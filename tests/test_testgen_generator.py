"""
Offline MockLLM tests for PR 3: TestSetGenerator, clause-role classification,
validation/security/sanity/dedup filters, blind independent coverage tagging,
stratified splitting, confirmation refresh, and end-to-end diagnose loop.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Dict, List

import pytest

from prompt_prism import (
    CheckSpec,
    CheckSuiteMetric,
    Experiment,
    Factor,
    FactorType,
    MockLLM,
    PromptTemplate,
    TestCase,
    TestSetGenerator,
)
from prompt_prism.testgen import (
    GENERATOR_PROMPT_HASH,
    CaseFilterPipeline,
    check_deterministic_sanity,
    is_safe_regex,
    token_jaccard,
    tokenize_for_jaccard,
)
from prompt_prism.testgen.generator import _extract_json_payload

# ---------------------------------------------------------------------------
# 1. Filters: ReDoS, sanity (including not_contains), dedup, and pipeline
# ---------------------------------------------------------------------------


def test_is_safe_regex_and_jaccard():
    assert is_safe_regex(r"^\d{4}-\d{2}-\d{2}$")[0] is True
    assert is_safe_regex(r"(a+)+")[0] is False
    assert is_safe_regex(r"([a-z]*)*")[0] is False
    assert is_safe_regex(r"[unclosed")[0] is False
    assert is_safe_regex("a" * 600)[0] is False

    t1 = tokenize_for_jaccard("The quick brown fox jumps over the lazy dog today")
    t2 = tokenize_for_jaccard("the quick brown fox jumps over the lazy dog")
    assert token_jaccard(t1, t2) >= 0.88


def test_deterministic_sanity_passable_failable_and_not_contains():
    # Good positive check: passes reference_pass, fails reference_fail, "", and echo
    tc_ok = TestCase(
        inputs={"text": "Invoice dated Jan 5, 2026"},
        target="2026-01-05",
        checks=[CheckSpec(kind="regex", value=r"^\d{4}-\d{2}-\d{2}$")],
        reference_pass="2026-01-05",
        reference_fail="January 5th",
        kind="baseline",
    )
    assert check_deterministic_sanity(tc_ok)[0] is True

    # Unpassable check: reference_pass violates regex -> dropped
    tc_unpassable = TestCase(
        inputs={"text": "Invoice"},
        checks=[CheckSpec(kind="regex", value=r"^\d{4}-\d{2}-\d{2}$")],
        reference_pass="not-a-date",
        reference_fail="bad",
        kind="baseline",
    )
    ok, reason = check_deterministic_sanity(tc_unpassable)
    assert ok is False and "reference_pass failed" in reason

    # Unfailable check: `.*` passes empty string "" -> dropped
    tc_trivial = TestCase(
        inputs={"text": "Invoice"},
        checks=[CheckSpec(kind="regex", value=r".*")],
        reference_pass="anything",
        reference_fail="wrong",
        kind="baseline",
    )
    ok, reason = check_deterministic_sanity(tc_trivial)
    assert ok is False

    # Pure `not_contains` check: MUST survive even though "" passes `not_contains`,
    # validated via reference_pass + reference_fail!
    tc_neg = TestCase(
        inputs={"text": "Return raw JSON without markdown fences."},
        checks=[CheckSpec(kind="not_contains", value="```")],
        reference_pass='{"status": "ok"}',
        reference_fail='```json\n{"status": "ok"}\n```',
        kind="targeted",
        intended_factor="B",
    )
    assert check_deterministic_sanity(tc_neg)[0] is True

    # Pure `not_contains` WITHOUT reference_fail cannot prove failability -> dropped
    tc_neg_no_fail = TestCase(
        inputs={"text": "Return raw JSON"},
        checks=[CheckSpec(kind="not_contains", value="```")],
        reference_pass='{"status": "ok"}',
        kind="targeted",
        intended_factor="B",
    )
    ok, reason = check_deterministic_sanity(tc_neg_no_fail)
    assert ok is False and "requires reference_fail" in reason

    # Pure llm_judge item skips deterministic probing without calling any judge
    tc_judge = TestCase(
        inputs={"text": "Write a calm apology."},
        checks=[CheckSpec(kind="llm_judge", value="Tone is calm and empathetic.")],
        kind="baseline",
    )
    assert check_deterministic_sanity(tc_judge)[0] is True


def test_case_filter_pipeline_drops_invalid_security_sanity_and_duplicates():
    pipe = CaseFilterPipeline(
        required_vars=["text"],
        judge_fallback=False,
        dedup_jaccard_threshold=0.9,
    )

    valid_raw = {
        "inputs": {
            "text": "Customer reported a stolen debit card while traveling abroad today"
        },
        "target": "lost_or_stolen_card",
        "checks": [{"kind": "exact", "value": "lost_or_stolen_card"}],
        "reference_pass": "lost_or_stolen_card",
        "reference_fail": "card_arrival",
    }
    first = pipe.filter_candidate(valid_raw, kind="baseline")
    assert first is not None

    # Exact duplicate dropped
    assert pipe.filter_candidate(valid_raw, kind="baseline") is None

    # Near-duplicate (10/11 shared tokens -> Jaccard ~0.91 >= 0.90) dropped
    near_dup = dict(valid_raw)
    near_dup["inputs"] = {
        "text": "Customer reported a stolen debit card while traveling abroad!"
    }
    assert pipe.filter_candidate(near_dup, kind="baseline") is None

    # Missing required var dropped
    assert (
        pipe.filter_candidate(
            {
                "inputs": {"wrong_var": "hi"},
                "target": "ok",
                "reference_pass": "ok",
                "reference_fail": "bad",
            },
            kind="baseline",
        )
        is None
    )

    # Synthetic custom check dropped
    assert (
        pipe.filter_candidate(
            {
                "inputs": {"text": "completely unique input alpha"},
                "checks": [{"kind": "custom", "value": "os:system"}],
                "reference_pass": "ok",
                "reference_fail": "bad",
            },
            kind="baseline",
        )
        is None
    )

    # llm_judge dropped when judge_fallback=False
    assert (
        pipe.filter_candidate(
            {
                "inputs": {"text": "completely unique input beta"},
                "checks": [{"kind": "llm_judge", "value": "Check tone."}],
            },
            kind="baseline",
        )
        is None
    )

    # ReDoS regex dropped
    assert (
        pipe.filter_candidate(
            {
                "inputs": {"text": "completely unique input gamma"},
                "checks": [{"kind": "regex", "value": "(a+)+$"}],
                "reference_pass": "aaa",
                "reference_fail": "bbb",
            },
            kind="baseline",
        )
        is None
    )

    assert pipe.stats.accepted == 1
    assert pipe.stats.dropped_duplicate == 2
    assert pipe.stats.dropped_template_vars == 1
    assert pipe.stats.dropped_security == 2
    assert pipe.stats.dropped_judge_disabled == 1


def test_extract_json_payload_handles_fences_and_prose():
    assert _extract_json_payload('Here you go:\n```json\n{"cases": [1, 2]}\n```') == {
        "cases": [1, 2]
    }
    assert _extract_json_payload('Sure! Result is {"a": {"b": 1}} done.') == {
        "a": {"b": 1}
    }
    assert _extract_json_payload("List: [1, 2, 3] trailing") == [1, 2, 3]


# ---------------------------------------------------------------------------
# 2. Factor validation & clause-role classification
# ---------------------------------------------------------------------------


def test_generator_validates_binary_non_param_factors():
    gen = TestSetGenerator(MockLLM(default_response="{}", latency_ms=0.0))
    tmpl = PromptTemplate.from_factors([], data_template="{{ text }}")

    with pytest.raises(ValueError, match="at least one Factor"):
        gen.generate([], tmpl)

    param_f = Factor.binary(
        "temp", level_0_content=0.0, level_1_content=0.7, factor_type=FactorType.PARAM
    )
    with pytest.raises(ValueError, match="factor_type=PARAM"):
        gen.generate([param_f], tmpl)

    multi_f = Factor(name="tone", levels=["formal", "casual", "brief"])
    with pytest.raises(ValueError, match="binary"):
        gen.generate([multi_f], tmpl)


def test_clause_role_classification_precedence_and_fallback():
    factors = [
        Factor.binary(
            "specialist_persona",
            level_0_content="",
            level_1_content="You are an expert analyst.",
        ),
        Factor.binary(
            "brief_reasoning",
            level_0_content="",
            level_1_content="Explain briefly before answering.",
            metadata={"clause_role": "global_scaffold"},
        ),
        Factor.binary(
            "null_date_rule",
            level_0_content="",
            level_1_content="If no date is mentioned, return null.",
        ),
        Factor.binary(
            "custom_clause",
            level_0_content="",
            level_1_content="Do X.",
        ),
    ]

    # LLM classifies custom_clause as global_scaffold, omits others -> heuristic fills rest
    def role_llm(prompt, kwargs):
        return json.dumps({"roles": {"D": "global_scaffold"}})

    gen = TestSetGenerator(
        MockLLM(response_generator=role_llm, latency_ms=0.0),
    )
    fset = gen._validate_factors(factors)
    roles = gen.classify_factor_roles(
        fset,
        explicit_roles={"null_date_rule": "conditional_rule"},
    )
    assert roles == {
        "A": "global_scaffold",  # heuristic
        "B": "global_scaffold",  # Factor.metadata["clause_role"]
        "C": "conditional_rule",  # explicit override
        "D": "global_scaffold",  # LLM pre-pass
    }


# ---------------------------------------------------------------------------
# 3. Full offline MockLLM generation, oversampling, blind tagging, split & refresh
# ---------------------------------------------------------------------------


def _build_mock_gen_and_tagger(
    *,
    inject_bad_candidates: bool = True,
    disagree_on_first_b: bool = False,
):
    """Build deterministic offline mock generator & blind tagger clients."""
    counter = {"base": 0, "A": 0, "B": 0}
    seen_gen_prompts: List[str] = []
    seen_tagger_prompts: List[str] = []

    def gen_fn(prompt, kwargs):
        p = str(prompt)
        seen_gen_prompts.append(p)
        if "[PROMPTPRISM_TASK:CLASSIFY_ROLES]" in p:
            return json.dumps(
                {"roles": {"A": "global_scaffold", "B": "conditional_rule"}}
            )

        m_n = re.search(r"Generate (\d+) ", p)
        n_req = int(m_n.group(1)) if m_n else 5
        cases = []

        if "[PROMPTPRISM_TASK:GENERATE_BASELINE]" in p:
            if inject_bad_candidates and counter["base"] == 0:
                # Inject bad candidates to verify filtering + oversampling recovery
                cases.append({"inputs": {"wrong": "x"}, "target": "ok"})
                cases.append(
                    {
                        "inputs": {"text": "Bad custom check"},
                        "checks": [{"kind": "custom", "value": "os:system"}],
                        "reference_pass": "ok",
                        "reference_fail": "no",
                    }
                )
            for _ in range(n_req):
                idx = counter["base"]
                counter["base"] += 1
                day = (idx % 28) + 1
                date_str = f"2026-04-{day:02d}"
                cases.append(
                    {
                        "inputs": {
                            "text": f"Baseline record alpha{idx} scheduled for {date_str} in sector {idx * 7}"
                        },
                        "target": date_str,
                        "checks": [{"kind": "exact", "value": date_str}],
                        "reference_pass": date_str,
                        "reference_fail": "unknown",
                    }
                )
            return json.dumps({"cases": cases})

        if "[PROMPTPRISM_TASK:GENERATE_TARGETED_SCAFFOLD]" in p:
            # Global scaffold A: hard dated inputs scored on task correctness
            for _ in range(n_req):
                idx = counter["A"]
                counter["A"] += 1
                day = (idx % 28) + 1
                date_str = f"2026-05-{day:02d}"
                cases.append(
                    {
                        "inputs": {
                            "text": f"Hard scaffold item beta{idx} ambiguous note dated {date_str} ref {idx * 13}"
                        },
                        "target": date_str,
                        "checks": [{"kind": "exact", "value": date_str}],
                        "reference_pass": date_str,
                        "reference_fail": "wrong_date",
                    }
                )
            return json.dumps({"cases": cases})

        if "[PROMPTPRISM_TASK:GENERATE_TARGETED_RULE]" in p:
            # Conditional rule B: undated items requiring "null"
            for _ in range(n_req):
                idx = counter["B"]
                counter["B"] += 1
                tag_word = (
                    "DISAGREE_ITEM" if (disagree_on_first_b and idx == 0) else "UNDATED"
                )
                cases.append(
                    {
                        "inputs": {
                            "text": f"{tag_word} memo gamma{idx} with zero calendar info ref {idx * 17}"
                        },
                        "target": "null",
                        "checks": [{"kind": "exact", "value": "null"}],
                        "reference_pass": "null",
                        "reference_fail": "2026-01-01",
                    }
                )
            return json.dumps({"cases": cases})

        return json.dumps({"cases": []})

    def tagger_fn(prompt, kwargs):
        p = str(prompt)
        seen_tagger_prompts.append(p)
        # Extract JSON array after "Cases:\n"
        raw_json = p.split("Cases:\n", 1)[1].strip()
        items = json.loads(raw_json)
        tags: Dict[str, List[str]] = {}
        for item in items:
            txt = str(item["inputs"].get("text", ""))
            matched: List[str] = []
            if "Baseline" in txt or "Hard scaffold" in txt:
                matched.append("A")
            if "UNDATED" in txt:
                matched.append("B")
            # DISAGREE_ITEM intentionally gets [] so tagger_agrees becomes False!
            tags[item["id"]] = matched
        return json.dumps({"tags": tags})

    gen_client = MockLLM(
        response_generator=gen_fn, latency_ms=0.0, model_id="mock-gen-v1"
    )
    tagger_client = MockLLM(
        response_generator=tagger_fn, latency_ms=0.0, model_id="mock-tagger-v1"
    )
    return gen_client, tagger_client, seen_gen_prompts, seen_tagger_prompts


def test_generator_full_pipeline_blind_tagging_and_e2e_diagnose():
    factors = [
        Factor.binary(
            "specialist_persona",
            level_0_content="",
            level_1_content="You are an expert date extraction specialist.",
        ),
        Factor.binary(
            "null_date_rule",
            level_0_content="",
            level_1_content="If no date is present, return null.",
        ),
    ]
    tmpl = PromptTemplate.from_factors(
        factors, system_prompt="Extract date.", data_template="Input: {{ text }}"
    )

    gen_client, tagger_client, gen_prompts, tagger_prompts = _build_mock_gen_and_tagger(
        inject_bad_candidates=True,
        disagree_on_first_b=True,
    )

    generator = TestSetGenerator(
        gen_client,
        tagger_client=tagger_client,
        n_baseline_discovery=20,
        n_baseline_confirmation=6,
        n_per_factor_discovery=21,
        n_per_factor_confirmation=6,
        batch_size=10,
        seed=42,
    )
    ts = generator.generate(
        factors,
        tmpl,
        task_description="Extract ISO dates from short texts.",
        seed_examples=[{"text": "Order placed 2026-01-15"}],
        freeze=True,
    )

    # 1. Clause-role branching verified in prompts
    assert any(
        "[PROMPTPRISM_TASK:GENERATE_TARGETED_SCAFFOLD]" in p for p in gen_prompts
    )
    assert any("[PROMPTPRISM_TASK:GENERATE_TARGETED_RULE]" in p for p in gen_prompts)
    assert ts.manifest.factor_roles == {
        "A": "global_scaffold",
        "B": "conditional_rule",
    }

    # 2. Tagger received strictly blind payloads (never saw `intended_factor` or `kind`)
    assert len(tagger_prompts) >= 1
    for tp in tagger_prompts:
        assert "intended_factor" not in tp
        assert '"kind"' not in tp

    # 3. Manifest provenance, oversampling recovery, and split counts
    assert ts.is_frozen
    ts.verify_integrity()
    assert ts.manifest.generator["model_id"] == "mock-gen-v1"
    assert ts.manifest.generator["tagger_model_id"] == "mock-tagger-v1"
    assert ts.manifest.generator["prompt_hash"] == GENERATOR_PROMPT_HASH
    assert ts.manifest.generator["filter_stats"]["dropped_template_vars"] == 1
    assert ts.manifest.generator["filter_stats"]["dropped_security"] == 1

    assert len(ts.baseline_sample_ids("discovery") or set()) == 20
    assert len(ts.baseline_sample_ids("confirmation") or set()) == 6

    # Tagger disagreed with 1 targeted case for B -> excluded from strict coverage!
    disagreed = [c for c in ts.cases if c.tagger_agrees is False]
    assert len(disagreed) == 1
    assert disagreed[0].intended_factor == "B"
    assert "B" not in disagreed[0].effective_targets_factors(require_tag_agreement=True)
    # Tagger also tagged baseline items as covering global scaffold A (free coverage!)
    disc_cov = ts.coverage("discovery", require_tag_agreement=True)
    assert len(disc_cov["A"]) == 20 + 21  # 20 baseline + 21 targeted scaffold
    assert len(disc_cov["B"]) >= 20

    # 4. End-to-end: run Experiment on the generated TestSet and diagnose!
    def sut_fn(prompt, kwargs):
        p = str(prompt)
        m = re.search(r"2026-\d{2}-\d{2}", p)
        if m:
            return m.group(0)
        return "null" if "If no date is present, return null." in p else "missing"

    sut_client = MockLLM(response_generator=sut_fn, latency_ms=0.0, model_id="mock-sut")
    exp = Experiment(
        factors=factors,
        template=tmpl,
        evaluator=[CheckSuiteMetric()],
        target_metric="check_suite",
    )
    exp.run(ts, client=sut_client, max_workers=2)
    audit = exp.diagnose(
        min_targeted_items=20, equivalence_margin=0.02, n_sign_flip=100
    )
    by_f = {f.factor_id: f for f in audit.factors}
    assert by_f["A"].pruning_verdict == "SAFE_TO_PRUNE"
    assert by_f["B"].pruning_verdict == "KEEP_TARGETED"
    assert audit.evidence_tier == "predicted (synthetic tests)"
    assert not any("Generator model" in w for w in audit.warnings)  # distinct models!


def test_refresh_confirmation_preserves_discovery_and_deduplicates():
    factors = [
        Factor.binary("persona", level_0_content="", level_1_content="Expert."),
        Factor.binary("null_rule", level_0_content="", level_1_content="Return null."),
    ]
    tmpl = PromptTemplate.from_factors(factors, data_template="Input: {{ text }}")

    gen_client, tagger_client, _, _ = _build_mock_gen_and_tagger(
        inject_bad_candidates=False
    )
    generator = TestSetGenerator(
        gen_client,
        tagger_client=tagger_client,
        n_baseline_discovery=6,
        n_baseline_confirmation=4,
        n_per_factor_discovery=4,
        n_per_factor_confirmation=3,
        seed=10,
    )
    ts1 = generator.generate(factors, tmpl, freeze=True)
    disc_before = sorted(
        [c.model_dump(mode="json") for c in ts1.cases if c.split == "discovery"],
        key=lambda d: d["id"],
    )
    conf_ids_before = {c.id for c in ts1.cases if c.split == "confirmation"}

    ts2 = generator.refresh_confirmation(ts1, factors, tmpl, seed_offset=5, freeze=True)
    disc_after = sorted(
        [c.model_dump(mode="json") for c in ts2.cases if c.split == "discovery"],
        key=lambda d: d["id"],
    )
    conf_ids_after = {c.id for c in ts2.cases if c.split == "confirmation"}
    disc_ids = {d["id"] for d in disc_after}

    assert disc_after == disc_before  # discovery split preserved bit-for-bit
    assert len(conf_ids_after) == 4 + 3 + 3  # 4 baseline + 3*2 targeted
    assert conf_ids_after.isdisjoint(disc_ids)  # zero overlap with discovery
    assert conf_ids_after.isdisjoint(
        conf_ids_before
    )  # genuinely fresh confirmation items
    assert ts2.is_frozen
    assert ts2.manifest.cases_sha256 != ts1.manifest.cases_sha256
    assert "confirmation_refreshed_at" in ts2.manifest.generator


def test_generator_shortfall_warns_and_preserves_confirmation_share():
    factors = [
        Factor.binary("null_rule", level_0_content="", level_1_content="Return null.")
    ]
    tmpl = PromptTemplate.from_factors(factors, data_template="{{ text }}")

    # Client emits only 4 unique items per prompt, then repeats (causing dedup exhaustion)
    def stingy_client(prompt, kwargs):
        p = str(prompt)
        if "[PROMPTPRISM_TASK:CLASSIFY_ROLES]" in p:
            return json.dumps({"roles": {"A": "conditional_rule"}})
        if "[PROMPTPRISM_TASK:TAG_COVERAGE]" in p:
            items = json.loads(p.split("Cases:\n", 1)[1].strip())
            return json.dumps({"tags": {it["id"]: ["A"] for it in items}})
        prefix = "base" if "GENERATE_BASELINE" in p else "targ"
        return json.dumps(
            {
                "cases": [
                    {
                        "inputs": {"text": f"{prefix} unique item {i}"},
                        "target": "ok",
                        "checks": [{"kind": "exact", "value": "ok"}],
                        "reference_pass": "ok",
                        "reference_fail": "bad",
                    }
                    for i in range(4)
                ]
            }
        )

    gen = TestSetGenerator(
        MockLLM(response_generator=stingy_client, latency_ms=0.0),
        n_baseline_discovery=10,
        n_baseline_confirmation=10,
        n_per_factor_discovery=10,
        n_per_factor_confirmation=10,
        seed=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts = gen.generate(factors, tmpl, freeze=True)

    assert any("fell short" in str(w.message) for w in caught)
    assert "shortfalls" in ts.manifest.generator
    # Despite shortfall (4 instead of 20), 50/50 split preserves 2 discovery & 2 confirmation!
    assert len(ts.baseline_sample_ids("discovery") or set()) == 2
    assert len(ts.baseline_sample_ids("confirmation") or set()) == 2
    assert len(ts.coverage("discovery")["A"]) == 4  # 2 baseline + 2 targeted
    assert len(ts.coverage("confirmation")["A"]) == 4


def test_pr3_review_hardening_json_ref_tagger_retry_heuristic_and_refresh_guards():
    from prompt_prism.testgen.generator import _heuristic_clause_role

    # 1. Dict/list reference_pass & reference_fail stay valid JSON for json_schema checks
    pipe = CaseFilterPipeline(required_vars=["text"])
    schema_case = pipe.filter_candidate(
        {
            "inputs": {"text": "Acme paid $50 on 2026-01-01"},
            "target": {"buyer": "Acme", "date": "2026-01-01"},
            "checks": [
                {
                    "kind": "json_schema",
                    "value": {
                        "type": "object",
                        "properties": {
                            "buyer": {"type": "string"},
                            "date": {"type": ["string", "null"]},
                        },
                        "required": ["buyer", "date"],
                    },
                }
            ],
            "reference_pass": {"buyer": "Acme", "date": "2026-01-01"},
            "reference_fail": {"buyer": 123},
        },
        kind="baseline",
    )
    assert schema_case is not None
    assert json.loads(schema_case.reference_pass or "") == {
        "buyer": "Acme",
        "date": "2026-01-01",
    }

    # 2. Heuristic role keeps scaffold names as global_scaffold even when body mentions JSON/format
    verif_factor = Factor.binary(
        "verification_pass",
        level_0_content="",
        level_1_content="Before finalizing, cross-check against source text for correct JSON formatting.",
    )
    few_shot_factor = Factor.binary(
        "few_shot_demos",
        level_0_content="",
        level_1_content='Example 1: {"date": null}',
    )
    assert _heuristic_clause_role(verif_factor) == "global_scaffold"
    assert _heuristic_clause_role(few_shot_factor) == "global_scaffold"

    # 3. Tagger retry recovers transient failure; permanent failure leaves tagger_agrees=None (NOT False!)
    factors = [
        Factor.binary("null_rule", level_0_content="", level_1_content="Return null.")
    ]
    tmpl = PromptTemplate.from_factors(factors, data_template="{{ text }}")
    gen_client, _, _, _ = _build_mock_gen_and_tagger(inject_bad_candidates=False)

    # 3a. Flaky tagger fails first attempt, succeeds on retry
    flaky_calls = {"n": 0}

    def flaky_tagger(prompt, kwargs):
        flaky_calls["n"] += 1
        if flaky_calls["n"] == 1:
            return "not valid json"
        items = json.loads(str(prompt).split("Cases:\n", 1)[1].strip())
        return json.dumps({"tags": {it["id"]: ["A"] for it in items}})

    gen_retry = TestSetGenerator(
        gen_client,
        tagger_client=MockLLM(
            response_generator=flaky_tagger, latency_ms=0.0, model_id=None
        ),
        n_baseline_discovery=2,
        n_baseline_confirmation=1,
        n_per_factor_discovery=2,
        n_per_factor_confirmation=1,
        seed=1,
    )
    ts_retry = gen_retry.generate(factors, tmpl, freeze=True)
    assert flaky_calls["n"] >= 2
    assert all(c.tagger_agrees is True for c in ts_retry.cases if c.kind == "targeted")
    # Distinct tagger client with model_id=None must report "unknown", NOT generator's model_id!
    assert ts_retry.manifest.generator["model_id"] == "mock-gen-v1"
    assert ts_retry.manifest.generator["tagger_model_id"] == "unknown"

    # 3b. Permanently failing tagger warns and leaves tagger_agrees=None (does not wipe coverage!)
    gen_fail_tagger = TestSetGenerator(
        gen_client,
        tagger_client=MockLLM(default_response="garbage", latency_ms=0.0),
        n_baseline_discovery=2,
        n_baseline_confirmation=1,
        n_per_factor_discovery=2,
        n_per_factor_confirmation=1,
        seed=2,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts_untagged = gen_fail_tagger.generate(factors, tmpl, freeze=True)
    assert any("tagger failed or omitted" in str(w.message) for w in caught)
    assert ts_untagged.manifest.generator["tagger_untagged_cases"] == 6
    assert all(c.tagger_agrees is None for c in ts_untagged.cases)
    # Targeted cases keep intended_factor when tagger failed (tagger_agrees is None, not False)
    assert len(ts_untagged.coverage("discovery")["A"]) == 2

    # 4. refresh_confirmation rejects changed factors/template and deduplicates retired confirmation items
    changed_factors = [
        Factor.binary(
            "null_rule", level_0_content="", level_1_content="CHANGED CLAUSE TEXT."
        )
    ]
    with pytest.raises(ValueError, match="factors_sha256"):
        gen_retry.refresh_confirmation(ts_retry, changed_factors, tmpl)

    changed_tmpl = PromptTemplate.from_factors(factors, data_template="NEW: {{ text }}")
    with pytest.raises(ValueError, match="template_sha256"):
        gen_retry.refresh_confirmation(ts_retry, factors, changed_tmpl)

    # Refresh with a generator that tries to replay the exact retired confirmation items:
    # retired items must be rejected as duplicates!
    retired_conf_inputs = [
        c.inputs["text"] for c in ts_retry.cases if c.split == "confirmation"
    ]

    def replay_then_fresh(prompt, kwargs):
        p = str(prompt)
        if "[PROMPTPRISM_TASK:TAG_COVERAGE]" in p:
            items = json.loads(p.split("Cases:\n", 1)[1].strip())
            return json.dumps({"tags": {it["id"]: ["A"] for it in items}})
        cases_out = [
            {
                "inputs": {"text": txt},
                "target": "ok",
                "checks": [{"kind": "exact", "value": "ok"}],
                "reference_pass": "ok",
                "reference_fail": "bad",
            }
            for txt in retired_conf_inputs
        ]
        # Plus fresh items
        prefix = "freshbase" if "GENERATE_BASELINE" in p else "freshtarg"
        for i in range(4):
            cases_out.append(
                {
                    "inputs": {"text": f"{prefix} completely novel item omega{i}"},
                    "target": "ok",
                    "checks": [{"kind": "exact", "value": "ok"}],
                    "reference_pass": "ok",
                    "reference_fail": "bad",
                }
            )
        return json.dumps({"cases": cases_out})

    gen_refresh = TestSetGenerator(
        MockLLM(response_generator=replay_then_fresh, latency_ms=0.0),
        n_baseline_discovery=2,
        n_baseline_confirmation=1,
        n_per_factor_discovery=2,
        n_per_factor_confirmation=1,
        seed=1,
    )
    ts_refreshed = gen_refresh.refresh_confirmation(
        ts_retry, factors, tmpl, freeze=True
    )
    new_conf_texts = {
        c.inputs["text"] for c in ts_refreshed.cases if c.split == "confirmation"
    }
    assert new_conf_texts.isdisjoint(set(retired_conf_inputs))
    assert ts_refreshed.manifest.generator["confirmation_filter_stats"][
        "dropped_duplicate"
    ] >= len(retired_conf_inputs)
