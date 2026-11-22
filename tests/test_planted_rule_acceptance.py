"""
Section 8.2: Planted-Rule End-to-End Acceptance Test (Offline, MockLLM).

Runs the complete PromptPrism 2.1 pipeline on 6 binary factors (3 global_scaffold +
3 conditional_rule) in a 16-run 2^(6-2)IV orthogonal design:
  TestSetGenerator -> CaseFilterPipeline -> blind tagger -> stratified split ->
  freeze -> Experiment.run(test_set, CheckSuiteMetric) -> Experiment.diagnose()

Validates the exact pass criteria for the rare conditional rule (`E: null_date_rule`):
1. Baseline-only, coverage=None        => SAFE_TO_PRUNE (reproduces the unguarded bug)
2. With coverage, n_T == 0             => UNTESTED
3. With coverage, n_T == 0 + -0.03 baseline penalty => CONFLICT_REVIEW
4. With targeted items (delta_T=+1.0, se_T=0.0)     => KEEP_TARGETED, never SAFE_TO_PRUNE
5. Truly dead covered clauses (A, F)   => SAFE_TO_PRUNE (gate does not block valid pruning)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np

from prompt_prism import (
    CheckSuiteMetric,
    Evaluator,
    Experiment,
    Factor,
    FactorSet,
    MockLLM,
    PromptTemplate,
    TestSetGenerator,
)
from prompt_prism.analysis.diagnostics import run_calibrated_attribution_audit

_TOPICS = [
    "invoice",
    "receipt",
    "shipment",
    "ledger",
    "payroll",
    "refund",
    "warranty",
    "booking",
    "policy",
    "ticket",
    "contract",
    "voucher",
    "rebate",
    "freight",
    "customs",
    "escrow",
    "deposit",
    "renewal",
    "payout",
    "billing",
    "audit",
    "claim",
    "order",
    "quote",
    "transfer",
    "dispatch",
    "permit",
    "census",
    "grant",
    "draft",
    "parcel",
    "credit",
    "debit",
    "budget",
    "filing",
    "record",
    "memo",
    "batch",
    "queue",
    "entry",
]


def _make_six_factors() -> FactorSet:
    return FactorSet(
        [
            Factor.binary(
                id="A",
                name="specialist_persona",
                level_0_content="",
                level_1_content="[CLAUSE_A] You are a senior operations extraction specialist.",
                description="Global persona framing (planted dead bloat)",
                metadata={"clause_role": "global_scaffold"},
            ),
            Factor.binary(
                id="B",
                name="brief_reasoning",
                level_0_content="",
                level_1_content="[CLAUSE_B] Think aloud briefly before giving your final output.",
                description="Global reasoning cue (planted harmful)",
                metadata={"clause_role": "global_scaffold"},
            ),
            Factor.binary(
                id="C",
                name="few_shot_demos",
                level_0_content="",
                level_1_content='[CLAUSE_C] Example: \'Order #12 on 2026-01-05\' -> {"status":"ok","date":"2026-01-05"}',
                description="Global few-shot demonstrations (planted helpful)",
                metadata={"clause_role": "global_scaffold"},
            ),
            Factor.binary(
                id="D",
                name="strict_output_contract",
                level_0_content="",
                level_1_content="[CLAUSE_D] Return strictly compact JSON with keys status, category, owner, priority, date.",
                description="Strict JSON schema contract (planted helpful)",
                metadata={"clause_role": "conditional_rule"},
            ),
            Factor.binary(
                id="E",
                name="null_date_rule",
                level_0_content="",
                level_1_content="[CLAUSE_E] If the input text has no date, set date to null (never invent a date).",
                description="Rare conditional missing-date rule (planted rare rule)",
                metadata={"clause_role": "conditional_rule"},
            ),
            Factor.binary(
                id="F",
                name="no_markdown_fences",
                level_0_content="",
                level_1_content="[CLAUSE_F] Do not wrap output in markdown code fences.",
                description="Conditional anti-fence rule (planted dead conditional rule)",
                metadata={"clause_role": "conditional_rule"},
            ),
        ]
    )


def _unique_tag(prefix: str, idx: int) -> str:
    w1 = _TOPICS[(idx - 1) % len(_TOPICS)]
    w2 = _TOPICS[((idx - 1) // len(_TOPICS) + idx * 3) % len(_TOPICS)]
    return f"{prefix}_{w1}_{w2}_{idx}"


def _build_mock_generator_client() -> MockLLM:
    counters: Dict[str, int] = {
        "base": 0,
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 0,
        "F": 0,
    }

    def _gen(prompt: Any, _ctx: Dict[str, Any]) -> str:
        text = str(prompt)
        if "[PROMPTPRISM_TASK:CLASSIFY_ROLES]" in text:
            return json.dumps(
                {
                    "roles": {
                        "A": "global_scaffold",
                        "B": "global_scaffold",
                        "C": "global_scaffold",
                        "D": "conditional_rule",
                        "E": "conditional_rule",
                        "F": "conditional_rule",
                    }
                }
            )

        m_n = re.search(r"Generate (\d+)", text)
        n = int(m_n.group(1)) if m_n else 10

        m_factor = re.search(r"(?:clause|scaffold) `([A-F])`", text)
        if m_factor:
            fid = m_factor.group(1)
            cases: List[Dict[str, Any]] = []
            for _ in range(n):
                counters[fid] += 1
                idx = counters[fid]
                tag = _unique_tag(f"targ{fid}", idx)
                if fid == "E":
                    # Rare conditional rule: missing-date inputs requiring `"date": null`
                    cases.append(
                        {
                            "inputs": {
                                "document": f"MISSING_DATE_TARGET {tag} customer record with no calendar timestamp"
                            },
                            "target": '{"status":"ok","date":null}',
                            "checks": [
                                {
                                    "kind": "contains",
                                    "value": '"date":null',
                                    "weight": 1.0,
                                    "description": "null date required when absent",
                                }
                            ],
                            "reference_pass": '{"status":"ok","date":null}',
                            "reference_fail": '{"status":"ok","date":"2026-01-01"}',
                        }
                    )
                elif fid == "D":
                    cases.append(
                        {
                            "inputs": {
                                "document": f"STRICT_JSON_TARGET {tag} dated 2026-05-10 requiring exact schema"
                            },
                            "target": '{"status":"ok","date":"2026-05-10"}',
                            "checks": [
                                {
                                    "kind": "json_schema",
                                    "value": {
                                        "type": "object",
                                        "required": ["status", "date"],
                                    },
                                    "weight": 1.0,
                                    "description": "strict json schema",
                                }
                            ],
                            "reference_pass": '{"status":"ok","date":"2026-05-10"}',
                            "reference_fail": "status=ok date=2026-05-10",
                        }
                    )
                elif fid == "F":
                    cases.append(
                        {
                            "inputs": {
                                "document": f"NO_FENCE_TARGET {tag} dated 2026-06-11 raw payload"
                            },
                            "target": '{"status":"ok","date":"2026-06-11"}',
                            "checks": [
                                {
                                    "kind": "not_contains",
                                    "value": "```",
                                    "weight": 1.0,
                                    "description": "no markdown fences",
                                }
                            ],
                            "reference_pass": '{"status":"ok","date":"2026-06-11"}',
                            "reference_fail": '```json\n{"status":"ok"}\n```',
                        }
                    )
                else:
                    # Global scaffolds A, B, C: scored strictly on baseline task correctness
                    cases.append(
                        {
                            "inputs": {
                                "document": f"HARD_SCAFFOLD_{fid} {tag} ambiguous priority routing dated 2026-04-02"
                            },
                            "target": '{"status":"ok","priority":"high","date":"2026-04-02"}',
                            "checks": [
                                {
                                    "kind": "contains",
                                    "value": '"priority":"high"',
                                    "weight": 1.0,
                                    "description": "baseline task correctness",
                                }
                            ],
                            "reference_pass": '{"status":"ok","priority":"high","date":"2026-04-02"}',
                            "reference_fail": '{"status":"ok","priority":"low","date":"2026-04-02"}',
                        }
                    )
            return json.dumps({"cases": cases})

        # Baseline traffic: 5% of baseline items genuinely have no date; multi-field task checks
        base_cases: List[Dict[str, Any]] = []
        for _ in range(n):
            counters["base"] += 1
            idx = counters["base"]
            tag = _unique_tag("base", idx)
            # Every 20th baseline item (exact 5% rate) has no date
            is_missing_date = (idx % 20) == 0
            # Separate small 3% subset where clause E slightly clashes on a non-missing-date edge
            is_clash = (not is_missing_date) and ((idx % 33) == 1)
            if is_missing_date:
                doc = (
                    f"BASE_MISSING_DATE {tag} routine summary lacking any calendar date"
                )
                date_check_val = '"date":null'
                ref_pass = '{"status":"ok","category":"ops","owner":"team","priority":"high","date":null}'
                ref_fail = '{"status":"err","category":"other","owner":"none","priority":"low","date":"2026-01-01"}'
            else:
                marker = "BASE_CLASH_E" if is_clash else "BASE_STANDARD"
                doc = f"{marker} {tag} dated 2026-03-14 routine operations record"
                date_check_val = '"date":"2026-03-14"'
                ref_pass = '{"status":"ok","category":"ops","owner":"team","priority":"high","date":"2026-03-14"}'
                ref_fail = '{"status":"err","category":"other","owner":"none","priority":"low","date":"unknown"}'

            # 5 equal-weight checks (each worth 0.20 of item score)
            base_cases.append(
                {
                    "inputs": {"document": doc},
                    "target": ref_pass,
                    "checks": [
                        {
                            "kind": "contains",
                            "value": '"status":"ok"',
                            "weight": 1.0,
                            "description": "status field",
                        },
                        {
                            "kind": "contains",
                            "value": '"category":"ops"',
                            "weight": 1.0,
                            "description": "category field",
                        },
                        {
                            "kind": "contains",
                            "value": '"owner":"team"',
                            "weight": 1.0,
                            "description": "owner field",
                        },
                        {
                            "kind": "contains",
                            "value": '"priority":"high"',
                            "weight": 1.0,
                            "description": "priority field",
                        },
                        {
                            "kind": "contains",
                            "value": date_check_val,
                            "weight": 1.0,
                            "description": "date field",
                        },
                    ],
                    "reference_pass": ref_pass,
                    "reference_fail": ref_fail,
                }
            )
        return json.dumps({"cases": base_cases})

    return MockLLM(
        response_generator=_gen, latency_ms=0.0, model_id="mock-generator-v1"
    )


def _build_mock_tagger_client() -> MockLLM:
    def _tag(prompt: Any, _ctx: Dict[str, Any]) -> str:
        text = str(prompt)
        m = re.search(r"Cases:\n(\[.*\])\s*$", text, flags=re.DOTALL)
        items = json.loads(m.group(1)) if m else []
        tags: Dict[str, List[str]] = {}
        for item in items:
            doc = str(item.get("inputs", {}).get("document", ""))
            matched: List[str] = []
            if "HARD_SCAFFOLD_A" in doc:
                matched.append("A")
            if "HARD_SCAFFOLD_B" in doc:
                matched.append("B")
            if "HARD_SCAFFOLD_C" in doc:
                matched.append("C")
            if "STRICT_JSON_TARGET" in doc:
                matched.append("D")
            if "MISSING_DATE_TARGET" in doc or "BASE_MISSING_DATE" in doc:
                matched.append("E")
            if "NO_FENCE_TARGET" in doc:
                matched.append("F")
            tags[item["id"]] = matched
        return json.dumps({"tags": tags})

    return MockLLM(response_generator=_tag, latency_ms=0.0, model_id="mock-tagger-v1")


def _build_mock_model_under_test(*, baseline_penalty_e: float = 0.0) -> MockLLM:
    """Deterministic model under test exhibiting the planted 6-factor effects.

    Factor ground truth:
      A (specialist_persona):     dead bloat (0.0 on baseline and targeted)
      B (brief_reasoning):        harmful (-0.12 baseline, -0.25 targeted_B)
      C (few_shot_demos):         helpful (+0.15 baseline, +1.00 targeted_C)
      D (strict_output_contract): helpful (+0.12 baseline, +1.00 targeted_D)
      E (null_date_rule):         rare conditional rule:
                                  - +0.20 on 5% of baseline (BASE_MISSING_DATE)
                                  - -0.20 on 3% of baseline (BASE_CLASH_E) -> net delta_F ≈ +0.004
                                  - +1.00 on 100% of targeted_E (se_T == 0.0, delta_T == +1.0)
                                  - Optional extra baseline penalty when `baseline_penalty_e < 0`
      F (no_markdown_fences):     dead conditional rule (model never emits fences -> 0.0 everywhere)
    """

    def _respond(prompt: Any, _ctx: Dict[str, Any]) -> str:
        text = (
            "\n".join(m["content"] for m in prompt)
            if isinstance(prompt, list)
            else str(prompt)
        )
        has_b = "[CLAUSE_B]" in text
        has_c = "[CLAUSE_C]" in text
        has_d = "[CLAUSE_D]" in text
        has_e = "[CLAUSE_E]" in text

        # Extract deterministic item index from document tag
        m_idx = re.search(r"_(\d+)\b", text)
        idx = int(m_idx.group(1)) if m_idx else 1

        if "MISSING_DATE_TARGET" in text:
            # 100% deterministic flip on targeted items for rare rule E (se_T == 0, delta_T == +1.0)
            return (
                '{"status":"ok","date":null}'
                if has_e
                else '{"status":"ok","date":"2026-01-01"}'
            )

        if "STRICT_JSON_TARGET" in text:
            return (
                '{"status":"ok","date":"2026-05-10"}'
                if has_d
                else "status=ok date=2026-05-10"
            )

        if "NO_FENCE_TARGET" in text:
            # Model already outputs clean unfenced JSON regardless of F -> F is truly dead
            return '{"status":"ok","date":"2026-06-11"}'

        if "HARD_SCAFFOLD_A" in text:
            # A has zero effect on its targeted hard items -> dead bloat
            return '{"status":"ok","priority":"high","date":"2026-04-02"}'

        if "HARD_SCAFFOLD_B" in text:
            # B hurts accuracy on its targeted items too
            prio = "low" if (has_b and idx % 2 == 0) else "high"
            return f'{{"status":"ok","priority":"{prio}","date":"2026-04-02"}}'

        if "HARD_SCAFFOLD_C" in text:
            # C helps on hard ambiguous items
            prio = "high" if has_c else "low"
            return f'{{"status":"ok","priority":"{prio}","date":"2026-04-02"}}'

        # Baseline items (5 equal-weight fields: status, category, owner, priority, date)
        # B hurts status on 60% of baseline items (-0.12 mean)
        status_ok = not (has_b and (idx % 5) in (0, 1, 2))
        # C helps category on 75% of baseline items (+0.15 mean)
        cat_ok = has_c or ((idx % 4) == 0)
        # D helps owner on 60% of baseline items (+0.12 mean)
        owner_ok = has_d or ((idx % 5) in (0, 1))

        # Priority field: unaffected unless baseline_penalty_e variant is enabled
        if baseline_penalty_e < 0.0 and has_e:
            # e.g. baseline_penalty_e = -0.03 -> fail 1 field (0.20) on ~15% of baseline items
            n_mod = max(1, int(round(0.20 / abs(baseline_penalty_e))))
            prio_ok = (idx % n_mod) != 0
        else:
            prio_ok = True

        # Date field: rare rule E flips missing-date baseline items (5%), mildly clashes on 3%
        if "BASE_MISSING_DATE" in text:
            date_val = "null" if has_e else '"2026-01-01"'
        elif "BASE_CLASH_E" in text:
            date_val = '"unknown"' if has_e else '"2026-03-14"'
        else:
            date_val = '"2026-03-14"'

        return (
            f'{{"status":"{"ok" if status_ok else "err"}",'
            f'"category":"{"ops" if cat_ok else "other"}",'
            f'"owner":"{"team" if owner_ok else "none"}",'
            f'"priority":"{"high" if prio_ok else "low"}",'
            f'"date":{date_val}}}'
        )

    return MockLLM(response_generator=_respond, latency_ms=0.0, model_id="mock-sut-v1")


def test_planted_rule_end_to_end_acceptance():
    """Verify full offline pipeline and all Section 8.2 planted-rule acceptance criteria."""
    factors = _make_six_factors()
    template = PromptTemplate.from_factors(
        factors,
        system_prompt="Extract structured operations JSON from the document.",
        data_template="Document: {{ document }}",
    )

    gen = TestSetGenerator(
        client=_build_mock_generator_client(),
        tagger_client=_build_mock_tagger_client(),
        n_baseline_discovery=100,
        n_baseline_confirmation=20,
        n_per_factor_discovery=25,
        n_per_factor_confirmation=10,
        seed=42,
    )
    test_set = gen.generate(
        factors,
        template,
        task_description="Structured document extraction with rare missing-date null rule",
        freeze=True,
    )
    assert test_set.is_frozen
    assert test_set.manifest.factor_roles == {
        "A": "global_scaffold",
        "B": "global_scaffold",
        "C": "global_scaffold",
        "D": "conditional_rule",
        "E": "conditional_rule",
        "F": "conditional_rule",
    }

    # Verify ~5% of baseline discovery items genuinely trigger E
    disc_base = [
        c for c in test_set.cases_for_split("discovery") if c.kind == "baseline"
    ]
    assert len(disc_base) == 100
    n_base_missing = sum(
        1 for c in disc_base if "BASE_MISSING_DATE" in c.inputs["document"]
    )
    assert 2 <= n_base_missing <= 8

    # Run 16-run 2^(6-2)IV orthogonal experiment on the frozen discovery split
    exp = Experiment(
        factors=factors,
        template=template,
        design="2(6-2)IV",
        evaluator=Evaluator([CheckSuiteMetric()]),
        target_metric="check_suite",
        title="Planted-Rule Acceptance Audit",
    )
    sut = _build_mock_model_under_test(baseline_penalty_e=0.0)
    results = exp.run(test_set, client=sut, max_workers=1)
    df_all = results.to_dataframe()
    base_ids = test_set.baseline_sample_ids("discovery")
    assert base_ids is not None
    df_baseline_only = df_all[df_all["sample_id"].isin(base_ids)].copy()

    # ------------------------------------------------------------------
    # Criterion 1: Baseline-only, coverage=None => rare rule E gets SAFE_TO_PRUNE
    # (documents the exact silent false-prune bug without the coverage gate)
    # ------------------------------------------------------------------
    audit_no_cov = run_calibrated_attribution_audit(
        data=df_baseline_only.drop(columns=["targets_factors"], errors="ignore"),
        factor_cols=factors.ids,
        target_col="check_suite",
        equivalence_margin=0.02,
        n_sign_flip=500,
        seed=42,
    )
    by_f_no_cov = {d.factor_id: d for d in audit_no_cov.factors}
    assert by_f_no_cov["E"].pruning_verdict == "SAFE_TO_PRUNE"
    assert any("No coverage map provided" in w for w in audit_no_cov.warnings)

    # ------------------------------------------------------------------
    # Criterion 2: With coverage map where n_T == 0 for E => UNTESTED
    # ------------------------------------------------------------------
    cov_zero_e = test_set.coverage("discovery")
    cov_zero_e["E"] = set()
    audit_zero_cov = exp.diagnose(
        results,
        test_set=test_set,
        coverage=cov_zero_e,
        equivalence_margin=0.02,
        min_targeted_items=20,
        n_sign_flip=500,
        seed=42,
    )
    by_f_zero = {d.factor_id: d for d in audit_zero_cov.factors}
    assert by_f_zero["E"].n_targeted == 0
    assert by_f_zero["E"].pruning_verdict == "UNTESTED"

    # ------------------------------------------------------------------
    # Criterion 3: Full frozen TestSet with targeted items => E gets KEEP_TARGETED
    # (never SAFE_TO_PRUNE), while dead covered clauses A and F still prune cleanly!
    # ------------------------------------------------------------------
    audit_full = exp.diagnose(
        results,
        test_set=test_set,
        equivalence_margin=0.02,
        min_targeted_items=20,
        n_sign_flip=500,
        seed=42,
    )
    assert audit_full.schema_version == "2.1.0"
    assert audit_full.evidence_tier == "predicted (synthetic tests)"
    assert audit_full.baseline_only_full_set is True
    assert audit_full.test_set_sha256 == test_set.manifest.cases_sha256
    assert not any("Split leakage" in w for w in audit_full.warnings)
    assert not any("equals model under test" in w for w in audit_full.warnings)

    by_f_full = {d.factor_id: d for d in audit_full.factors}
    # Targeted-only stats on E: check pure targeted subset (25 items -> delta_T=+1.0, se_T=0.0)
    # Note: E's effective coverage includes both the 25 targeted items AND the ~5 tagged
    # BASE_MISSING_DATE baseline items, all of which are positive -> KEEP_TARGETED!
    assert by_f_full["E"].pruning_verdict == "KEEP_TARGETED"
    assert by_f_full["E"].pruning_verdict != "SAFE_TO_PRUNE"
    assert by_f_full["E"].n_targeted >= 25
    assert by_f_full["E"].delta_targeted is not None
    assert by_f_full["E"].delta_targeted > 0.70

    # Also verify exact se_T == 0.0, delta_T == +1.0 when evaluated on E's 25 targeted items alone
    targeted_e_only = {
        c.id
        for c in test_set.cases_for_split("discovery")
        if c.kind == "targeted" and c.intended_factor == "E"
    }
    cov_exact_e = test_set.coverage("discovery")
    cov_exact_e["E"] = targeted_e_only
    audit_exact_e = exp.diagnose(
        results,
        test_set=test_set,
        coverage=cov_exact_e,
        equivalence_margin=0.02,
        min_targeted_items=20,
        n_sign_flip=500,
        seed=42,
    )
    e_diag = next(d for d in audit_exact_e.factors if d.factor_id == "E")
    assert e_diag.n_targeted == 25
    assert np.isclose(e_diag.delta_targeted, 1.0, atol=1e-12)
    assert np.isclose(e_diag.se_targeted, 0.0, atol=1e-12)
    assert e_diag.pruning_verdict == "KEEP_TARGETED"

    # Other 5 planted factors receive their exact ground-truth verdicts
    assert by_f_full["A"].pruning_verdict == "SAFE_TO_PRUNE"  # dead global_scaffold
    assert by_f_full["B"].pruning_verdict == "REMOVE_HARMFUL"  # harmful global_scaffold
    assert by_f_full["C"].pruning_verdict == "KEEP"  # helpful global_scaffold
    assert by_f_full["D"].pruning_verdict == "KEEP"  # helpful conditional_rule
    assert by_f_full["F"].pruning_verdict == "SAFE_TO_PRUNE"  # dead conditional_rule

    # ------------------------------------------------------------------
    # Criterion 4: Untested rule with -0.03 baseline penalty => CONFLICT_REVIEW
    # (never unguarded REMOVE_HARMFUL when n_T == 0!)
    # ------------------------------------------------------------------
    sut_penalized = _build_mock_model_under_test(baseline_penalty_e=-0.03)
    results_pen = exp.run(test_set, client=sut_penalized, max_workers=1)

    # With n_T == 0 for E: baseline hurts (~ -0.03), targeted untested => CONFLICT_REVIEW
    audit_pen_untested = exp.diagnose(
        results_pen,
        test_set=test_set,
        coverage=cov_zero_e,
        equivalence_margin=0.02,
        min_targeted_items=20,
        n_sign_flip=500,
        seed=42,
    )
    e_pen_untested = next(d for d in audit_pen_untested.factors if d.factor_id == "E")
    assert e_pen_untested.effect_delta < -0.015
    assert e_pen_untested.n_targeted == 0
    assert e_pen_untested.pruning_verdict == "CONFLICT_REVIEW"

    # With full targeted coverage: baseline hurts AND targeted strongly helps => CONFLICT_REVIEW
    audit_pen_covered = exp.diagnose(
        results_pen,
        test_set=test_set,
        equivalence_margin=0.02,
        min_targeted_items=20,
        n_sign_flip=500,
        seed=42,
    )
    e_pen_covered = next(d for d in audit_pen_covered.factors if d.factor_id == "E")
    assert e_pen_covered.pruning_verdict == "CONFLICT_REVIEW"
    assert (
        e_pen_covered.delta_targeted is not None and e_pen_covered.delta_targeted > 0.70
    )

    # Ensure TestSet confirmation split remains completely untouched throughout discovery
    disc_sample_ids = set(df_all["sample_id"].astype(str).unique())
    conf_ids = {c.id for c in test_set.cases_for_split("confirmation")}
    assert disc_sample_ids.isdisjoint(conf_ids)
    assert not test_set.detect_split_leakage(disc_sample_ids)
