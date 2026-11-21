"""
Tests for PR 2: TestSet schema, freezing/tamper verification, PromptTemplate serialization/vars,
JSONValidation schema support, CheckSuiteMetric (deterministic + offline LLMClient judge +
DeepEval isolation + accurate judge call estimation), and Experiment.run/diagnose(test_set=...).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import jsonschema
import pytest

from prompt_prism import (
    CallableLLM,
    CheckSpec,
    CheckSuiteMetric,
    Experiment,
    Factor,
    FactorSet,
    JSONValidation,
    JudgeCache,
    MockLLM,
    PromptTemplate,
    TestCase,
    TestSet,
)
from prompt_prism.testgen import compute_case_id


def _sample_custom_scorer(prediction, target=None, input_data=None) -> float:
    return 1.0 if "MAGIC" in str(prediction) else 0.0


# ---------------------------------------------------------------------------
# 1. PromptTemplate serialization & required_data_variables
# ---------------------------------------------------------------------------


def test_prompt_template_to_dict_sha256_and_required_vars():
    factors = FactorSet(
        [
            Factor.binary(
                "persona", level_0_content="", level_1_content="You are an expert."
            ),
            Factor.binary(
                "hint",
                level_0_content="",
                level_1_content="Domain hint: {{ domain }}",
            ),
        ]
    )
    tmpl = PromptTemplate.from_factors(
        factors=factors,
        system_prompt="Classify user queries.",
        data_template="Query: {{ query }}\nContext: {{ context }}",
    )
    tmpl.add_section(id="inactive_sec", content="{{ ignored_var }}", position=99)
    tmpl.sections[-1].is_active = False

    # Round-trip & deterministic hash
    d = tmpl.to_dict()
    rebuilt = PromptTemplate.from_dict(d)
    assert rebuilt.template_sha256() == tmpl.template_sha256()

    # Extracts Jinja vars from active sections + factor level contents, skips inactive
    assert tmpl.required_data_variables(factors) == {"query", "context", "domain"}

    # string.Template ($var) extraction
    tmpl_dollar = PromptTemplate()
    tmpl_dollar.add_section(id="s1", content="Ticket: $ticket_id — ${body}")
    assert tmpl_dollar.required_data_variables() == {"ticket_id", "body"}

    # master_template subtracts factor names, IDs, *_level, *_name
    tmpl_master = PromptTemplate(
        master_template="{{ persona }} {{ A }} {{ persona_level }} {{ A_level }} {{ persona_name }} :: {{ utterance }}"
    )
    assert tmpl_master.required_data_variables(factors) == {"utterance"}


# ---------------------------------------------------------------------------
# 2. JSONValidation with real JSON Schema validation
# ---------------------------------------------------------------------------


def test_json_validation_enforces_json_schema():
    schema = {
        "type": "object",
        "properties": {
            "date": {"type": ["string", "null"]},
            "count": {"type": "integer", "minimum": 0},
        },
        "required": ["date", "count"],
        "additionalProperties": False,
    }
    metric = JSONValidation(schema=schema)
    assert metric.compute('{"date": "2026-01-01", "count": 3}') == 1.0
    assert metric.compute('```json\n{"date": null, "count": 0}\n```') == 1.0
    assert metric.compute('{"date": "2026-01-01", "count": -1}') == 0.0
    assert metric.compute('{"date": "2026-01-01", "count": "3"}') == 0.0
    assert metric.compute('{"date": "2026-01-01", "count": 1, "extra": true}') == 0.0
    assert metric.compute("not json") == 0.0

    with pytest.raises(jsonschema.SchemaError):
        JSONValidation(schema={"type": "not_a_real_json_type"})


# ---------------------------------------------------------------------------
# 3. TestCase, CheckSpec, TestSet schema, splitting, and tamper-evident freeze
# ---------------------------------------------------------------------------


def test_check_spec_validation_and_security_guards():
    # Valid specs
    CheckSpec(kind="exact", value="ok")
    CheckSpec(kind="exact", value=None)
    CheckSpec(kind="regex", value=r"^\d{4}-\d{2}-\d{2}$")
    CheckSpec(kind="contains", value="approved")
    CheckSpec(kind="not_contains", value="sorry")
    CheckSpec(kind="json_schema", value={"type": "object", "required": ["id"]})
    CheckSpec(kind="custom", value="math:isnan")
    CheckSpec(kind="llm_judge", value="Output is polite and concise.")

    # Invalid specs rejected
    with pytest.raises(ValueError, match="Invalid regex"):
        CheckSpec(kind="regex", value="[unclosed")
    with pytest.raises(ValueError, match="non-empty string"):
        CheckSpec(kind="contains", value="   ")
    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        CheckSpec(kind="json_schema", value={"type": "bogus"})
    with pytest.raises(ValueError, match="module.path:callable_name"):
        CheckSpec(kind="custom", value="missing_colon")
    with pytest.raises(ValueError):
        CheckSpec(kind="exact", value="x", weight=0.0)

    # Security: synthetic TestCase cannot carry kind="custom"
    with pytest.raises(ValueError, match="forbidden on synthetic"):
        TestCase(
            inputs={"query": "hi"},
            checks=[CheckSpec(kind="custom", value="pkg.mod:fn")],
            kind="baseline",
            source="synthetic",
        )
    # Human TestCase may carry kind="custom"
    human_tc = TestCase(
        inputs={"query": "hi"},
        checks=[CheckSpec(kind="custom", value="pkg.mod:fn")],
        kind="baseline",
        source="human",
    )
    assert human_tc.id.startswith("tc_")

    # Reserved input key collision rejected
    with pytest.raises(ValueError, match="reserved key"):
        TestCase(inputs={"id": "x", "query": "hi"}, kind="baseline")
    with pytest.raises(ValueError, match="reserved key"):
        TestCase(inputs={"checks": [], "query": "hi"}, kind="baseline")

    # Baseline cannot have intended_factor
    with pytest.raises(ValueError, match="intended_factor"):
        TestCase(inputs={"query": "hi"}, kind="baseline", intended_factor="A")


def test_case_id_stable_across_workflow_mutations_and_tagger_agreement():
    tc = TestCase(
        inputs={"query": "Where is my card?"},
        target="lost_card",
        checks=[CheckSpec(kind="exact", value="lost_card")],
        kind="targeted",
        intended_factor="null_rule",
        tagger_factors=["other_factor"],
        tagger_agrees=False,
        review="pending",
    )
    orig_id = tc.id
    assert orig_id == compute_case_id(tc.inputs, tc.target, tc.checks)

    # Tagger disagreed & still pending -> intended_factor excluded by default
    assert tc.effective_targets_factors(require_tag_agreement=True) == ["other_factor"]
    assert tc.effective_targets_factors(require_tag_agreement=False) == [
        "null_rule",
        "other_factor",
    ]

    # Accepting in review restores intended_factor without altering content ID
    tc.review = "accepted"
    tc.split = "discovery"
    assert tc.effective_targets_factors(require_tag_agreement=True) == [
        "null_rule",
        "other_factor",
    ]
    assert tc.id == orig_id


def test_testset_split_freeze_save_load_and_tamper_refusal(tmp_path: Path):
    factors = FactorSet(
        [
            Factor.binary("persona", level_0_content="", level_1_content="Expert."),
            Factor.binary(
                "null_rule", level_0_content="", level_1_content="Return null."
            ),
        ]
    )
    tmpl = PromptTemplate.from_factors(factors, data_template="Q: {{ query }}")

    cases = [
        TestCase(
            inputs={"query": f"base query {i}"},
            target="ok",
            checks=[CheckSpec(kind="exact", value="ok")],
            kind="baseline",
            tagger_factors=["persona"],
        )
        for i in range(10)
    ]
    for fid in ("persona", "null_rule"):
        for i in range(6):
            cases.append(
                TestCase(
                    inputs={"query": f"targeted {fid} {i}"},
                    target="ok",
                    checks=[CheckSpec(kind="exact", value="ok")],
                    kind="targeted",
                    intended_factor=fid,
                    tagger_factors=[fid],
                    tagger_agrees=True,
                )
            )

    ts = TestSet.create(
        cases,
        factors=factors,
        template=tmpl,
        factor_roles={"persona": "global_scaffold", "null_rule": "conditional_rule"},
        generator={"model_id": "gen-model-v1", "seed": 7},
    )

    # Unsplit TestSet refuses default split="discovery" explicitly
    with pytest.raises(ValueError, match="no split assignments"):
        ts.to_dataset()

    ts.split_cases(discovery_ratio=0.7, seed=42)
    # Canonical factor normalization: no duplicate ID + name ghost keys!
    assert set(ts.manifest.counts["by_factor"].keys()) == {"A", "B"}
    assert ts.manifest.counts["by_factor"]["B"] == 6

    disc_cov = ts.coverage(split="discovery")
    conf_cov = ts.coverage(split="confirmation")
    assert set(disc_cov.keys()) == {"A", "B"}
    assert set(conf_cov.keys()) == {"A", "B"}
    assert len(disc_cov["B"]) == 4
    assert len(conf_cov["B"]) == 2
    assert set(ts.coverage(split="discovery", key_by="name").keys()) == {
        "persona",
        "null_rule",
    }
    assert len(ts.coverage(split="discovery", key_by="name")["null_rule"]) == 4
    assert len(ts.baseline_sample_ids(split="discovery") or set()) == 7
    assert len(ts.baseline_sample_ids(split="confirmation") or set()) == 3

    # Freeze prevents further splitting or accidental re-freeze without force=True
    ts.freeze()
    assert ts.is_frozen
    with pytest.raises(ValueError, match="already frozen"):
        ts.freeze()
    with pytest.raises(ValueError, match="already-frozen"):
        ts.split_cases()

    # Save & load round-trip
    out_dir = ts.save(tmp_path / "frozen_tests")
    loaded = TestSet.load(out_dir)
    assert loaded.is_frozen
    assert loaded.manifest.cases_sha256 == ts.manifest.cases_sha256
    assert loaded.manifest.template_sha256 == tmpl.template_sha256()
    assert loaded.provenance == "synthetic"

    # OS CRLF line endings must NOT break canonical JSON hash verification
    cases_file = out_dir / TestSet.CASES_FILENAME
    cases_file.write_bytes(
        cases_file.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8")
    )
    assert TestSet.load(out_dir).manifest.cases_sha256 == ts.manifest.cases_sha256

    # Tampering with a single case (even flipping review -> rejected) is refused on load
    raw_lines = cases_file.read_text(encoding="utf-8").splitlines()
    tampered_obj = json.loads(raw_lines[0])
    tampered_obj["review"] = "rejected"
    raw_lines[0] = json.dumps(tampered_obj)
    cases_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        TestSet.load(out_dir)


def test_testset_to_dataset_excludes_rejected_and_detects_leakage():
    c1 = TestCase(
        inputs={"query": "q1"},
        target="a1",
        checks=[CheckSpec(kind="exact", value="a1")],
        kind="baseline",
        split="discovery",
        source="human",
    )
    c2 = TestCase(
        inputs={"query": "q2"},
        target="a2",
        checks=[CheckSpec(kind="exact", value="a2")],
        kind="baseline",
        split="discovery",
        review="rejected",
        source="synthetic",
    )
    c3 = TestCase(
        inputs={"query": "q3"},
        target="a3",
        checks=[CheckSpec(kind="exact", value="a3")],
        kind="baseline",
        split="confirmation",
        source="log",
    )
    ts = TestSet.create([c1, c2, c3], freeze=True)
    # Active sources are human + log (rejected synthetic row excluded) -> mixed
    assert ts.provenance == "mixed"

    disc_ds = ts.to_dataset()  # defaults to split="discovery"
    assert [row["id"] for row in disc_ds] == [c1.id]
    assert "__test_inputs__" in disc_ds[0]
    assert not ts.detect_split_leakage([c1.id])
    assert ts.detect_split_leakage([c1.id, c3.id])

    # Duplicate case IDs rejected
    with pytest.raises(ValueError, match="Duplicate TestCase IDs"):
        TestSet.create([c1, c1.model_copy()])


# ---------------------------------------------------------------------------
# 4. CheckSuiteMetric (all check kinds, security, judge isolation, caching, estimation)
# ---------------------------------------------------------------------------


def test_check_suite_metric_deterministic_and_custom_checks():
    metric = CheckSuiteMetric()

    # Weighted exact + regex + contains + not_contains + json_schema
    checks = [
        CheckSpec(kind="exact", value=None, weight=1.0).model_dump(),  # uses target
        CheckSpec(kind="contains", value="status", weight=1.0).model_dump(),
        CheckSpec(kind="not_contains", value="error", weight=2.0).model_dump(),
        CheckSpec(kind="regex", value=r'"code"\s*:\s*200', weight=1.0).model_dump(),
        CheckSpec(
            kind="json_schema",
            value={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "code": {"type": "integer"},
                },
                "required": ["status", "code"],
            },
            weight=1.0,
        ).model_dump(),
    ]
    pred_ok = '{"status": "ok", "code": 200}'
    assert metric.compute(
        pred_ok, target=pred_ok, input_data={"checks": checks}
    ) == pytest.approx(1.0)

    # Fails exact (w=1) and not_contains (w=2), passes contains/regex/schema (w=3) -> 3/6 = 0.5
    pred_partial = '{"status": "error", "code": 200}'
    assert metric.compute(
        pred_partial, target=pred_ok, input_data={"checks": checks}
    ) == pytest.approx(0.5)

    # Custom check refused by default (security)
    custom_ref = f"{__name__}:_sample_custom_scorer"
    custom_checks = [CheckSpec(kind="custom", value=custom_ref).model_dump()]
    with pytest.raises(PermissionError, match="allow_custom_checks=True"):
        metric.compute("MAGIC", input_data={"checks": custom_checks})

    # Opt-in custom check works
    metric_opt_in = CheckSuiteMetric(allow_custom_checks=True)
    assert (
        metric_opt_in.compute("MAGIC word", input_data={"checks": custom_checks}) == 1.0
    )
    assert metric_opt_in.compute("plain", input_data={"checks": custom_checks}) == 0.0

    # Empty checks falls back to ExactMatch when target provided, else raises
    assert metric.compute("abc", target="ABC", input_data={"checks": []}) == 1.0
    with pytest.raises(ValueError, match="non-empty 'checks'"):
        metric.compute("abc", target=None, input_data={"checks": []})


def test_check_suite_metric_offline_judge_client_cache_and_prompt_isolation(
    tmp_path: Path,
):
    seen_prompts = []

    def stub_judge(prompt):
        seen_prompts.append(str(prompt))
        if "polite" in str(prompt).lower() and "please" in str(prompt).lower():
            return json.dumps({"score": 0.9, "reason": "Polite phrasing found."})
        return json.dumps({"score": 0.2, "reason": "Abrupt phrasing."})

    cache = JudgeCache(db_path=str(tmp_path / "judge.db"))
    metric = CheckSuiteMetric(
        judge_client=CallableLLM(stub_judge, model_id="mock-judge-1"),
        judge_cache=cache,
        judge_mode="score",
    )

    checks = [
        CheckSpec(kind="llm_judge", value="Check that reply is polite.").model_dump()
    ]
    ctx_run1 = {
        "query": "Help me reset my password",
        "__test_inputs__": '{"query": "Help me reset my password"}',
        "__prompt__": "SECRET_FACTOR_CLAUSE_LEVEL_1 :: Help me reset my password",
        "checks": checks,
    }
    ctx_run2 = {
        "query": "Help me reset my password",
        "__test_inputs__": '{"query": "Help me reset my password"}',
        "__prompt__": "DIFFERENT_FACTOR_CLAUSE_LEVEL_0 :: Help me reset my password",
        "checks": checks,
    }

    s1 = metric.compute("Please click reset.", target=None, input_data=ctx_run1)
    r1 = metric.pop_reason()
    # Second call with a DIFFERENT `__prompt__` MUST hit JudgeCache (proving zero prompt leakage!)
    s2 = metric.compute("Please click reset.", target=None, input_data=ctx_run2)
    r2 = metric.pop_reason()

    assert s1 == pytest.approx(0.9)
    assert s2 == pytest.approx(0.9)
    assert r1 == "Polite phrasing found."
    assert r2 == "Polite phrasing found."
    assert len(seen_prompts) == 1
    assert "SECRET_FACTOR_CLAUSE_LEVEL_1" not in seen_prompts[0]
    assert "Help me reset my password" in seen_prompts[0]

    # Judge failure yields NaN (not 0.0)
    failing_metric = CheckSuiteMetric(
        judge_client=CallableLLM(
            lambda p: (_ for _ in ()).throw(RuntimeError("timeout"))
        )
    )
    assert math.isnan(failing_metric.compute("hi", input_data=ctx_run1))
    assert "timeout" in (failing_metric.pop_reason() or "")


def test_check_suite_metric_deepeval_path_strips_prompt_and_isolates_input(monkeypatch):
    captured = {}

    class _StubDeepEvalMetric:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self._reason = "Looks accurate."

        def compute(self, prediction, target=None, input_data=None):
            captured["input_data"] = dict(input_data or {})
            return 0.85

        def pop_reason(self):
            return self._reason

    def _fake_deepeval_factory(kind, **kwargs):
        captured["kind"] = kind
        return _StubDeepEvalMetric(**kwargs)

    import prompt_prism.evaluation.check_suite as cs_mod

    monkeypatch.setattr(cs_mod, "deepeval_metric", _fake_deepeval_factory)

    metric = CheckSuiteMetric(judge_model="gpt-4o-mini", judge_mode="score")
    score = metric.compute(
        prediction="Answer text",
        target="Expected text",
        input_data={
            "utterance": "User utterance",
            "__prompt__": "SYSTEM CLAUSE LEVEL 1 :: User utterance",
            "checks": [{"kind": "llm_judge", "value": "Check faithfulness."}],
        },
    )
    assert score == pytest.approx(0.85)
    assert metric.pop_reason() == "Looks accurate."
    assert captured["kind"] == "g_eval"
    assert captured["init_kwargs"]["criteria"] == "Check faithfulness."
    assert captured["init_kwargs"]["input_key"] == cs_mod._CLEAN_INPUT_KEY
    assert "__prompt__" not in captured["input_data"]
    assert "User utterance" in captured["input_data"][cs_mod._CLEAN_INPUT_KEY]
    assert (
        "SYSTEM CLAUSE LEVEL 1" not in captured["input_data"][cs_mod._CLEAN_INPUT_KEY]
    )


def test_check_suite_metric_estimate_judge_calls_accurate_and_no_false_warning():
    factors = [Factor.binary("persona", level_0_content="", level_1_content="Expert.")]
    metric = CheckSuiteMetric(judge_client=MockLLM(default_response='{"score": 1.0}'))
    exp = Experiment.from_factors(
        factors=factors,
        data_template="{{ query }}",
        metrics=[metric],
        target_metric="check_suite",
    )

    det_dataset = [
        {
            "id": f"d_{i}",
            "query": f"q{i}",
            "checks": [{"kind": "contains", "value": "ok"}],
        }
        for i in range(5)
    ]
    assert exp.estimate_judge_calls(det_dataset) == 0

    # Zero UserWarning emitted on 100%-deterministic dataset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exp.run(det_dataset, client=MockLLM(default_response="ok"), max_workers=1)
    assert not any("Estimated judge calls" in str(w.message) for w in caught)

    # Mixed dataset: 1 item has 0 judge checks, 1 item has 2 judge checks -> 2 * n_runs
    mixed_dataset = [
        {"id": "m0", "query": "q0", "checks": [{"kind": "contains", "value": "ok"}]},
        {
            "id": "m1",
            "query": "q1",
            "checks": [
                {"kind": "llm_judge", "value": "rubric 1"},
                {"kind": "llm_judge", "value": "rubric 2"},
            ],
        },
    ]
    assert exp.estimate_judge_calls(mixed_dataset) == len(exp.design.runs) * 2


# ---------------------------------------------------------------------------
# 5. End-to-End Experiment.run(test_set) + Experiment.diagnose() + provenance
# ---------------------------------------------------------------------------


def test_experiment_run_and_diagnose_with_frozen_testset_and_guards():
    factors = [
        Factor.binary(
            "persona", level_0_content="", level_1_content="You are an expert."
        ),
        Factor.binary(
            "null_date_rule",
            level_0_content="",
            level_1_content="If no date is present, return null.",
        ),
    ]
    tmpl = PromptTemplate.from_factors(
        factors, system_prompt="Extract the date.", data_template="Text: {{ text }}"
    )

    cases = []
    # 20 baseline items (all have dates)
    for i in range(20):
        cases.append(
            TestCase(
                inputs={"text": f"Invoice dated 2026-03-{(i % 28) + 1:02d} (#{i})"},
                target="2026-03-01",
                checks=[CheckSpec(kind="regex", value=r"^2026-03-01$")],
                kind="baseline",
                tagger_factors=["persona"],
                split="discovery",
            )
        )
    # 20 targeted items for null_date_rule (no dates)
    for i in range(20):
        cases.append(
            TestCase(
                inputs={"text": f"Undated memo #{i}"},
                target="null",
                checks=[CheckSpec(kind="exact", value="null")],
                kind="targeted",
                intended_factor="null_date_rule",
                tagger_factors=["null_date_rule"],
                tagger_agrees=True,
                split="discovery",
            )
        )
    # 5 confirmation items (must stay untouched by default run!)
    for i in range(5):
        cases.append(
            TestCase(
                inputs={"text": f"Confirmation memo #{i}"},
                target="null",
                checks=[CheckSpec(kind="exact", value="null")],
                kind="targeted",
                intended_factor="null_date_rule",
                tagger_factors=["null_date_rule"],
                tagger_agrees=True,
                split="confirmation",
            )
        )

    draft_ts = TestSet.create(
        cases,
        factors=factors,
        template=tmpl,
        generator={"model_id": "gemini-2.5-flash"},
        freeze=False,
    )

    def sut_fn(prompt, kwargs):
        p = str(prompt)
        if "Undated memo" in p or "Confirmation memo" in p:
            return "null" if "If no date is present, return null." in p else "none"
        return "2026-03-01"

    sut_client = MockLLM(
        response_generator=sut_fn, latency_ms=0.0, model_id="gemini-2.5-flash"
    )
    exp = Experiment(
        factors=factors,
        template=tmpl,
        evaluator=[CheckSuiteMetric()],
        target_metric="check_suite",
    )

    # Running on unfrozen TestSet works, but diagnose() refuses unless frozen or allow_unfrozen=True
    results = exp.run(draft_ts, client=sut_client, max_workers=2)
    assert results.metadata["num_samples"] == 40  # 5 confirmation items untouched!
    assert results.metadata["model_id"] == "gemini-2.5-flash"

    with pytest.raises(ValueError, match="must be frozen"):
        exp.diagnose()

    # Freeze and re-run cleanly
    frozen_ts = draft_ts.freeze()
    exp.run(frozen_ts, client=sut_client, max_workers=2)
    audit = exp.diagnose(
        min_targeted_items=20, equivalence_margin=0.02, n_sign_flip=100
    )
    by_f = {f.factor_id: f for f in audit.factors}

    assert audit.n_items == 20  # baseline items only
    assert audit.baseline_only_full_set is True
    assert audit.test_set_sha256 == frozen_ts.manifest.cases_sha256
    assert audit.coverage_source == "synthetic"
    assert audit.evidence_tier == "predicted (synthetic tests)"
    assert audit.judge_scored_fraction == 0.0
    assert by_f["A"].pruning_verdict == "SAFE_TO_PRUNE"
    assert by_f["B"].pruning_verdict == "KEEP_TARGETED"
    assert by_f["B"].n_targeted == 20
    # Generator model == SUT model warning surfaced automatically
    assert any("Generator model" in w for w in audit.warnings)
    assert not any("Split leakage" in w for w in audit.warnings)

    # Running with split="all" triggers the split-leakage warning in diagnose()
    exp.run(frozen_ts, client=sut_client, max_workers=2, split="all")
    audit_leaked = exp.diagnose(min_targeted_items=20, n_sign_flip=0)
    assert any("Split leakage detected" in w for w in audit_leaked.warnings)


def test_pr2_review_hardening_guards(tmp_path: Path):
    from prompt_prism.evaluation.check_suite import _parse_judge_client_response
    from prompt_prism.evaluation.evaluator import Evaluator

    # 1. Judge response parser: fenced JSON, bool pass, PASS/FAIL, numeric, non-finite rejection
    assert _parse_judge_client_response(
        '```json\n{"pass": true, "reason": "ok"}\n```'
    ) == (
        1.0,
        "ok",
    )
    assert _parse_judge_client_response("PASS") == (1.0, "")
    assert _parse_judge_client_response("FAIL") == (0.0, "")
    assert _parse_judge_client_response("0.75") == (0.75, "")
    with pytest.raises(RuntimeError, match="non-finite"):
        _parse_judge_client_response('{"score": "Infinity"}')
    with pytest.raises(RuntimeError, match="non-finite"):
        _parse_judge_client_response('{"score": "NaN"}')

    # 2. judge_mode="pass" threshold behavior + cache hit
    calls = 0

    def judge_fn(prompt):
        nonlocal calls
        calls += 1
        return '{"score": 0.4, "reason": "below threshold"}'

    cache = JudgeCache(db_path=str(tmp_path / "pass_mode.db"))
    m_pass = CheckSuiteMetric(
        judge_client=judge_fn, judge_cache=cache, judge_mode="pass", judge_threshold=0.5
    )
    ctx = {"q": "hi", "checks": [{"kind": "llm_judge", "value": "be great"}]}
    assert m_pass.compute("out", input_data=ctx) == 0.0
    assert m_pass.compute("out", input_data=ctx) == 0.0
    assert calls == 1  # Second call hit cache in pass mode

    # 3. Evaluator.evaluate propagates PermissionError and ImportError (never swallows to 0.0)
    ev = Evaluator([CheckSuiteMetric(allow_custom_checks=False)])
    with pytest.raises(PermissionError, match="allow_custom_checks=True"):
        ev.evaluate(
            prediction="MAGIC",
            target=None,
            input_data={
                "checks": [
                    {"kind": "custom", "value": f"{__name__}:_sample_custom_scorer"}
                ]
            },
        )

    def _missing_dep(pred, target=None, input_data=None):
        raise ImportError("missing optional extra")

    ev_imp = Evaluator([_missing_dep])
    with pytest.raises(ImportError, match="missing optional extra"):
        ev_imp.evaluate("pred", "target")

    # 4. Experiment.estimate_judge_calls does not exhaust a one-shot generator
    factors = [Factor.binary("persona", level_0_content="", level_1_content="Expert.")]
    exp = Experiment.from_factors(
        factors=factors,
        data_template="{{ q }}",
        metrics=[CheckSuiteMetric(judge_client=judge_fn)],
        target_metric="check_suite",
    )
    gen = (
        {
            "id": f"g_{i}",
            "q": "hi",
            "checks": [{"kind": "llm_judge", "value": "rubric"}],
        }
        for i in range(3)
    )
    assert exp.estimate_judge_calls(gen) == len(exp.design.runs) * 3

    # 5. Experiment.run() fails fast on an in-memory tampered frozen TestSet before calling LLM
    tc = TestCase(
        inputs={"q": "hello"},
        target="ok",
        checks=[CheckSpec(kind="exact", value="ok")],
        kind="baseline",
        split="discovery",
    )
    ts = TestSet.create([tc], factors=factors, freeze=True)
    ts.cases[0].inputs["q"] = "tampered_in_memory"

    llm_calls = 0

    def spy_client(prompt):
        nonlocal llm_calls
        llm_calls += 1
        return "ok"

    with pytest.raises(ValueError, match="integrity check failed"):
        exp.run(ts, client=spy_client, max_workers=1)
    assert llm_calls == 0

    # 6. PromptTemplate.required_data_variables ignores stale sec.content when factor is bound
    tmpl_stale = PromptTemplate()
    tmpl_stale.add_section(
        id="persona_sec",
        factor_id="A",
        content="Stale {{ should_not_appear }}",
        position=0,
    )
    tmpl_stale.add_section(id="data_sec", content="{{ real_var }}", position=1)
    assert tmpl_stale.required_data_variables(factors) == {"real_var"}
