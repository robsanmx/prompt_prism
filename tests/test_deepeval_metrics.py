"""
Unit and Integration Tests for DeepEval Metric Adapter, JudgeCache, and Phase 0 Seams.
"""

import importlib.util
import math

import pandas as pd
import pytest

from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.analysis.optimizer import OptimalPromptFinder
from prompt_prism.core.factors import Factor
from prompt_prism.core.models import DesignMatrix, RunConfig
from prompt_prism.evaluation.deepeval_metrics import (
    DEEPEVAL_METRIC_SPECS,
    DeepEvalMetric,
    deepeval_metric,
)
from prompt_prism.evaluation.evaluator import Evaluator
from prompt_prism.evaluation.judge_cache import JudgeCache
from prompt_prism.evaluation.metrics import CustomMetric, ExactMatch
from prompt_prism.experiment import Experiment
from prompt_prism.runner.client import MockLLM
from prompt_prism.runner.runner import ExperimentRunner
from prompt_prism.template.composer import PromptComposer, PromptSection, PromptTemplate

HAS_DEEPEVAL = importlib.util.find_spec("deepeval") is not None

requires_deepeval = pytest.mark.skipif(
    not HAS_DEEPEVAL, reason="DeepEval metrics require the 'deepeval' extra."
)

# ---------------------------------------------------------------------------
# Phase 0: Seam Tests (Evaluator on_error, duplicates, prompt in context)
# ---------------------------------------------------------------------------


def test_evaluator_duplicate_metric_name_raises():
    evaluator = Evaluator([ExactMatch(name="exact_match")])
    with pytest.raises(ValueError, match="Duplicate metric name 'exact_match'"):
        evaluator.add_metric(ExactMatch(name="exact_match"))


def test_evaluator_on_error_handling():
    def flaky_fn(pred, target=None, **kw):
        raise RuntimeError("LLM Judge timeout")

    judge_metric = CustomMetric(flaky_fn, name="flaky_judge", is_llm_judge=True)
    local_metric = CustomMetric(flaky_fn, name="flaky_local", is_llm_judge=False)

    # 1. Default: LLM judge fails to NaN, local fails to 0.0
    evaluator_auto = Evaluator([judge_metric, local_metric])
    scores_auto = evaluator_auto.evaluate("pred", "target")
    assert math.isnan(scores_auto["flaky_judge"])
    assert scores_auto["flaky_local"] == 0.0
    assert "LLM Judge timeout" in evaluator_auto.last_errors["flaky_judge"]

    # 2. on_error="raise"
    evaluator_raise = Evaluator([judge_metric], on_error="raise")
    with pytest.raises(RuntimeError, match="LLM Judge timeout"):
        evaluator_raise.evaluate("pred", "target")

    # 3. on_error="zero"
    evaluator_zero = Evaluator([judge_metric], on_error="zero")
    scores_zero = evaluator_zero.evaluate("pred", "target")
    assert scores_zero["flaky_judge"] == 0.0


def test_runner_passes_prompt_in_context():
    seen_contexts = []

    def inspect_context(pred, target=None, input_data=None, **kw):
        seen_contexts.append(dict(input_data or {}))
        return 1.0

    context_metric = CustomMetric(
        inspect_context, name="inspect_ctx", wants_prompt=True
    )
    evaluator = Evaluator([context_metric])

    factors = [Factor.binary("f1", level_0_content="A", level_1_content="B")]
    template = PromptTemplate(
        sections=[
            PromptSection(id="sec_sys", content="System: Hello"),
            PromptSection(id="sec_f1", factor_id="f1"),
            PromptSection(id="sec_data", content="Data: {{ text }}"),
        ]
    )
    composer = PromptComposer(template, factors)

    runner = ExperimentRunner(
        composer=composer, client=MockLLM(default_response="OK"), evaluator=evaluator
    )
    design = DesignMatrix(
        plan_id="single",
        factor_ids=["f1"],
        runs=[RunConfig(run_id=1, factor_levels={"f1": 1})],
    )
    dataset = [{"id": "item_1", "text": "Sample text", "target": "Target val"}]

    res = runner.run(design=design, dataset=dataset)
    assert len(res.trials) == 1
    assert len(seen_contexts) == 1
    ctx = seen_contexts[0]
    assert "__prompt__" in ctx
    assert "System: Hello" in ctx["__prompt__"]
    assert ctx["__run_id__"] == 1
    assert ctx["__sample_id__"] == "item_1"
    assert ctx["text"] == "Sample text"


# ---------------------------------------------------------------------------
# Phase 1 & 2: DeepEval Adapter & JudgeCache Tests (Deterministic Stubs)
# ---------------------------------------------------------------------------


class StubDeepEvalJudge:
    def __init__(
        self,
        score=0.88,
        should_fail=False,
        reason="Clear reasoning and faithful facts.",
    ):
        self.score = score
        self.should_fail = should_fail
        self.measured_cases = []
        self.threshold = 0.5
        self.reason = reason
        self.model = "default"

    def measure(self, test_case):
        if self.should_fail:
            raise RuntimeError("API Rate limit exceeded on judge")
        self.measured_cases.append(test_case)
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


@requires_deepeval
def test_adapter_with_fake_judge(monkeypatch):
    stub = StubDeepEvalJudge(score=0.92)
    metric = DeepEvalMetric(metric_factory=lambda: stub, name="stub_relevancy")

    score = metric.compute(
        prediction="Berlin is the capital of Germany.",
        target="Berlin",
        input_data={"input": "What is the capital of Germany?"},
    )
    assert score == 0.92
    assert len(stub.measured_cases) == 1
    tc = stub.measured_cases[0]
    assert tc.input == "What is the capital of Germany?"
    assert tc.actual_output == "Berlin is the capital of Germany."
    assert tc.expected_output == "Berlin"


@requires_deepeval
def test_context_and_retrieval_context_fallbacks():
    # R4 & R24: Fallbacks for input (question/query) and retrieval_context (context/documents)
    recorded_cases = []

    def factory():
        stub = StubDeepEvalJudge(score=0.90)

        def measure_override(tc):
            recorded_cases.append(tc)
            return 0.90

        stub.measure = measure_override
        return stub

    # Test 1: explicit retrieval_context key in row
    m1 = deepeval_metric("faithfulness", name="faith_1")
    m1.metric_factory = factory
    m1.compute(
        "pred", input_data={"query": "q1", "retrieval_context": ["doc1", "doc2"]}
    )

    assert len(recorded_cases) == 1
    tc1 = recorded_cases[0]
    assert tc1.input == "q1"
    assert tc1.retrieval_context == ["doc1", "doc2"]

    # Test 2: fallback to 'context' when retrieval_context absent
    m2 = deepeval_metric("faithfulness", name="faith_2")
    m2.metric_factory = factory
    m2.compute("pred", input_data={"question": "q2", "context": "doc_single"})

    assert len(recorded_cases) == 2
    tc2 = recorded_cases[1]
    assert tc2.input == "q2"
    assert tc2.retrieval_context == ["doc_single"]

    # Test 3: explicit key present with value None falls back cleanly
    m3 = deepeval_metric(
        "faithfulness", name="faith_3", retrieval_context_key="custom_rc"
    )
    m3.metric_factory = factory
    m3.compute(
        "pred",
        input_data={"question": "q3", "custom_rc": None, "context": ["doc_fallback"]},
    )
    assert len(recorded_cases) == 3
    tc3 = recorded_cases[2]
    assert tc3.input == "q3"
    assert tc3.retrieval_context is None


@requires_deepeval
def test_judge_cache_content_addressing(tmp_path):
    # R5: Key covers contexts, criteria, and expected output
    db_file = str(tmp_path / "judge_test.db")
    cache = JudgeCache(db_path=db_file)

    call_count = 0

    def counting_factory():
        nonlocal call_count
        call_count += 1
        return StubDeepEvalJudge(score=0.95)

    metric = DeepEvalMetric(
        metric_factory=counting_factory,
        name="cached_judge",
        cache=cache,
        judge_model_id="default",
    )

    # Call 1: Context A
    s1 = metric.compute("pred", "target", {"input": "q", "retrieval_context": ["docA"]})
    assert s1 == 0.95
    assert call_count == 1

    # Call 2: Identical inputs -> Cache Hit
    s2 = metric.compute("pred", "target", {"input": "q", "retrieval_context": ["docA"]})
    assert s2 == 0.95
    assert call_count == 1

    # Call 3: Different context -> Cache Miss
    s3 = metric.compute("pred", "target", {"input": "q", "retrieval_context": ["docB"]})
    assert s3 == 0.95
    assert call_count == 2

    # Call 4: Different expected_output -> Cache Miss
    s4 = metric.compute(
        "pred", "different_target", {"input": "q", "retrieval_context": ["docA"]}
    )
    assert s4 == 0.95
    assert call_count == 3

    # Call 5: Non-string judge_model_id raises TypeError
    with pytest.raises(TypeError, match="judge_model_id must be a string"):
        cache.get(metric_name="m", judge_model_id=12345)


def test_judge_cache_schema_v1_migration(tmp_path):
    # R17/R38: a pre-existing database keeps working. R17 achieved this with a
    # schema_version column and an ALTER TABLE migration; R38 removed both, because
    # _hash_key already prefixes every key with v{SCHEMA_VERSION}. This test is the proof
    # that the simpler design needs no migration at all.
    import sqlite3

    db_file = str(tmp_path / "v1_cache.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE judge_cache (
                hash_key TEXT PRIMARY KEY,
                metric_name TEXT,
                judge_model TEXT,
                score REAL,
                reason TEXT,
                success INTEGER,
                token_usage TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    cache1 = JudgeCache(db_path=db_file)
    cache1.set(
        metric_name="faithfulness",
        input_text="q",
        actual_output="ans",
        score=0.91,
        reason="Good reasoning",
    )

    # Empty in-memory tier in second cache to verify persistence in SQLite
    cache2 = JudgeCache(db_path=db_file)
    res = cache2.get(
        metric_name="faithfulness",
        input_text="q",
        actual_output="ans",
    )
    assert res is not None
    assert res.score == 0.91
    assert res.reason == "Good reasoning"


@requires_deepeval
def test_judge_exception_propagation_to_evaluator():
    # R6: Exceptions propagate to Evaluator
    failing_stub = StubDeepEvalJudge(should_fail=True)
    metric = DeepEvalMetric(metric_factory=lambda: failing_stub, name="failing_judge")

    # DeepEvalMetric directly raises
    with pytest.raises(RuntimeError, match="API Rate limit exceeded"):
        metric.compute("pred", "target")

    # Evaluator handles it according to on_error
    ev_nan = Evaluator([metric], on_error="nan")
    res = ev_nan.evaluate("pred", "target")
    assert math.isnan(res["failing_judge"])
    assert "API Rate limit exceeded" in ev_nan.last_errors["failing_judge"]


@requires_deepeval
def test_fresh_instance_per_call():
    # R21: Metric factory produces a fresh instance per compute call
    instances = []

    def factory():
        inst = StubDeepEvalJudge(score=0.9)
        instances.append(inst)
        return inst

    metric = DeepEvalMetric(metric_factory=factory, name="fresh_metric")
    metric.compute("pred1", "target1", {"input": "q1"})
    metric.compute("pred2", "target2", {"input": "q2"})

    assert len(instances) == 2
    assert instances[0] is not instances[1]


@requires_deepeval
def test_judge_failure_is_nan_and_anova_survives():
    # R21: Raising judge yields NaN through Evaluator and ANOVA fits on surviving rows
    failing_stub = StubDeepEvalJudge(should_fail=True)
    failing_metric = DeepEvalMetric(
        metric_factory=lambda: failing_stub, name="failing_judge"
    )
    exact_m = ExactMatch(name="exact")
    evaluator = Evaluator([failing_metric, exact_m], on_error="nan")

    factors = [Factor.binary("A")]
    design = DesignMatrix(
        plan_id="p1",
        factor_ids=["A"],
        runs=[
            RunConfig(run_id=1, factor_levels={"A": 0}),
            RunConfig(run_id=2, factor_levels={"A": 1}),
        ],
    )
    dataset = [
        {"id": 1, "text": "item 1", "target": "item 1"},
        {"id": 2, "text": "item 2", "target": "item 2"},
    ]
    tmpl = PromptTemplate.from_factors(factors, data_template="{{ text }}")
    composer = PromptComposer(tmpl, factors)
    runner = ExperimentRunner(
        composer=composer,
        client=MockLLM(default_response="item 1"),
        evaluator=evaluator,
    )
    res = runner.run(design=design, dataset=dataset)
    df = res.to_dataframe()
    assert df["failing_judge"].isna().all()
    assert not df["exact"].isna().any()

    # ANOVA fits without error on the surviving exact metric
    anova_res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A"],
        target_col="exact",
    )
    assert anova_res.r_squared is not None


def test_deepeval_specs_and_directions():
    # R31: every supported deepeval (>=4.2.0) scores "1 is a pass", including the three
    # that were reversed in 4.2.0. Through 4.1.10 these scored the proportion of
    # violations; encoding that older direction makes the optimizer pick the worse prompt.
    assert DEEPEVAL_METRIC_SPECS["hallucination"]["higher_is_better"] is True
    assert DEEPEVAL_METRIC_SPECS["toxicity"]["higher_is_better"] is True
    assert DEEPEVAL_METRIC_SPECS["bias"]["higher_is_better"] is True
    assert DEEPEVAL_METRIC_SPECS["answer_relevancy"]["higher_is_better"] is True
    assert DEEPEVAL_METRIC_SPECS["faithfulness"]["higher_is_better"] is True

    assert deepeval_metric("toxicity").higher_is_better is True
    assert deepeval_metric("hallucination").higher_is_better is True
    assert deepeval_metric("bias").higher_is_better is True

    # An explicit override still wins, for anyone wrapping a custom-scored judge.
    assert deepeval_metric("toxicity", higher_is_better=False).higher_is_better is False

    with pytest.raises(ValueError, match="Unknown DeepEval metric kind"):
        deepeval_metric("non_existent_kind")


def test_toxicity_recommendation_points_at_the_cleaner_prompt():
    """R31 regression: the sign of the recommendation, end to end.

    A stubbed toxicity judge scores level 1 *cleaner* than level 0. Under deepeval
    >=4.2.0 semantics a higher toxicity score means less toxic, so the optimizer must
    select level 1 and label it ENABLE. With higher_is_better=False (the pre-4.2.0
    direction this branch used to encode) it selects level 0 - the more toxic prompt.
    """
    tox = deepeval_metric("toxicity", name="toxicity")
    assert tox.higher_is_better is True

    # Level 1 scores 0.9 (clean), level 0 scores 0.1 (toxic).
    rows = []
    for run_id, level, score in ((1, 0, 0.1), (2, 1, 0.9)):
        for sample in range(4):
            jitter = 0.01 * sample
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample,
                    "A": level,
                    "toxicity": score + jitter,
                }
            )
    df = pd.DataFrame(rows)

    anova_res = ANOVAEngine.run_anova(data=df, factor_cols=["A"], target_col="toxicity")
    opt = OptimalPromptFinder.find_optimal_prompt(
        anova_res, maximize=tox.higher_is_better
    )

    assert opt.optimal_factor_levels["A"] == 1, (
        "optimizer chose the more toxic prompt - deepeval >=4.2.0 scores the absence "
        "of toxicity, so the higher-scoring level is the cleaner one"
    )
    effect_a = next(e for e in anova_res.main_effects if e.factor_id == "A")
    assert effect_a.action_recommendation == "ENABLE"
    assert opt.expected_gain_absolute > 0


def test_json_correctness_schema_requirement():
    # R22: Require explicit expected_schema or schema alias for json_correctness
    with pytest.raises(ValueError, match="requires an explicit 'expected_schema'"):
        deepeval_metric("json_correctness")

    with pytest.raises(ValueError, match="requires an explicit 'expected_schema'"):
        deepeval_metric("json")

    from pydantic import BaseModel

    class MySchema(BaseModel):
        val: int

    m1 = deepeval_metric("json_correctness", schema=MySchema)
    assert m1.name == "deepeval_json_correctness"

    m2 = deepeval_metric("json_correctness", expected_schema=MySchema)
    assert m2.name == "deepeval_json_correctness"


# deepeval >=4.2.0 emits a DeprecationWarning when hallucination/toxicity/bias are
# constructed, announcing the score-direction reversal. We deliberately target the
# post-4.2.0 direction (see DEEPEVAL_METRIC_SPECS and the >=4.2.0,<5 pin), so the notice
# is expected here rather than a signal of misconfiguration. Filtered so it does not
# drown the suite's warning summary; the direction itself is asserted by
# test_deepeval_specs_and_directions and the optimizer regression test above.
@pytest.mark.filterwarnings(
    "ignore:.*now scores in the same direction.*:DeprecationWarning"
)
def test_deepeval_construction_all_kinds():
    # Test construction of all supported kinds with a DeepEvalBaseLLM dummy model
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError:
        pytest.skip("deepeval not installed")

    from pydantic import BaseModel

    class DummySchema(BaseModel):
        val: str

    class DummyJudgeLLM(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = "dummy-judge"

        def load_model(self):
            return None

        def generate(self, prompt: str) -> str:
            return "0.85"

        async def a_generate(self, prompt: str) -> str:
            return "0.85"

        def get_model_name(self):
            return self.model_name

    dummy_model = DummyJudgeLLM()
    for kind in DEEPEVAL_METRIC_SPECS:
        extra_kwargs = {}
        if kind in {"json", "json_correctness"}:
            extra_kwargs["expected_schema"] = DummySchema
        m = deepeval_metric(
            kind,
            model=dummy_model,
            criteria="helpful and concise",
            **extra_kwargs,
        )
        inst = m.metric_factory()
        assert inst is not None


@requires_deepeval
def test_reason_capture_in_trial_metadata(tmp_path):
    # R10: Judge reason lands in Trial.metadata and survives cache hits
    db_file = str(tmp_path / "judge_reasons.db")
    cache = JudgeCache(db_path=db_file)

    stub = StubDeepEvalJudge(score=0.88, reason="Accurate and grounded facts.")
    judge_m = DeepEvalMetric(
        metric_factory=lambda: stub, name="faithfulness", cache=cache
    )

    f1 = Factor.binary("f1", level_0_content="A", level_1_content="B")
    tmpl = PromptTemplate.from_factors([f1], data_template="Q: {{ q }}")
    composer = PromptComposer(tmpl, [f1])

    runner = ExperimentRunner(
        composer=composer,
        client=MockLLM(default_response="Response 42"),
        evaluator=[judge_m],
    )
    design = DesignMatrix(
        plan_id="p1",
        factor_ids=["f1"],
        runs=[RunConfig(run_id=1, factor_levels={"f1": 0})],
    )
    dataset = [{"id": 1, "q": "test q", "retrieval_context": ["doc1"]}]

    # Run 1: Cold judge execution
    res1 = runner.run(design=design, dataset=dataset)
    assert len(res1.trials) == 1
    assert (
        res1.trials[0].metadata.get("judge_reasons.faithfulness")
        == "Accurate and grounded facts."
    )

    # Run 2: Cache hit - reason is still present in metadata
    res2 = runner.run(design=design, dataset=dataset)
    assert len(res2.trials) == 1
    assert (
        res2.trials[0].metadata.get("judge_reasons.faithfulness")
        == "Accurate and grounded facts."
    )


def test_mode_pass():
    stub_high = StubDeepEvalJudge(score=0.85)
    stub_low = StubDeepEvalJudge(score=0.30)

    metric_pass = DeepEvalMetric(
        metric_factory=lambda: stub_high, name="pass_metric", mode="pass"
    )
    metric_fail = DeepEvalMetric(
        metric_factory=lambda: stub_low, name="fail_metric", mode="pass"
    )

    # Note: StubDeepEvalJudge without _build_test_case or when deepeval is available
    if not HAS_DEEPEVAL:
        # Mock _build_test_case if deepeval absent
        metric_pass._build_test_case = lambda p, t, d: None
        metric_fail._build_test_case = lambda p, t, d: None

    assert metric_pass.compute("pred") == 1.0
    assert metric_fail.compute("pred") == 0.0


def test_estimate_judge_calls():
    stub = StubDeepEvalJudge()
    judge_m1 = DeepEvalMetric(metric_factory=lambda: stub, name="judge_1")
    judge_m2 = DeepEvalMetric(metric_factory=lambda: stub, name="judge_2")
    exact_m = ExactMatch()

    factors = [Factor.binary("f1"), Factor.binary("f2"), Factor.binary("f3")]
    exp = Experiment.from_factors(
        factors=factors,
        design="2(3-1)III",  # 4 runs
        metrics=[judge_m1, judge_m2, exact_m],
    )

    dataset = [{"id": i, "text": f"item {i}"} for i in range(10)]
    assert exp.estimate_judge_calls(dataset) == 80


@requires_deepeval
def test_end_to_end_with_stub_judge():
    stub = StubDeepEvalJudge(score=0.85)
    judge_m = DeepEvalMetric(metric_factory=lambda: stub, name="faithfulness")

    factors = [
        Factor.binary("persona", level_0_content="Standard", level_1_content="Auditor"),
        Factor.binary("few_shot", level_0_content="", level_1_content="Ex 1..."),
        Factor.binary("cot", level_0_content="", level_1_content="Step by step"),
    ]
    exp = Experiment.from_factors(
        factors=factors,
        design="2(3-1)III",  # 4 runs
        metrics=[judge_m],
        target_metric="faithfulness",
    )

    dataset = [
        {"id": 1, "text": "Doc 1"},
        {"id": 2, "text": "Doc 2"},
    ]
    client = MockLLM(default_response="Faithful extraction response")

    results = exp.run(dataset=dataset, client=client)
    assert len(results.trials) == 8  # 4 runs * 2 samples

    report = exp.analyze(block_by="sample_id")
    assert report.anova_result.target_metric == "faithfulness"
    assert len(report.anova_result.main_effects) == 3


def test_cache_key_covers_factory_kwargs(tmp_path):
    """R33: metric config beyond the three named fields must reach the cache key.

    Two json_correctness metrics with the same name and threshold but different
    expected_schema must not share a cache entry - before this, tightening a schema and
    rerunning served the old schema's scores.
    """
    from pydantic import BaseModel

    class SchemaA(BaseModel):
        val: int

    class SchemaB(BaseModel):
        val: str
        extra: int = 0

    m_a = deepeval_metric("json_correctness", name="jc", expected_schema=SchemaA)
    m_b = deepeval_metric("json_correctness", name="jc", expected_schema=SchemaB)
    m_a2 = deepeval_metric("json_correctness", name="jc", expected_schema=SchemaA)

    cfg_a, cfg_b, cfg_a2 = (
        m_a._get_config_str(),
        m_b._get_config_str(),
        m_a2._get_config_str(),
    )
    assert (
        cfg_a != cfg_b
    ), "different expected_schema must produce a different cache key"
    assert (
        cfg_a == cfg_a2
    ), "identical config must produce a stable key, or nothing caches"

    # evaluation_params / evaluation_steps also change the key
    g1 = deepeval_metric("g_eval", name="g", criteria="c", evaluation_steps=["one"])
    g2 = deepeval_metric("g_eval", name="g", criteria="c", evaluation_steps=["two"])
    assert g1._get_config_str() != g2._get_config_str()

    # No memory addresses: the key must be reproducible in another process.
    assert "0x" not in cfg_a

    # End to end through a real SQLite cache: two rows, not one.
    cache = JudgeCache(db_path=str(tmp_path / "judge.db"))
    for metric, score in ((m_a, 0.25), (m_b, 0.75)):
        cache.set(
            metric_name=metric.name,
            metric_config=metric._get_config_str(),
            judge_model_id=metric.judge_model_id,
            input_text="i",
            actual_output="o",
            expected_output="e",
            score=score,
        )
    got_a = cache.get(
        metric_name=m_a.name,
        metric_config=m_a._get_config_str(),
        judge_model_id=m_a.judge_model_id,
        input_text="i",
        actual_output="o",
        expected_output="e",
    )
    got_b = cache.get(
        metric_name=m_b.name,
        metric_config=m_b._get_config_str(),
        judge_model_id=m_b.judge_model_id,
        input_text="i",
        actual_output="o",
        expected_output="e",
    )
    assert got_a is not None and got_b is not None
    assert got_a.score == 0.25 and got_b.score == 0.75


def test_cache_key_stable_for_unserializable_judge_model():
    """R33: a judge model instance must not leak its address into the key."""

    class FakeJudge:
        model_name = "judge-x"

    m1 = deepeval_metric("answer_relevancy", name="ar", model=FakeJudge())
    m2 = deepeval_metric("answer_relevancy", name="ar", model=FakeJudge())
    assert m1._get_config_str() == m2._get_config_str()
    assert "0x" not in m1._get_config_str()


def test_judge_cache_works_against_db_with_legacy_schema_version_column(tmp_path):
    """R38: databases written by the R17 build carry a schema_version column.

    The INSERT no longer names it; naming fewer columns than the table has is valid SQL,
    so those databases must keep working without a migration.
    """
    import sqlite3

    db_file = str(tmp_path / "legacy_v2.db")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE judge_cache (
                hash_key TEXT PRIMARY KEY,
                schema_version INTEGER DEFAULT 2,
                metric_name TEXT,
                judge_model TEXT,
                score REAL,
                reason TEXT,
                success INTEGER,
                token_usage TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    cache = JudgeCache(db_path=db_file)
    cache.set(
        metric_name="faithfulness",
        metric_config="{}",
        judge_model_id="gpt-4o",
        input_text="q",
        actual_output="a",
        expected_output="e",
        score=0.8,
        reason="grounded",
    )
    # A fresh instance must read it back from disk, not from the in-memory tier.
    got = JudgeCache(db_path=db_file).get(
        metric_name="faithfulness",
        metric_config="{}",
        judge_model_id="gpt-4o",
        input_text="q",
        actual_output="a",
        expected_output="e",
    )
    assert got is not None and got.score == 0.8


@requires_deepeval
def test_evaluator_does_not_mutate_input_data():
    """R39: judge reasons travel on an explicit channel, not the caller's dict."""
    import copy

    from prompt_prism.evaluation.deepeval_metrics import DeepEvalMetric
    from prompt_prism.evaluation.evaluator import Evaluator
    from prompt_prism.evaluation.metrics import ExactMatch

    class StubJudge:
        def __init__(self):
            self.score = 0.75
            self.reason = "looks grounded"

        def measure(self, test_case):
            return self.score

        def is_successful(self):
            return True

    judge = DeepEvalMetric(metric_factory=lambda: StubJudge(), name="judge")
    evaluator = Evaluator([judge, ExactMatch(name="exact")])

    payload = {"input": "q", "target": "a", "nested": {"k": [1, 2]}}
    snapshot = copy.deepcopy(payload)

    scores = evaluator.evaluate(prediction="a", target="a", input_data=payload)

    assert payload == snapshot, "evaluate() must not write into the caller's input_data"
    assert scores["judge"] == 0.75
    assert evaluator.last_reasons == {"judge": "looks grounded"}

    # Deterministic metrics contribute no reason.
    assert ExactMatch(name="e").pop_reason() is None


@requires_deepeval
def test_judge_reasons_are_thread_local():
    """R39: the runner evaluates trials concurrently; reasons must not cross threads."""
    import threading

    from prompt_prism.evaluation.deepeval_metrics import DeepEvalMetric
    from prompt_prism.evaluation.evaluator import Evaluator

    class EchoJudge:
        def __init__(self, reason):
            self.score = 1.0
            self.reason = reason

        def measure(self, test_case):
            return self.score

        def is_successful(self):
            return True

    metric = DeepEvalMetric(
        metric_factory=lambda: EchoJudge(threading.current_thread().name),
        name="judge",
    )
    evaluator = Evaluator([metric])
    seen = {}

    def work(name):
        evaluator.evaluate(prediction="p", target="t", input_data={"input": "i"})
        seen[name] = evaluator.last_reasons.get("judge")

    threads = [
        threading.Thread(target=work, args=(f"w{i}",), name=f"w{i}") for i in range(4)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert seen == {f"w{i}": f"w{i}" for i in range(4)}
