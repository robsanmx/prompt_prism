"""
Unit and Integration Tests for DeepEval Metric Adapter, JudgeCache, and Phase 0 Seams.
"""

import math
import sys
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.core.factors import Factor
from prompt_prism.core.models import DesignMatrix, RunConfig
from prompt_prism.evaluation.deepeval_metrics import DeepEvalMetric, deepeval_metric
from prompt_prism.evaluation.evaluator import Evaluator
from prompt_prism.evaluation.judge_cache import JudgeCache
from prompt_prism.evaluation.metrics import CustomMetric, ExactMatch, Metric
from prompt_prism.experiment import Experiment
from prompt_prism.runner.client import MockLLM
from prompt_prism.runner.runner import ExperimentRunner
from prompt_prism.template.composer import PromptComposer, PromptSection, PromptTemplate


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

    context_metric = CustomMetric(inspect_context, name="inspect_ctx", wants_prompt=True)
    evaluator = Evaluator([context_metric])

    factors = [Factor.binary("f1", level_0_content="A", level_1_content="B")]
    template = PromptTemplate(
        sections=[
            PromptSection(id="sec_sys", name="system", content="System: Hello"),
            PromptSection(id="sec_f1", name="f1", factor_id="f1"),
            PromptSection(id="sec_data", name="data", content="Data: {{ text }}"),
        ]
    )
    composer = PromptComposer(template, factors)

    runner = ExperimentRunner(composer=composer, client=MockLLM(default_response="OK"), evaluator=evaluator)
    design = DesignMatrix(plan_id="single", factor_ids=["f1"], runs=[RunConfig(run_id=1, factor_levels={"f1": 1})])
    dataset = [{"id": "item_1", "text": "Sample text", "target": "Target val"}]

    res = runner.run(design=design, dataset=dataset)
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
    def __init__(self, score=0.88, should_fail=False):
        self.score = score
        self.should_fail = should_fail
        self.measured_cases = []
        self.threshold = 0.5
        self.reason = "Clear reasoning and faithful facts."
        self.model = "default"

    def measure(self, test_case):
        if self.should_fail:
            raise RuntimeError("API Rate limit exceeded on judge")
        self.measured_cases.append(test_case)
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


def test_adapter_with_fake_judge():
    stub = StubDeepEvalJudge(score=0.92)
    metric = DeepEvalMetric(metric_factory=lambda: stub, name="stub_relevancy")

    # Mock deepeval.test_case import if not installed
    mock_tc_module = MagicMock()
    mock_tc_module.LLMTestCase = lambda **kw: MagicMock(**kw)
    sys.modules["deepeval"] = MagicMock()
    sys.modules["deepeval.test_case"] = mock_tc_module

    score = metric.compute(
        prediction="Berlin is the capital of Germany.",
        target="Berlin",
        input_data={"input": "What is the capital of Germany?"},
    )
    assert score == 0.92
    assert len(stub.measured_cases) == 1


def test_fresh_instance_per_call():
    instances_created = []

    def factory():
        inst = StubDeepEvalJudge(score=0.75)
        instances_created.append(inst)
        return inst

    metric = DeepEvalMetric(metric_factory=factory, name="threadsafe_judge")
    metric.compute("pred 1", "target")
    metric.compute("pred 2", "target")

    assert len(instances_created) == 2
    assert instances_created[0] is not instances_created[1]


def test_judge_failure_is_nan_and_anova_survives():
    failing_stub = StubDeepEvalJudge(should_fail=True)
    metric = DeepEvalMetric(metric_factory=lambda: failing_stub, name="failing_judge")

    score = metric.compute("pred", "target")
    assert math.isnan(score)

    # Test that ANOVAEngine drops NaN and still fits without crashing
    df = pd.DataFrame({
        "A": [0, 1, 0, 1, 0, 1, 0, 1],
        "B": [0, 0, 1, 1, 0, 0, 1, 1],
        "target_score": [0.8, 0.9, np.nan, 0.85, 0.7, 0.95, 0.6, 0.9],
        "sample_id": [1, 2, 3, 4, 1, 2, 3, 4],
    })
    anova_res = ANOVAEngine.run_anova(data=df, factor_cols=["A", "B"], target_col="target_score")
    assert anova_res.r_squared > 0.0
    assert len(anova_res.main_effects) == 2


def test_judge_cache_hit(tmp_path):
    db_file = str(tmp_path / "judge_test.db")
    cache = JudgeCache(db_path=db_file)

    call_count = [0]

    def counting_factory():
        call_count[0] += 1
        return StubDeepEvalJudge(score=0.95)

    metric = DeepEvalMetric(metric_factory=counting_factory, name="cached_judge", cache=cache, judge_model_id="default")

    # Call 1: Miss
    score1 = metric.compute(prediction="Output A", target="Target A", input_data={"input": "Prompt A"})
    assert score1 == 0.95
    assert call_count[0] == 1

    # Call 2: Hit (factory not invoked)
    score2 = metric.compute(prediction="Output A", target="Target A", input_data={"input": "Prompt A"})
    assert score2 == 0.95
    assert call_count[0] == 1  # Still 1!


def test_mode_pass():
    stub_high = StubDeepEvalJudge(score=0.85)
    stub_low = StubDeepEvalJudge(score=0.30)

    metric_pass = DeepEvalMetric(metric_factory=lambda: stub_high, name="pass_metric", mode="pass")
    metric_fail = DeepEvalMetric(metric_factory=lambda: stub_low, name="fail_metric", mode="pass")

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
    # 4 runs * 10 samples * 2 judge metrics = 80
    assert exp.estimate_judge_calls(dataset) == 80


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
