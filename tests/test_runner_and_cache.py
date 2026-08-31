"""
Tests for Runner, Cache, and Client modules.
"""

import time
import pytest
from prompt_prism.core.factors import Factor, FactorSet
from prompt_prism.core.models import DesignMatrix, RunConfig
from prompt_prism.design.generators import FractionalFactorialGenerator
from prompt_prism.evaluation.metrics import ExactMatch
from prompt_prism.runner.cache import ResponseCache
from prompt_prism.runner.client import CallableLLM, LLMResponse, MockLLM
from prompt_prism.runner.runner import ExperimentRunner
from prompt_prism.template.composer import PromptComposer, PromptTemplate


def test_response_cache_memory():
    cache = ResponseCache(enabled=True)
    res = LLMResponse(content="Cached output", latency_ms=50.0)
    
    assert cache.get("test prompt") is None
    cache.set("test prompt", res)
    cached_res = cache.get("test prompt")
    assert cached_res is not None
    assert cached_res.content == "Cached output"


def test_response_cache_sqlite(tmp_path):
    db_path = tmp_path / "cache.db"
    cache = ResponseCache(db_path=db_path, enabled=True)
    res = LLMResponse(content="SQLite output", latency_ms=30.0, token_usage={"tokens": 10})
    
    cache.set("prompt 1", res, params={"temp": 0.2})
    
    # Reload new cache instance from same db
    cache2 = ResponseCache(db_path=db_path, enabled=True)
    retrieved = cache2.get("prompt 1", params={"temp": 0.2})
    assert retrieved is not None
    assert retrieved.content == "SQLite output"
    assert retrieved.token_usage.get("tokens") == 10


def test_callable_llm_and_error_handling():
    def good_fn(prompt):
        return f"Response to: {prompt}"

    def error_fn(prompt):
        raise ValueError("API quota exceeded")

    client_good = CallableLLM(good_fn)
    resp = client_good.generate("hello")
    assert resp.content == "Response to: hello"
    assert resp.error is None

    client_err = CallableLLM(error_fn)
    resp_err = client_err.generate("hello")
    assert resp_err.error == "API quota exceeded"
    assert resp_err.content == ""


def test_experiment_runner_multithreaded():
    f1 = Factor.binary("f1", level_1_content="Feature 1")
    f_set = FactorSet([f1])
    tmpl = PromptTemplate.from_factors(f_set, data_template="Q: {{ q }}")
    composer = PromptComposer(tmpl, f_set)

    def simple_llm(p):
        return "42"

    runner = ExperimentRunner(
        composer=composer,
        client=simple_llm,
        evaluator=[ExactMatch()],
        max_workers=4,
    )

    design = DesignMatrix(
        plan_id="test_plan",
        factor_ids=["A"],
        runs=[
            RunConfig(run_id=1, factor_levels={"A": 0}),
            RunConfig(run_id=2, factor_levels={"A": 1}),
        ],
    )

    dataset = [{"id": 1, "q": "what is 6x7?", "target": "42"}, {"id": 2, "q": "what is 21x2?", "target": "42"}]
    results = runner.run(design=design, dataset=dataset)

    assert len(results.trials) == 4
    df = results.to_dataframe()
    assert len(df) == 4
    assert (df["exact_match"] == 1.0).all()
