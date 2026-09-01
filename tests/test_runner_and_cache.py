"""
Tests for Runner, Cache, and Client modules.
"""

from prompt_prism.core.factors import Factor, FactorSet
from prompt_prism.core.models import DesignMatrix, RunConfig
from prompt_prism.design.generators import FractionalFactorialGenerator
from prompt_prism.evaluation.metrics import ExactMatch
from prompt_prism.runner.cache import ResponseCache
from prompt_prism.runner.client import CallableLLM, LLMResponse
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
    res = LLMResponse(
        content="SQLite output", latency_ms=30.0, token_usage={"tokens": 10}
    )

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

    dataset = [
        {"id": 1, "q": "what is 6x7?", "target": "42"},
        {"id": 2, "q": "what is 21x2?", "target": "42"},
    ]
    results = runner.run(design=design, dataset=dataset)

    assert len(results.trials) == 4
    df = results.to_dataframe()
    assert len(df) == 4
    assert (df["exact_match"] == 1.0).all()


def test_runner_prompt_format_selection():
    # 1. Plain text template (no explicit roles)
    received_payloads = []

    def recording_client(p):
        received_payloads.append(p)
        return "OK"

    f1 = Factor.binary("f1", level_0_content="A", level_1_content="B")
    f_set = FactorSet([f1])
    tmpl_text = PromptTemplate.from_factors(f_set, data_template="Payload: {{ text }}")
    composer_text = PromptComposer(tmpl_text, f_set)

    runner_text = ExperimentRunner(
        composer=composer_text,
        client=recording_client,
        evaluator=[ExactMatch()],
    )
    design = DesignMatrix(
        plan_id="p1",
        factor_ids=["A"],
        runs=[RunConfig(run_id=1, factor_levels={"A": 0})],
    )
    res_text = runner_text.run(design=design, dataset=[{"id": 1, "text": "hello"}])

    assert len(received_payloads) == 1
    assert isinstance(received_payloads[0], str)
    assert isinstance(res_text.trials[0].prompt, str)
    assert "Payload: hello" in res_text.trials[0].prompt

    # 2. Chat messages template (explicit roles)
    received_chat_payloads = []

    def chat_recording_client(p):
        received_chat_payloads.append(p)
        return "OK"

    tmpl_chat = PromptTemplate()
    tmpl_chat.add_section(id="sys", content="System prompt", role="system")
    tmpl_chat.add_section(id="usr", content="User prompt: {{ text }}", role="user")
    composer_chat = PromptComposer(tmpl_chat)

    runner_chat = ExperimentRunner(
        composer=composer_chat,
        client=chat_recording_client,
        evaluator=[ExactMatch()],
    )
    res_chat = runner_chat.run(design=design, dataset=[{"id": 1, "text": "chat"}])

    assert len(res_chat.trials) == 1
    assert len(received_chat_payloads) == 1
    assert isinstance(received_chat_payloads[0], list)
    assert received_chat_payloads[0][0]["role"] == "system"


def test_runner_retry_limit_behavior():
    # 1. Soft failure (LLMResponse.error) that succeeds on retry
    calls = 0

    def flaky_soft_client(p):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(content="", error="Rate limit exceeded")
        return LLMResponse(content="Success output")

    tmpl = PromptTemplate.from_factors([Factor.binary("A")])
    f_set = FactorSet([Factor.binary("A")])
    composer = PromptComposer(tmpl, f_set)

    runner = ExperimentRunner(
        composer=composer,
        client=flaky_soft_client,
        evaluator=[ExactMatch()],
        retry_limit=2,
    )
    design = DesignMatrix(
        plan_id="p1",
        factor_ids=["A"],
        runs=[RunConfig(run_id=1, factor_levels={"A": 0})],
    )
    res = runner.run(design=design, dataset=[{"id": 1, "text": "test"}])

    assert len(res.trials) == 1
    assert res.trials[0].error is None
    assert res.trials[0].raw_response == "Success output"
    assert calls == 2

    # 2. Hard failure (Exception raised) that succeeds on retry
    hard_calls = 0

    def flaky_hard_client(p):
        nonlocal hard_calls
        hard_calls += 1
        if hard_calls == 1:
            raise ConnectionError("Network reset")
        return "Recovered"

    runner_hard = ExperimentRunner(
        composer=composer,
        client=flaky_hard_client,
        evaluator=[ExactMatch()],
        retry_limit=2,
    )
    res_hard = runner_hard.run(design=design, dataset=[{"id": 1, "text": "test"}])
    assert len(res_hard.trials) == 1
    assert res_hard.trials[0].error is None
    assert res_hard.trials[0].raw_response == "Recovered"
    assert hard_calls == 2

    # 3. Persistent failure exhausts retries
    fail_calls = 0

    def always_fail_client(p):
        nonlocal fail_calls
        fail_calls += 1
        raise ValueError("Permanent failure")

    runner_fail = ExperimentRunner(
        composer=composer,
        client=always_fail_client,
        evaluator=[ExactMatch()],
        retry_limit=2,
    )
    res_fail = runner_fail.run(design=design, dataset=[{"id": 1, "text": "test"}])
    assert len(res_fail.trials) == 1
    assert res_fail.trials[0].error == "Permanent failure"
    assert fail_calls == 3  # 1 initial + 2 retries


def test_to_dataframe_surfaces_metadata():
    # R20: Trial.metadata is flattened into DataFrame columns without breaking ANOVA
    from prompt_prism.analysis.anova import ANOVAEngine
    from prompt_prism.core.models import ExperimentResults, Trial

    design = FractionalFactorialGenerator.from_plan_id("2(3-1)III")
    trials = []
    for run in design.runs:
        for s_id in range(4):
            trials.append(
                Trial(
                    trial_id=f"t_{run.run_id}_{s_id}",
                    run_id=run.run_id,
                    sample_id=s_id,
                    factor_levels=dict(run.factor_levels),
                    metrics={
                        "exact_match": (
                            1.0 if run.factor_levels.get("A", 0) == 1 else 0.0
                        )
                    },
                    metadata={
                        "combination": f"comb_{run.run_id}",
                        "judge_reasons.exact_match": (
                            "Matched exactly."
                            if run.factor_levels.get("A", 0) == 1
                            else "Mismatch"
                        ),
                    },
                )
            )
    exp_res = ExperimentResults(experiment_id="test_meta", design=design, trials=trials)
    df = exp_res.to_dataframe()

    assert "combination" in df.columns
    assert "judge_reasons.exact_match" in df.columns
    assert df["combination"].iloc[0] == "comb_1"

    # ANOVA should still fit cleanly on the resulting frame even with string metadata columns present
    anova_res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=design.factor_ids,
        target_col="exact_match",
    )
    assert anova_res.r_squared > 0.9
