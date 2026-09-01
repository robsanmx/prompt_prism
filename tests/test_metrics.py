import math

from prompt_prism.evaluation.evaluator import Evaluator
from prompt_prism.evaluation.metrics import (
    CustomMetric,
    ExactMatch,
    F1Score,
    JSONValidation,
    KeyValuesExtractionOverlap,
    LevenshteinSimilarity,
)


def test_exact_match():
    em = ExactMatch(case_sensitive=False, strip=True)
    assert em.compute("Paris", "paris") == 1.0
    assert em.compute("Paris ", "paris") == 1.0
    assert em.compute("London", "paris") == 0.0


def test_f1_score():
    f1 = F1Score()
    assert f1.compute("the quick brown fox", "the fast brown fox") > 0.6
    assert f1.compute("exact same text", "exact same text") == 1.0
    assert f1.compute("completely different", "apples oranges") == 0.0


def test_json_validation():
    jv = JSONValidation(required_keys=["brand", "color"])
    assert jv.compute('{"brand": "Nike", "color": "red"}') == 1.0
    assert jv.compute('```json\n{"brand": "Nike", "color": "red"}\n```') == 1.0
    assert jv.compute('{"brand": "Nike"}') == 0.5
    assert jv.compute("Not json") == 0.0


def test_extraction_overlap():
    overlap = KeyValuesExtractionOverlap()
    pred = {"brand": "Nike", "model": "Air Max", "extra": "123"}
    target = {"brand": "Nike", "model": "Air Max", "size": "42"}

    score = overlap.compute(pred, target)
    assert 0.65 < score < 0.75  # P = 2/3, R = 2/3, F1 = 2/3 ≈ 0.667


def test_levenshtein_similarity():
    lev = LevenshteinSimilarity()
    assert lev.compute("kitten", "sitting") > 0.5
    assert lev.compute("same", "same") == 1.0
    assert lev.compute("", "") == 1.0
    assert lev.compute("abc", "") == 0.0
    assert lev.compute("", "abc") == 0.0
    assert lev.compute("disjoint", "xyz") == 0.0
    assert lev.compute("café", "cafe") == 0.75

    # Test long strings linear-space
    s1 = "a" * 4000
    s2 = "a" * 3999 + "b"
    assert lev.compute(s1, s2) == 3999 / 4000


def test_json_validation_dict_and_embedded():
    jv = JSONValidation(required_keys=["brand", "color"])
    # Dict input without str() round-trip failure
    assert jv.compute({"brand": "Nike", "color": "red"}) == 1.0
    assert jv.compute({"brand": "Nike"}) == 0.5

    # List input
    jv_no_keys = JSONValidation()
    assert jv_no_keys.compute([1, 2, 3]) == 1.0
    assert jv_no_keys.compute("[1, 2, 3]") == 1.0

    # JSON embedded in prose
    prose_json = 'Here is the extracted data: {"brand": "Nike", "color": "red"} and additional notes.'
    assert jv.compute(prose_json) == 1.0


def test_metric_modes_and_validation():
    import pytest

    with pytest.raises(ValueError, match="Unknown mode"):
        F1Score(mode="invalid_mode")

    with pytest.raises(ValueError, match="Unknown mode"):
        KeyValuesExtractionOverlap(mode="invalid_mode")

    overlap = KeyValuesExtractionOverlap(mode="precision")
    pred = {"brand": "Nike", "model": "Air Max", "extra": "123"}
    target = {"brand": "Nike", "model": "Air Max", "size": "42"}
    assert overlap.compute(pred, target) == 2 / 3


def test_custom_metric_introspection_and_errors():
    import pytest

    # score_fn deprecated alias
    with pytest.deprecated_call():
        cm_dep = CustomMetric(score_fn=lambda p: 1.0)
        assert cm_dep.compute("test") == 1.0

    # Missing callable raises TypeError
    with pytest.raises(TypeError, match="missing required argument"):
        CustomMetric()

    # User TypeError inside fn propagates without double-invocation
    call_count = 0

    def failing_fn(pred, target=None):
        nonlocal call_count
        call_count += 1
        raise TypeError("Custom user error inside metric function")

    cm_fail = CustomMetric(failing_fn)
    with pytest.raises(TypeError, match="Custom user error inside metric function"):
        cm_fail.compute("p", "t")
    assert call_count == 1

    # Context passing with various signatures
    seen = {}

    def sig1(pred, context=None):
        seen["sig1"] = context
        return 1.0

    CustomMetric(sig1).compute("p", input_data={"key": "val1"})
    assert seen["sig1"] == {"key": "val1"}

    def sig2(pred, target, input_data=None):
        seen["sig2"] = input_data
        return 1.0

    CustomMetric(sig2).compute("p", "t", input_data={"key": "val2"})
    assert seen["sig2"] == {"key": "val2"}

    def sig3(pred, target=None, **kwargs):
        seen["sig3"] = kwargs.get("input_data")
        return 1.0

    CustomMetric(sig3).compute("p", "t", input_data={"key": "val3"})
    assert seen["sig3"] == {"key": "val3"}

    # Keyword-only signature
    def sig_kw(pred, *, target=None, input_data=None):
        seen["sig_kw"] = (pred, target, input_data)
        return 0.88

    res_kw = CustomMetric(sig_kw).compute("p", "t", input_data={"k": "v"})
    assert res_kw == 0.88
    assert seen["sig_kw"] == ("p", "t", {"k": "v"})

    # Positional-only signature
    def sig_pos(pred, target, /):
        seen["sig_pos"] = (pred, target)
        return 0.77

    res_pos = CustomMetric(sig_pos).compute("p", "t", input_data={"k": "v"})
    assert res_pos == 0.77
    assert seen["sig_pos"] == ("p", "t")


def test_evaluator_suite():
    evaluator = Evaluator(
        [
            ExactMatch(),
            F1Score(),
            JSONValidation(),
            CustomMetric(
                lambda pred, target: len(str(pred)) / 100.0, name="length_ratio"
            ),
        ]
    )

    scores = evaluator.evaluate(
        prediction='{"name": "test"}',
        target='{"name": "test"}',
    )
    assert scores["exact_match"] == 1.0
    assert scores["json_validity"] == 1.0
    assert "length_ratio" in scores


def test_evaluator_concurrent_thread_isolated_errors():
    # R25: Concurrent evaluate calls across threads see only their own thread-local errors
    import concurrent.futures

    def thread_flaky_fn(pred, target=None, input_data=None):
        thread_id = input_data.get("thread_id") if input_data else "unknown"
        raise RuntimeError(f"Error from thread {thread_id}")

    m = CustomMetric(thread_flaky_fn, name="flaky_m", is_llm_judge=True)
    evaluator = Evaluator([m], on_error="nan")

    results = {}

    def worker(tid):
        scores = evaluator.evaluate("pred", "target", input_data={"thread_id": tid})
        err = evaluator.last_errors.get("flaky_m")
        results[tid] = (scores, err)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(16)]
        concurrent.futures.wait(futures)

    assert len(results) == 16
    for tid, (scores, err) in results.items():
        assert math.isnan(scores["flaky_m"])
        assert f"Error from thread {tid}" in err
