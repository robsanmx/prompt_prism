"""
Unit tests for evaluation metrics.
"""

from prompt_prism.evaluation.evaluator import Evaluator
from prompt_prism.evaluation.metrics import (
    CustomMetric,
    ExactMatch,
    F1Score,
    JSONValidation,
    KeyValuesExtractionOverlap,
    LevenshteinSimilarity,
    RegexMatch,
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
