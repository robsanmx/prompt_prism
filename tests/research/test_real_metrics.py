"""
Unit tests for Real-LLM v2 Deterministic Evaluators.
Verifies micro-averaged field F1, unigram token F1, abstention logic,
and intent classification accuracy.
"""

import pytest

from research.analysis.real_v2_metrics import (
    evaluate_grounded_qa,
    evaluate_intent_classification,
    evaluate_schema_extraction,
)


def test_schema_extraction_evaluator():
    target = {"buyer": "Apex Inc", "target": "Beta Corp", "amount": "$50M", "date": "May 1"}

    # Perfect match
    resp_perfect = '{"buyer": "Apex Inc", "target": "Beta Corp", "amount": "$50M", "date": "May 1"}'
    scores = evaluate_schema_extraction(resp_perfect, target)
    assert scores["field_f1"] == 1.0
    assert scores["exact_record_match"] == 1.0
    assert scores["schema_validity"] == 1.0

    # Partial match
    resp_partial = '{"buyer": "Apex Inc", "target": "Wrong Corp", "amount": "$50M", "date": null}'
    scores_part = evaluate_schema_extraction(resp_partial, target)
    assert 0.0 < scores_part["field_f1"] < 1.0
    assert scores_part["exact_record_match"] == 0.0
    assert scores_part["schema_validity"] == 1.0

    # Invalid JSON
    resp_invalid = "This is not JSON at all."
    scores_inv = evaluate_schema_extraction(resp_invalid, target)
    assert scores_inv["field_f1"] == 0.0
    assert scores_inv["exact_record_match"] == 0.0
    assert scores_inv["schema_validity"] == 0.0


def test_grounded_qa_evaluator():
    # Answerable - perfect
    scores_ans = evaluate_grounded_qa("Ariane 5", "Ariane 5", is_answerable=True)
    assert scores_ans["token_f1"] == 1.0
    assert scores_ans["exact_match"] == 1.0
    assert scores_ans["abstention_accuracy"] == 1.0

    # Answerable - partial
    scores_part = evaluate_grounded_qa("the Ariane rocket", "Ariane 5", is_answerable=True)
    assert 0.0 < scores_part["token_f1"] < 1.0

    # Unanswerable - correct abstention
    scores_unans = evaluate_grounded_qa("This question is unanswerable based on the passage.", "unanswerable", is_answerable=False)
    assert scores_unans["token_f1"] == 1.0
    assert scores_unans["abstention_accuracy"] == 1.0

    # Unanswerable - false hallucinated answer
    scores_halluc = evaluate_grounded_qa("The flight director was John Doe.", "unanswerable", is_answerable=False)
    assert scores_halluc["token_f1"] == 0.0
    assert scores_halluc["abstention_accuracy"] == 0.0


def test_intent_classification_evaluator():
    # Valid and correct
    scores_corr = evaluate_intent_classification("lost_card", "lost_card")
    assert scores_corr["accuracy"] == 1.0
    assert scores_corr["is_valid_label"] == 1.0

    # Embedded in conversational sentence
    scores_embed = evaluate_intent_classification("The user wants: lost_card.", "lost_card")
    assert scores_embed["accuracy"] == 1.0
    assert scores_embed["is_valid_label"] == 1.0

    # Incorrect valid class
    scores_inc = evaluate_intent_classification("balance_inquiry", "lost_card")
    assert scores_inc["accuracy"] == 0.0
    assert scores_inc["is_valid_label"] == 1.0

    # Invalid / hallucinated class
    scores_inv = evaluate_intent_classification("schedule_vacation", "lost_card")
    assert scores_inv["accuracy"] == 0.0
    assert scores_inv["is_valid_label"] == 0.0
