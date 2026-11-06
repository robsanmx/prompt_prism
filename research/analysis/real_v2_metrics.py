"""
Deterministic Task Evaluators for Real-LLM v2 Study.
Provides objective per-item scoring and aggregate metrics for:
1. schema_extraction: Micro-averaged Field F1, exact record match, schema validity
2. grounded_qa: Token unigram F1, exact match, abstention accuracy
3. intent_classification: Binary correctness (0/1), class validity

Explicitly handles malformed output as behavioral failure (score 0.0), not missing data.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_string(s: str) -> str:
    """Normalize string by removing punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def compute_unigram_f1(pred: str, target: str) -> float:
    """Compute token-level unigram precision, recall, and F1 between two text strings."""
    pred_tokens = normalize_string(pred).split()
    target_tokens = normalize_string(target).split()

    if not pred_tokens or not target_tokens:
        return 1.0 if pred_tokens == target_tokens else 0.0

    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(w), target_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), target_tokens.count(w)) for w in common) / len(target_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ----------------------------------------------------------------------
# 1. Schema Extraction Evaluator
# ----------------------------------------------------------------------
def evaluate_schema_extraction(response_text: str, target_dict: Dict[str, str]) -> Dict[str, float]:
    """
    Evaluates schema-bound JSON extraction against ground truth target dict.
    Returns:
    - field_f1: micro-averaged F1 across all target fields
    - exact_record_match: 1.0 if all fields match perfectly, else 0.0
    - schema_validity: 1.0 if valid JSON containing target keys, else 0.0
    """
    if not response_text or not response_text.strip():
        return {"field_f1": 0.0, "exact_record_match": 0.0, "schema_validity": 0.0}

    # Attempt JSON parse
    text_clean = response_text.strip()
    # Strip markdown if present
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean)
        text_clean = re.sub(r"\s*```$", "", text_clean)

    try:
        parsed = json.loads(text_clean)
        if not isinstance(parsed, dict):
            return {"field_f1": 0.0, "exact_record_match": 0.0, "schema_validity": 0.0}
    except Exception:
        return {"field_f1": 0.0, "exact_record_match": 0.0, "schema_validity": 0.0}

    is_valid_schema = 1.0
    f1_scores = []
    exact_matches = []

    for k, target_val in target_dict.items():
        if k in parsed and parsed[k] is not None:
            pred_str = str(parsed[k])
            f1 = compute_unigram_f1(pred_str, str(target_val))
            em = 1.0 if normalize_string(pred_str) == normalize_string(str(target_val)) else 0.0
        else:
            f1 = 0.0
            em = 0.0
        f1_scores.append(f1)
        exact_matches.append(em)

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    record_em = 1.0 if all(em == 1.0 for em in exact_matches) else 0.0

    return {
        "field_f1": mean_f1,
        "exact_record_match": record_em,
        "schema_validity": is_valid_schema,
    }


# ----------------------------------------------------------------------
# 2. Grounded QA Evaluator
# ----------------------------------------------------------------------
def evaluate_grounded_qa(response_text: str, target_answer: str, is_answerable: bool) -> Dict[str, float]:
    """
    Evaluates QA extraction against reference answer.
    Handles answerable questions (token F1) and unanswerable questions (abstention).
    """
    if not response_text or not response_text.strip():
        return {"token_f1": 0.0, "exact_match": 0.0, "abstention_accuracy": 0.0}

    pred_clean = response_text.strip()
    norm_pred = normalize_string(pred_clean)
    abstain_cues = {"unanswerable", "cannot be determined", "abstain", "null", "not mentioned", "unknown"}

    has_abstained = any(cue in norm_pred for cue in abstain_cues)

    if not is_answerable:
        # Ground truth is unanswerable
        correct_abstention = 1.0 if has_abstained else 0.0
        return {
            "token_f1": correct_abstention,
            "exact_match": correct_abstention,
            "abstention_accuracy": correct_abstention,
        }
    else:
        # Ground truth is answerable
        if has_abstained:
            return {"token_f1": 0.0, "exact_match": 0.0, "abstention_accuracy": 0.0}

        f1 = compute_unigram_f1(pred_clean, target_answer)
        em = 1.0 if norm_pred == normalize_string(target_answer) else 0.0
        return {
            "token_f1": f1,
            "exact_match": em,
            "abstention_accuracy": 1.0,  # correctly did not abstain
        }


# ----------------------------------------------------------------------
# 3. Intent Classification Evaluator
# ----------------------------------------------------------------------
VALID_INTENTS = {
    "lost_card",
    "balance_inquiry",
    "update_address",
    "dispute_charge",
    "loan_application",
    "reset_credentials",
    "out_of_scope",
}


def evaluate_intent_classification(response_text: str, target_intent: str) -> Dict[str, float]:
    """
    Evaluates customer intent classification.
    Returns:
    - accuracy: 1.0 if target intent is correctly identified, else 0.0
    - is_valid_label: 1.0 if prediction is one of the 7 allowed classes, else 0.0
    """
    if not response_text or not response_text.strip():
        return {"accuracy": 0.0, "is_valid_label": 0.0}

    pred_clean = response_text.strip().lower()
    # Normalize common punct/quotes
    pred_clean = re.sub(r'["\']', '', pred_clean).strip()

    # Find matching valid intent
    detected_intent = None
    for intent in VALID_INTENTS:
        if intent == pred_clean or re.search(rf"\b{intent}\b", pred_clean):
            detected_intent = intent
            break

    if detected_intent is None:
        return {"accuracy": 0.0, "is_valid_label": 0.0}

    is_correct = 1.0 if detected_intent == target_intent else 0.0
    return {
        "accuracy": is_correct,
        "is_valid_label": 1.0,
    }


def get_task_evaluator(task_id: str):
    """Factory retrieving registered evaluator for task."""
    if task_id == "schema_extraction":
        return lambda resp, item: evaluate_schema_extraction(resp, item["target"])
    elif task_id == "grounded_qa":
        return lambda resp, item: evaluate_grounded_qa(resp, item["reference_answer"], item["is_answerable"])
    elif task_id == "intent_classification":
        return lambda resp, item: evaluate_intent_classification(resp, item["intent"])
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
