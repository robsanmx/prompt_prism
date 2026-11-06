"""Deterministic per-item metrics for the real-v3 datasets."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Dict, Iterable, List, Optional


ENTITY_KEYS = ["person", "organization", "location", "money", "temporal"]


def normalize_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, target: str) -> float:
    pred = normalize_text(prediction).split()
    gold = normalize_text(target).split()
    if not pred or not gold:
        return 1.0 if pred == gold else 0.0
    overlap = Counter(pred) & Counter(gold)
    common = sum(overlap.values())
    if not common:
        return 0.0
    precision = common / len(pred)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_object(text: str) -> Optional[dict]:
    try:
        value = json.loads(_strip_fence(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _entity_set(record: Dict[str, Iterable[object]]) -> set[tuple[str, str]]:
    return {
        (key, normalize_text(value))
        for key in ENTITY_KEYS
        for value in record.get(key, [])
        if normalize_text(value)
    }


def evaluate_entity_extraction(response: str, target: Dict[str, List[str]]) -> Dict[str, float]:
    parsed = _parse_json_object(response)
    valid = bool(
        parsed is not None
        and set(parsed) == set(ENTITY_KEYS)
        and all(isinstance(parsed[key], list) for key in ENTITY_KEYS)
    )
    if not valid:
        return {"entity_f1": 0.0, "exact_match": 0.0, "schema_validity": 0.0}

    predicted = _entity_set(parsed)
    expected = _entity_set(target)
    if not predicted and not expected:
        f1 = 1.0
    elif not predicted or not expected:
        f1 = 0.0
    else:
        overlap = len(predicted & expected)
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        f1 = 0.0 if not overlap else 2 * precision * recall / (precision + recall)
    return {
        "entity_f1": f1,
        "exact_match": float(predicted == expected),
        "schema_validity": 1.0,
    }


def _answer_from_response(response: str) -> Optional[str]:
    parsed = _parse_json_object(response)
    if parsed is not None and set(parsed) == {"answer"}:
        answer = parsed["answer"]
        return None if answer is None else str(answer).strip()
    cleaned = _strip_fence(response)
    return cleaned or None


def evaluate_grounded_qa(
    response: str,
    accepted_answers: List[str],
    is_answerable: bool,
) -> Dict[str, float]:
    answer = _answer_from_response(response)
    norm = normalize_text(answer or "")
    abstained = answer is None or norm in {
        "unanswerable",
        "cannot be determined",
        "not answerable",
        "not mentioned",
        "unknown",
        "null",
    }
    if not is_answerable:
        score = float(abstained)
        return {"token_f1": score, "exact_match": score, "abstention_accuracy": score}
    if abstained or not accepted_answers:
        return {"token_f1": 0.0, "exact_match": 0.0, "abstention_accuracy": 0.0}

    f1 = max(token_f1(answer or "", target) for target in accepted_answers)
    exact = float(any(norm == normalize_text(target) for target in accepted_answers))
    return {"token_f1": f1, "exact_match": exact, "abstention_accuracy": 1.0}


def evaluate_intent(response: str, target: str, allowed_labels: List[str]) -> Dict[str, float]:
    parsed = _parse_json_object(response)
    if parsed is not None and set(parsed) == {"intent"}:
        prediction = parsed["intent"]
    else:
        prediction = _strip_fence(response).strip('"\' ')
    valid = isinstance(prediction, str) and prediction in allowed_labels
    return {
        "accuracy": float(valid and prediction == target),
        "valid_label": float(valid),
    }
