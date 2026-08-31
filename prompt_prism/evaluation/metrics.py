"""
Comprehensive Suite of Evaluation Metrics for Prompt Optimization Experiments.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union
import numpy as np


class Metric:
    """Abstract Base Metric."""
    name: str = "metric"
    higher_is_better: bool = True

    def compute(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        """Compute the metric score for a single prediction and target."""
        raise NotImplementedError

    def __call__(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        return self.compute(prediction, target, input_data)


class ExactMatch(Metric):
    """Exact string or value equality metric."""

    def __init__(
        self,
        name: str = "exact_match",
        case_sensitive: bool = False,
        strip: bool = True,
        ignore_punctuation: bool = False,
    ):
        self.name = name
        self.case_sensitive = case_sensitive
        self.strip = strip
        self.ignore_punctuation = ignore_punctuation

    def _normalize(self, val: Any) -> str:
        s = str(val or "")
        if self.strip:
            s = s.strip()
        if not self.case_sensitive:
            s = s.lower()
        if self.ignore_punctuation:
            s = re.sub(r"[^\w\s]", "", s)
        return re.sub(r"\s+", " ", s)

    def compute(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        norm_pred = self._normalize(prediction)
        norm_target = self._normalize(target)
        return 1.0 if norm_pred == norm_target else 0.0


class F1Score(Metric):
    """Token-level precision, recall, and F1 score for text overlap."""

    def __init__(self, name: str = "f1_score", mode: str = "f1"):
        self.name = name
        self.mode = mode  # 'f1', 'precision', 'recall'

    def _tokenize(self, text: Any) -> List[str]:
        return re.findall(r"\w+", str(text or "").lower())

    def compute(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        pred_tokens = self._tokenize(prediction)
        target_tokens = self._tokenize(target)

        if not pred_tokens and not target_tokens:
            return 1.0
        if not pred_tokens or not target_tokens:
            return 0.0

        common = set(pred_tokens) & set(target_tokens)
        num_same = sum(min(pred_tokens.count(t), target_tokens.count(t)) for t in common)
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(target_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if self.mode == "precision":
            return float(precision)
        elif self.mode == "recall":
            return float(recall)
        return float(f1)


class JSONValidation(Metric):
    """
    Checks if output is valid JSON and optionally conforms to required keys.
    """

    def __init__(
        self,
        name: str = "json_validity",
        required_keys: Optional[Sequence[str]] = None,
    ):
        self.name = name
        self.required_keys = set(required_keys) if required_keys else set()

    def compute(self, prediction: Any, target: Any = None, input_data: Optional[Dict[str, Any]] = None) -> float:
        if isinstance(prediction, dict):
            parsed = prediction
        else:
            # Strip markdown fences if present
            s = str(prediction or "").strip()
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\n|\n```$", "", s, flags=re.MULTILINE).strip()
            try:
                parsed = json.loads(s)
            except Exception:
                # Try finding JSON substring
                m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                    except Exception:
                        return 0.0
                else:
                    return 0.0

        if not isinstance(parsed, dict):
            return 0.5  # Valid JSON, but not an object

        if self.required_keys:
            found_keys = set(parsed.keys())
            matched = len(self.required_keys & found_keys)
            return float(matched / len(self.required_keys))

        return 1.0


class KeyValuesExtractionOverlap(Metric):
    """
    Evaluates key-value extraction precision, recall, and value match rate against target attributes.
    """

    def __init__(
        self,
        name: str = "extraction_f1",
        mode: str = "f1",  # 'f1', 'precision', 'recall', 'shared_keys', 'value_accuracy'
        case_sensitive: bool = False,
    ):
        self.name = name
        self.mode = mode
        self.case_sensitive = case_sensitive

    def _parse_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if not obj:
            return {}
        s = str(obj).strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\n|\n```$", "", s, flags=re.MULTILINE).strip()
        try:
            d = json.loads(s)
            return d if isinstance(d, dict) else {}
        except Exception:
            # Fallback regex key-value extraction
            matches = re.findall(r'["\']?([\w_]+)["\']?\s*[:=]\s*["\']?([^,\n\}]+)["\']?', s)
            return {k.strip(): v.strip() for k, v in matches}

    def _norm(self, s: Any) -> str:
        res = str(s or "").strip()
        return res if self.case_sensitive else res.lower()

    def compute(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        pred_dict = self._parse_dict(prediction)
        target_dict = self._parse_dict(target)

        pred_keys = {self._norm(k): v for k, v in pred_dict.items()}
        target_keys = {self._norm(k): v for k, v in target_dict.items()}

        if not pred_keys and not target_keys:
            return 1.0
        if not pred_keys or not target_keys:
            return 0.0

        shared_keys = set(pred_keys.keys()) & set(target_keys.keys())
        p = len(shared_keys) / len(pred_keys) if pred_keys else 0.0
        r = len(shared_keys) / len(target_keys) if target_keys else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        if self.mode == "shared_keys":
            return float(len(shared_keys))
        elif self.mode == "precision":
            return float(p)
        elif self.mode == "recall":
            return float(r)
        elif self.mode == "value_accuracy":
            if not shared_keys:
                return 0.0
            val_matches = sum(
                1 for k in shared_keys
                if self._norm(pred_keys[k]) == self._norm(target_keys[k])
                or self._norm(pred_keys[k]) in self._norm(target_keys[k])
            )
            return float(val_matches / len(shared_keys))
        return float(f1)


class LevenshteinSimilarity(Metric):
    """Normalized Levenshtein edit similarity in [0, 1]."""

    def __init__(self, name: str = "similarity"):
        self.name = name

    def compute(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        s1 = str(prediction or "").strip()
        s2 = str(target or "").strip()
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Dynamic programming edit distance
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i, c1 in enumerate(s1):
            new_dp = [i + 1] + [0] * n
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
            dp = new_dp

        dist = dp[n]
        max_len = max(m, n)
        return float(1.0 - (dist / max_len))


class RegexMatch(Metric):
    """Checks if regex pattern matches the prediction."""

    def __init__(self, pattern: str, name: str = "regex_match", flags: int = 0):
        self.name = name
        self.pattern = re.compile(pattern, flags)

    def compute(self, prediction: Any, target: Any = None, input_data: Optional[Dict[str, Any]] = None) -> float:
        s = str(prediction or "")
        return 1.0 if bool(self.pattern.search(s)) else 0.0


class CustomMetric(Metric):
    """Wraps any custom evaluation function: fn(prediction, target, input_data) -> float."""

    def __init__(
        self,
        score_fn: Callable[..., float],
        name: Optional[str] = None,
        higher_is_better: bool = True,
    ):
        self.score_fn = score_fn
        self.name = name or getattr(score_fn, "__name__", "custom_metric")
        self.higher_is_better = higher_is_better

    def compute(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None) -> float:
        try:
            return float(self.score_fn(prediction, target, input_data=input_data))
        except TypeError:
            return float(self.score_fn(prediction, target))
