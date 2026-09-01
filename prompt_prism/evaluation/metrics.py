"""
Comprehensive Suite of Evaluation Metrics for Prompt Optimization Experiments.
"""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence


class Metric:
    """
    Abstract Base Metric.

    Attributes:
        name: Unique metric identifier used as column name in ANOVA tables.
        higher_is_better: Whether higher score represents better performance.
        is_llm_judge: Whether this metric is network-bound/LLM-based (default False).
        wants_prompt: Whether this metric needs access to the composed prompt (__prompt__ in context).
    """

    name: str = "metric"
    higher_is_better: bool = True
    is_llm_judge: bool = False
    wants_prompt: bool = False

    def compute(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute the metric score for a single prediction and target."""
        raise NotImplementedError

    def __call__(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        return self.compute(prediction, target, input_data)

    def pop_reason(self) -> Optional[str]:
        """Return and clear an explanation for this thread's most recent `compute`.

        Metrics that produce a human-readable justification - LLM judges - override this;
        `Evaluator` collects the result into `last_reasons`. Deterministic metrics have no
        explanation to give, so the default is None. This is the explicit alternative to
        writing the reason back into the caller's `input_data` dict.
        """
        return None


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

    def compute(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        norm_pred = self._normalize(prediction)
        norm_target = self._normalize(target)
        return 1.0 if norm_pred == norm_target else 0.0


class F1Score(Metric):
    """Token-level precision, recall, and F1 score for text overlap."""

    def __init__(self, name: str = "f1_score", mode: str = "f1"):
        self.name = name
        self.mode = mode.lower()  # "f1", "precision", or "recall"
        if self.mode not in ("f1", "precision", "recall"):
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of: 'f1', 'precision', 'recall'"
            )

    def _tokenize(self, text: Any) -> List[str]:
        s = str(text or "").lower().strip()
        s = re.sub(r"[^\w\s]", " ", s)
        return s.split()

    def compute(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        pred_tokens = self._tokenize(prediction)
        target_tokens = self._tokenize(target)

        if not pred_tokens and not target_tokens:
            return 1.0
        if not pred_tokens or not target_tokens:
            return 0.0

        common = set(pred_tokens) & set(target_tokens)
        if not common:
            return 0.0

        # Token frequency intersection
        pred_counts = Counter(pred_tokens)
        target_counts = Counter(target_tokens)
        overlap = sum(min(pred_counts[token], target_counts[token]) for token in common)

        precision = overlap / len(pred_tokens)
        recall = overlap / len(target_tokens)

        if self.mode == "precision":
            return precision
        if self.mode == "recall":
            return recall

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class JSONValidation(Metric):
    """Validates if the LLM output is valid JSON and optionally conforms to schema/required keys."""

    def __init__(
        self,
        name: str = "json_validity",
        required_keys: Optional[Sequence[str]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extract_from_code_blocks: bool = True,
    ):
        self.name = name
        self.required_keys = list(required_keys) if required_keys else []
        self.schema = schema
        self.extract_from_code_blocks = extract_from_code_blocks

    def _extract_json_str(self, text: str) -> str:
        s = str(text or "").strip()
        if self.extract_from_code_blocks:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
            if match:
                return match.group(1).strip()
        return s

    def compute(
        self,
        prediction: Any,
        target: Any = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        if isinstance(prediction, (dict, list)):
            parsed = prediction
        else:
            json_str = self._extract_json_str(str(prediction))
            try:
                parsed = json.loads(json_str)
            except Exception:
                # Fallback: look for embedded JSON in surrounding prose
                match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", json_str)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        return 0.0
                else:
                    return 0.0

        if self.required_keys:
            if not isinstance(parsed, dict):
                return 0.0
            present = sum(1 for k in self.required_keys if k in parsed)
            return present / len(self.required_keys)

        return 1.0


class KeyValuesExtractionOverlap(Metric):
    """
    Measures precision, recall, or F1 / Jaccard similarity across extracted key-value dictionaries.
    """

    def __init__(
        self,
        name: str = "attribute_overlap",
        mode: str = "f1",
        case_sensitive: bool = False,
    ):
        self.name = name
        self.mode = mode.lower()
        if self.mode not in ("f1", "precision", "recall", "jaccard"):
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of: 'f1', 'precision', 'recall', 'jaccard'"
            )
        self.case_sensitive = case_sensitive

    def _to_kv_dict(self, val: Any) -> Dict[str, str]:
        if isinstance(val, dict):
            d = val
        elif isinstance(val, str):
            s = val.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = json.loads(s)
                except Exception:
                    d = {}
            else:
                d = {}
                for line in s.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        d[k.strip()] = v.strip()
        else:
            d = {}

        if not self.case_sensitive:
            return {
                str(k).lower().strip(): str(v).lower().strip() for k, v in d.items()
            }
        return {str(k).strip(): str(v).strip() for k, v in d.items()}

    def compute(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        pred_dict = self._to_kv_dict(prediction)
        target_dict = self._to_kv_dict(target)

        if not pred_dict and not target_dict:
            return 1.0
        if not pred_dict or not target_dict:
            return 0.0

        # Matched pairs
        matched_keys = set(pred_dict.keys()) & set(target_dict.keys())
        exact_matches = sum(1 for k in matched_keys if pred_dict[k] == target_dict[k])

        all_keys = set(pred_dict.keys()) | set(target_dict.keys())

        precision = exact_matches / len(pred_dict) if pred_dict else 0.0
        recall = exact_matches / len(target_dict) if target_dict else 0.0

        if self.mode == "precision":
            return precision
        elif self.mode == "recall":
            return recall
        elif self.mode == "jaccard":
            return exact_matches / len(all_keys) if all_keys else 0.0
        else:  # f1
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)


class LevenshteinSimilarity(Metric):
    """Normalized character-level Levenshtein similarity between prediction and target [0, 1]."""

    def __init__(self, name: str = "levenshtein_sim"):
        self.name = name

    def compute(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]] = None
    ) -> float:
        s1 = str(prediction or "")
        s2 = str(target or "")
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        previous_row = list(range(len2 + 1))
        current_row = [0] * (len2 + 1)

        for i in range(1, len1 + 1):
            current_row[0] = i
            c1 = s1[i - 1]
            for j in range(1, len2 + 1):
                cost = 0 if c1 == s2[j - 1] else 1
                current_row[j] = min(
                    previous_row[j] + 1,  # deletion
                    current_row[j - 1] + 1,  # insertion
                    previous_row[j - 1] + cost,  # substitution
                )
            previous_row, current_row = current_row, previous_row

        distance = previous_row[len2]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0


class RegexMatch(Metric):
    """Scores 1.0 if regular expression pattern matches the prediction, 0.0 otherwise."""

    def __init__(self, pattern: str, name: str = "regex_match", flags: int = 0):
        self.name = name
        self.pattern = re.compile(pattern, flags)

    def compute(
        self,
        prediction: Any,
        target: Any = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        s = str(prediction or "")
        return 1.0 if self.pattern.search(s) else 0.0


class CustomMetric(Metric):
    """Wraps any user-supplied callable `fn(prediction, target, input_data) -> float`."""

    def __init__(
        self,
        fn: Optional[Callable[..., float]] = None,
        name: Optional[str] = None,
        higher_is_better: bool = True,
        is_llm_judge: bool = False,
        wants_prompt: bool = False,
        score_fn: Optional[Callable[..., float]] = None,
    ):
        if fn is None and score_fn is not None:
            import warnings

            warnings.warn(
                "`score_fn` is deprecated; use `fn` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            fn = score_fn
        elif fn is None:
            raise TypeError("CustomMetric missing required argument: 'fn'")

        self.fn = fn
        self.name = name or getattr(fn, "__name__", "custom_metric")
        self.higher_is_better = higher_is_better
        self.is_llm_judge = is_llm_judge
        self.wants_prompt = wants_prompt
        self._caller = self._compile_caller(fn)

    @staticmethod
    def _compile_caller(
        fn: Callable[..., Any],
    ) -> Callable[[Any, Any, Optional[Dict[str, Any]]], float]:
        # Each shape is declared exactly once, as the (args, kwargs) it would be called
        # with. The bind probe and the real invocation therefore use the same expression -
        # writing them as two parallel lambdas let them drift into a wrong-argument bug.
        candidate_shapes = [
            lambda p, t, d: ((p, t, d), {}),
            lambda p, t, d: ((p, t), {"input_data": d}),
            lambda p, t, d: ((p,), {"target": t, "input_data": d}),
            lambda p, t, d: ((p, t), {"context": d}),
            lambda p, t, d: ((p,), {"target": t, "context": d}),
            lambda p, t, d: ((p,), {"input_data": d}),
            lambda p, t, d: ((p,), {"context": d}),
            lambda p, t, d: ((p, t), {}),
            lambda p, t, d: ((p,), {"target": t}),
            lambda p, t, d: ((p,), {}),
            lambda p, t, d: ((), {"prediction": p, "target": t, "input_data": d}),
        ]

        try:
            sig = inspect.signature(fn)
            for shape in candidate_shapes:
                probe_args, probe_kwargs = shape("pred", "target", {})
                try:
                    sig.bind(*probe_args, **probe_kwargs)
                except TypeError:
                    continue

                def caller(pred, target, input_data, _shape=shape):
                    call_args, call_kwargs = _shape(pred, target, input_data)
                    return float(fn(*call_args, **call_kwargs))

                return caller
        except (TypeError, ValueError):
            # inspect.signature fails on some builtins and C callables; fall through to
            # the canonical three-argument call and let any real mismatch surface there.
            pass

        return lambda pred, target, input_data: float(
            fn(pred, target, input_data=input_data)
        )

    def compute(
        self,
        prediction: Any,
        target: Any = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self._caller(prediction, target, input_data)
