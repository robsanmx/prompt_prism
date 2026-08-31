"""
Evaluation metrics and LLM judge modules.
"""

from .evaluator import Evaluator
from .metrics import (
    CustomMetric,
    ExactMatch,
    F1Score,
    JSONValidation,
    KeyValuesExtractionOverlap,
    LevenshteinSimilarity,
    Metric,
    RegexMatch,
)

__all__ = [
    "Metric",
    "ExactMatch",
    "F1Score",
    "JSONValidation",
    "KeyValuesExtractionOverlap",
    "LevenshteinSimilarity",
    "RegexMatch",
    "CustomMetric",
    "Evaluator",
    "DeepEvalMetric",
    "deepeval_metric",
    "JudgeCache",
]


def __getattr__(name: str):
    if name in {"DeepEvalMetric", "deepeval_metric"}:
        from .deepeval_metrics import DeepEvalMetric, deepeval_metric
        if name == "DeepEvalMetric":
            return DeepEvalMetric
        return deepeval_metric
    if name == "JudgeCache":
        from .judge_cache import JudgeCache
        return JudgeCache
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
