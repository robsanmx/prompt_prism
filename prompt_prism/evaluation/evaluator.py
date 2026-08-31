"""
Evaluator orchestrator for computing multiple metrics across experiment trials.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

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


class Evaluator:
    """
    Manages a suite of metrics and calculates evaluation scores for trials.
    
    Attributes:
        metrics: Registered list of metric instances.
        on_error: Error handling mode ("nan", "zero", "raise", or None for auto-detection).
        last_errors: Map of metric name to exception message from the most recent evaluation.
    """

    def __init__(
        self,
        metrics: Optional[Sequence[Union[Metric, Callable[..., float]]]] = None,
        on_error: Optional[str] = None,
    ):
        self.metrics: List[Metric] = []
        self.on_error = on_error
        self.last_errors: Dict[str, str] = {}

        if metrics:
            for m in metrics:
                self.add_metric(m)
        else:
            # Default to ExactMatch and F1
            self.add_metric(ExactMatch())
            self.add_metric(F1Score())

    def add_metric(self, metric: Union[Metric, Callable[..., float]], name: Optional[str] = None) -> Metric:
        """Add a metric to the suite. Raises ValueError on duplicate names."""
        if isinstance(metric, Metric):
            m_obj = metric
            if name:
                m_obj.name = name
        elif callable(metric):
            m_obj = CustomMetric(metric, name=name)
        else:
            raise TypeError(f"Expected Metric instance or callable, got {type(metric)}")

        if any(existing.name == m_obj.name for existing in self.metrics):
            raise ValueError(f"Duplicate metric name '{m_obj.name}' already registered in Evaluator.")

        self.metrics.append(m_obj)
        return m_obj

    def evaluate(
        self,
        prediction: Any,
        target: Any,
        input_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute all configured metrics for a single prediction.
        
        If a metric fails:
        - If on_error="raise", raises the exception.
        - If on_error="nan" (or metric.is_llm_judge=True and on_error is None), assigns float("nan").
        - If on_error="zero", assigns 0.0.
        """
        ctx = context if context is not None else input_data
        scores: Dict[str, float] = {}
        self.last_errors = {}

        for m in self.metrics:
            try:
                score = m.compute(prediction=prediction, target=target, input_data=ctx)
                scores[m.name] = float(score)
            except Exception as e:
                self.last_errors[m.name] = str(e)
                mode = self.on_error
                if mode is None:
                    mode = "nan" if getattr(m, "is_llm_judge", False) else "zero"

                if mode == "raise":
                    raise e
                elif mode == "nan":
                    scores[m.name] = float("nan")
                else:
                    scores[m.name] = 0.0

        return scores

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        pred_col: str = "prediction",
        target_col: str = "target",
    ) -> pd.DataFrame:
        """Compute all metrics across rows in a DataFrame."""
        results = []
        for _, row in df.iterrows():
            pred = row.get(pred_col)
            target = row.get(target_col)
            row_dict = row.to_dict()
            scores = self.evaluate(prediction=pred, target=target, input_data=row_dict)
            results.append(scores)

        scores_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
