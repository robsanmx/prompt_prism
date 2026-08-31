"""
Evaluator orchestrator for computing multiple metrics across experiment trials.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
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
    """

    def __init__(self, metrics: Optional[Sequence[Union[Metric, Callable[..., float]]]] = None):
        self.metrics: List[Metric] = []
        if metrics:
            for m in metrics:
                self.add_metric(m)
        else:
            # Default to ExactMatch and F1
            self.add_metric(ExactMatch())
            self.add_metric(F1Score())

    def add_metric(self, metric: Union[Metric, Callable[..., float]], name: Optional[str] = None) -> Metric:
        """Add a metric to the suite."""
        if isinstance(metric, Metric):
            m_obj = metric
        elif callable(metric):
            m_obj = CustomMetric(metric, name=name)
        else:
            raise TypeError(f"Expected Metric instance or callable, got {type(metric)}")
        self.metrics.append(m_obj)
        return m_obj

    def evaluate(
        self,
        prediction: Any,
        target: Any,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute all configured metrics for a single prediction."""
        scores = {}
        for m in self.metrics:
            try:
                score = m.compute(prediction=prediction, target=target, input_data=input_data)
                scores[m.name] = float(score)
            except Exception as e:
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
