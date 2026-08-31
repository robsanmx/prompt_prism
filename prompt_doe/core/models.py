"""
Core data structures for representing Design Matrices, Run Configurations, Trials, and Experiment Results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """
    Configuration for a single experimental run (a specific combination of factor levels).
    
    Attributes:
        run_id: Sequential or unique identifier of the run.
        factor_levels: Mapping of factor identifier (or name) to the assigned level code.
        factor_names: Optional mapping of factor name to level code.
        params: Model parameters (e.g. temperature, max_tokens) configured for this run.
        metadata: Extra metadata for tracking (e.g. plan_id, repetition).
    """
    run_id: int
    factor_levels: Dict[str, int] = Field(default_factory=dict)
    factor_names: Dict[str, int] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_level(self, factor_id_or_name: str) -> int:
        """Get level code for a factor by ID or name."""
        if factor_id_or_name in self.factor_levels:
            return self.factor_levels[factor_id_or_name]
        if factor_id_or_name in self.factor_names:
            return self.factor_names[factor_id_or_name]
        raise KeyError(f"Factor '{factor_id_or_name}' not found in RunConfig")

    @property
    def level_string(self) -> str:
        """Compact string representation of levels (e.g. '01001')."""
        return "".join(str(self.factor_levels[k]) for k in sorted(self.factor_levels.keys()))


class DesignMatrix(BaseModel):
    """
    A full Design of Experiments (DoE) matrix.
    
    Attributes:
        plan_id: Name or code of the design (e.g. '2(5-1)V', 'PB-12', 'Full-2^4').
        factor_ids: Ordered list of factor IDs (e.g. ['A', 'B', 'C', 'D', 'E']).
        runs: List of RunConfig objects representing each experimental condition.
        resolution: Design resolution (e.g., 3 for Res III, 4 for Res IV, 5 for Res V, None for PB/custom).
        generators: Generator formulas (e.g. ['E=ABCD']).
        metadata: Additional details (number of factors, fraction, design properties).
    """
    plan_id: str
    factor_ids: List[str]
    runs: List[RunConfig] = Field(default_factory=list)
    resolution: Optional[int] = None
    generators: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def num_factors(self) -> int:
        return len(self.factor_ids)

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    def to_dataframe(self, coded: bool = True) -> pd.DataFrame:
        """
        Convert the design matrix to a pandas DataFrame.
        
        Args:
            coded: If True, values are in {0, 1} (or {-1, +1} if specified).
        """
        records = []
        for run in self.runs:
            row: Dict[str, Any] = {"run_id": run.run_id}
            for fid in self.factor_ids:
                val = run.factor_levels.get(fid, 0)
                row[fid] = val if coded else val
            row["combination"] = run.level_string
            records.append(row)
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        factor_ids: Optional[List[str]] = None,
        plan_id: str = "custom",
        run_id_col: str = "run_id"
    ) -> DesignMatrix:
        """Create a DesignMatrix from a pandas DataFrame."""
        if factor_ids is None:
            # Exclude metadata columns
            exclude = {run_id_col, "combination", "order", "run", "EXPERIMENT_ORDER", "PLAN_ID", "FACTORS_COMBINATION"}
            factor_ids = [c for c in df.columns if c not in exclude]

        runs = []
        for idx, row in df.iterrows():
            run_id = int(row[run_id_col]) if run_id_col in row else int(idx) + 1
            f_levels = {fid: int(row[fid]) for fid in factor_ids}
            runs.append(RunConfig(run_id=run_id, factor_levels=f_levels))

        return cls(plan_id=plan_id, factor_ids=factor_ids, runs=runs)


class Trial(BaseModel):
    """
    The result of executing one RunConfig on a single sample dataset item.
    """
    trial_id: str
    run_id: int
    sample_id: Union[str, int]
    factor_levels: Dict[str, int] = Field(default_factory=dict)
    prompt: Union[str, List[Dict[str, str]]] = ""
    raw_response: Any = None
    parsed_response: Any = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResults(BaseModel):
    """
    Aggregated results of an entire prompt optimization experiment.
    """
    experiment_id: str
    design: DesignMatrix
    trials: List[Trial] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten all trial executions and metrics into a pandas DataFrame ready for ANOVA.
        """
        rows = []
        for trial in self.trials:
            row: Dict[str, Any] = {
                "trial_id": trial.trial_id,
                "run_id": trial.run_id,
                "sample_id": trial.sample_id,
            }
            # Factor levels
            for fid, val in trial.factor_levels.items():
                row[fid] = val
            # Metrics
            for m_name, m_val in trial.metrics.items():
                row[m_name] = m_val
            if trial.latency_ms is not None:
                row["latency_ms"] = trial.latency_ms
            if trial.token_usage:
                for k, v in trial.token_usage.items():
                    row[f"tokens_{k}"] = v
            if trial.error:
                row["error"] = trial.error
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_by_run(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Compute mean and std of metrics grouped by experimental run."""
        df = self.to_dataframe()
        factor_cols = self.design.factor_ids
        metric_cols = [c for c in df.columns if c not in factor_cols and c not in {"trial_id", "run_id", "sample_id", "error"}]
        
        agg_funcs = ["mean", "std", "count"]
        grouped = df.groupby(["run_id"] + factor_cols)[metric_cols].agg(agg_funcs)
        return grouped
