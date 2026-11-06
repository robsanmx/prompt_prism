"""
Formal Estimands and Statistical Evaluation Metrics for PromptPrism Validation.
Computes effect recovery (RMSE, sign agreement), calibration (interval coverage),
multiplicity control (Type I error, FDR, power), equivalence testing (TOST),
and held-out selection regret.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MethodResult:
    """
    Standardized container for the output of any prompt optimization / screening method.
    """

    method_id: str
    budget_configs: int
    budget_calls: int
    selected_config: Dict[str, int]
    estimated_main_effects: Dict[str, float]
    p_values: Dict[str, float]
    conf_intervals_95: Dict[str, Tuple[float, float]]
    conf_intervals_90: Dict[str, Tuple[float, float]]


@dataclass
class EvaluationMetrics:
    """
    Aggregated statistical evaluation metrics comparing a method against ground truth / oracle.
    """

    method_id: str
    rmse: float
    mae: float
    sign_accuracy_above_delta: float
    coverage_95: float
    type_1_error: float
    power_above_delta: float
    fdr: float
    held_out_regret: float
    uplift_captured_ratio: float
    optimal_config_selected: bool


class EstimandCalculator:
    """
    Computes preregistered estimands and verification statistics.
    """

    @staticmethod
    def calculate_regret(
        selected_config: Dict[str, int],
        oracle_best_config: Dict[str, int],
        confirmation_df: pd.DataFrame,
        factor_ids: List[str],
        metric_col: str = "metric",
        direction: str = "maximize",
    ) -> float:
        """
        Compute held-out selection regret on untouched confirmation items.
        Regret = mean_Y(c_oracle) - mean_Y(c_selected).
        """
        # Filter confirmation data for oracle config
        oracle_mask = pd.Series(True, index=confirmation_df.index)
        for f_id in factor_ids:
            oracle_mask &= (confirmation_df[f_id] == oracle_best_config[f_id])
        oracle_mean = confirmation_df.loc[oracle_mask, metric_col].mean()

        # Filter confirmation data for selected config
        selected_mask = pd.Series(True, index=confirmation_df.index)
        for f_id in factor_ids:
            selected_mask &= (confirmation_df[f_id] == selected_config[f_id])
        selected_mean = confirmation_df.loc[selected_mask, metric_col].mean()

        if direction == "maximize":
            return float(oracle_mean - selected_mean)
        else:
            return float(selected_mean - oracle_mean)

    @staticmethod
    def calculate_uplift_ratio(
        selected_config: Dict[str, int],
        oracle_best_config: Dict[str, int],
        baseline_config: Dict[str, int],
        confirmation_df: pd.DataFrame,
        factor_ids: List[str],
        metric_col: str = "metric",
        delta_sesoi: float = 0.0,
    ) -> float:
        """
        Compute proportion of oracle uplift captured by the selected config on confirmation items.
        If oracle uplift is below SESOI, returns NaN (per real-llm-experiment-plan Section 6.2).
        """
        def mean_for_config(cfg: Dict[str, int]) -> float:
            mask = pd.Series(True, index=confirmation_df.index)
            for f_id in factor_ids:
                mask &= (confirmation_df[f_id] == cfg[f_id])
            return float(confirmation_df.loc[mask, metric_col].mean())

        mean_selected = mean_for_config(selected_config)
        mean_oracle = mean_for_config(oracle_best_config)
        mean_base = mean_for_config(baseline_config)

        denom = mean_oracle - mean_base
        if denom <= delta_sesoi or abs(denom) < 1e-8:
            return float("nan")
        return float((mean_selected - mean_base) / denom)


    @staticmethod
    def evaluate_method_performance(
        result: MethodResult,
        oracle_marginal_effects: Dict[str, float],
        oracle_best_config: Dict[str, int],
        oracle_baseline_config: Dict[str, int],
        confirmation_df: pd.DataFrame,
        factor_ids: List[str],
        delta_sesoi: float = 0.03,
        alpha: float = 0.05,
    ) -> EvaluationMetrics:
        """
        Calculate full suite of statistical estimands: effect recovery, calibration, and regret.
        """
        errors = []
        abs_errors = []
        sign_matches = []
        covered_95 = []
        false_positives = 0
        true_negatives = 0
        true_positives = 0
        false_negatives = 0

        for f_id in factor_ids:
            true_tau = oracle_marginal_effects[f_id]
            est_tau = result.estimated_main_effects.get(f_id, 0.0)
            p_val = result.p_values.get(f_id, 1.0)
            ci_95 = result.conf_intervals_95.get(f_id, (est_tau, est_tau))

            # 1. Estimation errors
            err = est_tau - true_tau
            errors.append(err)
            abs_errors.append(abs(err))

            # 2. Sign agreement for meaningful effects
            if abs(true_tau) > delta_sesoi:
                sign_match = (np.sign(est_tau) == np.sign(true_tau))
                sign_matches.append(float(sign_match))

            # 3. 95% Confidence interval coverage
            is_covered = (ci_95[0] <= true_tau <= ci_95[1])
            covered_95.append(float(is_covered))

            # 4. Multiplicity & hypothesis testing classification
            is_null = abs(true_tau) < 1e-6
            is_rejected = (p_val < alpha)

            if is_null:
                if is_rejected:
                    false_positives += 1
                else:
                    true_negatives += 1
            elif abs(true_tau) > delta_sesoi:
                if is_rejected:
                    true_positives += 1
                else:
                    false_negatives += 1

        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(abs_errors))
        sign_acc = float(np.mean(sign_matches)) if sign_matches else 1.0
        coverage = float(np.mean(covered_95))

        total_null = false_positives + true_negatives
        type_1_err = float(false_positives / total_null) if total_null > 0 else 0.0

        total_active = true_positives + false_negatives
        power = float(true_positives / total_active) if total_active > 0 else 1.0

        total_discoveries = true_positives + false_positives
        fdr = float(false_positives / total_discoveries) if total_discoveries > 0 else 0.0

        # Regret & uplift
        regret = EstimandCalculator.calculate_regret(
            selected_config=result.selected_config,
            oracle_best_config=oracle_best_config,
            confirmation_df=confirmation_df,
            factor_ids=factor_ids,
        )

        uplift_ratio = EstimandCalculator.calculate_uplift_ratio(
            selected_config=result.selected_config,
            oracle_best_config=oracle_best_config,
            baseline_config=oracle_baseline_config,
            confirmation_df=confirmation_df,
            factor_ids=factor_ids,
        )

        optimal_selected = all(
            result.selected_config[f] == oracle_best_config[f] for f in factor_ids
        )

        return EvaluationMetrics(
            method_id=result.method_id,
            rmse=rmse,
            mae=mae,
            sign_accuracy_above_delta=sign_acc,
            coverage_95=coverage,
            type_1_error=type_1_err,
            power_above_delta=power,
            fdr=fdr,
            held_out_regret=regret,
            uplift_captured_ratio=uplift_ratio,
            optimal_config_selected=optimal_selected,
        )
