"""
Unit tests for EstimandCalculator in research/analysis/estimands.py.
Verifies hand-computed regret, uplift ratio, RMSE, sign accuracy, and coverage.
"""

import numpy as np
import pandas as pd
import pytest

from research.analysis.estimands import EstimandCalculator, MethodResult


def test_hand_computed_regret_and_uplift():
    """Verify regret and uplift formulas against hand-computed toy fixture."""
    factor_ids = ["A", "B"]
    # 4 items on confirmation set
    records = [
        # Config (0, 0): base -> mean = 0.50
        {"sample_id": "i1", "A": 0, "B": 0, "metric": 0.48},
        {"sample_id": "i2", "A": 0, "B": 0, "metric": 0.52},
        # Config (1, 0): -> mean = 0.65
        {"sample_id": "i1", "A": 1, "B": 0, "metric": 0.63},
        {"sample_id": "i2", "A": 1, "B": 0, "metric": 0.67},
        # Config (0, 1): -> mean = 0.60
        {"sample_id": "i1", "A": 0, "B": 1, "metric": 0.59},
        {"sample_id": "i2", "A": 0, "B": 1, "metric": 0.61},
        # Config (1, 1): oracle -> mean = 0.80
        {"sample_id": "i1", "A": 1, "B": 1, "metric": 0.79},
        {"sample_id": "i2", "A": 1, "B": 1, "metric": 0.81},
    ]
    conf_df = pd.DataFrame(records)

    oracle_best = {"A": 1, "B": 1}  # mean = 0.80
    baseline = {"A": 0, "B": 0}     # mean = 0.50
    selected_suboptimal = {"A": 1, "B": 0}  # mean = 0.65

    # Regret = 0.80 - 0.65 = 0.15
    regret = EstimandCalculator.calculate_regret(
        selected_config=selected_suboptimal,
        oracle_best_config=oracle_best,
        confirmation_df=conf_df,
        factor_ids=factor_ids,
    )
    assert np.isclose(regret, 0.15, atol=1e-5)

    # Uplift = (0.65 - 0.50) / (0.80 - 0.50) = 0.15 / 0.30 = 0.50 (50%)
    uplift = EstimandCalculator.calculate_uplift_ratio(
        selected_config=selected_suboptimal,
        oracle_best_config=oracle_best,
        baseline_config=baseline,
        confirmation_df=conf_df,
        factor_ids=factor_ids,
    )
    assert np.isclose(uplift, 0.50, atol=1e-5)


def test_hand_computed_evaluation_metrics():
    """Verify RMSE, sign agreement, and 95% CI coverage against known values."""
    factor_ids = ["A", "B", "C"]
    oracle_effects = {"A": 0.050, "B": -0.040, "C": 0.000}
    oracle_best = {"A": 1, "B": 0, "C": 0}
    oracle_base = {"A": 0, "B": 0, "C": 0}

    # Synthetic method result:
    # A: est = +0.060 (err = +0.010, sign match = True, covered = True)
    # B: est = -0.030 (err = +0.010, sign match = True, covered = True)
    # C: est = +0.020 (err = +0.020, null effect, covered = False)
    result = MethodResult(
        method_id="mock_method",
        budget_configs=16,
        budget_calls=1280,
        selected_config={"A": 1, "B": 0, "C": 0},
        estimated_main_effects={"A": 0.060, "B": -0.030, "C": 0.020},
        p_values={"A": 0.001, "B": 0.010, "C": 0.200},
        conf_intervals_95={
            "A": (0.040, 0.080),  # contains 0.050 -> True
            "B": (-0.050, -0.010), # contains -0.040 -> True
            "C": (0.005, 0.035),   # does NOT contain 0.000 -> False
        },
        conf_intervals_90={
            "A": (0.045, 0.075),
            "B": (-0.045, -0.015),
            "C": (0.008, 0.032),
        },
    )

    # Simple conf df
    records = [
        {"sample_id": "i1", "A": 0, "B": 0, "C": 0, "metric": 0.50},
        {"sample_id": "i1", "A": 1, "B": 0, "C": 0, "metric": 0.70},
    ]
    conf_df = pd.DataFrame(records)

    metrics = EstimandCalculator.evaluate_method_performance(
        result=result,
        oracle_marginal_effects=oracle_effects,
        oracle_best_config=oracle_best,
        oracle_baseline_config=oracle_base,
        confirmation_df=conf_df,
        factor_ids=factor_ids,
        delta_sesoi=0.025,
    )

    # Expected errors: [0.010, 0.010, 0.020]
    # Mean squared error: (0.0001 + 0.0001 + 0.0004) / 3 = 0.0006 / 3 = 0.0002
    # RMSE = sqrt(0.0002) = 0.014142
    assert np.isclose(metrics.rmse, np.sqrt(0.0002), atol=1e-5)

    # Sign accuracy above delta: A and B both match sign -> 1.0 (100%)
    assert metrics.sign_accuracy_above_delta == 1.0

    # Coverage 95%: 2 out of 3 factors covered -> 2/3 = 0.6667
    assert np.isclose(metrics.coverage_95, 2.0 / 3.0, atol=1e-4)

    # Regret: selected matches oracle best -> 0.0
    assert metrics.held_out_regret == 0.0
    assert metrics.optimal_config_selected is True
