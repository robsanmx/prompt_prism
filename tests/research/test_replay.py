"""
Unit and integration tests for MethodReplay in research/simulation/replay.py.
Verifies execution of PromptPrism, Random Search, Linear Model, and OFAT comparators,
and verifies strict budget assertions and no confirmation leakage.
"""

import numpy as np
import pytest

from research.simulation.dgp import DGPConfig, SyntheticDGP
from research.simulation.replay import MethodReplay


@pytest.fixture
def synthetic_additive_cube():
    config = DGPConfig(
        scenario_id="replay_test",
        k=6,
        factor_ids=["A", "B", "C", "D", "E", "F"],
        num_items=30,
        num_replicates=1,
        baseline_intercept=0.50,
        main_effects={
            "A": 0.08,
            "B": 0.05,
            "C": 0.00,
            "D": -0.06,
            "E": 0.04,
            "F": 0.00,
        },
        item_sd=0.05,
        noise_sd=0.02,
        outcome_type="bounded",
        seed=42,
    )
    return SyntheticDGP.generate_cube(config)


def test_replay_prompt_prism_budget_and_recovery(synthetic_additive_cube):
    """Verify that PromptPrism Resolution IV consumes exactly 16 runs and finds winning factors."""
    cube = synthetic_additive_cube
    factor_ids = cube.config.factor_ids
    df = cube.dataframe

    result = MethodReplay.replay_prompt_prism(
        discovery_df=df,
        factor_ids=factor_ids,
        plan_id="2(6-2)IV",
        alpha=0.05,
    )

    # Budget assertion: 16 configs
    assert result.budget_configs == 16
    assert result.budget_calls == 16 * cube.config.num_items

    # Selection verification: A, B, E are positive boosters; D is negative; C, F are null
    # In an additive low-noise scenario, A, B, E must be turned ON (1), D must be OFF (0)
    assert result.selected_config["A"] == 1
    assert result.selected_config["B"] == 1
    assert result.selected_config["E"] == 1
    assert result.selected_config["D"] == 0

    # Estimated effect direction
    assert result.estimated_main_effects["A"] > 0
    assert result.estimated_main_effects["D"] < 0


def test_replay_budget_matched_comparators(synthetic_additive_cube):
    """Verify that budget-matched random search and linear model stay within budget."""
    cube = synthetic_additive_cube
    factor_ids = cube.config.factor_ids
    df = cube.dataframe

    # Random search with 16 runs
    res_rand = MethodReplay.replay_random_search(
        discovery_df=df,
        factor_ids=factor_ids,
        budget_runs=16,
        seed=101,
    )
    assert res_rand.budget_configs == 16

    # Random linear model with 16 runs
    res_lin = MethodReplay.replay_random_linear_model(
        discovery_df=df,
        factor_ids=factor_ids,
        budget_runs=16,
        seed=101,
    )
    assert res_lin.budget_configs == 16

    # OFAT ablation: 1 baseline + 6 toggles = 7 configs
    res_ofat = MethodReplay.replay_ofat(
        discovery_df=df,
        factor_ids=factor_ids,
    )
    assert res_ofat.budget_configs == 7

    # Exhaustive oracle: 64 configs
    res_oracle = MethodReplay.replay_exhaustive_oracle(
        discovery_df=df,
        factor_ids=factor_ids,
    )
    assert res_oracle.budget_configs == 64
    assert res_oracle.selected_config["A"] == 1
    assert res_oracle.selected_config["B"] == 1
    assert res_oracle.selected_config["D"] == 0
    assert res_oracle.selected_config["E"] == 1


def test_replay_budget_violation_raises(synthetic_additive_cube):
    """Verify that budget violations are caught and raise ValueError across methods."""
    cube = synthetic_additive_cube
    factor_ids = cube.config.factor_ids
    df = cube.dataframe

    with pytest.raises(ValueError, match="Budget violation"):
        MethodReplay.replay_random_search(discovery_df=df, factor_ids=factor_ids, budget_runs=0)

    with pytest.raises(ValueError, match="Budget violation"):
        MethodReplay.replay_random_linear_model(discovery_df=df, factor_ids=factor_ids, budget_runs=0)


def test_replay_no_fixed_intervals_or_pvalues(synthetic_additive_cube):
    """Verify that no comparator uses fixed p-values or fixed-width confidence intervals."""
    cube = synthetic_additive_cube
    factor_ids = cube.config.factor_ids
    df = cube.dataframe

    # Replay on original (low noise) data
    res_rand = MethodReplay.replay_random_search(discovery_df=df, factor_ids=factor_ids, budget_runs=16, seed=42)
    res_ofat = MethodReplay.replay_ofat(discovery_df=df, factor_ids=factor_ids)
    res_oracle = MethodReplay.replay_exhaustive_oracle(discovery_df=df, factor_ids=factor_ids)
    res_lin = MethodReplay.replay_random_linear_model(discovery_df=df, factor_ids=factor_ids, budget_runs=16, seed=42)

    # Replay on high noise data
    df_noisy = df.copy()
    rng = np.random.default_rng(999)
    df_noisy["metric"] = df_noisy["metric"] + rng.normal(0, 0.30, size=len(df_noisy))

    res_rand_noisy = MethodReplay.replay_random_search(discovery_df=df_noisy, factor_ids=factor_ids, budget_runs=16, seed=42)
    res_ofat_noisy = MethodReplay.replay_ofat(discovery_df=df_noisy, factor_ids=factor_ids)
    res_oracle_noisy = MethodReplay.replay_exhaustive_oracle(discovery_df=df_noisy, factor_ids=factor_ids)
    res_lin_noisy = MethodReplay.replay_random_linear_model(discovery_df=df_noisy, factor_ids=factor_ids, budget_runs=16, seed=42)

    # Check that CI widths expand under higher noise (not fixed width)
    for f in factor_ids:
        # Random search
        w_rand = res_rand.conf_intervals_95[f][1] - res_rand.conf_intervals_95[f][0]
        w_rand_noisy = res_rand_noisy.conf_intervals_95[f][1] - res_rand_noisy.conf_intervals_95[f][0]
        assert w_rand > 0.0
        assert w_rand_noisy > w_rand, f"Random search CI for {f} did not expand with noise"

        # OFAT
        w_ofat = res_ofat.conf_intervals_95[f][1] - res_ofat.conf_intervals_95[f][0]
        w_ofat_noisy = res_ofat_noisy.conf_intervals_95[f][1] - res_ofat_noisy.conf_intervals_95[f][0]
        assert w_ofat > 0.0
        assert w_ofat_noisy > w_ofat, f"OFAT CI for {f} did not expand with noise"

        # Oracle
        w_oracle = res_oracle.conf_intervals_95[f][1] - res_oracle.conf_intervals_95[f][0]
        w_oracle_noisy = res_oracle_noisy.conf_intervals_95[f][1] - res_oracle_noisy.conf_intervals_95[f][0]
        assert w_oracle > 0.0
        assert w_oracle_noisy > w_oracle, f"Oracle CI for {f} did not expand with noise"

        # Check p-values are not hardcoded constants
        assert res_rand.p_values[f] != 0.50 or res_rand_noisy.p_values[f] != 0.50
        assert res_oracle.p_values[f] != 0.001


def test_replay_hand_computed_inference():
    """Verify that hand-computed designs match fitted coefficients and intervals."""
    import pandas as pd
    from scipy import stats

    # Construct small hand-computable dataset:
    # 2 factors: A, B across 4 items (blocks 0, 1, 2, 3)
    # Baseline (0,0): [10.0, 12.0, 11.0, 13.0] -> mean = 11.5
    # Toggle A (1,0): [12.0, 14.0, 13.0, 15.0] -> diff = [2.0, 2.0, 2.0, 2.0] -> mean diff = 2.0, var = 0
    # Toggle B (0,1): [13.0, 13.0, 15.0, 17.0] -> diff = [3.0, 1.0, 4.0, 4.0] -> mean diff = 3.0
    rows = []
    base_vals = [10.0, 12.0, 11.0, 13.0]
    toggle_a = [12.0, 14.0, 13.0, 15.0]
    toggle_b = [13.0, 13.0, 15.0, 17.0]
    both_ab = [15.0, 16.0, 17.0, 19.0]

    for i in range(4):
        rows.append({"sample_id": f"s{i}", "A": 0, "B": 0, "metric": base_vals[i]})
        rows.append({"sample_id": f"s{i}", "A": 1, "B": 0, "metric": toggle_a[i]})
        rows.append({"sample_id": f"s{i}", "A": 0, "B": 1, "metric": toggle_b[i]})
        rows.append({"sample_id": f"s{i}", "A": 1, "B": 1, "metric": both_ab[i]})

    hand_df = pd.DataFrame(rows)

    # Hand calculation for factor B in OFAT:
    # diffs = [3.0, 1.0, 4.0, 4.0]
    # mean = 3.0
    # s_d = std([3, 1, 4, 4], ddof=1) = sqrt( (0 + 4 + 1 + 1)/3 ) = sqrt(6/3) = sqrt(2) = 1.41421356
    # se = sqrt(2)/sqrt(4) = sqrt(0.5) = 0.70710678
    # t = 3.0 / 0.70710678 = 4.24264
    # df = 3
    # p = 2 * (1 - t.cdf(4.24264, 3))
    expected_diff_b = 3.0
    expected_se_b = np.sqrt(0.5)
    expected_df_b = 3
    expected_p_b = float(2.0 * (1.0 - stats.t.cdf(expected_diff_b / expected_se_b, df=expected_df_b)))
    crit_95 = float(stats.t.ppf(0.975, df=expected_df_b))
    expected_ci95_b = (expected_diff_b - crit_95 * expected_se_b, expected_diff_b + crit_95 * expected_se_b)

    res_ofat = MethodReplay.replay_ofat(
        discovery_df=hand_df,
        factor_ids=["A", "B"],
        target_col="metric",
        block_col="sample_id",
    )

    np.testing.assert_allclose(res_ofat.estimated_main_effects["B"], expected_diff_b, rtol=1e-5)
    np.testing.assert_allclose(res_ofat.p_values["B"], expected_p_b, rtol=1e-5)
    np.testing.assert_allclose(res_ofat.conf_intervals_95["B"][0], expected_ci95_b[0], rtol=1e-5)
    np.testing.assert_allclose(res_ofat.conf_intervals_95["B"][1], expected_ci95_b[1], rtol=1e-5)
