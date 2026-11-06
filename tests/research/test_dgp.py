"""
Tests for Data Generating Process (DGP) library for Study A.
Verifies noise-free exact contrast recovery, item difficulty variance,
outcome bounds, replicate structure, and deterministic reproducibility.
"""

import numpy as np
import pandas as pd

from research.simulation.dgp import DGPConfig, SyntheticDGP


def test_dgp_noise_free_exact_recovery():
    """Verify that in the absence of noise and bounding, observed contrasts match true effects exactly."""
    config = DGPConfig(
        scenario_id="exact_test",
        k=4,
        factor_ids=["A", "B", "C", "D"],
        num_items=10,
        num_replicates=1,
        baseline_intercept=0.50,
        main_effects={"A": 0.08, "B": -0.04, "C": 0.05, "D": 0.00},
        two_factor_interactions={("A", "B"): 0.03},
        item_sd=0.0,
        noise_sd=0.0,
        outcome_type="continuous",
        missing_rate=0.0,
        seed=123,
    )

    cube = SyntheticDGP.generate_cube(config)
    df = cube.dataframe

    # Verify run count: 2^4 = 16 runs * 10 items = 160 rows
    assert len(df) == 16 * 10

    # Compute observed marginal mean contrast for each factor from the dataframe
    for f_id in config.factor_ids:
        mean_1 = df[df[f_id] == 1]["metric"].mean()
        mean_0 = df[df[f_id] == 0]["metric"].mean()
        observed_tau = mean_1 - mean_0
        oracle_tau = cube.oracle_marginal_effects[f_id]

        assert np.isclose(observed_tau, oracle_tau, atol=1e-6)


def test_dgp_bounded_outcome_strictly_in_range():
    """Verify that bounded outcomes are strictly within [0.0, 1.0]."""
    config = DGPConfig(
        scenario_id="bounded_test",
        k=3,
        factor_ids=["A", "B", "C"],
        num_items=20,
        num_replicates=2,
        baseline_intercept=0.90,  # High baseline
        main_effects={"A": 0.30, "B": 0.20, "C": -0.10},  # Pushes z > 1.0
        item_sd=0.15,
        noise_sd=0.10,
        outcome_type="bounded",
        missing_rate=0.0,
        seed=42,
    )

    cube = SyntheticDGP.generate_cube(config)
    df = cube.dataframe

    assert df["metric"].min() >= 0.0
    assert df["metric"].max() <= 1.0


def test_dgp_binary_outcome_bernoulli():
    """Verify that binary outcomes only take values 0.0 or 1.0."""
    config = DGPConfig(
        scenario_id="binary_test",
        k=3,
        factor_ids=["A", "B", "C"],
        num_items=30,
        num_replicates=1,
        baseline_intercept=0.0,
        main_effects={"A": 0.5, "B": -0.5, "C": 0.0},
        item_sd=0.2,
        outcome_type="binary",
        seed=99,
    )

    cube = SyntheticDGP.generate_cube(config)
    df = cube.dataframe

    unique_vals = set(df["metric"].unique())
    assert unique_vals.issubset({0.0, 1.0})


def test_dgp_replicate_structure_and_item_blocks():
    """Verify that item blocks and replicate IDs are structured properly."""
    num_items = 15
    num_reps = 3
    k = 4
    config = DGPConfig(
        k=k,
        num_items=num_items,
        num_replicates=num_reps,
        seed=77,
    )
    cube = SyntheticDGP.generate_cube(config)
    df = cube.dataframe

    # Total rows = 2^k * num_items * num_reps
    expected_rows = (2**k) * num_items * num_reps
    assert len(df) == expected_rows

    # Each sample_id appears exactly (2^k * num_reps) times
    sample_counts = df["sample_id"].value_counts()
    assert (sample_counts == (2**k) * num_reps).all()

    # Replicates are 0, 1, 2
    assert set(df["replicate"].unique()) == {0, 1, 2}


def test_dgp_missingness_rate():
    """Verify that missing rate produces expected proportion of NaNs and errors."""
    config = DGPConfig(
        k=3,
        num_items=100,
        num_replicates=1,
        missing_rate=0.10,  # 10% missing
        seed=42,
    )
    cube = SyntheticDGP.generate_cube(config)
    df = cube.dataframe

    missing_fraction = df["metric"].isna().mean()
    # Should be close to 0.10 within binomial margin
    assert 0.06 <= missing_fraction <= 0.15
    assert (df["metric"].isna() == df["error"].notna()).all()


def test_dgp_deterministic_seed():
    """Verify that identical seeds produce bit-identical dataframes."""
    config1 = DGPConfig(k=4, num_items=25, seed=12345)
    config2 = DGPConfig(k=4, num_items=25, seed=12345)

    cube1 = SyntheticDGP.generate_cube(config1)
    cube2 = SyntheticDGP.generate_cube(config2)

    pd.testing.assert_frame_equal(cube1.dataframe, cube2.dataframe)
