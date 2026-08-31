"""
Unit tests for Fractional Factorial, Plackett-Burman, and Full Factorial Design Generators.
"""

import numpy as np
import pytest
from prompt_prism.design.catalog import CATALOG_DESIGNS, list_available_plans
from prompt_prism.design.generators import (
    FractionalFactorialGenerator,
    FullFactorialGenerator,
    PlackettBurmanGenerator,
)
from prompt_prism.design.recommender import recommend_design


def test_catalog_plans_exist():
    assert len(CATALOG_DESIGNS) >= 20
    plans_5 = list_available_plans(num_factors=5)
    assert len(plans_5) >= 2
    plan_ids = [p["plan_id"] for p in plans_5]
    assert "2(5-1)V" in plan_ids
    assert "2(5-2)III" in plan_ids


def test_fractional_factorial_2_5_minus_1():
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    assert design.plan_id == "2(5-1)V"
    assert design.num_factors == 5
    assert design.num_runs == 16
    assert design.resolution == 5
    assert design.factor_ids == ["A", "B", "C", "D", "E"]

    df = design.to_dataframe()
    assert len(df) == 16
    assert set(df.columns) >= {"run_id", "A", "B", "C", "D", "E", "combination"}

    # Test balance: each factor has exactly 8 runs at 0 and 8 runs at 1
    for col in ["A", "B", "C", "D", "E"]:
        counts = df[col].value_counts().to_dict()
        assert counts[0] == 8
        assert counts[1] == 8

    # Test orthogonality in {-1, +1} space
    X = np.where(df[["A", "B", "C", "D", "E"]].values == 1, 1, -1)
    XTX = X.T @ X
    expected_diag = 16 * np.eye(5)
    np.testing.assert_array_equal(XTX, expected_diag)


def test_fractional_factorial_2_7_minus_4():
    design = FractionalFactorialGenerator.from_plan_id("2(7-4)III")
    assert design.num_factors == 7
    assert design.num_runs == 8
    assert design.resolution == 3
    df = design.to_dataframe()
    assert len(df) == 8

    # Check orthogonality
    X = np.where(df[design.factor_ids].values == 1, 1, -1)
    XTX = X.T @ X
    expected_diag = 8 * np.eye(7)
    np.testing.assert_array_equal(XTX, expected_diag)


def test_fractional_factorial_saturated_2_15_minus_11():
    design = FractionalFactorialGenerator.from_plan_id("2(15-11)III")
    assert design.num_factors == 15
    assert design.num_runs == 16
    assert design.resolution == 3
    df = design.to_dataframe()
    assert len(df) == 16

    X = np.where(df[design.factor_ids].values == 1, 1, -1)
    XTX = X.T @ X
    expected_diag = 16 * np.eye(15)
    np.testing.assert_array_equal(XTX, expected_diag)


def test_plackett_burman_generator():
    pb12 = PlackettBurmanGenerator.create(num_factors=10)
    assert pb12.num_factors == 10
    assert pb12.num_runs == 12
    df = pb12.to_dataframe()
    assert len(df) == 12

    # Check balance: each factor has 6 runs at 1 and 6 runs at 0
    for col in pb12.factor_ids:
        counts = df[col].value_counts().to_dict()
        assert counts[0] == 6
        assert counts[1] == 6

    # Check orthogonality
    X = np.where(df[pb12.factor_ids].values == 1, 1, -1)
    XTX = X.T @ X
    expected_diag = 12 * np.eye(10)
    np.testing.assert_array_equal(XTX, expected_diag)


def test_full_factorial_generator():
    ff = FullFactorialGenerator.create(num_factors=3)
    assert ff.num_factors == 3
    assert ff.num_runs == 8
    df = ff.to_dataframe()
    assert len(df) == 8


def test_recommend_design():
    # 5 factors with max_runs=16 -> 2(5-1)V
    d1 = recommend_design(factors=5, max_runs=16)
    assert d1.num_runs == 16
    assert d1.resolution == 5

    # 7 factors with max_runs=8 -> 2(7-4)III
    d2 = recommend_design(factors=7, max_runs=8)
    assert d2.num_runs == 8
    assert d2.num_factors == 7

    # 10 factors with PB preferred -> PB-12
    d3 = recommend_design(factors=10, prefer_plackett_burman=True)
    assert d3.num_runs == 12
