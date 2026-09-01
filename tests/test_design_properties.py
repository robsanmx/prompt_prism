"""
Design Property Tests: Orthogonality, balance, and resolution verification for all DoE generators.
"""

import numpy as np

from prompt_prism.design.aliasing import AliasStructure
from prompt_prism.design.catalog import CATALOG_DESIGNS
from prompt_prism.design.generators import (
    PB_GENERATING_VECTORS,
    FractionalFactorialGenerator,
    PlackettBurmanGenerator,
)


def test_all_catalog_designs_orthogonal():
    """Verify that every single catalog fractional factorial design matrix is orthogonal."""
    for plan_id, entry in CATALOG_DESIGNS.items():
        design = FractionalFactorialGenerator.from_plan_id(plan_id)
        df = design.to_dataframe()
        factor_cols = design.factor_ids

        # Map {0, 1} to {-1, +1}
        X = np.where(df[factor_cols].values == 1, 1, -1)
        XTX = X.T @ X

        expected_diag = design.num_runs * np.eye(design.num_factors)
        np.testing.assert_array_almost_equal(
            XTX,
            expected_diag,
            err_msg=f"Plan {plan_id} is NOT orthogonal: off-diagonal elements detected.",
        )


def test_catalog_resolution_matches_label():
    """Verify that computed resolution matches the catalog entry's resolution and Roman numeral in plan_id."""
    roman_to_int = {
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
    }

    for plan_id, entry in CATALOG_DESIGNS.items():
        generators = entry.get("generators", [])
        if not generators:
            continue

        alias = AliasStructure(generators)
        computed_res = alias.resolution
        declared_res = entry["resolution"]

        # Check declared resolution
        assert (
            computed_res == declared_res
        ), f"Plan {plan_id}: computed resolution {computed_res} != declared resolution {declared_res}"

        # Check Roman numeral suffix in plan_id
        for roman, val in roman_to_int.items():
            if plan_id.endswith(roman):
                assert (
                    computed_res == val
                ), f"Plan {plan_id}: label suffix {roman} (={val}) != computed resolution {computed_res}"
                break


def test_all_pb_designs_orthogonal():
    """Verify that all Plackett-Burman designs (N = 8, 12, 16, 20, 24) are strictly orthogonal."""
    for n in sorted(PB_GENERATING_VECTORS.keys()):
        pb = PlackettBurmanGenerator.create(runs=n)
        df = pb.to_dataframe()
        factor_cols = pb.factor_ids

        X = np.where(df[factor_cols].values == 1, 1, -1)
        XTX = X.T @ X

        expected_diag = n * np.eye(pb.num_factors)
        np.testing.assert_array_almost_equal(
            XTX,
            expected_diag,
            err_msg=f"Plackett-Burman N={n} is NOT orthogonal.",
        )


def test_pb_run_count_and_balance():
    """Verify PB designs have shape (N, N-1) and each column has balanced 0s and 1s."""
    for n in sorted(PB_GENERATING_VECTORS.keys()):
        pb = PlackettBurmanGenerator.create(runs=n)
        df = pb.to_dataframe()
        assert pb.num_runs == n
        assert pb.num_factors == n - 1
        assert len(df) == n

        for col in pb.factor_ids:
            counts = df[col].value_counts().to_dict()
            assert counts[0] == n // 2
            assert counts[1] == n // 2
