"""
Design Generators: Fractional Factorial (2^(k-p)), Plackett-Burman, and Full Factorial.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from ..core.factors import Factor, FactorSet
from ..core.models import DesignMatrix, RunConfig
from .catalog import CATALOG_DESIGNS, get_catalog_entry


# Standard Plackett-Burman generating vectors (first row, coded as +1 / -1)
PB_GENERATING_VECTORS: Dict[int, List[int]] = {
    8: [1, 1, 1, -1, 1, -1, -1],  # 7 factors
    12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],  # 11 factors
    16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],  # 15 factors
    20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],  # 19 factors
    24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1],  # 23 factors
}


def parse_generator(gen_str: str) -> Tuple[str, List[str]]:
    """
    Parse generator string like 'E=ABCD' or 'E = A B C D' into (target_factor, [source_factors]).
    """
    parts = gen_str.replace(" ", "").split("=")
    if len(parts) != 2:
        raise ValueError(f"Invalid generator string: '{gen_str}'. Expected format like 'E=ABCD'")
    target = parts[0].strip()
    sources = list(parts[1].strip())
    return target, sources


class FractionalFactorialGenerator:
    """
    Generates 2^(k-p) Fractional Factorial design matrices.
    """

    @classmethod
    def from_plan_id(cls, plan_id: str, factor_names: Optional[Sequence[str]] = None) -> DesignMatrix:
        """Create a DesignMatrix from a standard catalog plan ID."""
        entry = get_catalog_entry(plan_id)
        if not entry:
            raise ValueError(f"Plan ID '{plan_id}' not found in catalog. Available: {list(CATALOG_DESIGNS.keys())}")

        num_factors = entry["num_factors"]
        base_factors = entry["base_factors"]
        generators = entry["generators"]
        factor_ids = list(entry["factors"])

        return cls.create(
            base_factors=base_factors,
            generators=generators,
            all_factor_ids=factor_ids,
            factor_names=factor_names,
            plan_id=entry["plan_id"],
            resolution=entry["resolution"],
        )

    @classmethod
    def create(
        cls,
        base_factors: Sequence[str],
        generators: Sequence[str],
        all_factor_ids: Optional[Sequence[str]] = None,
        factor_names: Optional[Sequence[str]] = None,
        plan_id: str = "custom_fractional",
        resolution: Optional[int] = None,
    ) -> DesignMatrix:
        """
        Construct fractional factorial design matrix from base factors and generator expressions.
        
        Args:
            base_factors: e.g. ['A', 'B', 'C', 'D'] (q = 4, so 2^4 = 16 runs)
            generators: e.g. ['E=ABC', 'F=BCD']
            all_factor_ids: Ordered list of all factor IDs.
            factor_names: Optional human-readable factor names mapped 1:1 to factor_ids.
            plan_id: Identifier name.
            resolution: Resolution level.
        """
        q = len(base_factors)
        num_runs = 2**q

        # Generate base factor combinations in standard Yates / binary order (0 and 1)
        base_cols: Dict[str, np.ndarray] = {}
        for i, b_factor in enumerate(base_factors):
            # Frequency repeats: 2^i
            col = np.array([int((r >> i) & 1) for r in range(num_runs)])
            base_cols[b_factor] = col

        # Build full table
        all_cols: Dict[str, np.ndarray] = dict(base_cols)

        # Compute generated columns via XOR (GF(2) arithmetic)
        gen_formulas = []
        for gen in generators:
            target, sources = parse_generator(gen)
            gen_formulas.append(f"{target}={''.join(sources)}")
            # XOR all sources
            res_col = np.zeros(num_runs, dtype=int)
            for s in sources:
                if s not in all_cols:
                    raise ValueError(f"Generator '{gen}' references undefined source factor '{s}'")
                res_col = res_col ^ all_cols[s]
            all_cols[target] = res_col

        # Determine final ordered factor IDs
        if all_factor_ids is None:
            ordered_ids = list(base_factors) + [parse_generator(g)[0] for g in generators]
        else:
            ordered_ids = list(all_factor_ids)

        # Build runs
        runs: List[RunConfig] = []
        name_map = {}
        if factor_names and len(factor_names) == len(ordered_ids):
            name_map = dict(zip(ordered_ids, factor_names))

        for r_idx in range(num_runs):
            f_levels: Dict[str, int] = {}
            f_names: Dict[str, int] = {}
            for fid in ordered_ids:
                val = int(all_cols[fid][r_idx])
                f_levels[fid] = val
                if fid in name_map:
                    f_names[name_map[fid]] = val

            runs.append(
                RunConfig(
                    run_id=r_idx + 1,
                    factor_levels=f_levels,
                    factor_names=f_names,
                    metadata={"plan_id": plan_id},
                )
            )

        return DesignMatrix(
            plan_id=plan_id,
            factor_ids=ordered_ids,
            runs=runs,
            resolution=resolution,
            generators=gen_formulas,
            metadata={
                "base_factors": list(base_factors),
                "fraction": len(generators),
                "num_runs": num_runs,
                "num_factors": len(ordered_ids),
            },
        )


class PlackettBurmanGenerator:
    """
    Generates Plackett-Burman screening designs for N = 8, 12, 16, 20, 24 runs.
    """

    @classmethod
    def create(
        cls,
        num_factors: int,
        factor_names: Optional[Sequence[str]] = None,
        factor_ids: Optional[Sequence[str]] = None,
        runs: Optional[int] = None,
    ) -> DesignMatrix:
        """
        Create a Plackett-Burman design for screening num_factors.
        
        Args:
            num_factors: Number of factors to screen (e.g. 7, 10, 11).
            factor_names: Optional names of factors.
            factor_ids: Optional IDs (A, B, C...).
            runs: Optional explicit run count (must be multiple of 4: 8, 12, 16, 20, 24).
        """
        # Find smallest available PB run size >= num_factors + 1
        available_sizes = sorted(PB_GENERATING_VECTORS.keys())
        if runs is not None:
            if runs not in PB_GENERATING_VECTORS:
                raise ValueError(f"Plackett-Burman run size {runs} not supported. Available: {available_sizes}")
            target_runs = runs
        else:
            valid_sizes = [s for s in available_sizes if s >= num_factors + 1]
            if not valid_sizes:
                raise ValueError(f"Plackett-Burman supports up to {max(available_sizes)-1} factors (requested {num_factors})")
            target_runs = valid_sizes[0]

        # Generate matrix
        first_row = PB_GENERATING_VECTORS[target_runs]
        k = len(first_row)
        
        # Cyclic permutation of first row
        matrix_pm = []
        for i in range(k):
            row = first_row[-i:] + first_row[:-i] if i > 0 else list(first_row)
            matrix_pm.append(row)
        # Final row of all -1
        matrix_pm.append([-1] * k)

        matrix = np.array(matrix_pm)  # shape (target_runs, k)
        # Convert -1/+1 to 0/1: -1 -> 0, +1 -> 1
        matrix_01 = np.where(matrix == 1, 1, 0)

        # Slice to requested number of factors
        matrix_01 = matrix_01[:, :num_factors]

        # Assign factor IDs
        if factor_ids is None:
            alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
            f_ids = [alphabet[i] if i < len(alphabet) else f"X{i+1}" for i in range(num_factors)]
        else:
            f_ids = list(factor_ids)[:num_factors]

        name_map = {}
        if factor_names and len(factor_names) == len(f_ids):
            name_map = dict(zip(f_ids, factor_names))

        run_configs: List[RunConfig] = []
        for r_idx in range(target_runs):
            f_levels = {f_ids[c]: int(matrix_01[r_idx, c]) for c in range(num_factors)}
            f_names = {name_map[f_ids[c]]: int(matrix_01[r_idx, c]) for c in range(num_factors) if f_ids[c] in name_map}
            run_configs.append(
                RunConfig(
                    run_id=r_idx + 1,
                    factor_levels=f_levels,
                    factor_names=f_names,
                    metadata={"design_type": "Plackett-Burman", "runs": target_runs},
                )
            )

        return DesignMatrix(
            plan_id=f"PB-{target_runs}(k={num_factors})",
            factor_ids=f_ids,
            runs=run_configs,
            resolution=3,  # PB designs are Resolution III screening designs
            metadata={
                "design_type": "Plackett-Burman",
                "num_factors": num_factors,
                "num_runs": target_runs,
            },
        )


class FullFactorialGenerator:
    """
    Generates Full Factorial designs (all 2^k combinations or multi-level grids).
    """

    @classmethod
    def create(
        cls,
        num_factors: Optional[int] = None,
        factor_ids: Optional[Sequence[str]] = None,
        factor_names: Optional[Sequence[str]] = None,
        levels_per_factor: Optional[Union[int, Sequence[int]]] = None,
    ) -> DesignMatrix:
        """
        Create a Full Factorial design.
        
        Args:
            num_factors: Number of factors.
            factor_ids: List of factor IDs (e.g. ['A', 'B', 'C']).
            factor_names: Optional factor names.
            levels_per_factor: Levels per factor (default 2 for binary design).
        """
        if factor_ids is None:
            if num_factors is None:
                raise ValueError("Either factor_ids or num_factors must be provided")
            alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
            factor_ids = [alphabet[i] if i < len(alphabet) else f"X{i+1}" for i in range(num_factors)]
        else:
            factor_ids = list(factor_ids)
            num_factors = len(factor_ids)

        if levels_per_factor is None:
            level_ranges = [range(2) for _ in range(num_factors)]
        elif isinstance(levels_per_factor, int):
            level_ranges = [range(levels_per_factor) for _ in range(num_factors)]
        else:
            level_ranges = [range(n) for n in levels_per_factor]

        combinations = list(itertools.product(*level_ranges))

        name_map = {}
        if factor_names and len(factor_names) == len(factor_ids):
            name_map = dict(zip(factor_ids, factor_names))

        runs: List[RunConfig] = []
        for r_idx, comb in enumerate(combinations):
            f_levels = {fid: int(val) for fid, val in zip(factor_ids, comb)}
            f_names = {name_map[fid]: int(val) for fid, val in zip(factor_ids, comb) if fid in name_map}
            runs.append(
                RunConfig(
                    run_id=r_idx + 1,
                    factor_levels=f_levels,
                    factor_names=f_names,
                    metadata={"design_type": "FullFactorial"},
                )
            )

        return DesignMatrix(
            plan_id=f"Full-2^{num_factors}" if levels_per_factor in (None, 2) else f"Full-Grid({len(combinations)}runs)",
            factor_ids=factor_ids,
            runs=runs,
            resolution=99,  # Full factorial has no confounding
            metadata={"design_type": "FullFactorial", "num_factors": num_factors, "num_runs": len(runs)},
        )
