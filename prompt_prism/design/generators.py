"""
Design Generators: Fractional Factorial (2^(k-p)), Plackett-Burman, and Full Factorial.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.models import DesignMatrix, RunConfig
from .catalog import get_catalog_entry

# Standard Plackett-Burman generating vectors (first row, coded as +1 / -1)
# fmt: off
PB_GENERATING_VECTORS: Dict[int, List[int]] = {
    8:  [1, 1, 1, -1, 1, -1, -1],  # 7 factors
    12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],  # 11 factors
    16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],  # 15 factors
    20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],  # 19 factors
    24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],  # 23 factors (corrected orthogonal)
}
# fmt: on


def parse_generator(gen_str: str) -> Tuple[str, List[str]]:
    """
    Parse generator string like 'E=ABCD' or 'E = A B C D' into (target_factor, [source_factors]).
    """
    parts = gen_str.replace(" ", "").split("=")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid generator string: '{gen_str}'. Expected format like 'E=ABCD'"
        )
    target = parts[0].strip()
    sources = list(parts[1].strip())
    return target, sources


class FractionalFactorialGenerator:
    """
    Generates 2^(k-p) Fractional Factorial design matrices.
    """

    @classmethod
    def from_plan_id(
        cls,
        plan_id: str,
        factor_names: Optional[Sequence[str]] = None,
    ) -> DesignMatrix:
        """
        Generate design matrix from a standard catalog plan ID (e.g. '2(5-1)V', '2(7-4)III').
        """
        entry = get_catalog_entry(plan_id)
        if not entry:
            raise ValueError(
                f"Unknown plan_id '{plan_id}'. Use list_available_plans() to see supported designs."
            )

        num_factors = entry["num_factors"]
        base_factors = entry["base_factors"]
        generators = entry["generators"]
        factor_letters = list(entry["factors"])

        names = (
            list(factor_names)
            if factor_names
            else [f"Factor_{f}" for f in factor_letters]
        )
        if len(names) < num_factors:
            names.extend(
                [f"Factor_{factor_letters[i]}" for i in range(len(names), num_factors)]
            )

        return cls.create(
            base_factors=base_factors,
            generators=generators,
            all_factors=factor_letters,
            factor_names=names,
            plan_id=entry["plan_id"],
            resolution=entry["resolution"],
            metadata={
                "base_factors": base_factors,
                "fraction": entry["fraction"],
                "num_runs": entry["runs"],
                "num_factors": num_factors,
                "identity": entry.get("identity", ""),
            },
        )

    @classmethod
    def create(
        cls,
        base_factors: Sequence[str],
        generators: Sequence[str],
        all_factors: Optional[Sequence[str]] = None,
        factor_names: Optional[Sequence[str]] = None,
        plan_id: str = "custom",
        resolution: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DesignMatrix:
        """
        Construct fractional factorial design matrix from base factors and generator equations.
        """
        q = len(base_factors)

        # Step 1: Generate Full Factorial grid for base factors in {0, 1}
        grid = list(itertools.product([0, 1], repeat=q))
        df_base = pd.DataFrame(grid, columns=list(base_factors))

        # Step 2: Compute generated factors using modulo-2 arithmetic (XOR in {0,1} space)
        df_full = df_base.copy()
        for gen in generators:
            target, sources = parse_generator(gen)
            # Modulo 2 sum of binary columns
            df_full[target] = df_full[sources].sum(axis=1) % 2

        # Step 3: Align column order
        if all_factors:
            col_order = [f for f in all_factors if f in df_full.columns]
        else:
            col_order = list(df_full.columns)

        df_final = df_full[col_order]

        # Step 4: Build RunConfig objects
        factor_names_list = (
            list(factor_names) if factor_names else [f"Factor_{c}" for c in col_order]
        )
        runs: List[RunConfig] = []
        for idx, row in df_final.iterrows():
            levels = {col: int(row[col]) for col in col_order}
            combination_str = "".join(str(levels[col]) for col in col_order)
            runs.append(
                RunConfig(
                    run_id=idx + 1,
                    factor_levels=levels,
                    combination_string=combination_str,
                )
            )

        return DesignMatrix(
            plan_id=plan_id,
            factor_ids=col_order,
            factor_names=factor_names_list[: len(col_order)],
            resolution=resolution,
            runs=runs,
            generators=list(generators),
            metadata=metadata or {},
        )


class PlackettBurmanGenerator:
    """
    Generates orthogonal Plackett-Burman screening design matrices.
    """

    @classmethod
    def create(
        cls,
        num_factors: Optional[int] = None,
        factor_ids: Optional[Sequence[str]] = None,
        factor_names: Optional[Sequence[str]] = None,
        runs: Optional[int] = None,
    ) -> DesignMatrix:
        """
        Build Plackett-Burman design matrix for up to N-1 factors.
        """
        # Determine number of runs N in {8, 12, 16, 20, 24}
        if runs is not None:
            if runs not in PB_GENERATING_VECTORS:
                valid_runs = sorted(PB_GENERATING_VECTORS.keys())
                raise ValueError(
                    f"Unsupported runs={runs} for Plackett-Burman. Choose from {valid_runs}."
                )
            n_runs = runs
        else:
            k = num_factors or (len(factor_ids) if factor_ids else 7)
            candidates = [n for n in sorted(PB_GENERATING_VECTORS.keys()) if n - 1 >= k]
            if not candidates:
                raise ValueError(
                    f"Plackett-Burman supports up to 23 factors. Requested: {k}"
                )
            n_runs = candidates[0]

        base_vec = PB_GENERATING_VECTORS[n_runs]
        max_factors = n_runs - 1

        # Cyclic permutations of first row
        rows: List[List[int]] = []
        k = max_factors
        for i in range(k):
            # Cyclic shift right by i positions
            shift = (k - i) % k
            row = base_vec[shift:] + base_vec[:shift]
            rows.append(row)

        # Last row is all -1
        rows.append([-1] * k)

        # Convert {-1, +1} to {0, 1} (where -1 -> 0, +1 -> 1)
        matrix_binary = np.where(np.array(rows) == 1, 1, 0)

        # Determine factors to keep
        if factor_ids:
            k_keep = min(len(factor_ids), max_factors)
            fids = list(factor_ids)[:k_keep]
        elif num_factors:
            k_keep = min(num_factors, max_factors)
            alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
            fids = [
                alphabet[i] if i < len(alphabet) else f"X{i+1}" for i in range(k_keep)
            ]
        else:
            k_keep = max_factors
            alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
            fids = [
                alphabet[i] if i < len(alphabet) else f"X{i+1}" for i in range(k_keep)
            ]

        fnames = (
            list(factor_names)[:k_keep]
            if factor_names
            else [f"Factor_{fid}" for fid in fids]
        )

        runs_list: List[RunConfig] = []
        for idx in range(n_runs):
            levels = {fids[j]: int(matrix_binary[idx, j]) for j in range(k_keep)}
            combination_str = "".join(str(levels[fid]) for fid in fids)
            runs_list.append(
                RunConfig(
                    run_id=idx + 1,
                    factor_levels=levels,
                    combination_string=combination_str,
                )
            )

        return DesignMatrix(
            plan_id=f"PB-{n_runs}",
            factor_ids=fids,
            factor_names=fnames,
            resolution=3,  # Plackett-Burman designs are Resolution III screening designs
            runs=runs_list,
            metadata={
                "num_runs": n_runs,
                "max_factors": max_factors,
                "design_type": "Plackett-Burman",
            },
        )


class FullFactorialGenerator:
    """
    Generates Full Factorial design matrices.
    """

    @classmethod
    def create(
        cls,
        num_factors: Optional[int] = None,
        factor_ids: Optional[Sequence[str]] = None,
        factor_names: Optional[Sequence[str]] = None,
        levels_per_factor: Optional[Sequence[int]] = None,
    ) -> DesignMatrix:
        """
        Create full factorial grid across all factor levels.
        """
        if factor_ids:
            fids = list(factor_ids)
        elif num_factors:
            alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
            fids = [
                alphabet[i] if i < len(alphabet) else f"X{i+1}"
                for i in range(num_factors)
            ]
        else:
            raise ValueError("Must provide either num_factors or factor_ids.")

        k = len(fids)
        fnames = (
            list(factor_names)[:k]
            if factor_names
            else [f"Factor_{fid}" for fid in fids]
        )

        if levels_per_factor:
            level_ranges = [list(range(num_lvls)) for num_lvls in levels_per_factor]
        else:
            level_ranges = [[0, 1] for _ in range(k)]

        grid = list(itertools.product(*level_ranges))

        runs_list: List[RunConfig] = []
        for idx, combination in enumerate(grid):
            levels = {fids[j]: int(combination[j]) for j in range(k)}
            comb_str = "".join(str(v) for v in combination)
            runs_list.append(
                RunConfig(
                    run_id=idx + 1,
                    factor_levels=levels,
                    combination_string=comb_str,
                )
            )

        return DesignMatrix(
            plan_id=(
                f"FullFactorial-2^{k}"
                if not levels_per_factor
                else f"FullFactorial-{len(grid)}runs"
            ),
            factor_ids=fids,
            factor_names=fnames,
            resolution=8,  # Full factorial has no confounding
            runs=runs_list,
            metadata={
                "num_runs": len(grid),
                "num_factors": k,
                "design_type": "FullFactorial",
            },
        )
