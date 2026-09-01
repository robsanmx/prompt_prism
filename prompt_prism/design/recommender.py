"""
Design Recommender: Recommends the optimal DoE design plan given factors, budget, and resolution goals.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from ..core.factors import Factor, FactorSet
from ..core.models import DesignMatrix
from .catalog import list_available_plans
from .generators import (
    FractionalFactorialGenerator,
    FullFactorialGenerator,
    PlackettBurmanGenerator,
)


def recommend_design(
    factors: Union[int, Sequence[Factor], Sequence[str], FactorSet],
    max_runs: Optional[int] = None,
    min_resolution: Optional[int] = None,
    prefer_plackett_burman: bool = False,
) -> DesignMatrix:
    """
    Select and build the most statistically powerful and cost-effective design matrix.

    Args:
        factors: Number of factors (int) OR list of Factor objects / names / FactorSet.
        max_runs: Optional hard upper limit on the number of prompt variations / runs.
        min_resolution: Minimum desired resolution (3 = screening, 4 = unaliased main effects, 5 = unaliased interactions).
        prefer_plackett_burman: If True and budget is tight, prefers PB screening designs.

    Returns:
        DesignMatrix ready for execution.
    """
    # Parse factor count and names
    factor_names: List[str] = []
    factor_ids: List[str] = []
    alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

    if isinstance(factors, int):
        num_factors = factors
        factor_ids = [
            alphabet[i] if i < len(alphabet) else f"X{i+1}" for i in range(num_factors)
        ]
        factor_names = [f"Factor_{fid}" for fid in factor_ids]
    elif isinstance(factors, FactorSet):
        num_factors = len(factors)
        factor_ids = factors.ids
        factor_names = factors.names
    else:
        num_factors = len(factors)
        for i, item in enumerate(factors):
            if isinstance(item, Factor):
                factor_ids.append(
                    item.id or (alphabet[i] if i < len(alphabet) else f"X{i+1}")
                )
                factor_names.append(item.name)
            else:
                factor_ids.append(
                    str(item)
                    if len(str(item)) <= 2
                    else (alphabet[i] if i < len(alphabet) else f"X{i+1}")
                )
                factor_names.append(str(item))

    # If PB screening is specifically requested
    if prefer_plackett_burman and (min_resolution is None or min_resolution <= 3):
        return PlackettBurmanGenerator.create(
            num_factors=num_factors,
            factor_ids=factor_ids,
            factor_names=factor_names,
            runs=max_runs,
        )

    # If small number of factors (<= 3) and budget allows, full factorial is great!
    if num_factors <= 3 and (max_runs is None or max_runs >= 2**num_factors):
        return FullFactorialGenerator.create(
            factor_ids=factor_ids,
            factor_names=factor_names,
        )

    # Check catalog plans for this number of factors
    matching_plans = list_available_plans(num_factors=num_factors)

    # Filter by constraints
    candidates = []
    for plan in matching_plans:
        runs = plan["runs"]
        res = plan["resolution"]
        if max_runs is not None and runs > max_runs:
            continue
        if min_resolution is not None and res < min_resolution:
            continue
        candidates.append(plan)

    # If we found catalog candidates, sort by highest resolution, then lowest runs
    if candidates:
        candidates.sort(key=lambda p: (-p["resolution"], p["runs"]))
        best_plan = candidates[0]
        return FractionalFactorialGenerator.from_plan_id(
            plan_id=best_plan["plan_id"],
            factor_names=factor_names,
        )

    # If no catalog plan fits budget, fallback to PB if resolution III acceptable
    if min_resolution is None or min_resolution <= 3:
        try:
            return PlackettBurmanGenerator.create(
                num_factors=num_factors,
                factor_ids=factor_ids,
                factor_names=factor_names,
                runs=max_runs,
            )
        except Exception:
            pass

    # Fallback to smallest available fractional or Full Factorial
    if matching_plans:
        matching_plans.sort(key=lambda p: p["runs"])
        return FractionalFactorialGenerator.from_plan_id(
            plan_id=matching_plans[0]["plan_id"],
            factor_names=factor_names,
        )

    return FullFactorialGenerator.create(
        factor_ids=factor_ids,
        factor_names=factor_names,
    )
