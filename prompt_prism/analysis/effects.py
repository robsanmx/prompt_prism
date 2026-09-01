"""
Factor Main Effects and 2-Factor Interaction Effect Calculations.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from ..design.aliasing import AliasStructure


class FactorEffect(BaseModel):
    """Estimated main effect for a single prompt factor."""

    factor_id: str
    factor_name: str
    mean_level_0: float
    mean_level_1: float
    effect_delta: float  # Level 1 Mean - Level 0 Mean
    relative_change_pct: float  # (effect_delta / baseline) * 100
    std_error: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0
    is_significant: bool = False
    aliased_with: List[str] = Field(default_factory=list)
    confounding_warning: str = ""
    action_recommendation: str = "KEEP_DEFAULT"  # 'ENABLE', 'DISABLE', 'OMIT_NEUTRAL'


class InteractionEffect(BaseModel):
    """Estimated 2-way interaction between two factors."""

    factor_1: str
    factor_2: str
    factor_1_name: str = ""
    factor_2_name: str = ""
    effect_delta: float
    is_synergistic: bool
    is_antagonistic: bool
    mean_00: float = 0.0
    mean_01: float = 0.0
    mean_10: float = 0.0
    mean_11: float = 0.0
    p_value: Optional[float] = None
    is_significant: bool = False
    aliased_with: List[str] = Field(default_factory=list)


class EffectAnalyzer:
    """Computes main effects, interaction effects, and effect sizes for DoE results."""

    @classmethod
    def compute_main_effects(
        cls,
        df: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
        factor_name_map: Optional[Dict[str, str]] = None,
        ols_model: Optional[Any] = None,
        col_to_safe: Optional[Dict[str, str]] = None,
        alias_structure: Optional[AliasStructure] = None,
        alpha: float = 0.05,
    ) -> List[FactorEffect]:
        """
        Compute main effect estimates for all factors on the target metric.
        If ols_model is provided, uses unified model coefficients (RCBD / OLS) for standard error,
        t-statistic, and p-value.
        """
        factor_name_map = factor_name_map or {}
        col_to_safe = col_to_safe or {}
        effects: List[FactorEffect] = []

        overall_std = df[target_col].std()
        if np.isnan(overall_std) or overall_std == 0:
            overall_std = 1e-9

        # Extract OLS params/bse/tvalues/pvalues if model is available
        ols_params = getattr(ols_model, "params", {}) if ols_model else {}
        ols_bse = getattr(ols_model, "bse", {}) if ols_model else {}
        ols_tvalues = getattr(ols_model, "tvalues", {}) if ols_model else {}
        ols_pvalues = getattr(ols_model, "pvalues", {}) if ols_model else {}

        for fid in factor_cols:
            if fid not in df.columns:
                continue

            # Descriptive means across level 0 and level 1
            vals_0 = df[df[fid] == 0][target_col].dropna()
            vals_1 = df[df[fid] == 1][target_col].dropna()

            m0 = float(vals_0.mean()) if len(vals_0) > 0 else 0.0
            m1 = float(vals_1.mean()) if len(vals_1) > 0 else 0.0

            safe_name = col_to_safe.get(fid, fid)
            # Look for categorical or numeric term in OLS model
            ols_term_key = None
            for key in ols_params.keys():
                if safe_name in key:
                    ols_term_key = key
                    break

            if ols_model and ols_term_key:
                delta = float(ols_params[ols_term_key])
                se = float(ols_bse.get(ols_term_key, 0.0))
                t_stat = float(ols_tvalues.get(ols_term_key, 0.0))
                pval = float(ols_pvalues.get(ols_term_key, 1.0))
            else:
                # Fallback to sample means and two-sample t-test
                delta = m1 - m0
                if len(vals_0) > 1 and len(vals_1) > 1:
                    t_stat, pval = stats.ttest_ind(vals_1, vals_0, equal_var=True)
                    t_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
                    pval = float(pval) if not np.isnan(pval) else 1.0
                    sp = np.sqrt(
                        (
                            (len(vals_1) - 1) * vals_1.var()
                            + (len(vals_0) - 1) * vals_0.var()
                        )
                        / (len(vals_1) + len(vals_0) - 2)
                    )
                    se = (
                        float(sp * np.sqrt(1 / len(vals_1) + 1 / len(vals_0)))
                        if sp > 0
                        else 0.0
                    )
                else:
                    se = 0.0
                    t_stat = 0.0
                    pval = 1.0

            rel_change = (delta / (m0 if abs(m0) > 1e-9 else 1.0)) * 100
            cohen_d = float(delta / overall_std) if overall_std > 0 else 0.0
            is_sig = bool(pval < alpha)

            # Check Aliasing
            aliases = (
                alias_structure.get_aliases_for_term(fid, max_order=3)
                if alias_structure
                else []
            )
            conf_warning = ""
            if alias_structure and alias_structure.resolution == 3 and aliases:
                conf_warning = (
                    f"Confounded with 2-factor interactions: {', '.join(aliases)}"
                )
            elif alias_structure and alias_structure.resolution == 4 and aliases:
                conf_warning = (
                    f"Confounded with 3-factor interactions: {', '.join(aliases)}"
                )

            # Action recommendation
            if is_sig:
                action = "ENABLE" if delta > 0 else "DISABLE"
            else:
                action = "OMIT_NEUTRAL"

            fname = factor_name_map.get(fid, fid)
            effects.append(
                FactorEffect(
                    factor_id=fid,
                    factor_name=fname,
                    mean_level_0=m0,
                    mean_level_1=m1,
                    effect_delta=delta,
                    relative_change_pct=rel_change,
                    std_error=se,
                    t_statistic=t_stat,
                    p_value=pval,
                    cohens_d=cohen_d,
                    is_significant=is_sig,
                    aliased_with=aliases,
                    confounding_warning=conf_warning,
                    action_recommendation=action,
                )
            )

        # Sort by absolute effect size descending
        effects.sort(key=lambda e: abs(e.effect_delta), reverse=True)
        return effects

    @classmethod
    def compute_interaction_effects(
        cls,
        df: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
        factor_name_map: Optional[Dict[str, str]] = None,
        ols_model: Optional[Any] = None,
        col_to_safe: Optional[Dict[str, str]] = None,
        alias_structure: Optional[AliasStructure] = None,
        alpha: float = 0.05,
    ) -> List[InteractionEffect]:
        """
        Compute 2-factor interaction effects. If cells are missing, sets effect_delta = NaN safely.
        """
        factor_name_map = factor_name_map or {}
        col_to_safe = col_to_safe or {}
        interactions: List[InteractionEffect] = []

        ols_params = getattr(ols_model, "params", {}) if ols_model else {}
        ols_pvalues = getattr(ols_model, "pvalues", {}) if ols_model else {}

        for f1, f2 in itertools.combinations(factor_cols, 2):
            if f1 not in df.columns or f2 not in df.columns:
                continue

            # Group by combinations of (f1, f2)
            grouped = df.groupby([f1, f2])[target_col].mean().to_dict()

            # Check if any cell is missing
            if (
                (0, 0) not in grouped
                or (0, 1) not in grouped
                or (1, 0) not in grouped
                or (1, 1) not in grouped
            ):
                # Cell is missing -> return NaN
                interactions.append(
                    InteractionEffect(
                        factor_1=f1,
                        factor_2=f2,
                        factor_1_name=factor_name_map.get(f1, f1),
                        factor_2_name=factor_name_map.get(f2, f2),
                        effect_delta=float(np.nan),
                        is_synergistic=False,
                        is_antagonistic=False,
                        mean_00=float(grouped.get((0, 0), np.nan)),
                        mean_01=float(grouped.get((0, 1), np.nan)),
                        mean_10=float(grouped.get((1, 0), np.nan)),
                        mean_11=float(grouped.get((1, 1), np.nan)),
                        p_value=None,
                        is_significant=False,
                    )
                )
                continue

            m00 = float(grouped[(0, 0)])
            m01 = float(grouped[(0, 1)])
            m10 = float(grouped[(1, 0)])
            m11 = float(grouped[(1, 1)])

            # Interaction effect = (m11 - m10 - m01 + m00) / 2
            interaction_delta = 0.5 * (m11 - m10 - m01 + m00)

            # Check OLS model for interaction p-value
            safe_1 = col_to_safe.get(f1, f1)
            safe_2 = col_to_safe.get(f2, f2)
            pval = None
            for key, p in ols_pvalues.items():
                if safe_1 in key and safe_2 in key:
                    pval = float(p)
                    break

            is_sig = bool(pval is not None and pval < alpha)
            aliases = (
                alias_structure.get_aliases_for_term(f1 + f2, max_order=2)
                if alias_structure
                else []
            )

            interactions.append(
                InteractionEffect(
                    factor_1=f1,
                    factor_2=f2,
                    factor_1_name=factor_name_map.get(f1, f1),
                    factor_2_name=factor_name_map.get(f2, f2),
                    effect_delta=interaction_delta,
                    is_synergistic=bool(interaction_delta > 0.01),
                    is_antagonistic=bool(interaction_delta < -0.01),
                    mean_00=m00,
                    mean_01=m01,
                    mean_10=m10,
                    mean_11=m11,
                    p_value=pval,
                    is_significant=is_sig,
                    aliased_with=aliases,
                )
            )

        interactions.sort(
            key=lambda e: (
                not np.isnan(e.effect_delta),
                abs(e.effect_delta) if not np.isnan(e.effect_delta) else -1,
            ),
            reverse=True,
        )
        return interactions
