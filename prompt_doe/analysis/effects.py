"""
Main Effects and Interaction Effects calculation for Design of Experiments.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FactorEffect(BaseModel):
    """Statistical summary of a single factor's effect on a target metric."""
    factor_id: str
    factor_name: str = ""
    mean_level_0: float = 0.0
    mean_level_1: float = 0.0
    effect_delta: float = 0.0        # mean_level_1 - mean_level_0
    relative_change_pct: float = 0.0 # (delta / mean_level_0) * 100
    std_error: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    cohen_d: float = 0.0
    sample_count_0: int = 0
    sample_count_1: int = 0
    is_significant: bool = False
    recommendation: str = "keep_default"  # 'enable', 'disable', 'neutral'


class InteractionEffect(BaseModel):
    """Statistical summary of a 2-factor interaction effect."""
    factor_pair: Tuple[str, str]
    factor_names: Tuple[str, str] = ("", "")
    interaction_term: str = ""
    effect_delta: float = 0.0
    mean_00: float = 0.0
    mean_01: float = 0.0
    mean_10: float = 0.0
    mean_11: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False


class EffectAnalyzer:
    """
    Computes Main Effects and 2-Factor Interactions from experiment results.
    """

    @classmethod
    def compute_main_effects(
        cls,
        df: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
        factor_name_map: Optional[Dict[str, str]] = None,
        alpha: float = 0.05,
    ) -> List[FactorEffect]:
        """Compute main effect estimates for all factors on the target metric."""
        effects: List[FactorEffect] = []
        factor_name_map = factor_name_map or {}

        for fid in factor_cols:
            if fid not in df.columns:
                continue

            sub_0 = df[df[fid] == 0][target_col].dropna()
            sub_1 = df[df[fid] == 1][target_col].dropna()

            n0 = len(sub_0)
            n1 = len(sub_1)
            if n0 == 0 or n1 == 0:
                continue

            m0 = float(sub_0.mean())
            m1 = float(sub_1.mean())
            v0 = float(sub_0.var(ddof=1)) if n0 > 1 else 0.0
            v1 = float(sub_1.var(ddof=1)) if n1 > 1 else 0.0

            delta = m1 - m0
            rel_pct = (delta / m0 * 100.0) if m0 != 0 else 0.0

            # Pooled variance
            df_pooled = (n0 - 1) + (n1 - 1)
            if df_pooled > 0:
                s_pooled = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / df_pooled)
                se = float(s_pooled * np.sqrt(1.0 / n0 + 1.0 / n1))
                t_stat = float(delta / se) if se > 0 else 0.0
                cohen_d = float(delta / s_pooled) if s_pooled > 0 else 0.0
                
                # Two-tailed p-value
                from scipy import stats
                p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_pooled)))
            else:
                se = 0.0
                t_stat = 0.0
                cohen_d = 0.0
                p_val = 1.0

            is_sig = bool(p_val < alpha)

            if is_sig and delta > 0:
                rec = "enable"
            elif is_sig and delta < 0:
                rec = "disable"
            else:
                rec = "neutral"

            fname = factor_name_map.get(fid, fid)
            effects.append(
                FactorEffect(
                    factor_id=fid,
                    factor_name=fname,
                    mean_level_0=m0,
                    mean_level_1=m1,
                    effect_delta=delta,
                    relative_change_pct=rel_pct,
                    std_error=se,
                    t_statistic=t_stat,
                    p_value=p_val,
                    cohen_d=cohen_d,
                    sample_count_0=n0,
                    sample_count_1=n1,
                    is_significant=is_sig,
                    recommendation=rec,
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
        alpha: float = 0.05,
    ) -> List[InteractionEffect]:
        """Compute all pairwise 2-factor interaction effects."""
        interactions: List[InteractionEffect] = []
        factor_name_map = factor_name_map or {}

        for f1, f2 in itertools.combinations(factor_cols, 2):
            if f1 not in df.columns or f2 not in df.columns:
                continue

            sub = df[[f1, f2, target_col]].dropna()
            g = sub.groupby([f1, f2])[target_col].mean()

            m00 = float(g.get((0, 0), 0.0))
            m01 = float(g.get((0, 1), 0.0))
            m10 = float(g.get((1, 0), 0.0))
            m11 = float(g.get((1, 1), 0.0))

            # Interaction delta = 0.5 * ((m11 - m10) - (m01 - m00))
            delta = 0.5 * ((m11 - m10) - (m01 - m00))
            term = f"{f1}:{f2}"

            fname1 = factor_name_map.get(f1, f1)
            fname2 = factor_name_map.get(f2, f2)

            interactions.append(
                InteractionEffect(
                    factor_pair=(f1, f2),
                    factor_names=(fname1, fname2),
                    interaction_term=term,
                    effect_delta=delta,
                    mean_00=m00,
                    mean_01=m01,
                    mean_10=m10,
                    mean_11=m11,
                    p_value=1.0,
                    is_significant=False,
                )
            )

        interactions.sort(key=lambda x: abs(x.effect_delta), reverse=True)
        return interactions
