"""
ANOVA (Analysis of Variance) and OLS Regression Engine for Prompt Optimization.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pydantic import BaseModel, ConfigDict, Field
from statsmodels.stats.multitest import multipletests

from ..design.aliasing import AliasStructure
from .effects import EffectAnalyzer, FactorEffect, InteractionEffect


class ConfoundedModelError(ValueError):
    """Raised when interaction terms are perfectly collinear or aliased with main effects."""

    pass


class ANOVARow(BaseModel):
    """Single row in the ANOVA table."""

    source: str
    factor_name: str = ""
    sum_sq: float
    df: float
    mean_sq: float
    f_statistic: Optional[float] = None
    p_value: Optional[float] = None
    p_value_bonferroni: Optional[float] = None
    p_value_fdr: Optional[float] = None
    partial_eta_sq: Optional[float] = None
    omega_sq: Optional[float] = None
    is_significant: bool = False


class ANOVAResult(BaseModel):
    """
    Complete ANOVA output containing the ANOVA table, effect sizes, model diagnostics, and factor rankings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_metric: str
    formula: str
    anova_table: List[ANOVARow] = Field(default_factory=list)
    main_effects: List[FactorEffect] = Field(default_factory=list)
    interactions: List[InteractionEffect] = Field(default_factory=list)
    omitted_interactions: List[str] = Field(default_factory=list)
    r_squared: float = 0.0
    r_squared_adj: float = 0.0
    f_statistic: float = 0.0
    model_p_value: float = 1.0
    residual_df: float = 0.0
    residual_std_error: float = 0.0
    significant_positive_factors: List[str] = Field(default_factory=list)
    significant_negative_factors: List[str] = Field(default_factory=list)
    neutral_factors: List[str] = Field(default_factory=list)
    alpha: float = 0.05
    block_col: Optional[str] = None
    alias_structure: Optional[AliasStructure] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ANOVA results table to pandas DataFrame."""
        records = []
        for r in self.anova_table:
            records.append(
                {
                    "Source": r.source,
                    "Factor Name": r.factor_name,
                    "Sum of Sq": r.sum_sq,
                    "DF": r.df,
                    "Mean Sq": r.mean_sq,
                    "F": r.f_statistic,
                    "PR(>F)": r.p_value,
                    "p (Bonferroni)": r.p_value_bonferroni,
                    "p (FDR)": r.p_value_fdr,
                    "Partial Eta^2": r.partial_eta_sq,
                    "Omega^2": r.omega_sq,
                    "Significant": r.is_significant,
                }
            )
        return pd.DataFrame(records)


class ANOVAEngine:
    """
    Orchestrates ANOVA analysis and model fitting for prompt factorial experiments.
    """

    @classmethod
    def _adjust_p_values(
        cls, p_values: Sequence[float], method: str = "fdr_bh"
    ) -> List[float]:
        """Adjust p-values using statsmodels multipletests (e.g. Benjamini-Hochberg FDR)."""
        if not p_values:
            return []
        _, adj_p, _, _ = multipletests(p_values, method=method)
        return [float(p) for p in adj_p]

    @classmethod
    def run_anova(
        cls,
        data: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
        block_col: Optional[str] = None,
        factor_name_map: Optional[Dict[str, str]] = None,
        alias_structure: Optional[AliasStructure] = None,
        include_interactions: bool = False,
        max_interaction_order: int = 2,
        ss_type: int = 2,
        alpha: float = 0.05,
        maximize: bool = True,
    ) -> ANOVAResult:
        """
        Run ANOVA on experimental data using Randomized Complete Block Design (RCBD) or standard Factorial OLS.

        Args:
            data: DataFrame containing factor columns (0/1), target metric, and optional block column.
            factor_cols: Names/IDs of the factor columns.
            target_col: Column name of the metric to analyze.
            block_col: Optional column name to model as a nuisance block (e.g. 'sample_id' / item difficulty).
            factor_name_map: Mapping of factor ID to descriptive name.
            alias_structure: Optional AliasStructure for resolution and confounding annotation.
            include_interactions: If True, fits 2-way interaction terms.
            max_interaction_order: 2 for pairwise interactions.
            ss_type: ANOVA Sum of Squares type (1, 2, or 3).
            alpha: Significance threshold (default 0.05).
            maximize: Whether higher score is better for the target metric (default True).
        """
        factor_name_map = factor_name_map or {}
        included_cols = [c for c in factor_cols if c in data.columns] + [target_col]
        has_blocking = bool(
            block_col and block_col in data.columns and data[block_col].nunique() > 1
        )
        if has_blocking and block_col not in included_cols:
            included_cols.append(block_col)

        clean_df = data[included_cols].dropna()

        valid_factors = [
            f
            for f in factor_cols
            if f in clean_df.columns and clean_df[f].nunique() > 1
        ]
        if not valid_factors:
            raise ValueError(
                f"No factors with multiple levels found in dataset columns: {factor_cols}"
            )

        # Map column names to safe identifiers to avoid Patsy keyword collisions (such as column 'C')
        col_to_safe = {f: f"__f_{i}_{f}__" for i, f in enumerate(valid_factors)}
        safe_to_col = {v: k for k, v in col_to_safe.items()}

        ols_df = clean_df.copy()
        ols_df.rename(columns=col_to_safe, inplace=True)
        safe_target = "__target_metric__"
        ols_df.rename(columns={target_col: safe_target}, inplace=True)

        terms: List[str] = []
        human_terms: List[str] = []

        # If blocking on sample_id, prepend block term
        if has_blocking and block_col:
            safe_block = "__block_var__"
            ols_df.rename(columns={block_col: safe_block}, inplace=True)
            terms.append(f"C({safe_block})")
            human_terms.append(f"C({block_col})")

        # Main effect terms
        for f in valid_factors:
            terms.append(f"C({col_to_safe[f]})")
            human_terms.append(f"C({f})")

        # 2-way interaction terms if requested
        if include_interactions and len(valid_factors) >= 2:
            for f1, f2 in itertools.combinations(valid_factors, 2):
                terms.append(f"C({col_to_safe[f1]}):C({col_to_safe[f2]})")
                human_terms.append(f"C({f1}):C({f2})")

        formula = f"{safe_target} ~ " + " + ".join(terms)
        human_formula = f"{target_col} ~ " + " + ".join(human_terms)

        # Fit OLS Model with statsmodels
        try:
            ols_model = smf.ols(formula, data=ols_df).fit()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                anova_df = sm.stats.anova_lm(ols_model, typ=ss_type)
        except Exception as e:
            # If rank deficiency occurred due to interactions in Res III/IV, fallback to main effects
            if include_interactions:
                fallback_terms = [f"C({col_to_safe[f]})" for f in valid_factors]
                fallback_human = [f"C({f})" for f in valid_factors]
                if has_blocking:
                    fallback_terms.insert(0, f"C({safe_block})")
                    fallback_human.insert(0, f"C({block_col})")

                fallback_formula = f"{safe_target} ~ " + " + ".join(fallback_terms)
                ols_model = smf.ols(fallback_formula, data=ols_df).fit()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anova_df = sm.stats.anova_lm(ols_model, typ=ss_type)
                formula = fallback_formula
                human_formula = f"{target_col} ~ " + " + ".join(fallback_human)
            else:
                raise e

        r_sq = float(ols_model.rsquared)
        r_sq_adj = float(ols_model.rsquared_adj)
        f_stat = (
            float(ols_model.fvalue)
            if hasattr(ols_model, "fvalue") and not np.isnan(ols_model.fvalue)
            else 0.0
        )
        model_pval = (
            float(ols_model.f_pvalue)
            if hasattr(ols_model, "f_pvalue") and not np.isnan(ols_model.f_pvalue)
            else 1.0
        )
        res_df = float(ols_model.df_resid)
        res_se = (
            float(np.sqrt(ols_model.mse_resid))
            if hasattr(ols_model, "mse_resid")
            else 0.0
        )

        # Compute unified main effects and interactions using fitted model coefficients
        main_effects = EffectAnalyzer.compute_main_effects(
            df=clean_df,
            factor_cols=factor_cols,
            target_col=target_col,
            factor_name_map=factor_name_map,
            ols_model=ols_model,
            col_to_safe=col_to_safe,
            alias_structure=alias_structure,
            alpha=alpha,
            maximize=maximize,
        )

        interactions = EffectAnalyzer.compute_interaction_effects(
            df=clean_df,
            factor_cols=factor_cols,
            target_col=target_col,
            factor_name_map=factor_name_map,
            ols_model=ols_model,
            col_to_safe=col_to_safe,
            alias_structure=alias_structure,
            alpha=alpha,
        )

        omitted_interactions = [
            f"{e.factor_1}:{e.factor_2}"
            for e in interactions
            if np.isnan(e.effect_delta)
        ]

        # Process ANOVA Table
        res_ss = (
            anova_df.loc["Residual", "sum_sq"] if "Residual" in anova_df.index else 1e-9
        )
        total_ss = anova_df["sum_sq"].sum()
        ms_resid = res_ss / res_df if res_df > 0 else 1e-9

        p_vals_list: List[Tuple[int, float]] = []
        row_objects: List[ANOVARow] = []

        for idx, row in anova_df.iterrows():
            source_raw = str(idx)
            # Revert safe names back to original factor IDs
            for safe_name, orig_name in safe_to_col.items():
                source_raw = source_raw.replace(safe_name, orig_name)

            if has_blocking and "__block_var__" in source_raw:
                source_clean = f"Block ({block_col})"
            else:
                source_clean = source_raw.replace("C(", "").replace(")", "").strip()

            ss = float(row["sum_sq"])
            df_val = float(row["df"])
            ms = ss / df_val if df_val > 0 else 0.0
            f_val = float(row["F"]) if "F" in row and not np.isnan(row["F"]) else None
            p_val = (
                float(row["PR(>F)"])
                if "PR(>F)" in row and not np.isnan(row["PR(>F)"])
                else None
            )

            # Effect sizes
            partial_eta = (
                (ss / (ss + res_ss))
                if (ss + res_ss) > 0 and source_clean != "Residual"
                else None
            )
            omega_sq = (
                (ss - df_val * ms_resid) / (total_ss + ms_resid)
                if source_clean != "Residual" and (total_ss + ms_resid) > 0
                else None
            )
            if omega_sq is not None and omega_sq < 0:
                omega_sq = 0.0

            # Exclude nuisance block and residual from multiple comparisons adjustment
            if (
                p_val is not None
                and source_clean != "Residual"
                and not source_clean.startswith("Block")
            ):
                p_vals_list.append((len(row_objects), p_val))

            fname = factor_name_map.get(source_clean, source_clean)
            row_objects.append(
                ANOVARow(
                    source=source_clean,
                    factor_name=fname,
                    sum_sq=ss,
                    df=df_val,
                    mean_sq=ms,
                    f_statistic=f_val,
                    p_value=p_val,
                    partial_eta_sq=partial_eta,
                    omega_sq=omega_sq,
                    is_significant=bool(p_val is not None and p_val < alpha),
                )
            )

        # Multiple comparisons p-value corrections (Bonferroni & Benjamini-Hochberg FDR)
        if p_vals_list:
            raw_p_values = [p for _, p in p_vals_list]
            fdr_p_values = cls._adjust_p_values(raw_p_values, method="fdr_bh")
            m_tests = len(raw_p_values)

            for i, (r_idx, p) in enumerate(p_vals_list):
                row_objects[r_idx].p_value_bonferroni = min(1.0, p * m_tests)
                row_objects[r_idx].p_value_fdr = fdr_p_values[i]

        # Factor classification strictly aligned with unified main effects and action recommendations
        sig_pos = [
            e.factor_id
            for e in main_effects
            if e.is_significant and e.action_recommendation == "ENABLE"
        ]
        sig_neg = [
            e.factor_id
            for e in main_effects
            if e.is_significant and e.action_recommendation == "DISABLE"
        ]
        neutral = [e.factor_id for e in main_effects if not e.is_significant]

        return ANOVAResult(
            target_metric=target_col,
            formula=human_formula,
            anova_table=row_objects,
            main_effects=main_effects,
            interactions=interactions,
            omitted_interactions=omitted_interactions,
            r_squared=r_sq,
            r_squared_adj=r_sq_adj,
            f_statistic=f_stat,
            model_p_value=model_pval,
            residual_df=res_df,
            residual_std_error=res_se,
            significant_positive_factors=sig_pos,
            significant_negative_factors=sig_neg,
            neutral_factors=neutral,
            alpha=alpha,
            block_col=block_col if has_blocking else None,
            alias_structure=alias_structure,
            metadata={
                "num_observations": len(clean_df),
                "ss_type": ss_type,
                "ols_model": ols_model,
                "target_min": float(clean_df[target_col].min()),
                "target_max": float(clean_df[target_col].max()),
                # The direction main_effects[*].action_recommendation was computed under,
                # so a consumer given a different one knows to re-derive.
                "maximize": maximize,
            },
        )
