"""
ANOVA (Analysis of Variance) and OLS Regression Engine for Prompt Optimization.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from scipy import stats
from .effects import EffectAnalyzer, FactorEffect, InteractionEffect


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
    target_metric: str
    formula: str
    anova_table: List[ANOVARow] = Field(default_factory=list)
    main_effects: List[FactorEffect] = Field(default_factory=list)
    interactions: List[InteractionEffect] = Field(default_factory=list)
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
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ANOVA results table to pandas DataFrame."""
        records = []
        for r in self.anova_table:
            records.append({
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
            })
        return pd.DataFrame(records)


class ANOVAEngine:
    """
    Orchestrates ANOVA analysis and model fitting for prompt factorial experiments.
    """

    @classmethod
    def run_anova(
        cls,
        data: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
        factor_name_map: Optional[Dict[str, str]] = None,
        include_interactions: bool = False,
        max_interaction_order: int = 2,
        ss_type: int = 2,
        alpha: float = 0.05,
    ) -> ANOVAResult:
        """
        Run ANOVA on experimental data.
        
        Args:
            data: DataFrame containing factor columns (0/1) and the target metric column.
            factor_cols: Names/IDs of the factor columns.
            target_col: Column name of the metric to analyze (e.g. 'f1_score', 'exact_match').
            factor_name_map: Mapping of factor ID to descriptive name.
            include_interactions: If True, tests 2-way interaction terms.
            max_interaction_order: 2 for pairwise interactions.
            ss_type: ANOVA Sum of Squares type (1, 2, or 3).
            alpha: Significance threshold (default 0.05).
        """
        factor_name_map = factor_name_map or {}
        clean_df = data[[c for c in factor_cols if c in data.columns] + [target_col]].dropna()

        # Compute direct main effects and interactions
        main_effects = EffectAnalyzer.compute_main_effects(
            df=clean_df,
            factor_cols=factor_cols,
            target_col=target_col,
            factor_name_map=factor_name_map,
            alpha=alpha,
        )

        interactions = EffectAnalyzer.compute_interaction_effects(
            df=clean_df,
            factor_cols=factor_cols,
            target_col=target_col,
            factor_name_map=factor_name_map,
            alpha=alpha,
        )

        valid_factors = [f for f in factor_cols if f in clean_df.columns and clean_df[f].nunique() > 1]
        if not valid_factors:
            raise ValueError(f"No factors with multiple levels found in dataset columns: {factor_cols}")

        # Map column names to safe identifiers (e.g. 'f_0_A', 'f_1_B') to avoid Patsy keyword collisions (such as column 'C')
        col_to_safe = {f: f"__f_{i}_{f}__" for i, f in enumerate(valid_factors)}
        safe_to_col = {v: k for k, v in col_to_safe.items()}

        ols_df = clean_df.copy()
        ols_df.rename(columns=col_to_safe, inplace=True)
        # Also ensure target col name is safe
        safe_target = "__target_metric__"
        ols_df.rename(columns={target_col: safe_target}, inplace=True)

        terms = [f"C({col_to_safe[f]})" for f in valid_factors]
        if include_interactions and len(valid_factors) >= 2:
            for f1, f2 in itertools.combinations(valid_factors, 2):
                terms.append(f"C({col_to_safe[f1]}):C({col_to_safe[f2]})")

        formula = f"{safe_target} ~ " + " + ".join(terms)
        human_formula = f"{target_col} ~ " + " + ".join([f"C({f})" for f in valid_factors])

        # Fit model
        if HAS_STATSMODELS:
            try:
                ols_model = smf.ols(formula, data=ols_df).fit()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anova_df = sm.stats.anova_lm(ols_model, typ=ss_type)
            except Exception:
                # If rank-deficient or singular, fallback to main effects only
                fallback_formula = f"{safe_target} ~ " + " + ".join([f"C({col_to_safe[f]})" for f in valid_factors])
                ols_model = smf.ols(fallback_formula, data=ols_df).fit()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anova_df = sm.stats.anova_lm(ols_model, typ=ss_type)
                formula = fallback_formula

            r_sq = float(ols_model.rsquared)
            r_sq_adj = float(ols_model.rsquared_adj)
            f_stat = float(ols_model.fvalue) if hasattr(ols_model, "fvalue") and not np.isnan(ols_model.fvalue) else 0.0
            model_pval = float(ols_model.f_pvalue) if hasattr(ols_model, "f_pvalue") and not np.isnan(ols_model.f_pvalue) else 1.0
            res_df = float(ols_model.df_resid)
            res_se = float(np.sqrt(ols_model.mse_resid)) if hasattr(ols_model, "mse_resid") else 0.0
        else:
            anova_df, r_sq, r_sq_adj, f_stat, model_pval, res_df, res_se = cls._manual_anova(
                clean_df, valid_factors, target_col
            )

        # Process ANOVA Table
        res_ss = anova_df.loc["Residual", "sum_sq"] if "Residual" in anova_df.index else 1e-9
        total_ss = anova_df["sum_sq"].sum()
        ms_resid = res_ss / res_df if res_df > 0 else 1e-9

        p_vals_list = []
        row_objects = []

        for idx, row in anova_df.iterrows():
            source_raw = str(idx)
            # Revert safe names back to original factor IDs
            for safe_name, orig_name in safe_to_col.items():
                source_raw = source_raw.replace(safe_name, orig_name)

            source_clean = source_raw.replace("C(", "").replace(")", "").strip()
            ss = float(row["sum_sq"])
            df_val = float(row["df"])
            ms = ss / df_val if df_val > 0 else 0.0
            f_val = float(row["F"]) if "F" in row and not np.isnan(row["F"]) else None
            p_val = float(row["PR(>F)"]) if "PR(>F)" in row and not np.isnan(row["PR(>F)"]) else None

            # Effect sizes
            partial_eta = (ss / (ss + res_ss)) if (ss + res_ss) > 0 and source_clean != "Residual" else None
            omega_sq = (
                (ss - df_val * ms_resid) / (total_ss + ms_resid)
                if source_clean != "Residual" and (total_ss + ms_resid) > 0
                else None
            )
            if omega_sq is not None and omega_sq < 0:
                omega_sq = 0.0

            if p_val is not None and source_clean != "Residual":
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

        # Multiple comparisons p-value corrections (Bonferroni & Benjamini-Hochberg)
        m_tests = len(p_vals_list)
        if m_tests > 0:
            for r_idx, p in p_vals_list:
                row_objects[r_idx].p_value_bonferroni = min(1.0, p * m_tests)

            sorted_p = sorted(p_vals_list, key=lambda x: x[1])
            for rank, (r_idx, p) in enumerate(sorted_p, start=1):
                fdr_q = min(1.0, (p * m_tests) / rank)
                row_objects[r_idx].p_value_fdr = fdr_q

        # Classify factor recommendations
        sig_pos = []
        sig_neg = []
        neutral = []

        effect_dict = {e.factor_id: e for e in main_effects}
        for f in valid_factors:
            eff = effect_dict.get(f)
            if eff and eff.is_significant:
                if eff.effect_delta > 0:
                    sig_pos.append(f)
                else:
                    sig_neg.append(f)
            else:
                neutral.append(f)

        return ANOVAResult(
            target_metric=target_col,
            formula=human_formula,
            anova_table=row_objects,
            main_effects=main_effects,
            interactions=interactions,
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
            metadata={"num_observations": len(clean_df), "ss_type": ss_type},
        )

    @classmethod
    def _manual_anova(
        cls, df: pd.DataFrame, factor_cols: Sequence[str], target_col: str
    ) -> Tuple[pd.DataFrame, float, float, float, float, float, float]:
        """Matrix-based manual OLS and ANOVA when statsmodels is not available."""
        y = df[target_col].values
        n = len(y)
        X_cols = [np.ones(n)]
        col_names = ["Intercept"]

        for f in factor_cols:
            X_cols.append(df[f].values)
            col_names.append(f)

        X = np.column_stack(X_cols)
        p = X.shape[1]

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        res = y - y_pred

        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(res ** 2)
        ss_reg = ss_tot - ss_res
        df_res = n - p
        df_reg = p - 1

        r_sq = ss_reg / ss_tot if ss_tot > 0 else 0.0
        r_sq_adj = 1.0 - (1.0 - r_sq) * (n - 1) / df_res if df_res > 0 else 0.0

        ms_res = ss_res / df_res if df_res > 0 else 1e-9
        ms_reg = ss_reg / df_reg if df_reg > 0 else 0.0
        f_stat = ms_reg / ms_res if ms_res > 0 else 0.0
        model_pval = float(1.0 - stats.f.cdf(f_stat, df_reg, df_res)) if df_res > 0 else 1.0

        records = {}
        for i, f in enumerate(factor_cols, start=1):
            ss_f = ss_reg / len(factor_cols)
            df_f = 1
            ms_f = ss_f
            f_f = ms_f / ms_res if ms_res > 0 else 0.0
            p_f = float(1.0 - stats.f.cdf(f_f, df_f, df_res)) if df_res > 0 else 1.0
            records[f"C({f})"] = {"sum_sq": ss_f, "df": df_f, "F": f_f, "PR(>F)": p_f}

        records["Residual"] = {"sum_sq": ss_res, "df": df_res, "F": np.nan, "PR(>F)": np.nan}
        return pd.DataFrame(records).T, r_sq, r_sq_adj, f_stat, model_pval, df_res, np.sqrt(ms_res)
