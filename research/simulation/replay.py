"""
Method Replay and Comparator Evaluation Interface.
Fits PromptPrism and budget-matched comparator algorithms to discovery partitions
of response surfaces without confirmation leakage.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.design.generators import FractionalFactorialGenerator, PlackettBurmanGenerator
from research.analysis.estimands import MethodResult


class MethodReplay:
    """
    Executes prompt optimization methods on discovery response surfaces.
    """

    @staticmethod
    def _extract_design_runs(
        discovery_df: pd.DataFrame,
        factor_ids: List[str],
        design_run_configs: List[Dict[str, int]],
    ) -> pd.DataFrame:
        """
        Filter discovery dataframe to keep only the rows matching the specified design runs.
        """
        # Create a set of tuples representing target configs
        target_tuples = set(tuple(cfg[f] for f in factor_ids) for cfg in design_run_configs)

        # Vectorized matching
        df_tuples = list(zip(*(discovery_df[f] for f in factor_ids)))
        mask = [t in target_tuples for t in df_tuples]
        return discovery_df.loc[mask].copy()

    @classmethod
    def replay_prompt_prism(
        cls,
        discovery_df: pd.DataFrame,
        factor_ids: List[str],
        plan_id: str = "2(6-2)IV",
        alpha: float = 0.05,
        target_col: str = "metric",
        block_col: str = "sample_id",
    ) -> MethodResult:
        """
        Replay PromptPrism fractional factorial screening with RCBD ANOVA blocking.
        """
        # 1. Generate design
        if plan_id.startswith("PB-"):
            n_runs = int(plan_id.split("-")[1])
            design = PlackettBurmanGenerator.generate(num_runs=n_runs, factor_names=factor_ids)
        else:
            design = FractionalFactorialGenerator.from_plan_id(plan_id)

        design_configs = [run.factor_levels for run in design.runs]
        filtered_df = cls._extract_design_runs(discovery_df, factor_ids, design_configs)

        # Budget assertions
        distinct_configs = filtered_df[factor_ids].drop_duplicates()
        if len(distinct_configs) > len(design.runs):
            raise ValueError(
                f"Budget violation: consumed {len(distinct_configs)} configs, budget was {len(design.runs)}"
            )

        # 2. Fit RCBD ANOVA
        anova_res = ANOVAEngine.run_anova(
            data=filtered_df,
            factor_cols=factor_ids,
            target_col=target_col,
            block_col=block_col,
            alpha=alpha,
        )

        estimated_effects = {}
        p_values = {}
        conf_95 = {}
        conf_90 = {}
        selected_config = {}

        for eff in anova_res.main_effects:
            fid = eff.factor_id
            delta = float(eff.effect_delta)
            se = float(eff.std_error) if eff.std_error > 0 else 0.01
            p_val = float(eff.p_value)
            ci_low = delta - 1.96 * se
            ci_high = delta + 1.96 * se
            margin_90 = 1.645 * se

            estimated_effects[fid] = delta
            p_values[fid] = p_val
            conf_95[fid] = (ci_low, ci_high)
            conf_90[fid] = (delta - margin_90, delta + margin_90)

            # Recommendation rule: enable if statistically significant positive booster
            if p_val < alpha and delta > 0.0:
                selected_config[fid] = 1
            else:
                selected_config[fid] = 0

        # Fill any missing factors
        for f in factor_ids:
            if f not in selected_config:
                selected_config[f] = 0
                estimated_effects[f] = 0.0
                p_values[f] = 1.0
                conf_95[f] = (0.0, 0.0)
                conf_90[f] = (0.0, 0.0)

        return MethodResult(
            method_id=f"prompt_prism_{plan_id}",
            budget_configs=len(design.runs),
            budget_calls=len(filtered_df),
            selected_config=selected_config,
            estimated_main_effects=estimated_effects,
            p_values=p_values,
            conf_intervals_95=conf_95,
            conf_intervals_90=conf_90,
        )

    @staticmethod
    def _two_sample_inference(
        series1: pd.Series,
        series0: pd.Series,
    ) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
        """
        Computes difference in means (series1 - series0), Welch's p-value, 95% CI, 90% CI.
        """
        from scipy import stats

        s1 = series1.dropna().to_numpy(dtype=float)
        s0 = series0.dropna().to_numpy(dtype=float)
        n1 = len(s1)
        n0 = len(s0)
        if n1 == 0 or n0 == 0:
            return 0.0, 1.0, (0.0, 0.0), (0.0, 0.0)

        m1 = float(np.mean(s1))
        m0 = float(np.mean(s0))
        diff = m1 - m0

        if n1 < 2 or n0 < 2:
            return diff, 1.0, (diff, diff), (diff, diff)

        v1 = float(np.var(s1, ddof=1))
        v0 = float(np.var(s0, ddof=1))
        se2 = v1 / n1 + v0 / n0
        se = float(np.sqrt(se2))

        if se <= 1e-12:
            pval = 1.0 if abs(diff) <= 1e-12 else 0.0
            return diff, pval, (diff, diff), (diff, diff)

        denom = (v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1)
        df = (se2 ** 2) / denom if denom > 0 else float(n1 + n0 - 2)
        df = max(1.0, df)

        t_stat = abs(diff / se)
        pval = float(2.0 * (1.0 - stats.t.cdf(t_stat, df=df)))
        crit_95 = float(stats.t.ppf(0.975, df=df))
        crit_90 = float(stats.t.ppf(0.95, df=df))

        ci_95 = (diff - crit_95 * se, diff + crit_95 * se)
        ci_90 = (diff - crit_90 * se, diff + crit_90 * se)
        return diff, pval, ci_95, ci_90

    @classmethod
    def replay_random_search(
        cls,
        discovery_df: pd.DataFrame,
        factor_ids: List[str],
        budget_runs: int = 16,
        seed: int = 42,
        target_col: str = "metric",
    ) -> MethodResult:
        """
        Replay budget-matched uniform random search (selecting empirical best observed config).
        """
        if budget_runs <= 0:
            raise ValueError(f"Budget violation: budget_runs must be at least 1, got {budget_runs}")

        rng = np.random.default_rng(seed)
        all_unique_configs = discovery_df[factor_ids].drop_duplicates().to_dict("records")

        # Sample budget_runs configurations without replacement
        sampled_indices = rng.choice(len(all_unique_configs), size=min(budget_runs, len(all_unique_configs)), replace=False)
        sampled_configs = [all_unique_configs[i] for i in sampled_indices]

        filtered_df = cls._extract_design_runs(discovery_df, factor_ids, sampled_configs)

        # Budget assertion
        distinct_configs = filtered_df[factor_ids].drop_duplicates()
        if len(distinct_configs) > budget_runs:
            raise ValueError(f"Budget violation: consumed {len(distinct_configs)} configs, budget was {budget_runs}")

        # Compute empirical mean for each sampled config
        grouped = filtered_df.groupby(factor_ids)[target_col].mean().reset_index()
        best_row = grouped.loc[grouped[target_col].idxmax()]
        selected_config = {f: int(best_row[f]) for f in factor_ids}

        # Inferential contrasts for main effects on the random subset
        estimated_effects = {}
        p_values = {}
        conf_95 = {}
        conf_90 = {}

        for f in factor_ids:
            sub1 = filtered_df[filtered_df[f] == 1][target_col]
            sub0 = filtered_df[filtered_df[f] == 0][target_col]
            diff, pval, ci95, ci90 = cls._two_sample_inference(sub1, sub0)
            estimated_effects[f] = diff
            p_values[f] = pval
            conf_95[f] = ci95
            conf_90[f] = ci90

        return MethodResult(
            method_id="uniform_random_search",
            budget_configs=budget_runs,
            budget_calls=len(filtered_df),
            selected_config=selected_config,
            estimated_main_effects=estimated_effects,
            p_values=p_values,
            conf_intervals_95=conf_95,
            conf_intervals_90=conf_90,
        )

    @classmethod
    def replay_random_linear_model(
        cls,
        discovery_df: pd.DataFrame,
        factor_ids: List[str],
        budget_runs: int = 16,
        seed: int = 42,
        target_col: str = "metric",
    ) -> MethodResult:
        """
        Replay random design with a fitted linear model (selects model-fitted optimum).
        """
        from scipy import stats

        if budget_runs <= 0:
            raise ValueError(f"Budget violation: budget_runs must be at least 1, got {budget_runs}")

        rng = np.random.default_rng(seed)
        all_unique_configs = discovery_df[factor_ids].drop_duplicates().to_dict("records")
        sampled_indices = rng.choice(len(all_unique_configs), size=min(budget_runs, len(all_unique_configs)), replace=False)
        sampled_configs = [all_unique_configs[i] for i in sampled_indices]

        filtered_df = cls._extract_design_runs(discovery_df, factor_ids, sampled_configs)

        distinct_configs = filtered_df[factor_ids].drop_duplicates()
        if len(distinct_configs) > budget_runs:
            raise ValueError(f"Budget violation: consumed {len(distinct_configs)} configs, budget was {budget_runs}")

        formula = f"{target_col} ~ " + " + ".join(factor_ids)
        try:
            model = ols(formula, data=filtered_df).fit()
            selected_config = {}
            estimated_effects = {}
            p_values = {}
            conf_95 = {}
            conf_90 = {}

            crit_95 = float(stats.t.ppf(0.975, df=model.df_resid)) if model.df_resid > 0 else 1.96
            crit_90 = float(stats.t.ppf(0.95, df=model.df_resid)) if model.df_resid > 0 else 1.645

            for f in factor_ids:
                if f in model.params and not np.isnan(model.params[f]):
                    coeff = float(model.params[f])
                    pval = float(model.pvalues[f]) if not np.isnan(model.pvalues[f]) else 1.0
                    bse = float(model.bse[f]) if not np.isnan(model.bse[f]) else 0.0
                    estimated_effects[f] = coeff
                    p_values[f] = pval
                    conf_95[f] = (coeff - crit_95 * bse, coeff + crit_95 * bse)
                    conf_90[f] = (coeff - crit_90 * bse, coeff + crit_90 * bse)
                    selected_config[f] = 1 if coeff > 0.0 else 0
                else:
                    estimated_effects[f] = 0.0
                    p_values[f] = 1.0
                    conf_95[f] = (0.0, 0.0)
                    conf_90[f] = (0.0, 0.0)
                    selected_config[f] = 0
        except Exception:
            # Fallback if rank deficient
            return cls.replay_random_search(discovery_df, factor_ids, budget_runs, seed, target_col)

        return MethodResult(
            method_id="random_design_linear_model",
            budget_configs=budget_runs,
            budget_calls=len(filtered_df),
            selected_config=selected_config,
            estimated_main_effects=estimated_effects,
            p_values=p_values,
            conf_intervals_95=conf_95,
            conf_intervals_90=conf_90,
        )

    @classmethod
    def replay_ofat(
        cls,
        discovery_df: pd.DataFrame,
        factor_ids: List[str],
        target_col: str = "metric",
        block_col: Optional[str] = "sample_id",
    ) -> MethodResult:
        """
        Replay One-Factor-At-A-Time (OFAT) ablation: baseline (all 0) + 1 toggle per factor.
        """
        from scipy import stats

        k = len(factor_ids)
        configs = []
        # Baseline
        base_cfg = {f: 0 for f in factor_ids}
        configs.append(base_cfg)

        # Toggles
        for f in factor_ids:
            toggle_cfg = dict(base_cfg)
            toggle_cfg[f] = 1
            configs.append(toggle_cfg)

        filtered_df = cls._extract_design_runs(discovery_df, factor_ids, configs)

        # Strict budget assertion: OFAT uses exactly 1 + k configurations
        distinct_configs = filtered_df[factor_ids].drop_duplicates()
        if len(distinct_configs) > len(configs):
            raise ValueError(
                f"Budget violation: consumed {len(distinct_configs)} configs, budget was {len(configs)}"
            )

        # Baseline subset
        base_mask = pd.Series(True, index=filtered_df.index)
        for f in factor_ids:
            base_mask &= (filtered_df[f] == 0)
        base_df = filtered_df[base_mask].copy()
        base_mean = float(base_df[target_col].mean()) if len(base_df) > 0 else 0.0

        selected_config = {}
        estimated_effects = {}
        p_values = {}
        conf_95 = {}
        conf_90 = {}

        for f in factor_ids:
            t_mask = pd.Series(True, index=filtered_df.index)
            for other in factor_ids:
                t_mask &= (filtered_df[other] == (1 if other == f else 0))
            t_df = filtered_df[t_mask].copy()

            if block_col and block_col in filtered_df.columns and len(base_df) > 0 and len(t_df) > 0:
                # Matched paired contrast across items
                merged = pd.merge(
                    t_df[[block_col, target_col]],
                    base_df[[block_col, target_col]],
                    on=block_col,
                    suffixes=("_toggle", "_base"),
                )
                if len(merged) >= 2:
                    diffs = (merged[f"{target_col}_toggle"] - merged[f"{target_col}_base"]).to_numpy(dtype=float)
                    diff = float(np.mean(diffs))
                    n_pairs = len(diffs)
                    s_d = float(np.std(diffs, ddof=1))
                    se = s_d / np.sqrt(n_pairs) if n_pairs > 0 else 0.0
                    df_paired = max(1, n_pairs - 1)

                    if se > 1e-12:
                        t_stat = abs(diff / se)
                        pval = float(2.0 * (1.0 - stats.t.cdf(t_stat, df=df_paired)))
                        crit_95 = float(stats.t.ppf(0.975, df=df_paired))
                        crit_90 = float(stats.t.ppf(0.95, df=df_paired))
                        ci95 = (diff - crit_95 * se, diff + crit_95 * se)
                        ci90 = (diff - crit_90 * se, diff + crit_90 * se)
                    else:
                        pval = 1.0 if abs(diff) <= 1e-12 else 0.0
                        ci95 = (diff, diff)
                        ci90 = (diff, diff)
                else:
                    diff, pval, ci95, ci90 = cls._two_sample_inference(t_df[target_col], base_df[target_col])
            else:
                diff, pval, ci95, ci90 = cls._two_sample_inference(t_df[target_col], base_df[target_col])

            estimated_effects[f] = diff
            selected_config[f] = 1 if diff > 0.0 else 0
            p_values[f] = pval
            conf_95[f] = ci95
            conf_90[f] = ci90

        return MethodResult(
            method_id="ofat_ablation",
            budget_configs=len(configs),
            budget_calls=len(filtered_df),
            selected_config=selected_config,
            estimated_main_effects=estimated_effects,
            p_values=p_values,
            conf_intervals_95=conf_95,
            conf_intervals_90=conf_90,
        )

    @classmethod
    def replay_exhaustive_oracle(
        cls,
        discovery_df: pd.DataFrame,
        factor_ids: List[str],
        target_col: str = "metric",
    ) -> MethodResult:
        """
        Replay exhaustive full factorial on discovery data with valid contrast inference.
        """
        grouped = discovery_df.groupby(factor_ids)[target_col].mean().reset_index()
        best_row = grouped.loc[grouped[target_col].idxmax()]
        selected_config = {f: int(best_row[f]) for f in factor_ids}

        estimated_effects = {}
        p_values = {}
        conf_95 = {}
        conf_90 = {}

        for f in factor_ids:
            sub1 = discovery_df[discovery_df[f] == 1][target_col]
            sub0 = discovery_df[discovery_df[f] == 0][target_col]
            diff, pval, ci95, ci90 = cls._two_sample_inference(sub1, sub0)
            estimated_effects[f] = diff
            p_values[f] = pval
            conf_95[f] = ci95
            conf_90[f] = ci90

        num_configs = len(grouped)

        return MethodResult(
            method_id="exhaustive_oracle",
            budget_configs=num_configs,
            budget_calls=len(discovery_df),
            selected_config=selected_config,
            estimated_main_effects=estimated_effects,
            p_values=p_values,
            conf_intervals_95=conf_95,
            conf_intervals_90=conf_90,
        )
