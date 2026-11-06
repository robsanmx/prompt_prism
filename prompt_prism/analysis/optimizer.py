"""
Optimal Prompt Configuration Finder based on Factorial Effects and ANOVA.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .anova import ANOVAResult
from .effects import EffectAnalyzer, FactorEffect


class OptimalPromptRecommendation(BaseModel):
    """Container for optimal prompt factor settings and expected performance gain."""

    target_metric: str
    optimal_factor_levels: Dict[str, int]
    factor_names_map: Dict[str, str] = Field(default_factory=dict)
    predicted_optimal_score: float
    baseline_score: float
    expected_gain_absolute: float
    expected_gain_pct: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    significant_positive_drivers: List[FactorEffect] = Field(default_factory=list)
    harmful_negative_factors: List[FactorEffect] = Field(default_factory=list)
    neutral_factors: List[FactorEffect] = Field(default_factory=list)
    summary_markdown: str = ""
    resolution: int = 8
    strategy: str = "additive"
    screen_best_factor_levels: Dict[str, int] = Field(default_factory=dict)
    screen_best_score: Optional[float] = None
    interaction_2fi_factor_levels: Dict[str, int] = Field(default_factory=dict)
    interaction_2fi_predicted_score: Optional[float] = None


class OptimalPromptFinder:
    """Finds the statistically optimal combination of prompt factors from ANOVA and effects."""

    VALID_STRATEGIES = ("additive", "screen_best", "interaction_2fi", "auto")

    @classmethod
    def _compute_screen_best(
        cls,
        anova_result: ANOVAResult,
        factor_ids: Sequence[str],
        default_levels: Dict[str, int],
        maximize: bool = True,
    ) -> Tuple[Dict[str, int], Optional[float]]:
        """Find the best factor configuration directly observed in the experimental screen."""
        cell_means: Dict[Tuple[int, ...], float] = anova_result.metadata.get(
            "cell_means", {}
        )
        valid_factors: List[str] = anova_result.metadata.get(
            "valid_factors", list(factor_ids)
        )
        if not cell_means or not valid_factors:
            return dict(default_levels), None

        best_key, best_val = (
            max(cell_means.items(), key=lambda kv: kv[1])
            if maximize
            else min(cell_means.items(), key=lambda kv: kv[1])
        )
        levels = dict(default_levels)
        for f_name, lvl in zip(valid_factors, best_key):
            levels[f_name] = int(lvl)
        return levels, float(best_val)

    @classmethod
    def _compute_interaction_2fi_best(
        cls,
        anova_result: ANOVAResult,
        factor_ids: Sequence[str],
        default_levels: Dict[str, int],
        bound_low: float,
        bound_high: float,
        maximize: bool = True,
    ) -> Tuple[Dict[str, int], Optional[float]]:
        """
        Evaluate the 2-factor interaction response surface over {0, 1}^k:
            y_hat(z) = grand_mean + sum_j (tau_j / 2) z_j + sum_{i<j} (tau_{ij} / 2) z_i z_j
        where z_j = 2*x_j - 1 in {-1, +1}.
        """
        if not factor_ids:
            return dict(default_levels), None

        valid_interactions = [
            inter
            for inter in anova_result.interactions
            if not np.isnan(inter.effect_delta)
        ]
        if not valid_interactions:
            return dict(default_levels), None

        grand_mean = anova_result.metadata.get("grand_mean")
        if grand_mean is None:
            all_means = [
                0.5 * (e.mean_level_0 + e.mean_level_1)
                for e in anova_result.main_effects
            ]
            grand_mean = float(np.mean(all_means)) if all_means else 0.5

        k = len(factor_ids)
        fid_to_idx = {fid: idx for idx, fid in enumerate(factor_ids)}

        beta_main = np.zeros(k, dtype=float)
        for eff in anova_result.main_effects:
            if eff.factor_id in fid_to_idx:
                beta_main[fid_to_idx[eff.factor_id]] = 0.5 * float(
                    eff.mean_level_1 - eff.mean_level_0
                )

        beta_inter = np.zeros((k, k), dtype=float)
        for inter in valid_interactions:
            if inter.factor_1 in fid_to_idx and inter.factor_2 in fid_to_idx:
                i = fid_to_idx[inter.factor_1]
                j = fid_to_idx[inter.factor_2]
                if i > j:
                    i, j = j, i
                beta_inter[i, j] = 0.5 * float(inter.effect_delta)

        if k <= 12:
            grid = np.array(
                list(itertools.product([0, 1], repeat=k)), dtype=int
            )  # (2^k, k)
        else:
            # For very high k, evaluate screen candidates + additive candidate + single-bit neighbors
            candidates = [tuple(default_levels.get(f, 0) for f in factor_ids)]
            base = list(candidates[0])
            for idx in range(k):
                flipped = list(base)
                flipped[idx] = 1 - flipped[idx]
                candidates.append(tuple(flipped))
            grid = np.array(list(set(candidates)), dtype=int)

        Z = 2.0 * grid - 1.0  # (N, k) in {-1, +1}
        lin_part = Z @ beta_main
        quad_part = np.sum((Z @ beta_inter) * Z, axis=1)
        preds = np.clip(float(grand_mean) + lin_part + quad_part, bound_low, bound_high)

        best_idx = int(np.argmax(preds) if maximize else np.argmin(preds))
        best_row = grid[best_idx]
        best_levels = {fid: int(best_row[i]) for i, fid in enumerate(factor_ids)}
        return best_levels, float(preds[best_idx])

    @classmethod
    def find_optimal_prompt(
        cls,
        anova_result: ANOVAResult,
        default_neutral_level: int = 0,
        factor_names_map: Optional[Dict[str, str]] = None,
        maximize: bool = True,
        strategy: str = "additive",
    ) -> OptimalPromptRecommendation:
        """
        Synthesize ANOVA and factor effect estimates into the optimal prompt configuration.

        Args:
            anova_result: Completed ANOVAResult from ANOVAEngine.run_anova.
            default_neutral_level: Level assigned to non-significant factors in 'additive' mode (0 or 1).
            factor_names_map: Optional display names for factors.
            maximize: True if higher metric values are better, False if minimizing.
            strategy: Selection rule for optimal_factor_levels:
                - 'additive' (default): Enables significant positive factors, disables negative factors,
                  and sets neutral factors to default_neutral_level.
                - 'screen_best': Selects the highest-scoring (or lowest-scoring if minimizing)
                  factor combination directly observed in the orthogonal experimental screen.
                  Avoids additive sign-composition extrapolation error when factors interact.
                - 'interaction_2fi': Optimizes the full main-effect + pairwise 2-factor interaction
                  (2FI) response surface over {0, 1}^k (recommended for Resolution V / full factorial).
                - 'auto': Automatically selects 'interaction_2fi' (for Res >= V with epistatic interactions),
                  'screen_best' (when strong interactions exist in Res III/IV), or 'additive'.
        """
        if strategy not in cls.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {cls.VALID_STRATEGIES}."
            )

        factor_names_map = factor_names_map or {}
        pos_drivers: List[FactorEffect] = []
        neg_factors: List[FactorEffect] = []
        neutral_factors: List[FactorEffect] = []

        additive_levels: Dict[str, int] = {}
        ordered_factor_ids: List[str] = []

        effects_maximize = anova_result.metadata.get("maximize", True)

        for effect in anova_result.main_effects:
            fid = effect.factor_id
            ordered_factor_ids.append(fid)
            fname = factor_names_map.get(fid, effect.factor_name or fid)
            effect.factor_name = fname

            # EffectAnalyzer is the single producer of this field. Read it; only re-derive
            # (via that same rule, into a local) when the caller asks for a direction the
            # effects were not computed under. Never write it back - these FactorEffect
            # objects belong to the ANOVAResult and other consumers read them too.
            if maximize == effects_maximize:
                action = effect.action_recommendation
            else:
                action = EffectAnalyzer.resolve_action(
                    effect.effect_delta, effect.is_significant, maximize
                )

            if action == "ENABLE":
                pos_drivers.append(effect)
                additive_levels[fid] = 1  # Enable booster
            elif action == "DISABLE":
                neg_factors.append(effect)
                additive_levels[fid] = 0  # Disable harmful
            else:
                neutral_factors.append(effect)
                additive_levels[fid] = default_neutral_level

        # Determine prediction bounds from observed target range
        target_min = anova_result.metadata.get("target_min")
        target_max = anova_result.metadata.get("target_max")
        if target_min is None or target_max is None:
            all_means = [e.mean_level_0 for e in anova_result.main_effects] + [
                e.mean_level_1 for e in anova_result.main_effects
            ]
            target_min = min(all_means) if all_means else 0.0
            target_max = max(all_means) if all_means else 1.0

        is_unit_interval = target_min >= 0.0 and target_max <= 1.0
        bound_low = 0.0 if is_unit_interval else float(target_min)
        bound_high = 1.0 if is_unit_interval else float(target_max)

        # Compute predicted optimal and baseline scores using fitted OLS model coefficients
        ols_model = anova_result.metadata.get("ols_model")
        res_resolution = (
            anova_result.alias_structure.resolution
            if anova_result.alias_structure
            else 8
        )

        if ols_model and hasattr(ols_model, "params"):
            params = dict(ols_model.params)
            intercept = float(params.get("Intercept", 0.0))

            # Sum positive drivers for optimal prediction
            opt_delta_sum = 0.0
            for effect in pos_drivers:
                # Find matching param key
                for k, v in params.items():
                    if effect.factor_id in k and "C(" in k and ":" not in k:
                        opt_delta_sum += float(v)
                        break

            baseline_score = (
                float(np.clip(intercept, bound_low, bound_high))
                if (intercept > 0 or not is_unit_interval)
                else float(
                    anova_result.main_effects[0].mean_level_0
                    if anova_result.main_effects
                    else 0.5
                )
            )
            pred_additive = float(
                np.clip(intercept + opt_delta_sum, bound_low, bound_high)
            )

            # Approximate 95% CI for predicted gain
            se_gain = (
                float(np.sqrt(sum(e.std_error**2 for e in pos_drivers)))
                if pos_drivers
                else 0.0
            )
        else:
            # Descriptive fallback
            m0_list = [e.mean_level_0 for e in anova_result.main_effects]
            baseline_score = (
                float(np.mean(m0_list))
                if m0_list
                else (0.5 if is_unit_interval else (bound_low + bound_high) / 2)
            )
            delta_sum = sum(e.effect_delta for e in pos_drivers)
            pred_additive = float(
                np.clip(baseline_score + delta_sum, bound_low, bound_high)
            )
            se_gain = None

        # Compute Screen-Best and Interaction-2FI alternatives
        valid_factor_order = anova_result.metadata.get(
            "valid_factors", ordered_factor_ids
        )
        screen_best_levels, screen_best_score = cls._compute_screen_best(
            anova_result=anova_result,
            factor_ids=valid_factor_order,
            default_levels=additive_levels,
            maximize=maximize,
        )
        inter_2fi_levels, inter_2fi_score = cls._compute_interaction_2fi_best(
            anova_result=anova_result,
            factor_ids=valid_factor_order,
            default_levels=additive_levels,
            bound_low=bound_low,
            bound_high=bound_high,
            maximize=maximize,
        )

        # Resolve 'auto' strategy if requested
        resolved_strategy = strategy
        if strategy == "auto":
            max_main = max(
                (abs(e.effect_delta) for e in anova_result.main_effects), default=0.0
            )
            valid_inters = [
                i for i in anova_result.interactions if not np.isnan(i.effect_delta)
            ]
            max_inter = max((abs(i.effect_delta) for i in valid_inters), default=0.0)
            has_strong_epistasis = (max_inter > 0.25 * max_main) or any(
                i.is_significant for i in valid_inters
            )
            if (
                res_resolution >= 5
                and has_strong_epistasis
                and inter_2fi_score is not None
            ):
                resolved_strategy = "interaction_2fi"
            elif has_strong_epistasis and screen_best_score is not None:
                resolved_strategy = "screen_best"
            else:
                resolved_strategy = "additive"

        if resolved_strategy == "screen_best" and screen_best_score is not None:
            optimal_levels = dict(screen_best_levels)
            pred_optimal = float(screen_best_score)
        elif resolved_strategy == "interaction_2fi" and inter_2fi_score is not None:
            optimal_levels = dict(inter_2fi_levels)
            pred_optimal = float(inter_2fi_score)
        else:
            optimal_levels = dict(additive_levels)
            pred_optimal = float(pred_additive)

        expected_gain = (
            (pred_optimal - baseline_score)
            if maximize
            else (baseline_score - pred_optimal)
        )
        if se_gain is not None:
            ci_low: Optional[float] = expected_gain - 1.96 * se_gain
            ci_high: Optional[float] = expected_gain + 1.96 * se_gain
        else:
            ci_low = None
            ci_high = None

        gain_pct = (
            expected_gain / (baseline_score if abs(baseline_score) > 1e-9 else 1.0)
        ) * 100

        # Generate summary markdown
        summary_md = cls.generate_summary_markdown(
            target_metric=anova_result.target_metric,
            pos_drivers=pos_drivers,
            neg_factors=neg_factors,
            neutral_factors=neutral_factors,
            optimal_levels=optimal_levels,
            baseline_score=baseline_score,
            pred_optimal=pred_optimal,
            expected_gain=expected_gain,
            gain_pct=gain_pct,
            ci_low=ci_low,
            ci_high=ci_high,
            resolution=res_resolution,
            alpha=anova_result.alpha,
            maximize=maximize,
            strategy=resolved_strategy,
            screen_best_levels=screen_best_levels,
            screen_best_score=screen_best_score,
            inter_2fi_levels=inter_2fi_levels,
            inter_2fi_score=inter_2fi_score,
        )

        return OptimalPromptRecommendation(
            target_metric=anova_result.target_metric,
            optimal_factor_levels=optimal_levels,
            factor_names_map=factor_names_map,
            predicted_optimal_score=pred_optimal,
            baseline_score=baseline_score,
            expected_gain_absolute=expected_gain,
            expected_gain_pct=gain_pct,
            ci_lower=ci_low,
            ci_upper=ci_high,
            significant_positive_drivers=pos_drivers,
            harmful_negative_factors=neg_factors,
            neutral_factors=neutral_factors,
            summary_markdown=summary_md,
            resolution=res_resolution,
            strategy=resolved_strategy,
            screen_best_factor_levels=screen_best_levels,
            screen_best_score=screen_best_score,
            interaction_2fi_factor_levels=inter_2fi_levels,
            interaction_2fi_predicted_score=inter_2fi_score,
        )

    @classmethod
    def generate_summary_markdown(
        cls,
        target_metric: str,
        pos_drivers: List[FactorEffect],
        neg_factors: List[FactorEffect],
        neutral_factors: List[FactorEffect],
        optimal_levels: Dict[str, int],
        baseline_score: float,
        pred_optimal: float,
        expected_gain: float,
        gain_pct: float,
        ci_low: Optional[float],
        ci_high: Optional[float],
        resolution: int,
        alpha: float,
        maximize: bool = True,
        strategy: str = "additive",
        screen_best_levels: Optional[Dict[str, int]] = None,
        screen_best_score: Optional[float] = None,
        inter_2fi_levels: Optional[Dict[str, int]] = None,
        inter_2fi_score: Optional[float] = None,
    ) -> str:
        """Format an executive prompt optimization recipe with resolution qualifications."""
        direction_label = " (maximize)" if maximize else " (minimize)"
        lines = [
            f"### 🎯 Optimal Prompt Configuration for `{target_metric}`{direction_label}",
            "",
            f"- **Selection Strategy:** `{strategy}`",
            f"- **Selected Factor Levels:** `{optimal_levels}`",
            f"- **Predicted Optimal Performance:** `{pred_optimal:.4f}`",
            f"- **Baseline Performance:** `{baseline_score:.4f}`",
            f"- **Expected Gain:** `{expected_gain:+.4f}` (`{gain_pct:+.1f}%`)",
        ]

        if ci_low is not None and ci_high is not None:
            lines.append(
                f"- **95% Confidence Interval for Gain:** `[{ci_low:+.4f}, {ci_high:+.4f}]`"
            )

        if screen_best_score is not None and screen_best_levels:
            lines.append(
                f"- **Best Observed in Orthogonal Screen (`screen_best`):** `{screen_best_score:.4f}` at `{screen_best_levels}`"
            )
        if inter_2fi_score is not None and inter_2fi_levels:
            lines.append(
                f"- **2-Factor Interaction Surface Optimum (`interaction_2fi`):** `{inter_2fi_score:.4f}` at `{inter_2fi_levels}`"
            )

        lines.append("")

        # Add resolution caution if screening design
        if resolution == 3:
            lines.extend(
                [
                    "> ⚠️ **Methodological Notice (Resolution III Screening):**",
                    "> This experiment uses a Resolution III screening design. Main effects are mathematically",
                    "> confounded with 2-factor interactions. Significant factors listed below are **candidates**",
                    "> that require a confirmation run or fold-over design before production deployment.",
                    "",
                ]
            )

        # Positive Drivers
        if resolution >= 5:
            header_pos = "#### 🏆 Statistically Significant Boosters (MUST ENABLE):"
        elif resolution == 4:
            header_pos = "#### 🏆 Statistically Significant Boosters (ENABLE - Res IV clean main effects):"
        else:
            header_pos = (
                "#### 🔍 Candidate Positive Drivers (CONFIRMATION RECOMMENDED):"
            )

        lines.append(header_pos)
        if pos_drivers:
            for d in pos_drivers:
                alias_str = (
                    f" [Aliased with: {', '.join(d.aliased_with)}]"
                    if d.aliased_with
                    else ""
                )
                lines.append(
                    f"  - **{d.factor_name}** (`{d.factor_id}`): effect = {d.effect_delta:+.4f} ({d.relative_change_pct:+.1f}%), "
                    f"p = {d.p_value:.4g}, Cohen's d = {d.cohens_d:.2f}{alias_str}"
                )
        else:
            lines.append(f"  - None detected at alpha = {alpha}.")

        lines.append("")

        # Negative Factors
        if resolution >= 5:
            header_neg = "#### ⚠️ Harmful Factors (MUST REMOVE/DISABLE):"
        else:
            header_neg = "#### ⚠️ Candidate Harmful Factors (RECOMMENDED TO DISABLE):"

        lines.append(header_neg)
        if neg_factors:
            for d in neg_factors:
                alias_str = (
                    f" [Aliased with: {', '.join(d.aliased_with)}]"
                    if d.aliased_with
                    else ""
                )
                lines.append(
                    f"  - **{d.factor_name}** (`{d.factor_id}`): effect = {d.effect_delta:+.4f} ({d.relative_change_pct:+.1f}%), "
                    f"p = {d.p_value:.4g}{alias_str}"
                )
        else:
            lines.append("  - None detected.")

        lines.append("")

        # Neutral Factors
        lines.append(
            "#### ℹ️ Neutral / Non-Significant Factors (Recommended to Omit to Save Tokens):"
        )
        if neutral_factors:
            for d in neutral_factors:
                lines.append(
                    f"  - **{d.factor_name}** (`{d.factor_id}`): effect = {d.effect_delta:+.4f} "
                    f"(p = {d.p_value:.4g}, not significant)"
                )
        else:
            lines.append(
                "  - None (all tested factors had statistically significant effects)."
            )

        return "\n".join(lines)
