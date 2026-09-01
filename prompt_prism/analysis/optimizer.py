"""
Optimal Prompt Configuration Finder based on Factorial Effects and ANOVA.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


class OptimalPromptFinder:
    """Finds the statistically optimal combination of prompt factors from ANOVA and effects."""

    @classmethod
    def find_optimal_prompt(
        cls,
        anova_result: ANOVAResult,
        default_neutral_level: int = 0,
        factor_names_map: Optional[Dict[str, str]] = None,
        maximize: bool = True,
    ) -> OptimalPromptRecommendation:
        """
        Synthesize ANOVA and factor effect estimates into the optimal prompt configuration.
        """
        factor_names_map = factor_names_map or {}
        pos_drivers: List[FactorEffect] = []
        neg_factors: List[FactorEffect] = []
        neutral_factors: List[FactorEffect] = []

        optimal_levels: Dict[str, int] = {}

        effects_maximize = anova_result.metadata.get("maximize", True)

        for effect in anova_result.main_effects:
            fid = effect.factor_id
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
                optimal_levels[fid] = 1  # Enable booster
            elif action == "DISABLE":
                neg_factors.append(effect)
                optimal_levels[fid] = 0  # Disable harmful
            else:
                neutral_factors.append(effect)
                optimal_levels[fid] = default_neutral_level

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
                    if effect.factor_id in k and "C(" in k:
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
            pred_optimal = float(
                np.clip(intercept + opt_delta_sum, bound_low, bound_high)
            )
            expected_gain = (
                (pred_optimal - baseline_score)
                if maximize
                else (baseline_score - pred_optimal)
            )

            # Approximate 95% CI for predicted gain
            se_gain = (
                float(np.sqrt(sum(e.std_error**2 for e in pos_drivers)))
                if pos_drivers
                else 0.0
            )
            ci_low = expected_gain - 1.96 * se_gain
            ci_high = expected_gain + 1.96 * se_gain
        else:
            # Descriptive fallback
            m0_list = [e.mean_level_0 for e in anova_result.main_effects]
            baseline_score = (
                float(np.mean(m0_list))
                if m0_list
                else (0.5 if is_unit_interval else (bound_low + bound_high) / 2)
            )
            delta_sum = sum(e.effect_delta for e in pos_drivers)
            pred_optimal = float(
                np.clip(baseline_score + delta_sum, bound_low, bound_high)
            )
            expected_gain = (
                (pred_optimal - baseline_score)
                if maximize
                else (baseline_score - pred_optimal)
            )
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
    ) -> str:
        """Format an executive prompt optimization recipe with resolution qualifications."""
        direction_label = " (maximize)" if maximize else " (minimize)"
        lines = [
            f"### 🎯 Optimal Prompt Configuration for `{target_metric}`{direction_label}",
            "",
            f"- **Predicted Optimal Performance:** `{pred_optimal:.4f}`",
            f"- **Baseline Performance:** `{baseline_score:.4f}`",
            f"- **Expected Gain:** `{expected_gain:+.4f}` (`{gain_pct:+.1f}%`)",
        ]

        if ci_low is not None and ci_high is not None:
            lines.append(
                f"- **95% Confidence Interval for Gain:** `[{ci_low:+.4f}, {ci_high:+.4f}]`"
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
