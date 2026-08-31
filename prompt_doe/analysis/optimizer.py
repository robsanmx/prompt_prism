"""
Optimal Prompt Configuration Finder based on statistically significant factors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..core.factors import Factor, FactorSet
from .anova import ANOVAResult
from .effects import FactorEffect


class OptimalPromptRecommendation(BaseModel):
    """
    Recommended optimal prompt configuration derived from ANOVA and effect analysis.
    """
    target_metric: str
    optimal_factor_levels: Dict[str, int] = Field(default_factory=dict)
    optimal_factor_names: Dict[str, int] = Field(default_factory=dict)
    baseline_levels: Dict[str, int] = Field(default_factory=dict)
    predicted_optimal_score: float = 0.0
    predicted_baseline_score: float = 0.0
    expected_gain_absolute: float = 0.0
    expected_gain_percent: float = 0.0
    significant_positive_drivers: List[FactorEffect] = Field(default_factory=list)
    harmful_negative_factors: List[FactorEffect] = Field(default_factory=list)
    neutral_removable_factors: List[FactorEffect] = Field(default_factory=list)
    summary_markdown: str = ""


class OptimalPromptFinder:
    """
    Finds the optimal combination of prompt factor levels that maximizes target metric performance.
    """

    @classmethod
    def find_optimal_prompt(
        cls,
        anova_result: ANOVAResult,
        factors: Optional[Union[FactorSet, Sequence[Factor]]] = None,
        default_baseline_level: int = 0,
        higher_is_better: bool = True,
    ) -> OptimalPromptRecommendation:
        """
        Derive the statistically optimal prompt configuration.
        """
        effects_map = {e.factor_id: e for e in anova_result.main_effects}
        all_factor_ids = list(effects_map.keys())

        optimal_levels: Dict[str, int] = {}
        optimal_names: Dict[str, int] = {}
        baseline_levels: Dict[str, int] = {}

        pos_drivers: List[FactorEffect] = []
        neg_factors: List[FactorEffect] = []
        neutral_factors: List[FactorEffect] = []

        baseline_score = 0.0
        predicted_optimal = 0.0

        for fid, eff in effects_map.items():
            baseline_levels[fid] = default_baseline_level
            
            # Baseline uses mean of level 0
            baseline_score += eff.mean_level_0 / len(effects_map) if len(effects_map) > 0 else 0.0

            if eff.is_significant:
                if (eff.effect_delta > 0 and higher_is_better) or (eff.effect_delta < 0 and not higher_is_better):
                    # Positive driver -> Enable (Level 1)
                    optimal_levels[fid] = 1
                    optimal_names[eff.factor_name] = 1
                    pos_drivers.append(eff)
                    predicted_optimal += eff.mean_level_1 / len(effects_map)
                else:
                    # Negative / Harmful factor -> Disable (Level 0)
                    optimal_levels[fid] = 0
                    optimal_names[eff.factor_name] = 0
                    neg_factors.append(eff)
                    predicted_optimal += eff.mean_level_0 / len(effects_map)
            else:
                # Neutral -> keep default (usually 0 to minimize prompt length / token cost)
                optimal_levels[fid] = default_baseline_level
                optimal_names[eff.factor_name] = default_baseline_level
                neutral_factors.append(eff)
                predicted_optimal += (eff.mean_level_0 if default_baseline_level == 0 else eff.mean_level_1) / len(effects_map)

        # Re-anchor predicted values to overall mean
        overall_mean = (
            sum(e.mean_level_0 * e.sample_count_0 + e.mean_level_1 * e.sample_count_1 for e in effects_map.values())
            / sum(e.sample_count_0 + e.sample_count_1 for e in effects_map.values())
            if effects_map else 0.0
        )

        delta_sum = sum(eff.effect_delta for eff in pos_drivers)
        predicted_optimal = overall_mean + (delta_sum / 2.0)
        predicted_baseline = overall_mean - (sum(abs(eff.effect_delta) for eff in pos_drivers) / 2.0)

        abs_gain = predicted_optimal - predicted_baseline
        rel_gain = (abs_gain / predicted_baseline * 100.0) if predicted_baseline != 0 else 0.0

        # Generate summary markdown
        lines = [
            f"### 🎯 Optimal Prompt Configuration for `{anova_result.target_metric}`",
            f"",
            f"- **Predicted Optimal Performance:** `{predicted_optimal:.4f}`",
            f"- **Baseline Performance:** `{predicted_baseline:.4f}`",
            f"- **Expected Gain:** `+{abs_gain:.4f}` (`+{rel_gain:.1f}%`)",
            f"",
            f"#### 🏆 Statistically Significant Boosters (MUST ENABLE):",
        ]
        if pos_drivers:
            for d in pos_drivers:
                lines.append(f"  - **{d.factor_name}** (`{d.factor_id}`): +{d.effect_delta:.4f} improvement (p = {d.p_value:.4g}, d = {d.cohen_d:.2f})")
        else:
            lines.append("  - None detected at alpha = 0.05.")

        lines.append(f"")
        lines.append(f"#### ⚠️ Harmful Factors (MUST REMOVE/DISABLE):")
        if neg_factors:
            for d in neg_factors:
                lines.append(f"  - **{d.factor_name}** (`{d.factor_id}`): {d.effect_delta:.4f} drop (p = {d.p_value:.4g})")
        else:
            lines.append("  - None detected.")

        lines.append(f"")
        lines.append(f"#### ℹ️ Neutral / Non-Significant Factors (Recommended to Omit to Save Tokens):")
        if neutral_factors:
            for d in neutral_factors:
                lines.append(f"  - **{d.factor_name}** (`{d.factor_id}`): effect = {d.effect_delta:+.4f} (p = {d.p_value:.4g}, not significant)")
        else:
            lines.append("  - None.")

        return OptimalPromptRecommendation(
            target_metric=anova_result.target_metric,
            optimal_factor_levels=optimal_levels,
            optimal_factor_names=optimal_names,
            baseline_levels=baseline_levels,
            predicted_optimal_score=predicted_optimal,
            predicted_baseline_score=predicted_baseline,
            expected_gain_absolute=abs_gain,
            expected_gain_percent=rel_gain,
            significant_positive_drivers=pos_drivers,
            harmful_negative_factors=neg_factors,
            neutral_removable_factors=neutral_factors,
            summary_markdown="\n".join(lines),
        )
