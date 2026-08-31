"""
Automated Reporting: Generates rich Markdown, HTML, and JSON reports from DoE & ANOVA results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..analysis.anova import ANOVAResult
from ..analysis.optimizer import OptimalPromptFinder, OptimalPromptRecommendation
from ..core.models import DesignMatrix, ExperimentResults
from ..design.aliasing import AliasStructure


class AnalysisReport:
    """
    Comprehensive report object containing full DoE ANOVA analysis, factor effects, and optimal prompt recipe.
    """

    def __init__(
        self,
        experiment_results: Optional[ExperimentResults] = None,
        anova_result: Optional[ANOVAResult] = None,
        optimal_recommendation: Optional[OptimalPromptRecommendation] = None,
        alias_structure: Optional[AliasStructure] = None,
        title: str = "Prompt Design of Experiments (DoE) & ANOVA Analysis Report",
    ):
        self.experiment_results = experiment_results
        self.anova_result = anova_result
        self.optimal_recommendation = optimal_recommendation
        self.alias_structure = alias_structure
        self.title = title

    @classmethod
    def generate(
        cls,
        experiment_results: ExperimentResults,
        target_metric: str,
        factor_name_map: Optional[Dict[str, str]] = None,
        block_col: Optional[str] = "sample_id",
        include_interactions: bool = False,
        alpha: float = 0.05,
        title: str = "Prompt Design of Experiments (DoE) & ANOVA Analysis Report",
    ) -> AnalysisReport:
        """
        Generate a complete AnalysisReport from ExperimentResults with RCBD blocking and aliasing.
        """
        from ..analysis.anova import ANOVAEngine
        
        df = experiment_results.to_dataframe()
        factor_cols = experiment_results.design.factor_ids

        # Alias structure
        alias_struct = None
        if experiment_results.design.generators:
            alias_struct = AliasStructure(experiment_results.design.generators)

        # Run ANOVA with blocking
        anova_res = ANOVAEngine.run_anova(
            data=df,
            factor_cols=factor_cols,
            target_col=target_metric,
            block_col=block_col if (block_col and block_col in df.columns) else None,
            factor_name_map=factor_name_map,
            alias_structure=alias_struct,
            include_interactions=include_interactions,
            alpha=alpha,
        )

        # Optimal prompt recommendation
        optimal_rec = OptimalPromptFinder.find_optimal_prompt(
            anova_result=anova_res,
            factor_names_map=factor_name_map,
        )

        return cls(
            experiment_results=experiment_results,
            anova_result=anova_res,
            optimal_recommendation=optimal_rec,
            alias_structure=alias_struct,
            title=title,
        )

    def to_markdown(self, save_path: Optional[Union[str, Path]] = None) -> str:
        """Render report as detailed GitHub-flavored Markdown."""
        lines = [
            f"# {self.title}",
            f"",
            f"**Target Metric:** `{self.anova_result.target_metric if self.anova_result else 'N/A'}`  ",
            f"**Model R²:** `{self.anova_result.r_squared:.4f}` (Adj. R²: `{self.anova_result.r_squared_adj:.4f}`)  ",
            f"**Significance Level (α):** `{self.anova_result.alpha if self.anova_result else 0.05}`  ",
        ]

        if self.anova_result and self.anova_result.block_col:
            lines.append(f"**Blocking Variable (RCBD):** `{self.anova_result.block_col}`  ")

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary & Optimal Prompt Recipe",
            f"",
        ])

        if self.optimal_recommendation:
            lines.append(self.optimal_recommendation.summary_markdown)
            lines.append("")

        lines.extend([
            f"---",
            f"",
            f"## 2. Factor Main Effects Ranking",
            f"",
            f"| Factor ID | Factor Name | Level 0 Mean | Level 1 Mean | Effect (Δ) | Rel % | t-statistic | p-value | p (Bonf) | Cohen's d | Significant? | Action |",
            f"|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
        ])

        if self.anova_result:
            for eff in self.anova_result.main_effects:
                sig_badge = "✅ YES" if eff.is_significant else "❌ NO"
                if eff.action_recommendation == "ENABLE":
                    action_badge = "🟢 ENABLE"
                elif eff.action_recommendation == "DISABLE":
                    action_badge = "🔴 DISABLE"
                else:
                    action_badge = "⚪ OMIT/DEFAULT"

                lines.append(
                    f"| **{eff.factor_id}** | {eff.factor_name} | {eff.mean_level_0:.4f} | {eff.mean_level_1:.4f} | "
                    f"**{eff.effect_delta:+.4f}** | {eff.relative_change_pct:+.1f}% | {eff.t_statistic:.2f} | "
                    f"{eff.p_value:.4g} | {min(1.0, eff.p_value * len(self.anova_result.main_effects)):.4g} | "
                    f"{eff.cohens_d:.2f} | {sig_badge} | {action_badge} |"
                )

        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Analysis of Variance (ANOVA) Table",
            f"",
            f"| Source | Factor Name | Sum of Sq (SS) | DF | Mean Sq (MS) | F-value | PR(>F) | Partial η² | Significant |",
            f"|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])

        if self.anova_result:
            for row in self.anova_result.anova_table:
                f_str = f"{row.f_statistic:.3f}" if row.f_statistic is not None else "-"
                p_str = f"{row.p_value:.4g}" if row.p_value is not None else "-"
                eta_str = f"{row.partial_eta_sq:.4f}" if row.partial_eta_sq is not None else "-"
                sig_str = "✅" if row.is_significant else ""
                lines.append(
                    f"| {row.source} | {row.factor_name} | {row.sum_sq:.4f} | {row.df:.0f} | {row.mean_sq:.4f} | {f_str} | {p_str} | {eta_str} | {sig_str} |"
                )

        if self.alias_structure:
            lines.extend([
                f"",
                f"---",
                f"",
                f"## 4. Fractional Design & Aliasing Structure",
                f"",
                f"- **Design Resolution:** `Res {self.alias_structure.resolution}`",
                f"- **Defining Relation:** `I = {' = '.join(self.alias_structure.defining_relation)}`",
                f"",
                f"> {self.alias_structure.summary()}",
            ])

        md_text = "\n".join(lines)
        if save_path:
            Path(save_path).write_text(md_text, encoding="utf-8")
        return md_text

    def to_html(self, save_path: Optional[Union[str, Path]] = None) -> str:
        """Render report as standalone, styled HTML."""
        md_content = self.to_markdown()
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{self.title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1, h2, h3 {{ color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.5em 0; font-size: 0.95em; }}
  th, td {{ border: 1px solid #dfe2e5; padding: 8px 12px; text-align: left; }}
  th {{ background-color: #f6f8fa; font-weight: 600; }}
  tr:nth-child(even) {{ background-color: #fcfcfc; }}
  blockquote {{ border-left: 4px solid #0366d6; margin: 1em 0; padding: 0.5em 1em; background-color: #f1f8ff; color: #24292e; }}
  code {{ background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; }}
</style>
</head>
<body>
<pre style="white-space: pre-wrap; font-family: inherit;">
{md_content}
</pre>
</body>
</html>"""
        if save_path:
            Path(save_path).write_text(html, encoding="utf-8")
        return html
