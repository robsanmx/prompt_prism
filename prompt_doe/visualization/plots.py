"""
Diagnostic and Effect Visualizations (Main Effects, Pareto Chart, Interaction Matrix, Daniel Plot, ASCII).
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..analysis.anova import ANOVAResult
from ..analysis.effects import FactorEffect, InteractionEffect


def plot_main_effects(
    anova_result: ANOVAResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> Optional[Any]:
    """
    Generate a Main Effects Plot showing response changes from Level 0 to Level 1 for each factor.
    """
    if not HAS_MATPLOTLIB:
        return None

    effects = anova_result.main_effects
    if not effects:
        return None

    n_factors = len(effects)
    cols = min(4, n_factors)
    rows = int(np.ceil(n_factors / cols))
    fig_size = figsize or (cols * 3.5, rows * 3.0)

    fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharey=True, squeeze=False)
    target_metric = anova_result.target_metric
    overall_title = title or f"Main Effects Plot for {target_metric}"
    fig.suptitle(overall_title, fontsize=14, fontweight="bold", y=1.02)

    # Compute overall y-range
    all_means = [e.mean_level_0 for e in effects] + [e.mean_level_1 for e in effects]
    y_min, y_max = min(all_means), max(all_means)
    y_pad = max(0.05, (y_max - y_min) * 0.15) if y_max != y_min else 0.1

    for idx, eff in enumerate(effects):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        x_vals = [0, 1]
        y_vals = [eff.mean_level_0, eff.mean_level_1]
        
        # Color based on significance
        color = "#2b8a3e" if eff.is_significant and eff.effect_delta > 0 else (
            "#c92a2a" if eff.is_significant and eff.effect_delta < 0 else "#495057"
        )

        ax.plot(x_vals, y_vals, marker="o", linewidth=2.5, markersize=8, color=color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0 (Off)", "1 (On)"], fontsize=9)
        ax.set_title(f"{eff.factor_name} ({eff.factor_id})\nΔ={eff.effect_delta:+.3f} (p={eff.p_value:.3g})", fontsize=10, fontweight="medium")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if c == 0:
            ax.set_ylabel(target_metric, fontsize=10)

    # Hide empty subplots
    for idx in range(n_factors, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_pareto_effects(
    anova_result: ANOVAResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> Optional[Any]:
    """
    Generate a Pareto Chart of Standardized Effects with statistical significance threshold line.
    """
    if not HAS_MATPLOTLIB:
        return None

    effects = anova_result.main_effects
    if not effects:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute t-statistic
    sorted_effs = sorted(effects, key=lambda e: abs(e.t_statistic), reverse=False)
    names = [f"{e.factor_name} ({e.factor_id})" for e in sorted_effs]
    t_vals = [abs(e.t_statistic) for e in sorted_effs]
    colors = ["#2b8a3e" if e.is_significant and e.effect_delta > 0 else (
        "#c92a2a" if e.is_significant and e.effect_delta < 0 else "#868e96"
    ) for e in sorted_effs]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, t_vals, color=colors, height=0.6, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("|Standardized Effect (t-value)|", fontsize=11, fontweight="medium")
    ax.set_title(title or f"Pareto Chart of Effects ({anova_result.target_metric})", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Add critical t line
    from scipy import stats
    df_res = anova_result.residual_df if anova_result.residual_df > 0 else 30
    t_crit = stats.t.ppf(1.0 - anova_result.alpha / 2.0, df=df_res)
    ax.axvline(t_crit, color="#d9480f", linestyle="--", linewidth=2, label=f"Significance (α={anova_result.alpha}, t={t_crit:.2f})")
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_interaction_effects(
    anova_result: ANOVAResult,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> Optional[Any]:
    """
    Generate an Interaction Plot grid showing 2-factor interactions.
    """
    if not HAS_MATPLOTLIB:
        return None

    interactions = anova_result.interactions
    if not interactions:
        return None

    # Take top 6 interactions by absolute magnitude
    top_int = interactions[:6]
    n_int = len(top_int)
    cols = min(3, n_int)
    rows = int(np.ceil(n_int / cols))
    fig_size = figsize or (cols * 4.0, rows * 3.2)

    fig, axes = plt.subplots(rows, cols, figsize=fig_size, sharey=True, squeeze=False)
    fig.suptitle(title or f"2-Factor Interaction Plots ({anova_result.target_metric})", fontsize=14, fontweight="bold", y=1.02)

    for idx, inter in enumerate(top_int):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        f1, f2 = inter.factor_pair
        n1, n2 = inter.factor_names

        # Line for f2 = 0
        y_f2_0 = [inter.mean_00, inter.mean_10]
        # Line for f2 = 1
        y_f2_1 = [inter.mean_01, inter.mean_11]

        ax.plot([0, 1], y_f2_0, marker="o", linewidth=2, label=f"{f2}=0", color="#1c7ed6")
        ax.plot([0, 1], y_f2_1, marker="s", linewidth=2, linestyle="--", label=f"{f2}=1", color="#f76707")

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{f1}=0", f"{f1}=1"], fontsize=9)
        ax.set_title(f"{f1} × {f2}\n(Δ={inter.effect_delta:+.3f})", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        if c == 0:
            ax.set_ylabel(anova_result.target_metric, fontsize=10)

    for idx in range(n_int, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r][c])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def generate_ascii_pareto(anova_result: ANOVAResult, max_width: int = 40) -> str:
    """Generate an ASCII Pareto chart for terminal or text summary display."""
    effects = anova_result.main_effects
    if not effects:
        return "No effects computed."

    max_t = max(abs(e.t_statistic) for e in effects) or 1.0
    lines = [
        f"=== Pareto Chart of Standardized Effects ({anova_result.target_metric}) ===",
        f"Factor ID | Name                     | Effect Δ | t-value | Chart",
        f"----------+--------------------------+----------+---------+-----------------------------------------",
    ]

    for e in effects:
        t_abs = abs(e.t_statistic)
        bar_len = int((t_abs / max_t) * max_width)
        symbol = "█" if e.is_significant else "░"
        sign_char = "+" if e.effect_delta >= 0 else "-"
        bar = symbol * bar_len
        flag = " [*** SIG ***]" if e.is_significant else ""
        lines.append(
            f"   {e.factor_id:<6} | {e.factor_name[:24]:<24} | {e.effect_delta:>+8.4f} | {t_abs:>7.2f} | {bar}{flag}"
        )

    return "\n".join(lines)
