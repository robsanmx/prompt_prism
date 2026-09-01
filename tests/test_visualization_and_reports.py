"""
Tests for Visualization and Reporting modules.
"""

import numpy as np
import pandas as pd
import pytest

from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.analysis.optimizer import OptimalPromptFinder
from prompt_prism.design.generators import FractionalFactorialGenerator
from prompt_prism.reporting.reporter import AnalysisReport
from prompt_prism.visualization.plots import (
    generate_ascii_pareto,
    plot_interaction_effects,
    plot_main_effects,
    plot_pareto_effects,
)


@pytest.fixture
def sample_anova_result():
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    base_df = design.to_dataframe()

    records = []
    np.random.seed(42)
    for s_id in range(8):
        for _, row in base_df.iterrows():
            a, b, c, d, e = row["A"], row["B"], row["C"], row["D"], row["E"]
            score = 0.5 + 0.3 * a - 0.2 * b + 0.1 * c + np.random.normal(0, 0.05)
            records.append(
                {
                    "run_id": row["run_id"],
                    "sample_id": s_id,
                    "A": a,
                    "B": b,
                    "C": c,
                    "D": d,
                    "E": e,
                    "accuracy": float(np.clip(score, 0.0, 1.0)),
                }
            )
    df = pd.DataFrame(records)
    return ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="accuracy",
        include_interactions=True,
    )


def test_plot_main_effects(sample_anova_result, tmp_path):
    save_file = str(tmp_path / "main_effects.png")
    fig = plot_main_effects(sample_anova_result, save_path=save_file)
    assert fig is not None
    assert (tmp_path / "main_effects.png").exists()


def test_plot_pareto_effects(sample_anova_result, tmp_path):
    save_file = str(tmp_path / "pareto.png")
    fig = plot_pareto_effects(sample_anova_result, save_path=save_file)
    assert fig is not None
    assert (tmp_path / "pareto.png").exists()


def test_plot_interaction_effects(sample_anova_result, tmp_path):
    save_file = str(tmp_path / "interactions.png")
    fig = plot_interaction_effects(sample_anova_result, save_path=save_file)
    assert fig is not None
    assert (tmp_path / "interactions.png").exists()


def test_generate_ascii_pareto(sample_anova_result):
    ascii_chart = generate_ascii_pareto(sample_anova_result)
    assert "Pareto Chart of Standardized Effects" in ascii_chart
    assert "Factor ID" in ascii_chart
    assert "accuracy" in ascii_chart


def test_analysis_report_export(sample_anova_result, tmp_path):
    opt = OptimalPromptFinder.find_optimal_prompt(sample_anova_result)
    report = AnalysisReport(
        anova_result=sample_anova_result,
        optimal_recommendation=opt,
        title="Sample Experiment Report",
    )

    md = report.to_markdown(save_path=tmp_path / "report.md")
    html = report.to_html(save_path=tmp_path / "report.html")

    assert "# Sample Experiment Report" in md
    assert "Factor Main Effects Ranking" in md
    assert "Analysis of Variance (ANOVA) Table" in md
    assert "<!DOCTYPE html>" in html
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
