"""
Unit tests for EffectAnalyzer, ANOVAEngine, and OptimalPromptFinder.
"""

import numpy as np
import pandas as pd
import pytest

from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.analysis.effects import EffectAnalyzer
from prompt_prism.analysis.optimizer import OptimalPromptFinder
from prompt_prism.design.generators import FractionalFactorialGenerator


@pytest.fixture
def synthetic_factorial_dataset():
    """Create a synthetic dataset from a 2^(5-1)V design with known ground-truth factor effects."""
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    base_df = design.to_dataframe()

    # Repeat runs across 10 sample items to create variance and degrees of freedom
    records = []
    np.random.seed(42)

    # True effects:
    # A has strong positive effect (+0.35)
    # B has harmful negative effect (-0.25)
    # C has mild positive effect (+0.15)
    # D and E are zero-effect noise
    for s_id in range(10):
        for _, row in base_df.iterrows():
            a = row["A"]
            b = row["B"]
            c = row["C"]
            d = row["D"]
            e = row["E"]

            noise = np.random.normal(0, 0.05)
            score = 0.50 + 0.35 * a - 0.25 * b + 0.15 * c + 0.0 * d + 0.0 * e + noise
            score = float(np.clip(score, 0.0, 1.0))

            records.append(
                {
                    "run_id": row["run_id"],
                    "sample_id": s_id,
                    "A": a,
                    "B": b,
                    "C": c,
                    "D": d,
                    "E": e,
                    "f1_score": score,
                }
            )

    return pd.DataFrame(records)


def test_main_effects_computation(synthetic_factorial_dataset):
    df = synthetic_factorial_dataset
    effects = EffectAnalyzer.compute_main_effects(
        df=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="f1_score",
    )
    eff_map = {e.factor_id: e for e in effects}

    # Verify effect sizes closely match ground truth
    assert 0.30 < eff_map["A"].effect_delta < 0.40
    assert -0.30 < eff_map["B"].effect_delta < -0.20
    assert 0.10 < eff_map["C"].effect_delta < 0.20
    assert abs(eff_map["D"].effect_delta) < 0.05
    assert abs(eff_map["E"].effect_delta) < 0.05

    assert eff_map["A"].is_significant
    assert eff_map["B"].is_significant
    assert eff_map["C"].is_significant
    assert not eff_map["D"].is_significant
    assert not eff_map["E"].is_significant


def test_anova_engine(synthetic_factorial_dataset):
    df = synthetic_factorial_dataset
    anova_res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="f1_score",
        alpha=0.05,
    )

    assert anova_res.r_squared > 0.85
    assert anova_res.target_metric == "f1_score"

    # Check factor classifications
    assert "A" in anova_res.significant_positive_factors
    assert "C" in anova_res.significant_positive_factors
    assert "B" in anova_res.significant_negative_factors
    assert "D" in anova_res.neutral_factors
    assert "E" in anova_res.neutral_factors

    # Check ANOVA table structure
    table_df = anova_res.to_dataframe()
    assert "F" in table_df.columns
    assert "PR(>F)" in table_df.columns
    assert "Partial Eta^2" in table_df.columns


def test_optimal_prompt_finder(synthetic_factorial_dataset):
    df = synthetic_factorial_dataset
    anova_res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="f1_score",
    )

    opt = OptimalPromptFinder.find_optimal_prompt(anova_res)

    # Optimal levels must enable positive drivers (A=1, C=1), disable negative (B=0), default neutral (D=0, E=0)
    assert opt.optimal_factor_levels["A"] == 1
    assert opt.optimal_factor_levels["B"] == 0
    assert opt.optimal_factor_levels["C"] == 1
    assert opt.optimal_factor_levels["D"] == 0
    assert opt.optimal_factor_levels["E"] == 0

    assert opt.expected_gain_absolute > 0.3
    assert len(opt.significant_positive_drivers) >= 2
    assert len(opt.harmful_negative_factors) == 1
    assert "Optimal Prompt Configuration" in opt.summary_markdown
