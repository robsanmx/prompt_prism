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

    opt = OptimalPromptFinder.find_optimal_prompt(anova_res, maximize=True)

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
    assert "(maximize)" in opt.summary_markdown


def test_optimal_prompt_finder_minimization(synthetic_factorial_dataset):
    df = synthetic_factorial_dataset
    anova_res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="f1_score",
    )

    # When minimizing (e.g. toxicity/bias/hallucination):
    # Factor B (negative effect_delta = -0.25) improves the metric (lowers it) -> B=1
    # Factors A & C (positive effect_delta) hurt the metric (raise it) -> A=0, C=0
    opt_min = OptimalPromptFinder.find_optimal_prompt(anova_res, maximize=False)

    assert opt_min.optimal_factor_levels["B"] == 1
    assert opt_min.optimal_factor_levels["A"] == 0
    assert opt_min.optimal_factor_levels["C"] == 0
    assert opt_min.optimal_factor_levels["D"] == 0
    assert opt_min.optimal_factor_levels["E"] == 0

    assert opt_min.expected_gain_absolute > 0  # Gain is positive (improvement)
    assert any(d.factor_id == "B" for d in opt_min.significant_positive_drivers)
    assert any(d.factor_id == "A" for d in opt_min.harmful_negative_factors)
    assert "(minimize)" in opt_min.summary_markdown


def test_optimal_prompt_unbounded_metric_minimization():
    # R26: Latency in ms (e.g. 200ms to 800ms) under --minimize
    records = []
    np.random.seed(42)
    for s_id in range(10):
        for a in (0, 1):
            for b in (0, 1):
                # A reduces latency by 150ms (good for minimize), B increases latency by 200ms (bad for minimize)
                lat = 500.0 - 150.0 * a + 200.0 * b + float(np.random.normal(0, 10))
                records.append({"sample_id": s_id, "A": a, "B": b, "latency_ms": lat})
    df = pd.DataFrame(records)
    anova_res = ANOVAEngine.run_anova(
        data=df, factor_cols=["A", "B"], target_col="latency_ms", maximize=False
    )
    opt = OptimalPromptFinder.find_optimal_prompt(anova_res, maximize=False)
    assert opt.optimal_factor_levels["A"] == 1
    assert opt.optimal_factor_levels["B"] == 0
    # Optimal and baseline should be on the milliseconds scale (~350ms and ~500ms), NOT clipped to [0, 1]
    assert 250.0 < opt.predicted_optimal_score < 450.0
    assert 400.0 < opt.baseline_score < 600.0
    assert opt.expected_gain_absolute > 100.0


def _minimize_frame():
    """Level 1 scores lower, which is *better* for a minimized metric."""
    rows = []
    for run_id, level, score in ((1, 0, 0.9), (2, 1, 0.1)):
        for sample in range(4):
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample,
                    "A": level,
                    "err_rate": score + 0.01 * sample,
                }
            )
    return pd.DataFrame(rows)


def test_optimizer_does_not_mutate_effect_actions():
    """R36: EffectAnalyzer is the sole producer; find_optimal_prompt only reads."""
    df = _minimize_frame()
    anova_res = ANOVAEngine.run_anova(
        data=df, factor_cols=["A"], target_col="err_rate", maximize=False
    )
    before = [e.action_recommendation for e in anova_res.main_effects]

    opt = OptimalPromptFinder.find_optimal_prompt(anova_res, maximize=False)
    after = [e.action_recommendation for e in anova_res.main_effects]

    assert before == after, "find_optimal_prompt must not write action_recommendation"
    # Minimizing: the lower-scoring level 1 is the improvement.
    assert opt.optimal_factor_levels["A"] == 1
    assert anova_res.main_effects[0].action_recommendation == "ENABLE"


def test_optimizer_rederives_when_direction_disagrees():
    """R36: a caller asking for the other direction still gets a correct answer."""
    df = _minimize_frame()
    # Effects computed for maximize, then asked to minimize.
    anova_res = ANOVAEngine.run_anova(
        data=df, factor_cols=["A"], target_col="err_rate", maximize=True
    )
    assert anova_res.main_effects[0].action_recommendation == "DISABLE"

    opt = OptimalPromptFinder.find_optimal_prompt(anova_res, maximize=False)
    assert opt.optimal_factor_levels["A"] == 1
    # ...and the shared effects were still not written to.
    assert anova_res.main_effects[0].action_recommendation == "DISABLE"


def test_both_plots_agree_on_direction_when_minimizing():
    """R34: Pareto and main-effects plots read the same field, so they cannot disagree."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from prompt_prism.visualization.plots import (
        DISABLE_COLOR,
        ENABLE_COLOR,
        _action_color,
    )

    df = _minimize_frame()
    anova_res = ANOVAEngine.run_anova(
        data=df, factor_cols=["A"], target_col="err_rate", maximize=False
    )
    effect = anova_res.main_effects[0]

    # The improving factor is green in both, because both derive colour from one field.
    assert effect.action_recommendation == "ENABLE"
    assert (
        _action_color(effect.action_recommendation, neutral="#868e96") == ENABLE_COLOR
    )
    assert (
        _action_color(effect.action_recommendation, neutral="#495057") == ENABLE_COLOR
    )
    assert _action_color("DISABLE", neutral="#868e96") == DISABLE_COLOR
