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


def test_optimizer_resolves_sign_composition_paradox_with_epistasis():
    """
    When two positive factors (A and B) clash epistatically when enabled together (A=1, B=1),
    additive sign composition selects the suboptimal corner (A=1, B=1), whereas 'screen_best',
    'interaction_2fi', and 'auto' avoid the antagonistic trap and recover the true optimum.
    """
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    base_df = design.to_dataframe()

    # Cell values for (A, B):
    # (0, 0) -> 0.50
    # (1, 0) -> 0.85  <-- True optimum!
    # (0, 1) -> 0.80
    # (1, 1) -> 0.55  <-- Antagonistic clash when both A and B are enabled!
    # Marginal means:
    # A=0: (0.50 + 0.80)/2 = 0.65, A=1: (0.85 + 0.55)/2 = 0.70 -> delta_A = +0.05 > 0
    # B=0: (0.50 + 0.85)/2 = 0.675, B=1: (0.80 + 0.55)/2 = 0.675 + 0.025 -> let's make (0, 1) = 0.84 so delta_B = +0.045 > 0
    cell_scores = {
        (0, 0): 0.50,
        (1, 0): 0.88,
        (0, 1): 0.84,
        (1, 1): 0.56,
    }
    # Check marginal means:
    # A=0 mean = (0.50 + 0.84)/2 = 0.67; A=1 mean = (0.88 + 0.56)/2 = 0.72 -> delta_A = +0.05 (positive!)
    # B=0 mean = (0.50 + 0.88)/2 = 0.69; B=1 mean = (0.84 + 0.56)/2 = 0.70 -> delta_B = +0.01 (or make (0,0)=0.44 so B=0 is 0.66, B=1 is 0.70 -> delta_B = +0.04!)
    cell_scores[(0, 0)] = 0.44

    records = []
    np.random.seed(7)
    for s_id in range(25):
        item_offset = (s_id - 12) * 0.005
        for _, row in base_df.iterrows():
            a, b = int(row["A"]), int(row["B"])
            score = cell_scores[(a, b)] + item_offset + np.random.normal(0, 0.005)
            records.append(
                {
                    "sample_id": s_id,
                    "run_id": row["run_id"],
                    "A": a,
                    "B": b,
                    "C": int(row["C"]),
                    "D": int(row["D"]),
                    "E": int(row["E"]),
                    "accuracy": float(np.clip(score, 0.0, 1.0)),
                }
            )

    df = pd.DataFrame(records)
    anova_res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="accuracy",
        block_col="sample_id",
        include_interactions=True,
    )

    # 1. Naive additive strategy enables both A=1 and B=1, falling into the (1, 1) trap
    opt_add = OptimalPromptFinder.find_optimal_prompt(anova_res, strategy="additive")
    assert opt_add.optimal_factor_levels["A"] == 1
    assert opt_add.optimal_factor_levels["B"] == 1

    # 2. Screen-Best strategy selects (A=1, B=0) with ~0.88 accuracy
    opt_screen = OptimalPromptFinder.find_optimal_prompt(
        anova_res, strategy="screen_best"
    )
    assert opt_screen.optimal_factor_levels["A"] == 1
    assert opt_screen.optimal_factor_levels["B"] == 0
    assert opt_screen.predicted_optimal_score > 0.85

    # 3. Interaction-2FI strategy models the antagonistic A:B interaction and selects (A=1, B=0)
    opt_2fi = OptimalPromptFinder.find_optimal_prompt(
        anova_res, strategy="interaction_2fi"
    )
    assert opt_2fi.optimal_factor_levels["A"] == 1
    assert opt_2fi.optimal_factor_levels["B"] == 0
    assert opt_2fi.predicted_optimal_score > 0.85

    # 4. Auto strategy detects strong epistasis in Res V and selects interaction_2fi
    opt_auto = OptimalPromptFinder.find_optimal_prompt(anova_res, strategy="auto")
    assert opt_auto.strategy == "interaction_2fi"
    assert opt_auto.optimal_factor_levels["A"] == 1
    assert opt_auto.optimal_factor_levels["B"] == 0

    with pytest.raises(ValueError, match="Unknown strategy"):
        OptimalPromptFinder.find_optimal_prompt(anova_res, strategy="invalid_mode")
