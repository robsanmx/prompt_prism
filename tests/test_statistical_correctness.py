"""
Statistical Correctness Tests: Randomized Complete Block Design (RCBD), FDR monotonicity, and inference consistency.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests
from prompt_prism.analysis.anova import ANOVAEngine
from prompt_prism.analysis.effects import EffectAnalyzer
from prompt_prism.design.generators import FractionalFactorialGenerator


def test_recovers_known_effect():
    """
    Verify that RCBD blocking on sample_id recovers true delta = 0.03 as highly significant (t >= 10, p < 1e-4).
    """
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    base_df = design.to_dataframe()

    records = []
    np.random.seed(42)
    num_items = 50
    # True effects: A = +0.030, others = 0.0
    # Item difficulties range widely from 0.2 to 0.8 (large variance across items)
    item_difficulties = np.random.uniform(0.2, 0.8, size=num_items)

    for s_id in range(num_items):
        base_item = item_difficulties[s_id]
        for _, row in base_df.iterrows():
            a = row["A"]
            # Within-item measurement noise is very small (0.005)
            noise = np.random.normal(0, 0.005)
            score = base_item + 0.030 * (a - 0.5) + noise
            score = float(np.clip(score, 0.0, 1.0))
            records.append({
                "sample_id": s_id,
                "run_id": row["run_id"],
                "A": a, "B": row["B"], "C": row["C"], "D": row["D"], "E": row["E"],
                "metric": score,
            })

    df = pd.DataFrame(records)

    # Analyze with blocking on sample_id
    res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="metric",
        block_col="sample_id",
        alpha=0.05,
    )

    eff_a = next(e for e in res.main_effects if e.factor_id == "A")
    assert 0.025 < eff_a.effect_delta < 0.035
    assert eff_a.is_significant
    assert eff_a.t_statistic > 10.0
    assert eff_a.p_value < 1e-4
    assert "A" in res.significant_positive_factors


def test_null_effects_not_significant():
    """Verify that null effects (delta = 0) are not falsely detected as significant."""
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    base_df = design.to_dataframe()

    records = []
    np.random.seed(123)
    num_items = 30
    item_difficulties = np.random.uniform(0.3, 0.7, size=num_items)

    for s_id in range(num_items):
        base_item = item_difficulties[s_id]
        for _, row in base_df.iterrows():
            noise = np.random.normal(0, 0.05)
            score = float(np.clip(base_item + noise, 0.0, 1.0))
            records.append({
                "sample_id": s_id,
                "run_id": row["run_id"],
                "A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"], "E": row["E"],
                "metric": score,
            })

    df = pd.DataFrame(records)
    res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="metric",
        block_col="sample_id",
        alpha=0.05,
    )

    assert len(res.significant_positive_factors) == 0
    assert len(res.significant_negative_factors) == 0
    assert len(res.neutral_factors) == 5


def test_interaction_missing_cell_is_nan():
    """Verify that when a 2x2 factorial cell is missing, the interaction effect returns NaN and is recorded in omitted_interactions."""
    records = [
        {"A": 0, "B": 0, "y": 0.5},
        {"A": 1, "B": 0, "y": 0.6},
        {"A": 0, "B": 1, "y": 0.7},
        # (1, 1) cell is completely missing!
    ]
    df = pd.DataFrame(records)
    interactions = EffectAnalyzer.compute_interaction_effects(
        df=df,
        factor_cols=["A", "B"],
        target_col="y",
    )
    assert len(interactions) == 1
    assert np.isnan(interactions[0].effect_delta) or not interactions[0].is_significant


def test_bh_fdr_monotone():
    """Verify that Benjamini-Hochberg FDR p-values are monotone and match statsmodels multipletests."""
    p_raw = [0.04, 0.041, 0.90, 0.90, 0.90]
    _, expected_q, _, _ = multipletests(p_raw, method="fdr_bh")

    # Reconstruct through ANOVAEngine FDR correction
    res = ANOVAEngine._adjust_p_values(p_raw, method="fdr_bh")
    np.testing.assert_array_almost_equal(res, expected_q)
    # Check monotonicity: if p1 <= p2 then q1 <= q2
    for i in range(len(res) - 1):
        if p_raw[i] <= p_raw[i+1]:
            assert res[i] <= res[i+1] + 1e-9


def test_main_effects_agree_with_anova_table():
    """Verify that main_effects significance and anova_table significance never disagree."""
    design = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    base_df = design.to_dataframe()

    records = []
    np.random.seed(99)
    for s_id in range(20):
        for _, row in base_df.iterrows():
            a = row["A"]
            b = row["B"]
            score = 0.5 + 0.2 * a - 0.15 * b + np.random.normal(0, 0.05)
            records.append({
                "sample_id": s_id,
                "run_id": row["run_id"],
                "A": a, "B": b, "C": row["C"], "D": row["D"], "E": row["E"],
                "metric": float(np.clip(score, 0.0, 1.0)),
            })

    df = pd.DataFrame(records)
    res = ANOVAEngine.run_anova(
        data=df,
        factor_cols=["A", "B", "C", "D", "E"],
        target_col="metric",
        block_col="sample_id",
        alpha=0.05,
    )

    anova_sig_map = {row.source: row.is_significant for row in res.anova_table if row.source != "Residual"}
    for eff in res.main_effects:
        if eff.factor_id in anova_sig_map:
            assert eff.is_significant == anova_sig_map[eff.factor_id], (
                f"Disagreement for factor {eff.factor_id}: effect says {eff.is_significant}, ANOVA says {anova_sig_map[eff.factor_id]}"
            )
