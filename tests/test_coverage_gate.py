"""
Unit and integration tests for PromptPrism Coverage Gate in `diagnostics.py`, `Experiment.diagnose()`, and CLI.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from prompt_prism import (
    PRUNING_VERDICTS,
    ExactMatch,
    Experiment,
    Factor,
    run_calibrated_attribution_audit,
)


def _make_paired_df(
    factor_effects: Dict[str, Dict[str, float]],
    *,
    item_ids: Sequence[str],
    noise_sd: float = 0.005,
    base_score: float = 0.50,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a balanced 2^k full-factorial paired DataFrame with per-item factor effects.

    `factor_effects[fid][item_id]` specifies the level-1 minus level-0 shift for `fid` on `item_id`
    (default 0.0 when omitted).
    """
    rng = np.random.default_rng(seed)
    fids = list(factor_effects.keys())
    k = len(fids)
    combos = [
        tuple((mask >> bit) & 1 for bit in range(k - 1, -1, -1))
        for mask in range(1 << k)
    ]

    rows: List[Dict[str, object]] = []
    for s_idx, sid in enumerate(item_ids):
        item_offset = 0.04 * ((s_idx % 5) - 2)
        for run_id, combo in enumerate(combos):
            y = base_score + item_offset
            rec: Dict[str, object] = {"run_id": run_id, "sample_id": sid}
            for fid, lvl in zip(fids, combo):
                rec[fid] = lvl
                if lvl == 1:
                    y += factor_effects[fid].get(sid, 0.0)
            if noise_sd > 0.0:
                y += float(rng.normal(0.0, noise_sd))
            rec["score"] = float(y)
            rows.append(rec)
    return pd.DataFrame(rows)


def _by_factor(audit) -> Dict[str, object]:
    return {f.factor_id: f for f in audit.factors}


# ---------------------------------------------------------------------------
# 1. Full 11-row verdict truth table (maximize=True)
# ---------------------------------------------------------------------------


def test_verdict_row1_f_helps_t_hurts_is_conflict_review():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    all_ids = base_ids + targ_ids
    effects = {
        "A": {**{s: +0.20 for s in base_ids}, **{s: -0.30 for s in targ_ids}},
    }
    df = _make_paired_df(effects, item_ids=all_ids, noise_sd=0.01, seed=1)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        n_sign_flip=0,
    )
    a = _by_factor(audit)["A"]
    assert a.pruning_verdict == "CONFLICT_REVIEW"
    assert a.effect_delta > 0.15
    assert a.delta_targeted < -0.25


def test_verdict_row2_f_helps_t_neutral_is_keep():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    effects = {"A": {**{s: +0.15 for s in base_ids}, **{s: 0.0 for s in targ_ids}}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.01, seed=2)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        n_sign_flip=0,
    )
    assert _by_factor(audit)["A"].pruning_verdict == "KEEP"


def test_verdict_row3_f_hurts_with_insufficient_targeted_coverage_is_conflict_review():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    small_targ = [f"t_{i:03d}" for i in range(5)]
    effects = {
        "A": {s: -0.15 for s in base_ids + small_targ},
        "B": {s: -0.15 for s in base_ids + small_targ},
    }
    df = _make_paired_df(effects, item_ids=base_ids + small_targ, noise_sd=0.01, seed=3)
    audit = run_calibrated_attribution_audit(
        df,
        ["A", "B"],
        "score",
        coverage={"A": [], "B": small_targ},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        n_sign_flip=0,
    )
    res = _by_factor(audit)
    assert res["A"].pruning_verdict == "CONFLICT_REVIEW"
    assert res["A"].n_targeted == 0
    assert res["B"].pruning_verdict == "CONFLICT_REVIEW"
    assert res["B"].n_targeted == 5
    assert any("insufficient targeted coverage" in w for w in audit.warnings)


def test_verdict_row4_f_hurts_t_helps_is_conflict_review():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    effects = {"A": {**{s: -0.10 for s in base_ids}, **{s: +0.40 for s in targ_ids}}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.01, seed=4)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        n_sign_flip=0,
    )
    assert _by_factor(audit)["A"].pruning_verdict == "CONFLICT_REVIEW"


def test_verdict_row5_f_hurts_t_no_benefit_with_coverage_is_remove_harmful():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    effects = {"A": {**{s: -0.15 for s in base_ids}, **{s: 0.0 for s in targ_ids}}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.01, seed=5)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        n_sign_flip=0,
    )
    assert _by_factor(audit)["A"].pruning_verdict == "REMOVE_HARMFUL"


def test_verdict_row6_f_neutral_n_t_zero_is_untested():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    effects = {"A": {s: 0.0 for s in base_ids}}
    df = _make_paired_df(effects, item_ids=base_ids, noise_sd=0.002, seed=6)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": []},
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    a = _by_factor(audit)["A"]
    assert a.tost_p_value < 0.05  # Would have been SAFE_TO_PRUNE without coverage gate
    assert a.n_targeted == 0
    assert a.pruning_verdict == "UNTESTED"


def test_verdict_row7_f_neutral_thin_coverage_is_inconclusive_including_n1():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    effects = {"A": {s: 0.0 for s in base_ids}, "B": {s: 0.0 for s in base_ids}}
    df = _make_paired_df(effects, item_ids=base_ids, noise_sd=0.002, seed=7)
    audit = run_calibrated_attribution_audit(
        df,
        ["A", "B"],
        "score",
        coverage={"A": base_ids[:1], "B": base_ids[:10]},
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    res = _by_factor(audit)
    assert res["A"].n_targeted == 1
    assert res["A"].se_targeted is None  # No invalid unpaired fallback for n_T == 1
    assert res["A"].p_value_targeted is None
    assert res["A"].pruning_verdict == "INCONCLUSIVE_NEED_MORE_ITEMS"
    assert res["B"].n_targeted == 10
    assert res["B"].pruning_verdict == "INCONCLUSIVE_NEED_MORE_ITEMS"


def test_verdict_row8_f_neutral_t_helps_is_keep_targeted():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    effects = {"A": {**{s: 0.0 for s in base_ids}, **{s: +0.60 for s in targ_ids}}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.005, seed=8)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    a = _by_factor(audit)["A"]
    assert a.tost_p_value < 0.05  # F is dead flat
    assert a.pruning_verdict == "KEEP_TARGETED"
    assert a.delta_targeted == pytest.approx(0.60, abs=0.02)


def test_verdict_row9_f_neutral_t_hurts_is_remove_harmful():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    effects = {"A": {**{s: 0.0 for s in base_ids}, **{s: -0.35 for s in targ_ids}}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.005, seed=9)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    a = _by_factor(audit)["A"]
    assert a.pruning_verdict == "REMOVE_HARMFUL"


def test_verdict_row10_f_tost_and_t_ni_pass_is_safe_to_prune_with_tokens():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(35)]
    effects = {"A": {s: 0.0 for s in base_ids + targ_ids}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.005, seed=10)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        targeted_margin=0.05,
        token_costs={"A": 42},
        n_sign_flip=0,
    )
    a = _by_factor(audit)["A"]
    assert a.pruning_verdict == "SAFE_TO_PRUNE"
    assert (
        a.p_noninferiority_targeted is not None and a.p_noninferiority_targeted < 0.05
    )
    assert audit.estimated_prunable_tokens == 42


def test_verdict_row11_f_tost_passes_but_t_ni_fails_is_inconclusive():
    """Regression test for the non-inferiority sign bug: a noisy positive targeted effect
    whose upper bound exceeds targeted_margin must NEVER become SAFE_TO_PRUNE."""
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(25)]
    # Construct targeted items with mean delta_T = +0.06 and wide variance (not significant,
    # ci_low_T > -0.05 would have falsely passed with the inverted sign, but upper bound >> +0.05!)
    rng = np.random.default_rng(11)
    targ_shifts = rng.normal(0.06, 0.20, size=len(targ_ids))
    targ_shifts = targ_shifts - np.mean(targ_shifts) + 0.06
    effects = {
        "A": {
            **{s: 0.0 for s in base_ids},
            **{s: float(v) for s, v in zip(targ_ids, targ_shifts)},
        }
    }
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.002, seed=11)
    audit = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        targeted_margin=0.05,
        n_sign_flip=0,
    )
    a = _by_factor(audit)["A"]
    assert a.tost_p_value < 0.05
    assert a.ci95_low_targeted is not None and a.ci95_low_targeted > -0.05
    assert (
        a.p_noninferiority_targeted is not None and a.p_noninferiority_targeted >= 0.05
    )
    assert a.pruning_verdict == "INCONCLUSIVE_NEED_MORE_ITEMS"


# ---------------------------------------------------------------------------
# 2. Orientation tests for maximize=False (lower-is-better metric, e.g. error rate)
# ---------------------------------------------------------------------------


def test_minimize_direction_targeted_noninferiority_and_verdicts():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(30)]
    # Under maximize=False (minimizing error rate):
    # - A lowers error rate by -0.50 on T (helps!) -> KEEP_TARGETED, never SAFE_TO_PRUNE
    # - B raises error rate by +0.40 on T (hurts!) -> REMOVE_HARMFUL
    # - C has ~0.00 effect everywhere -> SAFE_TO_PRUNE
    # - D lowers error by -0.06 (noisy, non-sig) on T -> INCONCLUSIVE (cannot prune!)
    rng = np.random.default_rng(12)
    d_shifts = rng.normal(-0.06, 0.20, size=len(targ_ids))
    d_shifts = d_shifts - np.mean(d_shifts) - 0.06
    effects = {
        "A": {**{s: 0.0 for s in base_ids}, **{s: -0.50 for s in targ_ids}},
        "B": {**{s: 0.0 for s in base_ids}, **{s: +0.40 for s in targ_ids}},
        "C": {s: 0.0 for s in base_ids + targ_ids},
        "D": {
            **{s: 0.0 for s in base_ids},
            **{s: float(v) for s, v in zip(targ_ids, d_shifts)},
        },
    }
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.003, seed=12)
    audit = run_calibrated_attribution_audit(
        df,
        ["A", "B", "C", "D"],
        "score",
        maximize=False,
        coverage={k: targ_ids for k in ("A", "B", "C", "D")},
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        targeted_margin=0.05,
        n_sign_flip=0,
    )
    res = _by_factor(audit)
    assert res["A"].pruning_verdict == "KEEP_TARGETED"
    assert res["A"].oriented_effect_targeted > 0.45
    assert res["A"].p_noninferiority_targeted > 0.99
    assert res["B"].pruning_verdict == "REMOVE_HARMFUL"
    assert res["C"].pruning_verdict == "SAFE_TO_PRUNE"
    assert res["D"].pruning_verdict == "INCONCLUSIVE_NEED_MORE_ITEMS"


# ---------------------------------------------------------------------------
# 3. Zero-variance unanimous deterministic contrast vs. exact zero null
# ---------------------------------------------------------------------------


def test_zero_variance_unanimous_nonzero_and_exact_null_and_json_safety():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(25)]
    effects = {
        "rare_rule": {**{s: 0.0 for s in base_ids}, **{s: +1.0 for s in targ_ids}},
        "global_rule": {s: +1.0 for s in base_ids + targ_ids},
        "dead_clause": {s: 0.0 for s in base_ids + targ_ids},
    }
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.0, seed=13)
    audit = run_calibrated_attribution_audit(
        df,
        ["rare_rule", "global_rule", "dead_clause"],
        "score",
        coverage={
            "rare_rule": targ_ids,
            "global_rule": targ_ids,
            "dead_clause": targ_ids,
        },
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        targeted_margin=0.05,
        n_sign_flip=500,
    )
    res = _by_factor(audit)

    # Global deterministic +1.0 -> KEEP with exact Rademacher p=2^(1-100) and finite t_statistic
    assert res["global_rule"].se_item_clustered == 0.0
    assert res["global_rule"].p_value_clustered == pytest.approx(2.0 ** (1 - 100))
    assert math.isfinite(res["global_rule"].t_statistic)
    assert res["global_rule"].pruning_verdict == "KEEP"

    # Rare deterministic +1.0 on T only -> KEEP_TARGETED with exact Rademacher p_T=2^(1-25), p_ni_T=1.0
    assert res["rare_rule"].se_targeted == 0.0
    assert res["rare_rule"].p_value_targeted == pytest.approx(2.0 ** (1 - 25))
    assert res["rare_rule"].p_value_sign_flip_targeted == pytest.approx(2.0 ** (1 - 25))
    assert res["rare_rule"].p_noninferiority_targeted == 1.0
    assert res["rare_rule"].pruning_verdict == "KEEP_TARGETED"

    # Dead deterministic 0.0 -> p=1.0, TOST=0.0, NI=0.0 -> SAFE_TO_PRUNE
    assert res["dead_clause"].p_value_clustered == 1.0
    assert res["dead_clause"].p_value_sign_flip == 1.0
    assert res["dead_clause"].p_noninferiority_targeted == 0.0
    assert res["dead_clause"].pruning_verdict == "SAFE_TO_PRUNE"

    # Strict JSON round-trip (no NaN/Infinity emitted)
    parsed = json.loads(audit.model_dump_json())
    assert parsed["schema_version"] == "2.1.0"
    for f in parsed["factors"]:
        assert f["pruning_verdict"] in PRUNING_VERDICTS


def test_zero_variance_unpaired_single_item_remains_conservative():
    # Only 1 unpaired row per level -> se=0, delta=1.0 -> must NOT claim p=0.0
    df = pd.DataFrame(
        [
            {"sample_id": "s0", "A": 0, "score": 0.0},
            {"sample_id": "s1", "A": 1, "score": 1.0},
        ]
    )
    audit = run_calibrated_attribution_audit(df, ["A"], "score", n_sign_flip=0)
    a = _by_factor(audit)["A"]
    assert a.p_value_clustered == 1.0
    assert a.pruning_verdict == "INCONCLUSIVE_NEED_MORE_ITEMS"


def test_zero_variance_tiny_n_is_not_anticonservative():
    """With n=2..5 unanimous flips, exact Rademacher p = 2^(1-n) >= 0.0625 must NOT falsely fire KEEP/CONFLICT_REVIEW."""
    for n_small, expected_p in ((2, 0.5), (5, 0.0625)):
        ids = [f"s_{i}" for i in range(n_small)]
        df = _make_paired_df({"A": {s: +1.0 for s in ids}}, item_ids=ids, noise_sd=0.0)
        audit = run_calibrated_attribution_audit(df, ["A"], "score", n_sign_flip=100)
        a = _by_factor(audit)["A"]
        assert a.p_value_clustered == pytest.approx(expected_p)
        assert a.p_value_sign_flip == pytest.approx(expected_p)
        assert a.pruning_verdict == "INCONCLUSIVE_NEED_MORE_ITEMS"

    # At n=6, 2^(1-6) = 0.03125 < 0.05 -> KEEP
    ids6 = [f"s_{i}" for i in range(6)]
    df6 = _make_paired_df({"A": {s: +1.0 for s in ids6}}, item_ids=ids6, noise_sd=0.0)
    audit6 = run_calibrated_attribution_audit(df6, ["A"], "score", n_sign_flip=100)
    a6 = _by_factor(audit6)["A"]
    assert a6.p_value_clustered == pytest.approx(0.03125)
    assert a6.pruning_verdict == "KEEP"

    # Tiny n_T=2 hurting on T must NOT falsely turn F-helps into CONFLICT_REVIEW
    base_ids = [f"b_{i:02d}" for i in range(30)]
    tiny_t = ["t_0", "t_1"]
    df_mix = _make_paired_df(
        {"A": {**{s: +0.30 for s in base_ids}, **{s: -1.0 for s in tiny_t}}},
        item_ids=base_ids + tiny_t,
        noise_sd=0.0,
    )
    audit_mix = run_calibrated_attribution_audit(
        df_mix,
        ["A"],
        "score",
        coverage={"A": tiny_t},
        baseline_sample_ids=base_ids,
        n_sign_flip=0,
    )
    a_mix = _by_factor(audit_mix)["A"]
    assert a_mix.p_value_targeted == pytest.approx(0.5)
    assert a_mix.pruning_verdict == "KEEP"


def test_input_validation_guards_bare_str_none_and_judge_fraction():
    ids = [f"s_{i:02d}" for i in range(25)]
    df = _make_paired_df({"A": {}}, item_ids=ids, noise_sd=0.002, seed=30)

    # Bare str / bytes in coverage values or baseline_sample_ids must raise TypeError
    with pytest.raises(TypeError, match="bare str"):
        run_calibrated_attribution_audit(
            df, ["A"], "score", coverage={"A": "s_00"}, n_sign_flip=0
        )
    with pytest.raises(TypeError, match="bare str"):
        run_calibrated_attribution_audit(
            df, ["A"], "score", baseline_sample_ids="s_00", n_sign_flip=0
        )

    # None value in coverage map treated as empty set (UNTESTED)
    audit_none = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": None},  # type: ignore[dict-item]
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    assert _by_factor(audit_none)["A"].n_targeted == 0
    assert _by_factor(audit_none)["A"].pruning_verdict == "UNTESTED"

    # judge_scored_fraction outside [0, 1] or NaN raises ValueError
    for bad_frac in (-0.01, 1.5, float("nan")):
        with pytest.raises(ValueError, match="judge_scored_fraction"):
            run_calibrated_attribution_audit(
                df, ["A"], "score", judge_scored_fraction=bad_frac, n_sign_flip=0
            )


# ---------------------------------------------------------------------------
# 4. Baseline estimand separation & T-family BH exclusion
# ---------------------------------------------------------------------------


def test_baseline_sample_ids_prevents_targeted_contamination_of_f():
    base_ids = [f"b_{i:03d}" for i in range(100)]
    targ_ids = [f"t_{i:03d}" for i in range(60)]
    effects = {"A": {**{s: 0.0 for s in base_ids}, **{s: +0.80 for s in targ_ids}}}
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.005, seed=14)

    audit_sep = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=base_ids,
        n_sign_flip=0,
    )
    audit_pooled = run_calibrated_attribution_audit(
        df,
        ["A"],
        "score",
        coverage={"A": targ_ids},
        baseline_sample_ids=None,
        n_sign_flip=0,
    )

    assert audit_sep.n_items == 100
    assert audit_sep.baseline_only_full_set is True
    assert _by_factor(audit_sep)["A"].effect_delta == pytest.approx(0.0, abs=0.005)
    assert _by_factor(audit_sep)["A"].pruning_verdict == "KEEP_TARGETED"

    assert audit_pooled.n_items == 160
    assert audit_pooled.baseline_only_full_set is False
    assert _by_factor(audit_pooled)["A"].effect_delta == pytest.approx(0.30, abs=0.01)


def test_t_family_bh_excludes_untested_factors():
    base_ids = [f"b_{i:03d}" for i in range(60)]
    targ_ids = [f"t_{i:03d}" for i in range(25)]
    effects = {
        "untested_1": {s: 0.0 for s in base_ids + targ_ids},
        "untested_2": {s: 0.0 for s in base_ids + targ_ids},
        "tested": {**{s: 0.0 for s in base_ids}, **{s: +0.025 for s in targ_ids}},
    }
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.04, seed=15)
    audit = run_calibrated_attribution_audit(
        df,
        ["untested_1", "untested_2", "tested"],
        "score",
        coverage={"untested_1": [], "untested_2": [], "tested": targ_ids},
        baseline_sample_ids=base_ids,
        n_sign_flip=0,
    )
    res = _by_factor(audit)
    assert res["untested_1"].p_value_targeted is None
    assert res["untested_1"].q_value_bh_targeted is None
    assert res["untested_2"].q_value_bh_targeted is None
    # With only 1 tested factor in the T family (m = 1), q_T must equal p_T
    assert res["tested"].q_value_bh_targeted == pytest.approx(
        res["tested"].p_value_targeted
    )


# ---------------------------------------------------------------------------
# 5. targets_factors column parsing & coverage key validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell_encoder",
    [
        lambda tags: list(tags),
        lambda tags: json.dumps(list(tags)),
        lambda tags: repr(list(tags)),
        lambda tags: ";".join(tags),
        lambda tags: ",".join(tags),
    ],
)
def test_targets_factors_column_formats(cell_encoder):
    base_ids = [f"b_{i:03d}" for i in range(30)]
    targ_ids = [f"t_{i:03d}" for i in range(25)]
    effects = {
        "A": {**{s: 0.0 for s in base_ids}, **{s: +0.50 for s in targ_ids}},
        "B": {s: 0.0 for s in base_ids + targ_ids},
    }
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.003, seed=16)
    df["targets_factors"] = [
        cell_encoder(["A"]) if sid in set(targ_ids) else np.nan
        for sid in df["sample_id"]
    ]

    audit = run_calibrated_attribution_audit(
        df,
        ["A", "B"],
        "score",
        baseline_sample_ids=base_ids,
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    res = _by_factor(audit)
    assert res["A"].n_targeted == 25
    assert res["A"].pruning_verdict == "KEEP_TARGETED"
    assert res["B"].n_targeted == 0
    assert res["B"].pruning_verdict == "UNTESTED"


def test_coverage_key_validation_and_empty_dict():
    ids = [f"s_{i:03d}" for i in range(30)]
    df = _make_paired_df({"A": {}, "B": {}}, item_ids=ids, noise_sd=0.002, seed=17)

    # Non-empty coverage with zero matching keys raises ValueError
    with pytest.raises(ValueError, match="do not match any factor_cols"):
        run_calibrated_attribution_audit(
            df, ["A", "B"], "score", coverage={"specialist_persona": ids}, n_sign_flip=0
        )

    # Partial mismatch warns and proceeds
    audit_partial = run_calibrated_attribution_audit(
        df,
        ["A", "B"],
        "score",
        coverage={"A": ids, "typo_col": ids},
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    assert any("typo_col" in w for w in audit_partial.warnings)
    assert _by_factor(audit_partial)["A"].pruning_verdict == "SAFE_TO_PRUNE"
    assert _by_factor(audit_partial)["B"].pruning_verdict == "UNTESTED"

    # Empty dict activates coverage gate cleanly (all UNTESTED)
    audit_empty = run_calibrated_attribution_audit(
        df, ["A", "B"], "score", coverage={}, equivalence_margin=0.02, n_sign_flip=0
    )
    assert all(f.pruning_verdict == "UNTESTED" for f in audit_empty.factors)


# ---------------------------------------------------------------------------
# 6. Experiment.run() + Experiment.diagnose() ID/Name resolution & provenance
# ---------------------------------------------------------------------------


def test_experiment_diagnose_resolves_factor_names_and_metadata():
    factors = [
        Factor.binary(
            "persona", level_0_content="", level_1_content="You are an expert."
        ),
        Factor.binary(
            "null_date_rule",
            level_0_content="",
            level_1_content="If no date, return null.",
        ),
    ]
    exp = Experiment.from_factors(
        factors=factors,
        system_prompt="Extract date.",
        data_template="Text: {{ text }}",
        metrics=[ExactMatch()],
        target_metric="exact_match",
    )

    dataset = []
    for i in range(20):
        dataset.append(
            {
                "id": f"base_{i:02d}",
                "text": f"Meeting on 2026-01-{(i % 28) + 1:02d}",
                "target": "2026-01-01",
                "targets_factors": ["persona"],
            }
        )
    for i in range(25):
        dataset.append(
            {
                "id": f"nodate_{i:02d}",
                "text": f"No date mentioned here #{i}",
                "target": "null",
                "targets_factors": ["null_date_rule"],
            }
        )

    def mock_client(prompt):
        p = str(prompt)
        if "No date mentioned" in p:
            return "null" if "If no date, return null." in p else "unknown"
        return "2026-01-01"

    exp.run(dataset=dataset, client=mock_client, max_workers=2)

    # Factor names in `targets_factors` and `token_costs` resolve automatically to design IDs ("A", "B")
    audit = exp.diagnose(
        baseline_sample_ids=[f"base_{i:02d}" for i in range(20)],
        min_targeted_items=20,
        equivalence_margin=0.02,
        token_costs={"persona": 5, "null_date_rule": 12},
        coverage_source="synthetic",
        test_set_sha256="deadbeef1234",
        judge_scored_fraction=0.25,
        generator_model_id="gemini-2.5-flash",
        model_id="gemini-2.5-flash",
        n_sign_flip=100,
    )
    res = _by_factor(audit)
    assert res["A"].pruning_verdict == "SAFE_TO_PRUNE"
    assert res["A"].token_cost == 5
    assert res["B"].pruning_verdict == "KEEP_TARGETED"
    assert res["B"].n_targeted == 25
    assert res["B"].token_cost == 12
    assert audit.estimated_prunable_tokens == 5
    assert audit.evidence_tier == "predicted (synthetic tests)"
    assert audit.test_set_sha256 == "deadbeef1234"
    assert any("Generator model" in w for w in audit.warnings)

    md = audit.to_markdown()
    assert "Covered N" in md
    assert "Δ targeted" in md
    assert "NI p (T)" in md
    assert "deadbeef1234" in md
    assert "25.0%" in md
    assert "`predicted (synthetic tests)`" in md

    # Explicit coverage dict keyed by factor names also resolves cleanly
    audit_explicit = exp.diagnose(
        coverage={
            "persona": [f"base_{i:02d}" for i in range(20)],
            "null_date_rule": [],
        },
        baseline_sample_ids=[f"base_{i:02d}" for i in range(20)],
        min_targeted_items=20,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    assert _by_factor(audit_explicit)["B"].pruning_verdict == "UNTESTED"


# ---------------------------------------------------------------------------
# 7. Exact backward compatibility when coverage=None
# ---------------------------------------------------------------------------


def test_backward_compatibility_without_coverage():
    ids = [f"item_{i:03d}" for i in range(120)]
    effects = {
        "A": {s: +0.25 for s in ids},
        "B": {s: -0.15 for s in ids},
        "C": {s: 0.0 for s in ids},
    }
    df = _make_paired_df(effects, item_ids=ids, noise_sd=0.005, seed=42)
    audit = run_calibrated_attribution_audit(
        df,
        ["A", "B", "C"],
        "score",
        equivalence_margin=0.01,
        token_costs={"A": 80, "B": 100, "C": 320},
        n_sign_flip=200,
        seed=42,
    )
    res = _by_factor(audit)
    assert res["A"].pruning_verdict == "KEEP"
    assert res["B"].pruning_verdict == "REMOVE_HARMFUL"
    assert res["C"].pruning_verdict == "SAFE_TO_PRUNE"
    assert audit.estimated_prunable_tokens == 320
    assert audit.coverage_source == "none"
    assert audit.evidence_tier == "predicted"
    assert audit.targeted_margin is None
    assert any("No coverage map provided" in w for w in audit.warnings)

    md = audit.to_markdown()
    assert "Covered N" not in md
    assert "`predicted`" in md


# ---------------------------------------------------------------------------
# 8. CLI diagnose with targets_factors column (including all-NaN guard)
# ---------------------------------------------------------------------------


def test_cli_diagnose_with_targets_factors_and_all_nan_guard(tmp_path):
    base_ids = [f"b_{i:03d}" for i in range(40)]
    targ_ids = [f"t_{i:03d}" for i in range(25)]
    effects = {
        "A": {**{s: 0.0 for s in base_ids}, **{s: +0.60 for s in targ_ids}},
        "B": {s: 0.0 for s in base_ids + targ_ids},
    }
    df = _make_paired_df(effects, item_ids=base_ids + targ_ids, noise_sd=0.003, seed=21)
    # 1. CSV with JSON targets_factors (A covered, B untested)
    df_tagged = df.copy()
    df_tagged["targets_factors"] = [
        json.dumps(["A"]) if sid in set(targ_ids) else ""
        for sid in df_tagged["sample_id"]
    ]
    csv_tagged = tmp_path / "tagged.csv"
    json_out = tmp_path / "out.json"
    df_tagged.to_csv(csv_tagged, index=False)

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "diagnose",
            "--data",
            str(csv_tagged),
            "--target",
            "score",
            "--prune-margin",
            "0.02",
            "--min-targeted-items",
            "20",
            "--targeted-margin",
            "0.05",
            "--sign-flip",
            "0",
            "--json-out",
            str(json_out),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    verdicts = {f["factor_id"]: f["pruning_verdict"] for f in payload["factors"]}
    assert set(verdicts.keys()) == {"A", "B"}
    assert verdicts["B"] == "UNTESTED"
    assert "Covered N" in res.stdout

    # 2. CSV with all-NaN targets_factors column: must NOT be auto-detected as a binary factor!
    df_empty_tags = df.copy()
    df_empty_tags["targets_factors"] = np.nan
    csv_empty = tmp_path / "empty_tags.csv"
    json_empty = tmp_path / "empty_out.json"
    df_empty_tags.to_csv(csv_empty, index=False)

    res2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "diagnose",
            "--data",
            str(csv_empty),
            "--target",
            "score",
            "--sign-flip",
            "0",
            "--json-out",
            str(json_empty),
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0, res2.stderr
    payload2 = json.loads(json_empty.read_text(encoding="utf-8"))
    verdicts2 = {f["factor_id"]: f["pruning_verdict"] for f in payload2["factors"]}
    assert set(verdicts2.keys()) == {"A", "B"}
    # A still helps strongly on F (KEEP); B is neutral on F with n_T=0 (UNTESTED)
    assert verdicts2["A"] == "KEEP"
    assert verdicts2["B"] == "UNTESTED"


def test_cli_auto_detect_skips_boolean_columns():
    from prompt_prism.cli.main import _auto_detect_binary_factors

    df = pd.DataFrame(
        {
            "run_id": [0, 1],
            "sample_id": ["s0", "s0"],
            "A": [0, 1],
            "is_synthetic": [True, False],
            "mixed_bool": pd.Series([True, None], dtype=object),
            "score": [0.5, 0.8],
        }
    )
    assert _auto_detect_binary_factors(df, "score") == ["A"]


def test_experiment_omitted_targets_factors_mutation_isolation_and_alias_collision():
    # Intentionally swap names/IDs to verify alias_to_col never shadows a real column ID
    factors = [
        Factor.binary("B", id="A", level_0_content="", level_1_content="RuleA"),
        Factor.binary("A", id="B", level_0_content="", level_1_content="RuleB"),
    ]
    exp = Experiment.from_factors(
        factors=factors,
        system_prompt="Task.",
        data_template="{{ text }}",
        metrics=[ExactMatch()],
        target_metric="exact_match",
    )

    shared_tags = ["A"]
    # Baseline items omit `targets_factors` entirely; targeted items include it
    dataset = [{"id": f"b_{i}", "text": "base", "target": "ok"} for i in range(10)] + [
        {
            "id": f"t_{i}",
            "text": "edge",
            "target": "edge_ok",
            "targets_factors": shared_tags,
        }
        for i in range(10)
    ]

    def client(p):
        s = str(p)
        if "edge" in s:
            return "edge_ok" if "RuleA" in s else "wrong"
        return "ok"

    exp.run(dataset=dataset, client=client, max_workers=1)
    # Mutating caller's tag list post-run must not mutate stored Trial.metadata
    shared_tags.append("B")

    audit = exp.diagnose(
        baseline_sample_ids=[f"b_{i}" for i in range(10)],
        min_targeted_items=10,
        equivalence_margin=0.02,
        n_sign_flip=0,
    )
    res = _by_factor(audit)
    assert res["A"].n_targeted == 10
    assert res["A"].pruning_verdict == "KEEP_TARGETED"
    assert res["B"].n_targeted == 0
    assert res["B"].pruning_verdict == "UNTESTED"

    # Bare str in Experiment.diagnose(coverage=...) also raises TypeError
    with pytest.raises(TypeError, match="bare str"):
        exp.diagnose(coverage={"A": "t_0"}, n_sign_flip=0)
