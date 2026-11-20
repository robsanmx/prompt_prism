"""
Unit, synthetic-statistical, empirical-invariant, and fixed-seed replay tests
for Phase 0 Task R1 (`research/hf_multi_prompt/effect_reliability.py`).

Covers:
1. Orthogonal Fourier basis, exact analytical item-cluster variance, item-bootstrap
   equivalence, paired two-way cluster bootstrap weights, and exact weighted Spearman rho.
2. Synthetic pure-noise cubes: near-zero split-half reliability, ~50% sign agreement,
   and near-zero noise-corrected RMS within Monte Carlo tolerance.
3. Synthetic known-signal hierarchical cubes: accurate recovery of true signal RMS
   across orders 1..5 after item-sampling noise correction.
4. Generated artifact schema, Resolution-V / Resolution-IV aliasing identities,
   cross-consistency with `headline_summary.json`, and effect-hierarchy audit fields.
5. Deterministic fixed-seed replay reproducing exact SHA-256 hashes of
   `effect_reliability.csv` and `effect_reliability_summary.json`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research.hf_multi_prompt.effect_reliability import (
    ESTIMATOR_ORDER,
    _bootstrap_pooled_spearman_single_split,
    _build_panel_bootstrap_weights,
    build_order_contrast_matrices,
    compute_cube_contrast_noise_and_rms,
    compute_cube_split_half_reliability,
    evaluate_panel_cubes_reliability,
    run_effect_reliability_analysis,
)
from research.hf_multi_prompt.run_experiments import (
    DATA_DIR,
    RESULTS_DIR,
    generate_item_splits,
    panel_cluster_bootstrap_ci,
    sha256_file,
    two_way_cluster_bootstrap_ci,
)

# ==============================================================================
# 1. Core Mathematical & Bootstrap Equivalence Unit Tests
# ==============================================================================


def test_order_contrast_matrices_orthogonality_and_dimensions():
    m5 = build_order_contrast_matrices(5)
    assert [m5[o][0].shape for o in range(1, 6)] == [
        (32, 5),
        (32, 10),
        (32, 10),
        (32, 5),
        (32, 1),
    ]
    assert m5[1][1] == ["A", "B", "C", "D", "E"]
    assert m5[5][1] == ["ABCDE"]

    X5 = np.column_stack([m5[o][0] for o in range(1, 6)])
    assert np.array_equal(X5.T @ X5, 32 * np.eye(31, dtype=int))

    m4 = build_order_contrast_matrices(4)
    assert [m4[o][0].shape for o in range(1, 5)] == [
        (16, 4),
        (16, 6),
        (16, 4),
        (16, 1),
    ]
    assert m4[4][1] == ["ABCD"]
    X4 = np.column_stack([m4[o][0] for o in range(1, 5)])
    assert np.array_equal(X4.T @ X4, 16 * np.eye(15, dtype=int))


def test_analytical_item_cluster_variance_and_item_bootstrap_match():
    rng = np.random.default_rng(20261199)
    K, N = 32, 500
    item_p = rng.beta(2.5, 2.5, size=(1, N))
    cell_shift = rng.normal(0.0, 0.04, size=(K, 1))
    probs = np.clip(item_p + cell_shift, 0.02, 0.98)
    Y = rng.binomial(1, probs).astype(float)

    res = compute_cube_contrast_noise_and_rms(
        Y, num_factors=5, n_item_boot=4000, boot_rng=np.random.default_rng(77)
    )
    order_mats = build_order_contrast_matrices(5)

    for o in range(1, 6):
        X_o, _ = order_mats[o]
        C_manual = (X_o.T @ Y) / 16.0
        tau_manual = C_manual.mean(axis=1)
        var_manual = C_manual.var(axis=1, ddof=1) / float(N)

        r_o = res[o]
        assert np.allclose(r_o["taus"], tau_manual, atol=1e-15)
        assert np.allclose(r_o["vars_analytical"], var_manual, atol=1e-15)
        assert np.isclose(
            r_o["rms_obs"] ** 2 - r_o["rms_noise"] ** 2,
            r_o["sigma2_signal_unclipped"],
            atol=1e-15,
        )
        # Bessel-corrected item bootstrap converges to exact analytical cluster variance
        assert np.isclose(
            r_o["rms_noise_item_boot_bessel"], r_o["rms_noise"], rtol=0.035
        )


def test_panel_bootstrap_weights_match_run_experiments_helpers():
    rng = np.random.default_rng(1234)

    # 1. 1D model vector (MMLU_POOLED)
    vec_1d = rng.normal(0.05, 0.01, size=15)
    meta_1d = pd.DataFrame(
        [{"model_id": f"m_{i:02d}", "task_key": "pool"} for i in range(15)]
    )
    W_1d, _ = _build_panel_bootstrap_weights(meta_1d, "1d_model", n_boot=300, seed=901)
    boot_1d = (W_1d @ vec_1d) / W_1d.sum(axis=1)
    ref_1d = two_way_cluster_bootstrap_ci(vec_1d, n_boot=300, seed=901)
    assert np.isclose(np.percentile(boot_1d, 2.5), ref_1d["ci_low"], atol=1e-14)
    assert np.isclose(np.percentile(boot_1d, 97.5), ref_1d["ci_high"], atol=1e-14)

    # 2. 2D balanced panel (MMLU_TASK)
    mat_2d = rng.normal(0.02, 0.005, size=(6, 9))
    meta_2d = pd.DataFrame(
        [
            {"model_id": f"m_{i:02d}", "task_key": f"t_{j:02d}"}
            for i in range(6)
            for j in range(9)
        ]
    )
    W_2d, _ = _build_panel_bootstrap_weights(
        meta_2d, "2d_balanced", n_boot=300, seed=902
    )
    boot_2d = (W_2d @ mat_2d.ravel()) / W_2d.sum(axis=1)
    ref_2d = two_way_cluster_bootstrap_ci(mat_2d, n_boot=300, seed=902)
    assert np.isclose(np.percentile(boot_2d, 2.5), ref_2d["ci_low"], atol=1e-14)
    assert np.isclose(np.percentile(boot_2d, 97.5), ref_2d["ci_high"], atol=1e-14)

    # 3. 2D unbalanced shared-model panel (SLAB)
    panel_rows = []
    for bench, m_count in [("BBH", 4), ("LMentry", 6)]:
        for t_i in range(3):
            for m_i in range(m_count):
                panel_rows.append(
                    {
                        "cohort": bench,
                        "model_id": f"model_{m_i:02d}",
                        "task": f"task_{t_i:02d}",
                        "task_key": f"{bench}/task_{t_i:02d}",
                        "val": float(rng.normal(0.1, 0.03)),
                    }
                )
    df_unb = pd.DataFrame(panel_rows)
    W_unb, _ = _build_panel_bootstrap_weights(
        df_unb, "2d_unbalanced", n_boot=300, seed=903
    )
    boot_unb = (W_unb @ df_unb["val"].to_numpy()) / W_unb.sum(axis=1)
    ref_unb = panel_cluster_bootstrap_ci(df_unb, value_col="val", n_boot=300, seed=903)
    assert np.isclose(np.percentile(boot_unb, 2.5), ref_unb["ci_low"], atol=1e-14)
    assert np.isclose(np.percentile(boot_unb, 97.5), ref_unb["ci_high"], atol=1e-14)


def test_vectorized_weighted_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(555)
    C, O, B = 35, 6, 40
    x_co = np.round(rng.normal(0.0, 1.0, size=(C, O)), 1)
    y_co = np.round(0.35 * x_co + rng.normal(0.0, 1.0, size=(C, O)), 1)
    x_co[0, :] = 0.0  # Explicit tied zeros
    W_cube = rng.integers(0, 4, size=(B, C)).astype(float)
    W_cube[:, 0] = np.maximum(1.0, W_cube[:, 0])

    fast_rho = _bootstrap_pooled_spearman_single_split(
        x_co, y_co, W_cube, batch_size=15
    )
    exact_rho = np.array(
        [
            stats.spearmanr(
                np.repeat(x_co.ravel(), np.repeat(W_cube[b].astype(int), O)),
                np.repeat(y_co.ravel(), np.repeat(W_cube[b].astype(int), O)),
            )[0]
            for b in range(B)
        ],
        dtype=float,
    )
    assert np.allclose(fast_rho, exact_rho, atol=1e-14)


# ==============================================================================
# 2. Synthetic Pure-Noise & Known-Signal Hierarchical Cubes
# ==============================================================================


def test_synthetic_pure_noise_cubes_recover_zero_reliability_and_zero_signal():
    rng = np.random.default_rng(20261201)
    M, T = 6, 10
    C_total = M * T
    K, N = 32, 240

    d_masks, c_masks, _ = generate_item_splits(N, n_splits=10, seed=7001)
    cube_mats = []
    for _ in range(C_total):
        # Item-difficulty heterogeneity + pure zero-contrast Bernoulli cell noise
        p_item = rng.uniform(0.20, 0.80, size=(1, N))
        Y_cube = rng.binomial(1, np.broadcast_to(p_item, (K, N))).astype(float)
        cube_mats.append(Y_cube)

    meta = pd.DataFrame(
        [
            {
                "cohort": "SYNTH_NOISE",
                "model_id": f"m_{i}",
                "task": f"t_{j}",
                "task_key": f"t_{j}",
            }
            for i in range(M)
            for j in range(T)
        ]
    )
    _, summary = evaluate_panel_cubes_reliability(
        cohort_label="SYNTH_NOISE",
        estimator_name="legacy_first",
        num_factors=5,
        cube_matrices=cube_mats,
        cube_d_masks=[d_masks for _ in range(C_total)],
        cube_c_masks=[c_masks for _ in range(C_total)],
        cube_meta=meta,
        panel_mode="2d_balanced",
        n_boot_cluster=300,
        cluster_seed=8001,
    )

    s20 = summary["order_reliability"]["splits20"]
    full = summary["order_reliability"]["full_cube"]
    for order in range(1, 6):
        rec_s20 = s20[f"order_{order}"]
        rec_full = full[f"order_{order}"]

        # Raw observed RMS is strictly positive (~0.010 on splits, ~0.007 on full), matching noise floor
        assert rec_s20["cohort_rms_obs"] > 0.008
        assert np.isclose(
            rec_s20["cohort_rms_obs"], rec_s20["cohort_rms_noise"], rtol=0.15
        )
        assert rec_s20["noise_variance_share"] > 0.80

        # Split-half reliability is near zero across every order 1..5
        assert abs(rec_s20["pooled_pearson_r"]) < 0.08
        assert abs(rec_s20["pooled_spearman_rho"]) < 0.08
        assert abs(rec_s20["sign_agreement_overall"] - 0.50) < 0.05

        # Unclipped signal variance is centered at 0 and noise-corrected cohort RMS is near zero
        assert rec_full["cohort_rms_corr"] < 0.0035
        assert rec_s20["cohort_rms_corr"] < 0.0045
        assert rec_s20["frac_cube_splits_truncated_zero"] > 0.35


def test_synthetic_hierarchical_known_signal_recovers_true_rms_orders_1_to_5():
    rng = np.random.default_rng(20261202)
    M, T = 6, 8
    C_total = M * T
    K, N = 32, 400
    order_mats = build_order_contrast_matrices(5)

    # Strict hierarchical ground-truth signal across orders 1..5
    target_order_sd = {1: 0.080, 2: 0.050, 3: 0.030, 4: 0.016, 5: 0.008}
    true_ms_by_order = {o: [] for o in range(1, 6)}

    d_masks, c_masks, _ = generate_item_splits(N, n_splits=10, seed=7002)
    cube_mats = []
    for _ in range(C_total):
        cell_signal = np.zeros(K, dtype=float)
        for o in range(1, 6):
            X_o, _ = order_mats[o]
            n_o = X_o.shape[1]
            # Draw contrasts and normalize per-cube RMS to exact target_order_sd[o] with random signs/orientations
            raw = rng.normal(0.0, 1.0, size=n_o)
            tau_true = raw * (target_order_sd[o] / np.sqrt(np.mean(raw**2)))
            true_ms_by_order[o].append(float(np.mean(tau_true**2)))
            cell_signal = cell_signal + 0.5 * (X_o @ tau_true)

        item_effect = rng.normal(0.50, 0.15, size=(1, N))
        noise = rng.normal(0.0, 0.32, size=(K, N))
        Y_cube = item_effect + cell_signal[:, None] + noise
        cube_mats.append(Y_cube)

    # Also check single-cube split-half helper on cube 0
    single_rel = compute_cube_split_half_reliability(
        cube_mats[0][:, ::2], cube_mats[0][:, 1::2], num_factors=5
    )
    assert single_rel[1]["within_cube_pearson_r"] > 0.85
    assert single_rel[1]["sign_agreement_gt_2se"] >= 0.95

    meta = pd.DataFrame(
        [
            {
                "cohort": "SYNTH_HIER",
                "model_id": f"m_{i}",
                "task": f"t_{j}",
                "task_key": f"t_{j}",
            }
            for i in range(M)
            for j in range(T)
        ]
    )
    _, summary = evaluate_panel_cubes_reliability(
        cohort_label="SYNTH_HIER",
        estimator_name="legacy_first",
        num_factors=5,
        cube_matrices=cube_mats,
        cube_d_masks=[d_masks for _ in range(C_total)],
        cube_c_masks=[c_masks for _ in range(C_total)],
        cube_meta=meta,
        panel_mode="2d_balanced",
        n_boot_cluster=300,
        cluster_seed=8002,
    )

    full = summary["order_reliability"]["full_cube"]
    s20 = summary["order_reliability"]["splits20"]

    recovered_full = []
    recovered_s20 = []
    for o in range(1, 6):
        true_rms = float(np.sqrt(np.mean(true_ms_by_order[o])))
        rec_f = full[f"order_{o}"]["cohort_rms_corr"]
        rec_s = s20[f"order_{o}"]["cohort_rms_corr"]
        obs_s = s20[f"order_{o}"]["cohort_rms_obs"]
        recovered_full.append(rec_f)
        recovered_s20.append(rec_s)

        # Noise-corrected RMS accurately recovers true hierarchical signal across ALL orders 1..5
        assert np.isclose(
            rec_f, true_rms, atol=0.0020
        ), f"Order {o} full: {rec_f} vs {true_rms}"
        assert np.isclose(
            rec_s, true_rms, atol=0.0025
        ), f"Order {o} s20: {rec_s} vs {true_rms}"
        # On weak high-order terms (orders 4, 5), uncorrected RMS_obs is materially upward-biased
        if o >= 4:
            assert abs(rec_s - true_rms) < abs(obs_s - true_rms)

    # Strict monotonic hierarchy recovered across orders 1 > 2 > 3 > 4 > 5
    assert all(recovered_full[i] > recovered_full[i + 1] for i in range(4))
    assert all(recovered_s20[i] > recovered_s20[i + 1] for i in range(4))

    # Split-half reliability decreases monotonically from strong Order 1 to weak Order 5
    r_vals = [s20[f"order_{o}"]["pooled_pearson_r"] for o in range(1, 6)]
    assert r_vals[0] > r_vals[1] > r_vals[2] > r_vals[3] > r_vals[4] > 0.25


# ==============================================================================
# 3. Empirical Artifact Invariants, Aliasing Identities, & Hierarchy Verdict
# ==============================================================================


def test_generated_artifacts_invariants_aliasing_and_hierarchy_audit():
    csv_path = RESULTS_DIR / "effect_reliability.csv"
    json_path = RESULTS_DIR / "effect_reliability_summary.json"
    assert csv_path.exists(), "effect_reliability.csv missing"
    assert json_path.exists(), "effect_reliability_summary.json missing"

    df = pd.read_csv(csv_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    headline = json.loads(
        (RESULTS_DIR / "headline_summary.json").read_text(encoding="utf-8")
    )

    assert set(df["cohort"].unique()) == {"MMLU_TASK", "MMLU_POOLED", "SLAB"}
    assert set(df["estimator"].unique()) == set(ESTIMATOR_ORDER)
    assert set(df["record_type"].unique()) == {"order_reliability", "design_esr"}

    # Check cube counts
    assert set(df[df["cohort"] == "MMLU_TASK"]["n_cubes"]) == {855}
    assert set(df[df["cohort"] == "MMLU_POOLED"]["n_cubes"]) == {15}
    assert set(df[df["cohort"] == "SLAB"]["n_cubes"]) == {178}

    # Exact Pythagorean variance identity across every row
    ord_rows = df[df["record_type"] == "order_reliability"]
    assert np.allclose(
        ord_rows["cohort_rms_obs"] ** 2 - ord_rows["cohort_rms_noise"] ** 2,
        ord_rows["mean_sigma2_signal_unclipped"],
        atol=1e-12,
    )
    assert (ord_rows["cohort_rms_corr"] <= ord_rows["cohort_rms_obs"] + 1e-12).all()
    assert (
        ord_rows["mean_cube_rms_corr"] <= ord_rows["mean_cube_rms_obs"] + 1e-12
    ).all()

    # Cross-consistency with `headline_summary.json`
    s1_leg = headline["cohort1_mmlu_legacy_first"]
    s2_leg = headline["cohort2_slab_legacy_first"]
    mt_s0 = summary["cohorts"]["MMLU_TASK"]["legacy_first"]["order_reliability"][
        "split0"
    ]
    sl_s0 = summary["cohorts"]["SLAB"]["legacy_first"]["order_reliability"]["split0"]

    assert np.isclose(
        mt_s0["order_1"]["mean_cube_rms_obs"],
        s1_leg["mean_disc0_rms_main_true"],
        atol=1e-12,
    )
    assert np.isclose(
        mt_s0["order_2"]["mean_cube_rms_obs"],
        s1_leg["mean_disc0_rms_2fi_true"],
        atol=1e-12,
    )
    assert np.isclose(
        mt_s0["order_3"]["mean_cube_rms_obs"],
        s1_leg["mean_disc0_rms_3fi_true"],
        atol=1e-12,
    )
    assert np.isclose(
        mt_s0["order_4"]["mean_cube_rms_obs"],
        s1_leg["mean_disc0_rms_4fi_true"],
        atol=1e-12,
    )
    assert np.isclose(
        sl_s0["order_1"]["mean_cube_rms_obs"],
        s2_leg["mean_disc0_rms_main_true"],
        atol=1e-12,
    )

    # Exact Resolution-V and Resolution-IV aliasing identities on Split 0 discovery
    mt_esr_s0 = summary["cohorts"]["MMLU_TASK"]["legacy_first"][
        "screening_esr_and_heldout_prediction"
    ]["split0"]
    sl_esr_s0 = summary["cohorts"]["SLAB"]["legacy_first"][
        "screening_esr_and_heldout_prediction"
    ]["split0"]
    assert np.isclose(
        mt_esr_s0["order_1"]["2(5-1)V_coset_avg"]["mean_cube_attr_rmse"],
        mt_s0["order_4"]["mean_cube_rms_obs"],
        atol=1e-12,
    )
    assert np.isclose(
        mt_esr_s0["order_2"]["2(5-1)V_coset_avg"]["mean_cube_attr_rmse"],
        mt_s0["order_3"]["mean_cube_rms_obs"],
        atol=1e-12,
    )
    assert np.isclose(
        sl_esr_s0["order_1"]["2(4-1)IV_coset_avg"]["mean_cube_attr_rmse"],
        sl_s0["order_3"]["mean_cube_rms_obs"],
        atol=1e-12,
    )

    # Item bootstrap verification agrees tightly with analytical variance across all 3 cohorts
    for key in (
        "MMLU_TASK_legacy_first",
        "MMLU_POOLED_legacy_first",
        "SLAB_legacy_first",
    ):
        ib = summary["item_bootstrap_verification"][key]["full_cube"]["order_1"]
        assert ib["pearson_r_cube_rms_noise_analytical_vs_bessel_boot"] > 0.99
        assert ib["median_rel_abs_diff_bessel_vs_analytical"] < 0.05

    # Effect hierarchy audit assertions
    audit = summary["effect_hierarchy_audit"]
    assert (
        audit["apparent_lack_of_hierarchy_survives_noise_floor_on_single_template"]
        is True
    )
    assert (
        audit["strict_monotonic_hierarchy_recovered_under_stratum_averaging_on_slab"]
        is True
    )
    assert audit["strict_monotonic_hierarchy_present_on_single_template_slab"] is False

    mt_nshare = audit["regime_comparison"]["mmlu_per_task_legacy_first_split0"][
        "noise_variance_share"
    ]
    assert mt_nshare["order_2"] > 0.75
    assert mt_nshare["order_3"] > 0.75

    mp_r = audit["regime_comparison"]["mmlu_pooled_14042_legacy_first_splits20"][
        "pooled_pearson_r"
    ]
    assert all(mp_r[f"order_{o}"] > 0.85 for o in range(1, 5))
    assert mp_r["order_5"] > 0.75

    # Screening ESR & held-out contrast prediction ordering across designs (20 splits)
    mt_esr_20 = summary["cohorts"]["MMLU_TASK"]["legacy_first"][
        "screening_esr_and_heldout_prediction"
    ]["splits20"]
    assert (
        mt_esr_20["order_1"]["full_cube"]["mean_cube_heldout_pred_rmse"]
        < mt_esr_20["order_1"]["2(5-1)V_coset_avg"]["mean_cube_heldout_pred_rmse"]
        < mt_esr_20["order_1"]["rand16_fullrank"]["mean_cube_heldout_pred_rmse"]
        < mt_esr_20["order_1"]["2(5-2)III_all60"]["mean_cube_heldout_pred_rmse"]
        < mt_esr_20["order_1"]["rand8_fullrank"]["mean_cube_heldout_pred_rmse"]
    )
    assert (
        mt_esr_20["order_2"]["full_cube"]["mean_cube_heldout_pred_rmse"]
        < mt_esr_20["order_2"]["2(5-1)V_coset_avg"]["mean_cube_heldout_pred_rmse"]
        < mt_esr_20["order_2"]["rand16_fullrank"]["mean_cube_heldout_pred_rmse"]
    )
    sl_esr_20 = summary["cohorts"]["SLAB"]["legacy_first"][
        "screening_esr_and_heldout_prediction"
    ]["splits20"]
    assert (
        sl_esr_20["order_1"]["full_cube"]["mean_cube_heldout_pred_rmse"]
        < sl_esr_20["order_1"]["2(4-1)IV_coset_avg"]["mean_cube_heldout_pred_rmse"]
        < sl_esr_20["order_1"]["rand8_fullrank"]["mean_cube_heldout_pred_rmse"]
    )


# ==============================================================================
# 4. Fixed-Seed Deterministic Replay Hash Verification
# ==============================================================================


@pytest.mark.skipif(
    not (DATA_DIR / "Ys.pickle").exists(),
    reason="Requires gitignored local PromptEval Ys.pickle dataset",
)
def test_fixed_seed_replay_reproduces_csv_and_json_sha256():
    expected_csv_sha = sha256_file(RESULTS_DIR / "effect_reliability.csv")
    expected_json_sha = sha256_file(RESULTS_DIR / "effect_reliability_summary.json")

    _, replay_summary, replay_csv_sha, replay_json_sha = (
        run_effect_reliability_analysis(
            n_boot_cluster=1000,
            n_item_boot_verify=500,
            write_outputs=False,
        )
    )
    assert replay_csv_sha == expected_csv_sha
    assert replay_json_sha == expected_json_sha
    assert (
        replay_summary["output_artifacts"]["effect_reliability_csv_sha256"]
        == expected_csv_sha
    )
