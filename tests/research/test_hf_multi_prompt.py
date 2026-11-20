"""
Focused unit, synthetic-statistical, and regression test suite for the revised
Hugging Face Multi-Prompt Factorial Study (`research/hf_multi_prompt/run_experiments.py`).

Covers all verification criteria in `tasks/paper_hf_revision_plan.md` and
`tasks/paper_hf_revision_spec.md` (Section 13).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.hf_multi_prompt.run_experiments import (
    DATA_DIR,
    EXPECTED_MMLU_META_SHA256,
    EXPECTED_YS_SHA256,
    FIGURES_DIR,
    RESULTS_DIR,
    SEEDS,
    TABLES_DIR,
    _two_way_cluster_bootstrap_correlation,
    build_cube_design_matrices,
    build_mmlu_mapping_table,
    build_multiplicity_table,
    build_slab_mapping_table_for_task,
    compute_all_reference_cells_recovery,
    compute_cube_rcbd_and_robust_inference,
    compute_oracle_uplift_recovery_stats,
    compute_tost_paired_cluster,
    concatenate_task_discovery_items,
    decompose_full_cube_effects,
    enumerate_fractional_cosets,
    extract_cell_item_matrix,
    extract_unique_cell_map,
    fit_design_ols,
    generate_balanced_random_designs,
    generate_item_splits,
    generate_random_designs_with_diagnostics,
    panel_cluster_bootstrap_ci,
    select_with_tie_policy,
    sha256_file,
    two_way_cluster_bootstrap_ci,
    validate_design_matrix,
    verify_source_hashes,
)

# ==============================================================================
# 1. Unit Tests: Rank & Condition Safeguards, Cosets, Splits, Oracle Uplift
# ==============================================================================


def test_validate_design_matrix_full_rank_and_rank_deficient():
    m5 = build_cube_design_matrices(5)
    D_2fi = m5["D_2fi"]
    cosets = enumerate_fractional_cosets()
    res5_pos = cosets["res5_cosets"]["2(5-1)V_coset_pos"]["indices"]

    # Orthogonal 2^(5-1)_V design is rank 16 with condition number 1.0
    rank, cond = validate_design_matrix(D_2fi[res5_pos], "2(5-1)V_pos", "test_2fi")
    assert rank == 16
    assert np.isclose(cond, 1.0)

    # Deliberately rank-deficient 16-run design (duplicate a row or use legacy rank-deficient subset)
    legacy_r16 = generate_balanced_random_designs(
        32,
        5,
        16,
        num_screens=20,
        seed=SEEDS["rand16_legacy"],
        require_2fi_full_rank=False,
    )
    rank_def_cand = next(c for c in legacy_r16 if np.linalg.matrix_rank(D_2fi[c]) < 16)
    with pytest.raises(ValueError, match="rank deficient"):
        validate_design_matrix(D_2fi[rank_def_cand], "legacy_rank_def", "test_2fi")

    with pytest.raises(ValueError, match="rank deficient"):
        fit_design_ols(
            D_2fi[rank_def_cand], np.zeros(16), "legacy_rank_def", "test_2fi"
        )

    # Ill-conditioned design rejected when max_condition_number threshold is exceeded
    full_rank_r16 = generate_balanced_random_designs(
        32,
        5,
        16,
        num_screens=10,
        seed=SEEDS["rand16_full_rank"],
        require_2fi_full_rank=True,
    )
    with pytest.raises(ValueError, match="exceeds maximum condition number"):
        validate_design_matrix(
            D_2fi[full_rank_r16[0]], "rand16_0", "test_2fi", max_condition_number=1.5
        )


def test_random_design_generation_diagnostics():
    accepted, diag_rows, stats_sum = generate_random_designs_with_diagnostics(
        5, 16, 25, SEEDS["rand16_full_rank"], True, "MMLU", "test_rand16"
    )
    assert len(accepted) == 25
    assert stats_sum["num_accepted"] == 25
    assert stats_sum["rejected_rank_deficient_balanced"] > 0
    m5 = build_cube_design_matrices(5)
    for c in accepted:
        assert np.linalg.matrix_rank(m5["D_2fi"][c]) == 16
        assert np.all(m5["X_01"][c].sum(axis=0) == 8)


def test_coset_enumeration_partitions_full_cubes():
    cosets = enumerate_fractional_cosets()

    # 2^(5-1)_V cosets partition {0..31} into two disjoint 16-run fractions
    pos5 = set(cosets["res5_cosets"]["2(5-1)V_coset_pos"]["indices"])
    neg5 = set(cosets["res5_cosets"]["2(5-1)V_coset_neg"]["indices"])
    assert len(pos5) == 16 and len(neg5) == 16
    assert pos5.isdisjoint(neg5)
    assert pos5 | neg5 == set(range(32))

    # 2^(5-2)_III 4 canonical cosets partition {0..31} into four disjoint 8-run fractions
    c_sets = [set(v["indices"]) for v in cosets["res3_canonical_cosets"].values()]
    assert len(c_sets) == 4
    for s in c_sets:
        assert len(s) == 8
    assert set().union(*c_sets) == set(range(32))
    for i in range(4):
        for j in range(i + 1, 4):
            assert c_sets[i].isdisjoint(c_sets[j])

    # All 60 regular 2^(5-2)_III designs have rank 6 for main effects
    m5 = build_cube_design_matrices(5)
    assert len(cosets["res3_all_60"]) == 60
    for did, info in cosets["res3_all_60"].items():
        rk, _ = validate_design_matrix(m5["D_main"][info["indices"]], did, "main")
        assert rk == 6

    # 2^(4-1)_IV cosets partition {0..15} into two disjoint 8-run fractions
    pos4 = set(cosets["res4_cosets"]["2(4-1)IV_coset_pos"]["indices"])
    neg4 = set(cosets["res4_cosets"]["2(4-1)IV_coset_neg"]["indices"])
    assert len(pos4) == 8 and len(neg4) == 8
    assert pos4.isdisjoint(neg4)
    assert pos4 | neg4 == set(range(16))


def test_item_splits_reproducibility_and_partitioning():
    d1, c1, ids1 = generate_item_splits(101, n_splits=20, seed=12345)
    d2, c2, ids2 = generate_item_splits(101, n_splits=20, seed=12345)
    assert ids1 == ids2
    assert len(ids1) == 21
    assert ids1[0] == "legacy_even_odd"
    assert np.array_equal(d1, d2) and np.array_equal(c1, c2)

    # Every split has disjoint discovery and confirmation items and normalized weights summing to 1
    assert np.allclose(d1.sum(axis=0), 1.0)
    assert np.allclose(c1.sum(axis=0), 1.0)
    assert np.allclose(np.sum(d1 * c1, axis=0), 0.0)


def test_oracle_uplift_edge_cases():
    base = np.array([0.50, 0.60, 0.70, 0.80])
    oracle = np.array(
        [0.60, 0.60, 0.65, 0.90]
    )  # cube 1 has 0 uplift, cube 2 has negative uplift
    method = np.array([0.58, 0.59, 0.68, 0.88])
    res = compute_oracle_uplift_recovery_stats(method, base, oracle)
    assert res["n_total_cubes"] == 4
    assert res["n_positive_uplift_cubes"] == 2
    assert res["n_nonpositive_uplift_cubes"] == 2
    assert np.isclose(
        res["ratio_of_means_pct"],
        (method.mean() - base.mean()) / (oracle.mean() - base.mean()) * 100.0,
    )

    # Zero mean denominator raises explicit ValueError
    with pytest.raises(ValueError, match="denominator is zero"):
        compute_oracle_uplift_recovery_stats(method, base, base.copy())


def test_template_mapping_catalog_order_invariance():
    df_mmlu = pd.read_csv(DATA_DIR / "templates" / "MMLU" / "abstract_algebra.csv")
    df_mmlu["original_index"] = np.arange(len(df_mmlu))
    meta = json.loads(
        (DATA_DIR / "templates" / "mmlu_templates_metadata.json").read_text()
    )

    _, _, canon_map_1 = build_mmlu_mapping_table(df_mmlu, meta)
    df_shuffled = df_mmlu.sample(frac=1.0, random_state=999).reset_index(drop=True)
    _, _, canon_map_2 = build_mmlu_mapping_table(df_shuffled, meta)

    assert canon_map_1 == canon_map_2

    # Verify extracted cell_mean and canonical_first matrices are strictly identical after catalog shuffling
    rng = np.random.default_rng(42)
    dummy_raw = rng.binomial(1, 0.5, size=(100, 50)).astype(float)
    all_cells = sorted(canon_map_1.keys())
    Y_cell_1 = extract_cell_item_matrix(
        dummy_raw, all_cells, canon_map_1, estimator="cell_mean"
    )
    Y_cell_2 = extract_cell_item_matrix(
        dummy_raw, all_cells, canon_map_2, estimator="cell_mean"
    )
    assert np.allclose(Y_cell_1, Y_cell_2)

    Y_can_1 = extract_cell_item_matrix(
        dummy_raw, all_cells, canon_map_1, estimator="canonical_first"
    )
    Y_can_2 = extract_cell_item_matrix(
        dummy_raw, all_cells, canon_map_2, estimator="canonical_first"
    )
    assert np.array_equal(Y_can_1, Y_can_2)

    # Also test SLAB task catalog order invariance
    df_slab = pd.read_csv(DATA_DIR / "templates" / "BBH" / "causal_judgement.csv")
    df_slab["original_index"] = np.arange(len(df_slab))
    _, _, slab_canon_1 = build_slab_mapping_table_for_task(
        df_slab, "BBH", "causal_judgement"
    )
    df_slab_shuf = df_slab.sample(frac=1.0, random_state=777).reset_index(drop=True)
    _, _, slab_canon_2 = build_slab_mapping_table_for_task(
        df_slab_shuf, "BBH", "causal_judgement"
    )
    assert slab_canon_1 == slab_canon_2


# ==============================================================================
# 2. Synthetic Statistical Tests: Exact Recovery, Aliasing, Coverage, RCBD
# ==============================================================================


def test_synthetic_noiseless_2fi_recovery_and_exact_aliasing():
    m5 = build_cube_design_matrices(5)
    X_pm = m5["X_pm"]
    X_2fi = m5["X_2fi"]
    X_3fi = m5["higher_orders"][3]
    X_4fi = m5["higher_orders"][4]
    D_2fi = m5["D_2fi"]

    rng = np.random.default_rng(2026)
    true_tau_main = rng.normal(0.0, 0.05, size=5)
    true_tau_2fi = rng.normal(0.0, 0.03, size=10)

    # Case 1: Pure main + 2FI response (zero higher-order effects, zero noise)
    y_pure = 0.60 + 0.5 * (X_pm @ true_tau_main) + 0.5 * (X_2fi @ true_tau_2fi)

    cosets = enumerate_fractional_cosets()
    res5_pos = cosets["res5_cosets"]["2(5-1)V_coset_pos"]["indices"]
    res5_neg = cosets["res5_cosets"]["2(5-1)V_coset_neg"]["indices"]
    full_rank_r16 = generate_balanced_random_designs(
        32,
        5,
        16,
        num_screens=10,
        seed=SEEDS["rand16_full_rank"],
        require_2fi_full_rank=True,
    )

    for idx_set in [res5_pos, res5_neg] + full_rank_r16:
        beta = fit_design_ols(
            D_2fi[idx_set], y_pure[idx_set], "test_design", "pure_2fi"
        )
        assert np.allclose(2.0 * beta[1:6], true_tau_main, atol=1e-13)
        assert np.allclose(2.0 * beta[6:], true_tau_2fi, atol=1e-13)

    # Case 2: Response with known 3FI and 4FI terms -> verify exact Resolution V aliasing
    true_tau_3fi = rng.normal(0.0, 0.02, size=10)
    true_tau_4fi = rng.normal(0.0, 0.015, size=5)
    y_aliased = y_pure + 0.5 * (X_3fi @ true_tau_3fi) + 0.5 * (X_4fi @ true_tau_4fi)
    decomp = decompose_full_cube_effects(y_aliased, 5)
    assert decomp["reconstruction_max_abs_err"] < 1e-14
    assert np.isclose(decomp["rms_main"], np.sqrt(np.mean(true_tau_main**2)))
    assert np.isclose(decomp["rms_2fi"], np.sqrt(np.mean(true_tau_2fi**2)))
    assert np.isclose(decomp["rms_3fi"], np.sqrt(np.mean(true_tau_3fi**2)))
    assert np.isclose(decomp["rms_4fi"], np.sqrt(np.mean(true_tau_4fi**2)))

    # On BOTH cosets of 2^(5-1)_V, main-effect RMSE equals RMS(4FI) and 2FI RMSE equals RMS(3FI)!
    for idx_set in [res5_pos, res5_neg]:
        beta = fit_design_ols(
            D_2fi[idx_set], y_aliased[idx_set], "res5_alias", "aliased_2fi"
        )
        rmse_main = np.sqrt(np.mean((2.0 * beta[1:6] - true_tau_main) ** 2))
        rmse_2fi = np.sqrt(np.mean((2.0 * beta[6:] - true_tau_2fi) ** 2))
        assert np.isclose(rmse_main, decomp["rms_4fi"], atol=1e-13)
        assert np.isclose(rmse_2fi, decomp["rms_3fi"], atol=1e-13)


def test_synthetic_two_way_cluster_bootstrap_null_coverage():
    rng = np.random.default_rng(10101)
    M, T = 15, 57
    n_trials = 120
    covered = 0
    for trial in range(n_trials):
        # Null data with model random effects + task random effects + cell noise
        model_re = rng.normal(0.0, 0.02, size=(M, 1))
        task_re = rng.normal(0.0, 0.02, size=(1, T))
        eps = rng.normal(0.0, 0.03, size=(M, T))
        Z = model_re + task_re + eps
        ci = two_way_cluster_bootstrap_ci(Z, n_boot=250, seed=5000 + trial, alpha=0.05)
        if ci["ci_low"] <= 0.0 <= ci["ci_high"]:
            covered += 1
    coverage_rate = covered / n_trials
    assert (
        0.87 <= coverage_rate <= 0.99
    ), f"Unexpected two-way cluster bootstrap coverage: {coverage_rate:.3f}"


def test_synthetic_rcbd_and_item_clustered_type1_and_power():
    m4 = build_cube_design_matrices(4)
    X4_pm = m4["X_pm"]
    N = 120
    rng = np.random.default_rng(7777)

    # 1. Synthetic Null with strong item difficulty + within-item correlation across cells (no factor effects)
    n_null_cubes = 100
    p_clust_null = []
    for _ in range(n_null_cubes):
        item_logit = rng.normal(0.0, 1.8, size=(1, N))
        # Shared random sensitivity to a latent nuisance direction inside each item
        latent_dir = X4_pm[:, [0]] * rng.normal(0.0, 0.8, size=(1, N))
        probs = 1.0 / (1.0 + np.exp(-(item_logit + latent_dir)))
        Y_null = rng.binomial(1, probs).astype(float)
        inf = compute_cube_rcbd_and_robust_inference(Y_null, X4_pm)
        # Factor 0 has heterogeneous item-level contrast variance due to latent_dir
        p_clust_null.append(inf["p_cluster"][0])

    type1_cluster = float(np.mean(np.array(p_clust_null) < 0.05))
    assert (
        type1_cluster <= 0.10
    ), f"Item-clustered t-test inflated Type-I error: {type1_cluster:.3f}"

    # 2. Synthetic Known Non-Null Effect (true statistical power simulation!)
    true_power_unb, true_power_rcbd, true_power_clust = 0, 0, 0
    n_alt_cubes = 80
    for _ in range(n_alt_cubes):
        item_logit = rng.normal(0.0, 2.5, size=(1, N))
        # Known true non-null effect on factor A (index 0)
        factor_shift = 0.12 * X4_pm[:, [0]]
        probs = 1.0 / (1.0 + np.exp(-(item_logit + factor_shift)))
        Y_alt = rng.binomial(1, probs).astype(float)
        inf = compute_cube_rcbd_and_robust_inference(Y_alt, X4_pm)
        if inf["p_unblocked"][0] < 0.05:
            true_power_unb += 1
        if inf["p_rcbd"][0] < 0.05:
            true_power_rcbd += 1
        if inf["p_cluster"][0] < 0.05:
            true_power_clust += 1

    assert true_power_clust > true_power_unb * 1.5
    assert true_power_rcbd > true_power_unb * 1.5

    # 3. Missing block / NaN check
    Y_bad = np.ones((16, 10))
    Y_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        compute_cube_rcbd_and_robust_inference(Y_bad, X4_pm)


# ==============================================================================
# 3. Regression & End-to-End Artifact Consistency Tests
# ==============================================================================


@pytest.mark.skipif(
    not (DATA_DIR / "Ys.pickle").exists(),
    reason="Requires gitignored local PromptEval Ys.pickle dataset",
)
def test_source_hash_integrity_and_tamper_rejection():
    hashes = verify_source_hashes()
    assert hashes["Ys_pickle_sha256"] == EXPECTED_YS_SHA256
    assert hashes["mmlu_metadata_sha256"] == EXPECTED_MMLU_META_SHA256

    with pytest.raises(RuntimeError, match="Source integrity failure"):
        verify_source_hashes(expected_ys="0" * 64)


def test_generated_artifacts_and_corrected_headline_values():
    headline_path = RESULTS_DIR / "headline_summary.json"
    manifest_path = RESULTS_DIR / "summary_manifest.json"
    assert (
        headline_path.exists()
    ), "Run research/hf_multi_prompt/run_experiments.py first"
    assert manifest_path.exists()

    headline = json.loads(headline_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    s1_leg = headline["cohort1_mmlu_legacy_first"]
    s2_leg = headline["cohort2_slab_legacy_first"]

    # 1. The regenerated legacy rank-deficient diagnostic is ~23.11x, while the
    # primary full-rank ratio is ~4.09x (~3.36x when well-conditioned).
    assert np.isclose(
        s1_leg["pool_2fi_rmse_ratio_legacy_rankdef_over_res5"], 23.11, atol=0.1
    )
    assert 3.0 <= s1_leg["pool_2fi_rmse_ratio_fullrank_over_res5"] <= 4.5
    assert 3.0 <= s1_leg["pool_2fi_rmse_ratio_wellcond_over_res5"] <= 4.0

    # 2. MMLU half-fraction coset recoveries: primary tie-averaged (`mean_tied`: 88.42% & 90.27%, 2FI: 79.90% & 52.15%)
    # and legacy `first_index` compatibility (~66.19% & ~86.13%, 2FI: ~42.27% on coset-)
    assert np.isclose(
        s1_leg["recovery_split0_res5_best_coset_pos"]["ratio_of_means_pct"],
        88.42,
        atol=0.1,
    )
    assert np.isclose(
        s1_leg["recovery_split0_res5_best_coset_neg"]["ratio_of_means_pct"],
        90.27,
        atol=0.1,
    )
    assert np.isclose(
        s1_leg["recovery_split0_res5_2fi_coset_avg"]["ratio_of_means_pct"],
        66.02,
        atol=0.1,
    )
    f1 = s1_leg["first_index_compatibility"]
    assert np.isclose(
        f1["recovery_split0_res5_best_coset_pos"]["ratio_of_means_pct"], 66.19, atol=0.1
    )
    assert np.isclose(
        f1["recovery_split0_res5_best_coset_neg"]["ratio_of_means_pct"], 86.13, atol=0.1
    )
    assert np.isclose(
        f1["recovery_split0_res5_2fi_coset_neg"]["ratio_of_means_pct"], 42.27, atol=0.1
    )

    # 3. ScreenBest vs unconditioned balanced-random conditional interval spans zero on both cohorts.
    ci_c1 = s1_leg["ci_screenbest_minus_rand16_best_20splits"]
    assert ci_c1["ci_low"] <= 0.0 <= ci_c1["ci_high"]
    ci_c2 = s2_leg["ci_screenbest_minus_rand8_best_20splits"]
    assert ci_c2["ci_low"] <= 0.0 <= ci_c2["ci_high"]

    # 4. Task-level discovery RMS effect magnitudes (~0.0105 main and ~0.0072 2FI on MMLU)
    assert np.isclose(s1_leg["mean_disc0_rms_main_true"], 0.01047, atol=1e-4)
    assert np.isclose(s1_leg["mean_disc0_rms_2fi_true"], 0.00724, atol=1e-4)

    # 5. All required output CSVs, figures, and tables exist and are recorded in manifest hashes
    for key in [
        "design_diagnostics",
        "coset_results",
        "split_selection_results",
        "template_mapping",
        "template_sensitivity",
        "factor_audit",
        "interaction_magnitudes",
        "rcbd_robustness",
        "multiplicity_results",
        "claim_evidence_map",
        "headline_summary",
        "fig1_effect_and_selection_pdf",
        "fig2_models_and_rcbd_pdf",
        "table_mmlu_models_tex",
        "table_cross_cohort_tex",
        "generated_macros_tex",
    ]:
        assert key in manifest["output_artifact_sha256"]


def test_audit_findings_1_to_12_regressions():
    """Verify that all 12 audit findings and V2 tasks are resolved across artifacts."""
    headline = json.loads(
        (RESULTS_DIR / "headline_summary.json").read_text(encoding="utf-8")
    )
    s1_leg = headline["cohort1_mmlu_legacy_first"]
    s2_cell = headline["cohort2_slab_cell_mean"]
    assoc = headline["interaction_regret_associations"]

    # Finding 1, 2 & V2 Task 5: Updated correlations, shared-model cluster bootstrap CIs, and partial correlations
    expected_aligned = {
        "SLAB_legacy_first": 0.47774,
        "SLAB_cell_mean": 0.52578,
        "MMLU_legacy_first": 0.21341,
        "MMLU_cell_mean": 0.25609,
    }
    for key, expected_r in expected_aligned.items():
        entry = assoc[key]
        sensitivity = entry["panel_reweighting_association_sensitivity"]
        assert np.isclose(
            entry["pearson_r_rms2fi_vs_main_regret"], expected_r, atol=1e-4
        )
        assert np.isclose(
            entry["pearson_r_rms2fi_vs_main_regret"],
            sensitivity["mean20_disc_rms2fi_vs_regret_main_20splits"]["pearson_r"],
        )
        assert (
            sensitivity["mean20_disc_rms2fi_vs_regret_main_20splits"][
                "confirmatory_null_test"
            ]
            is False
        )
    assert (
        assoc["SLAB_legacy_first"]["n_unique_models"] == 16
    )  # 11 shared models coupled across BBH & LMentry
    for k in [
        "SLAB_legacy_first",
        "SLAB_cell_mean",
        "MMLU_legacy_first",
        "MMLU_cell_mean",
    ]:
        assert "pearson_se_cluster_rms2fi_vs_main_regret" in assoc[k]
        assert (
            assoc[k]["pearson_ci_low_rms2fi_vs_main_regret"]
            < assoc[k]["pearson_r_rms2fi_vs_main_regret"]
        )
        assert (
            assoc[k]["pearson_r_rms2fi_vs_main_regret"]
            < assoc[k]["pearson_ci_high_rms2fi_vs_main_regret"]
        )
        assert "partial_pearson_r_rms2fi_vs_main_regret_given_rms_main" in assoc[k]

    # Finding 3 & V2 Task 7: TOST margins and Cohort-2 cell_mean orthogonal advantage (+0.206 pp vs unconditioned Rand-8, CI > 0)
    ci_c1_leg = s1_leg["ci_screenbest_minus_rand16_best_20splits"]
    assert "ci90_low" in ci_c1_leg and "equivalence_sensitivity_margins" in ci_c1_leg
    ci_c2_cell = s2_cell["ci_screenbest_minus_rand8_best_20splits"]
    assert np.isclose(ci_c2_cell["estimate"] * 100.0, 0.206, atol=0.01)
    assert ci_c2_cell["ci_low"] > 0.0
    assert ci_c2_cell["ci90_low"] > 0.0

    # Finding 4, 5, 8 & V2 Task 6: Multiplicity BY columns, conditional scope, and exact sign-flip detection count (240 on MMLU)
    df_mult = pd.read_csv(RESULTS_DIR / "multiplicity_results.csv")
    assert "detections_item_cluster_t_by" in df_mult.columns
    assert "inference_scope" in df_mult.columns
    m1_row = df_mult[
        (df_mult["cohort"] == "MMLU")
        & (df_mult["estimator"] == "legacy_first")
        & (df_mult["fdr_family"] == "per_cube_bh")
    ].iloc[0]
    assert int(m1_row["detections_unblocked"]) == 95
    assert int(m1_row["detections_item_cluster_t"]) == 243
    assert int(m1_row["detections_item_sign_flip_perm"]) == 240
    assert int(m1_row["detections_rcbd_fixed_block"]) == 565
    assert int(m1_row["detections_unblocked_by"]) == 67
    assert int(m1_row["detections_item_cluster_t_by"]) == 171
    assert int(m1_row["detections_rcbd_fixed_block_by"]) == 445

    # Finding 6 & 11: Exact study observation totals (265,856 Cohort 2; 7,006,016 total) & completeness audit
    obs = headline["study_observation_totals"]
    assert obs["cohort1_mmlu_cell_item_observations"] == 6740160
    assert obs["cohort2_slab_cell_item_observations"] == 265856
    assert obs["cohort2_bbh_cell_item_observations"] == 105600
    assert obs["cohort2_lmentry_cell_item_observations"] == 160256
    assert obs["total_study_cell_item_observations"] == 7006016
    assert obs["lmentry_word_not_containing_items"] == 26

    comp = headline["cohort2_completeness_audit"]
    assert comp["bbh_complete_16cell_tasks"] == 6 and comp["bbh_total_tasks"] == 15
    assert (
        comp["lmentry_complete_16cell_tasks"] == 7 and comp["lmentry_total_tasks"] == 10
    )
    assert len(comp["excluded_incomplete_tasks"]) == 12

    # Finding 7: Cell-0 diagnostic (row 65, cell mean 0.600652, +0.727 pp difference)
    c0 = headline["cell0_diagnostic"]
    assert c0["template_indices"] == [65, 76, 77, 96, 97]
    assert c0["legacy_selected_row_index"] == 65
    assert np.isclose(c0["split0_conf_legacy_first"], 0.607923, atol=1e-5)
    assert np.isclose(c0["split0_conf_cell_mean"], 0.600652, atol=1e-5)
    assert np.isclose(c0["split0_diff_legacy_minus_cell_mean_pp"], 0.7271, atol=1e-3)

    # Finding 9: All 60 regular 2^(5-2)_III fractions retained and ~1.53x task/pooled ratio.
    df_diag = pd.read_csv(RESULTS_DIR / "design_diagnostics.csv")
    df_all60 = df_diag[df_diag["design_family"] == "2(5-2)III_all60"]
    assert len(df_all60) == 60
    assert np.isclose(
        s1_leg["task_main_rmse_ratio_rand8_over_res3_all60"], 1.527, atol=0.01
    )
    assert np.isclose(
        s1_leg["pool_main_rmse_ratio_rand8_over_res3_all60"], 1.526, atol=0.01
    )
    assert np.isclose(
        s1_leg["task_main_rmse_ratio_rand8_over_res3_coset_avg"], 1.589, atol=0.01
    )

    # Finding 12: Positive oracle-uplift denominator fractions (under primary mean_tied and legacy first_index)
    assert np.isclose(
        s1_leg["recovery_split0_res5_best_coset_avg"]["positive_uplift_cube_fraction"],
        427 / 855,
        atol=1e-5,
    )
    assert np.isclose(
        s1_leg["recovery_20splits_res5_best_coset_avg"][
            "positive_uplift_cube_fraction"
        ],
        443 / 855,
        atol=1e-5,
    )
    assert np.isclose(
        s1_leg["first_index_compatibility"]["recovery_split0_res5_best_coset_avg"][
            "positive_uplift_cube_fraction"
        ],
        396 / 855,
        atol=1e-5,
    )

    # Verify manuscript text in paper_hf/main.tex contains no stale values or unsupported claims (when local paper_hf/ exists)
    tex_path = Path(__file__).resolve().parents[2] / "paper_hf" / "main.tex"
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        for forbidden in [
            "7{,}024{,}960",
            "284{,}800",
            "r = 0.420",
            "r = 0.065",
            "explaining why",
            "template index 20",
            "0.60002",
            "at $242$",
            "strict finite-sample validity",
            "achieve equivalent held-out accuracy",
        ]:
            assert (
                forbidden not in tex
            ), f"Stale/unsupported string still present in main.tex: {forbidden}"
        for required in [
            "7{,}006{,}016",
            "265{,}856",
            "0.478",
            "0.526",
            "0.213",
            "0.600652",
            "240",
            "1.53\\times",
            "6/15",
            "7/10",
            "46.3\\%",
        ]:
            assert (
                required in tex
            ), f"Required updated string missing from main.tex: {required}"


def test_v2_tasks_2_to_8_unit_and_manifest_integrity():
    """Unit and end-to-end verification for V2 Tasks 2, 3, 4, 5, 6, 7, and 8."""
    # Task 2: Tie-aware selection (`mean_tied`, `first_index`, `seeded_random`) & all-reference-cells RoM
    disc = np.array(
        [[0.80, 0.70], [0.80, 0.70], [0.50, 0.70]]
    )  # split 0 ties cells 0 & 1; split 1 ties all 3
    conf = np.array([[0.90, 0.60], [0.50, 0.90], [0.40, 0.30]])
    c_mt, tc_mt, _ = select_with_tie_policy(disc, conf, tie_policy="mean_tied")
    c_fi, _, _ = select_with_tie_policy(disc, conf, tie_policy="first_index")
    c_sr, _, _ = select_with_tie_policy(
        disc, conf, tie_policy="seeded_random", rng=np.random.default_rng(123)
    )
    assert np.array_equal(tc_mt, np.array([2, 3]))
    assert np.allclose(c_mt, np.array([0.70, 0.60]))
    assert np.allclose(c_fi, np.array([0.90, 0.60]))
    assert c_sr[0] in (0.90, 0.50) and c_sr[1] in (0.60, 0.90, 0.30)

    all_ref = compute_all_reference_cells_recovery(
        method_conf=np.array([0.72, 0.78]),
        oracle_or_cells_conf=np.array([0.80, 0.82]),
        cells_or_oracle_conf=np.array([[0.60, 0.70, 0.75], [0.65, 0.72, 0.80]]),
    )
    assert all_ref["n_reference_cells"] == 3
    assert all_ref["min_rom_pct"] <= all_ref["median_rom_pct"] <= all_ref["max_rom_pct"]

    # Task 3: Unconditioned balanced random designs accepted for ScreenBest even when rank-deficient for 2FI OLS
    uncond_16, _, uncond_stats = generate_random_designs_with_diagnostics(
        5,
        16,
        40,
        SEEDS["rand16_full_rank"],
        False,
        "MMLU",
        "rand16_screenbest_unconditioned",
        require_main_full_rank=False,
    )
    assert len(uncond_16) == 40
    assert uncond_stats["rejected_rank_deficient_balanced"] == 0
    assert uncond_stats["accepted_2fi_rank_deficient"] > 0

    # Task 4: Duplicate-row weighting audit (`cell_mean_unique` vs `cell_mean`)
    df_mmlu_map, leg_map, _ = build_mmlu_mapping_table()
    uniq_map = extract_unique_cell_map(df_mmlu_map)
    assert int(df_mmlu_map["is_duplicate_text"].sum()) == 7
    assert sum(len(v) for v in leg_map.values()) == 100
    assert sum(len(v) for v in uniq_map.values()) == 93

    # Task 5: Shared-model Cohort-2 cluster bootstrap & partial correlation
    df_synth = pd.DataFrame(
        [
            {
                "cohort": bench,
                "model_id": f"m_{m}",
                "task": f"t_{t}",
                "x": float(m + t),
                "y": float(2 * m + t + 0.1 * (m * t)),
                "z": float(m),
            }
            for bench, n_m in [("BBH", 4), ("LMentry", 6)]
            for m in range(n_m)
            for t in range(3)
        ]
    )
    corr_res = _two_way_cluster_bootstrap_correlation(
        df_synth, "x", "y", z_col="z", n_boot=50, seed=42
    )
    assert (
        corr_res["n_unique_models"] == 6
    )  # 4 shared models coupled across BBH & LMentry
    assert corr_res["n_unique_tasks"] == 6
    assert "partial_pearson_r" in corr_res

    # Task 6: Zero-variance contrast & survival-function (`sf`) non-zero tail p-value check
    m4 = build_cube_design_matrices(4)
    Y_const = np.ones((16, 40), dtype=float)
    inf_const = compute_cube_rcbd_and_robust_inference(Y_const, m4["X_pm"], n_perm=50)
    assert inf_const["n_zero_var_contrasts"] == 4
    assert np.all(inf_const["p_cluster"] == 1.0)
    assert np.isclose(inf_const["sign_flip_min_achievable_p"], 1.0 / 51.0)

    # Strong non-null contrast has positive survival-function p-value (never underflows to 0.0)
    rng = np.random.default_rng(99)
    Y_strong = 0.5 + 0.35 * m4["X_pm"][:, [0]] + rng.normal(0.0, 0.02, size=(16, 80))
    inf_strong = compute_cube_rcbd_and_robust_inference(Y_strong, m4["X_pm"], n_perm=0)
    assert 0.0 < inf_strong["p_cluster"][0] < 1e-20

    # Task 7: Nonconfirmatory normal-Wald margin sensitivity is labeled honestly
    Z_inside = rng.normal(0.0002, 0.0005, size=(15, 30))
    tost_in = compute_tost_paired_cluster(Z_inside, margin_pp=0.50, n_boot=200, seed=11)
    assert tost_in["method"] == "normal_wald_tost_sensitivity"
    assert tost_in["confirmatory_equivalence_test"] is False
    assert tost_in["wald_rejects_both_one_sided_nulls"] is True

    Z_outside = rng.normal(0.0080, 0.0005, size=(15, 30))
    tost_out = compute_tost_paired_cluster(
        Z_outside, margin_pp=0.25, n_boot=200, seed=11
    )
    assert tost_out["wald_rejects_both_one_sided_nulls"] is False
    assert tost_out["p_tost"] > 0.50

    # Task 8: Every SHA-256 digest in summary_manifest.json matches the exact bytes on disk
    manifest = json.loads(
        (RESULTS_DIR / "summary_manifest.json").read_text(encoding="utf-8")
    )
    path_lookup = {
        "design_diagnostics": RESULTS_DIR / "design_diagnostics.csv",
        "coset_results": RESULTS_DIR / "coset_results.csv",
        "split_selection_results": RESULTS_DIR / "split_selection_results.csv",
        "template_mapping": RESULTS_DIR / "template_mapping.csv",
        "template_sensitivity": RESULTS_DIR / "template_sensitivity.csv",
        "factor_audit": RESULTS_DIR / "factor_audit.csv",
        "interaction_magnitudes": RESULTS_DIR / "interaction_magnitudes.csv",
        "rcbd_robustness": RESULTS_DIR / "rcbd_robustness.csv",
        "multiplicity_results": RESULTS_DIR / "multiplicity_results.csv",
        "cohort1_mmlu_models": RESULTS_DIR / "cohort1_mmlu_models.csv",
        "cohort1_mmlu_cubes": RESULTS_DIR / "cohort1_mmlu_cubes.csv",
        "cohort2_slab_cubes": RESULTS_DIR / "cohort2_slab_cubes.csv",
        "claim_evidence_map": RESULTS_DIR / "claim_evidence_map.csv",
        "headline_summary": RESULTS_DIR / "headline_summary.json",
        "fig1_effect_and_selection_pdf": FIGURES_DIR / "fig1_effect_and_selection.pdf",
        "fig1_effect_and_selection_png": FIGURES_DIR / "fig1_effect_and_selection.png",
        "fig2_models_and_rcbd_pdf": FIGURES_DIR / "fig2_models_and_rcbd.pdf",
        "fig2_models_and_rcbd_png": FIGURES_DIR / "fig2_models_and_rcbd.png",
        "table_mmlu_models_tex": TABLES_DIR / "table_mmlu_models.tex",
        "table_cross_cohort_tex": TABLES_DIR / "table_cross_cohort.tex",
        "generated_macros_tex": TABLES_DIR / "generated_macros.tex",
    }
    paper_hf_root = TABLES_DIR.parent
    for key, expected_sha in manifest["output_artifact_sha256"].items():
        assert key in path_lookup, f"Unexpected artifact key in manifest: {key}"
        target_path = path_lookup[key]
        if not target_path.exists() and str(target_path).startswith(str(paper_hf_root)):
            continue
        actual_sha = sha256_file(target_path)
        assert (
            actual_sha == expected_sha
        ), f"SHA-256 mismatch for {key}: {actual_sha} != {expected_sha}"


def test_v3_zero_variance_nonzero_contrast_is_conservative():
    """A deterministic nonzero contrast has no estimable sampling variance; do not emit p≈0."""
    m4 = build_cube_design_matrices(4)
    factor_a_response = ((m4["X_pm"][:, [0]] + 1.0) / 2.0) @ np.ones((1, 40))

    inference = compute_cube_rcbd_and_robust_inference(
        factor_a_response, m4["X_pm"], n_perm=0
    )

    assert inference["zero_var_mask"][0]
    assert inference["taus"][0] != 0.0
    assert inference["p_cluster"][0] == 1.0
    assert inference["p_unblocked"][0] == 1.0
    assert inference["p_rcbd"][0] == 1.0


def test_v4_random_eight_run_designs_are_column_rank_deficient_for_full_2fi():
    """An 8-run screen cannot identify the 16-column intercept+main+2FI model."""
    accepted, diagnostics, stats = generate_random_designs_with_diagnostics(
        5,
        8,
        12,
        SEEDS["rand8_cohort1"],
        False,
        "MMLU",
        "test_rand8_screen",
        require_main_full_rank=False,
    )

    assert len(accepted) == 12
    accepted_rows = [row for row in diagnostics if row["accepted_primary"]]
    assert stats["accepted_2fi_rank_deficient"] == 12
    assert all(row["rank_2fi"] < 16 for row in accepted_rows)
    assert all(np.isinf(row["cond_2fi"]) for row in accepted_rows)


def test_v4_pooled_discovery_split_is_applied_within_each_task():
    """Odd task lengths must not flip even/odd discovery parity in later tasks."""
    task_a = np.arange(2 * 3, dtype=float).reshape(2, 3)
    task_b = (100 + np.arange(2 * 5, dtype=float)).reshape(2, 5)

    pooled = concatenate_task_discovery_items([task_a, task_b])

    expected = np.hstack([task_a[:, ::2], task_b[:, ::2]])
    globally_alternated = np.hstack([task_a, task_b])[:, ::2]
    assert np.array_equal(pooled, expected)
    assert not np.array_equal(pooled, globally_alternated)


def test_v4_unbalanced_panel_bootstrap_redraws_empty_replicates():
    panel = pd.DataFrame(
        [
            {"cohort": "BBH", "model_id": "m0", "task": "t0", "value": 0.25},
            {"cohort": "BBH", "model_id": "m1", "task": "t1", "value": 0.75},
        ]
    )

    result = panel_cluster_bootstrap_ci(panel, "value", n_boot=200, seed=7)

    assert result["n_valid_bootstrap"] == 200
    assert result["n_invalid_bootstrap_draws"] > 0
    assert np.isfinite(result["se"])
    assert np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_high"])


def test_v3_slab_per_model_family_couples_shared_model_ids():
    """A model appearing in BBH and LMentry must form one per-model correction family."""
    rows = []
    for cohort in ("BBH", "LMentry"):
        for factor, p_value in zip(("A", "B", "C", "D"), (0.01, 0.02, 0.20, 0.40)):
            rows.append(
                {
                    "cohort": cohort,
                    "estimator": "legacy_first",
                    "model_id": "shared-model",
                    "task": f"{cohort.lower()}-task",
                    "factor": factor,
                    "p_unblocked": p_value,
                    "p_rcbd": p_value,
                    "p_item_cluster": p_value,
                    "p_sign_flip_perm": p_value,
                    "zero_var_contrast": False,
                }
            )

    table = build_multiplicity_table(pd.DataFrame(rows), n_perm_rcbd=200)
    per_model = table[
        (table["cohort"] == "SLAB")
        & (table["estimator"] == "legacy_first")
        & (table["fdr_family"] == "per_model_bh")
    ].iloc[0]

    assert per_model["n_model_families"] == 1


def test_v3_summary_uses_all_cohort2_cubes_and_discloses_denominator():
    cubes = pd.read_csv(RESULTS_DIR / "cohort2_slab_cubes.csv")
    headline = json.loads(
        (RESULTS_DIR / "headline_summary.json").read_text(encoding="utf-8")
    )
    summary = headline["cohort2_slab_legacy_first"]

    assert len(cubes) == 178
    assert summary["mse_summary_cube_count"] == 178
    assert np.isclose(
        summary["mean_mse_reduction_pct"], cubes["mse_reduction_pct"].mean()
    )
    assert np.isclose(
        summary["mean_f_stat_multiplier"], cubes["f_stat_multiplier"].mean()
    )


def test_v3_equivalence_output_is_labeled_nonconfirmatory_sensitivity():
    rng = np.random.default_rng(20261111)
    paired_differences = rng.normal(0.0002, 0.0005, size=(15, 30))

    result = compute_tost_paired_cluster(
        paired_differences, margin_pp=0.50, n_boot=200, seed=11
    )

    assert result["method"] == "normal_wald_tost_sensitivity"
    assert result["confirmatory_equivalence_test"] is False
    assert "equivalent_at_alpha" not in result


def test_v3_source_integrity_covers_every_analysis_template_catalog():
    expected_catalogs = {
        str(path.relative_to(DATA_DIR))
        for pattern in (
            "templates/MMLU/*.csv",
            "templates/BBH/*.csv",
            "templates/LMentry/*.csv",
            "templates/mmlu_templates.json",
        )
        for path in DATA_DIR.glob(pattern)
    }
    assert len(expected_catalogs) == 83
    if (DATA_DIR / "Ys.pickle").exists():
        hashes = verify_source_hashes()
        catalog_hashes = hashes["template_catalog_sha256"]
        assert set(catalog_hashes) == expected_catalogs
        assert all(len(digest) == 64 for digest in catalog_hashes.values())

    manifest = json.loads(
        (RESULTS_DIR / "summary_manifest.json").read_text(encoding="utf-8")
    )
    expected_analysis_files = {
        "research/hf_multi_prompt/run_experiments.py",
        "research/hf_multi_prompt/build_release_manifest.py",
        "tests/research/test_hf_multi_prompt.py",
        "paper_hf/main.tex",
        "paper_hf/references.bib",
        "prompt_prism/analysis/anova.py",
        "prompt_prism/analysis/optimizer.py",
        "prompt_prism/design/catalog.py",
        "prompt_prism/design/generators.py",
    }
    assert set(manifest["analysis_definition_sha256"]) == expected_analysis_files
    project_root = RESULTS_DIR.parents[2]
    for relative_path, expected_sha in manifest["analysis_definition_sha256"].items():
        full_path = project_root / relative_path
        if relative_path.startswith("paper_hf/") and not full_path.exists():
            continue
        assert sha256_file(full_path) == expected_sha


@pytest.mark.skipif(
    not (TABLES_DIR.parent / "main.tex").exists(),
    reason="Requires gitignored local paper_hf/ directory",
)
def test_v3_manuscript_uses_generated_values_and_removes_prohibited_claims():
    main_tex = (TABLES_DIR.parent / "main.tex").read_text(encoding="utf-8")
    research_readme = (RESULTS_DIR.parent / "README.md").read_text(encoding="utf-8")

    for macro in (
        r"\TotalStudyObs",
        r"\MmluPoolTwoFiRatio",
        r"\MmluMseReductionMean",
        r"\SlabMseReductionMean",
    ):
        assert macro in main_tex

    combined = main_tex + "\n" + research_readme
    for prohibited in (
        "exact TOST",
        "cross-model and cross-task generalization is quantified",
        "full resolution for per-cube BH/BY",
        "5 unique templates",
        "Untuned Baseline (Cell 0)",
        "which omits $2\\mathrm{FI}$/higher-order",
        "Pairwise Interactions",
        "m=44",
        r"95\% CI",
    ):
        assert prohibited not in combined

    assert "five catalog rows and four unique templates" in main_tex
    assert r"10^{-6}" in main_tex
    assert "all 83 included" in research_readme
    assert "m=28" in main_tex


def test_v4_export_schemas_are_cohort_generic_and_canonical_inference_is_null():
    split_columns = set(
        pd.read_csv(RESULTS_DIR / "split_selection_results.csv", nrows=1).columns
    )
    interaction_columns = set(
        pd.read_csv(RESULTS_DIR / "interaction_magnitudes.csv", nrows=1).columns
    )

    assert "rmse_main_fraction" in split_columns
    assert "rmse_main_res5" not in split_columns
    assert "split0_rmse_main_fraction" in interaction_columns
    assert "disc0_rmse_main_res5" not in interaction_columns

    headline = json.loads(
        (RESULTS_DIR / "headline_summary.json").read_text(encoding="utf-8")
    )
    for cohort_key in ("cohort1_mmlu_canonical_first", "cohort2_slab_canonical_first"):
        summary = headline[cohort_key]
        assert summary["inference_computed"] is False
        assert summary["total_factor_tests"] is None
        assert summary["total_sig_unblocked_per_cube_bh"] is None
        assert summary["total_sig_rcbd_per_cube_bh"] is None
