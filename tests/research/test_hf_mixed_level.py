"""
Unit, synthetic-statistical, no-outcome-peeking, and replay reproducibility tests
for Phase 0 Task R2 (`research/hf_multi_prompt/mixed_level_analysis.py`).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.hf_multi_prompt.mixed_level_analysis import (
    MIXED_LEVEL_SEEDS,
    RESULTS_DIR,
    WELL_CONDITIONED_CAPS,
    audit_mmlu_mixed_level_catalog,
    build_additive_mixed_level_matrix,
    build_all_catalog_bases,
    compute_design_criteria,
    extract_binary_5bit_subsets_on_89,
    fit_mixed_level_ols,
    fit_prompteval_rasch_covariate,
    generate_mixed_level_random_subsets,
    generate_prompteval_sparse_mask,
    helmert_contrast_matrix,
    load_mmlu_catalog_tables,
    run_mixed_level_study,
    select_all_outcome_blind_designs,
    select_optimal_subset_fedorov,
    validate_mixed_level_design,
    write_mixed_level_artifacts,
)
from research.hf_multi_prompt.run_experiments import (
    DATA_DIR,
    MMLU_MODELS,
    sha256_file,
    verify_source_hashes,
)

# ==============================================================================
# 1. Catalog Audit (89 / 396), OA Impossibility, and Binary Level-Drop Audit
# ==============================================================================


def test_catalog_audit_89_of_396_and_oa_impossibility():
    if (DATA_DIR / "Ys.pickle").exists():
        hashes = verify_source_hashes()
        assert len(hashes["Ys_pickle_sha256"]) == 64
        assert len(hashes["mmlu_metadata_sha256"]) == 64

    bundle = load_mmlu_catalog_tables()
    assert bundle["all_57_csvs_identical"] is True
    audit = audit_mmlu_mixed_level_catalog(bundle)

    assert audit["axes"]["n_op"] == 4
    assert audit["axes"]["n_space"] == 9
    assert audit["axes"]["n_sep"] == 11
    assert audit["counts"]["full_cartesian_grid"] == 396
    assert audit["counts"]["catalog_rows"] == 100
    assert audit["counts"]["unique_outer_template_strings"] == 93
    assert audit["counts"]["observed_axis_combinations"] == 89
    assert audit["counts"]["missing_axis_combinations"] == 307

    # Exact 2-way projections
    pw = audit["pairwise_coverage"]
    assert pw["op_x_space_observed"] == 34 and pw["op_x_space_total"] == 36
    assert pw["op_x_sep_observed"] == 41 and pw["op_x_sep_total"] == 44
    assert pw["space_x_sep_observed"] == 68 and pw["space_x_sep_total"] == 99

    # Exact 1-way marginal sums and minimum marginal count
    for key, expected_sum in [
        ("catalog_100_rows", 100),
        ("unique_93_templates", 93),
        ("observed_89_combinations", 89),
    ]:
        m = audit["marginal_counts"][key]
        assert sum(m["op"].values()) == expected_sum
        assert sum(m["space_repr"].values()) == expected_sum
        assert sum(m["sep"].values()) == expected_sum

    assert audit["marginal_counts"]["observed_89_combinations"]["sep"]["]"] == 5

    # Number-theoretic OA non-existence below N = 396
    oa = audit["orthogonal_array_impossibility_proof"]
    assert oa["pairwise_coprime_levels"] is True
    assert oa["strength_1_min_balanced_runs_lcm_4_9_11"] == 396
    assert oa["strength_2_min_oa_runs_lcm_36_44_99"] == 396

    # Binary 5-bit cube projection onto 89 combinations:
    # canonical_first_32 is rank 22/22, while legacy_first_32 drops space='' and sep='-' (rank 20/22)
    bases_89 = build_all_catalog_bases(bundle["df_89"], "combinations_89")
    bases_100 = build_all_catalog_bases(bundle["df_100"], "catalog_rows_100")
    bin_sub = extract_binary_5bit_subsets_on_89(bundle)

    assert len(set(bin_sub["canonical_cube32_combo_indices"])) == 32
    assert len(set(bin_sub["canonical_res5_pos16_combo_indices"])) == 16
    assert len(set(bin_sub["canonical_res5_neg16_combo_indices"])) == 16
    assert set(bin_sub["canonical_res5_pos16_combo_indices"]).isdisjoint(
        set(bin_sub["canonical_res5_neg16_combo_indices"])
    )

    X22_89 = bases_89["additive_22p"]["X"]
    X22_100 = bases_100["additive_22p"]["X"]
    can_crit = compute_design_criteria(
        X22_89[bin_sub["canonical_cube32_combo_indices"]]
    )
    leg_crit = compute_design_criteria(X22_100[bin_sub["legacy_cube32_row_indices"]])
    assert can_crit["is_full_rank"] is True and can_crit["rank"] == 22
    assert leg_crit["is_full_rank"] is False and leg_crit["rank"] == 20


# ==============================================================================
# 2. Orthogonal Helmert Properties, Rotational Invariance, and Full-Rank Bases
# ==============================================================================


def test_helmert_orthogonality_rotational_invariance_and_basis_ranks():
    for L in (2, 3, 4, 5, 6, 9, 11):
        H = helmert_contrast_matrix(L)
        assert H.shape == (L, L - 1)
        assert np.allclose(H.sum(axis=0), 0.0, atol=1e-13)
        assert np.allclose((H.T @ H) / L, np.eye(L - 1), atol=1e-13)

    # Full 4 x 9 x 11 = 396 Cartesian grid has exact X^T X = 396 * I_22 and cond = 1.0
    full_rows = [
        {"op": f"op_{i}", "space": f"sp_{j}", "sep": f"sep_{k}"}
        for i, j, k in itertools.product(range(4), range(9), range(11))
    ]
    df_396 = pd.DataFrame(full_rows)
    X_396, info_396 = build_additive_mixed_level_matrix(df_396, ("op", "space", "sep"))
    assert X_396.shape == (396, 22)
    rk, cond = validate_mixed_level_design(X_396, "full_396")
    assert rk == 22
    assert np.isclose(cond, 1.0, atol=1e-13)
    assert np.allclose((X_396.T @ X_396) / 396.0, np.eye(22), atol=1e-13)

    # Verify D-optimality, A-optimality, and condition number are strictly invariant
    # to arbitrary orthogonal contrast rotations within each categorical factor
    bundle = load_mmlu_catalog_tables()
    bases_89 = build_all_catalog_bases(bundle["df_89"], "combinations_89")
    X22 = bases_89["additive_22p"]["X"]
    d_opt = select_optimal_subset_fedorov(X22, 24, "D", n_restarts=10, seed=123)
    idx_sub = d_opt["indices"]

    rng = np.random.default_rng(999)
    Q_op, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q_sp, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Q_sep, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    Q_blk = np.eye(22)
    Q_blk[1:4, 1:4] = Q_op
    Q_blk[4:12, 4:12] = Q_sp
    Q_blk[12:22, 12:22] = Q_sep

    X22_rot = X22 @ Q_blk
    c_orig = compute_design_criteria(X22[idx_sub])
    c_rot = compute_design_criteria(X22_rot[idx_sub])
    assert np.isclose(c_orig["condition_number"], c_rot["condition_number"], atol=1e-12)
    assert np.isclose(c_orig["logdet_xtx"], c_rot["logdet_xtx"], atol=1e-11)
    assert np.isclose(c_orig["trace_inv_xtx"], c_rot["trace_inv_xtx"], atol=1e-12)

    # All planned bases across 89, 93, and 100 catalogs have exact full column rank
    for label, df_cat in [
        ("89", bundle["df_89"]),
        ("93", bundle["df_93"]),
        ("100", bundle["df_100"]),
    ]:
        b_all = build_all_catalog_bases(df_cat, label)
        assert b_all["coarsened_main_13p"]["rank"] == 13
        assert b_all["coarsened_inter_15p"]["rank"] == 15
        assert b_all["binary_5bit_16p"]["rank"] == 16
        assert b_all["additive_22p"]["rank"] == 22
        assert b_all["additive22_plus_inter_28p"]["rank"] == 28


# ==============================================================================
# 3. Strict No-Outcome-Peeking Invariance (Requirement 7)
# ==============================================================================


def test_strict_no_outcome_peeking_invariance():
    """
    Permuting or replacing `Y` with random noise or adversarial spikes MUST produce
    the IDENTICAL D-optimal, A-optimal, random, and 5-bit template indices.
    """
    rng = np.random.default_rng(20261118)
    n_models = len(MMLU_MODELS)
    tasks_mini = ["task_alpha", "task_beta"]

    Ys_base = {
        "MMLU": {
            t: [
                rng.binomial(1, 0.55, size=(100, 40)).astype(np.int8)
                for _ in range(n_models)
            ]
            for t in tasks_mini
        }
    }
    # 1. Row- and item-permuted Ys
    Ys_perm = {
        "MMLU": {
            t: [
                mat[rng.permutation(100)][:, rng.permutation(40)].copy()
                for mat in Ys_base["MMLU"][t]
            ]
            for t in tasks_mini
        }
    }
    # 2. Independent noise Ys with completely different mean accuracy
    Ys_noise = {
        "MMLU": {
            t: [
                rng.binomial(1, 0.15, size=(100, 40)).astype(np.int8)
                for _ in range(n_models)
            ]
            for t in tasks_mini
        }
    }
    # 3. Adversarial spike: row 73 (not necessarily optimal in X) has 100% accuracy, all others 0%
    adv_mat = np.zeros((100, 40), dtype=np.int8)
    adv_mat[73, :] = 1
    Ys_adv = {
        "MMLU": {t: [adv_mat.copy() for _ in range(n_models)] for t in tasks_mini}
    }

    _, sum_base = run_mixed_level_study(
        Ys=Ys_base,
        n_random_draws=8,
        n_fedorov_restarts=8,
        n_pe_rand_draws_split0=2,
        n_splits=2,
        n_boot=40,
    )
    _, sum_perm = run_mixed_level_study(
        Ys=Ys_perm,
        n_random_draws=8,
        n_fedorov_restarts=8,
        n_pe_rand_draws_split0=2,
        n_splits=2,
        n_boot=40,
    )
    _, sum_noise = run_mixed_level_study(
        Ys=Ys_noise,
        n_random_draws=8,
        n_fedorov_restarts=8,
        n_pe_rand_draws_split0=2,
        n_splits=2,
        n_boot=40,
    )
    _, sum_adv = run_mixed_level_study(
        Ys=Ys_adv,
        n_random_draws=8,
        n_fedorov_restarts=8,
        n_pe_rand_draws_split0=2,
        n_splits=2,
        n_boot=40,
    )

    ref_designs = json.dumps(sum_base["selected_designs_89"], sort_keys=True)
    assert json.dumps(sum_perm["selected_designs_89"], sort_keys=True) == ref_designs
    assert json.dumps(sum_noise["selected_designs_89"], sort_keys=True) == ref_designs
    assert json.dumps(sum_adv["selected_designs_89"], sort_keys=True) == ref_designs


# ==============================================================================
# 4. Synthetic 4 x 9 x 11 Mixed-Level Fixtures & Rank-Deficient Rejection
# ==============================================================================


def test_synthetic_mixed_level_exact_recovery_and_rank_rejection():
    rng = np.random.default_rng(4242)

    # Part A: Complete 4 x 9 x 11 = 396 synthetic fixture
    full_rows = [
        {"op": f"op_{i}", "space": f"sp_{j}", "sep": f"sep_{k}"}
        for i, j, k in itertools.product(range(4), range(9), range(11))
    ]
    df_396 = pd.DataFrame(full_rows)
    X_396, info_396 = build_additive_mixed_level_matrix(df_396, ("op", "space", "sep"))

    true_tau_op = rng.normal(0.0, 0.08, size=3)
    true_tau_sp = rng.normal(0.0, 0.06, size=8)
    true_tau_sep = rng.normal(0.0, 0.05, size=10)
    true_tau_21 = np.concatenate([true_tau_op, true_tau_sp, true_tau_sep])
    y_396 = 0.62 + 0.5 * (X_396[:, 1:] @ true_tau_21)

    for B in (24, 32):
        for crit in ("D", "A"):
            opt = select_optimal_subset_fedorov(
                X_396, B, crit, n_restarts=8, seed=100 + B
            )
            idx = opt["indices"]
            out_idx = [u for u in range(396) if u not in set(idx)]
            beta_hat = fit_mixed_level_ols(X_396[idx], y_396[idx], f"{crit}_396_B{B}")
            tau_hat = 2.0 * beta_hat[1:]
            assert np.allclose(tau_hat[:3], true_tau_op, atol=1e-13)
            assert np.allclose(tau_hat[3:11], true_tau_sp, atol=1e-13)
            assert np.allclose(tau_hat[11:21], true_tau_sep, atol=1e-13)
            y_pred_out = X_396[out_idx] @ beta_hat
            assert np.allclose(y_pred_out, y_396[out_idx], atol=1e-13)

    # Part B: Observed incomplete 89-cell MMLU catalog fixture (both 22p additive & 28p interaction)
    bundle = load_mmlu_catalog_tables()
    bases_89 = build_all_catalog_bases(bundle["df_89"], "combinations_89")
    X22 = bases_89["additive_22p"]["X"]
    X28 = bases_89["additive22_plus_inter_28p"]["X"]
    X5b = bases_89["binary_5bit_16p"]["X"]

    y_89_add = 0.58 + 0.5 * (X22[:, 1:] @ true_tau_21)
    for B in (24, 32):
        for crit in ("D", "A"):
            opt89 = select_optimal_subset_fedorov(
                X22, B, crit, n_restarts=15, seed=200 + B
            )
            s89 = opt89["indices"]
            out89 = [u for u in range(89) if u not in set(s89)]
            beta89 = fit_mixed_level_ols(X22[s89], y_89_add[s89], f"{crit}_89_B{B}")
            assert np.allclose(2.0 * beta89[1:], true_tau_21, atol=1e-13)
            assert np.allclose(X22[out89] @ beta89, y_89_add[out89], atol=1e-13)

    # Conversely, 5-bit 32-cell binary model CANNOT recover general 21-df additive mixed-level surface!
    bin_sub = extract_binary_5bit_subsets_on_89(bundle)
    s_bin32 = bin_sub["canonical_cube32_combo_indices"]
    out_bin32 = [u for u in range(89) if u not in set(s_bin32)]
    beta_5b = fit_mixed_level_ols(X5b[s_bin32], y_89_add[s_bin32], "bin32_5bit")
    err_5bit_out = float(
        np.sqrt(np.mean((X5b[out_bin32] @ beta_5b - y_89_add[out_bin32]) ** 2))
    )
    assert err_5bit_out > 1e-3

    # Coarsened axis-interaction fixture on 28p basis at B=32
    true_tau_inter6 = rng.normal(0.0, 0.04, size=6)
    true_tau_27 = np.concatenate([true_tau_21, true_tau_inter6])
    y_89_inter = 0.58 + 0.5 * (X28[:, 1:] @ true_tau_27)
    dopt_28 = select_optimal_subset_fedorov(X28, 32, "D", n_restarts=15, seed=332)
    b28_hat = fit_mixed_level_ols(
        X28[dopt_28["indices"]], y_89_inter[dopt_28["indices"]], "D_89_28p"
    )
    assert np.allclose(2.0 * b28_hat[1:], true_tau_27, atol=1e-12)

    # Part C: Rank-deficient, under-budget, and ill-conditioned designs are strictly rejected
    # 1. Under-budget (B=16 < p=22)
    with pytest.raises(ValueError, match="fewer runs"):
        validate_mixed_level_design(X22[:16], "underbudget_16_for_22p")

    # 2. Subset omitting an entire level (`sep == ']'`) -> rank deficient
    no_bracket_idx = bundle["df_89"].index[bundle["df_89"]["sep"] != "]"].tolist()[:24]
    with pytest.raises(ValueError, match="rank deficient"):
        validate_mixed_level_design(X22[no_bracket_idx], "missing_sep_bracket")
    with pytest.raises(ValueError, match="rank deficient"):
        fit_mixed_level_ols(
            X22[no_bracket_idx], y_89_add[no_bracket_idx], "missing_sep_bracket"
        )

    # 3. Condition-number cap violation
    r_fr = generate_mixed_level_random_subsets(
        X22, 24, num_subsets=10, seed=77, require_full_rank=True
    )
    with pytest.raises(ValueError, match="exceeds maximum condition number"):
        validate_mixed_level_design(
            X22[r_fr["subsets"][0]], "ill_cond_check", max_condition_number=2.0
        )


# ==============================================================================
# 5. Equal Operational Budget & PromptEval Rasch-Covariate Recovery
# ==============================================================================


def test_prompteval_equal_budget_masks_and_synthetic_rasch_recovery():
    for B in (16, 24, 32):
        for n_disc in (50, 137):
            mask = generate_prompteval_sparse_mask(89, n_disc, B, seed=500 + B + n_disc)
            assert mask.shape == (89, n_disc)
            assert set(np.unique(mask)).issubset({0.0, 1.0})
            # Exact per-item budget = B and total budget = B * n_disc
            assert np.all(mask.sum(axis=0) == B)
            assert int(mask.sum()) == B * n_disc
            # Stratified across templates: max - min count <= 1
            row_counts = mask.sum(axis=1)
            assert int(row_counts.max() - row_counts.min()) <= 1

    # Synthetic 1PL Rasch + discrete covariates fixture
    rng = np.random.default_rng(8888)
    bundle = load_mmlu_catalog_tables()
    X22 = build_all_catalog_bases(bundle["df_89"], "89")["additive_22p"]["X"]
    dopt24 = select_optimal_subset_fedorov(X22, 24, "D", n_restarts=10, seed=55)[
        "indices"
    ]

    N_items = 600
    beta_true = np.zeros(22)
    beta_true[0] = 0.4
    beta_true[1:4] = [0.45, -0.35, 0.20]
    beta_true[4:12] = rng.normal(0.0, 0.25, size=8)
    beta_true[12:22] = rng.normal(0.0, 0.25, size=10)
    delta_true = rng.normal(0.0, 1.4, size=N_items)
    delta_true -= delta_true.mean()

    logits = (X22 @ beta_true)[:, None] - delta_true[None, :]
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    acc_true = probs_true.mean(axis=1)
    Y_sim = rng.binomial(1, probs_true).astype(float)

    # Subset PromptEval fit on D-optimal 24 templates x 600 items (Q = 14,400)
    fit_sub = fit_prompteval_rasch_covariate(
        Y_obs=Y_sim[dopt24, :], mask_obs=None, X_cov=X22[dopt24, :], X_pred=X22
    )
    out_idx = [u for u in range(89) if u not in set(dopt24)]
    rmse_sub_out = float(
        np.sqrt(np.mean((fit_sub["pred_acc"][out_idx] - acc_true[out_idx]) ** 2))
    )
    assert rmse_sub_out < 0.035
    assert float(np.corrcoef(fit_sub["pred_acc"], acc_true)[0, 1]) > 0.94

    # Sparse PromptEval fit across all 89 templates at exact same Q = 24 * 600 = 14,400 queries
    mask_sp = generate_prompteval_sparse_mask(89, N_items, 24, seed=909)
    assert int(mask_sp.sum()) == 24 * N_items
    fit_sp = fit_prompteval_rasch_covariate(
        Y_obs=Y_sim, mask_obs=mask_sp, X_cov=X22, X_pred=X22
    )
    rmse_sp_all = float(np.sqrt(np.mean((fit_sp["pred_acc"] - acc_true) ** 2)))
    assert rmse_sp_all < 0.025
    assert float(np.corrcoef(fit_sp["pred_acc"], acc_true)[0, 1]) > 0.96


# ==============================================================================
# 6. Generated Artifact Integrity, Optimality Dominance, & Fixed-Seed Replay
# ==============================================================================


def test_generated_artifacts_and_fixed_seed_replay(tmp_path: Path):
    csv_path = RESULTS_DIR / "mixed_level_results.csv"
    json_path = RESULTS_DIR / "mixed_level_summary.json"
    assert csv_path.exists(), f"Missing {csv_path}"
    assert json_path.exists(), f"Missing {json_path}"

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    df_csv = pd.read_csv(csv_path)

    # Verify recorded CSV SHA-256 matches actual file on disk
    actual_csv_sha = sha256_file(csv_path)
    assert (
        summary["artifact_hashes"]["mixed_level_results_csv_sha256"] == actual_csv_sha
    )

    # Verify D-optimal and A-optimal dominance over full-rank random designs at B=16, 24, 32
    sel89 = summary["selected_designs_89"]
    for b_key, basis_suffix in [("B16", "13p"), ("B24", "22p"), ("B32", "22p")]:
        d_opt = sel89[b_key][f"d_opt_{basis_suffix}"]
        a_opt = sel89[b_key][f"a_opt_{basis_suffix}"]
        r_wc = sel89[b_key][f"rand_wellcond_{basis_suffix}"]
        r_fr = sel89[b_key][f"rand_fullrank_{basis_suffix}"]

        assert d_opt["is_full_rank"] and a_opt["is_full_rank"]
        assert d_opt["logdet_xtx"] >= a_opt["logdet_xtx"] - 1e-9
        assert a_opt["trace_inv_xtx"] <= d_opt["trace_inv_xtx"] + 1e-9
        assert d_opt["logdet_xtx"] > r_wc["logdet_xtx_mean"] > r_fr["logdet_xtx_mean"]
        assert (
            a_opt["trace_inv_xtx"]
            < r_wc["trace_inv_xtx_mean"]
            < r_fr["trace_inv_xtx_mean"]
        )

    # Verify well-conditioned random designs strictly obey declared condition-number caps
    for (basis_id, B), cap in WELL_CONDITIONED_CAPS.items():
        sub_rows = df_csv[
            (df_csv["basis_id"] == basis_id)
            & (df_csv["budget"] == B)
            & (df_csv["method_id"] == "random_well_conditioned_ols")
        ]
        if len(sub_rows) > 0:
            assert (sub_rows["condition_number_mean"] <= cap + 1e-9).all()

    # Fixed-seed deterministic replay:
    # 1. Design optimality parity against `mixed_level_summary.json` + exact same-platform replay
    bundle = load_mmlu_catalog_tables()
    bases_89 = build_all_catalog_bases(bundle["df_89"], "combinations_89")
    replayed_designs = select_all_outcome_blind_designs(
        bundle,
        bases_89,
        seeds=MIXED_LEVEL_SEEDS,
        n_random_draws=200,
        n_fedorov_restarts=100,
    )
    replayed_designs_2 = select_all_outcome_blind_designs(
        bundle,
        bases_89,
        seeds=MIXED_LEVEL_SEEDS,
        n_random_draws=200,
        n_fedorov_restarts=100,
    )
    assert replayed_designs == replayed_designs_2
    for b_key, b_dict in replayed_designs.items():
        for d_key, d_val in b_dict.items():
            expected_entry = sel89[b_key][d_key]
            # Across BLAS backends (Accelerate ARM64 vs OpenBLAS x86_64), tied Fedorov optima can select
            # isomorphic optimal index sets with identical criterion values; verify scalar optimality parity.
            for field in (
                "budget",
                "rank",
                "n_params",
                "is_full_rank",
                "n_accepted",
                "n_rejected_rank_deficient",
            ):
                if field in expected_entry:
                    assert d_val[field] == expected_entry[field]
            for metric_field in (
                "logdet_xtx",
                "trace_inv_xtx",
                "logdet_xtx_mean",
                "trace_inv_xtx_mean",
            ):
                if metric_field in expected_entry:
                    val = expected_entry[metric_field]
                    if isinstance(val, float) and np.isfinite(val):
                        assert np.isclose(
                            d_val[metric_field], val, rtol=1e-4, atol=1e-4
                        )

    # 2. Byte-for-byte CSV and JSON SHA-256 hash replay on deterministic study execution
    rng = np.random.default_rng(7070)
    Ys_replay = {
        "MMLU": {
            "task_0": [
                rng.binomial(1, 0.6, size=(100, 50)).astype(np.int8)
                for _ in range(len(MMLU_MODELS))
            ],
            "task_1": [
                rng.binomial(1, 0.5, size=(100, 60)).astype(np.int8)
                for _ in range(len(MMLU_MODELS))
            ],
        }
    }
    df1, sum1 = run_mixed_level_study(
        Ys=Ys_replay,
        n_random_draws=10,
        n_fedorov_restarts=10,
        n_pe_rand_draws_split0=3,
        n_splits=3,
        n_boot=50,
    )
    h1 = write_mixed_level_artifacts(
        df1, sum1, csv_path=tmp_path / "run1.csv", json_path=tmp_path / "run1.json"
    )
    df2, sum2 = run_mixed_level_study(
        Ys=Ys_replay,
        n_random_draws=10,
        n_fedorov_restarts=10,
        n_pe_rand_draws_split0=3,
        n_splits=3,
        n_boot=50,
    )
    h2 = write_mixed_level_artifacts(
        df2, sum2, csv_path=tmp_path / "run2.csv", json_path=tmp_path / "run2.json"
    )
    assert h1 == h2
