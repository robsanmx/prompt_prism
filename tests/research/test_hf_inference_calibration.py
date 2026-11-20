"""
Focused unit, synthetic-statistical, anti-conservative perturbation, and deterministic
hash-replay tests for Phase 0 Task R3 (`research/hf_multi_prompt/inference_calibration.py`).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from research.hf_multi_prompt.inference_calibration import (
    CALIBRATION_SEEDS,
    CALIBRATION_TOLERANCE_GATES,
    NOMINAL_ALPHAS,
    PROCEDURES,
    REGIME_SPECIFICATIONS,
    adjust_pvalues_2d,
    evaluate_calibration_gates,
    evaluate_cubes_batch_inference,
    run_inference_calibration,
    run_regime_n1_additive_null,
    run_regime_n2_heterogeneous_2fi,
)
from research.hf_multi_prompt.run_experiments import (
    DATA_DIR,
    RESULTS_DIR,
    build_cube_design_matrices,
    build_mmlu_mapping_table,
    compute_cube_rcbd_and_robust_inference,
    extract_cell_item_matrix,
    sha256_file,
)

# ==============================================================================
# 1. Exact Mathematical Parity with `run_experiments.py` and `statsmodels`
# ==============================================================================


def test_adjust_pvalues_2d_exact_statsmodels_parity():
    rng = np.random.default_rng(20261118)
    for m in (4, 5, 28, 52, 285):
        P = rng.uniform(0.0, 1.0, size=(30, m))
        P[0, :2] = 1.0 / 201.0  # discrete permutation tie
        P[1, :] = 1.0  # all-one degenerate cube
        P[2, 0] = 0.0  # exact zero boundary
        for method in ("fdr_bh", "fdr_by"):
            q_fast = adjust_pvalues_2d(P, method=method)
            q_ref = np.array([multipletests(row, method=method)[1] for row in P])
            assert np.allclose(q_fast, q_ref, atol=1e-15)


@pytest.mark.skipif(
    not (DATA_DIR / "Ys.pickle").exists(),
    reason="Requires gitignored local PromptEval Ys.pickle dataset",
)
def test_batch_inference_matches_single_cube_reference_and_edge_cases():
    with open(DATA_DIR / "Ys.pickle", "rb") as f:
        Ys = pickle.load(f)

    _, leg_map, canon_map = build_mmlu_mapping_table()
    all_cells_5 = sorted(leg_map.keys())
    X5 = build_cube_design_matrices(5)["X_pm"]

    Y_list = [
        extract_cell_item_matrix(
            Ys["MMLU"]["abstract_algebra"][m_i],
            all_cells_5,
            canon_map,
            estimator="cell_mean",
        )
        for m_i in range(6)
    ]
    Y_batch = np.stack(Y_list, axis=0)
    batch_res = evaluate_cubes_batch_inference(Y_batch, X5)

    for idx, Y_single in enumerate(Y_list):
        ref = compute_cube_rcbd_and_robust_inference(Y_single, X5, n_perm=0)
        assert np.allclose(batch_res["taus"][idx], ref["taus"], atol=1e-14)
        assert np.allclose(batch_res["unblocked"][idx], ref["p_unblocked"], atol=1e-13)
        assert np.allclose(
            batch_res["fixed_block_rcbd"][idx], ref["p_rcbd"], atol=1e-13
        )
        assert np.allclose(
            batch_res["item_clustered_t"][idx], ref["p_cluster"], atol=1e-13
        )
        assert np.array_equal(batch_res["zero_var_mask"][idx], ref["zero_var_mask"])

    # Degenerate all-zero and single zero-variance contrast edge cases
    X4 = build_cube_design_matrices(4)["X_pm"]
    Y_zero = np.zeros((2, 16, 30), dtype=float)
    res_zero = evaluate_cubes_batch_inference(Y_zero, X4)
    for proc in PROCEDURES:
        assert np.all(res_zero[proc] == 1.0)

    # NaN block rejection
    Y_nan = np.ones((1, 16, 20), dtype=float)
    Y_nan[0, 3, 5] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        evaluate_cubes_batch_inference(Y_nan, X4)


# ==============================================================================
# 2. Anti-Conservative Perturbation & Underpowered Gate Failure Tests
# ==============================================================================


def test_gate_fails_explicitly_under_p_squared_anti_conservative_perturbation():
    """Scaling p -> p^2 makes p-values anti-conservative and MUST fail registered gates."""
    rows_perturbed, _ = run_regime_n1_additive_null(
        regime_id="N1_additive_null",
        cohort_label="SYNTHETIC_K32_N200",
        num_factors=5,
        n_items=200,
        n_models=6,
        n_tasks=8,
        n_replicates=40,
        seed_data=CALIBRATION_SEEDS["n1_standard_data"],
        seed_sf200=CALIBRATION_SEEDS["n1_standard_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n1_standard_sf2000"],
        pvalue_transform=lambda p: np.square(p),
    )
    df_pert = pd.DataFrame(rows_perturbed)
    gate_eval = evaluate_calibration_gates(df_pert, gates=CALIBRATION_TOLERANCE_GATES)

    assert gate_eval["primary_item_clustered_t_standard_n_pass"] is False
    verdict = gate_eval["procedure_verdicts"]["item_clustered_t"]
    assert verdict["verdict"] == "FAIL_ANTI_CONSERVATIVE"
    assert verdict["anti_conservative_violation"] is True
    clust_failures = [
        c for c in gate_eval["failed_checks"] if c["procedure"] == "item_clustered_t"
    ]
    assert len(clust_failures) >= 5  # Fails raw curve + BH/BY FDR checks


def test_gate_fails_explicitly_under_se_underestimation():
    """Underestimating item-clustered SE (0.70x) inflates Type-I error and MUST fail gates."""
    rows_under_se, _ = run_regime_n1_additive_null(
        regime_id="N1_additive_null",
        cohort_label="SYNTHETIC_K32_N200",
        num_factors=5,
        n_items=200,
        n_models=6,
        n_tasks=8,
        n_replicates=45,
        seed_data=CALIBRATION_SEEDS["n1_standard_data"],
        seed_sf200=CALIBRATION_SEEDS["n1_standard_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n1_standard_sf2000"],
        se_scale_cluster=0.70,
    )
    df_under = pd.DataFrame(rows_under_se)
    gate_eval = evaluate_calibration_gates(df_under, gates=CALIBRATION_TOLERANCE_GATES)

    assert gate_eval["primary_item_clustered_t_standard_n_pass"] is False
    verdict = gate_eval["procedure_verdicts"]["item_clustered_t"]
    assert verdict["verdict"] == "FAIL_ANTI_CONSERVATIVE"
    assert verdict["anti_conservative_violation"] is True


def test_gate_fails_explicitly_when_planted_alternative_power_collapses():
    """Compressing p-values into [0.5, 1.0] passes Type-I ceilings but MUST fail power gates."""
    rows_no_power, _ = run_regime_n2_heterogeneous_2fi(
        regime_id="N2b_planted_semi_null",
        cohort_label="SYNTHETIC_SEMI_NULL",
        planted_factor0_shift=0.24,
        n_models=6,
        n_tasks=8,
        n_items=200,
        n_replicates=30,
        seed_data=CALIBRATION_SEEDS["n2b_semi_data"],
        seed_sf200=CALIBRATION_SEEDS["n2b_semi_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n2b_semi_sf2000"],
        pvalue_transform=lambda p: np.clip(0.5 + 0.5 * p, 0.0, 1.0),
    )
    df_np = pd.DataFrame(rows_no_power)
    gate_eval = evaluate_calibration_gates(df_np, gates=CALIBRATION_TOLERANCE_GATES)

    assert gate_eval["primary_item_clustered_t_standard_n_pass"] is False
    verdict = gate_eval["procedure_verdicts"]["item_clustered_t"]
    assert verdict["low_power_violation"] is True
    assert verdict["verdict"] == "FAIL_UNDERPOWERED_CONSERVATIVE"


# ==============================================================================
# 3. Generated Artifact Integrity, Calibration Findings, and Hash Replay
# ==============================================================================


def test_committed_calibration_artifacts_and_scientific_findings():
    csv_path = RESULTS_DIR / "inference_calibration.csv"
    summary_path = RESULTS_DIR / "inference_calibration_summary.json"
    assert (
        csv_path.exists()
    ), "Run research/hf_multi_prompt/inference_calibration.py first"
    assert summary_path.exists()

    df = pd.read_csv(csv_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # 1. All 5 nominal alphas, 5 procedures, and 4 regime families N1..N4 are present
    assert set(df["nominal_alpha"].unique()) == set(NOMINAL_ALPHAS)
    assert set(df["procedure"].unique()) == set(PROCEDURES)
    assert set(df["regime_family"].unique()) == {"N1", "N2", "N3", "N4"}
    for reg_id in REGIME_SPECIFICATIONS:
        assert reg_id in summary["regime_specifications_and_exchangeability"]
        spec = summary["regime_specifications_and_exchangeability"][reg_id]
        assert len(spec["exchangeability_assumptions"]) > 40

    # 2. Primary `item_clustered_t` passes all standard-N calibration and power gates
    gates = summary["calibration_gate_evaluation"]
    assert gates["primary_item_clustered_t_standard_n_pass"] is True
    assert (
        gates["procedure_verdicts"]["item_clustered_t"]["verdict"]
        == "PASS_CALIBRATED_STANDARD_N"
    )

    # 3. `fixed_block_rcbd` passes exchangeable N1/N3a nulls, but FAILS N2, N3b, N4a, N4b!
    assert gates["fixed_block_rcbd_fails_anti_conservative"] is True
    assert (
        gates["procedure_verdicts"]["fixed_block_rcbd"]["verdict"]
        == "FAIL_ANTI_CONSERVATIVE"
    )

    # Verify bidirectional variance-pooling miscalibration of `fixed_block_rcbd` in N2a
    sub_n2a_005 = df[
        (df["regime_id"] == "N2a_hetero_2fi_null")
        & np.isclose(df["nominal_alpha"], 0.05)
    ]
    rcbd_hv = float(
        sub_n2a_005[
            (sub_n2a_005["procedure"] == "fixed_block_rcbd")
            & (sub_n2a_005["stratum"] == "high_var_null")
        ]["raw_type1_rate"].iloc[0]
    )
    rcbd_lv = float(
        sub_n2a_005[
            (sub_n2a_005["procedure"] == "fixed_block_rcbd")
            & (sub_n2a_005["stratum"] == "low_var_null")
        ]["raw_type1_rate"].iloc[0]
    )
    clust_hv = float(
        sub_n2a_005[
            (sub_n2a_005["procedure"] == "item_clustered_t")
            & (sub_n2a_005["stratum"] == "high_var_null")
        ]["raw_type1_rate"].iloc[0]
    )
    clust_lv = float(
        sub_n2a_005[
            (sub_n2a_005["procedure"] == "item_clustered_t")
            & (sub_n2a_005["stratum"] == "low_var_null")
        ]["raw_type1_rate"].iloc[0]
    )
    assert (
        rcbd_hv > 0.30
    ), f"Expected severe anti-conservatism on high_var_null, got {rcbd_hv}"
    assert (
        rcbd_lv < 0.015
    ), f"Expected severe conservatism on low_var_null, got {rcbd_lv}"
    assert 0.035 <= clust_hv <= 0.065
    assert 0.035 <= clust_lv <= 0.065

    # 4. Empirically anchored MMLU N3b & N4a expose real RCBD Type-I inflation (~11-13% overall, >25% Factor C)
    for reg_emp in ("N3b_contrast_rademacher_null", "N4a_bootstrap_centered_null"):
        sub_emp = df[
            (df["regime_id"] == reg_emp)
            & (df["cohort"] == "MMLU")
            & (df["estimator"] == "cell_mean")
            & np.isclose(df["nominal_alpha"], 0.05)
        ]
        rcbd_all = float(
            sub_emp[
                (sub_emp["procedure"] == "fixed_block_rcbd")
                & (sub_emp["stratum"] == "all_contrasts")
            ]["raw_type1_rate"].iloc[0]
        )
        rcbd_fc = float(
            sub_emp[
                (sub_emp["procedure"] == "fixed_block_rcbd")
                & (sub_emp["stratum"] == "factor_C_multiline")
            ]["raw_type1_rate"].iloc[0]
        )
        clust_all = float(
            sub_emp[
                (sub_emp["procedure"] == "item_clustered_t")
                & (sub_emp["stratum"] == "all_contrasts")
            ]["raw_type1_rate"].iloc[0]
        )
        assert rcbd_all > 0.10
        assert rcbd_fc > 0.25
        assert 0.040 <= clust_all <= 0.060

    # 5. Empirically anchored N4b (80% null + 20% planted signal) verifies BH/BY FDR control & B=200 vs B=2000 resolution
    sub_n4b = df[
        (df["regime_id"] == "N4b_bootstrap_partial_null_80_20")
        & (df["cohort"] == "MMLU")
        & (df["estimator"] == "cell_mean")
        & (df["stratum"] == "all_contrasts")
        & np.isclose(df["nominal_alpha"], 0.05)
    ].set_index("procedure")

    assert float(sub_n4b.loc["item_clustered_t", "bh_global_fdr"]) <= 0.060
    assert float(sub_n4b.loc["item_clustered_t", "by_global_fdr"]) <= 0.020
    assert float(sub_n4b.loc["item_clustered_t", "bh_global_power"]) >= 0.85
    assert float(sub_n4b.loc["fixed_block_rcbd", "bh_global_fdr"]) >= 0.18
    assert float(sub_n4b.loc["unblocked", "bh_global_power"]) <= 0.20

    # B=200 hits zero power under global BY on MMLU (min_p=1/201 > BY cutoff), whereas B=2000 recovers high power!
    assert float(sub_n4b.loc["sign_flip_200", "by_global_power"]) == 0.0
    assert float(sub_n4b.loc["sign_flip_2000", "by_global_power"]) >= 0.70


@pytest.mark.skipif(
    not (DATA_DIR / "Ys.pickle").exists(),
    reason="Requires gitignored local PromptEval Ys.pickle dataset",
)
def test_fixed_seed_replay_reproduces_exact_artifact_hashes(tmp_path: Path):
    expected_csv_hash = sha256_file(RESULTS_DIR / "inference_calibration.csv")
    expected_json_hash = sha256_file(RESULTS_DIR / "inference_calibration_summary.json")

    run_inference_calibration(output_dir=tmp_path)

    replay_csv_hash = sha256_file(tmp_path / "inference_calibration.csv")
    replay_json_hash = sha256_file(tmp_path / "inference_calibration_summary.json")

    assert replay_csv_hash == expected_csv_hash
    assert replay_json_hash == expected_json_hash
