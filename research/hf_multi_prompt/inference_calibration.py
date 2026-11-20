"""
Phase 0 Task R3: Item-Clustered Inference Calibration Under Registered Nulls and Partial Nulls.

Replaces "more detections" as implicit evidence of a superior test with explicit
Type-I error calibration curves across nominal alpha in {0.005, 0.01, 0.025, 0.05, 0.10}
and Benjamini-Hochberg (BH) / Benjamini-Yekutieli (BY) family-level FDR and FWER
evaluations (per-cube, per-model, and global cohort) with replicate-clustered Monte Carlo
standard errors across five inference procedure configurations:
  1. `unblocked`: Unblocked fixed-effects ANOVA F-test,
  2. `item_clustered_t`: Primary asymptotic item-clustered contrast one-sample t-test,
  3. `sign_flip_200`: Item-level Rademacher sign-flip randomization test (B = 200),
  4. `sign_flip_2000`: Item-level Rademacher sign-flip randomization test (B = 2000),
  5. `fixed_block_rcbd`: Exploratory additive fixed-block RCBD ANOVA F-test.

Evaluates four registered null and partial-null regimes:
  - Regime N1: Synthetic additive item-difficulty Bernoulli null (standard N=200 & small N=26),
  - Regime N2: Synthetic heterogeneous contrast-flip variance + non-null 2FI nuisance
    under complete main-effect null (N2a) and planted single-factor semi-null (N2b),
  - Regime N3: Empirically anchored sharp-null within-item cell permutation (N3a) and
    de-meaned empirical contrast Rademacher sign randomization (N3b) on real MMLU & SLAB cubes,
  - Regime N4: Empirically anchored item-bootstrap centered complete null (N4a, including
    standard-N vs 26-item `word_not_containing` stratification) and 80%-null / 20%-planted
    partial-null FDR & power audit (N4b) on real MMLU & SLAB cubes.
"""

from __future__ import annotations

import itertools
import json
import pickle
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels

from research.hf_multi_prompt.run_experiments import (
    BBH_MODELS,
    DATA_DIR,
    LMENTRY_MODELS,
    MMLU_MODELS,
    RESULTS_DIR,
    ROOT,
    build_cube_design_matrices,
    build_mmlu_mapping_table,
    build_slab_mapping_table_for_task,
    extract_cell_item_matrix,
    extract_unique_cell_map,
    sha256_file,
    verify_source_hashes,
)

NOMINAL_ALPHAS: Tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.10)

PROCEDURES: Tuple[str, ...] = (
    "unblocked",
    "item_clustered_t",
    "sign_flip_200",
    "sign_flip_2000",
    "fixed_block_rcbd",
)

SIGN_FLIP_B_MAP: Dict[str, Optional[int]] = {
    "unblocked": None,
    "item_clustered_t": None,
    "sign_flip_200": 200,
    "sign_flip_2000": 2000,
    "fixed_block_rcbd": None,
}

CALIBRATION_SEEDS: Dict[str, int] = {
    "n1_standard_data": 20260301,
    "n1_standard_sf200": 20260302,
    "n1_standard_sf2000": 20260303,
    "n1_small26_data": 20260304,
    "n1_small26_sf200": 20260305,
    "n1_small26_sf2000": 20260306,
    "n2a_hetero_data": 20260311,
    "n2a_hetero_sf200": 20260312,
    "n2a_hetero_sf2000": 20260313,
    "n2b_semi_data": 20260314,
    "n2b_semi_sf200": 20260315,
    "n2b_semi_sf2000": 20260316,
    "n3a_cell_perm": 20260321,
    "n3a_sf200": 20260322,
    "n3a_sf2000": 20260323,
    "n3b_contrast_rad": 20260324,
    "n3b_sf200": 20260325,
    "n3b_sf2000": 20260326,
    "n4a_item_boot": 20260331,
    "n4a_sf200": 20260332,
    "n4a_sf2000": 20260333,
    "n4b_planted_design": 20260341,
    "n4b_item_boot": 20260342,
    "n4b_sf200": 20260343,
    "n4b_sf2000": 20260344,
    "observed_sf200": 20260351,
    "observed_sf2000": 20260352,
}

DEFAULT_REPLICATE_COUNTS: Dict[str, int] = {
    "n1_standard": 150,
    "n1_small26": 200,
    "n2a_hetero": 150,
    "n2b_semi": 150,
    "empirical_mmlu": 50,
    "empirical_slab": 120,
}

CALIBRATION_TOLERANCE_GATES: Dict[str, Any] = {
    "primary_alpha": 0.05,
    "standard_n_min_items": 100,
    "raw_alpha_005_point_max": 0.060,
    "raw_alpha_005_mc_upper95_max": 0.065,
    "raw_mc_upper95_ceilings_by_alpha": {
        "0.005": 0.0085,
        "0.01": 0.0155,
        "0.025": 0.0340,
        "0.05": 0.0650,
        "0.1": 0.1220,
    },
    "severe_conservatism_alpha_005_min": 0.025,
    "bh_by_fdr_005_point_max_per_cube_or_model": 0.065,
    "bh_by_fdr_005_mc_upper95_max_per_cube_or_model": 0.080,
    "bh_by_fdr_005_point_max_n4a_per_cube": 0.062,
    "bh_by_fdr_005_mc_upper95_max_n4a_per_cube": 0.065,
    "bh_by_fdr_005_point_max_global_partial_null": 0.060,
    "bh_by_fdr_005_mc_upper95_max_global_partial_null": 0.065,
    "bh_by_fdr_005_point_max_global_complete_null": 0.090,
    "planted_power_005_min_bh_per_cube": 0.70,
    "planted_power_005_min_bh_global": 0.50,
}

REGIME_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "N1_additive_null": {
        "regime_family": "N1",
        "title": "Synthetic Additive Item-Difficulty Bernoulli Null (Standard N=200)",
        "dgp": (
            "theta_i ~ Normal(0.40, 2.20^2) i.i.d. across items i=1..N; "
            "Y_{c,i} | theta_i ~ Bernoulli(sigmoid(theta_i)) independently across K=32 cells; "
            "all main and interaction effects strictly zero."
        ),
        "exchangeability_assumptions": (
            "Full within-item cell exchangeability holds: conditional on item difficulty theta_i, "
            "(Y_{1,i}, ..., Y_{K,i}) is invariant under all K! cell permutations. Every orthogonal "
            "contrast c_{j,i} satisfies E[c_{j,i}|theta_i] = 0, conditional sign symmetry "
            "c_{j,i} =^d -c_{j,i}, joint {+1,-1}^N Rademacher invariance across independent items, "
            "and homogeneous variance across all K-1 contrasts."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t",
            "sign_flip_200",
            "sign_flip_2000",
            "fixed_block_rcbd",
        ],
    },
    "N1_small_N26_null": {
        "regime_family": "N1",
        "title": "Synthetic Additive Item-Difficulty Bernoulli Null (Small N=26)",
        "dgp": (
            "Same exchangeable item-difficulty Bernoulli DGP at K=16, N=26 (matching LMentry "
            "word_not_containing item count) to isolate small-N degrees-of-freedom effects under "
            "symmetric contrasts."
        ),
        "exchangeability_assumptions": (
            "Full K! within-item cell exchangeability and exact joint {+1,-1}^N sign symmetry hold; "
            "tests finite-sample Student-t (df=25) approximation when contrasts are symmetric."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t",
            "sign_flip_200",
            "sign_flip_2000",
            "fixed_block_rcbd",
        ],
    },
    "N2a_hetero_2fi_null": {
        "regime_family": "N2",
        "title": "Synthetic Heterogeneous Contrast-Flip Variance + Non-Null 2FI Complete Main-Effect Null",
        "dgp": (
            "Binary thresholded latent margin Y_{c,i} = 1{eta_{c,i} > 0} on K=32, N=200 with "
            "theta_i ~ Normal(0.40, 3.80^2), contrast-specific random item slopes "
            "gamma = (0.65, 0.30, 0.30, 0.08, 0.05) * Normal(0,1), fixed non-null 2FI delta_{23} = 0.40, "
            "random item-2FI gamma_{23} = 0.55, cell noise sigma_eps = 0.50, and tau_1 = ... = tau_5 = 0."
        ),
        "exchangeability_assumptions": (
            "Full K! within-item cell exchangeability FAILS because cells share factor-specific random "
            "slopes and a non-zero 2FI on (x_2, x_3). However, cube involutions x_j -> -x_j (for j in {1,4,5}) "
            "and simultaneous (x_2, x_3) -> (-x_2, -x_3) preserve x_2*x_3 and induce exact marginal and "
            "joint {+1,-1}^N sign symmetry on every main-effect contrast c_{j,i} with E[c_{j,i}] = 0. "
            "Heterogeneous Var(c_{j,i}) and unmodeled 2FI violate fixed_block_rcbd homoscedasticity."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t",
            "sign_flip_200",
            "sign_flip_2000",
        ],
    },
    "N2b_planted_semi_null": {
        "regime_family": "N2",
        "title": "Synthetic Planted Single-Factor Alternative + Heterogeneous Nulls + Non-Null 2FI",
        "dgp": (
            "Same K=32, N=200 binary latent-margin DGP with true non-null main effect on Factor 1 "
            "(mu_1 = 0.24, pi_1 = 0.20) while Factors 2..5 have exact tau_j = 0 (pi_0 = 0.80) under "
            "heterogeneous item-flip slopes gamma = (0.30, 0.65, 0.30, 0.30, 0.06) and non-null 2FI on (x_3, x_4)."
        ),
        "exchangeability_assumptions": (
            "Full cell exchangeability FAILS. Null contrasts j in {2,3,4,5} remain invariant under "
            "factor involutions x_2 -> -x_2, (x_3,x_4) -> (-x_3,-x_4), and x_5 -> -x_5 that hold x_1 "
            "and x_3*x_4 fixed, preserving E[c_{j,i}] = 0 and joint {+1,-1}^N sign symmetry on H_0."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t",
            "sign_flip_200",
            "sign_flip_2000",
        ],
    },
    "N3a_cell_perm_null": {
        "regime_family": "N3",
        "title": "Empirically Anchored Within-Item Uniform Cell Permutation Sharp Null (MMLU & SLAB)",
        "dgp": (
            "Permutes the K cell labels uniformly at random within each item i (shared across models "
            "evaluated on the same task item to preserve cross-model dependence), Y^{perm}_{c,i} = Y_{pi_i(c),i}."
        ),
        "exchangeability_assumptions": (
            "Enforces the sharp null of full K! within-item cell exchangeability conditional on the "
            "observed benchmark item responses, preserving exact per-item difficulty y_bar_{.i} and "
            "within-item marginal distribution, while homogenizing variance across all K-1 contrasts."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t",
            "sign_flip_200",
            "sign_flip_2000",
            "fixed_block_rcbd",
        ],
    },
    "N3b_contrast_rademacher_null": {
        "regime_family": "N3",
        "title": "Empirically Anchored De-Meaned Contrast Rademacher Sign-Randomization Null (MMLU & SLAB)",
        "dgp": (
            "De-means empirical item-level main-effect contrasts c~_{j,i} = c_{j,i} - c_bar_{j.}, "
            "multiplies by task-level i.i.d. Rademacher signs s_i in {-1,+1}, and reconstructs "
            "Y^{rad}_{c,i} = y_bar_{.i} + 0.5 * sum_j X_{c,j}(s_i c~_{j,i}) + HO_{c,i}."
        ),
        "exchangeability_assumptions": (
            "Does NOT assume K! cell exchangeability; instead enforces exact joint {+1,-1}^N "
            "sign-symmetry across items on the de-meaned main-effect contrast vector while preserving "
            "exact per-item difficulty y_bar_{.i}, empirical contrast heteroscedasticity s_{c_j}^2, "
            "within-item cross-contrast covariance, zero-variance contrasts, and higher-order residuals."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t",
            "sign_flip_200",
            "sign_flip_2000",
        ],
    },
    "N4a_bootstrap_centered_null": {
        "regime_family": "N4",
        "title": "Empirically Anchored Item-Bootstrap Centered Complete Null (MMLU, SLAB Standard-N & Small-N=26)",
        "dgp": (
            "Centers each cube's empirical main-effect contrasts Y~ = Y - 0.5 * X_main * c_bar so "
            "E^*[c^*_{j,i}] = 0 under the empirical item distribution, then resamples N_t task items "
            "with replacement (shared across models on task t)."
        ),
        "exchangeability_assumptions": (
            "Assumes only i.i.d. item sampling from the centered empirical task distribution (H_0: E[c_{j,i}] = 0) "
            "WITHOUT imposing within-item cell exchangeability or symmetric contrast distributions. "
            "Preserves discrete boundedness, empirical skewness, small-N tasks (N=26 word_not_containing), "
            "zero-variance contrasts, cross-contrast correlation, and cross-model item dependence."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t (asymptotic large-N; sensitive to small-N skewness)",
        ],
    },
    "N4b_bootstrap_partial_null_80_20": {
        "regime_family": "N4",
        "title": "Empirically Anchored Item-Bootstrap Partial-Null (80% Null + 20% Planted Signal) FDR Audit",
        "dgp": (
            "Starts from centered empirical cubes Y~ (pi_0 = 0.80 exact on MMLU, ~0.80 on SLAB) and "
            "plants known additive main-effect shifts Delta_tau = +-4.2 * max(s_{c_j}, 0.015) / sqrt(N_t) "
            "on a frozen 20% subset of non-zero-variance contrasts per model before task-level item bootstrap."
        ),
        "exchangeability_assumptions": (
            "Same nonparametric item-sampling model as N4a on the 80% null contrasts, coupled with "
            "realistic non-null alternatives (pi_1 = 0.20) under empirical cross-contrast and cross-model "
            "correlation to audit actual BH and BY false discovery rates and statistical power."
        ),
        "valid_procedures_by_theory": [
            "item_clustered_t (asymptotic large-N)",
        ],
    },
}


# ==============================================================================
# 1. Core Vectorized Inference & Multiplicity Primitives
# ==============================================================================

def _draw_packed_rademacher_bits(
    rng: np.random.Generator,
    n_perm: int,
    n_items: int,
) -> np.ndarray:
    """Deterministically generate a (n_perm, n_items) {0.0, 1.0} float64 matrix via PCG64 bytes."""
    n_bits = n_perm * n_items
    n_bytes = (n_bits + 7) // 8
    raw = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
    return np.unpackbits(raw)[:n_bits].reshape(n_perm, n_items).astype(np.float64)


def adjust_pvalues_2d(p_matrix: np.ndarray, method: str) -> np.ndarray:
    """
    Vectorized Benjamini-Hochberg (`fdr_bh`) and Benjamini-Yekutieli (`fdr_by`)
    step-up adjusted p-values across rows of shape `(n_families, m)`.
    Matches `statsmodels.stats.multitest.multipletests(..., method=...)[1]` to 1e-15.
    """
    arr = np.asarray(p_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Expected non-empty 2D p_matrix, got shape {arr.shape}")
    m = arr.shape[1]
    ranks = np.arange(1, m + 1, dtype=float)
    if method == "fdr_bh":
        scale = m / ranks
    elif method == "fdr_by":
        hm = float(np.sum(1.0 / ranks))
        scale = (m * hm) / ranks
    else:
        raise ValueError(f"Unsupported multiplicity method: {method!r}")

    order = np.argsort(arr, axis=1, kind="mergesort")
    p_sorted = np.take_along_axis(arr, order, axis=1)
    scaled = p_sorted * scale[None, :]
    q_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[:, ::-1], axis=1)[:, ::-1])
    q_out = np.empty_like(arr)
    np.put_along_axis(q_out, order, q_sorted, axis=1)
    return q_out


def evaluate_cubes_batch_inference(
    Y_batch: np.ndarray,
    X_pm: np.ndarray,
    rng_sf200: Optional[np.random.Generator] = None,
    rng_sf2000: Optional[np.random.Generator] = None,
    se_scale_cluster: float = 1.0,
    pvalue_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Vectorized batch evaluation of all five inference configurations across `(B_cubes, K, N)` cubes.
    Matches `compute_cube_rcbd_and_robust_inference` in `run_experiments.py` bit-for-bit on
    `unblocked`, `fixed_block_rcbd`, and `item_clustered_t` when `se_scale_cluster == 1.0`
    and `pvalue_transform is None`.
    """
    arr = np.asarray(Y_batch, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if np.isnan(arr).any():
        raise ValueError("Missing values detected in Y_batch; complete item blocks required.")
    B_cubes, K, N = arr.shape
    num_factors = X_pm.shape[1]
    if K != X_pm.shape[0]:
        raise ValueError(f"Row mismatch between Y_batch ({K}) and X_pm ({X_pm.shape[0]})")
    if N < 2:
        raise ValueError(f"At least 2 item blocks required, got N={N}")
    if se_scale_cluster <= 0.0:
        raise ValueError(f"se_scale_cluster must be positive, got {se_scale_cluster}")

    scale = K / 2.0
    C_item = np.einsum("ck,bcn->bkn", X_pm, arr) / scale  # (B_cubes, k, N)
    taus = C_item.mean(axis=2)  # (B_cubes, k)

    y_bar = arr.mean(axis=(1, 2), keepdims=True)  # (B_cubes, 1, 1)
    ss_tot = np.sum((arr - y_bar) ** 2, axis=(1, 2))  # (B_cubes,)
    item_means = arr.mean(axis=1)  # (B_cubes, N)
    ss_item = K * np.sum((item_means - y_bar[:, 0, :]) ** 2, axis=1)  # (B_cubes,)
    ss_factors = (K * N / 4.0) * (taus ** 2)  # (B_cubes, k)
    ss_model = np.sum(ss_factors, axis=1)  # (B_cubes,)

    df_u = K * N - 1 - num_factors
    df_b = K * N - 1 - num_factors - (N - 1)

    sd_c = C_item.std(axis=2, ddof=1)  # (B_cubes, k)
    zero_var_mask = sd_c <= 1e-15
    valid_tot = ss_tot > 1e-9

    mse_u = np.where(valid_tot, (ss_tot - ss_model) / df_u, 0.0)
    mse_b = np.where(valid_tot, (ss_tot - ss_model - ss_item) / df_b, 0.0)
    estimable_u = valid_tot & np.isfinite(mse_u) & (mse_u > 1e-12)
    estimable_b = valid_tot & np.isfinite(mse_b) & (mse_b > 1e-12)

    p_unblocked = np.ones((B_cubes, num_factors), dtype=float)
    if np.any(estimable_u):
        p_unblocked[estimable_u] = stats.f.sf(
            ss_factors[estimable_u] / mse_u[estimable_u, None], 1, df_u
        )

    p_rcbd = np.ones((B_cubes, num_factors), dtype=float)
    if np.any(estimable_b):
        p_rcbd[estimable_b] = stats.f.sf(
            ss_factors[estimable_b] / mse_b[estimable_b, None], 1, df_b
        )

    p_cluster = np.ones((B_cubes, num_factors), dtype=float)
    valid_cluster = valid_tot[:, None] & (~zero_var_mask)
    if np.any(valid_cluster):
        se_c = (sd_c[valid_cluster] / np.sqrt(N)) * se_scale_cluster
        t_vals = np.abs(taus[valid_cluster]) / se_c
        p_cluster[valid_cluster] = 2.0 * stats.t.sf(t_vals, df=N - 1)

    mse_reduction_pct = np.where(
        estimable_u & estimable_b,
        (mse_u - mse_b) / np.maximum(1e-30, mse_u) * 100.0,
        0.0,
    )
    f_over_t2 = np.where(
        estimable_b[:, None] & (~zero_var_mask),
        ((K / 4.0) * (sd_c ** 2)) / np.maximum(1e-30, mse_b[:, None]),
        1.0,
    )

    if rng_sf200 is None:
        rng_sf200 = np.random.default_rng(CALIBRATION_SEEDS["observed_sf200"])
    if rng_sf2000 is None:
        rng_sf2000 = np.random.default_rng(CALIBRATION_SEEDS["observed_sf2000"])

    C_flat = C_item.reshape(B_cubes * num_factors, N)
    c_sums = C_flat.sum(axis=1)[None, :]
    obs_abs = np.abs(taus).reshape(1, B_cubes * num_factors)
    invalid_perm = zero_var_mask | (~valid_tot[:, None])

    bits_200 = _draw_packed_rademacher_bits(rng_sf200, 200, N)
    perm_200 = np.abs(2.0 * (bits_200 @ C_flat.T) - c_sums) / N
    p_sf200 = ((1.0 + np.sum(perm_200 >= obs_abs - 1e-15, axis=0)) / 201.0).reshape(
        B_cubes, num_factors
    )
    p_sf200[invalid_perm] = 1.0

    bits_2000 = _draw_packed_rademacher_bits(rng_sf2000, 2000, N)
    perm_2000 = np.abs(2.0 * (bits_2000 @ C_flat.T) - c_sums) / N
    p_sf2000 = ((1.0 + np.sum(perm_2000 >= obs_abs - 1e-15, axis=0)) / 2001.0).reshape(
        B_cubes, num_factors
    )
    p_sf2000[invalid_perm] = 1.0

    out = {
        "unblocked": p_unblocked,
        "item_clustered_t": p_cluster,
        "sign_flip_200": p_sf200,
        "sign_flip_2000": p_sf2000,
        "fixed_block_rcbd": p_rcbd,
        "taus": taus,
        "zero_var_mask": zero_var_mask,
        "mse_reduction_pct": mse_reduction_pct,
        "f_rcbd_over_t_cluster_sq": f_over_t2,
    }
    if pvalue_transform is not None:
        for proc in PROCEDURES:
            out[proc] = np.clip(np.asarray(pvalue_transform(out[proc]), dtype=float), 0.0, 1.0)
    return out


# ==============================================================================
# 2. Multi-Family Accumulator Across Independent Monte Carlo Replicates
# ==============================================================================

def _mc_mean_and_se(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr))
    if arr.size <= 1:
        return mean_val, 0.0
    se_val = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean_val, se_val


def _family_fdp_fwe_stats(
    q_2d: np.ndarray,
    is_alt_2d: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given `q_2d` of shape `(n_families, m)` and boolean `is_alt_2d` of shape `(n_families, m)`,
    return `(mean_fdp_by_alpha, mean_fwe_by_alpha, sum_true_pos_by_alpha)` across the `n_families`.
    """
    is_null_2d = ~is_alt_2d
    n_alphas = len(alphas)
    fdp_means = np.empty(n_alphas, dtype=float)
    fwe_means = np.empty(n_alphas, dtype=float)
    tp_sums = np.empty(n_alphas, dtype=float)
    for a_i, alpha in enumerate(alphas):
        rej = q_2d < alpha
        r_tot = np.sum(rej, axis=1)
        v_false = np.sum(rej & is_null_2d, axis=1)
        s_true = np.sum(rej & is_alt_2d)
        fdp = np.where(r_tot > 0, v_false / np.maximum(1, r_tot), 0.0)
        fwe = (v_false >= 1).astype(float)
        fdp_means[a_i] = float(np.mean(fdp))
        fwe_means[a_i] = float(np.mean(fwe))
        tp_sums[a_i] = float(s_true)
    return fdp_means, fwe_means, tp_sums


def summarize_panel_replicates_to_rows(
    *,
    regime_id: str,
    cohort: str,
    estimator: str,
    p_replicates: Dict[str, np.ndarray],
    is_alt_mask: np.ndarray,
    cube_model_ids: Sequence[str],
    cube_n_items: np.ndarray,
    extra_contrast_strata: Optional[Dict[str, np.ndarray]] = None,
    alphas: Sequence[float] = NOMINAL_ALPHAS,
    gates: Dict[str, Any] = CALIBRATION_TOLERANCE_GATES,
) -> List[Dict[str, Any]]:
    """
    Convert stored Monte Carlo replicate p-value tensors `p_replicates[proc]` of shape
    `(R, n_cubes, k)` into tidy calibration curve & BH/BY multiplicity summary rows.
    """
    sample_arr = next(iter(p_replicates.values()))
    R, n_cubes, k = sample_arr.shape
    n_tests = n_cubes * k
    alt_2d = np.asarray(is_alt_mask, dtype=bool)
    if alt_2d.shape != (n_cubes, k):
        raise ValueError(f"is_alt_mask shape {alt_2d.shape} != ({n_cubes}, {k})")
    null_2d = ~alt_2d
    n_null_total = int(np.sum(null_2d))
    n_alt_total = int(np.sum(alt_2d))

    # Group cube indices by model_id (preserving shared models across BBH & LMentry)
    model_to_cube_idxs: Dict[str, List[int]] = {}
    for c_idx, m_id in enumerate(cube_model_ids):
        model_to_cube_idxs.setdefault(str(m_id), []).append(c_idx)
    # Partition models by family size m_model so adjust_pvalues_2d runs vectorized per size group
    size_to_model_cube_lists: Dict[int, List[List[int]]] = {}
    for _, c_list in sorted(model_to_cube_idxs.items()):
        size_to_model_cube_lists.setdefault(len(c_list) * k, []).append(c_list)
    n_model_families = len(model_to_cube_idxs)
    min_m_model = min(size_to_model_cube_lists.keys())

    min_items_n = int(np.min(cube_n_items))
    max_items_n = int(np.max(cube_n_items))
    regime_family = REGIME_SPECIFICATIONS[regime_id]["regime_family"]
    ceilings_map = gates["raw_mc_upper95_ceilings_by_alpha"]

    rows: List[Dict[str, Any]] = []

    for proc in PROCEDURES:
        P_all = p_replicates[proc]  # (R, n_cubes, k)
        b_perm = SIGN_FLIP_B_MAP[proc]
        min_ach_p = (1.0 / (b_perm + 1.0)) if b_perm is not None else float("nan")

        # 1. Primary `all_contrasts` stratum (raw + BH/BY across per_cube, per_model, global)
        raw_t1 = np.empty((R, len(alphas)), dtype=float)
        raw_pw = np.full((R, len(alphas)), np.nan, dtype=float)

        fam_keys = [
            "bh_per_cube",
            "by_per_cube",
            "bh_per_model",
            "by_per_model",
            "bh_global",
            "by_global",
        ]
        fam_fdr = {fk: np.empty((R, len(alphas)), dtype=float) for fk in fam_keys}
        fam_fwer = {fk: np.empty((R, len(alphas)), dtype=float) for fk in fam_keys}
        fam_power = {fk: np.full((R, len(alphas)), np.nan, dtype=float) for fk in fam_keys}

        alt_global = np.broadcast_to(alt_2d.reshape(1, n_tests), (R, n_tests))
        null_global = ~alt_global

        # Raw Type-I and power across all R replicates simultaneously
        P_null_all = P_all[:, null_2d]  # (R, n_null_total)
        P_alt_all = P_all[:, alt_2d] if n_alt_total > 0 else None
        for a_i, alpha in enumerate(alphas):
            raw_t1[:, a_i] = np.mean(P_null_all < alpha, axis=1)
            if P_alt_all is not None:
                raw_pw[:, a_i] = np.mean(P_alt_all < alpha, axis=1)

        # 1a. Per-cube BH & BY vectorized across (R * n_cubes, k)
        P_cubes_flat = P_all.reshape(R * n_cubes, k)
        alt_cubes_rep = np.broadcast_to(alt_2d[None, :, :], (R, n_cubes, k))
        null_cubes_rep = ~alt_cubes_rep
        for prefix, m_tag in [("bh", "fdr_bh"), ("by", "fdr_by")]:
            q_cubes = adjust_pvalues_2d(P_cubes_flat, m_tag).reshape(R, n_cubes, k)
            for a_i, alpha in enumerate(alphas):
                rej = q_cubes < alpha
                r_tot = np.sum(rej, axis=2)
                v_false = np.sum(rej & null_cubes_rep, axis=2)
                fdp = np.where(r_tot > 0, v_false / np.maximum(1, r_tot), 0.0)
                fwe = (v_false >= 1).astype(float)
                fam_fdr[f"{prefix}_per_cube"][:, a_i] = np.mean(fdp, axis=1)
                fam_fwer[f"{prefix}_per_cube"][:, a_i] = np.mean(fwe, axis=1)
                if n_alt_total > 0:
                    s_true = np.sum(rej & alt_cubes_rep, axis=(1, 2))
                    fam_power[f"{prefix}_per_cube"][:, a_i] = s_true / n_alt_total

        # 1b. Per-model BH & BY vectorized across (R * n_grp, m_grp) per model-size group
        for prefix, m_tag in [("bh", "fdr_bh"), ("by", "fdr_by")]:
            weighted_fdp = np.zeros((R, len(alphas)), dtype=float)
            weighted_fwe = np.zeros((R, len(alphas)), dtype=float)
            total_tp = np.zeros((R, len(alphas)), dtype=float)
            for m_size, c_lists in size_to_model_cube_lists.items():
                n_grp = len(c_lists)
                P_grp = np.stack(
                    [P_all[:, c_l, :].reshape(R, m_size) for c_l in c_lists], axis=1
                )  # (R, n_grp, m_size)
                alt_grp = np.broadcast_to(
                    np.stack([alt_2d[c_l].ravel() for c_l in c_lists], axis=0)[None, :, :],
                    (R, n_grp, m_size),
                )
                null_grp = ~alt_grp
                q_grp = adjust_pvalues_2d(
                    P_grp.reshape(R * n_grp, m_size), m_tag
                ).reshape(R, n_grp, m_size)
                for a_i, alpha in enumerate(alphas):
                    rej = q_grp < alpha
                    r_tot = np.sum(rej, axis=2)
                    v_false = np.sum(rej & null_grp, axis=2)
                    fdp = np.where(r_tot > 0, v_false / np.maximum(1, r_tot), 0.0)
                    fwe = (v_false >= 1).astype(float)
                    weighted_fdp[:, a_i] += np.sum(fdp, axis=1)
                    weighted_fwe[:, a_i] += np.sum(fwe, axis=1)
                    if n_alt_total > 0:
                        total_tp[:, a_i] += np.sum(rej & alt_grp, axis=(1, 2))
            fam_fdr[f"{prefix}_per_model"] = weighted_fdp / n_model_families
            fam_fwer[f"{prefix}_per_model"] = weighted_fwe / n_model_families
            if n_alt_total > 0:
                fam_power[f"{prefix}_per_model"] = total_tp / n_alt_total

        # 1c. Global cohort BH & BY vectorized across (R, n_tests)
        P_glob_all = P_all.reshape(R, n_tests)
        for prefix, m_tag in [("bh", "fdr_bh"), ("by", "fdr_by")]:
            q_glob = adjust_pvalues_2d(P_glob_all, m_tag)
            for a_i, alpha in enumerate(alphas):
                rej = q_glob < alpha
                r_tot = np.sum(rej, axis=1)
                v_false = np.sum(rej & null_global, axis=1)
                fam_fdr[f"{prefix}_global"][:, a_i] = np.where(
                    r_tot > 0, v_false / np.maximum(1, r_tot), 0.0
                )
                fam_fwer[f"{prefix}_global"][:, a_i] = (v_false >= 1).astype(float)
                if n_alt_total > 0:
                    fam_power[f"{prefix}_global"][:, a_i] = (
                        np.sum(rej & alt_global, axis=1) / n_alt_total
                    )

        idx_005 = list(alphas).index(0.05)
        t1_005_mean, _ = _mc_mean_and_se(raw_t1[:, idx_005])
        hm_cube = float(np.sum(1.0 / np.arange(1, k + 1, dtype=float)))

        for a_i, alpha in enumerate(alphas):
            t1_mean, t1_se = _mc_mean_and_se(raw_t1[:, a_i])
            t1_up95 = t1_mean + 1.96 * t1_se
            ceiling = float(ceilings_map[str(alpha)])
            if np.isclose(alpha, 0.05):
                passes_raw = bool(
                    (t1_mean <= gates["raw_alpha_005_point_max"]) and (t1_up95 <= ceiling)
                )
            else:
                passes_raw = bool(t1_up95 <= ceiling)

            if n_alt_total > 0:
                pw_mean, pw_se = _mc_mean_and_se(raw_pw[:, a_i])
            else:
                pw_mean, pw_se = float("nan"), float("nan")

            res_lim_global_bh = bool(
                (b_perm is not None) and ((alpha / n_tests) < min_ach_p)
            )
            res_lim_cube_by = bool(
                (b_perm is not None) and ((alpha / (k * hm_cube)) < min_ach_p)
            )
            res_lim_model_bh = bool(
                (b_perm is not None) and ((alpha / min_m_model) < min_ach_p)
            )

            row: Dict[str, Any] = {
                "regime_id": regime_id,
                "regime_family": regime_family,
                "cohort": cohort,
                "estimator": estimator,
                "stratum": "all_contrasts",
                "procedure": proc,
                "n_replicates": R,
                "n_cubes_per_rep": n_cubes,
                "n_model_families": n_model_families,
                "n_tests_per_rep": n_tests,
                "n_null_tests_per_rep": n_null_total,
                "n_alt_tests_per_rep": n_alt_total,
                "min_items_n": min_items_n,
                "max_items_n": max_items_n,
                "nominal_alpha": float(alpha),
                "raw_type1_rate": t1_mean,
                "raw_type1_mcse": t1_se,
                "raw_type1_mc_upper95": t1_up95,
                "raw_alpha_ceiling": ceiling,
                "passes_raw_anti_conservatism_gate": passes_raw,
                "is_severely_conservative_at_005": bool(
                    t1_005_mean < gates["severe_conservatism_alpha_005_min"]
                ),
                "raw_power_rate": pw_mean,
                "raw_power_mcse": pw_se,
                "sign_flip_min_achievable_p": min_ach_p,
                "resolution_limited_per_cube_by": res_lim_cube_by,
                "resolution_limited_per_model_bh": res_lim_model_bh,
                "resolution_limited_global_bh": res_lim_global_bh,
            }
            for fk in fam_keys:
                fdr_m, fdr_se = _mc_mean_and_se(fam_fdr[fk][:, a_i])
                fwe_m, fwe_se = _mc_mean_and_se(fam_fwer[fk][:, a_i])
                row[f"{fk}_fdr"] = fdr_m
                row[f"{fk}_fdr_mcse"] = fdr_se
                row[f"{fk}_fwer"] = fwe_m
                row[f"{fk}_fwer_mcse"] = fwe_se
                if n_alt_total > 0:
                    pw_f, _ = _mc_mean_and_se(fam_power[fk][:, a_i])
                    row[f"{fk}_power"] = pw_f
                else:
                    row[f"{fk}_power"] = float("nan")
            rows.append(row)

        # 2. Optional contrast-specific null strata (to expose variance-pooling miscalibration)
        if extra_contrast_strata:
            for stratum_name, stratum_mask in extra_contrast_strata.items():
                s_mask = np.asarray(stratum_mask, dtype=bool) & null_2d
                n_s_null = int(np.sum(s_mask))
                if n_s_null == 0:
                    continue
                s_t1 = np.empty((R, len(alphas)), dtype=float)
                for r in range(R):
                    p_sub = P_all[r][s_mask]
                    for a_i, alpha in enumerate(alphas):
                        s_t1[r, a_i] = float(np.mean(p_sub < alpha))
                s_005_mean, _ = _mc_mean_and_se(s_t1[:, idx_005])
                for a_i, alpha in enumerate(alphas):
                    t1_m, t1_s = _mc_mean_and_se(s_t1[:, a_i])
                    t1_u = t1_m + 1.96 * t1_s
                    ceiling = float(ceilings_map[str(alpha)])
                    if np.isclose(alpha, 0.05):
                        p_raw = bool((t1_m <= gates["raw_alpha_005_point_max"]) and (t1_u <= ceiling))
                    else:
                        p_raw = bool(t1_u <= ceiling)
                    s_row: Dict[str, Any] = {
                        "regime_id": regime_id,
                        "regime_family": regime_family,
                        "cohort": cohort,
                        "estimator": estimator,
                        "stratum": stratum_name,
                        "procedure": proc,
                        "n_replicates": R,
                        "n_cubes_per_rep": n_cubes,
                        "n_model_families": n_model_families,
                        "n_tests_per_rep": n_s_null,
                        "n_null_tests_per_rep": n_s_null,
                        "n_alt_tests_per_rep": 0,
                        "min_items_n": min_items_n,
                        "max_items_n": max_items_n,
                        "nominal_alpha": float(alpha),
                        "raw_type1_rate": t1_m,
                        "raw_type1_mcse": t1_s,
                        "raw_type1_mc_upper95": t1_u,
                        "raw_alpha_ceiling": ceiling,
                        "passes_raw_anti_conservatism_gate": p_raw,
                        "is_severely_conservative_at_005": bool(
                            s_005_mean < gates["severe_conservatism_alpha_005_min"]
                        ),
                        "raw_power_rate": float("nan"),
                        "raw_power_mcse": float("nan"),
                        "sign_flip_min_achievable_p": min_ach_p,
                        "resolution_limited_per_cube_by": False,
                        "resolution_limited_per_model_bh": False,
                        "resolution_limited_global_bh": False,
                    }
                    for fk in fam_keys:
                        s_row[f"{fk}_fdr"] = float("nan")
                        s_row[f"{fk}_fdr_mcse"] = float("nan")
                        s_row[f"{fk}_fwer"] = float("nan")
                        s_row[f"{fk}_fwer_mcse"] = float("nan")
                        s_row[f"{fk}_power"] = float("nan")
                    rows.append(s_row)

    return rows


# ==============================================================================
# 3. Synthetic Regimes N1 and N2
# ==============================================================================

def run_regime_n1_additive_null(
    *,
    regime_id: str,
    cohort_label: str,
    num_factors: int,
    n_items: int,
    n_models: int,
    n_tasks: int,
    n_replicates: int,
    seed_data: int,
    seed_sf200: int,
    seed_sf2000: int,
    mu_theta: float = 0.40,
    sigma_theta: float = 2.20,
    se_scale_cluster: float = 1.0,
    pvalue_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Regime N1: Synthetic Additive Item-Difficulty Null.
    Y_{c,i} ~ Bernoulli(sigmoid(theta_i)) independently across cells c=1..K given theta_i.
    """
    X_pm = build_cube_design_matrices(num_factors)["X_pm"]
    K = 1 << num_factors
    n_cubes = n_models * n_tasks
    cube_model_ids = [f"synth_model_{m:02d}" for m in range(n_models) for _ in range(n_tasks)]
    cube_n_items = np.full(n_cubes, n_items, dtype=int)

    rng_data = np.random.default_rng(seed_data)
    rng_200 = np.random.default_rng(seed_sf200)
    rng_2000 = np.random.default_rng(seed_sf2000)

    p_reps = {proc: np.empty((n_replicates, n_cubes, num_factors), dtype=float) for proc in PROCEDURES}
    mse_reds: List[float] = []

    for r in range(n_replicates):
        # Task-level item difficulties shared across models on the same synthetic task
        theta_tasks = rng_data.normal(mu_theta, sigma_theta, size=(n_tasks, 1, n_items))
        probs_tasks = 1.0 / (1.0 + np.exp(-theta_tasks))  # (n_tasks, 1, N)
        probs_panel = np.repeat(probs_tasks[None, :, :, :], n_models, axis=0).reshape(
            n_cubes, 1, n_items
        )
        Y_batch = (rng_data.random((n_cubes, K, n_items)) < probs_panel).astype(float)
        # Evaluate task-by-task so Rademacher flips match per-(replicate, task) structure
        for t_i in range(n_tasks):
            sl = slice(t_i, n_cubes, n_tasks)
            inf = evaluate_cubes_batch_inference(
                Y_batch[sl],
                X_pm,
                rng_sf200=rng_200,
                rng_sf2000=rng_2000,
                se_scale_cluster=se_scale_cluster,
                pvalue_transform=pvalue_transform,
            )
            for proc in PROCEDURES:
                p_reps[proc][r, sl, :] = inf[proc]
            mse_reds.append(float(np.mean(inf["mse_reduction_pct"])))

    is_alt = np.zeros((n_cubes, num_factors), dtype=bool)
    rows = summarize_panel_replicates_to_rows(
        regime_id=regime_id,
        cohort=cohort_label,
        estimator="synthetic_binary",
        p_replicates=p_reps,
        is_alt_mask=is_alt,
        cube_model_ids=cube_model_ids,
        cube_n_items=cube_n_items,
    )
    diag = {
        "mean_mse_reduction_pct": float(np.mean(mse_reds)),
        "mu_theta": mu_theta,
        "sigma_theta": sigma_theta,
    }
    return rows, diag


def run_regime_n2_heterogeneous_2fi(
    *,
    regime_id: str,
    cohort_label: str,
    planted_factor0_shift: float,
    n_models: int = 10,
    n_tasks: int = 12,
    n_items: int = 200,
    n_replicates: int = 150,
    seed_data: int = CALIBRATION_SEEDS["n2a_hetero_data"],
    seed_sf200: int = CALIBRATION_SEEDS["n2a_hetero_sf200"],
    seed_sf2000: int = CALIBRATION_SEEDS["n2a_hetero_sf2000"],
    se_scale_cluster: float = 1.0,
    pvalue_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Regime N2: Synthetic Heterogeneous Contrast-Specific Flip Variance + Non-Null 2FI.
    - When `planted_factor0_shift == 0.0` (N2a): All 5 main effects are strictly null
      (tau_1 = ... = tau_5 = 0), with:
        * j = 0: high-variance null contrast (gamma_0 = 0.65),
        * j = 1, 2: 2FI-coupled mid-variance null contrasts (gamma = 0.30, delta_23 = 0.40, gamma_23 = 0.55),
        * j = 3, 4: low-variance null contrasts (gamma = 0.08, 0.05).
    - When `planted_factor0_shift > 0.0` (N2b): Factor 0 has a true planted main effect
      (pi_1 = 0.20), while Factors 1..4 are strictly null (pi_0 = 0.80):
        * j = 0: planted true alternative (mu_0 = planted_factor0_shift, gamma_0 = 0.30),
        * j = 1: high-variance null contrast (gamma_1 = 0.65),
        * j = 2, 3: 2FI-coupled mid-variance null contrasts (gamma = 0.30, delta_34 = 0.40, gamma_34 = 0.55),
        * j = 4: low-variance null contrast (gamma_4 = 0.06).
    """
    num_factors = 5
    K = 32
    X_pm = build_cube_design_matrices(num_factors)["X_pm"]
    n_cubes = n_models * n_tasks
    cube_model_ids = [f"synth_model_{m:02d}" for m in range(n_models) for _ in range(n_tasks)]
    cube_n_items = np.full(n_cubes, n_items, dtype=int)

    is_semi_null = abs(planted_factor0_shift) > 1e-12
    if not is_semi_null:
        gammas = np.array([0.65, 0.30, 0.30, 0.08, 0.05], dtype=float)
        pair_cols = (1, 2)
        is_alt = np.zeros((n_cubes, num_factors), dtype=bool)
        extra_strata = {
            "high_var_null": np.broadcast_to(np.array([True, False, False, False, False]), (n_cubes, 5)),
            "twofi_coupled_null": np.broadcast_to(np.array([False, True, True, False, False]), (n_cubes, 5)),
            "low_var_null": np.broadcast_to(np.array([False, False, False, True, True]), (n_cubes, 5)),
        }
    else:
        gammas = np.array([0.30, 0.65, 0.30, 0.30, 0.06], dtype=float)
        pair_cols = (2, 3)
        is_alt = np.zeros((n_cubes, num_factors), dtype=bool)
        is_alt[:, 0] = True
        extra_strata = {
            "high_var_null": np.broadcast_to(np.array([False, True, False, False, False]), (n_cubes, 5)),
            "twofi_coupled_null": np.broadcast_to(np.array([False, False, True, True, False]), (n_cubes, 5)),
            "low_var_null": np.broadcast_to(np.array([False, False, False, False, True]), (n_cubes, 5)),
        }

    x_pair = (X_pm[:, pair_cols[0]] * X_pm[:, pair_cols[1]])[None, :, None]  # (1, K, 1)
    x_f0 = X_pm[:, 0][None, :, None]  # (1, K, 1)

    rng_data = np.random.default_rng(seed_data)
    rng_200 = np.random.default_rng(seed_sf200)
    rng_2000 = np.random.default_rng(seed_sf2000)

    p_reps = {proc: np.empty((n_replicates, n_cubes, num_factors), dtype=float) for proc in PROCEDURES}
    mse_reds: List[float] = []
    f_ratios: List[np.ndarray] = []

    for r in range(n_replicates):
        for t_i in range(n_tasks):
            # Shared base item difficulty across models on task t_i + model-specific item perturbation
            theta_base = rng_data.normal(0.40, 3.60, size=(1, 1, n_items))
            theta_mod = rng_data.normal(0.0, 1.20, size=(n_models, 1, n_items))
            Z = rng_data.normal(0.0, 1.0, size=(n_models, num_factors, n_items))
            scaled_Z = Z * gammas[None, :, None]
            contrast_term = np.einsum("ck,mkn->mcn", X_pm, scaled_Z)
            z_2fi = rng_data.normal(0.0, 0.55, size=(n_models, 1, n_items))
            sign_2fi = rng_data.choice([-1.0, 1.0], size=(n_models, 1, 1))
            sign_f0 = rng_data.choice([-1.0, 1.0], size=(n_models, 1, 1))
            eps = rng_data.normal(0.0, 0.50, size=(n_models, K, n_items))

            eta = (
                theta_base
                + theta_mod
                + sign_f0 * planted_factor0_shift * x_f0
                + contrast_term
                + sign_2fi * 0.40 * x_pair
                + x_pair * z_2fi
                + eps
            )
            Y_sub = (eta > 0.0).astype(float)
            inf = evaluate_cubes_batch_inference(
                Y_sub,
                X_pm,
                rng_sf200=rng_200,
                rng_sf2000=rng_2000,
                se_scale_cluster=se_scale_cluster,
                pvalue_transform=pvalue_transform,
            )
            sl = slice(t_i, n_cubes, n_tasks)
            for proc in PROCEDURES:
                p_reps[proc][r, sl, :] = inf[proc]
            mse_reds.append(float(np.mean(inf["mse_reduction_pct"])))
            f_ratios.append(np.mean(inf["f_rcbd_over_t_cluster_sq"], axis=0))

    rows = summarize_panel_replicates_to_rows(
        regime_id=regime_id,
        cohort=cohort_label,
        estimator="synthetic_binary",
        p_replicates=p_reps,
        is_alt_mask=is_alt,
        cube_model_ids=cube_model_ids,
        cube_n_items=cube_n_items,
        extra_contrast_strata=extra_strata,
    )
    diag = {
        "mean_mse_reduction_pct": float(np.mean(mse_reds)),
        "mean_f_rcbd_over_t_cluster_sq_by_factor": [
            float(x) for x in np.mean(np.array(f_ratios), axis=0)
        ],
        "planted_factor0_shift": float(planted_factor0_shift),
    }
    return rows, diag


# ==============================================================================
# 4. Empirically Anchored Regimes N3 and N4 on Real MMLU & SLAB Cubes
# ==============================================================================

def load_empirical_benchmark_panels(
    Ys: Dict[str, Any],
    estimators: Sequence[str] = ("legacy_first", "cell_mean", "cell_mean_unique"),
) -> Dict[str, Any]:
    """
    Extract all 855 MMLU cubes (15 models x 57 tasks, K=32) and all 178 completeness-selected
    SLAB cubes (6 BBH tasks x 11 models + 7 LMentry tasks x 16 models, K=16) using the exact
    template mappings from `run_experiments.py`.
    """
    df_mmlu_map, mmlu_leg_map, mmlu_canon_map = build_mmlu_mapping_table()
    mmlu_uniq_map = extract_unique_cell_map(df_mmlu_map)
    all_cells_5 = sorted(mmlu_leg_map.keys())
    all_cells_4 = list(itertools.product([0, 1], repeat=4))

    mmlu_maps = {
        "legacy_first": mmlu_leg_map,
        "cell_mean": mmlu_canon_map,
        "cell_mean_unique": mmlu_uniq_map,
    }
    tasks_mmlu = sorted(Ys["MMLU"].keys())
    mmlu_model_ids = [m[0] for m in MMLU_MODELS]

    # Organize by task so all models sharing a task's items are stacked as (n_models_task, K, N_task)
    mmlu_by_est: Dict[str, List[Dict[str, Any]]] = {est: [] for est in estimators}
    for t_name in tasks_mmlu:
        for est in estimators:
            mats = np.array(
                [
                    extract_cell_item_matrix(
                        Ys["MMLU"][t_name][m_i], all_cells_5, mmlu_maps[est], estimator=est
                    )
                    for m_i in range(len(MMLU_MODELS))
                ],
                dtype=float,
            )  # (15, 32, N_t)
            mmlu_by_est[est].append(
                {
                    "cohort": "MMLU",
                    "benchmark": "MMLU",
                    "task": t_name,
                    "model_ids": list(mmlu_model_ids),
                    "n_items": int(mats.shape[2]),
                    "Y": mats,
                }
            )

    slab_by_est: Dict[str, List[Dict[str, Any]]] = {est: [] for est in estimators}
    for bench, model_list in [("BBH", BBH_MODELS), ("LMentry", LMENTRY_MODELS)]:
        for fpath in sorted((DATA_DIR / "templates" / bench).glob("*.csv")):
            df_t = pd.read_csv(fpath)
            df_m, lmap, cmap = build_slab_mapping_table_for_task(df_t, bench, fpath.stem)
            if len(lmap) < 16:
                continue
            umap = extract_unique_cell_map(df_m)
            t_maps = {
                "legacy_first": lmap,
                "cell_mean": cmap,
                "cell_mean_unique": umap,
            }
            for est in estimators:
                mats = np.array(
                    [
                        extract_cell_item_matrix(
                            Ys[bench][fpath.stem][m_i], all_cells_4, t_maps[est], estimator=est
                        )
                        for m_i in range(len(model_list))
                    ],
                    dtype=float,
                )  # (n_m, 16, N_t)
                slab_by_est[est].append(
                    {
                        "cohort": bench,
                        "benchmark": bench,
                        "task": fpath.stem,
                        "model_ids": list(model_list),
                        "n_items": int(mats.shape[2]),
                        "Y": mats,
                    }
                )

    return {
        "MMLU": mmlu_by_est,
        "SLAB": slab_by_est,
        "X5_pm": build_cube_design_matrices(5)["X_pm"],
        "X4_pm": build_cube_design_matrices(4)["X_pm"],
    }


def _flatten_task_blocks_metadata(
    task_blocks: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, int]], List[str], np.ndarray]:
    slices: List[Tuple[int, int]] = []
    model_ids: List[str] = []
    n_items_list: List[int] = []
    cursor = 0
    for blk in task_blocks:
        m_count = len(blk["model_ids"])
        slices.append((cursor, cursor + m_count))
        cursor += m_count
        model_ids.extend(blk["model_ids"])
        n_items_list.extend([blk["n_items"]] * m_count)
    return slices, model_ids, np.array(n_items_list, dtype=int)


def run_regime_n3_sharp_randomization(
    *,
    panels: Dict[str, Any],
    estimators: Sequence[str] = ("legacy_first", "cell_mean"),
    n_reps_mmlu: int = DEFAULT_REPLICATE_COUNTS["empirical_mmlu"],
    n_reps_slab: int = DEFAULT_REPLICATE_COUNTS["empirical_slab"],
) -> List[Dict[str, Any]]:
    """
    Regime N3 on real MMLU (855 cubes) and SLAB (178 cubes):
      - N3a (`N3a_cell_perm_null`): Within-item uniform K!-cell permutation (shared across models
        on the same task item), preserving exact y_bar_{.i} and homogenizing contrast variances.
      - N3b (`N3b_contrast_rademacher_null`): Item-level Rademacher sign randomization on
        de-meaned empirical main-effect contrasts c~_{j,i} = c_{j,i} - c_bar_{j.}, preserving
        exact y_bar_{.i}, empirical contrast heteroscedasticity, zero-variance contrasts, and HO residuals.
    """
    rng_perm = np.random.default_rng(CALIBRATION_SEEDS["n3a_cell_perm"])
    rng_a_200 = np.random.default_rng(CALIBRATION_SEEDS["n3a_sf200"])
    rng_a_2000 = np.random.default_rng(CALIBRATION_SEEDS["n3a_sf2000"])

    rng_rad = np.random.default_rng(CALIBRATION_SEEDS["n3b_contrast_rad"])
    rng_b_200 = np.random.default_rng(CALIBRATION_SEEDS["n3b_sf200"])
    rng_b_2000 = np.random.default_rng(CALIBRATION_SEEDS["n3b_sf2000"])

    all_rows: List[Dict[str, Any]] = []

    for cohort_name, X_pm, n_reps, high_var_col, high_var_label in [
        ("MMLU", panels["X5_pm"], n_reps_mmlu, 2, "factor_C_multiline"),
        ("SLAB", panels["X4_pm"], n_reps_slab, 0, "factor_A_schema"),
    ]:
        num_factors = X_pm.shape[1]
        scale = X_pm.shape[0] / 2.0
        K = X_pm.shape[0]

        for est in estimators:
            blocks = panels[cohort_name][est]
            slices, model_ids, cube_n_items = _flatten_task_blocks_metadata(blocks)
            n_cubes = len(model_ids)
            is_alt = np.zeros((n_cubes, num_factors), dtype=bool)

            # Precompute de-meaned contrast decomposition for N3b
            n3b_precomp = []
            for blk in blocks:
                Y_blk = blk["Y"]  # (M_b, K, N)
                C_blk = np.einsum("ck,mcn->mkn", X_pm, Y_blk) / scale  # (M_b, k, N)
                C_tilde = C_blk - C_blk.mean(axis=2, keepdims=True)
                Y_no_main = Y_blk - 0.5 * np.einsum("ck,mkn->mcn", X_pm, C_blk)
                n3b_precomp.append((Y_no_main, C_tilde))

            p_n3a = {proc: np.empty((n_reps, n_cubes, num_factors), dtype=float) for proc in PROCEDURES}
            p_n3b = {proc: np.empty((n_reps, n_cubes, num_factors), dtype=float) for proc in PROCEDURES}

            for r in range(n_reps):
                for b_idx, blk in enumerate(blocks):
                    s0, s1 = slices[b_idx]
                    Y_blk = blk["Y"]
                    N_t = blk["n_items"]

                    # N3a: shared within-item cell permutation across models on task b_idx
                    perm_idx = np.argsort(rng_perm.random((K, N_t)), axis=0)  # (K, N_t)
                    Y_perm = np.take_along_axis(Y_blk, perm_idx[None, :, :], axis=1)
                    inf_a = evaluate_cubes_batch_inference(
                        Y_perm, X_pm, rng_sf200=rng_a_200, rng_sf2000=rng_a_2000
                    )
                    for proc in PROCEDURES:
                        p_n3a[proc][r, s0:s1, :] = inf_a[proc]

                    # N3b: shared item-level Rademacher sign flip on de-meaned empirical contrasts
                    Y_no_main, C_tilde = n3b_precomp[b_idx]
                    signs = 2.0 * _draw_packed_rademacher_bits(rng_rad, 1, N_t) - 1.0  # (1, N_t)
                    C_rad = C_tilde * signs[None, :, :]
                    Y_rad = Y_no_main + 0.5 * np.einsum("ck,mkn->mcn", X_pm, C_rad)
                    inf_b = evaluate_cubes_batch_inference(
                        Y_rad, X_pm, rng_sf200=rng_b_200, rng_sf2000=rng_b_2000
                    )
                    for proc in PROCEDURES:
                        p_n3b[proc][r, s0:s1, :] = inf_b[proc]

            all_rows.extend(
                summarize_panel_replicates_to_rows(
                    regime_id="N3a_cell_perm_null",
                    cohort=cohort_name,
                    estimator=est,
                    p_replicates=p_n3a,
                    is_alt_mask=is_alt,
                    cube_model_ids=model_ids,
                    cube_n_items=cube_n_items,
                )
            )

            hv_mask = np.zeros((n_cubes, num_factors), dtype=bool)
            hv_mask[:, high_var_col] = True
            all_rows.extend(
                summarize_panel_replicates_to_rows(
                    regime_id="N3b_contrast_rademacher_null",
                    cohort=cohort_name,
                    estimator=est,
                    p_replicates=p_n3b,
                    is_alt_mask=is_alt,
                    cube_model_ids=model_ids,
                    cube_n_items=cube_n_items,
                    extra_contrast_strata={high_var_label: hv_mask},
                )
            )

    return all_rows


def _build_planted_partial_null_mask(
    blocks: List[Dict[str, Any]],
    X_pm: np.ndarray,
    target_alt_fraction: float,
    rng_design: np.random.Generator,
    snr_theta: float = 4.2,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Construct centered + 20% planted-signal task blocks for Regime N4b.
    Selects `round(target_alt_fraction * m_model)` non-zero-variance contrasts per model family
    with frozen random signs and plants Delta_tau = sign * snr_theta * max(sd_c, 0.015) / sqrt(N_t).
    """
    num_factors = X_pm.shape[1]
    scale = X_pm.shape[0] / 2.0
    slices, model_ids, _ = _flatten_task_blocks_metadata(blocks)
    n_cubes = len(model_ids)

    sd_all = np.empty((n_cubes, num_factors), dtype=float)
    n_items_all = np.empty(n_cubes, dtype=float)
    centered_blocks: List[np.ndarray] = []

    for b_idx, blk in enumerate(blocks):
        s0, s1 = slices[b_idx]
        Y_blk = blk["Y"]
        C_blk = np.einsum("ck,mcn->mkn", X_pm, Y_blk) / scale
        c_bar = C_blk.mean(axis=2, keepdims=True)
        sd_blk = C_blk.std(axis=2, ddof=1)
        Y_tilde = Y_blk - 0.5 * np.einsum("ck,mkn->mcn", X_pm, c_bar)
        centered_blocks.append(Y_tilde)
        sd_all[s0:s1] = sd_blk
        n_items_all[s0:s1] = float(blk["n_items"])

    is_alt = np.zeros((n_cubes, num_factors), dtype=bool)
    alt_signs = np.zeros((n_cubes, num_factors), dtype=float)

    model_to_cubes: Dict[str, List[int]] = {}
    for c_i, m_id in enumerate(model_ids):
        model_to_cubes.setdefault(m_id, []).append(c_i)

    for m_id in sorted(model_to_cubes.keys()):
        c_list = model_to_cubes[m_id]
        eligible: List[Tuple[int, int]] = []
        for c_i in c_list:
            for f_j in range(num_factors):
                if sd_all[c_i, f_j] > 1e-12:
                    eligible.append((c_i, f_j))
        total_m = len(c_list) * num_factors
        n_plant = min(len(eligible), int(round(target_alt_fraction * total_m)))
        chosen_idx = rng_design.choice(len(eligible), size=n_plant, replace=False)
        for idx in chosen_idx:
            c_i, f_j = eligible[int(idx)]
            is_alt[c_i, f_j] = True
            alt_signs[c_i, f_j] = float(rng_design.choice([-1.0, 1.0]))

    planted_blocks: List[np.ndarray] = []
    for b_idx, Y_tilde in enumerate(centered_blocks):
        s0, s1 = slices[b_idx]
        sd_blk = sd_all[s0:s1]
        n_t = n_items_all[s0]
        delta_tau = (
            alt_signs[s0:s1]
            * is_alt[s0:s1].astype(float)
            * snr_theta
            * np.maximum(sd_blk, 0.015)
            / np.sqrt(n_t)
        )  # (M_b, k)
        shift = 0.5 * np.einsum("ck,mk->mc", X_pm, delta_tau)[:, :, None]
        planted_blocks.append(Y_tilde + shift)

    return planted_blocks, is_alt


def run_regime_n4_item_bootstrap(
    *,
    panels: Dict[str, Any],
    estimators: Sequence[str] = ("legacy_first", "cell_mean"),
    n_reps_mmlu: int = DEFAULT_REPLICATE_COUNTS["empirical_mmlu"],
    n_reps_slab: int = DEFAULT_REPLICATE_COUNTS["empirical_slab"],
) -> List[Dict[str, Any]]:
    """
    Regime N4 on real MMLU (855 cubes) and SLAB (178 cubes):
      - N4a (`N4a_bootstrap_centered_null`): Task-level item bootstrap with replacement from
        centered empirical cubes Y~ = Y - 0.5 * X_main * c_bar (also stratified on SLAB into
        standard N=100 [162 cubes] vs small N=26 `word_not_containing` [16 cubes]).
      - N4b (`N4b_bootstrap_partial_null_80_20`): Task-level item bootstrap from 80% centered
        null contrasts + 20% planted true signal contrasts, auditing actual BH and BY FDR & power.
    """
    rng_boot_a = np.random.default_rng(CALIBRATION_SEEDS["n4a_item_boot"])
    rng_a_200 = np.random.default_rng(CALIBRATION_SEEDS["n4a_sf200"])
    rng_a_2000 = np.random.default_rng(CALIBRATION_SEEDS["n4a_sf2000"])

    rng_design_b = np.random.default_rng(CALIBRATION_SEEDS["n4b_planted_design"])
    rng_boot_b = np.random.default_rng(CALIBRATION_SEEDS["n4b_item_boot"])
    rng_b_200 = np.random.default_rng(CALIBRATION_SEEDS["n4b_sf200"])
    rng_b_2000 = np.random.default_rng(CALIBRATION_SEEDS["n4b_sf2000"])

    all_rows: List[Dict[str, Any]] = []

    for cohort_name, X_pm, n_reps, high_var_col, high_var_label in [
        ("MMLU", panels["X5_pm"], n_reps_mmlu, 2, "factor_C_multiline"),
        ("SLAB", panels["X4_pm"], n_reps_slab, 0, "factor_A_schema"),
    ]:
        num_factors = X_pm.shape[1]
        scale = X_pm.shape[0] / 2.0

        for est in estimators:
            blocks = panels[cohort_name][est]
            slices, model_ids, cube_n_items = _flatten_task_blocks_metadata(blocks)
            n_cubes = len(model_ids)
            is_alt_zero = np.zeros((n_cubes, num_factors), dtype=bool)

            # Centered blocks for N4a
            centered_blocks: List[np.ndarray] = []
            for blk in blocks:
                Y_blk = blk["Y"]
                C_blk = np.einsum("ck,mcn->mkn", X_pm, Y_blk) / scale
                c_bar = C_blk.mean(axis=2, keepdims=True)
                centered_blocks.append(Y_blk - 0.5 * np.einsum("ck,mkn->mcn", X_pm, c_bar))

            # Planted 80/20 blocks for N4b
            planted_blocks, is_alt_planted = _build_planted_partial_null_mask(
                blocks, X_pm, target_alt_fraction=0.20, rng_design=rng_design_b, snr_theta=4.2
            )

            p_n4a = {proc: np.empty((n_reps, n_cubes, num_factors), dtype=float) for proc in PROCEDURES}
            p_n4b = {proc: np.empty((n_reps, n_cubes, num_factors), dtype=float) for proc in PROCEDURES}

            for r in range(n_reps):
                for b_idx, blk in enumerate(blocks):
                    s0, s1 = slices[b_idx]
                    N_t = blk["n_items"]

                    idx_a = rng_boot_a.integers(0, N_t, size=N_t)
                    Yb_a = centered_blocks[b_idx][:, :, idx_a]
                    inf_a = evaluate_cubes_batch_inference(
                        Yb_a, X_pm, rng_sf200=rng_a_200, rng_sf2000=rng_a_2000
                    )
                    for proc in PROCEDURES:
                        p_n4a[proc][r, s0:s1, :] = inf_a[proc]

                    idx_b = rng_boot_b.integers(0, N_t, size=N_t)
                    Yb_b = planted_blocks[b_idx][:, :, idx_b]
                    inf_b = evaluate_cubes_batch_inference(
                        Yb_b, X_pm, rng_sf200=rng_b_200, rng_sf2000=rng_b_2000
                    )
                    for proc in PROCEDURES:
                        p_n4b[proc][r, s0:s1, :] = inf_b[proc]

            hv_mask = np.zeros((n_cubes, num_factors), dtype=bool)
            hv_mask[:, high_var_col] = True
            all_rows.extend(
                summarize_panel_replicates_to_rows(
                    regime_id="N4a_bootstrap_centered_null",
                    cohort=cohort_name,
                    estimator=est,
                    p_replicates=p_n4a,
                    is_alt_mask=is_alt_zero,
                    cube_model_ids=model_ids,
                    cube_n_items=cube_n_items,
                    extra_contrast_strata={high_var_label: hv_mask},
                )
            )

            # Separate standard-N (N=100, 162 cubes) vs small-N (N=26 word_not_containing, 16 cubes) on SLAB
            if cohort_name == "SLAB":
                std_mask = cube_n_items >= CALIBRATION_TOLERANCE_GATES["standard_n_min_items"]
                sm_mask = ~std_mask
                for sub_cohort, c_mask in [
                    ("SLAB_STANDARD_N100", std_mask),
                    ("SLAB_SMALL_N26", sm_mask),
                ]:
                    p_sub = {proc: p_n4a[proc][:, c_mask, :] for proc in PROCEDURES}
                    sub_models = [m for m, keep in zip(model_ids, c_mask) if keep]
                    sub_n = cube_n_items[c_mask]
                    all_rows.extend(
                        summarize_panel_replicates_to_rows(
                            regime_id="N4a_bootstrap_centered_null",
                            cohort=sub_cohort,
                            estimator=est,
                            p_replicates=p_sub,
                            is_alt_mask=np.zeros((int(np.sum(c_mask)), num_factors), dtype=bool),
                            cube_model_ids=sub_models,
                            cube_n_items=sub_n,
                        )
                    )

            all_rows.extend(
                summarize_panel_replicates_to_rows(
                    regime_id="N4b_bootstrap_partial_null_80_20",
                    cohort=cohort_name,
                    estimator=est,
                    p_replicates=p_n4b,
                    is_alt_mask=is_alt_planted,
                    cube_model_ids=model_ids,
                    cube_n_items=cube_n_items,
                )
            )

    return all_rows


# ==============================================================================
# 5. Registered Calibration Gate Engine
# ==============================================================================

def evaluate_calibration_gates(
    df_cal: pd.DataFrame,
    gates: Dict[str, Any] = CALIBRATION_TOLERANCE_GATES,
) -> Dict[str, Any]:
    """
    Evaluate every inference procedure against registered Type-I, FDR, and planted-power
    tolerance gates across standard-N (N >= 100) regimes, and explicitly flag small-N (N < 100)
    and contrast-heterogeneity / pooled-RCBD miscalibration.
    """
    min_std_n = int(gates["standard_n_min_items"])
    ceilings_map = gates["raw_mc_upper95_ceilings_by_alpha"]

    failed_checks: List[Dict[str, Any]] = []
    small_n_flags: List[Dict[str, Any]] = []
    skewed_bootstrap_tail_flags: List[Dict[str, Any]] = []
    conservatism_flags: List[Dict[str, Any]] = []
    resolution_flags: List[Dict[str, Any]] = []

    proc_anti_conservative: Dict[str, bool] = {p: False for p in PROCEDURES}
    proc_low_power: Dict[str, bool] = {p: False for p in PROCEDURES}
    proc_resolution_limited: Dict[str, bool] = {p: False for p in PROCEDURES}

    for _, row in df_cal.iterrows():
        proc = str(row["procedure"])
        regime_id = str(row["regime_id"])
        reg_fam = str(row["regime_family"])
        cohort = str(row["cohort"])
        estimator = str(row["estimator"])
        stratum = str(row["stratum"])
        alpha = float(row["nominal_alpha"])
        alpha_key = str(alpha)
        is_standard_n = int(row["min_items_n"]) >= min_std_n
        is_mixed_slab = cohort == "SLAB"  # Mixed N=100 + N=26

        t1_point = float(row["raw_type1_rate"])
        t1_up95 = float(row["raw_type1_mc_upper95"])
        ceiling = float(ceilings_map[alpha_key])

        raw_violation = t1_up95 > ceiling or (
            np.isclose(alpha, 0.05) and t1_point > float(gates["raw_alpha_005_point_max"])
        )

        # 1. Small-N (< 100 items, e.g. 26-item word_not_containing) isolated qualification gate
        if not is_standard_n and not is_mixed_slab:
            if raw_violation:
                small_n_flags.append(
                    {
                        "regime_id": regime_id,
                        "cohort": cohort,
                        "estimator": estimator,
                        "stratum": stratum,
                        "procedure": proc,
                        "nominal_alpha": alpha,
                        "raw_type1_rate": t1_point,
                        "raw_type1_mc_upper95": t1_up95,
                        "standard_n_ceiling": ceiling,
                        "reason": "small_n_finite_sample_or_skewness_tail_inflation",
                    }
                )
            continue

        # For N4a on SLAB, standard-N eligibility is evaluated on SLAB_STANDARD_N100
        # so small-N=26 inflation is isolated in `small_n_flags` rather than conflated.
        if regime_id == "N4a_bootstrap_centered_null" and cohort == "SLAB" and stratum == "all_contrasts":
            continue

        # 2. Raw calibration curve check:
        #    - Across N1, N2, N3 (sign-invariant nulls): enforce full curve across all 5 alphas.
        #    - Under N4 (centered sparse skewed item-bootstrap where centering c - c_bar induces
        #      asymmetry on <=10-flip contrasts): enforce primary alpha=0.05 (<=0.060 point,
        #      <=0.065 MC upper95) as hard gate and log sub-0.05 Edgeworth skewness / sign-asymmetry
        #      exceedances in `skewed_bootstrap_tail_flags`.
        if raw_violation:
            if reg_fam in ("N1", "N2", "N3") or np.isclose(alpha, 0.05):
                proc_anti_conservative[proc] = True
                failed_checks.append(
                    {
                        "check_type": "raw_type1_anti_conservative",
                        "regime_id": regime_id,
                        "cohort": cohort,
                        "estimator": estimator,
                        "stratum": stratum,
                        "procedure": proc,
                        "nominal_alpha": alpha,
                        "observed_point": t1_point,
                        "observed_mc_upper95": t1_up95,
                        "ceiling": ceiling,
                    }
                )
            else:
                skewed_bootstrap_tail_flags.append(
                    {
                        "flag_type": "sub_005_skewed_centered_bootstrap_tail",
                        "regime_id": regime_id,
                        "cohort": cohort,
                        "estimator": estimator,
                        "stratum": stratum,
                        "procedure": proc,
                        "nominal_alpha": alpha,
                        "observed_point": t1_point,
                        "observed_mc_upper95": t1_up95,
                        "symmetric_null_ceiling": ceiling,
                    }
                )

        if np.isclose(alpha, 0.05) and stratum == "all_contrasts":
            if bool(row["is_severely_conservative_at_005"]):
                conservatism_flags.append(
                    {
                        "regime_id": regime_id,
                        "cohort": cohort,
                        "estimator": estimator,
                        "procedure": proc,
                        "raw_type1_at_005": t1_point,
                    }
                )
            if (
                bool(row["resolution_limited_per_cube_by"])
                or bool(row["resolution_limited_per_model_bh"])
                or bool(row["resolution_limited_global_bh"])
            ):
                proc_resolution_limited[proc] = True
                resolution_flags.append(
                    {
                        "regime_id": regime_id,
                        "cohort": cohort,
                        "estimator": estimator,
                        "procedure": proc,
                        "sign_flip_min_achievable_p": float(row["sign_flip_min_achievable_p"]),
                        "per_cube_by": bool(row["resolution_limited_per_cube_by"]),
                        "per_model_bh": bool(row["resolution_limited_per_model_bh"]),
                        "global_bh": bool(row["resolution_limited_global_bh"]),
                    }
                )

            has_alts = int(row["n_alt_tests_per_rep"]) > 0

            if reg_fam in ("N1", "N2", "N3") or has_alts:
                # Full BH & BY FDR gates across per_cube, per_model, and global_cohort!
                for fk in ["bh_per_cube", "by_per_cube", "bh_per_model", "by_per_model"]:
                    fdr_pt = float(row[f"{fk}_fdr"])
                    fdr_up = fdr_pt + 1.96 * float(row[f"{fk}_fdr_mcse"])
                    if (
                        fdr_pt > float(gates["bh_by_fdr_005_point_max_per_cube_or_model"])
                        or fdr_up > float(gates["bh_by_fdr_005_mc_upper95_max_per_cube_or_model"])
                    ):
                        proc_anti_conservative[proc] = True
                        failed_checks.append(
                            {
                                "check_type": f"{fk}_fdr_anti_conservative",
                                "regime_id": regime_id,
                                "cohort": cohort,
                                "estimator": estimator,
                                "stratum": stratum,
                                "procedure": proc,
                                "nominal_alpha": alpha,
                                "observed_point": fdr_pt,
                                "observed_mc_upper95": fdr_up,
                                "ceiling": float(gates["bh_by_fdr_005_mc_upper95_max_per_cube_or_model"]),
                            }
                        )

                for fk in ["bh_global", "by_global"]:
                    fdr_pt = float(row[f"{fk}_fdr"])
                    fdr_up = fdr_pt + 1.96 * float(row[f"{fk}_fdr_mcse"])
                    if has_alts:
                        pt_max = float(gates["bh_by_fdr_005_point_max_global_partial_null"])
                        up_max = float(gates["bh_by_fdr_005_mc_upper95_max_global_partial_null"])
                        violated = (fdr_pt > pt_max) or (fdr_up > up_max)
                        ceil_used = up_max
                    else:
                        pt_max = float(gates["bh_by_fdr_005_point_max_global_complete_null"])
                        violated = fdr_pt > pt_max
                        ceil_used = pt_max
                    if violated:
                        proc_anti_conservative[proc] = True
                        failed_checks.append(
                            {
                                "check_type": f"{fk}_fdr_anti_conservative",
                                "regime_id": regime_id,
                                "cohort": cohort,
                                "estimator": estimator,
                                "stratum": stratum,
                                "procedure": proc,
                                "nominal_alpha": alpha,
                                "observed_point": fdr_pt,
                                "observed_mc_upper95": fdr_up,
                                "ceiling": ceil_used,
                            }
                        )
            else:
                # N4a complete-null centered skewed item-bootstrap:
                # Hard per-cube BH/BY gate for parametric/RCBD tests; unstudentized sign-flip
                # asymmetry and large-m (per_model, global_cohort) ~1e-5 extreme-tail Edgeworth
                # skewness FWER are logged in `skewed_bootstrap_tail_flags` to qualify global
                # complete-null multiplicity claims on sparse binary contrasts.
                for fk in ["bh_per_cube", "by_per_cube"]:
                    fdr_pt = float(row[f"{fk}_fdr"])
                    fdr_up = fdr_pt + 1.96 * float(row[f"{fk}_fdr_mcse"])
                    pt_lim = float(gates["bh_by_fdr_005_point_max_n4a_per_cube"])
                    up_lim = float(gates["bh_by_fdr_005_mc_upper95_max_n4a_per_cube"])
                    if (fdr_pt > pt_lim) or (fdr_up > up_lim):
                        if SIGN_FLIP_B_MAP[proc] is None:
                            proc_anti_conservative[proc] = True
                            failed_checks.append(
                                {
                                    "check_type": f"{fk}_fdr_anti_conservative",
                                    "regime_id": regime_id,
                                    "cohort": cohort,
                                    "estimator": estimator,
                                    "stratum": stratum,
                                    "procedure": proc,
                                    "nominal_alpha": alpha,
                                    "observed_point": fdr_pt,
                                    "observed_mc_upper95": fdr_up,
                                    "ceiling": up_lim,
                                }
                            )
                        else:
                            skewed_bootstrap_tail_flags.append(
                                {
                                    "flag_type": f"{fk}_unstudentized_sign_asymmetry_on_centered_bootstrap",
                                    "regime_id": regime_id,
                                    "cohort": cohort,
                                    "estimator": estimator,
                                    "procedure": proc,
                                    "observed_point": fdr_pt,
                                    "observed_mc_upper95": fdr_up,
                                    "studentized_ceiling": up_lim,
                                }
                            )

                for fk in ["bh_per_model", "by_per_model", "bh_global", "by_global"]:
                    fdr_pt = float(row[f"{fk}_fdr"])
                    fdr_up = fdr_pt + 1.96 * float(row[f"{fk}_fdr_mcse"])
                    if fdr_pt > float(gates["bh_by_fdr_005_point_max_global_complete_null"]):
                        skewed_bootstrap_tail_flags.append(
                            {
                                "flag_type": f"{fk}_complete_null_extreme_tail_skewness_fwer",
                                "regime_id": regime_id,
                                "cohort": cohort,
                                "estimator": estimator,
                                "procedure": proc,
                                "observed_point": fdr_pt,
                                "observed_mc_upper95": fdr_up,
                                "reference_ceiling": float(
                                    gates["bh_by_fdr_005_point_max_global_complete_null"]
                                ),
                            }
                        )

            # Planted-alternative power gates on N2b and N4b (standard-N MMLU & synthetic)
            if has_alts and cohort in ("SYNTHETIC_SEMI_NULL", "MMLU"):
                pw_cube = float(row["bh_per_cube_power"])
                pw_glob = float(row["bh_global_power"])
                if (
                    pw_cube < float(gates["planted_power_005_min_bh_per_cube"])
                    or pw_glob < float(gates["planted_power_005_min_bh_global"])
                ):
                    proc_low_power[proc] = True
                    failed_checks.append(
                        {
                            "check_type": "planted_alternative_insufficient_power",
                            "regime_id": regime_id,
                            "cohort": cohort,
                            "estimator": estimator,
                            "stratum": stratum,
                            "procedure": proc,
                            "nominal_alpha": alpha,
                            "bh_per_cube_power": pw_cube,
                            "bh_global_power": pw_glob,
                            "required_per_cube_power": float(gates["planted_power_005_min_bh_per_cube"]),
                            "required_global_power": float(gates["planted_power_005_min_bh_global"]),
                        }
                    )

    procedure_verdicts: Dict[str, Dict[str, Any]] = {}
    for proc in PROCEDURES:
        if proc_anti_conservative[proc]:
            status = "FAIL_ANTI_CONSERVATIVE"
            paper_disposition = "WITHDRAWN_OR_EXPLORATORY_UNCALIBRATED"
        elif proc_low_power[proc] and proc_resolution_limited[proc]:
            status = "QUALIFIED_MONTE_CARLO_RESOLUTION_FLOOR"
            paper_disposition = "QUALIFIED_INCREASE_PERMUTATIONS_FOR_MULTIPLICITY"
        elif proc_low_power[proc]:
            status = "FAIL_UNDERPOWERED_CONSERVATIVE"
            paper_disposition = "RETAINED_CONSERVATIVE_BASELINE_ONLY"
        elif proc_resolution_limited[proc]:
            status = "PASS_CALIBRATED_WITH_BY_RESOLUTION_CAVEAT"
            paper_disposition = "RETAINED_CORROBORATING_RANDOMIZATION"
        else:
            status = "PASS_CALIBRATED_STANDARD_N"
            paper_disposition = "RETAINED_PRIMARY_CALIBRATED_STANDARD_N"

        has_small_n_flag = any(f["procedure"] == proc for f in small_n_flags)
        has_skew_tail_flag = any(f["procedure"] == proc for f in skewed_bootstrap_tail_flags)
        procedure_verdicts[proc] = {
            "verdict": status,
            "paper_claim_disposition": paper_disposition,
            "anti_conservative_violation": proc_anti_conservative[proc],
            "low_power_violation": proc_low_power[proc],
            "monte_carlo_resolution_limited": proc_resolution_limited[proc],
            "small_n_qualification_required": has_small_n_flag,
            "extreme_tail_skewness_qualification_required": has_skew_tail_flag,
        }

    return {
        "primary_item_clustered_t_standard_n_pass": (
            procedure_verdicts["item_clustered_t"]["verdict"] == "PASS_CALIBRATED_STANDARD_N"
        ),
        "fixed_block_rcbd_fails_anti_conservative": bool(
            proc_anti_conservative["fixed_block_rcbd"]
        ),
        "procedure_verdicts": procedure_verdicts,
        "n_failed_checks": len(failed_checks),
        "failed_checks": failed_checks,
        "n_small_n_flags": len(small_n_flags),
        "small_n_flags": small_n_flags,
        "n_skewed_bootstrap_tail_flags": len(skewed_bootstrap_tail_flags),
        "skewed_bootstrap_tail_flags": skewed_bootstrap_tail_flags,
        "n_conservatism_flags": len(conservatism_flags),
        "conservatism_flags": conservatism_flags,
        "n_resolution_flags": len(resolution_flags),
        "resolution_flags": resolution_flags,
    }


# ==============================================================================
# 6. Post-Calibration Observed Benchmark Detection Audit (1,033 Real Cubes)
# ==============================================================================

def compute_post_calibration_observed_audit(
    panels: Dict[str, Any],
    gate_results: Dict[str, Any],
    estimators: Sequence[str] = ("legacy_first", "cell_mean", "cell_mean_unique"),
) -> Dict[str, Any]:
    """
    Evaluate observed rejection counts across `per_cube`, `per_model`, and `global_cohort`
    BH and BY (at alpha = 0.05) on the real MMLU (855 cubes) and SLAB (178 cubes, also
    split into 162 standard N=100 cubes vs 16 small N=26 `word_not_containing` cubes)
    AFTER attaching calibration verdicts and empirical F_rcbd / t_cluster^2 inflation diagnostics.
    """
    rng_200 = np.random.default_rng(CALIBRATION_SEEDS["observed_sf200"])
    rng_2000 = np.random.default_rng(CALIBRATION_SEEDS["observed_sf2000"])
    verdicts = gate_results["procedure_verdicts"]

    audit_records: List[Dict[str, Any]] = []
    variance_ratio_diagnostics: Dict[str, Any] = {}

    for cohort_name, X_pm, factor_names in [
        ("MMLU", panels["X5_pm"], list("ABCDE")),
        ("SLAB", panels["X4_pm"], list("ABCD")),
    ]:
        num_factors = X_pm.shape[1]
        for est in estimators:
            blocks = panels[cohort_name][est]
            slices, model_ids, cube_n_items = _flatten_task_blocks_metadata(blocks)
            n_cubes = len(model_ids)

            p_obs = {proc: np.empty((n_cubes, num_factors), dtype=float) for proc in PROCEDURES}
            zero_var = np.empty((n_cubes, num_factors), dtype=bool)
            f_ratios = np.empty((n_cubes, num_factors), dtype=float)
            mse_reds = np.empty(n_cubes, dtype=float)

            for b_idx, blk in enumerate(blocks):
                s0, s1 = slices[b_idx]
                inf = evaluate_cubes_batch_inference(
                    blk["Y"], X_pm, rng_sf200=rng_200, rng_sf2000=rng_2000
                )
                for proc in PROCEDURES:
                    p_obs[proc][s0:s1] = inf[proc]
                zero_var[s0:s1] = inf["zero_var_mask"]
                f_ratios[s0:s1] = inf["f_rcbd_over_t_cluster_sq"]
                mse_reds[s0:s1] = inf["mse_reduction_pct"]

            variance_ratio_diagnostics[f"{cohort_name}_{est}"] = {
                "n_cubes": n_cubes,
                "n_zero_var_contrasts": int(np.sum(zero_var)),
                "mean_mse_reduction_pct": float(np.mean(mse_reds)),
                "mean_f_rcbd_over_t_cluster_sq_by_factor": {
                    f_name: float(np.mean(f_ratios[:, f_i]))
                    for f_i, f_name in enumerate(factor_names)
                },
                "p95_f_rcbd_over_t_cluster_sq_by_factor": {
                    f_name: float(np.percentile(f_ratios[:, f_i], 95.0))
                    for f_i, f_name in enumerate(factor_names)
                },
                "max_f_rcbd_over_t_cluster_sq_by_factor": {
                    f_name: float(np.max(f_ratios[:, f_i]))
                    for f_i, f_name in enumerate(factor_names)
                },
            }

            subsets = [(cohort_name, np.ones(n_cubes, dtype=bool))]
            if cohort_name == "SLAB":
                std_m = cube_n_items >= CALIBRATION_TOLERANCE_GATES["standard_n_min_items"]
                subsets.append(("SLAB_STANDARD_N100", std_m))
                subsets.append(("SLAB_SMALL_N26", ~std_m))

            for sub_label, c_mask in subsets:
                sub_models = [m for m, keep in zip(model_ids, c_mask) if keep]
                n_sub_cubes = int(np.sum(c_mask))
                n_sub_tests = n_sub_cubes * num_factors

                mod_map: Dict[str, List[int]] = {}
                for local_i, m_id in enumerate(sub_models):
                    mod_map.setdefault(m_id, []).append(local_i)

                for proc in PROCEDURES:
                    P_sub = p_obs[proc][c_mask]
                    q_cube_bh = adjust_pvalues_2d(P_sub, "fdr_bh")
                    q_cube_by = adjust_pvalues_2d(P_sub, "fdr_by")

                    sig_mod_bh, sig_mod_by = 0, 0
                    for _, c_idxs in sorted(mod_map.items()):
                        p_m = P_sub[c_idxs].reshape(1, -1)
                        sig_mod_bh += int(np.sum(adjust_pvalues_2d(p_m, "fdr_bh") < 0.05))
                        sig_mod_by += int(np.sum(adjust_pvalues_2d(p_m, "fdr_by") < 0.05))

                    p_glob = P_sub.reshape(1, -1)
                    sig_glob_bh = int(np.sum(adjust_pvalues_2d(p_glob, "fdr_bh") < 0.05))
                    sig_glob_by = int(np.sum(adjust_pvalues_2d(p_glob, "fdr_by") < 0.05))

                    is_small_n_sub = sub_label == "SLAB_SMALL_N26"
                    disp = verdicts[proc]["paper_claim_disposition"]
                    if is_small_n_sub and verdicts[proc]["small_n_qualification_required"]:
                        disp = f"{disp}__QUALIFIED_SMALL_N26_SKEWNESS"

                    audit_records.append(
                        {
                            "cohort_subset": sub_label,
                            "estimator": est,
                            "procedure": proc,
                            "calibration_verdict": verdicts[proc]["verdict"],
                            "claim_disposition": disp,
                            "n_cubes": n_sub_cubes,
                            "n_tests": n_sub_tests,
                            "raw_p_lt_005": int(np.sum(P_sub < 0.05)),
                            "detections_per_cube_bh": int(np.sum(q_cube_bh < 0.05)),
                            "detections_per_cube_by": int(np.sum(q_cube_by < 0.05)),
                            "detections_per_model_bh": sig_mod_bh,
                            "detections_per_model_by": sig_mod_by,
                            "detections_global_bh": sig_glob_bh,
                            "detections_global_by": sig_glob_by,
                        }
                    )

    return {
        "empirical_variance_pooling_diagnostics": variance_ratio_diagnostics,
        "observed_detections_with_calibration_status": audit_records,
    }


# ==============================================================================
# 7. Deterministic End-to-End Orchestrator
# ==============================================================================

def _extract_headline_metrics(df_cal: pd.DataFrame) -> Dict[str, Any]:
    """Extract compact headline numbers at alpha = 0.05 for rapid manuscript integration."""
    sub_005 = df_cal[np.isclose(df_cal["nominal_alpha"], 0.05)]
    out: Dict[str, Any] = {}
    for (reg, coh, est, strat), grp in sub_005.groupby(
        ["regime_id", "cohort", "estimator", "stratum"], sort=True
    ):
        key = f"{reg}__{coh}__{est}__{strat}"
        proc_map: Dict[str, Any] = {}
        for _, r in grp.iterrows():
            entry: Dict[str, Any] = {
                "raw_type1_005": float(r["raw_type1_rate"]),
                "raw_type1_mcse": float(r["raw_type1_mcse"]),
                "raw_type1_mc_upper95": float(r["raw_type1_mc_upper95"]),
                "passes_raw_gate": bool(r["passes_raw_anti_conservatism_gate"]),
            }
            if strat == "all_contrasts":
                entry["bh_per_cube_fdr"] = float(r["bh_per_cube_fdr"])
                entry["by_per_cube_fdr"] = float(r["by_per_cube_fdr"])
                entry["bh_per_model_fdr"] = float(r["bh_per_model_fdr"])
                entry["by_per_model_fdr"] = float(r["by_per_model_fdr"])
                entry["bh_global_fdr"] = float(r["bh_global_fdr"])
                entry["by_global_fdr"] = float(r["by_global_fdr"])
                entry["bh_per_cube_fwer"] = float(r["bh_per_cube_fwer"])
                entry["bh_global_fwer"] = float(r["bh_global_fwer"])
                if int(r["n_alt_tests_per_rep"]) > 0:
                    entry["raw_power_005"] = float(r["raw_power_rate"])
                    entry["bh_per_cube_power"] = float(r["bh_per_cube_power"])
                    entry["bh_per_model_power"] = float(r["bh_per_model_power"])
                    entry["bh_global_power"] = float(r["bh_global_power"])
                    entry["by_per_cube_power"] = float(r["by_per_cube_power"])
                    entry["by_global_power"] = float(r["by_global_power"])
            proc_map[str(r["procedure"])] = entry
        out[key] = proc_map
    return out


def run_inference_calibration(
    output_dir: Path = RESULTS_DIR,
    replicate_counts: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete Task R3 inference calibration study deterministically and write:
      - `<output_dir>/inference_calibration.csv`
      - `<output_dir>/inference_calibration_summary.json`
    """
    rep_cfg = dict(DEFAULT_REPLICATE_COUNTS)
    if replicate_counts is not None:
        rep_cfg.update(replicate_counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = verify_source_hashes()

    with open(DATA_DIR / "Ys.pickle", "rb") as f:
        Ys = pickle.load(f)

    panels = load_empirical_benchmark_panels(
        Ys, estimators=("legacy_first", "cell_mean", "cell_mean_unique")
    )

    all_rows: List[Dict[str, Any]] = []

    # 1. Regime N1 (standard N=200 & small N=26)
    rows_n1_std, diag_n1_std = run_regime_n1_additive_null(
        regime_id="N1_additive_null",
        cohort_label="SYNTHETIC_K32_N200",
        num_factors=5,
        n_items=200,
        n_models=10,
        n_tasks=12,
        n_replicates=rep_cfg["n1_standard"],
        seed_data=CALIBRATION_SEEDS["n1_standard_data"],
        seed_sf200=CALIBRATION_SEEDS["n1_standard_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n1_standard_sf2000"],
    )
    all_rows.extend(rows_n1_std)

    rows_n1_sm, diag_n1_sm = run_regime_n1_additive_null(
        regime_id="N1_small_N26_null",
        cohort_label="SYNTHETIC_K16_N26",
        num_factors=4,
        n_items=26,
        n_models=8,
        n_tasks=5,
        n_replicates=rep_cfg["n1_small26"],
        seed_data=CALIBRATION_SEEDS["n1_small26_data"],
        seed_sf200=CALIBRATION_SEEDS["n1_small26_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n1_small26_sf2000"],
    )
    all_rows.extend(rows_n1_sm)

    # 2. Regime N2 (complete null N2a & planted single-factor semi-null N2b)
    rows_n2a, diag_n2a = run_regime_n2_heterogeneous_2fi(
        regime_id="N2a_hetero_2fi_null",
        cohort_label="SYNTHETIC_HETERO_2FI",
        planted_factor0_shift=0.0,
        n_replicates=rep_cfg["n2a_hetero"],
        seed_data=CALIBRATION_SEEDS["n2a_hetero_data"],
        seed_sf200=CALIBRATION_SEEDS["n2a_hetero_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n2a_hetero_sf2000"],
    )
    all_rows.extend(rows_n2a)

    rows_n2b, diag_n2b = run_regime_n2_heterogeneous_2fi(
        regime_id="N2b_planted_semi_null",
        cohort_label="SYNTHETIC_SEMI_NULL",
        planted_factor0_shift=0.24,
        n_replicates=rep_cfg["n2b_semi"],
        seed_data=CALIBRATION_SEEDS["n2b_semi_data"],
        seed_sf200=CALIBRATION_SEEDS["n2b_semi_sf200"],
        seed_sf2000=CALIBRATION_SEEDS["n2b_semi_sf2000"],
    )
    all_rows.extend(rows_n2b)

    # 3. Regime N3 (empirically anchored within-item cell perm N3a & contrast Rademacher N3b)
    all_rows.extend(
        run_regime_n3_sharp_randomization(
            panels=panels,
            estimators=("legacy_first", "cell_mean"),
            n_reps_mmlu=rep_cfg["empirical_mmlu"],
            n_reps_slab=rep_cfg["empirical_slab"],
        )
    )

    # 4. Regime N4 (empirically anchored centered item-bootstrap N4a & 80/20 partial-null N4b)
    all_rows.extend(
        run_regime_n4_item_bootstrap(
            panels=panels,
            estimators=("legacy_first", "cell_mean"),
            n_reps_mmlu=rep_cfg["empirical_mmlu"],
            n_reps_slab=rep_cfg["empirical_slab"],
        )
    )

    df_cal = pd.DataFrame(all_rows)
    csv_path = output_dir / "inference_calibration.csv"
    df_cal.to_csv(csv_path, index=False, lineterminator="\n")
    csv_sha256 = sha256_file(csv_path)

    gate_results = evaluate_calibration_gates(df_cal, gates=CALIBRATION_TOLERANCE_GATES)
    observed_audit = compute_post_calibration_observed_audit(
        panels=panels,
        gate_results=gate_results,
        estimators=("legacy_first", "cell_mean", "cell_mean_unique"),
    )

    summary_payload: Dict[str, Any] = {
        "task": "Phase 0 Task R3: Calibrate item-clustered inference under registered nulls",
        "nominal_alphas": list(NOMINAL_ALPHAS),
        "procedures": list(PROCEDURES),
        "random_seeds": CALIBRATION_SEEDS,
        "replicate_counts": rep_cfg,
        "registered_tolerance_gates": CALIBRATION_TOLERANCE_GATES,
        "regime_specifications_and_exchangeability": REGIME_SPECIFICATIONS,
        "synthetic_regime_diagnostics": {
            "N1_additive_null": diag_n1_std,
            "N1_small_N26_null": diag_n1_sm,
            "N2a_hetero_2fi_null": diag_n2a,
            "N2b_planted_semi_null": diag_n2b,
        },
        "calibration_gate_evaluation": gate_results,
        "headline_alpha_005_by_regime_and_stratum": _extract_headline_metrics(df_cal),
        "post_calibration_observed_detection_audit": observed_audit,
        "source_data_sha256": source_hashes,
        "analysis_code_sha256": {
            "research/hf_multi_prompt/inference_calibration.py": sha256_file(Path(__file__)),
            "research/hf_multi_prompt/run_experiments.py": sha256_file(
                ROOT / "research" / "hf_multi_prompt" / "run_experiments.py"
            ),
        },
        "output_artifact_sha256": {
            "research/hf_multi_prompt/results/inference_calibration.csv": csv_sha256,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "statsmodels_version": statsmodels.__version__,
        },
    }

    summary_path = output_dir / "inference_calibration_summary.json"
    summary_text = json.dumps(summary_payload, indent=2, allow_nan=False) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")

    return df_cal, summary_payload


def main() -> None:
    df_cal, summary = run_inference_calibration(output_dir=RESULTS_DIR)
    gates = summary["calibration_gate_evaluation"]
    print(f"Wrote {len(df_cal)} calibration rows to {RESULTS_DIR / 'inference_calibration.csv'}")
    print(
        "Primary item_clustered_t standard-N gate pass:",
        gates["primary_item_clustered_t_standard_n_pass"],
    )
    print(
        "Fixed-block RCBD anti-conservative gate failure detected:",
        gates["fixed_block_rcbd_fails_anti_conservative"],
    )
    for proc, info in gates["procedure_verdicts"].items():
        print(f"  {proc:20s}: {info['verdict']} -> {info['paper_claim_disposition']}")


if __name__ == "__main__":
    main()
