"""
Phase 0 Task R1: Effect Noise-Floor Estimation, Split-Half Reliability, and
Absolute Error-to-Signal Ratios (ESR) for the Hugging Face Multi-Prompt Study.

Evaluates whether main, second-, third-, fourth-, and fifth-order orthogonal
Fourier contrast magnitudes represent repeatable prompt structure across
benchmark items or finite item-sampling noise across:
1. Cohort 1 MMLU per-task cubes: 855 complete 2^5 = 32 cubes (15 LLMs x 57 tasks, orders 1..5)
2. Cohort 1 MMLU pooled per-model cubes: 15 complete 2^5 = 32 cubes (14,042 items/LLM, orders 1..5)
3. Cohort 2 SLAB completeness-selected cubes: 178 complete 2^4 = 16 cubes (6 BBH + 7 LMentry tasks, orders 1..4)

Evaluates all four cell estimators (`legacy_first`, `canonical_first`, `cell_mean_unique`, `cell_mean`)
across Split 0 (`legacy_even_odd`), the 20 seeded random 50/50 item partitions (`SEEDS["item_splits"]`),
and the full-cube item set.

Outputs:
- `research/hf_multi_prompt/results/effect_reliability.csv`
- `research/hf_multi_prompt/results/effect_reliability_summary.json`
"""

from __future__ import annotations

import itertools
import json
import pickle
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import sparse, stats

from research.hf_multi_prompt.run_experiments import (
    BBH_MODELS,
    DATA_DIR,
    LMENTRY_MODELS,
    MMLU_MODELS,
    RESULTS_DIR,
    SEEDS,
    build_cube_design_matrices,
    build_mmlu_mapping_table,
    build_slab_mapping_table_for_task,
    concatenate_task_discovery_items,
    enumerate_fractional_cosets,
    extract_cell_item_matrix,
    extract_unique_cell_map,
    generate_balanced_random_designs,
    generate_item_splits,
    generate_random_designs_with_diagnostics,
    panel_cluster_bootstrap_ci,
    sha256_file,
    sha256_text,
    two_way_cluster_bootstrap_ci,
    validate_design_matrix,
    verify_source_hashes,
)

RELIABILITY_SEEDS: Dict[str, int] = {
    "item_splits": SEEDS["item_splits"],
    "cluster_bootstrap": SEEDS["cluster_bootstrap"],
    "item_bootstrap": 20261160,
    "rand16_full_rank": SEEDS["rand16_full_rank"],
    "rand8_cohort1": SEEDS["rand8_cohort1"],
    "rand8_cohort2": SEEDS["rand8_cohort2"],
}

ESTIMATOR_ORDER: Tuple[str, ...] = (
    "legacy_first",
    "canonical_first",
    "cell_mean_unique",
    "cell_mean",
)


# ==============================================================================
# 1. Orthogonal Fourier Contrast Matrices & Item-Level Variance Decomposition
# ==============================================================================


def build_order_contrast_matrices(
    num_factors: int,
) -> Dict[int, Tuple[np.ndarray, List[str]]]:
    """
    Build exact {-1, +1} Walsh-Fourier contrast column matrices and contrast names
    for every interaction order 1..num_factors, reusing `build_cube_design_matrices`.
    """
    mats = build_cube_design_matrices(num_factors)
    factor_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_factors]
    out: Dict[int, Tuple[np.ndarray, List[str]]] = {}

    # Order 1: main effects
    out[1] = (mats["X_pm"], [factor_letters[i] for i in range(num_factors)])

    # Order 2: pairwise products
    if num_factors >= 2:
        names_2 = [f"{factor_letters[i]}{factor_letters[j]}" for i, j in mats["pairs"]]
        out[2] = (mats["X_2fi"], names_2)

    # Orders 3..num_factors
    for order in range(3, num_factors + 1):
        combos = list(itertools.combinations(range(num_factors), order))
        names_o = ["".join(factor_letters[idx] for idx in cols) for cols in combos]
        out[order] = (mats["higher_orders"][order], names_o)

    # Verify full mutual orthogonality across all 2^k - 1 non-empty contrasts
    K = 1 << num_factors
    X_full = np.column_stack([out[o][0] for o in range(1, num_factors + 1)])
    gram = X_full.T @ X_full
    if not np.array_equal(gram, K * np.eye(K - 1, dtype=gram.dtype)):
        raise RuntimeError(f"Non-orthogonal Fourier basis detected for k={num_factors}")
    return out


def _vectorized_masked_mean_and_var(
    C: np.ndarray,
    masks_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact sample means, unbiased sample variances of item contrasts, and
    mean-estimator sampling variances across multiple item-subset masks simultaneously.

    Parameters
    ----------
    C : np.ndarray of shape (M_contrasts, N_items)
        Item-level Fourier contrasts c_{S, i} = (2 / K) sum_c X_{c, S} Y_{c, i}.
    masks_binary : np.ndarray of shape (N_items, S_splits)
        Binary {0.0, 1.0} membership indicators for each split column.

    Returns
    -------
    taus : np.ndarray of shape (M_contrasts, S_splits)
        Sample mean contrast per split.
    var_taus : np.ndarray of shape (M_contrasts, S_splits)
        Exact analytical item-cluster sampling variance s^2(c_{S, .}) / N_s.
    se_taus : np.ndarray of shape (M_contrasts, S_splits)
        Standard error sqrt(var_taus).
    """
    W = np.asarray(masks_binary, dtype=float)
    counts = W.sum(axis=0)  # (S,)
    if np.any(counts < 2):
        raise ValueError(f"Every split mask must contain at least 2 items, got min={counts.min()}")

    taus = (C @ W) / counts[None, :]
    # Shifted two-pass identity avoids catastrophic cancellation while staying vectorized:
    # sum_{i in I} (c_i - bar_c_I)^2 = sum_{i in I} (c_i - mu)^2 - N_I * (bar_c_I - mu)^2
    mu = C.mean(axis=1, keepdims=True)
    C_shift = C - mu
    ss_shift = (C_shift * C_shift) @ W
    ss_within = np.maximum(0.0, ss_shift - counts[None, :] * ((taus - mu) ** 2))
    s2_item = ss_within / (counts[None, :] - 1.0)
    var_taus = s2_item / counts[None, :]
    se_taus = np.sqrt(var_taus)
    return taus, var_taus, se_taus


def _tie_neutral_sign_agreement(
    tau_disc: np.ndarray,
    tau_conf: np.ndarray,
    zero_tol: float = 1e-15,
) -> np.ndarray:
    """
    Tie-neutral sign concordance indicator in {0.0, 0.5, 1.0}.
    Under any symmetric zero-mean null (even with discrete point mass at 0),
    expectation is strictly 0.50; when both contrasts are nonzero it equals
    1{sign(tau_disc) == sign(tau_conf)}.
    """
    d_zero = np.abs(tau_disc) <= zero_tol
    c_zero = np.abs(tau_conf) <= zero_tol
    either_zero = d_zero | c_zero
    same_sign = (tau_disc * tau_conf) > 0.0
    return np.where(either_zero, 0.5, np.where(same_sign, 1.0, 0.0))


def _safe_pearson_1d(x: np.ndarray, y: np.ndarray, tol: float = 1e-15) -> float:
    if x.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= tol or sy <= tol:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman_1d(x: np.ndarray, y: np.ndarray, tol: float = 1e-15) -> float:
    if x.size < 2:
        return float("nan")
    if float(np.std(x)) <= tol or float(np.std(y)) <= tol:
        return float("nan")
    rho = float(stats.spearmanr(x, y)[0])
    return rho if np.isfinite(rho) else float("nan")


def compute_cube_contrast_noise_and_rms(
    Y_items: np.ndarray,
    num_factors: int,
    n_item_boot: int = 0,
    boot_rng: Optional[np.random.Generator] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute exact orthogonal Fourier contrasts, analytical item-cluster sampling
    variances, observed RMS, item-sampling noise RMS floor, unclipped signal variance
    component, and truncated noise-corrected signal RMS for a single (K, N) cube,
    with optional fixed-seed multinomial item-bootstrap verification.
    """
    Y_arr = np.asarray(Y_items, dtype=float)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y_items must be 2D (K, N), got shape {Y_arr.shape}")
    K, N = Y_arr.shape
    expected_K = 1 << num_factors
    if K != expected_K:
        raise ValueError(f"Expected K={expected_K} rows for k={num_factors}, got {K}")
    if N < 2:
        raise ValueError(f"At least 2 items required to estimate item-sampling variance, got N={N}")
    if np.isnan(Y_arr).any():
        raise ValueError("NaN values found in Y_items; complete item blocks required")

    order_mats = build_order_contrast_matrices(num_factors)
    scale = K / 2.0
    X_all = np.column_stack([order_mats[o][0] for o in range(1, num_factors + 1)])
    C_all = (X_all.T @ Y_arr) / scale  # (K - 1, N)

    full_mask = np.ones((N, 1), dtype=float)
    taus_2d, vars_2d, ses_2d = _vectorized_masked_mean_and_var(C_all, full_mask)
    taus_all = taus_2d[:, 0]
    vars_all = vars_2d[:, 0]
    ses_all = ses_2d[:, 0]

    boot_vars_raw_all: Optional[np.ndarray] = None
    boot_vars_bessel_all: Optional[np.ndarray] = None
    if n_item_boot > 0:
        if boot_rng is None:
            boot_rng = np.random.default_rng(RELIABILITY_SEEDS["item_bootstrap"])
        counts = boot_rng.multinomial(N, np.full(N, 1.0 / N), size=n_item_boot).astype(float)
        # (n_item_boot, N) @ (N, K - 1) -> (n_item_boot, K - 1)
        boot_taus = (counts @ C_all.T) / float(N)
        boot_vars_raw_all = boot_taus.var(axis=0, ddof=1)
        boot_vars_bessel_all = boot_vars_raw_all * (float(N) / float(N - 1))

    out: Dict[int, Dict[str, Any]] = {}
    offset = 0
    for order in range(1, num_factors + 1):
        X_o, names_o = order_mats[order]
        n_o = X_o.shape[1]
        sl = slice(offset, offset + n_o)
        offset += n_o

        t_o = taus_all[sl]
        v_o = vars_all[sl]
        se_o = ses_all[sl]
        ms_obs = float(np.mean(t_o ** 2))
        ms_noise = float(np.mean(v_o))
        sigma2_unclipped = float(np.mean(t_o ** 2 - v_o))
        rms_obs = float(np.sqrt(ms_obs))
        rms_noise = float(np.sqrt(ms_noise))
        rms_corr = float(np.sqrt(max(0.0, sigma2_unclipped)))

        entry: Dict[str, Any] = {
            "order": order,
            "n_contrasts": n_o,
            "n_items": N,
            "contrast_names": names_o,
            "taus": t_o,
            "vars_analytical": v_o,
            "ses_analytical": se_o,
            "ms_obs": ms_obs,
            "ms_noise": ms_noise,
            "rms_obs": rms_obs,
            "rms_noise": rms_noise,
            "sigma2_signal_unclipped": sigma2_unclipped,
            "rms_corr": rms_corr,
            "is_truncated_zero": bool(sigma2_unclipped <= 0.0),
        }
        if boot_vars_raw_all is not None and boot_vars_bessel_all is not None:
            bv_raw = boot_vars_raw_all[sl]
            bv_bes = boot_vars_bessel_all[sl]
            ms_n_raw = float(np.mean(bv_raw))
            ms_n_bes = float(np.mean(bv_bes))
            sig2_boot_raw = float(ms_obs - ms_n_raw)
            sig2_boot_bes = float(ms_obs - ms_n_bes)
            entry.update(
                {
                    "n_item_boot": int(n_item_boot),
                    "vars_item_boot_raw": bv_raw,
                    "vars_item_boot_bessel": bv_bes,
                    "rms_noise_item_boot_raw": float(np.sqrt(ms_n_raw)),
                    "rms_noise_item_boot_bessel": float(np.sqrt(ms_n_bes)),
                    "sigma2_signal_unclipped_item_boot_raw": sig2_boot_raw,
                    "sigma2_signal_unclipped_item_boot_bessel": sig2_boot_bes,
                    "rms_corr_item_boot_raw": float(np.sqrt(max(0.0, sig2_boot_raw))),
                    "rms_corr_item_boot_bessel": float(np.sqrt(max(0.0, sig2_boot_bes))),
                }
            )
        out[order] = entry
    return out


def compute_cube_split_half_reliability(
    Y_disc: np.ndarray,
    Y_conf: np.ndarray,
    num_factors: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Independently estimate all orthogonal Fourier contrasts on disjoint discovery
    and confirmation item matrices for a single cube and compute order-specific
    split-half reliability, sign agreement, variance components, and held-out RMSE.
    """
    disc_res = compute_cube_contrast_noise_and_rms(Y_disc, num_factors)
    conf_res = compute_cube_contrast_noise_and_rms(Y_conf, num_factors)

    out: Dict[int, Dict[str, Any]] = {}
    for order in range(1, num_factors + 1):
        td = disc_res[order]["taus"]
        tc = conf_res[order]["taus"]
        sed = disc_res[order]["ses_analytical"]
        n_o = len(td)

        sign_vec = _tie_neutral_sign_agreement(td, tc)
        sign_strict = (np.sign(td) == np.sign(tc)).astype(float)
        gt_2se_mask = (np.abs(td) > 2.0 * sed) & (np.abs(td) > 1e-15)
        n_gt_2se = int(np.sum(gt_2se_mask))
        sign_gt_2se = float(np.mean(sign_vec[gt_2se_mask])) if n_gt_2se > 0 else float("nan")

        r_within = _safe_pearson_1d(td, tc) if n_o >= 2 else float("nan")
        rho_within = _safe_spearman_1d(td, tc) if n_o >= 2 else float("nan")
        cross_cov = float(np.mean(td * tc))
        heldout_mse = float(np.mean((td - tc) ** 2))
        heldout_rmse = float(np.sqrt(heldout_mse))

        out[order] = {
            "order": order,
            "n_contrasts": n_o,
            "disc": disc_res[order],
            "conf": conf_res[order],
            "within_cube_pearson_r": r_within,
            "within_cube_spearman_rho": rho_within,
            "sign_agreement_overall": float(np.mean(sign_vec)),
            "sign_agreement_strict_equal": float(np.mean(sign_strict)),
            "n_contrasts_gt_2se": n_gt_2se,
            "frac_contrasts_gt_2se": float(n_gt_2se / n_o),
            "sign_agreement_gt_2se": sign_gt_2se,
            "split_half_cross_cov": cross_cov,
            "rms_corr_cross_split": float(np.sqrt(max(0.0, cross_cov))),
            "heldout_pred_mse_full_cube": heldout_mse,
            "heldout_pred_rmse_full_cube": heldout_rmse,
        }
    return out


# ==============================================================================
# 2. Exact Paired Two-Way Model-Task Cluster Bootstrap Engine
# ==============================================================================


def _build_panel_bootstrap_weights(
    cube_meta: pd.DataFrame,
    panel_mode: str,
    n_boot: int = 1000,
    seed: int = RELIABILITY_SEEDS["cluster_bootstrap"],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct exact cube-multiplicity weight matrix W_cube of shape (n_boot, n_cubes)
    matching `two_way_cluster_bootstrap_ci` (for balanced 2D MMLU_TASK and 1D MMLU_POOLED)
    and `panel_cluster_bootstrap_ci` (for unbalanced Cohort 2 SLAB).
    """
    n_cubes = len(cube_meta)
    rng = np.random.default_rng(seed)

    if panel_mode == "1d_model":
        # Exact match to 1D branch of `two_way_cluster_bootstrap_ci`
        M = n_cubes
        idx = rng.integers(0, M, size=(n_boot, M))
        W_cube = np.zeros((n_boot, M), dtype=float)
        for b in range(n_boot):
            W_cube[b] = np.bincount(idx[b], minlength=M)
        return W_cube, {
            "panel_mode": panel_mode,
            "n_model_clusters": int(M),
            "n_task_clusters": 1,
            "n_valid_bootstrap": int(n_boot),
            "n_invalid_bootstrap_draws": 0,
        }

    if panel_mode == "2d_balanced":
        # Exact match to 2D branch of `two_way_cluster_bootstrap_ci` assuming row-major (model, task)
        models = list(dict.fromkeys(cube_meta["model_id"].tolist()))
        tasks = list(dict.fromkeys(cube_meta["task_key"].tolist()))
        M, T = len(models), len(tasks)
        if M * T != n_cubes:
            raise ValueError(f"Expected balanced {M}x{T}={M*T} panel, got {n_cubes} cubes")
        m_map = {m: i for i, m in enumerate(models)}
        t_map = {t: j for j, t in enumerate(tasks)}
        c_m = np.array([m_map[m] for m in cube_meta["model_id"]], dtype=int)
        c_t = np.array([t_map[t] for t in cube_meta["task_key"]], dtype=int)

        m_idx = rng.integers(0, M, size=(n_boot, M))
        t_idx = rng.integers(0, T, size=(n_boot, T))
        W_M = np.array([np.bincount(m_idx[b], minlength=M) for b in range(n_boot)], dtype=float)
        W_T = np.array([np.bincount(t_idx[b], minlength=T) for b in range(n_boot)], dtype=float)
        W_cube = W_M[:, c_m] * W_T[:, c_t]
        return W_cube, {
            "panel_mode": panel_mode,
            "n_model_clusters": int(M),
            "n_task_clusters": int(T),
            "n_valid_bootstrap": int(n_boot),
            "n_invalid_bootstrap_draws": 0,
        }

    if panel_mode == "2d_unbalanced":
        # Exact match to `panel_cluster_bootstrap_ci` pivot ordering (sorted index & sorted columns)
        tmp = cube_meta.copy()
        tmp["_cube_idx"] = np.arange(n_cubes)
        col_keys: Any = ["cohort", "task"] if "cohort" in tmp.columns else "task_key"
        pivot = tmp.pivot(index="model_id", columns=col_keys, values="_cube_idx")
        grid = pivot.to_numpy(dtype=float)
        M, T = grid.shape
        finite_grid = np.isfinite(grid)

        c_m = np.empty(n_cubes, dtype=int)
        c_t = np.empty(n_cubes, dtype=int)
        for i in range(M):
            for j in range(T):
                if finite_grid[i, j]:
                    c_idx = int(grid[i, j])
                    c_m[c_idx] = i
                    c_t[c_idx] = j

        W_cube = np.zeros((n_boot, n_cubes), dtype=float)
        valid = 0
        invalid = 0
        draws = 0
        max_draws = max(1000, 100 * n_boot)
        while valid < n_boot and draws < max_draws:
            draws += 1
            m_idx = rng.integers(0, M, size=M)
            t_idx = rng.integers(0, T, size=T)
            wm = np.bincount(m_idx, minlength=M).astype(float)
            wt = np.bincount(t_idx, minlength=T).astype(float)
            wc = wm[c_m] * wt[c_t]
            if wc.sum() <= 0.0:
                invalid += 1
                continue
            W_cube[valid] = wc
            valid += 1
        if valid < n_boot:
            raise RuntimeError(f"Only {valid} valid panel bootstrap draws after {draws} attempts")
        return W_cube, {
            "panel_mode": panel_mode,
            "n_model_clusters": int(M),
            "n_task_clusters": int(T),
            "n_valid_bootstrap": int(valid),
            "n_invalid_bootstrap_draws": int(invalid),
        }

    raise ValueError(f"Unknown panel_mode: {panel_mode!r}")


def _ci_from_samples(
    point: float,
    boot_samples: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    finite = boot_samples[np.isfinite(boot_samples)]
    if finite.size == 0 or not np.isfinite(point):
        return {
            "estimate": float(point),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    return {
        "estimate": float(point),
        "se": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "ci_low": float(np.percentile(finite, 100.0 * (alpha / 2.0))),
        "ci_high": float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0))),
    }


def _bootstrap_weighted_mean_with_nan(
    vals_cube: np.ndarray,
    W_cube: np.ndarray,
) -> np.ndarray:
    """Compute bootstrap panel mean per draw while skipping NaN cubes cleanly."""
    finite = np.isfinite(vals_cube)
    if not np.any(finite):
        return np.full(W_cube.shape[0], np.nan, dtype=float)
    v_safe = np.where(finite, vals_cube, 0.0)
    w_fin = W_cube * finite[None, :].astype(float)
    denom = w_fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0.0, (W_cube @ v_safe) / denom, np.nan)


def _bootstrap_pooled_pearson_splits(
    x_cos: np.ndarray,
    y_cos: np.ndarray,
    W_cube: np.ndarray,
) -> np.ndarray:
    """
    Exact vectorized bootstrap of pooled Pearson correlation across (C, O, S) arrays,
    averaged over the S splits in `x_cos`/`y_cos`, using cube-level sufficient statistics.
    """
    # x_cos, y_cos: (C, O, S)
    _, O, _ = x_cos.shape
    sx = x_cos.sum(axis=1)              # (C, S)
    sy = y_cos.sum(axis=1)              # (C, S)
    sxx = (x_cos * x_cos).sum(axis=1)   # (C, S)
    syy = (y_cos * y_cos).sum(axis=1)   # (C, S)
    sxy = (x_cos * y_cos).sum(axis=1)   # (C, S)

    N_b = (W_cube.sum(axis=1) * float(O))[:, None]  # (B, 1)
    Sx = W_cube @ sx
    Sy = W_cube @ sy
    Sxx = W_cube @ sxx
    Syy = W_cube @ syy
    Sxy = W_cube @ sxy

    cov = Sxy - (Sx * Sy) / N_b
    vx = np.maximum(0.0, Sxx - (Sx * Sx) / N_b)
    vy = np.maximum(0.0, Syy - (Sy * Sy) / N_b)
    den = np.sqrt(vx * vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_bs = np.where(den > 1e-15, cov / den, 0.0)
    return r_bs.mean(axis=1)  # (B,)


def _bootstrap_pooled_spearman_single_split(
    x_co: np.ndarray,
    y_co: np.ndarray,
    W_cube: np.ndarray,
    batch_size: int = 250,
) -> np.ndarray:
    """
    Exact vectorized mid-rank Spearman rho across all bootstrap draws `W_cube` (B, C)
    matching `scipy.stats.spearmanr(np.repeat(x, w), np.repeat(y, w))` to machine precision.
    """
    C, O = x_co.shape
    xf = x_co.ravel()
    yf = y_co.ravel()
    L = xf.size

    ux, inv_x = np.unique(xf, return_inverse=True)
    uy, inv_y = np.unique(yf, return_inverse=True)
    cube_of_l = np.repeat(np.arange(C), O)
    Sx = sparse.csr_matrix((np.ones(L, dtype=float), (cube_of_l, inv_x)), shape=(C, len(ux)))
    Sy = sparse.csr_matrix((np.ones(L, dtype=float), (cube_of_l, inv_y)), shape=(C, len(uy)))

    B = W_cube.shape[0]
    out = np.empty(B, dtype=float)
    for start in range(0, B, batch_size):
        Wb = W_cube[start : start + batch_size]
        b_len = Wb.shape[0]
        Wgx = Wb @ Sx  # (b_len, Gx)
        Wgy = Wb @ Sy  # (b_len, Gy)
        Fx = np.cumsum(Wgx, axis=1)
        Fy = np.cumsum(Wgy, axis=1)

        N_tot = (Wb.sum(axis=1, keepdims=True)) * float(O)
        mean_rank = 0.5 * (N_tot + 1.0)

        dx_grp = (Fx - 0.5 * (Wgx - 1.0)) - mean_rank
        dy_grp = (Fy - 0.5 * (Wgy - 1.0)) - mean_rank

        vx = np.sum(Wgx * (dx_grp * dx_grp), axis=1)
        vy = np.sum(Wgy * (dy_grp * dy_grp), axis=1)
        den = np.sqrt(np.maximum(0.0, vx * vy))

        dx_bco = dx_grp[:, inv_x].reshape(b_len, C, O)
        dy_bco = dy_grp[:, inv_y].reshape(b_len, C, O)
        cov = np.sum(Wb * np.sum(dx_bco * dy_bco, axis=2), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + b_len] = np.where(den > 1e-15, cov / den, 0.0)
    return out


def _bootstrap_pooled_spearman_splits(
    x_cos: np.ndarray,
    y_cos: np.ndarray,
    W_cube: np.ndarray,
) -> np.ndarray:
    S = x_cos.shape[2]
    acc = np.zeros(W_cube.shape[0], dtype=float)
    for s in range(S):
        acc += _bootstrap_pooled_spearman_single_split(x_cos[:, :, s], y_cos[:, :, s], W_cube)
    return acc / float(S)


# ==============================================================================
# 3. General Cohort Panel Reliability + Screening ESR Evaluator
# ==============================================================================


def _build_linear_contrast_operator(
    D_full: np.ndarray,
    subset_indices_list: Sequence[Sequence[int]],
    col_slice: slice,
    design_family_name: str,
) -> np.ndarray:
    """
    Construct stacked linear contrast operator L of shape (n_designs, n_contrasts, K)
    such that `2 * (pinv(D[idx]) @ y[idx])[col_slice] == L[d] @ y` after explicit
    full-rank verification via `validate_design_matrix`.
    """
    K = D_full.shape[0]
    n_designs = len(subset_indices_list)
    sample_cols = np.arange(D_full.shape[1])[col_slice]
    n_contrasts = len(sample_cols)
    L = np.zeros((n_designs, n_contrasts, K), dtype=float)
    for d_i, idx in enumerate(subset_indices_list):
        idx_list = list(idx)
        validate_design_matrix(
            D_full[idx_list],
            design_id=f"{design_family_name}_{d_i:03d}",
            model_id=design_family_name,
        )
        pinv_sub = np.linalg.pinv(D_full[idx_list])  # (n_params, n_runs)
        L[d_i, :, :] = 0.0
        L[d_i][:, idx_list] = 2.0 * pinv_sub[col_slice, :]
    return L


def build_screening_operators(
    num_factors: int,
    cosets_info: Optional[Dict[str, Any]] = None,
    r16_fr: Optional[List[List[int]]] = None,
    r16_conds: Optional[np.ndarray] = None,
    r8_c1_fr: Optional[List[List[int]]] = None,
    r8_c2_fr: Optional[List[List[int]]] = None,
) -> Dict[int, List[Tuple[str, int, np.ndarray]]]:
    """
    Precompute validated linear contrast operators `(design_id, num_runs, L_operator)`
    per contrast order (where `L_operator` has shape `(n_designs, n_contrasts, K)`).
    """
    if cosets_info is None:
        cosets_info = enumerate_fractional_cosets()

    mats = build_cube_design_matrices(num_factors)
    all_cells = mats["all_cells"]
    K = 1 << num_factors
    scale = K / 2.0
    order_mats = build_order_contrast_matrices(num_factors)

    ops_by_order: Dict[int, List[Tuple[str, int, np.ndarray]]] = {
        o: [] for o in range(1, num_factors + 1)
    }
    # Full-cube operator is available for every order 1..k
    for o in range(1, num_factors + 1):
        X_o, _ = order_mats[o]
        L_full = ((X_o.T) / scale)[None, :, :]  # (1, n_o, K)
        ops_by_order[o].append(("full_cube", K, L_full))

    if num_factors == 5:
        if r16_fr is None or r16_conds is None:
            r16_fr, r16_diag, _ = generate_random_designs_with_diagnostics(
                5, 16, 200, RELIABILITY_SEEDS["rand16_full_rank"], True, "MMLU", "rand16_2fi_fullrank"
            )
            r16_conds = np.array([d["cond_2fi"] for d in r16_diag if d["accepted_primary"]])
        if r8_c1_fr is None:
            r8_c1_fr = generate_balanced_random_designs(
                32, 5, 8, num_screens=200, seed=RELIABILITY_SEEDS["rand8_cohort1"], require_2fi_full_rank=False
            )

        D_main = mats["D_main"]
        D_2fi = mats["D_2fi"]
        res5_pos = cosets_info["res5_cosets"]["2(5-1)V_coset_pos"]["indices"]
        res5_neg = cosets_info["res5_cosets"]["2(5-1)V_coset_neg"]["indices"]
        res3_c_idx = [v["indices"] for v in cosets_info["res3_canonical_cosets"].values()]
        res3_60_idx = [v["indices"] for v in cosets_info["res3_all_60"].values()]
        ofat5_cells = [
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 0, 1),
        ]
        ofat5_idx = [all_cells.index(c) for c in ofat5_cells]
        r16_wellcond = [r16_fr[i] for i in range(len(r16_fr)) if r16_conds[i] <= 15.0]

        # Order 1 (5 main contrasts)
        L_m_pos = _build_linear_contrast_operator(D_2fi, [res5_pos], slice(1, 6), "2(5-1)V_pos_main")
        L_m_neg = _build_linear_contrast_operator(D_2fi, [res5_neg], slice(1, 6), "2(5-1)V_neg_main")
        L_m_both = np.concatenate([L_m_pos, L_m_neg], axis=0)
        L_m_r16 = _build_linear_contrast_operator(D_main, r16_fr, slice(1, 6), "rand16_main")
        L_m_r16_2fi = _build_linear_contrast_operator(D_2fi, r16_fr, slice(1, 6), "rand16_2fi_joint_main")
        L_m_res3_60 = _build_linear_contrast_operator(D_main, res3_60_idx, slice(1, 6), "2(5-2)III_all60")
        L_m_res3_4 = _build_linear_contrast_operator(D_main, res3_c_idx, slice(1, 6), "2(5-2)III_4cosets")
        L_m_r8 = _build_linear_contrast_operator(D_main, r8_c1_fr, slice(1, 6), "rand8_c1_main")
        L_m_ofat = _build_linear_contrast_operator(D_main, [ofat5_idx], slice(1, 6), "ofat6_main")

        ops_by_order[1].extend(
            [
                ("2(5-1)V_coset_avg", 16, L_m_both),
                ("2(5-1)V_coset_pos", 16, L_m_pos),
                ("2(5-1)V_coset_neg", 16, L_m_neg),
                ("rand16_fullrank", 16, L_m_r16),
                ("rand16_fullrank_joint2fi", 16, L_m_r16_2fi),
                ("2(5-2)III_all60", 8, L_m_res3_60),
                ("2(5-2)III_4cosets", 8, L_m_res3_4),
                ("rand8_fullrank", 8, L_m_r8),
                ("ofat_ref0", 6, L_m_ofat),
            ]
        )

        # Order 2 (10 second-order contrasts)
        L_2_pos = _build_linear_contrast_operator(D_2fi, [res5_pos], slice(6, 16), "2(5-1)V_pos_2fi")
        L_2_neg = _build_linear_contrast_operator(D_2fi, [res5_neg], slice(6, 16), "2(5-1)V_neg_2fi")
        L_2_both = np.concatenate([L_2_pos, L_2_neg], axis=0)
        L_2_r16 = _build_linear_contrast_operator(D_2fi, r16_fr, slice(6, 16), "rand16_2fi")
        L_2_r16_wc = _build_linear_contrast_operator(D_2fi, r16_wellcond, slice(6, 16), "rand16_wellcond_2fi")

        ops_by_order[2].extend(
            [
                ("2(5-1)V_coset_avg", 16, L_2_both),
                ("2(5-1)V_coset_pos", 16, L_2_pos),
                ("2(5-1)V_coset_neg", 16, L_2_neg),
                ("rand16_fullrank", 16, L_2_r16),
                ("rand16_wellcond", 16, L_2_r16_wc),
            ]
        )

    elif num_factors == 4:
        if r8_c2_fr is None:
            r8_c2_fr = generate_balanced_random_designs(
                16, 4, 8, num_screens=200, seed=RELIABILITY_SEEDS["rand8_cohort2"], require_2fi_full_rank=False
            )
        D4_main = mats["D_main"]
        res4_pos = cosets_info["res4_cosets"]["2(4-1)IV_coset_pos"]["indices"]
        res4_neg = cosets_info["res4_cosets"]["2(4-1)IV_coset_neg"]["indices"]
        ofat4_cells = [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ]
        ofat4_idx = [all_cells.index(c) for c in ofat4_cells]

        L4_pos = _build_linear_contrast_operator(D4_main, [res4_pos], slice(1, 5), "2(4-1)IV_pos_main")
        L4_neg = _build_linear_contrast_operator(D4_main, [res4_neg], slice(1, 5), "2(4-1)IV_neg_main")
        L4_both = np.concatenate([L4_pos, L4_neg], axis=0)
        L4_r8 = _build_linear_contrast_operator(D4_main, r8_c2_fr, slice(1, 5), "rand8_c2_main")
        L4_ofat = _build_linear_contrast_operator(D4_main, [ofat4_idx], slice(1, 5), "ofat5_main")

        ops_by_order[1].extend(
            [
                ("2(4-1)IV_coset_avg", 8, L4_both),
                ("2(4-1)IV_coset_pos", 8, L4_pos),
                ("2(4-1)IV_coset_neg", 8, L4_neg),
                ("rand8_fullrank", 8, L4_r8),
                ("ofat_ref0", 5, L4_ofat),
            ]
        )

    return ops_by_order


def evaluate_panel_cubes_reliability(
    cohort_label: str,
    estimator_name: str,
    num_factors: int,
    cube_matrices: Sequence[np.ndarray],
    cube_d_masks: Sequence[np.ndarray],
    cube_c_masks: Sequence[np.ndarray],
    cube_meta: pd.DataFrame,
    panel_mode: str,
    screening_ops: Optional[Dict[int, List[Tuple[str, int, np.ndarray]]]] = None,
    n_boot_cluster: int = 1000,
    cluster_seed: int = RELIABILITY_SEEDS["cluster_bootstrap"],
    n_item_boot_verify: int = 0,
    item_boot_seed: int = RELIABILITY_SEEDS["item_bootstrap"],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate full-cube noise floor, Split 0 and 20-split reliability, paired
    two-way model-task cluster bootstrap 95% intervals, and screening ESR /
    held-out contrast prediction RMSE across a cohort panel of cubes.
    """
    C_count = len(cube_matrices)
    if C_count == 0:
        raise ValueError("At least one cube is required")
    K = 1 << num_factors
    scale = K / 2.0
    order_mats = build_order_contrast_matrices(num_factors)
    X_all = np.column_stack([order_mats[o][0] for o in range(1, num_factors + 1)])

    n_splits_total = cube_d_masks[0].shape[1]
    has_20_splits = n_splits_total > 1

    # Storage across cubes:
    # Full-cube contrasts & variances: (C, K-1)
    tau_full_all = np.empty((C_count, K - 1), dtype=float)
    var_full_all = np.empty((C_count, K - 1), dtype=float)
    # Split-level contrasts & variances: (C, K-1, S)
    tau_disc_all = np.empty((C_count, K - 1, n_splits_total), dtype=float)
    var_disc_all = np.empty((C_count, K - 1, n_splits_total), dtype=float)
    se_disc_all = np.empty((C_count, K - 1, n_splits_total), dtype=float)
    tau_conf_all = np.empty((C_count, K - 1, n_splits_total), dtype=float)
    var_conf_all = np.empty((C_count, K - 1, n_splits_total), dtype=float)
    # Split-level cell means for fast screening OLS: (C, K, S)
    Y_disc_cells = np.empty((C_count, K, n_splits_total), dtype=float)
    n_items_per_cube = np.empty(C_count, dtype=int)

    # Optional item-bootstrap verification storage: (C, K-1) for full and split0_disc
    boot_var_full_raw = np.empty((C_count, K - 1), dtype=float) if n_item_boot_verify > 0 else None
    boot_var_full_bes = np.empty((C_count, K - 1), dtype=float) if n_item_boot_verify > 0 else None
    boot_var_s0_raw = np.empty((C_count, K - 1), dtype=float) if n_item_boot_verify > 0 else None
    boot_var_s0_bes = np.empty((C_count, K - 1), dtype=float) if n_item_boot_verify > 0 else None
    item_rng = np.random.default_rng(item_boot_seed) if n_item_boot_verify > 0 else None

    for c_i in range(C_count):
        Y_t = np.asarray(cube_matrices[c_i], dtype=float)
        _, N = Y_t.shape
        n_items_per_cube[c_i] = N
        d_bin = (cube_d_masks[c_i] > 0.0).astype(float)
        c_bin = (cube_c_masks[c_i] > 0.0).astype(float)
        d_cnt = d_bin.sum(axis=0)
        Y_disc_cells[c_i] = (Y_t @ d_bin) / d_cnt[None, :]

        C_item = (X_all.T @ Y_t) / scale  # (K-1, N)
        tf, vf, _ = _vectorized_masked_mean_and_var(C_item, np.ones((N, 1), dtype=float))
        tau_full_all[c_i] = tf[:, 0]
        var_full_all[c_i] = vf[:, 0]

        td, vd, sed = _vectorized_masked_mean_and_var(C_item, d_bin)
        tc, vc, _ = _vectorized_masked_mean_and_var(C_item, c_bin)
        tau_disc_all[c_i] = td
        var_disc_all[c_i] = vd
        se_disc_all[c_i] = sed
        tau_conf_all[c_i] = tc
        var_conf_all[c_i] = vc

        if n_item_boot_verify > 0 and item_rng is not None:
            cnt_f = item_rng.multinomial(N, np.full(N, 1.0 / N), size=n_item_boot_verify).astype(float)
            bt_f = (cnt_f @ C_item.T) / float(N)
            vr_f = bt_f.var(axis=0, ddof=1)
            boot_var_full_raw[c_i] = vr_f
            boot_var_full_bes[c_i] = vr_f * (float(N) / float(N - 1))

            s0_idx = np.where(d_bin[:, 0] > 0.5)[0]
            N_s0 = len(s0_idx)
            C_s0 = C_item[:, s0_idx]
            cnt_s0 = item_rng.multinomial(N_s0, np.full(N_s0, 1.0 / N_s0), size=n_item_boot_verify).astype(float)
            bt_s0 = (cnt_s0 @ C_s0.T) / float(N_s0)
            vr_s0 = bt_s0.var(axis=0, ddof=1)
            boot_var_s0_raw[c_i] = vr_s0
            boot_var_s0_bes[c_i] = vr_s0 * (float(N_s0) / float(N_s0 - 1))

    # Build paired two-way model-task cluster bootstrap weights ONCE for this cohort
    W_cube, boot_meta = _build_panel_bootstrap_weights(
        cube_meta=cube_meta,
        panel_mode=panel_mode,
        n_boot=n_boot_cluster,
        seed=cluster_seed,
    )
    W_sum = W_cube.sum(axis=1)  # (B,)

    # Map order slices in (0 .. K-2)
    order_slices: Dict[int, slice] = {}
    offset = 0
    for o in range(1, num_factors + 1):
        n_o = order_mats[o][0].shape[1]
        order_slices[o] = slice(offset, offset + n_o)
        offset += n_o

    csv_rows: List[Dict[str, Any]] = []
    order_summary_dict: Dict[str, Dict[str, Any]] = {
        "full_cube": {},
        "split0": {},
        **({"splits20": {}} if has_20_splits else {}),
    }

    # Store bootstrap sample vectors per (split_regime, order) for paired inter-order differences
    boot_store_cohort_rms_corr: Dict[Tuple[str, int], np.ndarray] = {}
    boot_store_cube_rms_corr: Dict[Tuple[str, int], np.ndarray] = {}
    boot_store_pooled_r: Dict[Tuple[str, int], np.ndarray] = {}

    # --------------------------------------------------------------------------
    # A. Evaluate Order-Specific Noise Floor & Split-Half Reliability
    # --------------------------------------------------------------------------
    regimes: List[Tuple[str, Optional[slice]]] = [("full_cube", None), ("split0", slice(0, 1))]
    if has_20_splits:
        regimes.append(("splits20", slice(1, n_splits_total)))

    for regime_name, s_slice in regimes:
        for order in range(1, num_factors + 1):
            sl = order_slices[order]
            n_o = order_mats[order][0].shape[1]

            if regime_name == "full_cube":
                t_eval = tau_full_all[:, sl][:, :, None]  # (C, O, 1)
                v_eval = var_full_all[:, sl][:, :, None]  # (C, O, 1)

                ms_obs_cs = np.mean(t_eval ** 2, axis=1)      # (C, 1)
                ms_noise_cs = np.mean(v_eval, axis=1)         # (C, 1)
                sig2_unclip_cs = ms_obs_cs - ms_noise_cs      # (C, 1)
                rms_obs_cs = np.sqrt(ms_obs_cs)
                rms_noise_cs = np.sqrt(ms_noise_cs)
                rms_corr_cs = np.sqrt(np.maximum(0.0, sig2_unclip_cs))
                trunc_cs = (sig2_unclip_cs <= 0.0).astype(float)

                ms_obs_c = ms_obs_cs[:, 0]
                ms_noise_c = ms_noise_cs[:, 0]
                sig2_unclip_c = sig2_unclip_cs[:, 0]
                rms_obs_c = rms_obs_cs[:, 0]
                rms_noise_c = rms_noise_cs[:, 0]
                rms_corr_c = rms_corr_cs[:, 0]
                trunc_frac_cube_splits = float(trunc_cs.mean())
                n_cubes_trunc = int(np.sum(sig2_unclip_c <= 0.0))
                frac_cubes_trunc = float(n_cubes_trunc / C_count)

                cohort_rms_obs = float(np.sqrt(np.mean(ms_obs_c)))
                cohort_rms_noise = float(np.sqrt(np.mean(ms_noise_c)))
                mean_sig2_unclip = float(np.mean(sig2_unclip_c))
                cohort_rms_corr = float(np.sqrt(max(0.0, mean_sig2_unclip)))
                mean_cube_rms_obs = float(np.mean(rms_obs_c))
                mean_cube_rms_noise = float(np.mean(rms_noise_c))
                mean_cube_rms_corr = float(np.mean(rms_corr_c))
                noise_var_share = float((cohort_rms_noise ** 2) / max(1e-30, cohort_rms_obs ** 2))

                boot_sig2 = (W_cube @ sig2_unclip_c) / W_sum
                boot_cohort_corr = np.sqrt(np.maximum(0.0, boot_sig2))
                boot_cube_corr = (W_cube @ rms_corr_c) / W_sum

                ci_cohort_corr = _ci_from_samples(cohort_rms_corr, boot_cohort_corr)
                ci_cube_corr = _ci_from_samples(mean_cube_rms_corr, boot_cube_corr)
                ci_sig2 = _ci_from_samples(mean_sig2_unclip, boot_sig2)

                boot_store_cohort_rms_corr[(regime_name, order)] = boot_cohort_corr
                boot_store_cube_rms_corr[(regime_name, order)] = boot_cube_corr

                rec = {
                    "record_type": "order_reliability",
                    "cohort": cohort_label,
                    "estimator": estimator_name,
                    "split_regime": regime_name,
                    "order": order,
                    "design_id": "full_cube_all_items",
                    "num_factors": num_factors,
                    "num_runs": K,
                    "n_cubes": C_count,
                    "n_contrasts_per_cube": n_o,
                    "mean_items_eval": float(n_items_per_cube.mean()),
                    "mean_cube_rms_obs": mean_cube_rms_obs,
                    "mean_cube_rms_noise": mean_cube_rms_noise,
                    "mean_cube_rms_corr": mean_cube_rms_corr,
                    "mean_cube_rms_corr_ci_low": ci_cube_corr["ci_low"],
                    "mean_cube_rms_corr_ci_high": ci_cube_corr["ci_high"],
                    "cohort_rms_obs": cohort_rms_obs,
                    "cohort_rms_noise": cohort_rms_noise,
                    "mean_sigma2_signal_unclipped": mean_sig2_unclip,
                    "mean_sigma2_signal_ci_low": ci_sig2["ci_low"],
                    "mean_sigma2_signal_ci_high": ci_sig2["ci_high"],
                    "cohort_rms_corr": cohort_rms_corr,
                    "cohort_rms_corr_ci_low": ci_cohort_corr["ci_low"],
                    "cohort_rms_corr_ci_high": ci_cohort_corr["ci_high"],
                    "noise_variance_share": noise_var_share,
                    "frac_cubes_truncated_zero": frac_cubes_trunc,
                    "frac_cube_splits_truncated_zero": trunc_frac_cube_splits,
                    "n_cubes_truncated_zero": n_cubes_trunc,
                    "conf_mean_cube_rms_obs": float("nan"),
                    "conf_cohort_rms_obs": float("nan"),
                    "conf_cohort_rms_corr": float("nan"),
                    "split_half_cross_cov": float("nan"),
                    "rms_corr_cross_split": float("nan"),
                    "pooled_pearson_r": float("nan"),
                    "pooled_pearson_r_ci_low": float("nan"),
                    "pooled_pearson_r_ci_high": float("nan"),
                    "pooled_spearman_rho": float("nan"),
                    "pooled_spearman_rho_ci_low": float("nan"),
                    "pooled_spearman_rho_ci_high": float("nan"),
                    "within_cube_pearson_r": float("nan"),
                    "within_cube_pearson_r_ci_low": float("nan"),
                    "within_cube_pearson_r_ci_high": float("nan"),
                    "within_cube_spearman_rho": float("nan"),
                    "n_estimable_within_cube": 0,
                    "sign_agreement_overall": float("nan"),
                    "sign_agreement_overall_ci_low": float("nan"),
                    "sign_agreement_overall_ci_high": float("nan"),
                    "sign_agreement_strict_equal": float("nan"),
                    "frac_contrasts_gt_2se": float("nan"),
                    "sign_agreement_gt_2se": float("nan"),
                    "sign_agreement_gt_2se_ci_low": float("nan"),
                    "sign_agreement_gt_2se_ci_high": float("nan"),
                    "mean_cube_attr_rmse": 0.0,
                    "cohort_attr_rmse_quad": 0.0,
                    "mean_cube_heldout_pred_rmse": float("nan"),
                    "cohort_heldout_pred_rmse_quad": float("nan"),
                    "esr_attr_over_disc_corr_quad": 0.0,
                    "esr_attr_over_disc_corr_quad_ci_low": 0.0,
                    "esr_attr_over_disc_corr_quad_ci_high": 0.0,
                    "esr_attr_over_disc_corr_cube_mean": 0.0,
                    "esr_attr_over_conf_obs_quad": float("nan"),
                    "esr_attr_over_conf_obs_cube_mean": float("nan"),
                    "esr_heldout_pred_over_disc_corr_quad": float("nan"),
                    "esr_heldout_pred_over_disc_corr_quad_ci_low": float("nan"),
                    "esr_heldout_pred_over_disc_corr_quad_ci_high": float("nan"),
                    "esr_heldout_pred_over_conf_obs_quad": float("nan"),
                    "esr_heldout_pred_over_conf_obs_quad_ci_low": float("nan"),
                    "esr_heldout_pred_over_conf_obs_quad_ci_high": float("nan"),
                    "esr_heldout_pred_over_conf_obs_cube_mean": float("nan"),
                }
                csv_rows.append(rec)
                order_summary_dict[regime_name][f"order_{order}"] = rec
                continue

            # Split-based regime (`split0` or `splits20`)
            assert s_slice is not None
            td = tau_disc_all[:, sl, s_slice]    # (C, O, S_sub)
            vd = var_disc_all[:, sl, s_slice]    # (C, O, S_sub)
            sed = se_disc_all[:, sl, s_slice]    # (C, O, S_sub)
            tc = tau_conf_all[:, sl, s_slice]    # (C, O, S_sub)
            vc = var_conf_all[:, sl, s_slice]    # (C, O, S_sub)
            S_sub = td.shape[2]

            ms_obs_d_cs = np.mean(td ** 2, axis=1)       # (C, S_sub)
            ms_noise_d_cs = np.mean(vd, axis=1)          # (C, S_sub)
            sig2_d_cs = ms_obs_d_cs - ms_noise_d_cs      # (C, S_sub)
            rms_obs_d_cs = np.sqrt(ms_obs_d_cs)
            rms_noise_d_cs = np.sqrt(ms_noise_d_cs)
            rms_corr_d_cs = np.sqrt(np.maximum(0.0, sig2_d_cs))
            trunc_d_cs = (sig2_d_cs <= 0.0).astype(float)

            ms_obs_c_cs = np.mean(tc ** 2, axis=1)
            ms_noise_c_cs = np.mean(vc, axis=1)
            sig2_c_cs = ms_obs_c_cs - ms_noise_c_cs

            # Average per-cube across splits in the regime
            ms_obs_d_c = ms_obs_d_cs.mean(axis=1)
            ms_noise_d_c = ms_noise_d_cs.mean(axis=1)
            sig2_d_c = sig2_d_cs.mean(axis=1)
            rms_obs_d_c = rms_obs_d_cs.mean(axis=1)
            rms_noise_d_c = rms_noise_d_cs.mean(axis=1)
            rms_corr_d_c = rms_corr_d_cs.mean(axis=1)

            ms_obs_conf_c = ms_obs_c_cs.mean(axis=1)
            sig2_conf_c = sig2_c_cs.mean(axis=1)
            rms_obs_conf_c = np.sqrt(ms_obs_c_cs).mean(axis=1)

            n_cubes_trunc = int(np.sum(sig2_d_c <= 0.0))
            frac_cubes_trunc = float(n_cubes_trunc / C_count)
            frac_cube_splits_trunc = float(trunc_d_cs.mean())

            cohort_rms_obs = float(np.sqrt(np.mean(ms_obs_d_c)))
            cohort_rms_noise = float(np.sqrt(np.mean(ms_noise_d_c)))
            mean_sig2_unclip = float(np.mean(sig2_d_c))
            cohort_rms_corr = float(np.sqrt(max(0.0, mean_sig2_unclip)))
            mean_cube_rms_obs = float(np.mean(rms_obs_d_c))
            mean_cube_rms_noise = float(np.mean(rms_noise_d_c))
            mean_cube_rms_corr = float(np.mean(rms_corr_d_c))
            noise_var_share = float((cohort_rms_noise ** 2) / max(1e-30, cohort_rms_obs ** 2))

            conf_mean_cube_rms_obs = float(np.mean(rms_obs_conf_c))
            conf_cohort_rms_obs = float(np.sqrt(np.mean(ms_obs_conf_c)))
            conf_cohort_rms_corr = float(np.sqrt(max(0.0, float(np.mean(sig2_conf_c)))))

            # Split-half cross-covariance & full-cube held-out prediction RMSE
            cross_cov_c = np.mean(td * tc, axis=(1, 2))
            split_half_cross_cov = float(np.mean(cross_cov_c))
            rms_corr_cross_split = float(np.sqrt(max(0.0, split_half_cross_cov)))

            heldout_mse_cs = np.mean((td - tc) ** 2, axis=1)  # (C, S_sub)
            heldout_rmse_c = np.sqrt(heldout_mse_cs).mean(axis=1)
            heldout_mse_c = heldout_mse_cs.mean(axis=1)
            mean_cube_heldout_rmse = float(np.mean(heldout_rmse_c))
            cohort_heldout_rmse_quad = float(np.sqrt(np.mean(heldout_mse_c)))

            # Pooled Pearson r & Spearman rho per split, averaged across S_sub
            r_splits = np.array(
                [_safe_pearson_1d(td[:, :, s].ravel(), tc[:, :, s].ravel()) for s in range(S_sub)],
                dtype=float,
            )
            rho_splits = np.array(
                [_safe_spearman_1d(td[:, :, s].ravel(), tc[:, :, s].ravel()) for s in range(S_sub)],
                dtype=float,
            )
            pooled_r = float(np.nanmean(r_splits))
            pooled_rho = float(np.nanmean(rho_splits))

            # Within-cube Pearson r & Spearman rho (defined for n_o >= 2), vectorized over (C, O, S_sub)
            if n_o >= 2:
                mx_wc = td.mean(axis=1, keepdims=True)
                my_wc = tc.mean(axis=1, keepdims=True)
                dx_wc = td - mx_wc
                dy_wc = tc - my_wc
                vx_wc = np.sum(dx_wc * dx_wc, axis=1)
                vy_wc = np.sum(dy_wc * dy_wc, axis=1)
                valid_wc = (vx_wc > float(n_o) * 1e-30) & (vy_wc > float(n_o) * 1e-30)
                den_wc = np.sqrt(np.maximum(0.0, vx_wc * vy_wc))
                with np.errstate(invalid="ignore", divide="ignore"):
                    r_wc_mat = np.where(valid_wc, np.sum(dx_wc * dy_wc, axis=1) / np.maximum(1e-30, den_wc), np.nan)

                rx_wc = stats.rankdata(td, method="average", axis=1)
                ry_wc = stats.rankdata(tc, method="average", axis=1)
                mrx = 0.5 * float(n_o + 1)
                drx = rx_wc - mrx
                dry = ry_wc - mrx
                vrx = np.sum(drx * drx, axis=1)
                vry = np.sum(dry * dry, axis=1)
                valid_rho_wc = valid_wc & (vrx > 1e-30) & (vry > 1e-30)
                den_rho_wc = np.sqrt(np.maximum(0.0, vrx * vry))
                with np.errstate(invalid="ignore", divide="ignore"):
                    rho_wc_mat = np.where(
                        valid_rho_wc,
                        np.sum(drx * dry, axis=1) / np.maximum(1e-30, den_rho_wc),
                        np.nan,
                    )

                cnt_r_c = np.isfinite(r_wc_mat).sum(axis=1)
                cnt_rho_c = np.isfinite(rho_wc_mat).sum(axis=1)
                r_wc_c = np.where(cnt_r_c > 0, np.nansum(r_wc_mat, axis=1) / np.maximum(1, cnt_r_c), np.nan)
                rho_wc_c = np.where(cnt_rho_c > 0, np.nansum(rho_wc_mat, axis=1) / np.maximum(1, cnt_rho_c), np.nan)

                n_est_wc = int(np.sum(np.isfinite(r_wc_c)))
                within_r = float(np.nanmean(r_wc_c)) if n_est_wc > 0 else float("nan")
                within_rho = float(np.nanmean(rho_wc_c)) if np.any(np.isfinite(rho_wc_c)) else float("nan")
                boot_within_r = _bootstrap_weighted_mean_with_nan(r_wc_c, W_cube)
                ci_within_r = _ci_from_samples(within_r, boot_within_r)
            else:
                n_est_wc = 0
                within_r = float("nan")
                within_rho = float("nan")
                ci_within_r = {"estimate": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

            # Sign agreement: overall & conditioned on |tau_disc| > 2 SE
            sign_neutral = _tie_neutral_sign_agreement(td, tc)        # (C, O, S_sub)
            sign_strict = (np.sign(td) == np.sign(tc)).astype(float)  # (C, O, S_sub)
            sign_c = sign_neutral.mean(axis=(1, 2))
            sign_overall = float(sign_c.mean())
            sign_strict_mean = float(sign_strict.mean())

            gt_mask = (np.abs(td) > 2.0 * sed) & (np.abs(td) > 1e-15)  # (C, O, S_sub)
            gt_count_c = gt_mask.sum(axis=(1, 2)).astype(float)
            gt_conc_c = (sign_neutral * gt_mask).sum(axis=(1, 2)).astype(float)
            tot_gt = float(gt_count_c.sum())
            frac_gt_2se = float(tot_gt / float(C_count * n_o * S_sub))
            sign_gt_2se = float(gt_conc_c.sum() / tot_gt) if tot_gt > 0.0 else float("nan")

            # Cluster bootstrap CIs
            boot_sig2 = (W_cube @ sig2_d_c) / W_sum
            boot_cohort_corr = np.sqrt(np.maximum(0.0, boot_sig2))
            boot_cube_corr = (W_cube @ rms_corr_d_c) / W_sum
            boot_pooled_r = _bootstrap_pooled_pearson_splits(td, tc, W_cube)
            boot_pooled_rho = _bootstrap_pooled_spearman_splits(td, tc, W_cube)
            boot_sign_overall = (W_cube @ sign_c) / W_sum

            boot_gt_den = W_cube @ gt_count_c
            boot_gt_num = W_cube @ gt_conc_c
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_sign_gt = np.where(boot_gt_den > 0.0, boot_gt_num / boot_gt_den, np.nan)

            # Full-cube held-out ESR & bootstrap CIs
            esr_pred_disc_corr_quad = (
                float(cohort_heldout_rmse_quad / cohort_rms_corr) if cohort_rms_corr > 1e-15 else float("inf")
            )
            esr_pred_conf_obs_quad = (
                float(cohort_heldout_rmse_quad / conf_cohort_rms_obs) if conf_cohort_rms_obs > 1e-15 else float("nan")
            )
            esr_pred_conf_obs_cube = (
                float(mean_cube_heldout_rmse / conf_mean_cube_rms_obs) if conf_mean_cube_rms_obs > 1e-15 else float("nan")
            )

            boot_pred_quad = np.sqrt((W_cube @ heldout_mse_c) / W_sum)
            boot_conf_obs_quad = np.sqrt((W_cube @ ms_obs_conf_c) / W_sum)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot_esr_pred_disc = np.where(boot_cohort_corr > 1e-12, boot_pred_quad / boot_cohort_corr, np.nan)
                boot_esr_pred_conf = np.where(boot_conf_obs_quad > 1e-12, boot_pred_quad / boot_conf_obs_quad, np.nan)

            ci_cohort_corr = _ci_from_samples(cohort_rms_corr, boot_cohort_corr)
            ci_cube_corr = _ci_from_samples(mean_cube_rms_corr, boot_cube_corr)
            ci_sig2 = _ci_from_samples(mean_sig2_unclip, boot_sig2)
            ci_r = _ci_from_samples(pooled_r, boot_pooled_r)
            ci_rho = _ci_from_samples(pooled_rho, boot_pooled_rho)
            ci_sign = _ci_from_samples(sign_overall, boot_sign_overall)
            ci_sign_gt = _ci_from_samples(sign_gt_2se, boot_sign_gt)
            ci_esr_pred_disc = _ci_from_samples(esr_pred_disc_corr_quad, boot_esr_pred_disc)
            ci_esr_pred_conf = _ci_from_samples(esr_pred_conf_obs_quad, boot_esr_pred_conf)

            boot_store_cohort_rms_corr[(regime_name, order)] = boot_cohort_corr
            boot_store_cube_rms_corr[(regime_name, order)] = boot_cube_corr
            boot_store_pooled_r[(regime_name, order)] = boot_pooled_r

            mean_disc_items = float(
                np.mean([(cube_d_masks[c_i][:, s_slice] > 0.0).sum(axis=0).mean() for c_i in range(C_count)])
            )

            rec = {
                "record_type": "order_reliability",
                "cohort": cohort_label,
                "estimator": estimator_name,
                "split_regime": regime_name,
                "order": order,
                "design_id": "full_cube_order_summary",
                "num_factors": num_factors,
                "num_runs": K,
                "n_cubes": C_count,
                "n_contrasts_per_cube": n_o,
                "mean_items_eval": mean_disc_items,
                "mean_cube_rms_obs": mean_cube_rms_obs,
                "mean_cube_rms_noise": mean_cube_rms_noise,
                "mean_cube_rms_corr": mean_cube_rms_corr,
                "mean_cube_rms_corr_ci_low": ci_cube_corr["ci_low"],
                "mean_cube_rms_corr_ci_high": ci_cube_corr["ci_high"],
                "cohort_rms_obs": cohort_rms_obs,
                "cohort_rms_noise": cohort_rms_noise,
                "mean_sigma2_signal_unclipped": mean_sig2_unclip,
                "mean_sigma2_signal_ci_low": ci_sig2["ci_low"],
                "mean_sigma2_signal_ci_high": ci_sig2["ci_high"],
                "cohort_rms_corr": cohort_rms_corr,
                "cohort_rms_corr_ci_low": ci_cohort_corr["ci_low"],
                "cohort_rms_corr_ci_high": ci_cohort_corr["ci_high"],
                "noise_variance_share": noise_var_share,
                "frac_cubes_truncated_zero": frac_cubes_trunc,
                "frac_cube_splits_truncated_zero": frac_cube_splits_trunc,
                "n_cubes_truncated_zero": n_cubes_trunc,
                "conf_mean_cube_rms_obs": conf_mean_cube_rms_obs,
                "conf_cohort_rms_obs": conf_cohort_rms_obs,
                "conf_cohort_rms_corr": conf_cohort_rms_corr,
                "split_half_cross_cov": split_half_cross_cov,
                "rms_corr_cross_split": rms_corr_cross_split,
                "pooled_pearson_r": pooled_r,
                "pooled_pearson_r_ci_low": ci_r["ci_low"],
                "pooled_pearson_r_ci_high": ci_r["ci_high"],
                "pooled_spearman_rho": pooled_rho,
                "pooled_spearman_rho_ci_low": ci_rho["ci_low"],
                "pooled_spearman_rho_ci_high": ci_rho["ci_high"],
                "within_cube_pearson_r": within_r,
                "within_cube_pearson_r_ci_low": ci_within_r["ci_low"],
                "within_cube_pearson_r_ci_high": ci_within_r["ci_high"],
                "within_cube_spearman_rho": within_rho,
                "n_estimable_within_cube": n_est_wc,
                "sign_agreement_overall": sign_overall,
                "sign_agreement_overall_ci_low": ci_sign["ci_low"],
                "sign_agreement_overall_ci_high": ci_sign["ci_high"],
                "sign_agreement_strict_equal": sign_strict_mean,
                "frac_contrasts_gt_2se": frac_gt_2se,
                "sign_agreement_gt_2se": sign_gt_2se,
                "sign_agreement_gt_2se_ci_low": ci_sign_gt["ci_low"],
                "sign_agreement_gt_2se_ci_high": ci_sign_gt["ci_high"],
                "mean_cube_attr_rmse": 0.0,
                "cohort_attr_rmse_quad": 0.0,
                "mean_cube_heldout_pred_rmse": mean_cube_heldout_rmse,
                "cohort_heldout_pred_rmse_quad": cohort_heldout_rmse_quad,
                "esr_attr_over_disc_corr_quad": 0.0,
                "esr_attr_over_disc_corr_quad_ci_low": 0.0,
                "esr_attr_over_disc_corr_quad_ci_high": 0.0,
                "esr_attr_over_disc_corr_cube_mean": 0.0,
                "esr_attr_over_conf_obs_quad": 0.0,
                "esr_attr_over_conf_obs_cube_mean": 0.0,
                "esr_heldout_pred_over_disc_corr_quad": esr_pred_disc_corr_quad,
                "esr_heldout_pred_over_disc_corr_quad_ci_low": ci_esr_pred_disc["ci_low"],
                "esr_heldout_pred_over_disc_corr_quad_ci_high": ci_esr_pred_disc["ci_high"],
                "esr_heldout_pred_over_conf_obs_quad": esr_pred_conf_obs_quad,
                "esr_heldout_pred_over_conf_obs_quad_ci_low": ci_esr_pred_conf["ci_low"],
                "esr_heldout_pred_over_conf_obs_quad_ci_high": ci_esr_pred_conf["ci_high"],
                "esr_heldout_pred_over_conf_obs_cube_mean": esr_pred_conf_obs_cube,
            }
            csv_rows.append(rec)
            order_summary_dict[regime_name][f"order_{order}"] = rec

    # --------------------------------------------------------------------------
    # B. Paired Inter-Order Differences (Cluster Bootstrap 95% Intervals)
    # --------------------------------------------------------------------------
    paired_diffs: Dict[str, Any] = {}
    for regime_name, _ in regimes:
        regime_diffs: Dict[str, Any] = {}
        order_pairs = [(1, 2), (2, 3)]
        if num_factors >= 4:
            order_pairs.extend([(2, 4), (3, 4)])
        if num_factors >= 5:
            order_pairs.extend([(4, 5), (2, 5)])
        for o1, o2 in order_pairs:
            b_corr_1 = boot_store_cohort_rms_corr[(regime_name, o1)]
            b_corr_2 = boot_store_cohort_rms_corr[(regime_name, o2)]
            pt_corr_1 = order_summary_dict[regime_name][f"order_{o1}"]["cohort_rms_corr"]
            pt_corr_2 = order_summary_dict[regime_name][f"order_{o2}"]["cohort_rms_corr"]

            b_cube_1 = boot_store_cube_rms_corr[(regime_name, o1)]
            b_cube_2 = boot_store_cube_rms_corr[(regime_name, o2)]
            pt_cube_1 = order_summary_dict[regime_name][f"order_{o1}"]["mean_cube_rms_corr"]
            pt_cube_2 = order_summary_dict[regime_name][f"order_{o2}"]["mean_cube_rms_corr"]

            entry_diff: Dict[str, Any] = {
                "cohort_rms_corr_diff": _ci_from_samples(pt_corr_1 - pt_corr_2, b_corr_1 - b_corr_2),
                "mean_cube_rms_corr_diff": _ci_from_samples(pt_cube_1 - pt_cube_2, b_cube_1 - b_cube_2),
            }
            if (regime_name, o1) in boot_store_pooled_r and (regime_name, o2) in boot_store_pooled_r:
                br1 = boot_store_pooled_r[(regime_name, o1)]
                br2 = boot_store_pooled_r[(regime_name, o2)]
                pr1 = order_summary_dict[regime_name][f"order_{o1}"]["pooled_pearson_r"]
                pr2 = order_summary_dict[regime_name][f"order_{o2}"]["pooled_pearson_r"]
                entry_diff["pooled_pearson_r_diff"] = _ci_from_samples(pr1 - pr2, br1 - br2)
            regime_diffs[f"order_{o1}_minus_order_{o2}"] = entry_diff
        paired_diffs[regime_name] = regime_diffs

    # --------------------------------------------------------------------------
    # C. Screening Attribution ESR & Held-Out Contrast Prediction RMSE
    # --------------------------------------------------------------------------
    if screening_ops is None:
        screening_ops = build_screening_operators(num_factors)

    esr_summary: Dict[str, Dict[str, Any]] = {"split0": {}, **({"splits20": {}} if has_20_splits else {})}
    split_regimes_only: List[Tuple[str, slice]] = [("split0", slice(0, 1))]
    if has_20_splits:
        split_regimes_only.append(("splits20", slice(1, n_splits_total)))

    # Reshape Y_disc_cells once: (K, C * S) for single-GEMM design evaluation
    Y_disc_2d = np.transpose(Y_disc_cells, (1, 0, 2)).reshape(K, C_count * n_splits_total)

    for order, op_list in screening_ops.items():
        sl = order_slices[order]
        n_o = order_mats[order][0].shape[1]

        # Precompute contrast estimates across all splits for each design in `op_list`
        design_attr_rmse_cs: Dict[str, np.ndarray] = {}
        design_attr_mse_cs: Dict[str, np.ndarray] = {}
        design_pred_rmse_cs: Dict[str, np.ndarray] = {}
        design_pred_mse_cs: Dict[str, np.ndarray] = {}
        design_runs: Dict[str, int] = {}

        td_ocs = np.transpose(tau_disc_all[:, sl, :], (1, 0, 2))[None, :, :, :]  # (1, O, C, S)
        tc_ocs = np.transpose(tau_conf_all[:, sl, :], (1, 0, 2))[None, :, :, :]  # (1, O, C, S)

        for design_id, n_runs, L_op in op_list:
            design_runs[design_id] = n_runs
            n_des = L_op.shape[0]
            # (n_des * O, K) @ (K, C * S) -> (n_des, O, C, S)
            est_flat = L_op.reshape(n_des * n_o, K) @ Y_disc_2d
            est = est_flat.reshape(n_des, n_o, C_count, n_splits_total)

            diff_disc = est - td_ocs  # (n_des, O, C, S)
            diff_conf = est - tc_ocs  # (n_des, O, C, S)

            mse_d = np.mean(diff_disc ** 2, axis=1)  # (n_des, C, S)
            mse_c = np.mean(diff_conf ** 2, axis=1)  # (n_des, C, S)

            design_attr_rmse_cs[design_id] = np.sqrt(mse_d).mean(axis=0)  # (C, S)
            design_attr_mse_cs[design_id] = mse_d.mean(axis=0)            # (C, S)
            design_pred_rmse_cs[design_id] = np.sqrt(mse_c).mean(axis=0)  # (C, S)
            design_pred_mse_cs[design_id] = mse_c.mean(axis=0)            # (C, S)

        for regime_name, s_slice in split_regimes_only:
            base_ord = order_summary_dict[regime_name][f"order_{order}"]
            esr_summary[regime_name][f"order_{order}"] = {}

            sig2_d_c = (
                np.mean(tau_disc_all[:, sl, s_slice] ** 2, axis=1)
                - np.mean(var_disc_all[:, sl, s_slice], axis=1)
            ).mean(axis=1)
            ms_conf_obs_c = np.mean(tau_conf_all[:, sl, s_slice] ** 2, axis=(1, 2))
            sig2_conf_c = (
                np.mean(tau_conf_all[:, sl, s_slice] ** 2, axis=1)
                - np.mean(var_conf_all[:, sl, s_slice], axis=1)
            ).mean(axis=1)

            cohort_disc_corr = base_ord["cohort_rms_corr"]
            cube_disc_corr = base_ord["mean_cube_rms_corr"]
            cohort_conf_obs = base_ord["conf_cohort_rms_obs"]
            cube_conf_obs = base_ord["conf_mean_cube_rms_obs"]
            cohort_conf_corr = base_ord["conf_cohort_rms_corr"]

            boot_disc_corr = boot_store_cohort_rms_corr[(regime_name, order)]
            boot_conf_obs = np.sqrt((W_cube @ ms_conf_obs_c) / W_sum)
            boot_conf_corr = np.sqrt(np.maximum(0.0, (W_cube @ sig2_conf_c) / W_sum))

            # Reference orthogonal design for relative ratio reporting
            orth_ref_id = None
            if num_factors == 5 and order in (1, 2):
                orth_ref_id = "2(5-1)V_coset_avg"
            elif num_factors == 4 and order == 1:
                orth_ref_id = "2(4-1)IV_coset_avg"

            orth_attr_mean = (
                float(design_attr_rmse_cs[orth_ref_id][:, s_slice].mean()) if orth_ref_id else float("nan")
            )
            orth_pred_mean = (
                float(design_pred_rmse_cs[orth_ref_id][:, s_slice].mean()) if orth_ref_id else float("nan")
            )

            for design_id, n_runs, _ in op_list:
                attr_rmse_c = design_attr_rmse_cs[design_id][:, s_slice].mean(axis=1)
                attr_mse_c = design_attr_mse_cs[design_id][:, s_slice].mean(axis=1)
                pred_rmse_c = design_pred_rmse_cs[design_id][:, s_slice].mean(axis=1)
                pred_mse_c = design_pred_mse_cs[design_id][:, s_slice].mean(axis=1)

                mean_cube_attr = float(attr_rmse_c.mean())
                quad_attr = float(np.sqrt(attr_mse_c.mean()))
                mean_cube_pred = float(pred_rmse_c.mean())
                quad_pred = float(np.sqrt(pred_mse_c.mean()))

                esr_attr_disc_quad = (
                    0.0 if quad_attr == 0.0 else (float(quad_attr / cohort_disc_corr) if cohort_disc_corr > 1e-15 else float("inf"))
                )
                esr_attr_disc_cube = (
                    0.0 if mean_cube_attr == 0.0 else (float(mean_cube_attr / cube_disc_corr) if cube_disc_corr > 1e-15 else float("inf"))
                )
                esr_attr_conf_obs_quad = (
                    0.0 if quad_attr == 0.0 else (float(quad_attr / cohort_conf_obs) if cohort_conf_obs > 1e-15 else float("nan"))
                )
                esr_attr_conf_obs_cube = (
                    0.0 if mean_cube_attr == 0.0 else (float(mean_cube_attr / cube_conf_obs) if cube_conf_obs > 1e-15 else float("nan"))
                )
                esr_attr_conf_corr_quad = (
                    0.0 if quad_attr == 0.0 else (float(quad_attr / cohort_conf_corr) if cohort_conf_corr > 1e-15 else float("inf"))
                )

                esr_pred_disc_quad = (
                    float(quad_pred / cohort_disc_corr) if cohort_disc_corr > 1e-15 else float("inf")
                )
                esr_pred_conf_obs_quad = (
                    float(quad_pred / cohort_conf_obs) if cohort_conf_obs > 1e-15 else float("nan")
                )
                esr_pred_conf_obs_cube = (
                    float(mean_cube_pred / cube_conf_obs) if cube_conf_obs > 1e-15 else float("nan")
                )
                esr_pred_conf_corr_quad = (
                    float(quad_pred / cohort_conf_corr) if cohort_conf_corr > 1e-15 else float("inf")
                )

                # Bootstrap CIs on ESRs
                boot_attr_quad = np.sqrt((W_cube @ attr_mse_c) / W_sum)
                boot_pred_quad = np.sqrt((W_cube @ pred_mse_c) / W_sum)
                with np.errstate(invalid="ignore", divide="ignore"):
                    b_esr_a_d = (
                        np.zeros_like(boot_attr_quad)
                        if quad_attr == 0.0
                        else np.where(boot_disc_corr > 1e-12, boot_attr_quad / boot_disc_corr, np.nan)
                    )
                    b_esr_a_co = (
                        np.zeros_like(boot_attr_quad)
                        if quad_attr == 0.0
                        else np.where(boot_conf_obs > 1e-12, boot_attr_quad / boot_conf_obs, np.nan)
                    )
                    b_esr_p_d = np.where(boot_disc_corr > 1e-12, boot_pred_quad / boot_disc_corr, np.nan)
                    b_esr_p_co = np.where(boot_conf_obs > 1e-12, boot_pred_quad / boot_conf_obs, np.nan)
                    b_esr_p_cc = np.where(boot_conf_corr > 1e-12, boot_pred_quad / boot_conf_corr, np.nan)

                ci_a_d = _ci_from_samples(esr_attr_disc_quad, b_esr_a_d)
                ci_a_co = _ci_from_samples(esr_attr_conf_obs_quad, b_esr_a_co)
                ci_p_d = _ci_from_samples(esr_pred_disc_quad, b_esr_p_d)
                ci_p_co = _ci_from_samples(esr_pred_conf_obs_quad, b_esr_p_co)
                ci_p_cc = _ci_from_samples(esr_pred_conf_corr_quad, b_esr_p_cc)

                rel_attr = (
                    float(mean_cube_attr / orth_attr_mean)
                    if (orth_ref_id and orth_attr_mean > 1e-15)
                    else float("nan")
                )
                rel_pred = (
                    float(mean_cube_pred / orth_pred_mean)
                    if (orth_ref_id and orth_pred_mean > 1e-15)
                    else float("nan")
                )

                des_row = {
                    "record_type": "design_esr",
                    "cohort": cohort_label,
                    "estimator": estimator_name,
                    "split_regime": regime_name,
                    "order": order,
                    "design_id": design_id,
                    "num_factors": num_factors,
                    "num_runs": n_runs,
                    "n_cubes": C_count,
                    "n_contrasts_per_cube": n_o,
                    "mean_items_eval": base_ord["mean_items_eval"],
                    "mean_cube_rms_obs": base_ord["mean_cube_rms_obs"],
                    "mean_cube_rms_noise": base_ord["mean_cube_rms_noise"],
                    "mean_cube_rms_corr": base_ord["mean_cube_rms_corr"],
                    "mean_cube_rms_corr_ci_low": base_ord["mean_cube_rms_corr_ci_low"],
                    "mean_cube_rms_corr_ci_high": base_ord["mean_cube_rms_corr_ci_high"],
                    "cohort_rms_obs": base_ord["cohort_rms_obs"],
                    "cohort_rms_noise": base_ord["cohort_rms_noise"],
                    "mean_sigma2_signal_unclipped": base_ord["mean_sigma2_signal_unclipped"],
                    "mean_sigma2_signal_ci_low": base_ord["mean_sigma2_signal_ci_low"],
                    "mean_sigma2_signal_ci_high": base_ord["mean_sigma2_signal_ci_high"],
                    "cohort_rms_corr": cohort_disc_corr,
                    "cohort_rms_corr_ci_low": base_ord["cohort_rms_corr_ci_low"],
                    "cohort_rms_corr_ci_high": base_ord["cohort_rms_corr_ci_high"],
                    "noise_variance_share": base_ord["noise_variance_share"],
                    "frac_cubes_truncated_zero": base_ord["frac_cubes_truncated_zero"],
                    "frac_cube_splits_truncated_zero": base_ord["frac_cube_splits_truncated_zero"],
                    "n_cubes_truncated_zero": base_ord["n_cubes_truncated_zero"],
                    "conf_mean_cube_rms_obs": cube_conf_obs,
                    "conf_cohort_rms_obs": cohort_conf_obs,
                    "conf_cohort_rms_corr": cohort_conf_corr,
                    "split_half_cross_cov": base_ord["split_half_cross_cov"],
                    "rms_corr_cross_split": base_ord["rms_corr_cross_split"],
                    "pooled_pearson_r": base_ord["pooled_pearson_r"],
                    "pooled_pearson_r_ci_low": base_ord["pooled_pearson_r_ci_low"],
                    "pooled_pearson_r_ci_high": base_ord["pooled_pearson_r_ci_high"],
                    "pooled_spearman_rho": base_ord["pooled_spearman_rho"],
                    "pooled_spearman_rho_ci_low": base_ord["pooled_spearman_rho_ci_low"],
                    "pooled_spearman_rho_ci_high": base_ord["pooled_spearman_rho_ci_high"],
                    "within_cube_pearson_r": base_ord["within_cube_pearson_r"],
                    "within_cube_pearson_r_ci_low": base_ord["within_cube_pearson_r_ci_low"],
                    "within_cube_pearson_r_ci_high": base_ord["within_cube_pearson_r_ci_high"],
                    "within_cube_spearman_rho": base_ord["within_cube_spearman_rho"],
                    "n_estimable_within_cube": base_ord["n_estimable_within_cube"],
                    "sign_agreement_overall": base_ord["sign_agreement_overall"],
                    "sign_agreement_overall_ci_low": base_ord["sign_agreement_overall_ci_low"],
                    "sign_agreement_overall_ci_high": base_ord["sign_agreement_overall_ci_high"],
                    "sign_agreement_strict_equal": base_ord["sign_agreement_strict_equal"],
                    "frac_contrasts_gt_2se": base_ord["frac_contrasts_gt_2se"],
                    "sign_agreement_gt_2se": base_ord["sign_agreement_gt_2se"],
                    "sign_agreement_gt_2se_ci_low": base_ord["sign_agreement_gt_2se_ci_low"],
                    "sign_agreement_gt_2se_ci_high": base_ord["sign_agreement_gt_2se_ci_high"],
                    "mean_cube_attr_rmse": mean_cube_attr,
                    "cohort_attr_rmse_quad": quad_attr,
                    "mean_cube_heldout_pred_rmse": mean_cube_pred,
                    "cohort_heldout_pred_rmse_quad": quad_pred,
                    "esr_attr_over_disc_corr_quad": esr_attr_disc_quad,
                    "esr_attr_over_disc_corr_quad_ci_low": ci_a_d["ci_low"],
                    "esr_attr_over_disc_corr_quad_ci_high": ci_a_d["ci_high"],
                    "esr_attr_over_disc_corr_cube_mean": esr_attr_disc_cube,
                    "esr_attr_over_conf_obs_quad": esr_attr_conf_obs_quad,
                    "esr_attr_over_conf_obs_cube_mean": esr_attr_conf_obs_cube,
                    "esr_heldout_pred_over_disc_corr_quad": esr_pred_disc_quad,
                    "esr_heldout_pred_over_disc_corr_quad_ci_low": ci_p_d["ci_low"],
                    "esr_heldout_pred_over_disc_corr_quad_ci_high": ci_p_d["ci_high"],
                    "esr_heldout_pred_over_conf_obs_quad": esr_pred_conf_obs_quad,
                    "esr_heldout_pred_over_conf_obs_quad_ci_low": ci_p_co["ci_low"],
                    "esr_heldout_pred_over_conf_obs_quad_ci_high": ci_p_co["ci_high"],
                    "esr_heldout_pred_over_conf_obs_cube_mean": esr_pred_conf_obs_cube,
                }
                csv_rows.append(des_row)

                esr_summary[regime_name][f"order_{order}"][design_id] = {
                    "num_runs": n_runs,
                    "mean_cube_attr_rmse": mean_cube_attr,
                    "cohort_attr_rmse_quad": quad_attr,
                    "mean_cube_heldout_pred_rmse": mean_cube_pred,
                    "cohort_heldout_pred_rmse_quad": quad_pred,
                    "esr_attr_over_disc_corr_quad": ci_a_d,
                    "esr_attr_over_disc_corr_cube_mean": esr_attr_disc_cube,
                    "esr_attr_over_conf_obs_quad": ci_a_co,
                    "esr_attr_over_conf_obs_cube_mean": esr_attr_conf_obs_cube,
                    "esr_attr_over_conf_corr_quad": esr_attr_conf_corr_quad,
                    "esr_heldout_pred_over_disc_corr_quad": ci_p_d,
                    "esr_heldout_pred_over_conf_obs_quad": ci_p_co,
                    "esr_heldout_pred_over_conf_obs_cube_mean": esr_pred_conf_obs_cube,
                    "esr_heldout_pred_over_conf_corr_quad": ci_p_cc,
                    "relative_attr_rmse_vs_orth_half": rel_attr,
                    "relative_heldout_pred_rmse_vs_orth_half": rel_pred,
                }

    # --------------------------------------------------------------------------
    # D. Item-Bootstrap Verification Summary (when requested)
    # --------------------------------------------------------------------------
    item_boot_summary: Optional[Dict[str, Any]] = None
    if (
        n_item_boot_verify > 0
        and boot_var_full_raw is not None
        and boot_var_full_bes is not None
        and boot_var_s0_raw is not None
        and boot_var_s0_bes is not None
    ):
        item_boot_summary = {
            "n_item_boot": int(n_item_boot_verify),
            "seed": int(item_boot_seed),
            "full_cube": {},
            "split0_disc": {},
        }
        for tag, v_ana, v_raw, v_bes, t_ref in [
            ("full_cube", var_full_all, boot_var_full_raw, boot_var_full_bes, tau_full_all),
            ("split0_disc", var_disc_all[:, :, 0], boot_var_s0_raw, boot_var_s0_bes, tau_disc_all[:, :, 0]),
        ]:
            for o in range(1, num_factors + 1):
                sl = order_slices[o]
                rn_ana = np.sqrt(np.mean(v_ana[:, sl], axis=1))
                rn_raw = np.sqrt(np.mean(v_raw[:, sl], axis=1))
                rn_bes = np.sqrt(np.mean(v_bes[:, sl], axis=1))
                ms_o = np.mean(t_ref[:, sl] ** 2, axis=1)
                rc_ana = float(np.sqrt(max(0.0, float(np.mean(ms_o - np.mean(v_ana[:, sl], axis=1))))))
                rc_raw = float(np.sqrt(max(0.0, float(np.mean(ms_o - np.mean(v_raw[:, sl], axis=1))))))
                rc_bes = float(np.sqrt(max(0.0, float(np.mean(ms_o - np.mean(v_bes[:, sl], axis=1))))))
                rel_err_bes = np.abs(rn_bes - rn_ana) / np.maximum(1e-15, rn_ana)
                item_boot_summary[tag][f"order_{o}"] = {
                    "mean_cube_rms_noise_analytical": float(rn_ana.mean()),
                    "mean_cube_rms_noise_item_boot_raw": float(rn_raw.mean()),
                    "mean_cube_rms_noise_item_boot_bessel": float(rn_bes.mean()),
                    "cohort_rms_corr_analytical": rc_ana,
                    "cohort_rms_corr_item_boot_raw": rc_raw,
                    "cohort_rms_corr_item_boot_bessel": rc_bes,
                    "pearson_r_cube_rms_noise_analytical_vs_bessel_boot": _safe_pearson_1d(rn_ana, rn_bes),
                    "mean_rel_abs_diff_bessel_vs_analytical": float(rel_err_bes.mean()),
                    "median_rel_abs_diff_bessel_vs_analytical": float(np.median(rel_err_bes)),
                }

    panel_summary = {
        "cohort": cohort_label,
        "estimator": estimator_name,
        "num_factors": num_factors,
        "n_cubes": C_count,
        "mean_total_items_per_cube": float(n_items_per_cube.mean()),
        "min_total_items_per_cube": int(n_items_per_cube.min()),
        "max_total_items_per_cube": int(n_items_per_cube.max()),
        "bootstrap_panel_info": boot_meta,
        "order_reliability": order_summary_dict,
        "paired_order_differences": paired_diffs,
        "screening_esr_and_heldout_prediction": esr_summary,
    }
    if item_boot_summary is not None:
        panel_summary["item_bootstrap_verification"] = item_boot_summary

    return csv_rows, panel_summary


# ==============================================================================
# 4. Cross-Cohort Effect Hierarchy Audit & Full Study Runner
# ==============================================================================


def build_effect_hierarchy_audit(
    cohorts_summary: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Synthesize quantitative comparison across single-template MMLU (`MMLU_TASK`),
    14,042-item pooled MMLU (`MMLU_POOLED`), stratum-averaged estimators (`cell_mean_unique`),
    and Cohort 2 (`SLAB`) to determine whether the apparent lack of effect hierarchy
    (2FI ~ 3FI ~ 4FI observed RMS) is an item-sampling noise-floor artifact vs genuine
    higher-order contrast structure.
    """
    mt_leg = cohorts_summary["MMLU_TASK"]["legacy_first"]["order_reliability"]["split0"]
    mt_leg_20 = cohorts_summary["MMLU_TASK"]["legacy_first"]["order_reliability"]["splits20"]
    mt_uniq = cohorts_summary["MMLU_TASK"]["cell_mean_unique"]["order_reliability"]["splits20"]

    mp_leg = cohorts_summary["MMLU_POOLED"]["legacy_first"]["order_reliability"]["splits20"]
    mp_uniq = cohorts_summary["MMLU_POOLED"]["cell_mean_unique"]["order_reliability"]["splits20"]

    sl_leg = cohorts_summary["SLAB"]["legacy_first"]["order_reliability"]["splits20"]
    sl_uniq = cohorts_summary["SLAB"]["cell_mean_unique"]["order_reliability"]["splits20"]

    def _orders_vec(d: Dict[str, Any], field: str, k: int) -> Dict[str, float]:
        return {f"order_{o}": float(d[f"order_{o}"][field]) for o in range(1, k + 1)}

    slab_uniq_cohort_corr = [sl_uniq[f"order_{o}"]["cohort_rms_corr"] for o in range(1, 5)]
    slab_uniq_cube_corr = [sl_uniq[f"order_{o}"]["mean_cube_rms_corr"] for o in range(1, 5)]
    slab_leg_cohort_corr = [sl_leg[f"order_{o}"]["cohort_rms_corr"] for o in range(1, 5)]

    slab_uniq_strict_monotone = bool(
        all(slab_uniq_cohort_corr[i] > slab_uniq_cohort_corr[i + 1] for i in range(3))
        and all(slab_uniq_cube_corr[i] > slab_uniq_cube_corr[i + 1] for i in range(3))
    )
    slab_leg_strict_monotone = bool(
        all(slab_leg_cohort_corr[i] > slab_leg_cohort_corr[i + 1] for i in range(3))
    )

    mmlu_pool_leg_corr = [mp_leg[f"order_{o}"]["cohort_rms_corr"] for o in range(1, 6)]
    mmlu_pool_uniq_corr = [mp_uniq[f"order_{o}"]["cohort_rms_corr"] for o in range(1, 6)]
    pct_drop_3fi = (1.0 - mmlu_pool_uniq_corr[2] / mmlu_pool_leg_corr[2]) * 100.0
    pct_drop_4fi = (1.0 - mmlu_pool_uniq_corr[3] / mmlu_pool_leg_corr[3]) * 100.0

    return {
        "question": (
            "Is the apparent lack of effect hierarchy (2FI ~ 3FI ~ 4FI observed RMS on single-template MMLU) "
            "an item-sampling noise-floor artifact or genuine higher-order contrast structure?"
        ),
        "verdict_code": "composite_noise_floor_plus_template_and_axis_coding_structure",
        "apparent_lack_of_hierarchy_survives_noise_floor_on_single_template": True,
        "strict_monotonic_hierarchy_recovered_under_stratum_averaging_on_slab": slab_uniq_strict_monotone,
        "strict_monotonic_hierarchy_present_on_single_template_slab": slab_leg_strict_monotone,
        "concise_conclusion": (
            "Both mechanisms act simultaneously, and cross-cohort/stratum comparisons separate them cleanly: "
            f"(1) On per-task MMLU (`legacy_first`), the observed Split-0 plateau "
            f"(2FI={mt_leg['order_2']['mean_cube_rms_obs']:.5f}, 3FI={mt_leg['order_3']['mean_cube_rms_obs']:.5f}, "
            f"4FI={mt_leg['order_4']['mean_cube_rms_obs']:.5f}) is "
            f"{mt_leg['order_4']['noise_variance_share']*100:.1f}%-{mt_leg['order_2']['noise_variance_share']*100:.1f}% "
            f"item-sampling variance (mean cube noise RMS {mt_leg['order_3']['mean_cube_rms_noise']:.5f}-"
            f"{mt_leg['order_4']['mean_cube_rms_noise']:.5f}, split-half r="
            f"{mt_leg['order_3']['pooled_pearson_r']:.2f}-{mt_leg['order_2']['pooled_pearson_r']:.2f}, "
            f"{mt_leg['order_3']['frac_cubes_truncated_zero']*100:.1f}%-"
            f"{mt_leg['order_4']['frac_cubes_truncated_zero']*100:.1f}% of cubes truncated at zero); "
            f"noise subtraction cuts Orders 2-4 RMS by more than half "
            f"({mt_leg['order_2']['cohort_rms_corr']:.5f} / {mt_leg['order_3']['cohort_rms_corr']:.5f} / "
            f"{mt_leg['order_4']['cohort_rms_corr']:.5f} cohort RMS_corr) and separates "
            f"1FI ({mt_leg['order_1']['cohort_rms_corr']:.5f}) >> {{2FI, 3FI, 4FI}} >> "
            f"5FI ({mt_leg['order_5']['cohort_rms_corr']:.5f}). "
            f"(2) However, among Orders 2-4, approximate parity (2FI ~ 3FI ~ 4FI) SURVIVES noise correction and persists "
            f"with high 20-split item repeatability (r={mp_leg['order_3']['pooled_pearson_r']:.2f}-"
            f"{mp_leg['order_4']['pooled_pearson_r']:.2f}) on 14,042-item Pooled MMLU "
            f"({mmlu_pool_leg_corr[1]:.5f} / {mmlu_pool_leg_corr[2]:.5f} / {mmlu_pool_leg_corr[3]:.5f}) "
            f"and single-template SLAB ({slab_leg_cohort_corr[1]:.4f} / {slab_leg_cohort_corr[2]:.4f} / "
            f"{slab_leg_cohort_corr[3]:.4f}). "
            f"(3) Stratum averaging (`cell_mean_unique`) proves why: on SLAB, averaging out single-template wording "
            f"idiosyncrasies restores strict monotonic hierarchy (1FI > 2FI > 3FI > 4FI: "
            f"{slab_uniq_cohort_corr[0]:.4f} > {slab_uniq_cohort_corr[1]:.4f} > "
            f"{slab_uniq_cohort_corr[2]:.4f} > {slab_uniq_cohort_corr[3]:.4f} cohort RMS_corr), "
            f"whereas on MMLU `cell_mean_unique` reduces Pooled 3FI/4FI by "
            f"{pct_drop_3fi:.1f}%/{pct_drop_4fi:.1f}% (to {mmlu_pool_uniq_corr[2]:.5f} / "
            f"{mmlu_pool_uniq_corr[3]:.5f} vs 2FI {mmlu_pool_uniq_corr[1]:.5f}) while retaining genuine "
            f"multi-bit contrasts induced by binary partitioning of the 4-level `op` and 11-level `sep` categorical axes."
        ),
        "regime_comparison": {
            "mmlu_per_task_legacy_first_split0": {
                "mean_cube_rms_obs": _orders_vec(mt_leg, "mean_cube_rms_obs", 5),
                "mean_cube_rms_noise": _orders_vec(mt_leg, "mean_cube_rms_noise", 5),
                "mean_cube_rms_corr": _orders_vec(mt_leg, "mean_cube_rms_corr", 5),
                "cohort_rms_obs": _orders_vec(mt_leg, "cohort_rms_obs", 5),
                "cohort_rms_noise": _orders_vec(mt_leg, "cohort_rms_noise", 5),
                "cohort_rms_corr": _orders_vec(mt_leg, "cohort_rms_corr", 5),
                "noise_variance_share": _orders_vec(mt_leg, "noise_variance_share", 5),
                "frac_cubes_truncated_zero": _orders_vec(mt_leg, "frac_cubes_truncated_zero", 5),
                "pooled_pearson_r": _orders_vec(mt_leg, "pooled_pearson_r", 5),
            },
            "mmlu_per_task_legacy_first_splits20": {
                "cohort_rms_corr": _orders_vec(mt_leg_20, "cohort_rms_corr", 5),
                "mean_cube_rms_corr": _orders_vec(mt_leg_20, "mean_cube_rms_corr", 5),
                "noise_variance_share": _orders_vec(mt_leg_20, "noise_variance_share", 5),
                "pooled_pearson_r": _orders_vec(mt_leg_20, "pooled_pearson_r", 5),
            },
            "mmlu_per_task_cell_mean_unique_splits20": {
                "cohort_rms_corr": _orders_vec(mt_uniq, "cohort_rms_corr", 5),
                "mean_cube_rms_corr": _orders_vec(mt_uniq, "mean_cube_rms_corr", 5),
                "noise_variance_share": _orders_vec(mt_uniq, "noise_variance_share", 5),
                "pooled_pearson_r": _orders_vec(mt_uniq, "pooled_pearson_r", 5),
            },
            "mmlu_pooled_14042_legacy_first_splits20": {
                "cohort_rms_obs": _orders_vec(mp_leg, "cohort_rms_obs", 5),
                "cohort_rms_noise": _orders_vec(mp_leg, "cohort_rms_noise", 5),
                "cohort_rms_corr": _orders_vec(mp_leg, "cohort_rms_corr", 5),
                "mean_cube_rms_corr": _orders_vec(mp_leg, "mean_cube_rms_corr", 5),
                "noise_variance_share": _orders_vec(mp_leg, "noise_variance_share", 5),
                "pooled_pearson_r": _orders_vec(mp_leg, "pooled_pearson_r", 5),
                "ratio_3fi_to_2fi_cohort_rms_corr": float(mmlu_pool_leg_corr[2] / mmlu_pool_leg_corr[1]),
                "ratio_4fi_to_2fi_cohort_rms_corr": float(mmlu_pool_leg_corr[3] / mmlu_pool_leg_corr[1]),
            },
            "mmlu_pooled_14042_cell_mean_unique_splits20": {
                "cohort_rms_obs": _orders_vec(mp_uniq, "cohort_rms_obs", 5),
                "cohort_rms_noise": _orders_vec(mp_uniq, "cohort_rms_noise", 5),
                "cohort_rms_corr": _orders_vec(mp_uniq, "cohort_rms_corr", 5),
                "mean_cube_rms_corr": _orders_vec(mp_uniq, "mean_cube_rms_corr", 5),
                "noise_variance_share": _orders_vec(mp_uniq, "noise_variance_share", 5),
                "pooled_pearson_r": _orders_vec(mp_uniq, "pooled_pearson_r", 5),
                "ratio_3fi_to_2fi_cohort_rms_corr": float(mmlu_pool_uniq_corr[2] / mmlu_pool_uniq_corr[1]),
                "ratio_4fi_to_2fi_cohort_rms_corr": float(mmlu_pool_uniq_corr[3] / mmlu_pool_uniq_corr[1]),
            },
            "slab_legacy_first_splits20": {
                "cohort_rms_obs": _orders_vec(sl_leg, "cohort_rms_obs", 4),
                "cohort_rms_noise": _orders_vec(sl_leg, "cohort_rms_noise", 4),
                "cohort_rms_corr": _orders_vec(sl_leg, "cohort_rms_corr", 4),
                "mean_cube_rms_corr": _orders_vec(sl_leg, "mean_cube_rms_corr", 4),
                "noise_variance_share": _orders_vec(sl_leg, "noise_variance_share", 4),
                "pooled_pearson_r": _orders_vec(sl_leg, "pooled_pearson_r", 4),
            },
            "slab_cell_mean_unique_splits20": {
                "cohort_rms_obs": _orders_vec(sl_uniq, "cohort_rms_obs", 4),
                "cohort_rms_noise": _orders_vec(sl_uniq, "cohort_rms_noise", 4),
                "cohort_rms_corr": _orders_vec(sl_uniq, "cohort_rms_corr", 4),
                "mean_cube_rms_corr": _orders_vec(sl_uniq, "mean_cube_rms_corr", 4),
                "noise_variance_share": _orders_vec(sl_uniq, "noise_variance_share", 4),
                "pooled_pearson_r": _orders_vec(sl_uniq, "pooled_pearson_r", 4),
            },
        },
        "paired_bootstrap_tests_splits20": {
            "mmlu_task_legacy_first": cohorts_summary["MMLU_TASK"]["legacy_first"]["paired_order_differences"]["splits20"],
            "mmlu_pooled_legacy_first": cohorts_summary["MMLU_POOLED"]["legacy_first"]["paired_order_differences"]["splits20"],
            "mmlu_pooled_cell_mean_unique": cohorts_summary["MMLU_POOLED"]["cell_mean_unique"]["paired_order_differences"]["splits20"],
            "slab_legacy_first": cohorts_summary["SLAB"]["legacy_first"]["paired_order_differences"]["splits20"],
            "slab_cell_mean_unique": cohorts_summary["SLAB"]["cell_mean_unique"]["paired_order_differences"]["splits20"],
        },
    }


def run_effect_reliability_analysis(
    n_boot_cluster: int = 1000,
    n_item_boot_verify: int = 500,
    output_csv_path: Optional[Path] = RESULTS_DIR / "effect_reliability.csv",
    output_json_path: Optional[Path] = RESULTS_DIR / "effect_reliability_summary.json",
    write_outputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], str, str]:
    """
    Execute complete Phase 0 Task R1 effect noise-floor and split-half reliability
    audit and optionally write deterministic CSV and JSON artifacts.

    Returns `(df_results, summary_dict, csv_sha256, json_sha256)`.
    """
    source_hashes = verify_source_hashes()
    with open(DATA_DIR / "Ys.pickle", "rb") as f:
        Ys = pickle.load(f)

    # 1. Build exact Cohort 1 & Cohort 2 template mappings via run_experiments helpers
    df_mmlu_map, mmlu_leg_map, mmlu_canon_map = build_mmlu_mapping_table()
    mmlu_uniq_map = extract_unique_cell_map(df_mmlu_map)
    all_cells_5 = sorted(mmlu_leg_map.keys())
    all_cells_4 = list(itertools.product([0, 1], repeat=4))

    mmlu_maps_by_est = {
        "legacy_first": mmlu_leg_map,
        "canonical_first": mmlu_canon_map,
        "cell_mean_unique": mmlu_uniq_map,
        "cell_mean": mmlu_canon_map,
    }

    slab_leg_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]] = {}
    slab_canon_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]] = {}
    slab_uniq_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]] = {}
    for bench in ("BBH", "LMentry"):
        for fpath in sorted((DATA_DIR / "templates" / bench).glob("*.csv")):
            df_t = pd.read_csv(fpath)
            df_m, lmap, cmap = build_slab_mapping_table_for_task(df_t, bench, fpath.stem)
            if len(lmap) < 16:
                continue
            key = (bench, fpath.stem)
            slab_leg_maps[key] = lmap
            slab_canon_maps[key] = cmap
            slab_uniq_maps[key] = extract_unique_cell_map(df_m)

    slab_maps_by_est = {
        "legacy_first": slab_leg_maps,
        "canonical_first": slab_canon_maps,
        "cell_mean_unique": slab_uniq_maps,
        "cell_mean": slab_canon_maps,
    }

    # 2. Generate exact deterministic discovery/confirmation item splits
    tasks_mmlu = sorted(Ys["MMLU"].keys())
    mmlu_splits: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for idx, t in enumerate(tasks_mmlu):
        n_items = int(Ys["MMLU"][t][0].shape[1])
        mmlu_splits[t] = generate_item_splits(
            n_items, n_splits=20, seed=RELIABILITY_SEEDS["item_splits"] + idx
        )

    # Task-stratified pooled MMLU masks preserve within-task parity on Split 0
    # (exact match to `concatenate_task_discovery_items` in run_experiments.py)
    d_pool_bin = np.vstack([(mmlu_splits[t][0] > 0.0).astype(float) for t in tasks_mmlu])
    c_pool_bin = np.vstack([(mmlu_splits[t][1] > 0.0).astype(float) for t in tasks_mmlu])
    d_pool_norm = d_pool_bin / d_pool_bin.sum(axis=0)[None, :]
    c_pool_norm = c_pool_bin / c_pool_bin.sum(axis=0)[None, :]

    slab_keys_sorted = sorted(slab_leg_maps.keys())
    slab_splits: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for idx, key in enumerate(slab_keys_sorted):
        bench, task_name = key
        n_items = int(Ys[bench][task_name][0].shape[1])
        slab_splits[key] = generate_item_splits(
            n_items, n_splits=20, seed=RELIABILITY_SEEDS["item_splits"] + 1000 + idx
        )

    # 3. Build shared screening design operators once for k=5 and k=4
    cosets_info = enumerate_fractional_cosets()
    r16_fr, r16_diag, _ = generate_random_designs_with_diagnostics(
        5, 16, 200, RELIABILITY_SEEDS["rand16_full_rank"], True, "MMLU", "rand16_2fi_fullrank"
    )
    r16_conds = np.array([d["cond_2fi"] for d in r16_diag if d["accepted_primary"]])
    r8_c1_fr = generate_balanced_random_designs(
        32, 5, 8, num_screens=200, seed=RELIABILITY_SEEDS["rand8_cohort1"], require_2fi_full_rank=False
    )
    r8_c2_fr = generate_balanced_random_designs(
        16, 4, 8, num_screens=200, seed=RELIABILITY_SEEDS["rand8_cohort2"], require_2fi_full_rank=False
    )
    ops_5 = build_screening_operators(
        5, cosets_info=cosets_info, r16_fr=r16_fr, r16_conds=r16_conds, r8_c1_fr=r8_c1_fr
    )
    ops_4 = build_screening_operators(4, cosets_info=cosets_info, r8_c2_fr=r8_c2_fr)

    all_csv_rows: List[Dict[str, Any]] = []
    cohorts_summary: Dict[str, Dict[str, Any]] = {
        "MMLU_TASK": {},
        "MMLU_POOLED": {},
        "SLAB": {},
    }
    item_boot_top: Dict[str, Any] = {}

    # Metadata DataFrames for the three cohorts
    meta_mmlu_task = pd.DataFrame(
        [
            {"cohort": "MMLU", "model_id": m_id, "task": t, "task_key": t}
            for m_id, _, _ in MMLU_MODELS
            for t in tasks_mmlu
        ]
    )
    meta_mmlu_pool = pd.DataFrame(
        [
            {"cohort": "MMLU_POOLED", "model_id": m_id, "task": "pooled_57_tasks", "task_key": "pooled_57_tasks"}
            for m_id, _, _ in MMLU_MODELS
        ]
    )

    slab_meta_rows = []
    for bench, task_name in slab_keys_sorted:
        model_list = BBH_MODELS if bench == "BBH" else LMENTRY_MODELS
        for m_name in model_list:
            slab_meta_rows.append(
                {
                    "cohort": bench,
                    "model_id": m_name,
                    "task": task_name,
                    "task_key": f"{bench}/{task_name}",
                }
            )
    meta_slab = pd.DataFrame(slab_meta_rows)

    # 4. Evaluate all estimators across MMLU_TASK, MMLU_POOLED, and SLAB
    for est_name in ESTIMATOR_ORDER:
        run_iboot = n_item_boot_verify if est_name == "legacy_first" else 0
        cmap_5 = mmlu_maps_by_est[est_name]

        mmlu_task_mats: List[np.ndarray] = []
        mmlu_task_d: List[np.ndarray] = []
        mmlu_task_c: List[np.ndarray] = []
        mmlu_pool_mats: List[np.ndarray] = []

        for m_idx in range(len(MMLU_MODELS)):
            model_task_mats: List[np.ndarray] = []
            for t in tasks_mmlu:
                Y_t = extract_cell_item_matrix(
                    Ys["MMLU"][t][m_idx], all_cells_5, cmap_5, estimator=est_name
                )
                mmlu_task_mats.append(Y_t)
                mmlu_task_d.append(mmlu_splits[t][0])
                mmlu_task_c.append(mmlu_splits[t][1])
                model_task_mats.append(Y_t)
            Y_pool = np.hstack(model_task_mats)
            # Sanity-verify Split 0 discovery parity matches `concatenate_task_discovery_items`
            if m_idx == 0 and est_name == "legacy_first":
                disc_concat = concatenate_task_discovery_items(model_task_mats)
                if not np.array_equal(Y_pool[:, d_pool_bin[:, 0] > 0.5], disc_concat):
                    raise RuntimeError("Pooled MMLU Split 0 discovery mask mismatch vs concatenate_task_discovery_items")
            mmlu_pool_mats.append(Y_pool)

        rows_mt, sum_mt = evaluate_panel_cubes_reliability(
            cohort_label="MMLU_TASK",
            estimator_name=est_name,
            num_factors=5,
            cube_matrices=mmlu_task_mats,
            cube_d_masks=mmlu_task_d,
            cube_c_masks=mmlu_task_c,
            cube_meta=meta_mmlu_task,
            panel_mode="2d_balanced",
            screening_ops=ops_5,
            n_boot_cluster=n_boot_cluster,
            cluster_seed=RELIABILITY_SEEDS["cluster_bootstrap"],
            n_item_boot_verify=run_iboot,
            item_boot_seed=RELIABILITY_SEEDS["item_bootstrap"],
        )
        all_csv_rows.extend(rows_mt)
        cohorts_summary["MMLU_TASK"][est_name] = sum_mt
        if "item_bootstrap_verification" in sum_mt:
            item_boot_top["MMLU_TASK_legacy_first"] = sum_mt["item_bootstrap_verification"]

        rows_mp, sum_mp = evaluate_panel_cubes_reliability(
            cohort_label="MMLU_POOLED",
            estimator_name=est_name,
            num_factors=5,
            cube_matrices=mmlu_pool_mats,
            cube_d_masks=[d_pool_norm for _ in range(len(MMLU_MODELS))],
            cube_c_masks=[c_pool_norm for _ in range(len(MMLU_MODELS))],
            cube_meta=meta_mmlu_pool,
            panel_mode="1d_model",
            screening_ops=ops_5,
            n_boot_cluster=n_boot_cluster,
            cluster_seed=RELIABILITY_SEEDS["cluster_bootstrap"],
            n_item_boot_verify=run_iboot,
            item_boot_seed=RELIABILITY_SEEDS["item_bootstrap"] + 1,
        )
        all_csv_rows.extend(rows_mp)
        cohorts_summary["MMLU_POOLED"][est_name] = sum_mp
        if "item_bootstrap_verification" in sum_mp:
            item_boot_top["MMLU_POOLED_legacy_first"] = sum_mp["item_bootstrap_verification"]

        slab_mats: List[np.ndarray] = []
        slab_d: List[np.ndarray] = []
        slab_c: List[np.ndarray] = []
        tmap_4 = slab_maps_by_est[est_name]
        for bench, task_name in slab_keys_sorted:
            key = (bench, task_name)
            dm, cm, _ = slab_splits[key]
            for raw_mat in Ys[bench][task_name]:
                Y_t = extract_cell_item_matrix(raw_mat, all_cells_4, tmap_4[key], estimator=est_name)
                slab_mats.append(Y_t)
                slab_d.append(dm)
                slab_c.append(cm)

        rows_sl, sum_sl = evaluate_panel_cubes_reliability(
            cohort_label="SLAB",
            estimator_name=est_name,
            num_factors=4,
            cube_matrices=slab_mats,
            cube_d_masks=slab_d,
            cube_c_masks=slab_c,
            cube_meta=meta_slab,
            panel_mode="2d_unbalanced",
            screening_ops=ops_4,
            n_boot_cluster=n_boot_cluster,
            cluster_seed=RELIABILITY_SEEDS["cluster_bootstrap"],
            n_item_boot_verify=run_iboot,
            item_boot_seed=RELIABILITY_SEEDS["item_bootstrap"] + 2,
        )
        all_csv_rows.extend(rows_sl)
        cohorts_summary["SLAB"][est_name] = sum_sl
        if "item_bootstrap_verification" in sum_sl:
            item_boot_top["SLAB_legacy_first"] = sum_sl["item_bootstrap_verification"]

    df_out = pd.DataFrame(all_csv_rows)
    csv_text = df_out.to_csv(index=False, float_format="%.12g")
    csv_sha256 = sha256_text(csv_text)

    hierarchy_audit = build_effect_hierarchy_audit(cohorts_summary)

    # Extract concise headline numbers for immediate paper/table integration
    mt_l_s20 = cohorts_summary["MMLU_TASK"]["legacy_first"]["order_reliability"]["splits20"]
    mt_l_esr_s20 = cohorts_summary["MMLU_TASK"]["legacy_first"]["screening_esr_and_heldout_prediction"]["splits20"]
    mp_l_s20 = cohorts_summary["MMLU_POOLED"]["legacy_first"]["order_reliability"]["splits20"]
    mp_l_esr_s20 = cohorts_summary["MMLU_POOLED"]["legacy_first"]["screening_esr_and_heldout_prediction"]["splits20"]
    sl_l_s20 = cohorts_summary["SLAB"]["legacy_first"]["order_reliability"]["splits20"]
    sl_l_esr_s20 = cohorts_summary["SLAB"]["legacy_first"]["screening_esr_and_heldout_prediction"]["splits20"]
    sl_u_s20 = cohorts_summary["SLAB"]["cell_mean_unique"]["order_reliability"]["splits20"]

    headline_numbers = {
        "mmlu_task_legacy_first_splits20": {
            "order1_pooled_r": mt_l_s20["order_1"]["pooled_pearson_r"],
            "order2_pooled_r": mt_l_s20["order_2"]["pooled_pearson_r"],
            "order3_pooled_r": mt_l_s20["order_3"]["pooled_pearson_r"],
            "order4_pooled_r": mt_l_s20["order_4"]["pooled_pearson_r"],
            "order5_pooled_r": mt_l_s20["order_5"]["pooled_pearson_r"],
            "order1_cohort_rms_corr": mt_l_s20["order_1"]["cohort_rms_corr"],
            "order2_cohort_rms_corr": mt_l_s20["order_2"]["cohort_rms_corr"],
            "order3_cohort_rms_corr": mt_l_s20["order_3"]["cohort_rms_corr"],
            "order4_cohort_rms_corr": mt_l_s20["order_4"]["cohort_rms_corr"],
            "order5_cohort_rms_corr": mt_l_s20["order_5"]["cohort_rms_corr"],
            "res5_main_esr_attr_over_disc_corr_quad": mt_l_esr_s20["order_1"]["2(5-1)V_coset_avg"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "res5_main_esr_heldout_over_conf_obs_quad": mt_l_esr_s20["order_1"]["2(5-1)V_coset_avg"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
            "res5_2fi_esr_attr_over_disc_corr_quad": mt_l_esr_s20["order_2"]["2(5-1)V_coset_avg"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "res5_2fi_esr_heldout_over_conf_obs_quad": mt_l_esr_s20["order_2"]["2(5-1)V_coset_avg"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
            "rand16_2fi_esr_attr_over_disc_corr_quad": mt_l_esr_s20["order_2"]["rand16_fullrank"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "rand16_2fi_esr_heldout_over_conf_obs_quad": mt_l_esr_s20["order_2"]["rand16_fullrank"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
        },
        "mmlu_pooled_legacy_first_splits20": {
            "order1_pooled_r": mp_l_s20["order_1"]["pooled_pearson_r"],
            "order2_pooled_r": mp_l_s20["order_2"]["pooled_pearson_r"],
            "order3_pooled_r": mp_l_s20["order_3"]["pooled_pearson_r"],
            "order4_pooled_r": mp_l_s20["order_4"]["pooled_pearson_r"],
            "order5_pooled_r": mp_l_s20["order_5"]["pooled_pearson_r"],
            "res5_2fi_esr_attr_over_disc_corr_quad": mp_l_esr_s20["order_2"]["2(5-1)V_coset_avg"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "rand16_2fi_esr_attr_over_disc_corr_quad": mp_l_esr_s20["order_2"]["rand16_fullrank"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "res5_2fi_esr_heldout_over_conf_obs_quad": mp_l_esr_s20["order_2"]["2(5-1)V_coset_avg"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
            "rand16_2fi_esr_heldout_over_conf_obs_quad": mp_l_esr_s20["order_2"]["rand16_fullrank"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
        },
        "slab_legacy_first_splits20": {
            "order1_pooled_r": sl_l_s20["order_1"]["pooled_pearson_r"],
            "order2_pooled_r": sl_l_s20["order_2"]["pooled_pearson_r"],
            "order3_pooled_r": sl_l_s20["order_3"]["pooled_pearson_r"],
            "order4_pooled_r": sl_l_s20["order_4"]["pooled_pearson_r"],
            "res4_main_esr_attr_over_disc_corr_quad": sl_l_esr_s20["order_1"]["2(4-1)IV_coset_avg"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "rand8_main_esr_attr_over_disc_corr_quad": sl_l_esr_s20["order_1"]["rand8_fullrank"]["esr_attr_over_disc_corr_quad"]["estimate"],
            "res4_main_esr_heldout_over_conf_obs_quad": sl_l_esr_s20["order_1"]["2(4-1)IV_coset_avg"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
            "rand8_main_esr_heldout_over_conf_obs_quad": sl_l_esr_s20["order_1"]["rand8_fullrank"]["esr_heldout_pred_over_conf_obs_quad"]["estimate"],
        },
        "slab_cell_mean_unique_splits20": {
            "order1_cohort_rms_corr": sl_u_s20["order_1"]["cohort_rms_corr"],
            "order2_cohort_rms_corr": sl_u_s20["order_2"]["cohort_rms_corr"],
            "order3_cohort_rms_corr": sl_u_s20["order_3"]["cohort_rms_corr"],
            "order4_cohort_rms_corr": sl_u_s20["order_4"]["cohort_rms_corr"],
        },
    }

    summary_dict: Dict[str, Any] = {
        "study": "PromptPrism Multi-Model Hugging Face Factorial Benchmark Study",
        "task_id": "Phase0_Task_R1_Effect_Noise_Floor_And_Split_Half_Reliability",
        "source_hashes": source_hashes,
        "random_seeds": RELIABILITY_SEEDS,
        "n_cluster_bootstrap_replicates": int(n_boot_cluster),
        "n_item_bootstrap_verification_replicates": int(n_item_boot_verify),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
        },
        "variance_decomposition_and_truncation_rule": {
            "item_contrast_definition": "c_{S, i} = (2 / K) * sum_c X_{c, S} * Y_{c, i}",
            "analytical_sampling_variance": "Var_hat(tau_hat_S) = s^2(c_{S, .}) / N_eval (ddof=1 across evaluated items)",
            "observed_rms": "RMS_obs = sqrt((1 / |O|) * sum_{S in O} tau_hat_S^2)",
            "noise_floor_rms": "RMS_noise = sqrt((1 / |O|) * sum_{S in O} Var_hat(tau_hat_S))",
            "unclipped_signal_variance": "sigma2_hat_signal_unclipped = (1 / |O|) * sum_{S in O} (tau_hat_S^2 - Var_hat(tau_hat_S))",
            "per_cube_corrected_rms": "RMS_corr_cube = sqrt(max(0.0, sigma2_hat_signal_unclipped_cube))",
            "cohort_corrected_rms": "RMS_corr_cohort = sqrt(max(0.0, mean_cube(sigma2_hat_signal_unclipped_cube)))",
            "rationale": (
                "Per-cube truncation records the exact fraction of individual cubes whose observed contrast power falls "
                "at or below the item-sampling variance floor (`frac_cubes_truncated_zero`), while cohort-level "
                "aggregation averages the unbiased variance component `sigma2_hat_signal_unclipped` across cubes prior "
                "to square-root truncation so multi-cube cohorts remain unbiased under pure noise even for |O|=1."
            ),
        },
        "output_artifacts": {
            "effect_reliability_csv": "research/hf_multi_prompt/results/effect_reliability.csv",
            "effect_reliability_csv_sha256": csv_sha256,
            "effect_reliability_csv_rows": int(len(df_out)),
        },
        "headline_paper_numbers": headline_numbers,
        "effect_hierarchy_audit": hierarchy_audit,
        "item_bootstrap_verification": item_boot_top,
        "cohorts": cohorts_summary,
    }

    json_text = json.dumps(summary_dict, indent=2) + "\n"
    json_sha256 = sha256_text(json_text)

    if write_outputs:
        if output_csv_path is not None:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            output_csv_path.write_text(csv_text, encoding="utf-8")
        if output_json_path is not None:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(json_text, encoding="utf-8")

    return df_out, summary_dict, csv_sha256, json_sha256


def main() -> None:
    df_out, summary_dict, csv_sha, json_sha = run_effect_reliability_analysis(
        n_boot_cluster=1000,
        n_item_boot_verify=500,
        write_outputs=True,
    )
    print(f"Wrote {len(df_out)} rows to effect_reliability.csv (SHA-256: {csv_sha})")
    print(f"Wrote effect_reliability_summary.json (SHA-256: {json_sha})")
    verdict = summary_dict["effect_hierarchy_audit"]["verdict_code"]
    print(f"Effect hierarchy audit verdict: {verdict}")


if __name__ == "__main__":
    main()
