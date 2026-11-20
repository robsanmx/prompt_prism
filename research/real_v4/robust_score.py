"""Simultaneous worst-environment lower-confidence-bound scoring for robust validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

TIE_TOL: float = 1e-12


@dataclass(frozen=True)
class CandidateRobustScore:
    run_id: int
    robust_lower_bound: float
    worst_env_mean: float
    worst_env_id: str
    per_env_means: Dict[str, float]
    per_env_se: Dict[str, float]


def _orient_scores(arr: np.ndarray, higher_is_better: bool) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    return vals if higher_is_better else (1.0 - vals)


def compute_candidate_robust_scores(
    validation_scores_by_env: Mapping[str, Mapping[int, np.ndarray]],
    *,
    group_ids: Optional[Sequence[str]] = None,
    higher_is_better: bool = True,
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    seed: int = 20260401,
    partition_tag: str = "validation",
) -> Dict[int, CandidateRobustScore]:
    """Compute simultaneous lower confidence bounds on worst-environment validation score.

    Enforces confirmation quarantine (`partition_tag` cannot be confirmation) and paired
    group/item bootstrap across all validation environments.
    """
    if "confirm" in partition_tag.lower():
        raise PermissionError(
            f"Confirmation leakage violation: robust validation refused partition_tag={partition_tag!r}"
        )
    if not validation_scores_by_env:
        raise ValueError("validation_scores_by_env cannot be empty")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

    env_ids = sorted(validation_scores_by_env.keys())
    cand_ids = sorted(validation_scores_by_env[env_ids[0]].keys())
    if not cand_ids:
        raise ValueError("No candidate run_ids provided in validation_scores_by_env")

    first_arr = np.asarray(validation_scores_by_env[env_ids[0]][cand_ids[0]], dtype=float)
    n_items = len(first_arr)
    if n_items == 0:
        raise ValueError("Validation score arrays cannot be empty")

    if group_ids is None:
        g_tuple = tuple(f"item_{i:04d}" for i in range(n_items))
    else:
        g_tuple = tuple(group_ids)
        if len(g_tuple) != n_items:
            raise ValueError(f"group_ids length {len(g_tuple)} != n_items {n_items}")

    unique_groups = []
    g_map: Dict[str, int] = {}
    for g in g_tuple:
        if g not in g_map:
            g_map[g] = len(unique_groups)
            unique_groups.append(g)
    n_groups = len(unique_groups)
    g_idx = np.array([g_map[g] for g in g_tuple], dtype=int)
    g_counts = np.bincount(g_idx, minlength=n_groups).astype(float)

    # Validate shapes and precompute group sums
    oriented_group_sums: Dict[Tuple[int, str], np.ndarray] = {}
    point_means: Dict[Tuple[int, str], float] = {}
    for env_id in env_ids:
        env_map = validation_scores_by_env[env_id]
        if sorted(env_map.keys()) != cand_ids:
            raise ValueError(f"Candidate ID mismatch in validation env {env_id!r}")
        for cid in cand_ids:
            raw = np.asarray(env_map[cid], dtype=float)
            if raw.ndim != 1 or len(raw) != n_items:
                raise ValueError(f"Expected 1D validation array of length {n_items} for run {cid} in {env_id}")
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"Non-finite validation score for run {cid} in {env_id}")
            ori = _orient_scores(raw, higher_is_better=higher_is_better)
            oriented_group_sums[(cid, env_id)] = np.bincount(g_idx, weights=ori, minlength=n_groups)
            point_means[(cid, env_id)] = float(np.mean(ori))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))
    denom = g_counts[draws].sum(axis=1)  # shape (n_bootstrap,)

    results: Dict[int, CandidateRobustScore] = {}
    for cid in cand_ids:
        boot_mat = np.column_stack(
            [oriented_group_sums[(cid, env_id)][draws].sum(axis=1) / denom for env_id in env_ids]
        )  # (n_bootstrap, n_envs)
        mu_vec = np.array([point_means[(cid, env_id)] for env_id in env_ids], dtype=float)
        se_vec = np.std(boot_mat, axis=0, ddof=1) if n_groups > 1 else np.zeros(len(env_ids), dtype=float)

        worst_idx = int(np.argmin(mu_vec))
        worst_env_id = env_ids[worst_idx]
        worst_mean = float(mu_vec[worst_idx])

        if np.all(se_vec < 1e-15):
            lcb_sim = worst_mean
        else:
            se_safe = np.maximum(se_vec, 1e-12)
            pivots = (boot_mat - mu_vec[None, :]) / se_safe[None, :]
            t_min_boot = np.min(pivots, axis=1)
            q_low = float(np.percentile(t_min_boot, 100.0 * alpha))
            eff_se = np.where(se_vec < 1e-15, 0.0, se_vec)
            lcb_sim = float(np.min(mu_vec + q_low * eff_se))

        results[cid] = CandidateRobustScore(
            run_id=cid,
            robust_lower_bound=lcb_sim,
            worst_env_mean=worst_mean,
            worst_env_id=worst_env_id,
            per_env_means={env_ids[k]: float(mu_vec[k]) for k in range(len(env_ids))},
            per_env_se={env_ids[k]: float(se_vec[k]) for k in range(len(env_ids))},
        )
    return results


def select_robust_winner(
    robust_scores: Mapping[int, CandidateRobustScore],
    *,
    tol: float = TIE_TOL,
) -> CandidateRobustScore:
    """Select the candidate maximizing simultaneous lower bound; ties pick lowest `run_id`."""
    if not robust_scores:
        raise ValueError("robust_scores cannot be empty")
    max_lcb = max(score.robust_lower_bound for score in robust_scores.values())
    tied = [
        score
        for score in robust_scores.values()
        if score.robust_lower_bound >= max_lcb - tol
    ]
    tied.sort(key=lambda s: s.run_id)
    return tied[0]
