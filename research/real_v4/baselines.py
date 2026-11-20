"""Equal-budget factor-balanced Random and sequential Bayesian Optimization baselines (Task 5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.real_v4.budget import BudgetLedger
from research.real_v4.robust_score import (
    TIE_TOL,
    CandidateRobustScore,
    _orient_scores,
    compute_candidate_robust_scores,
    select_robust_winner,
)
from research.real_v4.schemas import CandidateProvenance
from research.real_v4.selection import (
    FACTOR_IDS,
    full_cube_configs,
    run_id_to_levels,
)

BALANCED_RANDOM_VERSION: str = "balanced_random_v4.0.0"
BAYESIAN_OPT_VERSION: str = "bayesian_opt_gp_ucb_v4.0.0"


@dataclass(frozen=True)
class BaselineTrajectoryResult:
    method_id: str
    trajectory_id: str
    screen_run_ids: Tuple[int, ...]
    candidates: Tuple[CandidateProvenance, ...]
    selected_candidate: CandidateProvenance
    selected_robust_score: CandidateRobustScore


def generate_frozen_balanced_random_screens(
    *,
    num_designs: int = 20,
    subset_size: int = 16,
    seed: int = 20261101,
) -> Tuple[Tuple[int, ...], ...]:
    """Pre-freeze factor-balanced, full-rank 16-run screening designs without outcome access."""
    if subset_size % 2 != 0:
        raise ValueError("subset_size must be even for exact factor balance")
    cube = full_cube_configs()
    x_pm = np.array([[2 * cube[r][f] - 1 for f in FACTOR_IDS] for r in range(64)], dtype=float)
    d_main = np.column_stack([np.ones(64), x_pm])  # (64, 7)

    rng = np.random.default_rng(seed)
    accepted: List[Tuple[int, ...]] = []
    seen = set()
    attempts = 0
    while len(accepted) < num_designs:
        attempts += 1
        if attempts > 500_000:
            raise RuntimeError("Could not sample requested number of full-rank balanced random designs")
        cand = tuple(sorted(int(r) for r in rng.choice(64, size=subset_size, replace=False)))
        if cand in seen:
            continue
        if not np.all(x_pm[list(cand)].sum(axis=0) == 0.0):
            continue
        if int(np.linalg.matrix_rank(d_main[list(cand)])) != 7:
            continue
        seen.add(cand)
        accepted.append(cand)
    return tuple(accepted)


def _generate_balanced_initial_4_runs(seed: int) -> Tuple[int, ...]:
    """Generate 4 distinct full-cube runs with exact 2/2 factor balance across A..F."""
    cube = full_cube_configs()
    x_pm = np.array([[2 * cube[r][f] - 1 for f in FACTOR_IDS] for r in range(64)], dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(100_000):
        cand = tuple(sorted(int(r) for r in rng.choice(64, size=4, replace=False)))
        if np.all(x_pm[list(cand)].sum(axis=0) == 0.0):
            # Also ensure pairwise Hamming diversity (min pairwise distance >= 2)
            sub = x_pm[list(cand)]
            dists = [
                0.5 * np.sum(np.abs(sub[i] - sub[j]))
                for i in range(4)
                for j in range(i + 1, 4)
            ]
            if min(dists) >= 2.0:
                return cand
    raise RuntimeError("Could not generate balanced 4-run initial design")


def _hamming_rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 3.0, sigma_f: float = 0.15) -> np.ndarray:
    """Isotropic Hamming exponential kernel over binary {0, 1}^6 vectors."""
    # x1: (n1, 6), x2: (n2, 6)
    hamming = np.sum(np.abs(x1[:, None, :] - x2[None, :, :]), axis=-1)
    return (sigma_f ** 2) * np.exp(-hamming / length_scale)


class BalancedRandomTwoStageBaseline:
    """Equal-budget two-stage factor-balanced random baseline retaining all 20 frozen designs."""

    method_id: str = "balanced_random"
    algorithm_version: str = BALANCED_RANDOM_VERSION
    is_primary_comparator: bool = True

    def __init__(
        self,
        *,
        num_designs: int = 20,
        design_seed: int = 20261101,
        max_candidates: int = 4,
        alpha_validation: float = 0.05,
        n_bootstrap_validation: int = 500,
    ) -> None:
        self.num_designs = int(num_designs)
        self.design_seed = int(design_seed)
        self.max_candidates = int(max_candidates)
        self.alpha_validation = float(alpha_validation)
        self.n_bootstrap_validation = int(n_bootstrap_validation)
        self.frozen_designs: Tuple[Tuple[int, ...], ...] = generate_frozen_balanced_random_screens(
            num_designs=self.num_designs,
            subset_size=16,
            seed=self.design_seed,
        )

    def run_all_trajectories(
        self,
        *,
        cell_id: str,
        screen_evaluator: Callable[[int], np.ndarray],
        validation_evaluator: Callable[[int, str], np.ndarray],
        screen_item_ids: Sequence[str],
        validation_item_ids: Sequence[str],
        validation_env_ids: Sequence[str],
        validation_group_ids: Optional[Sequence[str]] = None,
        higher_is_better: bool = True,
        ledger: Optional[BudgetLedger] = None,
        validation_seed_base: int = 20260500,
    ) -> Tuple[BaselineTrajectoryResult, ...]:
        """Execute all frozen random designs under strict budget accounting with zero cherry-picking."""
        results: List[BaselineTrajectoryResult] = []
        for d_idx, screen_runs in enumerate(self.frozen_designs):
            traj_id = f"design_{d_idx:02d}"
            screen_means_ori: List[Tuple[float, float, int]] = []
            for rid in screen_runs:
                raw_scores = np.asarray(screen_evaluator(rid), dtype=float)
                if ledger is not None:
                    ledger.record_screen_batch(
                        cell_id=cell_id,
                        method_id=self.method_id,
                        trajectory_id=traj_id,
                        run_id=rid,
                        item_ids=screen_item_ids,
                        repeat_idx=0,
                        partition_tag="screen",
                    )
                ori = _orient_scores(raw_scores, higher_is_better=higher_is_better)
                screen_means_ori.append((float(np.mean(ori)), float(np.mean(raw_scores)), rid))

            # Deterministic top-4 selection: highest oriented score, lowest run_id on ties
            ranked = sorted(
                screen_means_ori,
                key=lambda item: (-round(item[0] / TIE_TOL), item[2]),
            )
            top_k = ranked[: self.max_candidates]
            candidates = tuple(
                CandidateProvenance(
                    run_id=rid,
                    factor_levels=run_id_to_levels(rid),
                    source_rule="random_screen_top_k",
                    screen_score=float(np.clip(raw_mu, 0.0, 1.0)),
                    rank_in_screen=rank_idx,
                )
                for rank_idx, (_ori_mu, raw_mu, rid) in enumerate(top_k, start=1)
            )

            val_scores: Dict[str, Dict[int, np.ndarray]] = {e: {} for e in validation_env_ids}
            for cand in candidates:
                for env_id in validation_env_ids:
                    arr = np.asarray(validation_evaluator(cand.run_id, env_id), dtype=float)
                    if ledger is not None:
                        ledger.record_validation_batch(
                            cell_id=cell_id,
                            method_id=self.method_id,
                            trajectory_id=traj_id,
                            run_id=cand.run_id,
                            env_id=env_id,
                            item_ids=validation_item_ids,
                            repeat_idx=0,
                            partition_tag="validation",
                        )
                    val_scores[env_id][cand.run_id] = arr

            rob_map = compute_candidate_robust_scores(
                val_scores,
                group_ids=validation_group_ids,
                higher_is_better=higher_is_better,
                alpha=self.alpha_validation,
                n_bootstrap=self.n_bootstrap_validation,
                seed=validation_seed_base + d_idx,
                partition_tag="validation",
            )
            winner_rob = select_robust_winner(rob_map)
            cand_by_id = {c.run_id: c for c in candidates}
            results.append(
                BaselineTrajectoryResult(
                    method_id=self.method_id,
                    trajectory_id=traj_id,
                    screen_run_ids=screen_runs,
                    candidates=candidates,
                    selected_candidate=cand_by_id[winner_rob.run_id],
                    selected_robust_score=winner_rob,
                )
            )
        return tuple(results)


class SequentialBayesianOptBaseline:
    """Equal-budget sequential GP-UCB Bayesian optimization baseline over {0, 1}^6."""

    method_id: str = "bayesian_opt"
    algorithm_version: str = BAYESIAN_OPT_VERSION
    is_primary_comparator: bool = True

    def __init__(
        self,
        *,
        num_trajectories: int = 20,
        base_seed: int = 20261201,
        n_initial: int = 4,
        n_total_screen: int = 16,
        max_candidates: int = 4,
        length_scale: float = 3.0,
        sigma_f: float = 0.15,
        ucb_beta: float = 1.96,
        alpha_validation: float = 0.05,
        n_bootstrap_validation: int = 500,
    ) -> None:
        if n_initial < 2 or n_initial >= n_total_screen:
            raise ValueError("Require 2 <= n_initial < n_total_screen")
        self.num_trajectories = int(num_trajectories)
        self.base_seed = int(base_seed)
        self.n_initial = int(n_initial)
        self.n_total_screen = int(n_total_screen)
        self.max_candidates = int(max_candidates)
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.ucb_beta = float(ucb_beta)
        self.alpha_validation = float(alpha_validation)
        self.n_bootstrap_validation = int(n_bootstrap_validation)

    def _gp_posterior_all_64(
        self,
        x_all_01: np.ndarray,
        visited_ids: Sequence[int],
        y_ori_visited: np.ndarray,
        noise_var: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_obs = x_all_01[list(visited_ids)]
        y_center = float(np.mean(y_ori_visited))
        y_centered = y_ori_visited - y_center

        k_oo = _hamming_rbf_kernel(x_obs, x_obs, self.length_scale, self.sigma_f) + (
            max(noise_var, 1e-5) * np.eye(len(visited_ids))
        )
        k_ao = _hamming_rbf_kernel(x_all_01, x_obs, self.length_scale, self.sigma_f)
        alpha_vec = np.linalg.solve(k_oo, y_centered)
        post_mean = y_center + (k_ao @ alpha_vec)

        v_mat = np.linalg.solve(k_oo, k_ao.T)  # (t, 64)
        prior_diag = self.sigma_f ** 2
        post_var = np.maximum(0.0, prior_diag - np.sum(k_ao.T * v_mat, axis=0))
        return post_mean, np.sqrt(post_var)

    def run_all_trajectories(
        self,
        *,
        cell_id: str,
        screen_evaluator: Callable[[int], np.ndarray],
        validation_evaluator: Callable[[int, str], np.ndarray],
        screen_item_ids: Sequence[str],
        validation_item_ids: Sequence[str],
        validation_env_ids: Sequence[str],
        validation_group_ids: Optional[Sequence[str]] = None,
        higher_is_better: bool = True,
        ledger: Optional[BudgetLedger] = None,
        validation_seed_base: int = 20260600,
    ) -> Tuple[BaselineTrajectoryResult, ...]:
        """Run all 20 sequential GP-UCB trajectories strictly within the frozen operational budget."""
        cube = full_cube_configs()
        x_all_01 = np.array([[cube[r][f] for f in FACTOR_IDS] for r in range(64)], dtype=float)
        results: List[BaselineTrajectoryResult] = []

        for t_idx in range(self.num_trajectories):
            traj_id = f"bo_seed_{t_idx:02d}"
            init_runs = list(_generate_balanced_initial_4_runs(self.base_seed + t_idx))
            visited_ids: List[int] = []
            visited_ori_means: List[float] = []
            visited_raw_means: Dict[int, float] = {}
            visited_item_vars: List[float] = []

            def _observe_run(rid: int) -> None:
                raw_arr = np.asarray(screen_evaluator(rid), dtype=float)
                if ledger is not None:
                    ledger.record_screen_batch(
                        cell_id=cell_id,
                        method_id=self.method_id,
                        trajectory_id=traj_id,
                        run_id=rid,
                        item_ids=screen_item_ids,
                        repeat_idx=0,
                        partition_tag="screen",
                    )
                ori_arr = _orient_scores(raw_arr, higher_is_better=higher_is_better)
                visited_ids.append(rid)
                visited_ori_means.append(float(np.mean(ori_arr)))
                visited_raw_means[rid] = float(np.mean(raw_arr))
                n_it = len(ori_arr)
                sem2 = float(np.var(ori_arr, ddof=1) / n_it) if n_it > 1 else 1e-4
                visited_item_vars.append(max(sem2, 1e-5))

            for rid in init_runs:
                _observe_run(rid)

            while len(visited_ids) < self.n_total_screen:
                noise_var = float(np.mean(visited_item_vars))
                post_mean, post_std = self._gp_posterior_all_64(
                    x_all_01,
                    visited_ids,
                    np.array(visited_ori_means, dtype=float),
                    noise_var=noise_var,
                )
                ucb = post_mean + self.ucb_beta * post_std
                unvisited = [r for r in range(64) if r not in visited_ids]
                max_ucb = max(float(ucb[r]) for r in unvisited)
                tied_next = [r for r in unvisited if float(ucb[r]) >= max_ucb - TIE_TOL]
                next_run = min(tied_next)
                _observe_run(next_run)

            # Final GP posterior over all 64 runs to include posterior-mean optimum if distinct from top-3 observed
            final_post_mean, _ = self._gp_posterior_all_64(
                x_all_01,
                visited_ids,
                np.array(visited_ori_means, dtype=float),
                noise_var=float(np.mean(visited_item_vars)),
            )
            ranked_obs = sorted(
                zip(visited_ori_means, visited_ids),
                key=lambda pair: (-round(pair[0] / TIE_TOL), pair[1]),
            )
            chosen_cands: List[Tuple[int, str]] = []
            seen_cands = set()
            for _mu, rid in ranked_obs[: max(1, self.max_candidates - 1)]:
                seen_cands.add(rid)
                chosen_cands.append((rid, "bo_observed_top"))

            max_pm = float(np.max(final_post_mean))
            pm_best = min(r for r in range(64) if float(final_post_mean[r]) >= max_pm - TIE_TOL)
            if pm_best not in seen_cands and len(chosen_cands) < self.max_candidates:
                seen_cands.add(pm_best)
                chosen_cands.append((pm_best, "bo_posterior_mean_opt"))

            for _mu, rid in ranked_obs:
                if len(chosen_cands) >= self.max_candidates:
                    break
                if rid not in seen_cands:
                    seen_cands.add(rid)
                    chosen_cands.append((rid, "bo_observed_runner_up"))

            candidates = tuple(
                CandidateProvenance(
                    run_id=rid,
                    factor_levels=run_id_to_levels(rid),
                    source_rule=rule,
                    screen_score=float(
                        np.clip(
                            visited_raw_means.get(
                                rid,
                                float(final_post_mean[rid]) if higher_is_better else float(1.0 - final_post_mean[rid]),
                            ),
                            0.0,
                            1.0,
                        )
                    ),
                    rank_in_screen=idx,
                )
                for idx, (rid, rule) in enumerate(chosen_cands, start=1)
            )

            val_scores: Dict[str, Dict[int, np.ndarray]] = {e: {} for e in validation_env_ids}
            for cand in candidates:
                for env_id in validation_env_ids:
                    arr = np.asarray(validation_evaluator(cand.run_id, env_id), dtype=float)
                    if ledger is not None:
                        ledger.record_validation_batch(
                            cell_id=cell_id,
                            method_id=self.method_id,
                            trajectory_id=traj_id,
                            run_id=cand.run_id,
                            env_id=env_id,
                            item_ids=validation_item_ids,
                            repeat_idx=0,
                            partition_tag="validation",
                        )
                    val_scores[env_id][cand.run_id] = arr

            rob_map = compute_candidate_robust_scores(
                val_scores,
                group_ids=validation_group_ids,
                higher_is_better=higher_is_better,
                alpha=self.alpha_validation,
                n_bootstrap=self.n_bootstrap_validation,
                seed=validation_seed_base + t_idx,
                partition_tag="validation",
            )
            winner_rob = select_robust_winner(rob_map)
            cand_by_id = {c.run_id: c for c in candidates}
            results.append(
                BaselineTrajectoryResult(
                    method_id=self.method_id,
                    trajectory_id=traj_id,
                    screen_run_ids=tuple(visited_ids),
                    candidates=candidates,
                    selected_candidate=cand_by_id[winner_rob.run_id],
                    selected_robust_score=winner_rob,
                )
            )
        return tuple(results)


def ofat_reference_run_ids() -> Tuple[int, ...]:
    """Secondary lower-budget 7-run OFAT reference (run 0 + single-factor flips)."""
    return (0, 32, 16, 8, 4, 2, 1)


def untuned_reference_run_id() -> int:
    """Secondary untuned reference prompt (all factors at level 0)."""
    return 0
