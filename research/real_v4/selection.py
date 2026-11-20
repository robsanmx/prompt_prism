"""Deterministic four-candidate PromptPrism selector and Resolution-IV screen geometry."""

from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.real_v4.robust_score import (
    TIE_TOL,
    CandidateRobustScore,
    _orient_scores,
    compute_candidate_robust_scores,
    select_robust_winner,
)
from research.real_v4.schemas import CandidateProvenance

FACTOR_IDS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
PROMPTPRISM_SELECTOR_VERSION: str = "promptprism_robust_v4.0.0"


def full_cube_configs() -> Dict[int, Dict[str, int]]:
    """Map deterministic run_id 0..63 to binary {0, 1} levels over A..F."""
    return {
        run_id: dict(zip(FACTOR_IDS, bits))
        for run_id, bits in enumerate(itertools.product((0, 1), repeat=len(FACTOR_IDS)))
    }


def run_id_to_levels(run_id: int) -> Dict[str, int]:
    if not (0 <= run_id < 64):
        raise ValueError(f"run_id must be in 0..63, got {run_id}")
    return {
        factor: (run_id >> (5 - idx)) & 1
        for idx, factor in enumerate(FACTOR_IDS)
    }


def levels_to_run_id(levels: Mapping[str, int]) -> int:
    run_id = 0
    for idx, factor in enumerate(FACTOR_IDS):
        bit = int(levels[factor])
        if bit not in (0, 1):
            raise ValueError(f"Factor {factor!r} bit must be 0 or 1, got {bit}")
        run_id |= bit << (5 - idx)
    return run_id


def resolution_iv_screen_run_ids() -> Tuple[int, ...]:
    """Return the 16 full-cube run IDs for 2^(6-2)_IV with E=ABC and F=BCD in {-1, +1} coding."""
    selected: List[int] = []
    for run_id, levels in full_cube_configs().items():
        pm = {f: (2 * v - 1) for f, v in levels.items()}
        if pm["E"] == pm["A"] * pm["B"] * pm["C"] and pm["F"] == pm["B"] * pm["C"] * pm["D"]:
            selected.append(run_id)
    assert len(selected) == 16
    return tuple(selected)


def _build_res4_orthogonal_2fi_basis() -> Tuple[np.ndarray, np.ndarray]:
    """Build full-cube (64, 14) and screen (16, 14) orthogonal main + 7-chain 2FI matrices.

    Under I = ABCE = BCDF = ADEF, the 15 two-factor products partition into 7 orthogonal
    alias chains on the 16-run screen:
      AB (=CE), AC (=BE), AD (=EF), AE (=BC=DF), AF (=DE), BD (=CF), BF (=CD).
    Choosing one canonical representative column per chain yields 1 + 6 + 7 = 14 strictly
    orthogonal columns on the 16-run fraction. To avoid arbitrary representative bias when
    extrapolating across all 64 cube corners, we average predictions equally across each
    alias chain's constituent 2FI products!
    """
    cube = full_cube_configs()
    x_pm_full = np.array(
        [[2 * cube[r][f] - 1 for f in FACTOR_IDS] for r in range(64)],
        dtype=float,
    )  # (64, 6)
    screen_ids = np.array(resolution_iv_screen_run_ids(), dtype=int)

    alias_chains: Tuple[Tuple[Tuple[int, int], ...], ...] = (
        ((0, 1), (2, 4)),          # AB = CE
        ((0, 2), (1, 4)),          # AC = BE
        ((0, 3), (4, 5)),          # AD = EF
        ((0, 4), (1, 2), (3, 5)),  # AE = BC = DF
        ((0, 5), (3, 4)),          # AF = DE
        ((1, 3), (2, 5)),          # BD = CF
        ((1, 5), (2, 3)),          # BF = CD
    )

    rep_cols_screen = [
        x_pm_full[screen_ids, chain[0][0]] * x_pm_full[screen_ids, chain[0][1]]
        for chain in alias_chains
    ]
    d_screen = np.column_stack([np.ones(16), x_pm_full[screen_ids], *rep_cols_screen])  # (16, 14)

    avg_chain_cols_full = [
        np.mean(
            np.column_stack([x_pm_full[:, i] * x_pm_full[:, j] for (i, j) in chain]),
            axis=1,
        )
        for chain in alias_chains
    ]
    d_full_symmetric = np.column_stack([np.ones(64), x_pm_full, *avg_chain_cols_full])  # (64, 14)
    return d_screen, d_full_symmetric


class PromptPrismRobustSelector:
    """Two-stage Resolution-IV four-candidate robust PromptPrism selector."""

    algorithm_version: str = PROMPTPRISM_SELECTOR_VERSION

    def __init__(
        self,
        *,
        practical_threshold: float = 0.005,
        alpha_screen: float = 0.10,
        alpha_validation: float = 0.05,
        n_bootstrap_screen: int = 500,
        n_bootstrap_validation: int = 500,
        max_candidates: int = 4,
        pad_to_max: bool = True,
    ) -> None:
        if not (1 <= max_candidates <= 4):
            raise ValueError(f"max_candidates must be in 1..4, got {max_candidates}")
        self.practical_threshold = float(practical_threshold)
        self.alpha_screen = float(alpha_screen)
        self.alpha_validation = float(alpha_validation)
        self.n_bootstrap_screen = int(n_bootstrap_screen)
        self.n_bootstrap_validation = int(n_bootstrap_validation)
        self.max_candidates = int(max_candidates)
        self.pad_to_max = bool(pad_to_max)

    @staticmethod
    def screen_run_ids() -> Tuple[int, ...]:
        return resolution_iv_screen_run_ids()

    def generate_candidates(
        self,
        screen_scores_by_run: Mapping[int, np.ndarray],
        *,
        group_ids: Optional[Sequence[str]] = None,
        higher_is_better: bool = True,
        seed: int = 20260402,
        partition_tag: str = "screen",
    ) -> Tuple[CandidateProvenance, ...]:
        """Deterministically construct up to 4 unique full-cube candidates from screen-only data."""
        if "confirm" in partition_tag.lower() or "val" in partition_tag.lower():
            raise PermissionError(
                f"Screen candidate generation only accepts screen partition data; got {partition_tag!r}"
            )

        expected_runs = self.screen_run_ids()
        if tuple(sorted(screen_scores_by_run.keys())) != expected_runs:
            raise ValueError(
                f"screen_scores_by_run keys {sorted(screen_scores_by_run.keys())} "
                f"do not match registered Resolution-IV runs {list(expected_runs)}"
            )

        n_items = len(np.asarray(screen_scores_by_run[expected_runs[0]]))
        if n_items == 0:
            raise ValueError("Screen item arrays cannot be empty")

        raw_mat = np.vstack([np.asarray(screen_scores_by_run[r], dtype=float) for r in expected_runs])
        if raw_mat.shape != (16, n_items) or not np.all(np.isfinite(raw_mat)):
            raise ValueError("Invalid or non-finite screen matrix")

        ori_mat = _orient_scores(raw_mat, higher_is_better=higher_is_better)  # (16, n_items)
        mean_ori_by_pos = np.mean(ori_mat, axis=1)  # (16,)
        mean_raw_by_pos = np.mean(raw_mat, axis=1)

        # Rank the 16 screen runs deterministically: highest oriented score, lowest run_id on ties
        ranked_positions = sorted(
            range(16),
            key=lambda pos: (-round(float(mean_ori_by_pos[pos]) / TIE_TOL) * TIE_TOL, expected_runs[pos]),
        )
        # Exact tie-aware sort: group anything within TIE_TOL of local leaders cleanly
        ranked_positions = sorted(
            range(16),
            key=lambda pos: (-float(mean_ori_by_pos[pos]), expected_runs[pos]),
        )
        # Enforce TIE_TOL tie rule explicitly for screen_best
        max_ori = float(np.max(mean_ori_by_pos))
        best_tied_pos = [
            pos for pos in range(16) if float(mean_ori_by_pos[pos]) >= max_ori - TIE_TOL
        ]
        best_pos = min(best_tied_pos, key=lambda pos: expected_runs[pos])
        screen_best_run_id = expected_runs[best_pos]

        # Compute exact orthogonal main-effect contrasts: (6, n_items)
        x_pm_screen = np.array(
            [[2 * run_id_to_levels(r)[f] - 1 for f in FACTOR_IDS] for r in expected_runs],
            dtype=float,
        )  # (16, 6)
        item_contrasts = (x_pm_screen.T @ ori_mat) / 8.0  # (6, n_items), since K_screen/2 = 8
        tau_hat = np.mean(item_contrasts, axis=1)  # (6,)

        # Group-aware bootstrap lower bounds on tau_hat for conservative recipe
        if group_ids is None:
            g_tuple = tuple(f"item_{i:04d}" for i in range(n_items))
        else:
            g_tuple = tuple(group_ids)
            if len(g_tuple) != n_items:
                raise ValueError("group_ids length mismatch in generate_candidates")

        uniq_g: List[str] = []
        g_map: Dict[str, int] = {}
        for g in g_tuple:
            if g not in g_map:
                g_map[g] = len(uniq_g)
                uniq_g.append(g)
        n_groups = len(uniq_g)
        g_idx = np.array([g_map[g] for g in g_tuple], dtype=int)
        g_counts = np.bincount(g_idx, minlength=n_groups).astype(float)
        contrast_group_sums = np.vstack(
            [np.bincount(g_idx, weights=item_contrasts[j], minlength=n_groups) for j in range(6)]
        )  # (6, n_groups)

        if n_groups > 1:
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, n_groups, size=(self.n_bootstrap_screen, n_groups))
            denom = g_counts[draws].sum(axis=1)  # (B,)
            tau_boot = np.column_stack(
                [contrast_group_sums[j][draws].sum(axis=1) / denom for j in range(6)]
            )  # (B, 6)
            tau_lcb = np.percentile(tau_boot, 100.0 * self.alpha_screen, axis=0)
        else:
            tau_lcb = tau_hat.copy()

        # Rule 1: sign_composed (zero contrast maps to 0 / -1)
        sign_levels = {
            FACTOR_IDS[j]: (1 if float(tau_hat[j]) > TIE_TOL else 0)
            for j in range(6)
        }
        sign_run_id = levels_to_run_id(sign_levels)

        # Rule 3: conservative_thresholded (activate to 1 iff LCB > practical_threshold)
        cons_levels = {
            FACTOR_IDS[j]: (1 if float(tau_lcb[j]) > self.practical_threshold + TIE_TOL else 0)
            for j in range(6)
        }
        cons_run_id = levels_to_run_id(cons_levels)

        # Rule 4: weakest_factor_alternative (flip weakest active factor in sign_levels, or weakest overall if all 0)
        active_indices = [j for j in range(6) if sign_levels[FACTOR_IDS[j]] == 1]
        search_indices = active_indices if active_indices else list(range(6))
        weakest_j = min(
            search_indices,
            key=lambda j: (round(abs(float(tau_hat[j])) / TIE_TOL), j),
        )
        weak_levels = dict(sign_levels)
        weak_levels[FACTOR_IDS[weakest_j]] = 1 - weak_levels[FACTOR_IDS[weakest_j]]
        weak_run_id = levels_to_run_id(weak_levels)

        # Orthogonal main + aliased-2FI prediction across all 64 runs (for screen_score proxy on unobserved runs & fallback)
        d_screen, d_full = _build_res4_orthogonal_2fi_basis()
        beta_14 = np.linalg.pinv(d_screen) @ mean_ori_by_pos
        pred_ori_64 = d_full @ beta_14  # (64,)
        pred_raw_64 = pred_ori_64 if higher_is_better else (1.0 - pred_ori_64)
        pred_raw_clipped = np.clip(pred_raw_64, 0.0, 1.0)

        observed_raw_map = {expected_runs[pos]: float(np.clip(mean_raw_by_pos[pos], 0.0, 1.0)) for pos in range(16)}

        primary_proposals: List[Tuple[int, str]] = [
            (sign_run_id, "sign_composed"),
            (screen_best_run_id, "screen_best"),
            (cons_run_id, "conservative_thresholded"),
            (weak_run_id, "weakest_factor_alternative"),
        ]

        chosen: List[Tuple[int, str]] = []
        seen_ids = set()
        for rid, rule_name in primary_proposals:
            if rid not in seen_ids and len(chosen) < self.max_candidates:
                seen_ids.add(rid)
                chosen.append((rid, rule_name))

        if self.pad_to_max and len(chosen) < self.max_candidates:
            # Fallback A: symmetric aliased-2FI full-cube optimum (lowest run_id on ties)
            max_2fi = float(np.max(pred_ori_64))
            best_2fi_runs = [r for r in range(64) if float(pred_ori_64[r]) >= max_2fi - TIE_TOL]
            opt_2fi_run = min(best_2fi_runs)
            if opt_2fi_run not in seen_ids:
                seen_ids.add(opt_2fi_run)
                chosen.append((opt_2fi_run, "aliased_2fi_fallback"))

            # Fallback B: deterministic screen runner-ups (highest oriented mean, lowest run_id on ties)
            runner_up_order = sorted(
                range(16),
                key=lambda pos: (-round(float(mean_ori_by_pos[pos]) / TIE_TOL), expected_runs[pos]),
            )
            for pos in runner_up_order:
                if len(chosen) >= self.max_candidates:
                    break
                rid = expected_runs[pos]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    chosen.append((rid, "screen_runner_up"))

        provenances: List[CandidateProvenance] = []
        for idx, (rid, rule_name) in enumerate(chosen, start=1):
            s_score = observed_raw_map.get(rid, float(pred_raw_clipped[rid]))
            provenances.append(
                CandidateProvenance(
                    run_id=rid,
                    factor_levels=run_id_to_levels(rid),
                    source_rule=rule_name,
                    screen_score=s_score,
                    rank_in_screen=idx,
                )
            )
        return tuple(provenances)

    def select_final_prompt(
        self,
        candidates: Sequence[CandidateProvenance],
        validation_scores_by_env: Mapping[str, Mapping[int, np.ndarray]],
        *,
        group_ids: Optional[Sequence[str]] = None,
        higher_is_better: bool = True,
        seed: int = 20260403,
        partition_tag: str = "validation",
    ) -> Tuple[CandidateProvenance, CandidateRobustScore, Dict[int, CandidateRobustScore]]:
        """Select final prompt maximizing simultaneous validation LCB; lowest run_id breaks ties."""
        if not candidates or len(candidates) > 4:
            raise ValueError(f"Expected 1..4 unique candidates, got {len(candidates)}")
        cand_map = {c.run_id: c for c in candidates}
        if len(cand_map) != len(candidates):
            raise ValueError("Duplicate candidate run_ids passed to select_final_prompt")

        filtered_val = {
            env_id: {cid: np.asarray(env_scores[cid], dtype=float) for cid in cand_map}
            for env_id, env_scores in validation_scores_by_env.items()
        }
        robust_scores = compute_candidate_robust_scores(
            filtered_val,
            group_ids=group_ids,
            higher_is_better=higher_is_better,
            alpha=self.alpha_validation,
            n_bootstrap=self.n_bootstrap_validation,
            seed=seed,
            partition_tag=partition_tag,
        )
        winner_score = select_robust_winner(robust_scores)
        return cand_map[winner_score.run_id], winner_score, robust_scores
