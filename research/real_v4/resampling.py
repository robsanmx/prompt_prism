"""Paired hierarchical, leave-one-task-out, multiway, and nested adaptive resampling (Task 11)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from research.real_v4.decisions import (
    PROMPTPRISM_METHOD_ID,
    DecisionThresholds,
    StudyDecisionInput,
    TaskEnvironmentGroupData,
    evaluate_study_decisions,
)
from research.real_v4.schemas import OperatingEnvelope
from research.real_v4.selection import PromptPrismRobustSelector


def leave_one_task_out_sensitivity(
    strata: Sequence[TaskEnvironmentGroupData],
    envelope: OperatingEnvelope,
    comparators: Tuple[str, ...] = ("balanced_random", "bayesian_opt"),
) -> Dict[str, Any]:
    """Compute equal-cell-weighted Claim P and Claim S point summaries leaving out one task at a time."""
    tasks = envelope.tasks
    models = envelope.models
    pert_envs = envelope.perturbation_env_ids

    # Precompute cell means per (task, model, env, method)
    cell_means: Dict[Tuple[str, str, str, str], float] = {}
    for st in strata:
        for model_id in models:
            for method_id, arr in st.scores[model_id].items():
                cell_means[(st.task_id, model_id, st.env_id, method_id)] = float(np.mean(arr))

    per_dropped_task: Dict[str, Any] = {}
    for omitted_task in tasks:
        kept_tasks = [t for t in tasks if t != omitted_task]
        n_kept_cells = len(kept_tasks) * len(models)

        g_select: Dict[str, float] = {}
        for comp in comparators:
            diffs = [
                cell_means[(t, m, "ref", PROMPTPRISM_METHOD_ID)] - cell_means[(t, m, "ref", comp)]
                for t in kept_tasks
                for m in models
            ]
            g_select[comp] = float(np.mean(diffs))

        ref_mu = float(
            np.mean([cell_means[(t, m, "ref", PROMPTPRISM_METHOD_ID)] for t in kept_tasks for m in models])
        )
        drops = {
            e_id: ref_mu
            - float(np.mean([cell_means[(t, m, e_id, PROMPTPRISM_METHOD_ID)] for t in kept_tasks for m in models]))
            for e_id in pert_envs
        }
        per_dropped_task[omitted_task] = {
            "n_retained_cells": n_kept_cells,
            "g_select_by_comparator": g_select,
            "d_stability_max_drop": float(max(drops.values())),
        }

    return {
        "analysis_type": "leave_one_task_out_sensitivity",
        "confirmatory_primary": False,
        "by_omitted_task": per_dropped_task,
    }


def multiway_task_model_bootstrap_sensitivity(
    strata: Sequence[TaskEnvironmentGroupData],
    envelope: OperatingEnvelope,
    comparators: Tuple[str, ...] = ("balanced_random", "bayesian_opt"),
    n_bootstrap: int = 2000,
    seed: int = 20261199,
) -> Dict[str, Any]:
    """Two-way pigeonhole bootstrap across tasks and models (robustness diagnostic only, not population claim)."""
    tasks = envelope.tasks
    models = envelope.models
    T = len(tasks)
    M = len(models)

    ref_prism = np.zeros((T, M), dtype=float)
    ref_comp = {c: np.zeros((T, M), dtype=float) for c in comparators}

    st_ref = {st.task_id: st for st in strata if st.env_id == "ref"}
    for t_i, t_id in enumerate(tasks):
        for m_i, m_id in enumerate(models):
            ref_prism[t_i, m_i] = float(np.mean(st_ref[t_id].scores[m_id][PROMPTPRISM_METHOD_ID]))
            for c in comparators:
                ref_comp[c][t_i, m_i] = float(np.mean(st_ref[t_id].scores[m_id][c]))

    rng = np.random.default_rng(seed)
    t_idx = rng.integers(0, T, size=(n_bootstrap, T))
    m_idx = rng.integers(0, M, size=(n_bootstrap, M))

    out_comp: Dict[str, Any] = {}
    for c in comparators:
        diff_mat = ref_prism - ref_comp[c]
        boots = np.array([diff_mat[t_idx[b]][:, m_idx[b]].mean() for b in range(n_bootstrap)], dtype=float)
        out_comp[c] = {
            "estimate": float(diff_mat.mean()),
            "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)),
        }

    return {
        "analysis_type": "multiway_task_model_pigeonhole_bootstrap",
        "confirmatory_primary": False,
        "scope_note": "Finite-panel reweighting sensitivity only; six tasks and three models do not justify broad population generalization.",
        "comparators": out_comp,
    }


def nested_adaptive_selection_bootstrap(
    *,
    screen_cubes_by_cell: Mapping[Tuple[str, str], Mapping[int, np.ndarray]],
    screen_groups_by_task: Mapping[str, Sequence[str]],
    val_cubes_by_cell: Mapping[Tuple[str, str], Mapping[str, Mapping[int, np.ndarray]]],
    val_groups_by_task: Mapping[str, Sequence[str]],
    conf_cubes_by_cell: Mapping[Tuple[str, str], Mapping[int, np.ndarray]],
    conf_groups_by_task: Mapping[str, Sequence[str]],
    tasks: Sequence[str],
    models: Sequence[str],
    n_bootstrap: int = 200,
    seed: int = 20261250,
) -> Dict[str, Any]:
    """Nested bootstrap re-running PromptPrism candidate generation and validation inside each group resample."""
    selector = PromptPrismRobustSelector(pad_to_max=True, n_bootstrap_screen=80, n_bootstrap_validation=80)
    rng = np.random.default_rng(seed)

    # Compute baseline locked run_ids on unresampled data for selection-agreement diagnostic
    locked_runs: Dict[Tuple[str, str], int] = {}
    for t_id in tasks:
        for m_id in models:
            cands = selector.generate_candidates(
                screen_cubes_by_cell[(t_id, m_id)],
                group_ids=screen_groups_by_task[t_id],
                seed=seed,
            )
            win, _, _ = selector.select_final_prompt(
                cands,
                val_cubes_by_cell[(t_id, m_id)],
                group_ids=val_groups_by_task[t_id],
                seed=seed + 1,
            )
            locked_runs[(t_id, m_id)] = win.run_id

    def _group_index_lists(g_seq: Sequence[str]) -> List[np.ndarray]:
        buckets: Dict[str, List[int]] = {}
        for idx, g in enumerate(g_seq):
            buckets.setdefault(str(g), []).append(idx)
        return [np.array(v, dtype=int) for v in buckets.values()]

    scr_g_lists = {t: _group_index_lists(screen_groups_by_task[t]) for t in tasks}
    val_g_lists = {t: _group_index_lists(val_groups_by_task[t]) for t in tasks}
    conf_g_lists = {t: _group_index_lists(conf_groups_by_task[t]) for t in tasks}

    nested_conf_means = np.zeros(n_bootstrap, dtype=float)
    agreement_counts = 0
    total_cell_draws = n_bootstrap * len(tasks) * len(models)

    for b in range(n_bootstrap):
        cell_scores = []
        for t_id in tasks:
            sg = scr_g_lists[t_id]
            vg = val_g_lists[t_id]
            cg = conf_g_lists[t_id]
            s_pick = np.concatenate([sg[i] for i in rng.integers(0, len(sg), size=len(sg))])
            v_pick = np.concatenate([vg[i] for i in rng.integers(0, len(vg), size=len(vg))])
            c_pick = np.concatenate([cg[i] for i in rng.integers(0, len(cg), size=len(cg))])

            s_g_boot = [str(screen_groups_by_task[t_id][i]) for i in s_pick]
            v_g_boot = [str(val_groups_by_task[t_id][i]) for i in v_pick]

            for m_id in models:
                s_boot = {r: arr[s_pick] for r, arr in screen_cubes_by_cell[(t_id, m_id)].items()}
                v_boot = {
                    e_id: {r: arr[v_pick] for r, arr in r_map.items()}
                    for e_id, r_map in val_cubes_by_cell[(t_id, m_id)].items()
                }
                cands_b = selector.generate_candidates(s_boot, group_ids=s_g_boot, seed=seed + 10 + b)
                win_b, _, _ = selector.select_final_prompt(cands_b, v_boot, group_ids=v_g_boot, seed=seed + 20 + b)
                if win_b.run_id == locked_runs[(t_id, m_id)]:
                    agreement_counts += 1
                c_arr = conf_cubes_by_cell[(t_id, m_id)][win_b.run_id][c_pick]
                cell_scores.append(float(np.mean(c_arr)))
        nested_conf_means[b] = float(np.mean(cell_scores))

    return {
        "analysis_type": "nested_adaptive_selection_bootstrap",
        "confirmatory_primary": False,
        "n_bootstrap": int(n_bootstrap),
        "mean_nested_confirmation_score": float(np.mean(nested_conf_means)),
        "ci95_low": float(np.percentile(nested_conf_means, 2.5)),
        "ci95_high": float(np.percentile(nested_conf_means, 97.5)),
        "exact_prompt_selection_agreement": float(agreement_counts / max(1, total_cell_draws)),
    }
