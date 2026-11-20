"""Preregistered hierarchical selection, stability, drift, repeatability, and supporting analysis (Task 11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.real_v4.decisions import (
    PRIMARY_COMPARATORS,
    PROMPTPRISM_METHOD_ID,
    DecisionThresholds,
    StudyDecisionInput,
    TaskEnvironmentGroupData,
    evaluate_study_decisions,
)
from research.real_v4.resampling import (
    leave_one_task_out_sensitivity,
    multiway_task_model_bootstrap_sensitivity,
)
from research.real_v4.schemas import (
    CURRENT_SCHEMA_VERSION,
    OperatingEnvelope,
    SupportingDiagnostics,
    canonical_sha256,
)
from research.real_v4.selection import (
    FACTOR_IDS,
    full_cube_configs,
    resolution_iv_screen_run_ids,
)


def compute_supporting_effect_recovery(
    full_cube_screen_means_by_cell: Mapping[str, Mapping[int, float]],
    random_designs: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Compute supporting H_effect (6-factor main-effect RMSE vs 64-run cube; non-rescuing)."""
    cube = full_cube_configs()
    x_pm = np.array([[2 * cube[r][f] - 1 for f in FACTOR_IDS] for r in range(64)], dtype=float)
    d_main = np.column_stack([np.ones(64), x_pm])
    res4_ids = list(resolution_iv_screen_run_ids())
    pinv_res4 = np.linalg.pinv(d_main[res4_ids])
    pinv_rands = [np.linalg.pinv(d_main[list(d)]) for d in random_designs]

    prism_rmses: List[float] = []
    rand_rmses: List[float] = []
    per_cell: Dict[str, Any] = {}

    for cell_id, run_means in sorted(full_cube_screen_means_by_cell.items()):
        if len(run_means) < 64:
            continue
        y_full = np.array([float(run_means[r]) for r in range(64)], dtype=float)
        tau_true = (x_pm.T @ y_full) / 32.0  # (6,)

        tau_prism = 2.0 * (pinv_res4 @ y_full[res4_ids])[1:]
        rmse_p = float(np.sqrt(np.mean((tau_prism - tau_true) ** 2)))

        r_list = []
        for d_idx, d_runs in enumerate(random_designs):
            tau_r = 2.0 * (pinv_rands[d_idx] @ y_full[list(d_runs)])[1:]
            r_list.append(float(np.sqrt(np.mean((tau_r - tau_true) ** 2))))
        rmse_r = float(np.mean(r_list))

        prism_rmses.append(rmse_p)
        rand_rmses.append(rmse_r)
        per_cell[cell_id] = {
            "rmse_promptprism": rmse_p,
            "rmse_random_mean": rmse_r,
            "gain_rmse_reduction": float(rmse_r - rmse_p),
        }

    if not prism_rmses:
        return {
            "available": False,
            "supporting_only_non_rescuing": True,
        }

    mean_p = float(np.mean(prism_rmses))
    mean_r = float(np.mean(rand_rmses))
    return {
        "available": True,
        "supporting_only_non_rescuing": True,
        "mean_rmse_promptprism": mean_p,
        "mean_rmse_random": mean_r,
        "g_effect_gain": float(mean_r - mean_p),
        "relative_rmse_reduction_pct": float((mean_r - mean_p) / max(1e-12, mean_r) * 100.0),
        "per_cell": per_cell,
    }


def analyze_locked_manifests(
    *,
    certificate_id: str,
    study_id: str,
    algorithm_version: str,
    envelope: OperatingEnvelope,
    final_selection_lock: Mapping[str, Any],
    confirmation_records: Iterable[Mapping[str, Any]],
    stability_records: Iterable[Mapping[str, Any]],
    thresholds: DecisionThresholds = DecisionThresholds(
        alpha=0.05, delta_select=0.01, epsilon_stability=0.02, n_bootstrap=2000, seed=20261118
    ),
    full_cube_screen_means_by_cell: Optional[Mapping[str, Mapping[int, float]]] = None,
    random_designs: Sequence[Sequence[int]] = (),
    selection_resample_agreement: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the frozen Real-v4 analysis once from locked final selections and raw confirmation/stability records."""
    all_records = [*confirmation_records, *stability_records]

    # Index raw records by (task_id, model_id, env_id, run_id, sample_id)
    rec_index: Dict[Tuple[str, str, str, int], List[Mapping[str, Any]]] = {}
    observed_versions: Dict[str, Dict[str, str]] = {}
    drift_table: List[Dict[str, Any]] = []

    total_cost_usd = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    terminal_failures = 0

    for rec in all_records:
        t_id = str(rec["task_id"])
        m_id = str(rec["model_id"])
        e_id = str(rec["env_id"])
        r_id = int(rec["run_id"])
        rec_index.setdefault((t_id, m_id, e_id, r_id), []).append(rec)

        prov = rec.get("provenance", {})
        total_cost_usd += float(prov.get("cost_usd", 0.0))
        total_tokens_in += int(prov.get("tokens_input", 0))
        total_tokens_out += int(prov.get("tokens_output", 0))
        is_terminal_failure = rec.get("status") == "terminal_failure"
        if is_terminal_failure:
            terminal_failures += 1

        # Per SAP §3.2 & §3.3: terminal execution failures (e.g., HTTP_429 / timeout after bounded retries)
        # receive score=0.0 in the denominator; only completed responses carry a provider model_version string
        # eligible to trigger PROVIDER_VERSION_DRIFT.
        obs_env_map = observed_versions.setdefault(e_id, {})
        pinned = envelope.model_versions_pinned.get(m_id, "")
        if not is_terminal_failure:
            ret_ver = str(rec.get("returned_model_version", ""))
            if m_id not in obs_env_map or ret_ver != pinned:
                obs_env_map[m_id] = ret_ver
            if ret_ver != pinned:
                drift_table.append(
                    {
                        "env_id": e_id,
                        "task_id": t_id,
                        "model_id": m_id,
                        "pinned_version": pinned,
                        "observed_version": ret_ver,
                        "sample_id": str(rec["sample_id"]),
                    }
                )
        elif m_id not in obs_env_map:
            obs_env_map[m_id] = pinned

    # Assemble paired TaskEnvironmentGroupData strata across all registered environments and tasks
    strata: List[TaskEnvironmentGroupData] = []
    cell_breakdown: Dict[str, Any] = {}
    trajectory_distributions: Dict[str, Dict[str, Any]] = {}

    repeat_exact_matches: List[float] = []
    repeat_item_variances: List[float] = []

    cells_lock = final_selection_lock["cells"]

    for env_spec in envelope.registered_environments:
        e_id = env_spec.env_id
        for t_id in envelope.tasks:
            # Determine canonical sample_id order for (t_id, e_id) from first model's promptprism run
            first_m = envelope.models[0]
            first_c = f"{t_id}::{first_m}"
            first_p_run = int(cells_lock[first_c]["promptprism"]["primary"])
            anchor_rows = sorted(
                rec_index.get((t_id, first_m, e_id, first_p_run), []),
                key=lambda r: str(r["sample_id"]),
            )
            if not anchor_rows:
                # Missing environment or task stratum -> omit so decision engine catches missing cells cleanly
                continue

            ordered_sample_ids = [str(r["sample_id"]) for r in anchor_rows]
            group_ids = tuple(str(r["group_id"]) for r in anchor_rows)
            scores_by_model: Dict[str, Dict[str, np.ndarray]] = {}

            complete_task_env = True
            for m_id in envelope.models:
                cell_id = f"{t_id}::{m_id}"
                c_sel = cells_lock[cell_id]
                p_rid = int(c_sel["promptprism"]["primary"])

                def _extract_aligned_scores(rid: int) -> Optional[Tuple[np.ndarray, List[str]]]:
                    rows = {
                        str(r["sample_id"]): r
                        for r in rec_index.get((t_id, m_id, e_id, rid), [])
                    }
                    if any(sid not in rows for sid in ordered_sample_ids):
                        return None
                    arr = np.array([float(rows[sid]["score"]) for sid in ordered_sample_ids], dtype=float)
                    texts = [str(rows[sid].get("raw_text", "")) for sid in ordered_sample_ids]
                    return arr, texts

                p_res = _extract_aligned_scores(p_rid)
                if p_res is None:
                    complete_task_env = False
                    continue
                p_arr, p_texts = p_res

                # Repeatability diagnostics on PromptPrism across ref and all registered REPEAT environments
                if e_id == "ref":
                    rep_env_ids = [
                        es.env_id
                        for es in envelope.registered_environments
                        if es.family.value == "repeat"
                    ]
                    rep_maps = [
                        {str(r["sample_id"]): r for r in rec_index.get((t_id, m_id, r_eid, p_rid), [])}
                        for r_eid in rep_env_ids
                    ]
                    if rep_maps and all(all(sid in rm for sid in ordered_sample_ids) for rm in rep_maps):
                        for idx_s, sid in enumerate(ordered_sample_ids):
                            tset = {p_texts[idx_s], *(str(rm[sid].get("raw_text", "")) for rm in rep_maps)}
                            repeat_exact_matches.append(1.0 if len(tset) == 1 else 0.0)
                            svec = [float(p_arr[idx_s]), *(float(rm[sid]["score"]) for rm in rep_maps)]
                            repeat_item_variances.append(float(np.var(svec, ddof=1)))

                model_method_map: Dict[str, np.ndarray] = {PROMPTPRISM_METHOD_ID: p_arr}
                missing_comp = False
                for comp in PRIMARY_COMPARATORS:
                    traj_map = c_sel[comp]
                    traj_arrays: List[np.ndarray] = []
                    for _tid, tr_rid in sorted(traj_map.items()):
                        tr_res = _extract_aligned_scores(int(tr_rid))
                        if tr_res is None:
                            missing_comp = True
                            break
                        traj_arrays.append(tr_res[0])
                    if missing_comp or not traj_arrays:
                        missing_comp = True
                        break
                    traj_mat = np.vstack(traj_arrays)  # (n_traj, n_items)
                    model_method_map[comp] = np.mean(traj_mat, axis=0)

                    if e_id == "ref":
                        traj_means = np.mean(traj_mat, axis=1)
                        trajectory_distributions.setdefault(cell_id, {})[comp] = {
                            "n_trajectories": int(len(traj_means)),
                            "mean": float(np.mean(traj_means)),
                            "min": float(np.min(traj_means)),
                            "p05": float(np.percentile(traj_means, 5.0)),
                            "median": float(np.median(traj_means)),
                            "p95": float(np.percentile(traj_means, 95.0)),
                            "max": float(np.max(traj_means)),
                        }

                if not missing_comp:
                    scores_by_model[m_id] = model_method_map
                    if e_id == "ref":
                        cell_breakdown[cell_id] = {
                            "promptprism_mean": float(np.mean(p_arr)),
                            "balanced_random_mean": float(np.mean(model_method_map["balanced_random"])),
                            "bayesian_opt_mean": float(np.mean(model_method_map["bayesian_opt"])),
                            "diff_vs_random": float(np.mean(p_arr - model_method_map["balanced_random"])),
                            "diff_vs_bayesian_opt": float(np.mean(p_arr - model_method_map["bayesian_opt"])),
                        }

            if complete_task_env:
                strata.append(
                    TaskEnvironmentGroupData(
                        task_id=t_id,
                        env_id=e_id,
                        group_ids=group_ids,
                        scores=scores_by_model,
                    )
                )

    eff_rec = (
        compute_supporting_effect_recovery(full_cube_screen_means_by_cell, random_designs)
        if full_cube_screen_means_by_cell
        else {"available": False, "supporting_only_non_rescuing": True}
    )

    sup_diag = SupportingDiagnostics(
        effect_recovery_rmse_prism=eff_rec.get("mean_rmse_promptprism"),
        effect_recovery_rmse_random=eff_rec.get("mean_rmse_random"),
        effect_recovery_gain=eff_rec.get("g_effect_gain"),
        identical_repeat_exact_match_rate=(
            float(np.mean(repeat_exact_matches)) if repeat_exact_matches else None
        ),
        score_variance_across_repeats=(
            float(np.mean(repeat_item_variances)) if repeat_item_variances else None
        ),
        selection_resample_agreement=selection_resample_agreement,
    )

    decision_res = evaluate_study_decisions(
        StudyDecisionInput(
            certificate_id=certificate_id,
            study_id=study_id,
            algorithm_version=algorithm_version,
            envelope=envelope,
            strata=strata,
            observed_model_versions=observed_versions,
            supporting_diagnostics=sup_diag,
        ),
        thresholds=thresholds,
    )

    # Sensitivity analyses run only when envelope strata are complete
    if decision_res.certificate.status.value != "INVALID/OUT_OF_ENVELOPE":
        loto = leave_one_task_out_sensitivity(strata, envelope)
        multiway = multiway_task_model_bootstrap_sensitivity(
            strata, envelope, n_bootstrap=min(1000, thresholds.n_bootstrap), seed=thresholds.seed + 11
        )
        c_means_lomo: Dict[Tuple[str, str, str, str], float] = {}
        for st in strata:
            for model_id in envelope.models:
                for method_id, arr in st.scores[model_id].items():
                    c_means_lomo[(st.task_id, model_id, st.env_id, method_id)] = float(np.mean(arr))
        by_omitted_model: Dict[str, Any] = {}
        for om in envelope.models:
            kept_m = [m for m in envelope.models if m != om]
            if kept_m:
                g_sel = {
                    comp: float(
                        np.mean(
                            [
                                c_means_lomo[(t, m, "ref", PROMPTPRISM_METHOD_ID)] - c_means_lomo[(t, m, "ref", comp)]
                                for t in envelope.tasks
                                for m in kept_m
                            ]
                        )
                    )
                    for comp in PRIMARY_COMPARATORS
                }
                ref_m = float(
                    np.mean([c_means_lomo[(t, m, "ref", PROMPTPRISM_METHOD_ID)] for t in envelope.tasks for m in kept_m])
                )
                d_max = float(
                    max(
                        ref_m
                        - float(
                            np.mean(
                                [c_means_lomo[(t, m, e_id, PROMPTPRISM_METHOD_ID)] for t in envelope.tasks for m in kept_m]
                            )
                        )
                        for e_id in envelope.perturbation_env_ids
                    )
                )
                by_omitted_model[om] = {
                    "n_retained_cells": len(envelope.tasks) * len(kept_m),
                    "g_select_by_comparator": g_sel,
                    "d_stability_max_drop": d_max,
                }
        lomo = {
            "analysis_type": "leave_one_model_out_sensitivity",
            "confirmatory_primary": False,
            "by_omitted_model": by_omitted_model,
        }
    else:
        loto = {"analysis_type": "leave_one_task_out_sensitivity", "available": False}
        lomo = {"analysis_type": "leave_one_model_out_sensitivity", "available": False}
        multiway = {"analysis_type": "multiway_task_model_pigeonhole_bootstrap", "available": False}

    summary: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": study_id,
        "primary_verdict": {
            "verdict_code": decision_res.verdict_code.value,
            "allowed_conclusion": decision_res.allowed_conclusion,
            "overall_pass": decision_res.overall_pass,
            "claim_p_pass": decision_res.claim_p_pass,
            "claim_p_practical_pass": decision_res.claim_p_practical_pass,
            "claim_s_pass": decision_res.claim_s_pass,
            "cell_consistency_gate_pass": decision_res.cell_consistency_gate_pass,
            "positive_cells_by_comparator": decision_res.positive_cells_by_comparator,
            "model_family_means_by_comparator": decision_res.model_family_means_by_comparator,
            "selection_contrasts": {
                k: v.model_dump(mode="json") for k, v in decision_res.selection_contrasts.items()
            },
            "stability_summary": (
                decision_res.stability_summary.model_dump(mode="json")
                if decision_res.stability_summary is not None
                else None
            ),
            "certificate_status": decision_res.certificate.status.value,
            "certificate_sha256": decision_res.certificate.certificate_sha256(),
        },
        "stability_certificate": json.loads(decision_res.certificate.canonical_json()),
        "cell_level_breakdown": {
            "equal_cell_weight": 1.0 / (len(envelope.tasks) * len(envelope.models)),
            "cells": cell_breakdown,
            "baseline_trajectory_distributions": trajectory_distributions,
        },
        "sensitivity_analyses": {
            "leave_one_task_out": loto,
            "leave_one_model_out": lomo,
            "multiway_task_model_bootstrap": multiway,
        },
        "supporting_effect_recovery": eff_rec,
        "repeatability_diagnostics": {
            "supporting_only_non_rescuing": True,
            "identical_repeat_exact_match_rate": sup_diag.identical_repeat_exact_match_rate,
            "mean_item_score_variance_across_repeats": sup_diag.score_variance_across_repeats,
            "n_items_evaluated": len(repeat_exact_matches),
        },
        "drift_report": {
            "drift_detected": len(drift_table) > 0,
            "drift_event_count": len(drift_table),
            "events": drift_table[:50],
        },
        "resource_accounting": {
            "total_confirmation_and_stability_records": len(all_records),
            "terminal_failure_records_scored_zero": terminal_failures,
            "total_tokens_input": total_tokens_in,
            "total_tokens_output": total_tokens_out,
            "total_hosted_cost_usd": round(total_cost_usd, 6),
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary
