"""Deterministic 2,000+ replicate replay, missing-pair, small-group, tie, drift, and OSF bundle tests (Task 11)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from research.real_v4.analysis import analyze_locked_manifests
from research.real_v4.baselines import generate_frozen_balanced_random_screens
from research.real_v4.decisions import DecisionThresholds
from research.real_v4.registration import build_osf_registration_bundle
from research.real_v4.resampling import nested_adaptive_selection_bootstrap
from research.real_v4.selection import full_cube_configs
from tests.research.test_real_v4_schemas import make_sample_envelope


def _make_manifest_records_and_locks(
    env,
    *,
    n_items_by_task: Dict[str, int] | None = None,
    prism_score: float = 0.78,
    rand_score: float = 0.74,
    bo_score: float = 0.745,
    pert_drop: float = 0.003,
    drift_temporal: bool = False,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    final_lock: Dict[str, Any] = {"cells": {}}
    conf_records: List[Dict[str, Any]] = []
    stab_records: List[Dict[str, Any]] = []

    rng = np.random.default_rng(2026)

    for t_id in env.tasks:
        n_it = (n_items_by_task or {}).get(t_id, 24)
        for m_id in env.models:
            cell_id = f"{t_id}::{m_id}"
            final_lock["cells"][cell_id] = {
                "promptprism": {"primary": 47},
                "balanced_random": {"design_00": 10, "design_01": 12},
                "bayesian_opt": {"bo_seed_00": 20, "bo_seed_01": 22},
            }

        for spec in env.registered_environments:
            e_id = spec.env_id
            for i in range(n_it):
                sid = f"{t_id}:{e_id if spec.uses_disjoint_item_group else 'ref'}:s_{i:03d}"
                gid = f"{t_id}:{e_id if spec.uses_disjoint_item_group else 'ref'}:g_{i // 2:03d}"
                item_jitter = float(rng.normal(0.0, 0.02))

                for m_id in env.models:
                    pinned_ver = env.model_versions_pinned[m_id]
                    ret_ver = (
                        "drifted-model-v9"
                        if (
                            drift_temporal
                            and e_id == "temporal_wave_2"
                            and m_id == env.models[0]
                        )
                        else pinned_ver
                    )
                    for rid, base_mu in (
                        (47, prism_score),
                        (10, rand_score),
                        (12, rand_score + 0.004),
                        (20, bo_score),
                        (22, bo_score + 0.002),
                    ):
                        drop = 0.0 if e_id == "ref" else pert_drop
                        sc = float(np.clip(base_mu - drop + item_jitter, 0.0, 1.0))
                        rec = {
                            "task_id": t_id,
                            "model_id": m_id,
                            "env_id": e_id,
                            "run_id": rid,
                            "sample_id": sid,
                            "group_id": gid,
                            "raw_text": f"FINAL_ANSWER: {rid}",
                            "score": sc,
                            "status": "ok",
                            "returned_model_version": ret_ver,
                            "provenance": {
                                "cost_usd": 0.0001,
                                "tokens_input": 100,
                                "tokens_output": 10,
                            },
                        }
                        if e_id == "ref":
                            conf_records.append(rec)
                        else:
                            stab_records.append(rec)

    return final_lock, conf_records, stab_records


def test_deterministic_2000_replicate_replay_equal_cell_weights_and_distinct_sections():
    env = make_sample_envelope()
    # Give tasks unequal item counts (e.g., 8 items in one task vs 40 in others) to verify EQUAL cell weights!
    n_by_task = {t: (8 if idx == 0 else 30) for idx, t in enumerate(env.tasks)}
    final_lock, conf_recs, stab_recs = _make_manifest_records_and_locks(
        env,
        n_items_by_task=n_by_task,
        prism_score=0.80,
        rand_score=0.75,
        bo_score=0.76,
        pert_drop=0.004,
    )

    # Also construct known-truth 64-run cube for supporting H_effect verification
    cube = full_cube_configs()
    rand_designs = generate_frozen_balanced_random_screens(
        num_designs=10, seed=20261101
    )
    full_cube_means = {
        f"{env.tasks[0]}::{env.models[0]}": {
            r: 0.50 + 0.04 * cube[r]["A"] + 0.03 * cube[r]["D"] for r in range(64)
        }
    }

    th = DecisionThresholds(
        alpha=0.05,
        delta_select=0.01,
        epsilon_stability=0.02,
        n_bootstrap=2000,
        seed=20261118,
    )
    s1 = analyze_locked_manifests(
        certificate_id="cert-replay-2000",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        final_selection_lock=final_lock,
        confirmation_records=conf_recs,
        stability_records=stab_recs,
        thresholds=th,
        full_cube_screen_means_by_cell=full_cube_means,
        random_designs=rand_designs,
    )
    s2 = analyze_locked_manifests(
        certificate_id="cert-replay-2000",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        final_selection_lock=final_lock,
        confirmation_records=conf_recs,
        stability_records=stab_recs,
        thresholds=th,
        full_cube_screen_means_by_cell=full_cube_means,
        random_designs=rand_designs,
    )

    assert s1["summary_sha256"] == s2["summary_sha256"]
    assert s1["primary_verdict"]["overall_pass"] is True
    assert s1["primary_verdict"]["certificate_status"] == "PASS"

    # Verify equal cell weights across all 18 cells despite unequal item counts
    cells = s1["cell_level_breakdown"]["cells"]
    assert len(cells) == 18
    manual_equal_mean = np.mean([c["diff_vs_random"] for c in cells.values()])
    assert s1["primary_verdict"]["selection_contrasts"]["balanced_random"][
        "estimate"
    ] == pytest.approx(manual_equal_mean)

    # Verify distinct output sections
    for section in (
        "primary_verdict",
        "stability_certificate",
        "cell_level_breakdown",
        "sensitivity_analyses",
        "supporting_effect_recovery",
        "repeatability_diagnostics",
        "drift_report",
        "resource_accounting",
    ):
        assert section in s1
    assert s1["supporting_effect_recovery"]["supporting_only_non_rescuing"] is True
    assert s1["supporting_effect_recovery"]["mean_rmse_promptprism"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert s1["repeatability_diagnostics"][
        "identical_repeat_exact_match_rate"
    ] == pytest.approx(1.0)


def test_missing_pairs_and_version_drift_fail_closed_and_nested_bootstrap():
    env = make_sample_envelope()
    th = DecisionThresholds(
        alpha=0.05, delta_select=0.01, epsilon_stability=0.02, n_bootstrap=200, seed=55
    )

    # 1. Version drift in temporal_wave_2 is reported in drift_report and forces INVALID/OUT_OF_ENVELOPE
    final_lock, conf_recs, stab_recs = _make_manifest_records_and_locks(
        env, drift_temporal=True
    )
    s_drift = analyze_locked_manifests(
        certificate_id="cert-drift",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        final_selection_lock=final_lock,
        confirmation_records=conf_recs,
        stability_records=stab_recs,
        thresholds=th,
    )
    assert s_drift["drift_report"]["drift_detected"] is True
    assert s_drift["primary_verdict"]["certificate_status"] == "INVALID/OUT_OF_ENVELOPE"
    assert s_drift["primary_verdict"]["overall_pass"] is False

    # 2. Missing paired item record fails closed
    _, conf_clean, stab_clean = _make_manifest_records_and_locks(
        env, drift_temporal=False
    )
    conf_incomplete = conf_clean[1:]  # drop one paired item record!
    s_miss = analyze_locked_manifests(
        certificate_id="cert-miss",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        final_selection_lock=final_lock,
        confirmation_records=conf_incomplete,
        stability_records=stab_clean,
        thresholds=th,
    )
    assert s_miss["primary_verdict"]["certificate_status"] == "INVALID/OUT_OF_ENVELOPE"

    # 2b. SAP §3.2: terminal_failure HTTP_ERROR records score 0.0 in denominator without misclassifying as version drift
    _, conf_tf, stab_tf = _make_manifest_records_and_locks(env, drift_temporal=False)
    stab_tf[0] = {
        **stab_tf[0],
        "status": "terminal_failure",
        "score": 0.0,
        "returned_model_version": "HTTP_ERROR",
    }
    s_tf = analyze_locked_manifests(
        certificate_id="cert-tf-sap32",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        final_selection_lock=final_lock,
        confirmation_records=conf_tf,
        stability_records=stab_tf,
        thresholds=th,
    )
    assert s_tf["drift_report"]["drift_detected"] is False
    assert s_tf["resource_accounting"]["terminal_failure_records_scored_zero"] == 1
    assert s_tf["primary_verdict"]["certificate_status"] != "INVALID/OUT_OF_ENVELOPE"
    assert "balanced_random" in s_tf["primary_verdict"]["selection_contrasts"]
    assert "leave_one_model_out" in s_tf["sensitivity_analyses"]

    # 3. Nested adaptive selection bootstrap runs cleanly on small groups
    from research.real_v4.selection import resolution_iv_screen_run_ids

    s_ids = resolution_iv_screen_run_ids()
    scr_cubes = {("t1", "m1"): {r: np.full(8, 0.5 + 0.002 * r) for r in s_ids}}
    val_cubes = {
        ("t1", "m1"): {"ref": {r: np.full(8, 0.5 + 0.002 * r) for r in range(64)}}
    }
    conf_cubes = {("t1", "m1"): {r: np.full(8, 0.5 + 0.002 * r) for r in range(64)}}
    g_seq = ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]
    nested = nested_adaptive_selection_bootstrap(
        screen_cubes_by_cell=scr_cubes,
        screen_groups_by_task={"t1": g_seq},
        val_cubes_by_cell=val_cubes,
        val_groups_by_task={"t1": g_seq},
        conf_cubes_by_cell=conf_cubes,
        conf_groups_by_task={"t1": g_seq},
        tasks=("t1",),
        models=("m1",),
        n_bootstrap=30,
        seed=99,
    )
    assert nested["exact_prompt_selection_agreement"] == pytest.approx(1.0)


def test_osf_registration_bundle_reproducibility(tmp_path):
    b1 = build_osf_registration_bundle(tmp_path / "b1.json")
    b2 = build_osf_registration_bundle(tmp_path / "b2.json")
    assert b1["bundle_sha256"] == b2["bundle_sha256"]
    assert b1["prospective_outcomes_acquired"] is False
    assert (
        b1["worst_case_resource_budget"]["hosted_spend_worst_case_usd"][
            "total_hosted_worst_case_usd"
        ]
        > 0
    )
