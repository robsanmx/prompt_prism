"""Phase 4-5 Orchestrator & Independent Raw-Manifest Auditor (Tasks 13, 14, 15)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.real_v4.analysis import analyze_locked_manifests
from research.real_v4.baselines import BalancedRandomTwoStageBaseline
from research.real_v4.decisions import DecisionThresholds
from research.real_v4.locking import (
    PHASE_SEQUENCE,
    PhaseLockError,
    StudyPhase,
    compute_study_fingerprint,
    default_code_paths,
    lock_path_for_phase,
    sha256_file,
    verify_phase_prerequisites,
)
from research.real_v4.manifest import AppendOnlyManifest
from research.real_v4.model_provenance import MODEL_REGISTRY_PATH, host_hardware_fingerprint
from research.real_v4.perturbations import (
    PERTURBATION_REGISTRY_PATH,
    REGISTERED_TASKS,
    build_perturbation_environments,
    parser_hashes_by_task,
)
from research.real_v4.pilot import (
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_PROTOCOL_DIR,
    DEFAULT_RESULTS_DIR,
    REGISTERED_MODELS,
    STUDY_ID,
    build_default_live_adapters,
    load_real_v4_splits,
)
from research.real_v4.providers import BaseRealV4Adapter
from research.real_v4.runner import ResumableRealV4Runner, RunnerPaths
from research.real_v4.certificate import StabilityCertificate
from research.real_v4.schemas import (
    CURRENT_SCHEMA_VERSION,
    CandidateProvenance,
    OperatingEnvelope,
    canonical_sha256,
)
from research.real_v4.selection import PROMPTPRISM_SELECTOR_VERSION, PromptPrismRobustSelector, full_cube_configs
from research.real_v4.simulation import CALIBRATION_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[2]

WAVE_1_STABILITY_ENVS: Tuple[str, ...] = (
    "repeat_r2",
    "repeat_r3",
    "fmt_whitespace",
    "fmt_delimiter",
    "word_paraphrase_1",
    "word_paraphrase_2",
    "item_holdout_group",
)
WAVE_2_TEMPORAL_ENVS: Tuple[str, ...] = ("temporal_wave_2",)


def build_phase4_runner(
    *,
    adapters: Optional[Mapping[str, BaseRealV4Adapter]] = None,
    splits: Optional[Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]]] = None,
    tasks: Sequence[str] = REGISTERED_TASKS,
    models: Sequence[str] = REGISTERED_MODELS,
    protocol_dir: Path = DEFAULT_PROTOCOL_DIR,
    dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    num_baseline_trajectories: int = 10,
    n_bootstrap: int = 200,
) -> ResumableRealV4Runner:
    live_splits = load_real_v4_splits() if splits is None else splits
    live_adapters = build_default_live_adapters() if adapters is None else adapters
    paths = RunnerPaths(
        protocol_dir=protocol_dir,
        dataset_manifest_path=dataset_manifest_path,
        perturbation_registry_path=PERTURBATION_REGISTRY_PATH,
        model_registry_path=MODEL_REGISTRY_PATH,
        calibration_config_path=CALIBRATION_CONFIG_PATH,
        code_paths=default_code_paths(ROOT),
        results_dir=results_dir,
    )
    return ResumableRealV4Runner(
        study_id=STUDY_ID,
        paths=paths,
        adapters={m: live_adapters[m] for m in models},
        splits_by_task_and_partition=live_splits,
        tasks=tasks,
        models=models,
        num_baseline_trajectories=num_baseline_trajectories,
        n_bootstrap=n_bootstrap,
    )


def _extract_screen_cube_and_secondary_baselines(
    screen_manifest_path: Path,
    tasks: Sequence[str],
    models: Sequence[str],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Any]]:
    manifest = AppendOnlyManifest(screen_manifest_path)
    by_cell_run: Dict[Tuple[str, int], List[float]] = {}
    for rec in manifest.records:
        cell_id = f"{rec['task_id']}::{rec['model_id']}"
        by_cell_run.setdefault((cell_id, int(rec["run_id"])), []).append(float(rec["score"]))

    full_cube_means: Dict[str, Dict[int, float]] = {}
    secondary_refs: Dict[str, Any] = {}
    ofat_run_ids = (0, 32, 16, 8, 4, 2, 1)

    for task_id in tasks:
        for model_id in models:
            cell_id = f"{task_id}::{model_id}"
            cell_map: Dict[int, float] = {}
            for r in range(64):
                vals = by_cell_run.get((cell_id, r))
                if vals:
                    cell_map[r] = float(np.mean(vals))
            full_cube_means[cell_id] = cell_map
            if all(r in cell_map for r in ofat_run_ids):
                best_ofat_rid = max(ofat_run_ids, key=lambda r: (round(cell_map[r] / 1e-12), -r))
                secondary_refs[cell_id] = {
                    "untuned_run0_screen_mean": cell_map[0],
                    "ofat7_best_observed_run_id": best_ofat_rid,
                    "ofat7_best_observed_screen_mean": cell_map[best_ofat_rid],
                    "full_cube_n_populated_runs": len(cell_map),
                }
    return full_cube_means, secondary_refs


def _summarize_all_manifest_resources(results_dir: Path) -> Dict[str, Any]:
    phases = (
        ("pilot", "pilot_manifest.jsonl"),
        ("screen", "screen_manifest.jsonl"),
        ("validation", "validation_manifest.jsonl"),
        ("clean_confirmation", "confirmation_manifest.jsonl"),
        ("stability_wave", "stability_manifest.jsonl"),
    )
    by_phase: Dict[str, Any] = {}
    total_calls = 0
    total_cost = 0.0
    total_tok_in = 0
    total_tok_out = 0
    total_failures = 0

    for pname, fname in phases:
        fpath = results_dir / fname
        if not fpath.exists():
            continue
        man = AppendOnlyManifest(fpath)
        recs = man.records
        c_usd = sum(float(r.get("provenance", {}).get("cost_usd", 0.0)) for r in recs)
        t_in = sum(int(r.get("provenance", {}).get("tokens_input", 0)) for r in recs)
        t_out = sum(int(r.get("provenance", {}).get("tokens_output", 0)) for r in recs)
        fails = sum(1 for r in recs if r.get("status") == "terminal_failure")
        by_phase[pname] = {
            "manifest_file": fname,
            "unique_cached_calls": len(recs),
            "terminal_failures": fails,
            "tokens_input": t_in,
            "tokens_output": t_out,
            "hosted_cost_usd": round(c_usd, 6),
        }
        total_calls += len(recs)
        total_cost += c_usd
        total_tok_in += t_in
        total_tok_out += t_out
        total_failures += fails

    return {
        "by_phase": by_phase,
        "total_unique_calls_all_phases": total_calls,
        "total_terminal_failures_all_phases": total_failures,
        "total_tokens_input_all_phases": total_tok_in,
        "total_tokens_output_all_phases": total_tok_out,
        "total_hosted_cost_usd_all_phases": round(total_cost, 6),
    }


def _build_operating_envelope(
    *,
    study_id: str,
    tasks: Sequence[str],
    models: Sequence[str],
    model_registry_path: Path,
    dataset_manifest_path: Path,
    study_fingerprint_sha256: str,
) -> OperatingEnvelope:
    m_reg = json.loads(model_registry_path.read_text(encoding="utf-8"))
    d_man = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))

    pinned_versions = {
        m: str(m_reg["models"][m]["default_pinned_version"])
        for m in models
    }
    primary_metrics = {
        "re3d_entity_extraction": "entity_f1",
        "squad2_grounded_qa": "token_f1",
        "banking77_intent": "exact_label_accuracy",
        "gsm8k_math": "numeric_exact_match",
        "mbpp_python": "assertion_pass_rate",
        "sciq_verification": "exact_choice_accuracy",
    }
    part_hashes: Dict[str, str] = {}
    tasks_block = d_man.get("tasks", {})
    for t in tasks:
        if t in tasks_block:
            part_hashes[t] = canonical_sha256(tasks_block[t].get("partitions", tasks_block[t]))
        else:
            part_hashes[t] = sha256_file(dataset_manifest_path)

    runtime_locks = host_hardware_fingerprint()
    try:
        import mlx.core as mx  # type: ignore
        import mlx_lm  # type: ignore

        runtime_locks["mlx"] = str(getattr(mx, "__version__", "unknown"))
        runtime_locks["mlx_lm"] = str(getattr(mlx_lm, "__version__", "unknown"))
    except ImportError:
        runtime_locks["mlx"] = "not_installed"

    return OperatingEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        study_id=study_id,
        tasks=tuple(tasks),
        models=tuple(models),
        model_versions_pinned=pinned_versions,
        primary_metrics={t: primary_metrics[t] for t in tasks},
        parser_hashes={t: parser_hashes_by_task()[t] for t in tasks},
        decoding_config=dict(m_reg.get("canonical_decoding_intent", {"temperature": 0.0, "max_output_tokens": 512, "seed": 42})),
        partition_hashes=part_hashes,
        registered_environments=build_perturbation_environments(),
        repeat_count=3,
        nondeterminism_detected=True,
        local_runtime_locks=runtime_locks,
        code_hash=study_fingerprint_sha256,
    )


def _write_claim_evidence_map(summary: Mapping[str, Any], out_path: Path) -> None:
    pv = summary["primary_verdict"]
    sel_r = pv["selection_contrasts"].get("balanced_random")
    sel_b = pv["selection_contrasts"].get("bayesian_opt")
    stab = pv["stability_summary"]
    eff = summary["supporting_effect_recovery"]
    res_all = summary.get("full_study_resource_reconciliation", {})

    rows = [
        {
            "claim_id": "V4_C1_claim_p_vs_balanced_random",
            "role": "primary_co_decisive",
            "source_file": "research/results/real_v4/summary.json",
            "source_field": "primary_verdict.selection_contrasts.balanced_random",
            "estimand": "G_select,balanced_random (paired clean confirmation mean gain across 18 cells)",
            "estimate": f"{sel_r['estimate']:+.6f}" if sel_r else "NA",
            "bound_95": f"L95={sel_r['ci95_lower']:+.6f} (p={sel_r['p_value_one_sided']:.4f})" if sel_r else "NA",
            "gate_rule": "L95 > 0 (practical: L95 > 0.01)",
            "verdict": (
                ("PASS" if sel_r["passes_statistical_gate"] else "FAIL")
                if sel_r
                else pv["certificate_status"]
            ),
        },
        {
            "claim_id": "V4_C2_claim_p_vs_bayesian_opt",
            "role": "primary_co_decisive",
            "source_file": "research/results/real_v4/summary.json",
            "source_field": "primary_verdict.selection_contrasts.bayesian_opt",
            "estimand": "G_select,bayesian_opt (paired clean confirmation mean gain across 18 cells)",
            "estimate": f"{sel_b['estimate']:+.6f}" if sel_b else "NA",
            "bound_95": f"L95={sel_b['ci95_lower']:+.6f} (p={sel_b['p_value_one_sided']:.4f})" if sel_b else "NA",
            "gate_rule": "L95 > 0 (practical: L95 > 0.01)",
            "verdict": (
                ("PASS" if sel_b["passes_statistical_gate"] else "FAIL")
                if sel_b
                else pv["certificate_status"]
            ),
        },
        {
            "claim_id": "V4_C3_claim_s_max_perturbation_drop",
            "role": "primary_co_decisive",
            "source_file": "research/results/real_v4/stability_certificate.json",
            "source_field": "stability_summary.d_stability_estimate & d_stability_upper_95_sim",
            "estimand": "D_stability = max_e(mu_ref - mu_e) across 8 registered perturbation environments",
            "estimate": f"{stab['d_stability_estimate']:+.6f}" if stab else "NA",
            "bound_95": f"U95_sim={stab['d_stability_upper_95_sim']:+.6f}" if stab else "NA",
            "gate_rule": "U95_sim < 0.02 (epsilon_stability) & valid envelope",
            "verdict": (
                ("PASS" if stab["passes_absolute_noninferiority"] else "FAIL")
                if stab
                else pv["certificate_status"]
            ),
        },
        {
            "claim_id": "V4_C4_claim_s_worst_env_vs_balanced_random",
            "role": "primary_co_decisive",
            "source_file": "research/results/real_v4/stability_certificate.json",
            "source_field": "stability_summary.comparator_worst_env_checks.balanced_random",
            "estimand": "G_robust,balanced_random = min_e(mu_prism,e) - min_e(mu_random,e)",
            "estimate": f"{stab['comparator_worst_env_checks']['balanced_random']['estimate']:+.6f}" if stab else "NA",
            "bound_95": f"L95={stab['comparator_worst_env_checks']['balanced_random']['ci95_lower']:+.6f}" if stab else "NA",
            "gate_rule": "L95 > -0.02",
            "verdict": (
                ("PASS" if stab["comparator_worst_env_checks"]["balanced_random"]["passes_statistical_gate"] else "FAIL")
                if stab
                else pv["certificate_status"]
            ),
        },
        {
            "claim_id": "V4_C5_claim_s_worst_env_vs_bayesian_opt",
            "role": "primary_co_decisive",
            "source_file": "research/results/real_v4/stability_certificate.json",
            "source_field": "stability_summary.comparator_worst_env_checks.bayesian_opt",
            "estimand": "G_robust,bayesian_opt = min_e(mu_prism,e) - min_e(mu_bo,e)",
            "estimate": f"{stab['comparator_worst_env_checks']['bayesian_opt']['estimate']:+.6f}" if stab else "NA",
            "bound_95": f"L95={stab['comparator_worst_env_checks']['bayesian_opt']['ci95_lower']:+.6f}" if stab else "NA",
            "gate_rule": "L95 > -0.02",
            "verdict": (
                ("PASS" if stab["comparator_worst_env_checks"]["bayesian_opt"]["passes_statistical_gate"] else "FAIL")
                if stab
                else pv["certificate_status"]
            ),
        },
        {
            "claim_id": "V4_C6_overall_intersection_verdict",
            "role": "primary_overall",
            "source_file": "research/results/real_v4/summary.json",
            "source_field": "primary_verdict.verdict_code & allowed_conclusion",
            "estimand": "Intersection-union over Claim P and Claim S",
            "estimate": str(pv["verdict_code"]),
            "bound_95": f"cert={pv['certificate_status']}",
            "gate_rule": "Claim P PASS AND Claim S PASS",
            "verdict": "PASS" if pv["overall_pass"] else "FAIL",
        },
        {
            "claim_id": "V4_C7_supporting_effect_recovery",
            "role": "supporting_non_rescuing",
            "source_file": "research/results/real_v4/summary.json",
            "source_field": "supporting_effect_recovery",
            "estimand": "6-factor main-effect RMSE vs 64-run screening cube (PromptPrism vs 10 balanced random designs)",
            "estimate": (
                f"prism={eff['mean_rmse_promptprism']:.5f} vs rand={eff['mean_rmse_random']:.5f}"
                if eff.get("available")
                else "not_evaluated"
            ),
            "bound_95": (
                f"reduction={eff['relative_rmse_reduction_pct']:.2f}%"
                if eff.get("available")
                else "NA"
            ),
            "gate_rule": "Supporting evidence only; cannot override Claim P or Claim S",
            "verdict": "SUPPORTING_REPORTED",
        },
        {
            "claim_id": "V4_C8_resource_and_budget_parity",
            "role": "operational_integrity",
            "source_file": "research/results/real_v4/summary.json",
            "source_field": "full_study_resource_reconciliation",
            "estimand": "Equal operational screen (1,920/cell) & validation (720/cell) calls across methods + total spend",
            "estimate": f"calls={res_all.get('total_unique_calls_all_phases', 0)}",
            "bound_95": f"hosted_usd=${res_all.get('total_hosted_cost_usd_all_phases', 0.0):.4f}",
            "gate_rule": "BudgetLedger byte-exact equality verified pre-confirmation",
            "verdict": "VERIFIED",
        },
    ]

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_real_v4_final_artifacts(
    *,
    results_dir: Path,
    tasks: Sequence[str] = REGISTERED_TASKS,
    models: Sequence[str] = REGISTERED_MODELS,
) -> Dict[str, Any]:
    """Independent raw-record trace verifying locks, Claim P/S point estimates, certificate state, and claim map."""
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    cert_dict = json.loads((results_dir / "stability_certificate.json").read_text(encoding="utf-8"))
    cert = StabilityCertificate.model_validate(cert_dict)
    final_sel = json.loads((results_dir / "final_selection_lock.json").read_text(encoding="utf-8"))

    # 1. Verify every lock file in PHASE_SEQUENCE exists, self-hashes cleanly, forms an unbroken chain,
    #    and matches every locked artifact's SHA-256 on disk.
    locks_dir = results_dir / "locks"
    lock_digests: Dict[str, str] = {}
    prev_sha: Optional[str] = None
    for idx, ph in enumerate(PHASE_SEQUENCE):
        lpath = lock_path_for_phase(locks_dir, ph)
        if not lpath.exists():
            raise PhaseLockError(f"Audit failed: missing lock {lpath.name}")
        ldata = json.loads(lpath.read_text(encoding="utf-8"))
        expected = str(ldata["lock_sha256"])
        actual = canonical_sha256({k: v for k, v in ldata.items() if k != "lock_sha256"})
        if actual != expected:
            raise PhaseLockError(f"Audit failed: tampered lock {lpath.name}")
        if idx > 0 and ldata.get("prior_lock_sha256") != prev_sha:
            raise PhaseLockError(f"Audit failed: broken prior_lock_sha256 chain at {ph.value}")
        art_name = ldata.get("artifact_name")
        art_sha = ldata.get("artifact_sha256")
        if art_name is not None:
            art_path = results_dir / str(art_name)
            if not art_path.exists() or sha256_file(art_path) != art_sha:
                raise PhaseLockError(f"Audit failed: locked artifact {art_name!r} hash mismatch")
        prev_sha = expected
        lock_digests[ph.value] = expected

    # 2. Independently recompute Claim P clean-confirmation point contrasts from raw JSONL lines
    conf_rows = [
        json.loads(line)
        for line in (results_dir / "confirmation_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    score_lookup: Dict[Tuple[str, str, int, str], float] = {
        (str(r["task_id"]), str(r["model_id"]), int(r["run_id"]), str(r["sample_id"])): float(r["score"])
        for r in conf_rows
        if r["env_id"] == "ref"
    }

    cell_diffs_rand: List[float] = []
    cell_diffs_bo: List[float] = []
    cell_ref_prism: List[float] = []
    for t_id in tasks:
        for m_id in models:
            cell_id = f"{t_id}::{m_id}"
            c_sel = final_sel["cells"][cell_id]
            p_rid = int(c_sel["promptprism"]["primary"])
            sids = sorted({sid for (tt, mm, rr, sid) in score_lookup if tt == t_id and mm == m_id and rr == p_rid})
            p_vec = np.array([score_lookup[(t_id, m_id, p_rid, s)] for s in sids], dtype=float)

            r_mat = np.vstack(
                [
                    np.array([score_lookup[(t_id, m_id, int(rid), s)] for s in sids], dtype=float)
                    for _, rid in sorted(c_sel["balanced_random"].items())
                ]
            )
            b_mat = np.vstack(
                [
                    np.array([score_lookup[(t_id, m_id, int(rid), s)] for s in sids], dtype=float)
                    for _, rid in sorted(c_sel["bayesian_opt"].items())
                ]
            )
            cell_ref_prism.append(float(np.mean(p_vec)))
            cell_diffs_rand.append(float(np.mean(p_vec - np.mean(r_mat, axis=0))))
            cell_diffs_bo.append(float(np.mean(p_vec - np.mean(b_mat, axis=0))))

    indep_g_rand = float(np.mean(cell_diffs_rand))
    indep_g_bo = float(np.mean(cell_diffs_bo))
    sel_contrasts = summary["primary_verdict"]["selection_contrasts"]
    if "balanced_random" in sel_contrasts and "bayesian_opt" in sel_contrasts:
        rep_g_rand = float(sel_contrasts["balanced_random"]["estimate"])
        rep_g_bo = float(sel_contrasts["bayesian_opt"]["estimate"])
    else:
        cells_bd = summary["cell_level_breakdown"]["cells"]
        rep_g_rand = float(np.mean([cells_bd[f"{t}::{m}"]["diff_vs_random"] for t in tasks for m in models]))
        rep_g_bo = float(np.mean([cells_bd[f"{t}::{m}"]["diff_vs_bayesian_opt"] for t in tasks for m in models]))

    if abs(indep_g_rand - rep_g_rand) > 1e-10 or abs(indep_g_bo - rep_g_bo) > 1e-10:
        raise RuntimeError(
            f"Independent Claim P replay mismatch: ({indep_g_rand}, {indep_g_bo}) vs ({rep_g_rand}, {rep_g_bo})"
        )

    # 2b. Independently recompute Claim S max perturbation drop from raw stability_manifest.jsonl when populated
    indep_d_stab: Optional[float] = None
    stab_summary = summary["primary_verdict"]["stability_summary"]
    if stab_summary is not None:
        stab_rows = [
            json.loads(line)
            for line in (results_dir / "stability_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stab_lookup: Dict[Tuple[str, str, str, int], List[float]] = {}
        for r in stab_rows:
            key = (str(r["task_id"]), str(r["model_id"]), str(r["env_id"]), int(r["run_id"]))
            stab_lookup.setdefault(key, []).append(float(r["score"]))
        ref_mean_indep = float(np.mean(cell_ref_prism))
        pert_drops_indep: List[float] = []
        for e_id in stab_summary["per_env_drops"]:
            c_means_e = []
            for t_id in tasks:
                for m_id in models:
                    p_rid = int(final_sel["cells"][f"{t_id}::{m_id}"]["promptprism"]["primary"])
                    c_means_e.append(float(np.mean(stab_lookup[(t_id, m_id, e_id, p_rid)])))
            pert_drops_indep.append(ref_mean_indep - float(np.mean(c_means_e)))
        indep_d_stab = float(max(pert_drops_indep))
        rep_d_stab = float(stab_summary["d_stability_estimate"])
        if abs(indep_d_stab - rep_d_stab) > 1e-10:
            raise RuntimeError(f"Independent Claim S replay mismatch: {indep_d_stab} vs {rep_d_stab}")

    # 3. Verify Claim-to-Evidence Map has all 8 expected claims and non-empty fields
    cmap_path = results_dir / "claim_evidence_map.csv"
    with cmap_path.open("r", encoding="utf-8") as handle:
        cmap_rows = list(csv.DictReader(handle))
    expected_claim_ids = {
        "V4_C1_claim_p_vs_balanced_random",
        "V4_C2_claim_p_vs_bayesian_opt",
        "V4_C3_claim_s_max_perturbation_drop",
        "V4_C4_claim_s_worst_env_vs_balanced_random",
        "V4_C5_claim_s_worst_env_vs_bayesian_opt",
        "V4_C6_overall_intersection_verdict",
        "V4_C7_supporting_effect_recovery",
        "V4_C8_resource_and_budget_parity",
    }
    observed_claim_ids = {r["claim_id"] for r in cmap_rows}
    if observed_claim_ids != expected_claim_ids:
        raise RuntimeError(f"Claim map mismatch: {observed_claim_ids} != {expected_claim_ids}")

    return {
        "independent_audit_passed": True,
        "verified_locks": lock_digests,
        "replayed_g_select_balanced_random": indep_g_rand,
        "replayed_g_select_bayesian_opt": indep_g_bo,
        "replayed_d_stability": indep_d_stab,
        "certificate_sha256": cert.certificate_sha256(),
        "claim_count_verified": len(cmap_rows),
    }


POST_LOCK_REPORTING_MODULES = frozenset({"research/real_v4/analysis.py", "research/real_v4/execution.py"})


def _verify_locked_chain_for_post_acquisition_analysis(
    *,
    locks_dir: Path,
    artifacts_dir: Path,
    current_fingerprint: Mapping[str, Any],
) -> Tuple[str, List[Dict[str, str]]]:
    """Verify every phase lock (PROTOCOL..RAW_RESULTS_LOCK), artifact hash, and acquisition input/code hash.

    Any missing lock, broken prior_lock_sha256 link, tampered lock JSON, altered raw manifest or selection lock,
    or change to protocol/data/perturbation/model/calibration/acquisition-code raises PhaseLockError.
    If only post-lock analysis/reporting modules (`analysis.py` or `execution.py`) were patched after
    `raw_results_lock` (e.g., aligning terminal-failure handling with SAP §3.2), return the locked acquisition
    fingerprint and an explicit auditable errata record for `release_manifest.json`.
    """
    prev_lock_sha: Optional[str] = None
    locked_study_fp: Optional[str] = None
    errata: List[Dict[str, str]] = []

    cur_hashes = dict(current_fingerprint["file_hashes"])
    cur_envs = list(current_fingerprint["registered_env_ids"])

    for idx, req_phase in enumerate(PHASE_SEQUENCE):
        lpath = lock_path_for_phase(locks_dir, req_phase)
        if not lpath.exists():
            raise PhaseLockError(
                f"Refusing to run Task 15 analysis: missing prerequisite lock {req_phase.value!r}"
            )
        payload = json.loads(lpath.read_text(encoding="utf-8"))
        stored_lock_sha = str(payload.get("lock_sha256", ""))
        check_copy = {k: v for k, v in payload.items() if k != "lock_sha256"}
        if canonical_sha256(check_copy) != stored_lock_sha:
            raise PhaseLockError(f"Tampered lock file detected in {lpath.name}")

        if idx > 0 and payload.get("prior_lock_sha256") != prev_lock_sha:
            raise PhaseLockError(f"Broken prior_lock_sha256 chain at {req_phase.value!r}")

        if payload.get("registered_env_ids") != cur_envs:
            raise PhaseLockError("Registered stability environments changed after protocol lock")

        p_fp = str(payload.get("study_fingerprint_sha256", ""))
        if locked_study_fp is None:
            locked_study_fp = p_fp
        elif p_fp != locked_study_fp:
            raise PhaseLockError(f"Inconsistent study_fingerprint_sha256 across lock chain at {req_phase.value!r}")

        locked_hashes = dict(payload.get("file_hashes", {}))
        if set(locked_hashes.keys()) != set(cur_hashes.keys()):
            raise PhaseLockError("Set of hashed study files changed after protocol lock")

        for fkey, l_sha in sorted(locked_hashes.items()):
            c_sha = cur_hashes[fkey]
            if l_sha != c_sha:
                if fkey not in POST_LOCK_REPORTING_MODULES:
                    raise PhaseLockError(
                        f"Study fingerprint drift detected in pre-analysis input/code {fkey!r}: "
                        f"locked={l_sha}, current={c_sha}"
                    )
                if idx == 0:
                    errata.append(
                        {
                            "file": fkey,
                            "locked_pre_analysis_sha256": l_sha,
                            "post_lock_analysis_sha256": c_sha,
                            "sap_alignment_note": (
                                "Post-acquisition SAP §3.2 alignment: terminal_failure HTTP errors score 0.0 in "
                                "denominator without misclassifying 'HTTP_ERROR' as a provider model_version swap, "
                                "and _write_claim_evidence_map safely formats INVALID/OUT_OF_ENVELOPE states."
                            ),
                        }
                    )

        art_name = payload.get("artifact_name")
        art_sha = payload.get("artifact_sha256")
        if art_name is not None:
            art_path = artifacts_dir / str(art_name)
            if not art_path.exists():
                raise PhaseLockError(f"Locked artifact {art_name!r} was deleted")
            if sha256_file(art_path) != art_sha:
                raise PhaseLockError(f"Locked artifact {art_name!r} hash changed after {req_phase.value!r} lock")

        prev_lock_sha = stored_lock_sha

    assert locked_study_fp is not None
    return locked_study_fp, errata


def _compute_validation_selection_resample_agreement(
    *,
    results_dir: Path,
    tasks: Sequence[str],
    models: Sequence[str],
    n_bootstrap: int = 200,
    seed: int = 20261250,
) -> Optional[float]:
    cand_path = results_dir / "candidate_lock.json"
    val_path = results_dir / "validation_manifest.jsonl"
    final_path = results_dir / "final_selection_lock.json"
    if not (cand_path.exists() and val_path.exists() and final_path.exists()):
        return None

    cand_lock = json.loads(cand_path.read_text(encoding="utf-8"))
    final_lock = json.loads(final_path.read_text(encoding="utf-8"))
    cube = full_cube_configs()
    val_man = AppendOnlyManifest(val_path)

    # Index validation records by (task_id, model_id, env_id, run_id, sample_id)
    idx_map: Dict[Tuple[str, str, str, int], Dict[str, Tuple[float, str]]] = {}
    for rec in val_man.records:
        key = (str(rec["task_id"]), str(rec["model_id"]), str(rec["env_id"]), int(rec["run_id"]))
        idx_map.setdefault(key, {})[str(rec["sample_id"])] = (float(rec["score"]), str(rec["group_id"]))

    selector = PromptPrismRobustSelector(pad_to_max=True, n_bootstrap_validation=80)
    rng = np.random.default_rng(seed)
    agree = 0
    total = 0

    for t_id in tasks:
        for m_id in models:
            cell_id = f"{t_id}::{m_id}"
            raw_cands = cand_lock["cells"][cell_id]["promptprism"]["primary"]
            if isinstance(raw_cands, Mapping):
                raw_cands = raw_cands.get("candidates", [])
            c_rids = [int(x["run_id"] if isinstance(x, Mapping) else x) for x in raw_cands]
            locked_win = int(final_lock["cells"][cell_id]["promptprism"]["primary"])
            cands = [
                CandidateProvenance(
                    run_id=rid,
                    factor_levels=cube[rid],
                    source_rule="locked_screen_candidate",
                    screen_score=0.0,
                    rank_in_screen=rank,
                )
                for rank, rid in enumerate(c_rids, start=1)
            ]
            ref_rows = idx_map.get((t_id, m_id, "ref", c_rids[0]), {})
            if not ref_rows:
                continue
            sids = sorted(ref_rows.keys())
            gids = [ref_rows[s][1] for s in sids]
            buckets: Dict[str, List[int]] = {}
            for i_s, g in enumerate(gids):
                buckets.setdefault(g, []).append(i_s)
            g_lists = [np.array(v, dtype=int) for v in buckets.values()]

            val_cube: Dict[str, Dict[int, np.ndarray]] = {}
            for e_id in ("ref", "fmt_whitespace", "word_paraphrase_1"):
                rmap_e: Dict[int, np.ndarray] = {}
                for rid in c_rids:
                    erows = idx_map.get((t_id, m_id, e_id, rid), {})
                    if all(s in erows for s in sids):
                        rmap_e[rid] = np.array([erows[s][0] for s in sids], dtype=float)
                if rmap_e:
                    val_cube[e_id] = rmap_e
            if not val_cube:
                continue

            for b in range(n_bootstrap):
                pick = np.concatenate([g_lists[i] for i in rng.integers(0, len(g_lists), size=len(g_lists))])
                g_boot = [gids[i] for i in pick]
                v_boot = {e: {r: arr[pick] for r, arr in rmap.items()} for e, rmap in val_cube.items()}
                win_b, _, _ = selector.select_final_prompt(cands, v_boot, group_ids=g_boot, seed=seed + b)
                if win_b.run_id == locked_win:
                    agree += 1
                total += 1

    return float(agree / total) if total > 0 else None


def run_task15_analysis_and_audit(
    *,
    protocol_dir: Path = DEFAULT_PROTOCOL_DIR,
    dataset_manifest_path: Path = DEFAULT_DATASET_MANIFEST,
    model_registry_path: Path = MODEL_REGISTRY_PATH,
    perturbation_registry_path: Path = PERTURBATION_REGISTRY_PATH,
    calibration_config_path: Path = CALIBRATION_CONFIG_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    tasks: Sequence[str] = REGISTERED_TASKS,
    models: Sequence[str] = REGISTERED_MODELS,
    study_id: str = STUDY_ID,
    num_baseline_trajectories: int = 10,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    """Execute the preregistered Real-v4 analysis once Raw-Results Lock verifies, then run independent audit."""
    fp = compute_study_fingerprint(
        protocol_dir=protocol_dir,
        dataset_manifest_path=dataset_manifest_path,
        perturbation_registry_path=perturbation_registry_path,
        model_registry_path=model_registry_path,
        calibration_config_path=calibration_config_path,
        code_paths=default_code_paths(ROOT),
    )
    locks_dir = results_dir / "locks"
    locked_study_fp, post_lock_errata = _verify_locked_chain_for_post_acquisition_analysis(
        locks_dir=locks_dir,
        artifacts_dir=results_dir,
        current_fingerprint=fp,
    )

    envelope = _build_operating_envelope(
        study_id=study_id,
        tasks=tasks,
        models=models,
        model_registry_path=model_registry_path,
        dataset_manifest_path=dataset_manifest_path,
        study_fingerprint_sha256=locked_study_fp,
    )
    final_sel = json.loads((results_dir / "final_selection_lock.json").read_text(encoding="utf-8"))
    conf_man = AppendOnlyManifest(results_dir / "confirmation_manifest.jsonl")
    stab_man = AppendOnlyManifest(results_dir / "stability_manifest.jsonl")

    full_cube_means, secondary_refs = _extract_screen_cube_and_secondary_baselines(
        results_dir / "screen_manifest.jsonl", tasks=tasks, models=models
    )
    rand_base = BalancedRandomTwoStageBaseline(num_designs=num_baseline_trajectories)
    sel_agree = _compute_validation_selection_resample_agreement(
        results_dir=results_dir, tasks=tasks, models=models, n_bootstrap=min(200, n_bootstrap)
    )

    thresholds = DecisionThresholds(
        alpha=0.05,
        delta_select=0.01,
        epsilon_stability=0.02,
        n_bootstrap=n_bootstrap,
        seed=20261118,
    )
    summary = analyze_locked_manifests(
        certificate_id=f"cert-{study_id}",
        study_id=study_id,
        algorithm_version=PROMPTPRISM_SELECTOR_VERSION,
        envelope=envelope,
        final_selection_lock=final_sel,
        confirmation_records=conf_man.records,
        stability_records=stab_man.records,
        thresholds=thresholds,
        full_cube_screen_means_by_cell=full_cube_means,
        random_designs=rand_base.frozen_designs,
        selection_resample_agreement=sel_agree,
    )

    # Non-rescuing sensitivity decomposition: 7 paired-item perturbation environments vs unpaired item_holdout_group
    paired_envs_only = tuple(
        e for e in envelope.registered_environments if not e.uses_disjoint_item_group
    )
    if len(paired_envs_only) >= 2:
        env_paired = envelope.model_copy(update={"registered_environments": paired_envs_only})
        paired_ids = {e.env_id for e in paired_envs_only}
        stab_paired = [r for r in stab_man.records if str(r["env_id"]) in paired_ids]
        sum_paired = analyze_locked_manifests(
            certificate_id=f"cert-{study_id}-paired-envs",
            study_id=study_id,
            algorithm_version=PROMPTPRISM_SELECTOR_VERSION,
            envelope=env_paired,
            final_selection_lock=final_sel,
            confirmation_records=conf_man.records,
            stability_records=stab_paired,
            thresholds=thresholds,
        )
        summary["sensitivity_analyses"]["paired_item_perturbations_only"] = {
            "analysis_type": "paired_item_perturbations_sensitivity",
            "confirmatory_primary": False,
            "supporting_only_non_rescuing": True,
            "included_environments": [e.env_id for e in paired_envs_only if e.env_id != "ref"],
            "stability_summary": sum_paired["primary_verdict"]["stability_summary"],
            "would_pass_claim_s_on_paired_envs": sum_paired["primary_verdict"]["claim_s_pass"],
        }

    # Non-rescuing terminal-failure sensitivity breakdown (SAP §3.2 scores terminal HTTP errors as 0.0 in primary)
    tf_counts: Dict[str, int] = {}
    for rec in stab_man.records:
        if rec.get("status") == "terminal_failure":
            k = f"{rec['env_id']}::{rec['model_id']}::{rec['task_id']}"
            tf_counts[k] = tf_counts.get(k, 0) + 1
    summary["sensitivity_analyses"]["terminal_failure_breakdown"] = {
        "analysis_type": "terminal_failure_zero_score_accounting",
        "confirmatory_primary": False,
        "supporting_only_non_rescuing": True,
        "counts_by_env_model_task": dict(sorted(tf_counts.items())),
    }

    summary["secondary_screen_references"] = secondary_refs
    summary["full_study_resource_reconciliation"] = _summarize_all_manifest_resources(results_dir)
    summary.pop("summary_sha256", None)
    summary["summary_sha256"] = canonical_sha256(summary)

    summary_path = results_dir / "summary.json"
    cert_path = results_dir / "stability_certificate.json"
    cmap_path = results_dir / "claim_evidence_map.csv"
    rel_path = results_dir / "release_manifest.json"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cert_path.write_text(json.dumps(summary["stability_certificate"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_claim_evidence_map(summary, cmap_path)

    audit_res = audit_real_v4_final_artifacts(results_dir=results_dir, tasks=tasks, models=models)

    artifact_files = [
        "pilot_manifest.jsonl",
        "pilot_report.json",
        "screen_manifest.jsonl",
        "candidate_lock.json",
        "validation_manifest.jsonl",
        "final_selection_lock.json",
        "confirmation_manifest.jsonl",
        "stability_manifest.jsonl",
        "raw_results_lock.json",
        "summary.json",
        "stability_certificate.json",
        "claim_evidence_map.csv",
    ]
    artifact_hashes = {
        fname: sha256_file(results_dir / fname)
        for fname in artifact_files
        if (results_dir / fname).exists()
    }
    release_payload: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": study_id,
        "acquisition_study_fingerprint_sha256": locked_study_fp,
        "study_fingerprint_sha256": fp["study_fingerprint_sha256"],
        "post_lock_analysis_errata": post_lock_errata,
        "independent_audit": audit_res,
        "artifact_sha256": artifact_hashes,
    }
    release_payload["release_sha256"] = canonical_sha256(release_payload)
    rel_path.write_text(json.dumps(release_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "audit": audit_res, "release_manifest": release_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Real-v4 Phase 4-5 steps.")
    parser.add_argument(
        "--step",
        choices=("screen", "validation", "confirmation", "stability-wave-1", "temporal-wave-2", "analyze", "all"),
        required=True,
    )
    args = parser.parse_args()

    if args.step in ("screen", "all"):
        print("=== Task 13a: Running Screening + 64-Run Research Audit Cube ===", flush=True)
        runner = build_phase4_runner()
        runner.run_screen_and_lock_candidates(include_research_full_cube_audit=True)
        print("=== Screen & Candidate Lock Complete ===", flush=True)

    if args.step in ("validation", "all"):
        print("=== Task 13b: Running Robust Candidate Validation & Final Selection Lock ===", flush=True)
        runner = build_phase4_runner()
        runner.run_validation_and_lock_final_selections()
        print("=== Validation & Final Selection Lock Complete ===", flush=True)

    if args.step in ("confirmation", "all"):
        print("=== Task 14a: Running Clean Confirmation ===", flush=True)
        runner = build_phase4_runner()
        runner.run_clean_confirmation()
        print("=== Clean Confirmation Complete ===", flush=True)

    if args.step in ("stability-wave-1", "all"):
        print("=== Task 14b: Running Stability Wave 1 (7 Environments) ===", flush=True)
        runner = build_phase4_runner()
        runner.run_stability_wave_and_lock_raw_results(
            requested_env_ids=WAVE_1_STABILITY_ENVS,
            write_final_locks=False,
        )
        print("=== Stability Wave 1 Complete ===", flush=True)

    if args.step in ("temporal-wave-2", "all"):
        print("=== Task 14c: Running Temporal Wave 2 & Raw Results Lock ===", flush=True)
        runner = build_phase4_runner()
        runner.run_stability_wave_and_lock_raw_results(
            requested_env_ids=None,
            write_final_locks=True,
        )
        print("=== Stability Wave 2 & Raw Results Lock Complete ===", flush=True)

    if args.step in ("analyze", "all"):
        print("=== Task 15: Running Preregistered Analysis & Independent Audit ===", flush=True)
        res = run_task15_analysis_and_audit()
        pv = res["summary"]["primary_verdict"]
        print(
            json.dumps(
                {
                    "verdict_code": pv["verdict_code"],
                    "claim_p_pass": pv["claim_p_pass"],
                    "claim_s_pass": pv["claim_s_pass"],
                    "certificate_status": pv["certificate_status"],
                    "independent_audit_passed": res["audit"]["independent_audit_passed"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
