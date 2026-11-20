"""Task 12: Cross-provider quarantined pilot execution, repeatability policy freeze, and go/no-go report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv
import numpy as np

from research.data.real_v4.build import PARTITION_SIZES, V4_SPLITS_DIR
from research.real_v4.locking import default_code_paths
from research.real_v4.manifest import AppendOnlyManifest, deterministic_call_key
from research.real_v4.model_provenance import (
    MODEL_REGISTRY_PATH,
    evaluate_pilot_repeatability,
)
from research.real_v4.perturbations import PERTURBATION_REGISTRY_PATH, REGISTERED_TASKS
from research.real_v4.providers import (
    BaseRealV4Adapter,
    RealV4GeminiAdapter,
    RealV4LocalMistralMLXAdapter,
    RealV4LocalQwenMLXAdapter,
)
from research.real_v4.runner import ResumableRealV4Runner, RunnerPaths
from research.real_v4.schemas import CURRENT_SCHEMA_VERSION, canonical_sha256
from research.real_v4.simulation import CALIBRATION_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "prompt-prism-real-v4-selection-stability-2026"
DEFAULT_PROTOCOL_DIR = ROOT / "research" / "protocol" / "real_v4"
DEFAULT_DATASET_MANIFEST = ROOT / "research" / "data" / "real_v4" / "dataset_manifest.json"
DEFAULT_RESULTS_DIR = ROOT / "research" / "results" / "real_v4"

REGISTERED_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen3-8B-4bit",
)

PARTITIONS: Tuple[str, ...] = tuple(PARTITION_SIZES.keys())
SPLITS_DIR: Path = V4_SPLITS_DIR

PILOT_GATES: Dict[str, float] = {
    "min_completion_rate": 0.95,
    "max_truncation_rate": 0.10,
    "min_nonempty_response_rate": 0.95,
}


def load_real_v4_splits(
    splits_dir: Path = SPLITS_DIR,
    tasks: Sequence[str] = REGISTERED_TASKS,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    splits: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for task_id in tasks:
        for part in PARTITIONS:
            fpath = splits_dir / f"{task_id}_{part}.jsonl"
            rows = [
                json.loads(line)
                for line in fpath.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            splits[(task_id, part)] = rows
    return splits


def audit_pilot_quarantine(
    splits: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    tasks: Sequence[str] = REGISTERED_TASKS,
) -> Dict[str, Any]:
    """Verify pilot sample_ids and group_ids have zero overlap with screen/validation/confirmation splits."""
    downstream_parts = tuple(p for p in PARTITIONS if p != "pilot")
    leaks: List[str] = []
    per_task: Dict[str, Dict[str, Any]] = {}

    for task_id in tasks:
        pilot_rows = splits[(task_id, "pilot")]
        pilot_samples = {str(r["sample_id"]) for r in pilot_rows}
        pilot_groups = {str(r["group_id"]) for r in pilot_rows}
        down_samples = set()
        down_groups = set()
        for part in downstream_parts:
            for r in splits[(task_id, part)]:
                down_samples.add(str(r["sample_id"]))
                down_groups.add(str(r["group_id"]))

        s_overlap = sorted(pilot_samples & down_samples)
        g_overlap = sorted(pilot_groups & down_groups)
        if s_overlap or g_overlap:
            leaks.append(f"{task_id}: sample_overlap={s_overlap}, group_overlap={g_overlap}")

        per_task[task_id] = {
            "pilot_items": len(pilot_rows),
            "pilot_unique_groups": len(pilot_groups),
            "downstream_items": len(down_samples),
            "sample_overlap_count": len(s_overlap),
            "group_overlap_count": len(g_overlap),
        }

    if leaks:
        raise RuntimeError(f"Pilot quarantine breach detected: {'; '.join(leaks)}")

    return {
        "quarantine_verified": True,
        "downstream_partitions_checked": list(downstream_parts),
        "per_task": per_task,
    }


def build_default_live_adapters() -> Dict[str, BaseRealV4Adapter]:
    load_dotenv(ROOT / ".env")
    return {
        "gemini-2.5-flash": RealV4GeminiAdapter(),
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit": RealV4LocalMistralMLXAdapter(),
        "mlx-community/Qwen3-8B-4bit": RealV4LocalQwenMLXAdapter(),
    }


def build_pilot_report(
    *,
    manifest_path: Path,
    splits: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    tasks: Sequence[str] = REGISTERED_TASKS,
    models: Sequence[str] = REGISTERED_MODELS,
    pilot_repeats: int = 3,
    pilot_run_id: int = 12,
    study_id: str = STUDY_ID,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit `pilot_manifest.jsonl`, freeze repeatability policy, and emit `pilot_report.json`."""
    quarantine = audit_pilot_quarantine(splits, tasks=tasks)
    manifest = AppendOnlyManifest(manifest_path)

    expected_keys: List[str] = []
    for task_id in tasks:
        for model_id in models:
            for rep_idx in range(pilot_repeats):
                for item in splits[(task_id, "pilot")]:
                    expected_keys.append(
                        deterministic_call_key(
                            study_id=study_id,
                            phase="pilot",
                            task_id=task_id,
                            model_id=model_id,
                            env_id="ref",
                            run_id=pilot_run_id,
                            sample_id=str(item["sample_id"]),
                            repeat_idx=rep_idx,
                        )
                    )

    replay_audit = manifest.replay_audit(expected_keys)

    cell_reports: Dict[str, Dict[str, Any]] = {}
    model_repeatability: Dict[str, Dict[str, Any]] = {}
    all_cells_pass = True

    for model_id in models:
        model_pass_outputs: List[List[str]] = [[] for _ in range(pilot_repeats)]
        model_pass_scores: List[List[float]] = [[] for _ in range(pilot_repeats)]
        model_latencies: List[float] = []
        model_tok_in: List[int] = []
        model_tok_out: List[int] = []
        model_costs: List[float] = []
        returned_versions = set()
        within_env_all = True
        missing_prov_ids = 0

        for task_id in tasks:
            cell_id = f"{task_id}::{model_id}"
            items = splits[(task_id, "pilot")]
            pass_outputs: List[List[str]] = []
            pass_scores: List[List[float]] = []
            ok_count = 0
            trunc_count = 0
            nonempty_count = 0
            total_calls = 0
            cell_costs = 0.0
            cell_latencies: List[float] = []

            for rep_idx in range(pilot_repeats):
                r_texts: List[str] = []
                r_scores: List[float] = []
                for item in items:
                    ck = deterministic_call_key(
                        study_id=study_id,
                        phase="pilot",
                        task_id=task_id,
                        model_id=model_id,
                        env_id="ref",
                        run_id=pilot_run_id,
                        sample_id=str(item["sample_id"]),
                        repeat_idx=rep_idx,
                    )
                    rec = manifest.get_record(ck)
                    total_calls += 1
                    prov = rec["provenance"]
                    raw_text = str(rec["raw_text"])
                    score = float(rec["score"])
                    status_ok = rec["status"] == "ok"
                    f_reason = str(prov.get("finish_reason", "")).upper()
                    is_trunc = f_reason in {"MAX_TOKENS", "LENGTH"}

                    if status_ok:
                        ok_count += 1
                    if is_trunc:
                        trunc_count += 1
                    if raw_text.strip():
                        nonempty_count += 1
                    if not str(rec.get("response_id", "")).strip() or not str(rec.get("returned_model_version", "")).strip():
                        missing_prov_ids += 1
                    if not bool(rec.get("within_pinned_envelope", False)):
                        within_env_all = False

                    returned_versions.add(str(rec.get("returned_model_version", "")))
                    lat = float(prov.get("latency_ms", 0.0))
                    cell_latencies.append(lat)
                    model_latencies.append(lat)
                    model_tok_in.append(int(prov.get("tokens_input", 0)))
                    model_tok_out.append(int(prov.get("tokens_output", 0)))
                    c_usd = float(prov.get("cost_usd", 0.0))
                    cell_costs += c_usd
                    model_costs.append(c_usd)

                    r_texts.append(raw_text)
                    r_scores.append(score)
                    model_pass_outputs[rep_idx].append(raw_text)
                    model_pass_scores[rep_idx].append(score)

                pass_outputs.append(r_texts)
                pass_scores.append(r_scores)

            rep_dec = evaluate_pilot_repeatability(pass_outputs, pass_scores)
            comp_rate = ok_count / max(1, total_calls)
            trunc_rate = trunc_count / max(1, total_calls)
            nonempty_rate = nonempty_count / max(1, total_calls)
            cell_pass = (
                comp_rate >= PILOT_GATES["min_completion_rate"]
                and trunc_rate <= PILOT_GATES["max_truncation_rate"]
                and nonempty_rate >= PILOT_GATES["min_nonempty_response_rate"]
                and within_env_all
                and missing_prov_ids == 0
            )
            if not cell_pass:
                all_cells_pass = False

            cell_reports[cell_id] = {
                "task_id": task_id,
                "model_id": model_id,
                "total_calls": total_calls,
                "completion_rate": comp_rate,
                "truncation_rate": trunc_rate,
                "nonempty_response_rate": nonempty_rate,
                "mean_primary_score": float(np.mean(pass_scores)),
                "mean_latency_ms": float(np.mean(cell_latencies)),
                "total_cost_usd": round(cell_costs, 6),
                "repeatability": {
                    "exact_output_agreement": rep_dec.exact_output_agreement,
                    "exact_score_agreement": rep_dec.exact_score_agreement,
                    "score_std_across_repeats": rep_dec.score_std_across_repeats,
                    "nondeterminism_detected": rep_dec.nondeterminism_detected,
                    "selected_repeat_count": rep_dec.frozen_repeat_count,
                    "aggregation_rule": rep_dec.aggregation_rule,
                },
                "cell_gate_pass": cell_pass,
            }

        mod_rep = evaluate_pilot_repeatability(model_pass_outputs, model_pass_scores)
        mean_cost_per_call = float(np.mean(model_costs)) if model_costs else 0.0
        mean_lat_s = float(np.mean(model_latencies) / 1000.0) if model_latencies else 0.0
        # Full study upper bound per model across 6 tasks (up to 37,260 unique calls/cell if 64-run audit + all trajectories disjoint)
        est_phase4_calls_typical = 6 * 12_500  # ~12.5k unique cached calls/task-model cell across union of designs
        est_phase4_calls_worst = 6 * 37_260

        model_repeatability[model_id] = {
            "observed_returned_versions": sorted(returned_versions),
            "within_pinned_envelope": within_env_all,
            "missing_provenance_ids": missing_prov_ids,
            "exact_output_agreement": mod_rep.exact_output_agreement,
            "exact_score_agreement": mod_rep.exact_score_agreement,
            "score_std_across_repeats": mod_rep.score_std_across_repeats,
            "nondeterminism_detected": mod_rep.nondeterminism_detected,
            "frozen_repeat_count": mod_rep.frozen_repeat_count,
            "aggregation_rule": mod_rep.aggregation_rule,
            "mean_tokens_input": float(np.mean(model_tok_in)) if model_tok_in else 0.0,
            "mean_tokens_output": float(np.mean(model_tok_out)) if model_tok_out else 0.0,
            "mean_latency_ms": float(np.mean(model_latencies)) if model_latencies else 0.0,
            "pilot_spend_usd": round(float(np.sum(model_costs)), 6),
            "projected_phase4_typical_usd": round(mean_cost_per_call * est_phase4_calls_typical, 2),
            "projected_phase4_worst_case_usd": round(mean_cost_per_call * est_phase4_calls_worst, 2),
            "projected_sequential_wall_hours_typical": round((mean_lat_s * est_phase4_calls_typical) / 3600.0, 2),
        }

    report: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": study_id,
        "pilot_run_id": int(pilot_run_id),
        "pilot_repeats": int(pilot_repeats),
        "claim_margins_unchanged": {
            "delta_select": 0.01,
            "epsilon_stability": 0.02,
            "methods_modified": False,
        },
        "registered_pilot_gates": PILOT_GATES,
        "pilot_quarantine_audit": quarantine,
        "manifest_replay_audit": replay_audit,
        "model_execution_policy_and_projections": model_repeatability,
        "cell_reports": cell_reports,
        "all_pilot_gates_passed": bool(all_cells_pass and quarantine["quarantine_verified"]),
        "human_go_no_go_status": "READY_FOR_HUMAN_APPROVAL" if all_cells_pass else "BLOCKED_SEE_CELL_REPORTS",
    }
    report["report_sha256"] = canonical_sha256(report)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_task12_pilot_cli(
    *,
    models: Sequence[str] = REGISTERED_MODELS,
    pilot_repeats: int = 3,
    pilot_run_id: int = 12,
    max_new_calls: Optional[int] = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    splits = load_real_v4_splits()
    audit_pilot_quarantine(splits)
    adapters_all = build_default_live_adapters()
    adapters = {m: adapters_all[m] for m in models}

    paths = RunnerPaths(
        protocol_dir=DEFAULT_PROTOCOL_DIR,
        dataset_manifest_path=DEFAULT_DATASET_MANIFEST,
        perturbation_registry_path=PERTURBATION_REGISTRY_PATH,
        model_registry_path=MODEL_REGISTRY_PATH,
        calibration_config_path=CALIBRATION_CONFIG_PATH,
        code_paths=default_code_paths(ROOT),
        results_dir=results_dir,
    )
    runner = ResumableRealV4Runner(
        study_id=STUDY_ID,
        paths=paths,
        adapters=adapters,
        splits_by_task_and_partition=splits,
        tasks=REGISTERED_TASKS,
        models=models,
    )
    if not (paths.locks_dir / "protocol.lock.json").exists():
        runner.freeze_protocol()

    runner.run_pilot(
        pilot_repeats=pilot_repeats,
        pilot_run_id=pilot_run_id,
        max_new_calls=max_new_calls,
    )
    return build_pilot_report(
        manifest_path=results_dir / "pilot_manifest.jsonl",
        splits=splits,
        tasks=REGISTERED_TASKS,
        models=models,
        pilot_repeats=pilot_repeats,
        pilot_run_id=pilot_run_id,
        study_id=STUDY_ID,
        output_path=results_dir / "pilot_report.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute Real-v4 Task 12 quarantined cross-provider pilot.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(REGISTERED_MODELS),
        help="Subset or full list of registered models to run.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run-id", type=int, default=12)
    parser.add_argument("--max-new-calls", type=int, default=None)
    args = parser.parse_args()

    rep = run_task12_pilot_cli(
        models=tuple(args.models),
        pilot_repeats=args.repeats,
        pilot_run_id=args.run_id,
        max_new_calls=args.max_new_calls,
    )
    print(json.dumps({"all_pilot_gates_passed": rep["all_pilot_gates_passed"], "status": rep["human_go_no_go_status"]}, indent=2))
