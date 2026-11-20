"""Checkpoint C preregistration bundle and worst-case cost/runtime estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from research.data.real_v4.build import AUDIT_REPORT_PATH, MANIFEST_PATH as DATASET_MANIFEST_PATH
from research.real_v4.locking import (
    ROOT,
    compute_study_fingerprint,
    default_code_paths,
    sha256_file,
)
from research.real_v4.model_provenance import MODEL_REGISTRY_PATH
from research.real_v4.perturbations import PERTURBATION_REGISTRY_PATH
from research.real_v4.schemas import CURRENT_SCHEMA_VERSION, canonical_sha256
from research.real_v4.simulation import (
    CALIBRATION_CONFIG_PATH,
    CALIBRATION_SUMMARY_PATH,
)

REAL_V4_PROTOCOL_DIR = ROOT / "research" / "protocol" / "real_v4"
OSF_BUNDLE_PATH = REAL_V4_PROTOCOL_DIR / "osf_registration_bundle.json"


def estimate_worst_case_resources() -> Dict[str, Any]:
    """Compute conservative worst-case hosted spend and local MLX runtime across all phases."""
    # Upper-bound unique configurations per cell (M_traj=10):
    # Pilot: 1 config x 20 items x 3 repeats = 60 calls
    # Screen: 64 unique cube configs x 120 items = 7,680 calls (covers all methods + 64-run oracle audit!)
    # Validation: at most 20 unique candidate union configs x 60 items x 3 val envs = 3,600 calls
    # Clean + Stability Confirmation: at most 8 unique final winners x (240 * 9) = 17,280 calls
    calls_per_cell_upper = 60 + 7680 + 3600 + 17280  # 28,620 worst-case unique calls per task-model cell
    n_tasks = 6

    avg_tok_in = 260
    avg_tok_out = 60

    def _hosted_cost(calls: int, in_per_m: float, out_per_m: float) -> float:
        return round(
            calls * (avg_tok_in * in_per_m + avg_tok_out * out_per_m) / 1_000_000.0,
            2,
        )

    gemini_calls = n_tasks * calls_per_cell_upper
    local_mlx_calls_per_model = n_tasks * calls_per_cell_upper

    gemini_usd = _hosted_cost(gemini_calls, 0.30, 2.50)

    # Single-method operational deployment cost (16x120 screen + 4x60x3 val + 1x240 ref = 2,880 calls/cell)
    single_method_calls_per_cell = 16 * 120 + 4 * 60 * 3 + 240  # 2,880

    # Local MLX batch_generate (batch=32) on Apple M4 Max at ~0.07s-0.12s/prompt average across the 6 tasks
    mlx_wall_hours_two_models = round((2 * local_mlx_calls_per_model * 0.095) / 3600.0, 2)

    return {
        "assumptions": {
            "avg_tokens_input": avg_tok_in,
            "avg_tokens_output_worst_case": avg_tok_out,
            "worst_case_unique_calls_per_cell_including_64run_audit": calls_per_cell_upper,
            "single_method_operational_calls_per_cell": single_method_calls_per_cell,
        },
        "hosted_spend_worst_case_usd": {
            "gemini_2_5_flash": gemini_usd,
            "local_mlx_models": 0.0,
            "total_hosted_worst_case_usd": gemini_usd,
        },
        "local_compute_worst_case": {
            "models": [
                "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "mlx-community/Qwen3-8B-4bit",
            ],
            "total_calls_upper_bound_both_models": 2 * local_mlx_calls_per_model,
            "estimated_wall_hours": mlx_wall_hours_two_models,
            "estimated_peak_memory_gb": 9.2,
        },
    }


def build_osf_registration_bundle(
    output_path: Path = OSF_BUNDLE_PATH,
    *,
    external_registration_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble deterministic Checkpoint-C preregistration package."""
    fp = compute_study_fingerprint(
        protocol_dir=REAL_V4_PROTOCOL_DIR,
        dataset_manifest_path=DATASET_MANIFEST_PATH,
        perturbation_registry_path=PERTURBATION_REGISTRY_PATH,
        model_registry_path=MODEL_REGISTRY_PATH,
        calibration_config_path=CALIBRATION_CONFIG_PATH,
        code_paths=default_code_paths(ROOT),
    )

    calib_summary = json.loads(CALIBRATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    data_audit = json.loads(AUDIT_REPORT_PATH.read_text(encoding="utf-8"))

    if not calib_summary.get("all_scientific_feasibility_gates_passed"):
        raise RuntimeError("Cannot generate registration bundle: Checkpoint B calibration gates did not pass")
    if not data_audit.get("audit_passed"):
        raise RuntimeError("Cannot generate registration bundle: Task 8 dataset audit did not pass")

    status = (
        "LOCKED_PRE_SCREENING_TIMESTAMPED"
        if external_registration_url
        else "READY_FOR_EXTERNAL_OSF_TIMESTAMP"
    )
    bundle: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": "prompt-prism-real-v4-selection-stability-2026",
        "registration_status": status,
        "prospective_outcomes_acquired": False,
        "external_registry_target": "OSF (https://osf.io) / Immutable Git Tag",
        "external_registration_url": external_registration_url,
        "public_api_promotion_policy": {
            "default_promotion_requires": ["claim_p_pass", "claim_s_pass"],
            "mixed_verdict_policy": "experimental_opt_in_only_no_stability_badge",
        },
        "study_fingerprint": fp,
        "calibration_summary_sha256": sha256_file(CALIBRATION_SUMMARY_PATH),
        "dataset_audit_sha256": sha256_file(AUDIT_REPORT_PATH),
        "worst_case_resource_budget": estimate_worst_case_resources(),
    }
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-url", type=str, default=None)
    args = parser.parse_args()
    out = build_osf_registration_bundle(external_registration_url=args.registration_url)
    print(f"OSF bundle written: sha256={out['bundle_sha256']}")
