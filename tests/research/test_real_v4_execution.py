"""End-to-end synthetic verification of Tasks 13-15 orchestrator & independent artifact audit (`execution.py`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from research.real_v4.execution import (
    WAVE_1_STABILITY_ENVS,
    _write_claim_evidence_map,
    run_task15_analysis_and_audit,
)
from research.real_v4.locking import PhaseLockError, default_code_paths
from research.real_v4.model_provenance import ResponseProvenance
from research.real_v4.perturbations import PERTURBATION_REGISTRY_PATH, REGISTERED_TASKS
from research.real_v4.providers import (
    BaseRealV4Adapter,
    CanonicalAdapterRequest,
    CanonicalAdapterResponse,
)
from research.real_v4.runner import ResumableRealV4Runner, RunnerPaths
from research.real_v4.simulation import CALIBRATION_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[2]
REAL_V4_PROTOCOL_DIR = ROOT / "research" / "protocol" / "real_v4"


def _stub_item_for_task(task_id: str, part: str, idx: int) -> Dict[str, Any]:
    base = {
        "sample_id": f"{task_id}_{part}_{idx}",
        "group_id": f"{task_id}_grp_{part}_{idx}",
    }
    if task_id == "re3d_entity_extraction":
        return {
            **base,
            "input_text": "Prime Minister Sanae Takaichi visited Tokyo.",
            "target": {
                "person": ["Sanae Takaichi"],
                "organization": [],
                "location": ["Tokyo"],
                "money": [],
                "temporal": [],
            },
        }
    if task_id == "squad2_grounded_qa":
        return {
            **base,
            "context": "Oxygen was isolated in 1774.",
            "question": "When?",
            "target": ["1774"],
        }
    if task_id == "banking77_intent":
        return {
            **base,
            "input_text": "My card was rejected.",
            "options": ["card_arrival", "declined_card_payment"],
            "target": "declined_card_payment",
        }
    if task_id == "gsm8k_math":
        return {**base, "question": "What is 40 + 2?", "target": "42"}
    if task_id == "mbpp_python":
        return {
            **base,
            "prompt": "Write add_two(x) returning x + 2.",
            "signature_hint": "def add_two(x):",
            "target": ["assert add_two(3) == 5"],
        }
    if task_id == "sciq_verification":
        return {
            **base,
            "support": "Sound cannot travel in vacuum.",
            "question": "Where cannot sound travel?",
            "options": {"A": "Water", "B": "Steel", "C": "Vacuum", "D": "Air"},
            "target": "C",
        }
    raise ValueError(task_id)


def _good_output_for_task(task_id: str) -> str:
    if task_id == "re3d_entity_extraction":
        return '{"person": ["Sanae Takaichi"], "organization": [], "location": ["Tokyo"], "money": [], "temporal": []}'
    if task_id == "squad2_grounded_qa":
        return "FINAL_ANSWER: 1774"
    if task_id == "banking77_intent":
        return "FINAL_ANSWER: declined_card_payment"
    if task_id == "gsm8k_math":
        return "FINAL_ANSWER: 42"
    if task_id == "mbpp_python":
        return "```python\ndef add_two(x):\n    return x + 2\n```"
    if task_id == "sciq_verification":
        return "FINAL_ANSWER: C"
    raise ValueError(task_id)


class StructuredEndToEndStubAdapter(BaseRealV4Adapter):
    """Deterministic response surface across all 6 tasks."""

    adapter_id = "e2e_stub"

    def __init__(self, model_id: str, pinned_ver: str) -> None:
        self.model_id = model_id
        self.pinned_model_version = pinned_ver
        self.calls = 0

    def serialize_request_payload(
        self, request: CanonicalAdapterRequest
    ) -> Dict[str, Any]:
        return {"s": request.system_text, "u": request.user_text}

    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        self.calls += 1
        good = ("STRICT CONTRACT" in request.user_text) and (
            "Rely strictly on the provided input" in request.user_text
        )
        txt = _good_output_for_task(request.task_id) if good else "WRONG"
        prov = ResponseProvenance(
            adapter_id=self.adapter_id,
            requested_model_id=self.model_id,
            pinned_model_version=self.pinned_model_version,
            returned_model_version=self.pinned_model_version,
            response_id=f"e2e_{self.model_id}_{self.calls:06d}",
            finish_reason="STOP",
            tokens_input=100,
            tokens_output=6,
            latency_ms=1.0,
            cost_usd=0.00005,
            within_pinned_envelope=True,
        )
        return CanonicalAdapterResponse(raw_text=txt, provenance=prov)


def test_tasks_13_to_15_full_orchestration_and_independent_audit(tmp_path: Path):
    proto_copy = tmp_path / "protocol"
    proto_copy.mkdir()
    for fname in (
        "study_plan.md",
        "preregistration.md",
        "statistical_analysis_plan.md",
    ):
        (proto_copy / fname).write_text(
            (REAL_V4_PROTOCOL_DIR / fname).read_text(encoding="utf-8"), encoding="utf-8"
        )

    tasks = REGISTERED_TASKS
    dman = tmp_path / "dataset_manifest.json"
    dman.write_text(
        json.dumps({"tasks": {t: {"partitions": {"clean": f"h_{t}"}} for t in tasks}})
        + "\n",
        encoding="utf-8",
    )

    models = ("m_fast", "m_local_1", "m_local_2")
    mreg = tmp_path / "model_registry.json"
    mreg.write_text(
        json.dumps(
            {
                "canonical_decoding_intent": {
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "seed": 42,
                },
                "models": {m: {"default_pinned_version": f"ver_{m}"} for m in models},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    adapters = {m: StructuredEndToEndStubAdapter(m, f"ver_{m}") for m in models}
    # Avoid subprocess python execution overhead on mbpp by scoring 2 items/split cleanly
    splits: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for t in tasks:
        for part, n in (
            ("pilot", 2),
            ("screen", 2),
            ("validation", 2),
            ("clean_confirmation", 4),
            ("stability_confirmation", 4),
        ):
            splits[(t, part)] = [_stub_item_for_task(t, part, i) for i in range(n)]

    res_dir = tmp_path / "results"
    paths = RunnerPaths(
        protocol_dir=proto_copy,
        dataset_manifest_path=dman,
        perturbation_registry_path=PERTURBATION_REGISTRY_PATH,
        model_registry_path=mreg,
        calibration_config_path=CALIBRATION_CONFIG_PATH,
        code_paths=default_code_paths(ROOT),
        results_dir=res_dir,
    )
    runner = ResumableRealV4Runner(
        study_id="test-e2e-v4",
        paths=paths,
        adapters=adapters,
        splits_by_task_and_partition=splits,
        tasks=tasks,
        models=models,
        num_baseline_trajectories=2,
        n_bootstrap=40,
    )

    runner.freeze_protocol()
    runner.run_pilot(pilot_repeats=2, pilot_run_id=12)
    runner.run_screen_and_lock_candidates(include_research_full_cube_audit=True)
    runner.run_validation_and_lock_final_selections()
    runner.run_clean_confirmation()

    with pytest.raises(PhaseLockError):
        run_task15_analysis_and_audit(
            protocol_dir=proto_copy,
            dataset_manifest_path=dman,
            model_registry_path=mreg,
            results_dir=res_dir,
            tasks=tasks,
            models=models,
            study_id="test-e2e-v4",
            num_baseline_trajectories=2,
            n_bootstrap=120,
        )

    w1 = runner.run_stability_wave_and_lock_raw_results(
        requested_env_ids=WAVE_1_STABILITY_ENVS,
        write_final_locks=False,
    )
    assert w1["status"] == "partial_wave_complete"
    runner.run_stability_wave_and_lock_raw_results(
        requested_env_ids=None, write_final_locks=True
    )

    out = run_task15_analysis_and_audit(
        protocol_dir=proto_copy,
        dataset_manifest_path=dman,
        model_registry_path=mreg,
        results_dir=res_dir,
        tasks=tasks,
        models=models,
        study_id="test-e2e-v4",
        num_baseline_trajectories=2,
        n_bootstrap=120,
    )
    assert out["audit"]["independent_audit_passed"] is True
    assert out["summary"]["supporting_effect_recovery"]["available"] is True
    assert (res_dir / "summary.json").exists()
    assert (res_dir / "stability_certificate.json").exists()
    assert (res_dir / "claim_evidence_map.csv").exists()
    assert (res_dir / "release_manifest.json").exists()

    # Verify _write_claim_evidence_map handles INVALID/OUT_OF_ENVELOPE (empty selection_contrasts) without KeyError
    out_of_env_summary = {
        "primary_verdict": {
            "verdict_code": "INVALID_OUT_OF_ENVELOPE",
            "certificate_status": "INVALID/OUT_OF_ENVELOPE",
            "overall_pass": False,
            "selection_contrasts": {},
            "stability_summary": None,
        },
        "supporting_effect_recovery": {"available": False},
        "full_study_resource_reconciliation": {
            "total_unique_calls_all_phases": 10,
            "total_hosted_cost_usd_all_phases": 0.0,
        },
    }
    cmap_invalid = tmp_path / "claim_map_invalid.csv"
    _write_claim_evidence_map(out_of_env_summary, cmap_invalid)
    assert cmap_invalid.exists()
    assert "INVALID/OUT_OF_ENVELOPE" in cmap_invalid.read_text(encoding="utf-8")
