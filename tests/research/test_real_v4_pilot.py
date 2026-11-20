"""Focused unit & integration tests for Task 12 quarantined cross-provider pilot and pilot_report.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from research.real_v4.locking import default_code_paths
from research.real_v4.model_provenance import MODEL_REGISTRY_PATH, ResponseProvenance
from research.real_v4.perturbations import PERTURBATION_REGISTRY_PATH
from research.real_v4.pilot import (
    audit_pilot_quarantine,
    build_pilot_report,
    load_real_v4_splits,
)
from research.real_v4.providers import (
    BaseRealV4Adapter,
    CanonicalAdapterRequest,
    CanonicalAdapterResponse,
)
from research.real_v4.runner import ResumableRealV4Runner, RunnerPaths
from research.real_v4.simulation import CALIBRATION_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[2]
REAL_V4_PROTOCOL_DIR = ROOT / "research" / "protocol" / "real_v4"


class PilotSimAdapter(BaseRealV4Adapter):
    adapter_id = "pilot_sim"

    def __init__(self, model_id: str, *, nondeterministic: bool = False) -> None:
        self.model_id = model_id
        self.pinned_model_version = f"{model_id}-pinned"
        self.nondeterministic = nondeterministic
        self.calls = 0

    def serialize_request_payload(
        self, request: CanonicalAdapterRequest
    ) -> Dict[str, Any]:
        return {"system": request.system_text, "user": request.user_text}

    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        self.calls += 1
        ans = "42" if (not self.nondeterministic or self.calls % 4 != 0) else "41"
        prov = ResponseProvenance(
            adapter_id=self.adapter_id,
            requested_model_id=self.model_id,
            pinned_model_version=self.pinned_model_version,
            returned_model_version=self.pinned_model_version,
            response_id=f"p_{self.model_id}_{self.calls:05d}",
            finish_reason="STOP",
            tokens_input=120,
            tokens_output=12,
            latency_ms=45.0,
            cost_usd=0.0001,
            within_pinned_envelope=True,
        )
        return CanonicalAdapterResponse(
            raw_text=f"FINAL_ANSWER: {ans}", provenance=prov
        )


def test_real_v4_actual_pilot_quarantine_across_all_six_tasks():
    splits = load_real_v4_splits()
    audit = audit_pilot_quarantine(splits)
    assert audit["quarantine_verified"] is True
    assert len(audit["per_task"]) == 6
    for info in audit["per_task"].values():
        assert info["pilot_items"] == 20
        assert info["sample_overlap_count"] == 0
        assert info["group_overlap_count"] == 0


def test_pilot_quarantine_detects_group_leakage():
    bad_splits: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        ("gsm8k_math", "pilot"): [
            {
                "sample_id": "p0",
                "group_id": "shared_grp",
                "question": "q",
                "target": "1",
            }
        ],
        ("gsm8k_math", "screen"): [
            {
                "sample_id": "s0",
                "group_id": "shared_grp",
                "question": "q2",
                "target": "2",
            }
        ],
        ("gsm8k_math", "validation"): [],
        ("gsm8k_math", "clean_confirmation"): [],
        ("gsm8k_math", "stability_confirmation"): [],
    }
    with pytest.raises(RuntimeError, match="Pilot quarantine breach"):
        audit_pilot_quarantine(bad_splits, tasks=("gsm8k_math",))


def test_three_repeat_pilot_selects_repeat_policy_and_preserves_frozen_margins(
    tmp_path: Path,
):
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
    dman = tmp_path / "dataset_manifest.json"
    dman.write_text('{"v":"1"}\n', encoding="utf-8")

    tasks = ("gsm8k_math",)
    models = ("det-model", "nondet-model")
    adapters = {
        "det-model": PilotSimAdapter("det-model", nondeterministic=False),
        "nondet-model": PilotSimAdapter("nondet-model", nondeterministic=True),
    }
    splits: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for part, n in (
        ("pilot", 5),
        ("screen", 2),
        ("validation", 2),
        ("clean_confirmation", 2),
        ("stability_confirmation", 2),
    ):
        splits[("gsm8k_math", part)] = [
            {
                "sample_id": f"id_{part}_{i}",
                "group_id": f"grp_{part}_{i}",
                "question": "40+2?",
                "target": "42",
            }
            for i in range(n)
        ]

    paths = RunnerPaths(
        protocol_dir=proto_copy,
        dataset_manifest_path=dman,
        perturbation_registry_path=PERTURBATION_REGISTRY_PATH,
        model_registry_path=MODEL_REGISTRY_PATH,
        calibration_config_path=CALIBRATION_CONFIG_PATH,
        code_paths=default_code_paths(ROOT),
        results_dir=tmp_path / "results",
    )
    runner = ResumableRealV4Runner(
        study_id="pilot-test-v4",
        paths=paths,
        adapters=adapters,
        splits_by_task_and_partition=splits,
        tasks=tasks,
        models=models,
    )
    runner.freeze_protocol()
    runner.run_pilot(pilot_repeats=3, pilot_run_id=8)

    report = build_pilot_report(
        manifest_path=paths.results_dir / "pilot_manifest.jsonl",
        splits=splits,
        tasks=tasks,
        models=models,
        pilot_repeats=3,
        pilot_run_id=8,
        study_id="pilot-test-v4",
        output_path=paths.results_dir / "pilot_report.json",
    )
    assert report["all_pilot_gates_passed"] is True
    assert report["claim_margins_unchanged"] == {
        "delta_select": 0.01,
        "epsilon_stability": 0.02,
        "methods_modified": False,
    }
    pol = report["model_execution_policy_and_projections"]
    assert pol["det-model"]["nondeterminism_detected"] is False
    assert pol["det-model"]["frozen_repeat_count"] == 1
    assert pol["nondet-model"]["nondeterminism_detected"] is True
    assert pol["nondet-model"]["frozen_repeat_count"] == 3
