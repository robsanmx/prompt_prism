"""Integration tests for phase locks, tampering rejection, confirmation quarantine, and resume (Task 10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from research.real_v4.locking import PhaseLockError, default_code_paths
from research.real_v4.manifest import AppendOnlyManifest, ManifestIntegrityError
from research.real_v4.model_provenance import MODEL_REGISTRY_PATH, ResponseProvenance
from research.real_v4.perturbations import PERTURBATION_REGISTRY_PATH
from research.real_v4.providers import (
    BaseRealV4Adapter,
    CanonicalAdapterRequest,
    CanonicalAdapterResponse,
)
from research.real_v4.runner import (
    ResumableRealV4Runner,
    RunnerInterrupted,
    RunnerPaths,
)
from research.real_v4.simulation import CALIBRATION_CONFIG_PATH

ROOT = Path(__file__).resolve().parents[2]
REAL_V4_PROTOCOL_DIR = ROOT / "research" / "protocol" / "real_v4"


class CountingStubAdapter(BaseRealV4Adapter):
    """Deterministic stub adapter tracking exact live calls and supporting injected HTTP failures."""

    adapter_id = "counting_stub"

    def __init__(self, model_id: str, fail_sample_ids: set[str] | None = None) -> None:
        self.model_id = model_id
        self.pinned_model_version = f"{model_id}-v1"
        self.fail_sample_ids = fail_sample_ids or set()
        self.call_count = 0

    def serialize_request_payload(
        self, request: CanonicalAdapterRequest
    ) -> Dict[str, Any]:
        return {"system": request.system_text, "user": request.user_text}

    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload

    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        self.call_count += 1
        if request.sample_id in self.fail_sample_ids:
            prov_err = ResponseProvenance(
                adapter_id=self.adapter_id,
                requested_model_id=self.model_id,
                pinned_model_version=self.pinned_model_version,
                returned_model_version="HTTP_ERROR",
                response_id="err",
                finish_reason="HTTP_ERROR",
                tokens_input=0,
                tokens_output=0,
                latency_ms=1.0,
                retry_count=3,
                within_pinned_envelope=False,
                envelope_breach_reason="HTTP_503",
            )
            return CanonicalAdapterResponse(
                raw_text="", provenance=prov_err, error="HTTP 503"
            )

        prov = ResponseProvenance(
            adapter_id=self.adapter_id,
            requested_model_id=self.model_id,
            pinned_model_version=self.pinned_model_version,
            returned_model_version=self.pinned_model_version,
            response_id=f"resp_{self.model_id}_{self.call_count:06d}",
            finish_reason="STOP",
            tokens_input=20,
            tokens_output=5,
            latency_ms=1.5,
            retry_count=0,
            within_pinned_envelope=True,
        )
        return CanonicalAdapterResponse(raw_text="FINAL_ANSWER: 42", provenance=prov)


def _setup_runner_workspace(
    tmp_path: Path, fail_sample_ids: set[str] | None = None
) -> Tuple[ResumableRealV4Runner, Dict[str, CountingStubAdapter], Path]:
    proto_copy = tmp_path / "protocol"
    proto_copy.mkdir()
    for fname in (
        "study_plan.md",
        "preregistration.md",
        "statistical_analysis_plan.md",
    ):
        (proto_copy / fname).write_text(
            (REAL_V4_PROTOCOL_DIR / fname).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    data_manifest = tmp_path / "dataset_manifest.json"
    data_manifest.write_text('{"version":"mini_test"}\n', encoding="utf-8")

    paths = RunnerPaths(
        protocol_dir=proto_copy,
        dataset_manifest_path=data_manifest,
        perturbation_registry_path=PERTURBATION_REGISTRY_PATH,
        model_registry_path=MODEL_REGISTRY_PATH,
        calibration_config_path=CALIBRATION_CONFIG_PATH,
        code_paths=default_code_paths(ROOT),
        results_dir=tmp_path / "results",
    )

    tasks = ("gsm8k_math",)
    models = ("gemini-2.5-flash", "claude-3-5-haiku-20241022")
    adapters = {
        m: CountingStubAdapter(m, fail_sample_ids=fail_sample_ids) for m in models
    }

    splits: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for part, n in (
        ("pilot", 2),
        ("screen", 4),
        ("validation", 4),
        ("clean_confirmation", 4),
        ("stability_confirmation", 4),
    ):
        splits[("gsm8k_math", part)] = [
            {
                "sample_id": f"gsm_{part}_{i}",
                "group_id": f"grp_{part}_{i}",
                "question": f"What is 40 + 2 (item {i})?",
                "target": "42",
            }
            for i in range(n)
        ]

    runner = ResumableRealV4Runner(
        study_id="test-study-v4",
        paths=paths,
        adapters=adapters,
        splits_by_task_and_partition=splits,
        tasks=tasks,
        models=models,
        validation_env_ids=("ref", "fmt_whitespace"),
        stability_env_ids=("repeat_r2", "fmt_whitespace", "item_holdout_group"),
        num_baseline_trajectories=2,
        n_bootstrap=100,
    )
    return runner, adapters, proto_copy


def test_confirmation_blocked_before_locks_interruption_resume_and_tamper_rejection(
    tmp_path: Path,
):
    runner, adapters, proto_copy = _setup_runner_workspace(
        tmp_path, fail_sample_ids={"gsm_clean_confirmation_3"}
    )

    # 1. Confirmation cannot run before protocol/pilot/screen/validation/final_selection locks
    with pytest.raises(PhaseLockError, match="missing prerequisite lock"):
        runner.run_clean_confirmation()

    runner.freeze_protocol()
    with pytest.raises(PhaseLockError, match="missing prerequisite lock 'pilot'"):
        runner.run_screen_and_lock_candidates()

    # 2. Interrupt pilot mid-run after 2 calls, then resume to completion with zero duplicate adapter calls!
    with pytest.raises(RunnerInterrupted):
        runner.run_pilot(max_new_calls=2)
    calls_after_interrupt = sum(a.call_count for a in adapters.values())
    assert calls_after_interrupt == 2

    pilot_lock = runner.run_pilot(max_new_calls=None)
    calls_after_resume = sum(a.call_count for a in adapters.values())
    assert (
        calls_after_resume == 4
    )  # 1 task * 2 models * 2 pilot items = 4 total calls, zero duplicates!
    assert pilot_lock["metadata"]["replay_audit"]["total_observed"] == 4

    # 3. Interrupt screen mid-phase, then resume cleanly and lock candidates
    with pytest.raises(RunnerInterrupted):
        runner.run_screen_and_lock_candidates(max_new_calls=15)
    runner.run_screen_and_lock_candidates(max_new_calls=None)

    # Confirmation STILL blocked before validation & final_selection_lock!
    with pytest.raises(PhaseLockError, match="missing prerequisite lock 'validation'"):
        runner.run_clean_confirmation()

    runner.run_validation_and_lock_final_selections()

    # 4. Unregistered confirmation prompt is refused even after final_selection_lock
    with pytest.raises(PhaseLockError, match="unregistered confirmation run_id"):
        runner.run_clean_confirmation(extra_unregistered_run_id=61)

    # 5. Clean confirmation runs and records terminal failure (`gsm_clean_confirmation_3`) as score=0.0
    conf_lock = runner.run_clean_confirmation()
    conf_audit = conf_lock["metadata"]["replay_audit"]
    assert conf_audit["terminal_failure_calls"] >= 2  # failed item across models

    # 6. Adding an unregistered stability environment after registration is refused
    with pytest.raises(PhaseLockError, match="not registered in protocol lock"):
        runner.run_stability_wave_and_lock_raw_results(
            requested_env_ids=("unregistered_new_env",)
        )

    raw_lock = runner.run_stability_wave_and_lock_raw_results()
    assert raw_lock["phase"] == "raw_results_lock"

    # 7. Tampering with a protocol document invalidates subsequent fingerprint verification!
    (proto_copy / "preregistration.md").write_text(
        (proto_copy / "preregistration.md").read_text(encoding="utf-8")
        + "\n<!-- silent edit -->\n",
        encoding="utf-8",
    )
    with pytest.raises(PhaseLockError, match="Study fingerprint drift"):
        runner._verify_for(
            runner.prism_selector
            and raw_lock
            and __import__(
                "research.real_v4.locking", fromlist=["StudyPhase"]
            ).StudyPhase.RAW_RESULTS_LOCK
        )


def test_manifest_detects_truncated_partial_line_and_duplicate_response_id(
    tmp_path: Path,
):
    mpath = tmp_path / "partial.jsonl"
    m = AppendOnlyManifest(mpath)
    rec = {
        "call_key": "k1",
        "phase": "pilot",
        "task_id": "t",
        "model_id": "m",
        "env_id": "ref",
        "run_id": 0,
        "sample_id": "s1",
        "score": 1.0,
        "status": "ok",
        "response_id": "unique_r1",
    }
    m.append(rec)

    # Duplicate response_id rejected
    with pytest.raises(ManifestIntegrityError, match="Duplicate response_id"):
        m.append({**rec, "call_key": "k2", "sample_id": "s2"})

    # Truncated partial line (missing newline / cut JSON) detected on reload
    with mpath.open("a", encoding="utf-8") as handle:
        handle.write('{"call_key":"k_partial"')
    with pytest.raises(ManifestIntegrityError, match="Truncated/partial"):
        AppendOnlyManifest(mpath)
