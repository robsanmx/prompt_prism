"""Cryptographic phase-lock chain enforcing append-only progression and zero post-lock tampering (Task 10)."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from research.real_v4.schemas import (
    CURRENT_SCHEMA_VERSION,
    canonical_sha256,
    lint_protocol_documents,
)

ROOT = Path(__file__).resolve().parents[2]


class PhaseLockError(RuntimeError):
    """Raised when a phase prerequisite is missing or any hashed input/manifest was modified."""


class StudyPhase(str, Enum):
    PROTOCOL = "protocol"
    PILOT = "pilot"
    SCREEN = "screen"
    CANDIDATE_LOCK = "candidate_lock"
    VALIDATION = "validation"
    FINAL_SELECTION_LOCK = "final_selection_lock"
    CLEAN_CONFIRMATION = "clean_confirmation"
    STABILITY_WAVE = "stability_wave"
    RAW_RESULTS_LOCK = "raw_results_lock"


PHASE_SEQUENCE: Tuple[StudyPhase, ...] = (
    StudyPhase.PROTOCOL,
    StudyPhase.PILOT,
    StudyPhase.SCREEN,
    StudyPhase.CANDIDATE_LOCK,
    StudyPhase.VALIDATION,
    StudyPhase.FINAL_SELECTION_LOCK,
    StudyPhase.CLEAN_CONFIRMATION,
    StudyPhase.STABILITY_WAVE,
    StudyPhase.RAW_RESULTS_LOCK,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_code_paths(root: Path = ROOT) -> Tuple[Path, ...]:
    pkg = root / "research" / "real_v4"
    return tuple(
        sorted(
            p for p in pkg.glob("*.py") if p.is_file()
        )
    )


def compute_study_fingerprint(
    *,
    protocol_dir: Path,
    dataset_manifest_path: Path,
    perturbation_registry_path: Path,
    model_registry_path: Path,
    calibration_config_path: Path,
    code_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Compute canonical SHA-256 fingerprint across protocol, data, perturbations, models, config, and code."""
    # Lint protocol documents first (rejects placeholders and missing contract fields)
    protocol_hashes = lint_protocol_documents(protocol_dir)

    required_files = [
        dataset_manifest_path,
        perturbation_registry_path,
        model_registry_path,
        calibration_config_path,
        *code_paths,
    ]
    file_hashes: Dict[str, str] = {f"protocol/{k}": v for k, v in sorted(protocol_hashes.items())}
    for path in required_files:
        if not path.exists():
            raise PhaseLockError(f"Missing required study input file for lock fingerprint: {path}")
        key = path.name if not str(path).startswith(str(ROOT)) else path.relative_to(ROOT).as_posix()
        file_hashes[key] = sha256_file(path)

    pert_payload = json.loads(perturbation_registry_path.read_text(encoding="utf-8"))
    registered_env_ids = tuple(e["env_id"] for e in pert_payload.get("environments", []))
    if not registered_env_ids:
        raise PhaseLockError("Perturbation registry declares zero environments")

    fingerprint_sha = canonical_sha256(file_hashes)
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "file_hashes": dict(sorted(file_hashes.items())),
        "registered_env_ids": list(registered_env_ids),
        "study_fingerprint_sha256": fingerprint_sha,
    }


def lock_path_for_phase(locks_dir: Path, phase: StudyPhase) -> Path:
    return locks_dir / f"{phase.value}.lock.json"


def write_phase_lock(
    *,
    locks_dir: Path,
    phase: StudyPhase,
    fingerprint: Mapping[str, Any],
    artifact_path: Optional[Path] = None,
    prior_phase: Optional[StudyPhase] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    locks_dir.mkdir(parents=True, exist_ok=True)
    prior_sha: Optional[str] = None
    if prior_phase is not None:
        p_path = lock_path_for_phase(locks_dir, prior_phase)
        if not p_path.exists():
            raise PhaseLockError(f"Cannot write {phase.value} lock without prior {prior_phase.value} lock")
        prior_payload = json.loads(p_path.read_text(encoding="utf-8"))
        prior_sha = str(prior_payload["lock_sha256"])

    artifact_sha: Optional[str] = None
    artifact_name: Optional[str] = None
    if artifact_path is not None:
        if not artifact_path.exists():
            raise PhaseLockError(f"Phase {phase.value} artifact missing at lock time: {artifact_path}")
        artifact_sha = sha256_file(artifact_path)
        artifact_name = artifact_path.name

    payload: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "phase": phase.value,
        "study_fingerprint_sha256": str(fingerprint["study_fingerprint_sha256"]),
        "file_hashes": dict(fingerprint["file_hashes"]),
        "registered_env_ids": list(fingerprint["registered_env_ids"]),
        "prior_phase": prior_phase.value if prior_phase is not None else None,
        "prior_lock_sha256": prior_sha,
        "artifact_name": artifact_name,
        "artifact_sha256": artifact_sha,
        "metadata": dict(extra_metadata or {}),
    }
    payload["lock_sha256"] = canonical_sha256(payload)
    out = lock_path_for_phase(locks_dir, phase)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_phase_prerequisites(
    *,
    target_phase: StudyPhase,
    locks_dir: Path,
    artifacts_dir: Path,
    current_fingerprint: Mapping[str, Any],
) -> Dict[StudyPhase, Dict[str, Any]]:
    """Verify full cryptographic lock chain from PROTOCOL through `target_phase`'s immediate predecessor."""
    target_idx = PHASE_SEQUENCE.index(target_phase)
    verified: Dict[StudyPhase, Dict[str, Any]] = {}
    prev_lock_sha: Optional[str] = None

    for idx in range(target_idx):
        req_phase = PHASE_SEQUENCE[idx]
        lpath = lock_path_for_phase(locks_dir, req_phase)
        if not lpath.exists():
            raise PhaseLockError(
                f"Refusing to run phase {target_phase.value!r}: missing prerequisite lock {req_phase.value!r}"
            )
        payload = json.loads(lpath.read_text(encoding="utf-8"))
        stored_lock_sha = payload.get("lock_sha256")
        check_copy = {k: v for k, v in payload.items() if k != "lock_sha256"}
        if canonical_sha256(check_copy) != stored_lock_sha:
            raise PhaseLockError(f"Tampered lock file detected in {lpath.name}")

        if payload.get("study_fingerprint_sha256") != current_fingerprint["study_fingerprint_sha256"]:
            raise PhaseLockError(
                f"Study fingerprint drift detected before {target_phase.value!r}: "
                f"locked={payload.get('study_fingerprint_sha256')}, "
                f"current={current_fingerprint['study_fingerprint_sha256']}"
            )
        if payload.get("registered_env_ids") != list(current_fingerprint["registered_env_ids"]):
            raise PhaseLockError("Registered stability environments changed after protocol lock")

        if idx > 0 and payload.get("prior_lock_sha256") != prev_lock_sha:
            raise PhaseLockError(f"Broken prior_lock_sha256 chain at {req_phase.value!r}")

        art_name = payload.get("artifact_name")
        art_sha = payload.get("artifact_sha256")
        if art_name is not None:
            art_path = artifacts_dir / str(art_name)
            if not art_path.exists():
                raise PhaseLockError(f"Locked artifact {art_name!r} was deleted")
            if sha256_file(art_path) != art_sha:
                raise PhaseLockError(f"Locked artifact {art_name!r} hash changed after {req_phase.value!r} lock")

        prev_lock_sha = str(stored_lock_sha)
        verified[req_phase] = payload

    return verified
