"""Content locks separating discovery selection from confirmation execution."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_discovery_only(manifest_path: Path) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "discovery":
                raise ValueError(
                    f"Recommendation lock requires a discovery-only manifest; line {number} "
                    f"has split={record.get('split')!r}"
                )


def create_recommendation_lock(
    discovery_manifest: Path,
    recommendations: Path,
    lock_path: Path,
    protocol_hash: str,
) -> dict:
    _assert_discovery_only(discovery_manifest)
    payload = {
        "lock_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol_hash,
        "discovery_manifest_path": str(discovery_manifest),
        "discovery_manifest_sha256": file_sha256(discovery_manifest),
        "recommendations_path": str(recommendations),
        "recommendations_sha256": file_sha256(recommendations),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def assert_lock_matches(
    discovery_manifest: Path,
    recommendations: Path,
    lock_path: Path,
    protocol_hash: str,
) -> dict:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != protocol_hash:
        raise ValueError("protocol hash does not match the recommendation lock")
    if payload.get("discovery_manifest_sha256") != file_sha256(discovery_manifest):
        raise ValueError("discovery manifest hash does not match the recommendation lock")
    if payload.get("recommendations_sha256") != file_sha256(recommendations):
        raise ValueError("recommendation hash does not match the recommendation lock")
    _assert_discovery_only(discovery_manifest)
    return payload
