"""Append-only content-addressed JSONL execution manifest and replay auditor (Task 10)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set

from research.real_v4.schemas import canonical_json, canonical_sha256


class ManifestIntegrityError(RuntimeError):
    """Raised on partial/corrupted JSONL lines, duplicate call keys, duplicate response IDs, or missing cells."""


def deterministic_call_key(
    *,
    study_id: str,
    phase: str,
    task_id: str,
    model_id: str,
    env_id: str,
    run_id: int,
    sample_id: str,
    repeat_idx: int = 0,
) -> str:
    raw = f"{study_id}|{phase}|{task_id}|{model_id}|{env_id}|{run_id}|{sample_id}|r{repeat_idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AppendOnlyManifest:
    """Crash-safe append-only JSONL manifest with deterministic cache-key and response-ID deduplication."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records_by_key: Dict[str, Dict[str, Any]] = {}
        self._seen_response_ids: Set[str] = set()
        if self.path.exists():
            self._load_and_validate()

    def _load_and_validate(self) -> None:
        raw_bytes = self.path.read_bytes()
        if not raw_bytes:
            return
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestIntegrityError(f"Non-UTF8 corrupted manifest {self.path}") from exc

        if not text.endswith("\n"):
            raise ManifestIntegrityError(
                f"Truncated/partial trailing JSONL line detected in {self.path} (missing newline)"
            )

        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise ManifestIntegrityError(f"Empty line {line_no} in manifest {self.path}")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestIntegrityError(
                    f"Malformed JSON at line {line_no} of {self.path}: {exc}"
                ) from exc
            self._register_record(rec, source_desc=f"{self.path.name}:{line_no}")

    def _register_record(self, rec: Mapping[str, Any], source_desc: str) -> None:
        for req_field in ("call_key", "phase", "task_id", "model_id", "env_id", "run_id", "sample_id", "score", "status"):
            if req_field not in rec:
                raise ManifestIntegrityError(f"Missing field {req_field!r} in {source_desc}")
        ck = str(rec["call_key"])
        if ck in self._records_by_key:
            raise ManifestIntegrityError(f"Duplicate call_key {ck} in {source_desc}")

        resp_id = str(rec.get("response_id") or "")
        if rec["status"] == "ok":
            if not resp_id:
                raise ManifestIntegrityError(f"Status 'ok' record missing response_id in {source_desc}")
            if resp_id in self._seen_response_ids:
                raise ManifestIntegrityError(f"Duplicate response_id {resp_id!r} in {source_desc}")
            self._seen_response_ids.add(resp_id)
        elif rec["status"] == "terminal_failure":
            if float(rec["score"]) != 0.0:
                raise ManifestIntegrityError(f"Terminal failure must record score=0.0 in {source_desc}")
        else:
            raise ManifestIntegrityError(f"Unknown record status {rec['status']!r} in {source_desc}")

        self._records_by_key[ck] = dict(rec)

    def has_key(self, call_key: str) -> bool:
        return call_key in self._records_by_key

    def get_record(self, call_key: str) -> Dict[str, Any]:
        return self._records_by_key[call_key]

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records_by_key.values())

    def append(self, record: Mapping[str, Any]) -> None:
        self._register_record(record, source_desc="append")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = canonical_json(dict(record)) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

    def append_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        lines: List[str] = []
        for record in records:
            self._register_record(record, source_desc="append_many")
            lines.append(canonical_json(dict(record)) + "\n")
        if not lines:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())

    def replay_audit(self, expected_call_keys: Iterable[str]) -> Dict[str, Any]:
        """Re-read disk manifest from scratch and reconcile against exact expected `call_key` set."""
        reloaded = AppendOnlyManifest(self.path)
        expected_set = set(expected_call_keys)
        observed_set = set(reloaded._records_by_key.keys())

        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        if missing or unexpected:
            raise ManifestIntegrityError(
                f"Manifest replay mismatch in {self.path.name}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        ok_count = sum(1 for r in reloaded.records if r["status"] == "ok")
        fail_count = sum(1 for r in reloaded.records if r["status"] == "terminal_failure")
        return {
            "manifest_name": self.path.name,
            "total_expected": len(expected_set),
            "total_observed": len(observed_set),
            "ok_calls": ok_count,
            "terminal_failure_calls": fail_count,
            "unique_response_ids": len(reloaded._seen_response_ids),
            "records_canonical_sha256": canonical_sha256(
                [reloaded._records_by_key[k] for k in sorted(observed_set)]
            ),
        }
