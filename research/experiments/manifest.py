"""
Provenance Manifest and Content-Addressed Execution Tracker for Studies B and C.
Implements immutable audit trails, content-addressed caching, and strict
isolation between discovery and confirmation partitions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd


def get_current_git_commit() -> str:
    """Retrieve current repository commit SHA if in git repo."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def sha256_hash(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class ProvenanceRecord:
    """
    Complete immutable provenance record for a single model generation trial.
    """

    trial_id: str
    experiment_id: str
    git_commit: str
    task_id: str
    model_id: str
    provider: str
    run_id: int
    factor_levels: Dict[str, int]
    sample_id: str
    split: str  # "discovery" or "confirmation"
    prompt_text: str
    prompt_hash: str
    response_text: str
    response_hash: str
    metrics: Dict[str, float]
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: float
    timestamp: str
    seed: int
    retry_count: int = 0
    error: Optional[str] = None
    cache_key: str = ""
    data_origin: str = "real"  # "real" or "simulated"
    request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    system_fingerprint: Optional[str] = None

    def __post_init__(self):
        if not self.prompt_hash:
            self.prompt_hash = sha256_hash(self.prompt_text)
        if not self.response_hash:
            self.response_hash = sha256_hash(self.response_text or "")
        if not self.cache_key:
            self.cache_key = ExperimentManifest.compute_cache_key(
                model_id=self.model_id,
                prompt_text=self.prompt_text,
                sample_id=self.sample_id,
                seed=self.seed,
            )


class ExperimentManifest:
    """
    Resumable, append-only experiment manifest.
    """

    def __init__(self, experiment_id: str, manifest_path: Optional[Path] = None):
        self.experiment_id = experiment_id
        self.manifest_path = manifest_path
        self.records: List[ProvenanceRecord] = []
        self._cache_index: Dict[str, ProvenanceRecord] = {}
        self.is_confirmation_locked: bool = True

        if manifest_path and manifest_path.exists():
            self.load_from_disk(manifest_path)

    @staticmethod
    def compute_cache_key(
        model_id: str,
        prompt_text: str,
        sample_id: str,
        seed: int,
        request_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Content-addressed cache key covering model, parameters, prompt, item, and seed.
        """
        params_str = json.dumps(request_params or {}, sort_keys=True)
        raw_key = f"{model_id}|{params_str}|{sha256_hash(prompt_text)}|{sample_id}|{seed}"
        return sha256_hash(raw_key)

    def add_record(self, record: ProvenanceRecord, persist: bool = True):
        """Append record to manifest and memory index."""
        self.records.append(record)
        self._cache_index[record.cache_key] = record

        if persist and self.manifest_path:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    def get_cached_record(self, cache_key: str) -> Optional[ProvenanceRecord]:
        """Look up existing record by cache key."""
        return self._cache_index.get(cache_key)

    def load_from_disk(self, path: Path, require_origin: Optional[str] = None):
        """Load manifest from JSONL file, enforcing data origin boundaries."""
        self.records = []
        self._cache_index = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    # Detect v1 simulated records if data_origin is missing
                    if "data_origin" not in data:
                        if data.get("experiment_id") in ("confirmatory_study_v1", "pilot_v1") or "confirmatory_manifest.jsonl" in str(path):
                            data["data_origin"] = "simulated"
                        else:
                            data["data_origin"] = "real"
                    rec = ProvenanceRecord(**data)
                    if require_origin and rec.data_origin != require_origin:
                        raise ValueError(
                            f"Origin mismatch: Manifest {path} contains record with data_origin='{rec.data_origin}', "
                            f"but empirical loader requires '{require_origin}'. Quarantined simulated fixtures cannot enter empirical analysis."
                        )
                    self.records.append(rec)
                    self._cache_index[rec.cache_key] = rec

    def get_discovery_dataframe(self) -> pd.DataFrame:
        """
        Safe API to load discovery trials only. Never returns confirmation rows.
        """
        disc_records = [r for r in self.records if r.split == "discovery"]
        rows = []
        for r in disc_records:
            row = {
                "trial_id": r.trial_id,
                "task_id": r.task_id,
                "model_id": r.model_id,
                "provider": r.provider,
                "run_id": r.run_id,
                "sample_id": r.sample_id,
                "split": r.split,
                **r.factor_levels,
                **r.metrics,
                "error": r.error,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def get_confirmation_dataframe(self, unlock_token: str) -> pd.DataFrame:
        """
        Protected API to load confirmation data.
        Enforces locking until explicitly unlocked with frozen commit token.
        """
        if self.is_confirmation_locked:
            if not unlock_token or unlock_token != "CONFIRMATORY_ANALYSIS_UNLOCKED":
                raise PermissionError(
                    "Confirmation split is locked! Cannot load confirmation data until "
                    "all screening recommendations are frozen and the unlock token is provided."
                )

        conf_records = [r for r in self.records if r.split == "confirmation"]
        rows = []
        for r in conf_records:
            row = {
                "trial_id": r.trial_id,
                "task_id": r.task_id,
                "model_id": r.model_id,
                "provider": r.provider,
                "run_id": r.run_id,
                "sample_id": r.sample_id,
                "split": r.split,
                **r.factor_levels,
                **r.metrics,
                "error": r.error,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Run automated data integrity checks on recorded trials.
        """
        total = len(self.records)
        unique_cache_keys = len(self._cache_index)
        has_duplicates = (total != unique_cache_keys)

        splits = [r.split for r in self.records]
        num_disc = splits.count("discovery")
        num_conf = splits.count("confirmation")

        errors = [r for r in self.records if r.error is not None]

        return {
            "total_records": total,
            "unique_trials": unique_cache_keys,
            "has_duplicate_keys": has_duplicates,
            "discovery_count": num_disc,
            "confirmation_count": num_conf,
            "error_count": len(errors),
            "manifest_hash": sha256_hash("".join(r.cache_key for r in self.records)),
        }


def load_real_v2_manifest(path: Path, allow_pilot: bool = False) -> ExperimentManifest:
    """
    Strict empirical loader for real_v2 manifests.
    Guarantees no simulated or v1 fixture artifact enters the empirical analysis,
    and enforces that no pilot partition records leak into confirmatory analysis
    when allow_pilot is False.
    """
    if not path.exists():
        raise FileNotFoundError(f"V2 manifest not found: {path}")
    manifest = ExperimentManifest(experiment_id="real_v2_manifest", manifest_path=path)
    manifest.load_from_disk(path, require_origin="real")
    if not allow_pilot:
        pilot_records = [r for r in manifest.records if r.split == "pilot" or r.sample_id.startswith("pilot_")]
        if pilot_records:
            raise ValueError(
                f"Quarantine violation: manifest {path} contains {len(pilot_records)} pilot records. "
                "Pilot items are quarantined from confirmatory analysis."
            )
    return manifest

