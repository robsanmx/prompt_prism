"""Model provenance, local weight-hash verification, repeatability policy, and model registry (Task 9)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.real_v4.schemas import (
    CURRENT_SCHEMA_VERSION,
    _VersionedBaseModel,
    canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
QWEN_INVENTORY_PATH = ROOT / "research" / "protocol" / "real_v4" / "local_qwen_inventory.json"
MISTRAL_INVENTORY_PATH = ROOT / "research" / "protocol" / "real_v4" / "local_mistral_inventory.json"
MODEL_REGISTRY_PATH = ROOT / "research" / "protocol" / "real_v4" / "model_registry.json"


class ProvenanceEnvelopeError(RuntimeError):
    """Raised when required model provenance is missing or violates frozen hashes."""


class ResponseProvenance(_VersionedBaseModel):
    adapter_id: str
    requested_model_id: str
    pinned_model_version: str
    returned_model_version: str
    response_id: str
    finish_reason: str
    tokens_input: int
    tokens_output: int
    tokens_thoughts: int = 0
    latency_ms: float
    retry_count: int = 0
    cost_usd: float = 0.0
    peak_memory_mb: Optional[float] = None
    throughput_tok_per_s: Optional[float] = None
    local_weights_inventory_sha256: Optional[str] = None
    tokenizer_sha256: Optional[str] = None
    chat_template_sha256: Optional[str] = None
    runtime_versions: Dict[str, str] = {}
    hardware_platform: Dict[str, str] = {}
    within_pinned_envelope: bool
    envelope_breach_reason: Optional[str] = None


@dataclass(frozen=True)
class RepeatabilityPolicyDecision:
    exact_output_agreement: float
    exact_score_agreement: float
    score_std_across_repeats: float
    nondeterminism_detected: bool
    frozen_repeat_count: int
    aggregation_rule: str


def load_qwen_inventory(inventory_path: Path = QWEN_INVENTORY_PATH) -> Dict[str, Any]:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if len(files) != 9:
        raise ProvenanceEnvelopeError(f"Expected 9 frozen Qwen files in inventory, got {len(files)}")
    file_map = {f["name"]: f["sha256"] for f in files}
    inv_digest = canonical_sha256(file_map)
    return {
        "repo_id": payload["repo_id"],
        "revision": payload["revision"],
        "cache_snapshot": payload["cache_snapshot"],
        "files": files,
        "file_sha256_map": file_map,
        "weights_inventory_sha256": inv_digest,
        "tokenizer_sha256": file_map["tokenizer.json"],
        "tokenizer_config_sha256": file_map["tokenizer_config.json"],
        "quantization": payload["model_config"]["quantization"],
    }


def verify_local_qwen_files_on_disk(
    snapshot_dir: Optional[Path] = None,
    inventory_path: Path = QWEN_INVENTORY_PATH,
    *,
    verify_full_safetensors_bytes: bool = False,
) -> Dict[str, Any]:
    """Verify local Qwen snapshot directory metadata (and optional full 4.6GB SHA-256) against inventory."""
    inv = load_qwen_inventory(inventory_path)
    snap = snapshot_dir or Path(inv["cache_snapshot"]).expanduser()
    if not snap.exists():
        raise ProvenanceEnvelopeError(f"Local Qwen snapshot directory missing: {snap}")

    for entry in inv["files"]:
        fpath = snap / entry["name"]
        if not fpath.exists():
            raise ProvenanceEnvelopeError(f"Missing frozen Qwen file: {fpath}")
        actual_size = fpath.stat().st_size
        if actual_size != int(entry["size_bytes"]):
            raise ProvenanceEnvelopeError(
                f"Size mismatch for {entry['name']}: expected {entry['size_bytes']}, got {actual_size}"
            )
        if verify_full_safetensors_bytes or entry["name"] != "model.safetensors":
            h = hashlib.sha256()
            with fpath.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != entry["sha256"]:
                raise ProvenanceEnvelopeError(f"SHA-256 mismatch for {entry['name']}")
    return inv


def evaluate_pilot_repeatability(
    repeat_raw_outputs: Sequence[Sequence[str]],
    repeat_scores: Sequence[Sequence[float]],
) -> RepeatabilityPolicyDecision:
    """Evaluate pilot repeat passes (`R >= 2` passes over `N` items) and freeze replicate rule.

    `repeat_raw_outputs[r][i]` and `repeat_scores[r][i]` hold pass `r` on item `i`.
    """
    n_passes = len(repeat_raw_outputs)
    if n_passes < 2 or len(repeat_scores) != n_passes:
        raise ValueError("Require at least 2 repeat passes with matching output and score dimensions")
    n_items = len(repeat_raw_outputs[0])
    if n_items == 0:
        raise ValueError("Pilot repeat items cannot be empty")

    exact_text_matches = 0
    exact_score_matches = 0
    item_stds: List[float] = []

    for i in range(n_items):
        texts = [str(repeat_raw_outputs[r][i]) for r in range(n_passes)]
        scores = [float(repeat_scores[r][i]) for r in range(n_passes)]
        if len(set(texts)) == 1:
            exact_text_matches += 1
        if max(scores) - min(scores) <= 1e-12:
            exact_score_matches += 1
        item_stds.append(float(np.std(scores, ddof=1)))

    out_agree = exact_text_matches / n_items
    score_agree = exact_score_matches / n_items
    mean_std = float(np.mean(item_stds))
    nondet = (out_agree < 1.0 - 1e-12) or (score_agree < 1.0 - 1e-12)

    return RepeatabilityPolicyDecision(
        exact_output_agreement=out_agree,
        exact_score_agreement=score_agree,
        score_std_across_repeats=mean_std,
        nondeterminism_detected=nondet,
        frozen_repeat_count=3 if nondet else 1,
        aggregation_rule="mean_across_3_identical_repeats" if nondet else "single_deterministic_pass_with_stability_audit_r2_r3",
    )


def host_hardware_fingerprint() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def load_mistral_inventory(inventory_path: Path = MISTRAL_INVENTORY_PATH) -> Dict[str, Any]:
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if len(files) != 7:
        raise ProvenanceEnvelopeError(f"Expected 7 frozen Mistral files in inventory, got {len(files)}")
    file_map = {f["name"]: f["sha256"] for f in files}
    inv_digest = canonical_sha256(file_map)
    return {
        "repo_id": payload["repo_id"],
        "revision": payload["revision"],
        "cache_snapshot": payload["cache_snapshot"],
        "files": files,
        "file_sha256_map": file_map,
        "weights_inventory_sha256": inv_digest,
        "tokenizer_sha256": file_map["tokenizer.json"],
        "tokenizer_config_sha256": file_map["tokenizer_config.json"],
        "quantization": payload["model_config"]["quantization"],
    }


def verify_local_mistral_files_on_disk(
    snapshot_dir: Optional[Path] = None,
    inventory_path: Path = MISTRAL_INVENTORY_PATH,
    *,
    verify_full_safetensors_bytes: bool = False,
) -> Dict[str, Any]:
    inv = load_mistral_inventory(inventory_path)
    snap = snapshot_dir or Path(inv["cache_snapshot"]).expanduser()
    if not snap.exists():
        raise ProvenanceEnvelopeError(f"Local Mistral snapshot directory missing: {snap}")

    for entry in inv["files"]:
        fpath = snap / entry["name"]
        if not fpath.exists():
            raise ProvenanceEnvelopeError(f"Missing frozen Mistral file: {fpath}")
        actual_size = fpath.stat().st_size
        if actual_size != int(entry["size_bytes"]):
            raise ProvenanceEnvelopeError(
                f"Size mismatch for {entry['name']}: expected {entry['size_bytes']}, got {actual_size}"
            )
        if verify_full_safetensors_bytes or entry["name"] != "model.safetensors":
            h = hashlib.sha256()
            with fpath.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != entry["sha256"]:
                raise ProvenanceEnvelopeError(f"SHA-256 mismatch for {entry['name']}")
    return inv


def write_model_registry(output_path: Path = MODEL_REGISTRY_PATH) -> Dict[str, Any]:
    inv = load_qwen_inventory()
    m_inv = load_mistral_inventory()
    registry: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": "prompt-prism-real-v4-selection-stability-2026",
        "canonical_decoding_intent": {
            "temperature": 0.0,
            "max_output_tokens": 512,
            "thinking_enabled": False,
            "seed": 42,
        },
        "models": {
            "gemini-2.5-flash": {
                "provider": "google_gemini",
                "deployment_mode": "hosted_api",
                "requested_model_id": "gemini-2.5-flash",
                "pinned_version_policy": "lock_exact_returned_modelVersion_at_pilot",
                "default_pinned_version": "gemini-2.5-flash",
                "pricing_usd_per_1m_input": 0.30,
                "pricing_usd_per_1m_output": 2.50,
                "decoding": {
                    "temperature": 0.0,
                    "topP": 1.0,
                    "maxOutputTokens": 512,
                    "seed": 42,
                    "thinkingBudget": 0,
                },
            },
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit": {
                "provider": "local_mlx",
                "deployment_mode": "local_weights",
                "requested_model_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                "pinned_version_policy": "exact_snapshot_and_7_file_sha256",
                "default_pinned_version": m_inv["revision"],
                "weights_inventory_sha256": m_inv["weights_inventory_sha256"],
                "tokenizer_sha256": m_inv["tokenizer_sha256"],
                "quantization": m_inv["quantization"],
                "pricing_usd_per_1m_input": 0.0,
                "pricing_usd_per_1m_output": 0.0,
                "decoding": {
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "enable_thinking": False,
                },
            },
            "mlx-community/Qwen3-8B-4bit": {
                "provider": "local_mlx",
                "deployment_mode": "local_weights",
                "requested_model_id": "mlx-community/Qwen3-8B-4bit",
                "pinned_version_policy": "exact_snapshot_and_9_file_sha256",
                "default_pinned_version": inv["revision"],
                "weights_inventory_sha256": inv["weights_inventory_sha256"],
                "tokenizer_sha256": inv["tokenizer_sha256"],
                "quantization": inv["quantization"],
                "pricing_usd_per_1m_input": 0.0,
                "pricing_usd_per_1m_output": 0.0,
                "decoding": {
                    "temperature": 0.0,
                    "max_tokens": 512,
                    "enable_thinking": False,
                },
            },
        },
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry


if __name__ == "__main__":
    write_model_registry()
