"""
Tests for Zero Pilot Leakage into Confirmatory Pipeline (Phase 4 Task 10).
Verifies:
1. Pilot partition items are strictly disjoint from discovery and confirmation items.
2. Confirmatory loaders explicitly reject pilot items.
3. Pilot manifest records cannot contaminate confirmatory analysis.
"""

import json
from pathlib import Path

import pytest

from research.data.real_v2.validate import load_dataset_records
from research.experiments.manifest import ExperimentManifest, load_real_v2_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "research" / "data" / "real_v2"


def test_pilot_items_strictly_disjoint():
    """Verify zero overlap between pilot and non-pilot items in all task files."""
    task_files = [
        "schema_extraction.jsonl",
        "grounded_qa.jsonl",
        "intent_classification.jsonl",
    ]

    for fname in task_files:
        path = DATA_DIR / fname
        records = load_dataset_records(path)

        pilot_ids = {r["id"] for r in records if r["split"] == "pilot"}
        disc_ids = {r["id"] for r in records if r["split"] == "discovery"}
        conf_ids = {r["id"] for r in records if r["split"] == "confirmation"}

        assert len(pilot_ids) == 20
        assert len(disc_ids) == 80
        assert len(conf_ids) == 80

        # Disjoint ID sets
        assert (
            len(pilot_ids & disc_ids) == 0
        ), f"Pilot items leak into discovery in {fname}"
        assert (
            len(pilot_ids & conf_ids) == 0
        ), f"Pilot items leak into confirmation in {fname}"

        # Text uniqueness
        if "context" in records[0]:
            pilot_texts = {r["context"] for r in records if r["split"] == "pilot"}
            conf_texts = {r["context"] for r in records if r["split"] == "confirmation"}
            assert len(pilot_texts & conf_texts) == 0
        elif "utterance" in records[0]:
            pilot_texts = {r["utterance"] for r in records if r["split"] == "pilot"}
            conf_texts = {
                r["utterance"] for r in records if r["split"] == "confirmation"
            }
            assert len(pilot_texts & conf_texts) == 0


def test_confirmatory_manifest_loader_rejects_pilot_split(tmp_path):
    """Verify that load_real_v2_manifest raises ValueError if any records belong to the pilot split."""
    pilot_record = {
        "trial_id": "trial_pilot_001",
        "experiment_id": "promptprism_real_v2_pilot",
        "git_commit": "HEAD",
        "task_id": "schema_extraction",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "provider": "local_vllm",
        "run_id": 1,
        "factor_levels": {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1},
        "sample_id": "pilot_001",
        "split": "pilot",
        "prompt_text": "Test prompt",
        "prompt_hash": "hash_prompt",
        "response_text": '{"name": "test"}',
        "response_hash": "hash_response",
        "metrics": {"field_f1": 0.85},
        "tokens_input": 100,
        "tokens_output": 50,
        "cost_usd": 0.0,
        "latency_ms": 250.0,
        "timestamp": "2026-09-26T20:00:00Z",
        "seed": 42,
        "data_origin": "real",
        "request_id": "req_pilot_1",
        "finish_reason": "stop",
        "system_fingerprint": "fp_test",
    }

    manifest_file = tmp_path / "mixed_pilot_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(pilot_record) + "\n")

    # load_real_v2_manifest should raise ValueError when loaded for confirmatory analysis
    with pytest.raises(ValueError, match="pilot|non-confirmatory"):
        ExperimentManifest(experiment_id="test_pilot", manifest_path=manifest_file)
        load_real_v2_manifest(manifest_file, allow_pilot=False)
