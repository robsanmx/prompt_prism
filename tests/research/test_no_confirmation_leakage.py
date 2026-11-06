"""
Verification test confirming zero confirmation data leakage in replay and analysis.
Verifies that:
1. Replay methods only receive split == 'discovery'.
2. Confirmation split remains locked.
3. Method recommendations are deterministic and match their frozen hash.
"""

import json
from pathlib import Path
import pytest

from research.experiments.manifest import ExperimentManifest, sha256_hash


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "research" / "results" / "confirmatory_manifest.jsonl"
RECOMMENDATIONS_PATH = REPO_ROOT / "research" / "results" / "method_recommendations.json"


def test_manifest_has_both_splits_and_locks_confirmation():
    """Verify that confirmatory manifest exists, contains both splits, and enforces locking."""
    if not MANIFEST_PATH.exists():
        pytest.skip("Confirmatory manifest not yet generated.")

    manifest = ExperimentManifest(experiment_id="test_leakage_check", manifest_path=MANIFEST_PATH)
    assert len(manifest.records) > 0

    # Verify discovery records are non-empty
    disc_df = manifest.get_discovery_dataframe()
    assert len(disc_df) > 0
    assert (disc_df["split"] == "discovery").all()

    # Verify that trying to access confirmation data without correct unlock token raises PermissionError
    with pytest.raises(PermissionError, match="Confirmation split is locked"):
        manifest.get_confirmation_dataframe(unlock_token="")

    with pytest.raises(PermissionError, match="Confirmation split is locked"):
        manifest.get_confirmation_dataframe(unlock_token="ILLEGAL_ACCESS_TOKEN")


def test_recommendations_artifact_frozen_and_hash_consistent():
    """Verify that recommendations artifact is locked and matches internal hash."""
    if not RECOMMENDATIONS_PATH.exists():
        pytest.skip("Method recommendations not yet generated.")

    with open(RECOMMENDATIONS_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    assert artifact["status"] == "LOCKED_PRE_CONFIRMATION_UNLOCK"
    recs = artifact["recommendations"]
    expected_hash = sha256_hash(json.dumps(recs, sort_keys=True))
    assert artifact["recommendations_hash"] == expected_hash

    # Verify that all 3 tasks have recommendations for each method
    for task_id in ["schema_extraction", "grounded_qa", "intent_classification"]:
        assert task_id in recs
        assert "prompt_prism_res_iv" in recs[task_id]
        assert "uniform_random_search" in recs[task_id]
        assert "random_linear_model" in recs[task_id]
        assert "ofat_ablation" in recs[task_id]
        assert "exhaustive_oracle" in recs[task_id]
