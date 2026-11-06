"""
Unit tests for Real-LLM v2 Datasets and Partition Isolation.
Verifies item counts, schema compliance, absence of ID or text overlap,
and reproducible split allocation.
"""

import json
from pathlib import Path

import pytest

from research.data.real_v2.validate import REQUIRED_DATASETS, audit_dataset

DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "research" / "data" / "real_v2"
)


@pytest.mark.parametrize("filename,spec", list(REQUIRED_DATASETS.items()))
def test_dataset_integrity_and_zero_leakage(filename, spec):
    """Verify that each real_v2 dataset passes integrity, schema, and leakage audits."""
    report = audit_dataset(filename, spec)
    assert report["total_items"] == 180
    assert report["pilot_count"] == 20
    assert report["discovery_count"] == 80
    assert report["confirmation_count"] == 80
    assert report["status"] == "PASSED_INTEGRITY_AUDIT"


def test_schema_field_types():
    """Verify schema extraction target structure."""
    path = DATA_DIR / "schema_extraction.jsonl"
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            target = item["target"]
            assert isinstance(target, dict)
            for k in ["buyer", "target", "amount", "date"]:
                assert k in target
                assert isinstance(target[k], str)


def test_intent_classes_distribution():
    """Verify customer intent categories are valid."""
    path = DATA_DIR / "intent_classification.jsonl"
    valid_classes = {
        "lost_card",
        "balance_inquiry",
        "update_address",
        "dispute_charge",
        "loan_application",
        "reset_credentials",
        "out_of_scope",
    }
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            assert item["intent"] in valid_classes
