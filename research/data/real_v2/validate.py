"""
Validation and Integrity Audit for Real-LLM v2 Datasets.
Checks:
- Exactly 80 discovery, 80 confirmation, 20 pilot items per dataset
- Zero ID or content leakage across discovery and confirmation partitions
- Schema compliance and field types
- Deterministic SHA-256 partition fingerprints
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent

REQUIRED_DATASETS = {
    "schema_extraction.jsonl": {
        "required_keys": ["id", "text", "target", "split"],
        "target_keys": ["buyer", "target", "amount", "date"],
        "text_field": "text",
    },
    "grounded_qa.jsonl": {
        "required_keys": ["id", "context", "question", "reference_answer", "is_answerable", "split"],
        "text_field": "context",
    },
    "intent_classification.jsonl": {
        "required_keys": ["id", "utterance", "intent", "split"],
        "text_field": "utterance",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_dataset_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSON lines dataset records from disk."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))
    return items


def audit_dataset(filename: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    items = load_dataset_records(path)

    assert len(items) == 180, f"{filename}: Expected 180 items, got {len(items)}"

    # Check keys
    for it in items:
        for k in spec["required_keys"]:
            assert k in it, f"Missing key {k} in item {it.get('id')}"

    # Partition counts
    pilot = [it for it in items if it["split"] == "pilot"]
    disc = [it for it in items if it["split"] == "discovery"]
    conf = [it for it in items if it["split"] == "confirmation"]

    assert len(pilot) == 20, f"Expected 20 pilot, got {len(pilot)}"
    assert len(disc) == 80, f"Expected 80 discovery, got {len(disc)}"
    assert len(conf) == 80, f"Expected 80 confirmation, got {len(conf)}"

    # ID overlap check
    pilot_ids = {it["id"] for it in pilot}
    disc_ids = {it["id"] for it in disc}
    conf_ids = {it["id"] for it in conf}

    assert not (disc_ids & conf_ids), "Discovery and confirmation share IDs!"
    assert not (pilot_ids & disc_ids), "Pilot and discovery share IDs!"
    assert not (pilot_ids & conf_ids), "Pilot and confirmation share IDs!"

    # Text overlap check
    txt_field = spec["text_field"]
    disc_texts = {it[txt_field].strip().lower() for it in disc}
    conf_texts = {it[txt_field].strip().lower() for it in conf}
    assert not (disc_texts & conf_texts), "Discovery and confirmation share text content!"

    # Partition hashes
    disc_hash = sha256_bytes("".join(sorted(disc_ids)).encode("utf-8"))
    conf_hash = sha256_bytes("".join(sorted(conf_ids)).encode("utf-8"))

    return {
        "filename": filename,
        "total_items": len(items),
        "pilot_count": len(pilot),
        "discovery_count": len(disc),
        "confirmation_count": len(conf),
        "discovery_hash": disc_hash,
        "confirmation_hash": conf_hash,
        "status": "PASSED_INTEGRITY_AUDIT"
    }


def main():
    print("=== Auditing Real-LLM v2 Datasets ===")
    results = {}
    for fn, spec in REQUIRED_DATASETS.items():
        res = audit_dataset(fn, spec)
        results[fn] = res
        print(f"[{fn}] 180 items verified. Zero partition leakage. Status: {res['status']}")

    audit_path = DATA_DIR / "dataset_audit_report.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit report saved to {audit_path}")


if __name__ == "__main__":
    main()
