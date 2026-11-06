"""Build immutable, group-disjoint real-v3 evaluation splits from public data."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable

from research.real_v3.design import stable_seed
from research.real_v3.locking import file_sha256


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "research" / "data" / "real_v3"
SOURCE_ROOT = DATA_ROOT / "sources"
SPLIT_SIZES = {"pilot": 20, "discovery": 80, "confirmation": 80}
ENTITY_TYPES = {
    "Person": "person",
    "Organisation": "organization",
    "Location": "location",
    "Money": "money",
    "Temporal": "temporal",
}
INTENT_LABELS = [
    "card_arrival",
    "card_delivery_estimate",
    "card_not_working",
    "virtual_card_not_working",
    "pending_card_payment",
    "declined_card_payment",
    "card_payment_not_recognised",
    "pending_cash_withdrawal",
    "declined_cash_withdrawal",
    "cash_withdrawal_not_recognised",
]


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sentence_spans(text: str) -> Iterable[tuple[int, int]]:
    starts = [0] + [match.end() for match in re.finditer(r"(?<=[.!?])\s+|\n+", text)]
    starts = sorted(set(starts))
    for index, begin in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if begin < end:
            yield begin, end


def build_entity_items() -> dict[str, list[dict]]:
    """Create sentence items while assigning whole documents to one split."""
    candidates: dict[str, list[dict]] = defaultdict(list)
    re3d_root = SOURCE_ROOT / "re3d"
    for source_dir in sorted(path for path in re3d_root.iterdir() if path.is_dir()):
        documents = {row["_id"]: row for row in _read_jsonl(source_dir / "documents.json")}
        entities: dict[str, list[dict]] = defaultdict(list)
        for entity in _read_jsonl(source_dir / "entities.json"):
            if entity["type"] in ENTITY_TYPES:
                entities[entity["documentId"]].append(entity)

        for document_id, document in documents.items():
            digest_bucket = stable_seed(document_id, base=42) % 100
            split = "pilot" if digest_bucket < 12 else (
                "discovery" if digest_bucket < 56 else "confirmation"
            )
            text = document["text"]
            for begin, end in _sentence_spans(text):
                if not 40 <= end - begin <= 600:
                    continue
                included = [
                    entity for entity in entities[document_id]
                    if entity["begin"] >= begin and entity["end"] <= end
                ]
                if not included:
                    continue
                target = {value: [] for value in ENTITY_TYPES.values()}
                seen = {value: set() for value in ENTITY_TYPES.values()}
                for entity in sorted(included, key=lambda row: (row["begin"], row["end"], row["type"])):
                    key = ENTITY_TYPES[entity["type"]]
                    value = text[entity["begin"]:entity["end"]]
                    if value not in seen[key]:
                        seen[key].add(value)
                        target[key].append(value)
                candidates[split].append({
                    "sample_id": f"re3d:{source_dir.name}:{document_id}:{begin}:{end}",
                    "group_id": f"re3d:{source_dir.name}:{document_id}",
                    "source": source_dir.name,
                    "source_url": document.get("sourceUrl"),
                    "text": text[begin:end],
                    "target": target,
                })

    selected = {}
    for split, size in SPLIT_SIZES.items():
        rows = candidates[split]
        random.Random(stable_seed(f"re3d:{split}")).shuffle(rows)
        if len(rows) < size:
            raise ValueError(f"Only {len(rows)} re3d candidates for {split}; need {size}")
        selected[split] = rows[:size]
    return selected


def build_qa_items() -> dict[str, list[dict]]:
    """Select one SQuAD question per context to prevent passage leakage."""
    source = SOURCE_ROOT / "squad2" / "dev-v2.0.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    contexts: list[dict] = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
            answerable = [qa for qa in paragraph["qas"] if not qa.get("is_impossible", False)]
            unanswerable = [qa for qa in paragraph["qas"] if qa.get("is_impossible", False)]
            if answerable and unanswerable:
                contexts.append({
                    "title": article["title"],
                    "context": context,
                    "context_hash": context_hash,
                    "answerable": answerable,
                    "unanswerable": unanswerable,
                })

    random.Random(stable_seed("squad:contexts")).shuffle(contexts)
    needed = sum(SPLIT_SIZES.values())
    if len(contexts) < needed:
        raise ValueError(f"Only {len(contexts)} eligible SQuAD contexts; need {needed}")

    selected: dict[str, list[dict]] = {}
    offset = 0
    unanswerable_counts = {"pilot": 3, "discovery": 12, "confirmation": 12}
    for split, size in SPLIT_SIZES.items():
        block = contexts[offset:offset + size]
        offset += size
        rows = []
        for index, record in enumerate(block):
            use_unanswerable = index < unanswerable_counts[split]
            pool = record["unanswerable"] if use_unanswerable else record["answerable"]
            qa = min(pool, key=lambda row: stable_seed(row["id"], base=113))
            answers = []
            for answer in qa.get("answers", []):
                if answer["text"] not in answers:
                    answers.append(answer["text"])
            rows.append({
                "sample_id": f"squad2:{qa['id']}",
                "group_id": f"squad2:{record['context_hash']}",
                "source": "SQuAD 2.0 dev",
                "title": record["title"],
                "context": record["context"],
                "question": qa["question"],
                "answers": answers,
                "is_answerable": not use_unanswerable,
            })
        random.Random(stable_seed(f"squad:{split}")).shuffle(rows)
        selected[split] = rows
    return selected


def build_intent_items() -> dict[str, list[dict]]:
    """Build ten-class Banking77 splits balanced within every split."""
    source = SOURCE_ROOT / "banking77" / "test.csv"
    by_label: dict[str, list[str]] = defaultdict(list)
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] in INTENT_LABELS:
                by_label[row["category"]].append(row["text"])

    selected: dict[str, list[dict]] = {split: [] for split in SPLIT_SIZES}
    allocations = [("pilot", 2), ("discovery", 8), ("confirmation", 8)]
    for label in INTENT_LABELS:
        utterances = by_label[label]
        random.Random(stable_seed(f"banking77:{label}")).shuffle(utterances)
        if len(utterances) < 18:
            raise ValueError(f"Only {len(utterances)} test utterances for {label}; need 18")
        offset = 0
        for split, count in allocations:
            for within_label, utterance in enumerate(utterances[offset:offset + count]):
                digest = hashlib.sha256(utterance.encode("utf-8")).hexdigest()[:16]
                selected[split].append({
                    "sample_id": f"banking77:{label}:{digest}",
                    "group_id": f"banking77:{label}:{digest}",
                    "source": "Banking77 official test split",
                    "utterance": utterance,
                    "target": label,
                })
            offset += count
    for split in selected:
        random.Random(stable_seed(f"banking77:{split}")).shuffle(selected[split])
    return selected


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_disjoint(task_splits: dict[str, list[dict]]) -> None:
    group_sets = {split: {row["group_id"] for row in rows} for split, rows in task_splits.items()}
    for first in SPLIT_SIZES:
        for second in SPLIT_SIZES:
            if first < second and group_sets[first] & group_sets[second]:
                raise ValueError(f"Group leakage between {first} and {second}")
    for split, expected in SPLIT_SIZES.items():
        rows = task_splits[split]
        if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
            raise ValueError(f"Invalid {split} count or duplicate IDs")


def build_all(output_root: Path = DATA_ROOT / "splits") -> dict:
    tasks = {
        "schema_extraction": build_entity_items(),
        "grounded_qa": build_qa_items(),
        "intent_classification": build_intent_items(),
    }
    files = {}
    summaries = {}
    for task_id, splits in tasks.items():
        _validate_disjoint(splits)
        summaries[task_id] = {}
        for split, rows in splits.items():
            path = output_root / f"{task_id}_{split}.jsonl"
            _write_jsonl(path, rows)
            files[str(path.relative_to(ROOT))] = file_sha256(path)
            summaries[task_id][split] = {
                "count": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
            }

    source_files = {}
    for path in sorted(SOURCE_ROOT.rglob("*")):
        if path.is_file():
            source_files[str(path.relative_to(ROOT))] = file_sha256(path)
    manifest = {
        "dataset_version": "real-v3",
        "selection_seed": 42,
        "split_sizes": SPLIT_SIZES,
        "intent_labels": INTENT_LABELS,
        "sources": {
            "re3d": {
                "repository": "https://github.com/dstl/re3d",
                "commit": "b061ff06713cad4df7e5cfaed3fd07bf15b2598c",
                "license": "per-source licenses retained beside source files",
            },
            "squad2": {
                "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
                "sha256": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
                "license": "CC BY-SA 4.0",
            },
            "banking77": {
                "repository": "https://github.com/PolyAI-LDN/task-specific-datasets",
                "commit": "57ec275d8078af65b7731c2a98be812d844a6d6b",
                "license": "CC BY 4.0",
            },
        },
        "source_file_sha256": source_files,
        "split_file_sha256": files,
        "summaries": summaries,
    }
    manifest_path = DATA_ROOT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2, sort_keys=True))
