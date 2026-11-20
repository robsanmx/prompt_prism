"""Acquire, partition, hash, and audit the six licensed Real-v4 task datasets (Task 8).

Guarantees:
1. Six objective tasks across five distinct capability families (`entity_f1`, `token_f1`,
   `exact_label_accuracy`, `numeric_exact_match`, `assertion_pass_rate`, `exact_choice_accuracy`).
2. Strict zero group or item overlap with `research/data/real_v3/splits/*`.
3. Whole source groups assigned to exactly one of (`pilot`, `screen`, `validation`,
   `clean_confirmation`, `stability_confirmation`), with complexity-stratified group allocation
   so `clean_confirmation` (240 items) and `stability_confirmation` (240 items) have matched
   structural difficulty distributions.
4. Complete source commits/URLs, licenses, raw file SHA-256s, split SHA-256s, and automated
   audit report (`audit_report.json`).
"""

from __future__ import annotations

import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import httpx

from research.real_v3.datasets import ENTITY_TYPES, _read_jsonl, _sentence_spans
from research.real_v4.schemas import CURRENT_SCHEMA_VERSION, canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
V3_DATA_ROOT = ROOT / "research" / "data" / "real_v3"
V4_DATA_ROOT = ROOT / "research" / "data" / "real_v4"
V4_SOURCES_DIR = V4_DATA_ROOT / "sources"
V4_SPLITS_DIR = V4_DATA_ROOT / "splits"
MANIFEST_PATH = V4_DATA_ROOT / "dataset_manifest.json"
AUDIT_REPORT_PATH = V4_DATA_ROOT / "audit_report.json"

PARTITION_SIZES: Dict[str, int] = {
    "pilot": 20,
    "screen": 120,
    "validation": 60,
    "clean_confirmation": 240,
    "stability_confirmation": 240,
}
PARTITION_ORDER: Tuple[str, ...] = (
    "pilot",
    "screen",
    "validation",
    "clean_confirmation",
    "stability_confirmation",
)
SELECTION_SEED: int = 20261118

SOURCE_METADATA: Dict[str, Dict[str, str]] = {
    "re3d": {
        "repository": "https://github.com/dstl/re3d",
        "revision": "b061ff06713cad4df7e5cfaed3fd07bf15b2598c",
        "license": "Open Government / CC-BY per-corpus licenses retained alongside source files",
    },
    "few_nerd_supervised": {
        "repository": "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
        "revision": "205f3e9c9f35ea1ce5b05f49aa04f6b29a1f7191",
        "license": "CC BY-SA 4.0",
    },
    "squad2": {
        "repository": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "revision": "sha256:80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
        "license": "CC BY-SA 4.0",
    },
    "banking77": {
        "repository": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "license": "CC BY 4.0",
    },
    "gsm8k": {
        "repository": "https://github.com/openai/grade-school-math",
        "revision": "master/grade_school_math/data/test.jsonl",
        "license": "MIT License",
    },
    "mbpp": {
        "repository": "https://github.com/google-research/google-research/tree/master/mbpp",
        "revision": "master/mbpp/mbpp.jsonl",
        "license": "CC BY 4.0",
    },
    "sciq": {
        "repository": "https://huggingface.co/datasets/allenai/sciq",
        "revision": "2c94ad3e1aaf4e780ef5dc40d2fc0254b84bd5b4",
        "license": "CC BY-NC 3.0",
    },
}

_TEMPORAL_OR_AMBIGUOUS_RE = re.compile(
    r"\b(\d{4}s?|\d{1,2}(?:st|nd|rd|th)|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|yesterday|tomorrow|century|decade)\b",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_seed(label: str, base: int = SELECTION_SEED) -> int:
    digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def load_v3_quarantine_sets() -> Tuple[Set[str], Set[str], Set[str]]:
    """Return `(v3_group_ids, v3_sample_ids, v3_banking_labels)` strictly quarantined from v4."""
    groups: Set[str] = set()
    samples: Set[str] = set()
    v3_splits = V3_DATA_ROOT / "splits"
    for path in sorted(v3_splits.glob("*.jsonl")):
        for row in _read_jsonl(path):
            if "group_id" in row:
                groups.add(str(row["group_id"]))
            if "sample_id" in row:
                samples.add(str(row["sample_id"]))
    v3_manifest = json.loads((V3_DATA_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))
    v3_labels = set(v3_manifest.get("intent_labels", []))
    return groups, samples, v3_labels


def ensure_cached_sources(sources_dir: Path = V4_SOURCES_DIR) -> Dict[str, Path]:
    """Download and cache GSM8K, MBPP, SciQ, and Few-NERD raw source files once if missing."""
    sources_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gsm8k": sources_dir / "gsm8k_test.jsonl",
        "mbpp": sources_dir / "mbpp_full.jsonl",
        "sciq": sources_dir / "sciq_test_val.jsonl",
        "few_nerd": sources_dir / "few_nerd_supervised_test.jsonl",
    }

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        if not paths["gsm8k"].exists():
            r = client.get(
                "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
            )
            r.raise_for_status()
            paths["gsm8k"].write_bytes(r.content)

        if not paths["mbpp"].exists():
            r = client.get(
                "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
            )
            r.raise_for_status()
            paths["mbpp"].write_bytes(r.content)

        if not paths["sciq"].exists():
            rows_out: List[str] = []
            for split_name in ("test", "validation"):
                for offset in range(0, 1000, 100):
                    url = (
                        f"https://datasets-server.huggingface.co/rows?"
                        f"dataset=allenai%2Fsciq&config=default&split={split_name}&offset={offset}&length=100"
                    )
                    resp = client.get(url)
                    resp.raise_for_status()
                    for entry in resp.json().get("rows", []):
                        rec = {"split_origin": split_name, "row_idx": entry["row_idx"], **entry["row"]}
                        rows_out.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            paths["sciq"].write_text("\n".join(rows_out) + "\n", encoding="utf-8")

        if not paths["few_nerd"].exists():
            fn_rows: List[str] = []
            for offset in range(0, 2200, 100):
                url = (
                    f"https://datasets-server.huggingface.co/rows?"
                    f"dataset=DFKI-SLT%2Ffew-nerd&config=supervised&split=test&offset={offset}&length=100"
                )
                resp = client.get(url)
                resp.raise_for_status()
                batch = resp.json().get("rows", [])
                if not batch:
                    break
                for entry in batch:
                    rec = {"row_idx": entry["row_idx"], **entry["row"]}
                    fn_rows.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            paths["few_nerd"].write_text("\n".join(fn_rows) + "\n", encoding="utf-8")

    return paths


def _allocate_groups_stratified(
    task_id: str,
    groups_to_items: Mapping[str, List[Dict[str, Any]]],
    difficulty_fn,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministically allocate whole source groups to partitions with matched difficulty strata.

    Uses one item per group (or whole group up to 2 items) while strictly guaranteeing that no
    `group_id` ever appears in more than one partition, and matching `clean_confirmation` and
    `stability_confirmation` across difficulty quantiles.
    """
    total_needed = sum(PARTITION_SIZES.values())  # 680

    # Take 1 canonical representative item per group first if enough unique groups exist;
    # otherwise allow whole multi-item groups without ever splitting a group!
    unique_group_ids = sorted(groups_to_items.keys())
    rng = random.Random(_stable_seed(f"alloc:{task_id}"))

    if len(unique_group_ids) >= total_needed:
        one_per_group: List[Tuple[float, str, Dict[str, Any]]] = []
        for gid in unique_group_ids:
            items = sorted(groups_to_items[gid], key=lambda x: str(x["sample_id"]))
            chosen = items[0]
            one_per_group.append((float(difficulty_fn(chosen)), gid, chosen))
        # Shuffle within ties deterministically, then sort by difficulty metric
        rng.shuffle(one_per_group)
        one_per_group.sort(key=lambda pair: pair[0])
        pool = [item for (_d, _g, item) in one_per_group[:total_needed]]

        # Interleave across strata of 34 items (ratio 1 : 6 : 3 : 12 : 12 = 34 -> x 20 = 680!)
        buckets: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PARTITION_ORDER}
        block_template = (
            ["pilot"] * 1
            + ["screen"] * 6
            + ["validation"] * 3
            + ["clean_confirmation"] * 12
            + ["stability_confirmation"] * 12
        )  # length 34; 34 * 20 = 680 exact!
        for block_idx in range(20):
            block = pool[block_idx * 34 : (block_idx + 1) * 34]
            perm = list(block_template)
            rng.shuffle(perm)
            for item, part in zip(block, perm):
                buckets[part].append(item)
        return buckets

    # Multi-item whole-group allocation fallback when unique groups < 680
    group_entries = []
    for gid in unique_group_ids:
        items = sorted(groups_to_items[gid], key=lambda x: str(x["sample_id"]))[:3]
        avg_diff = sum(float(difficulty_fn(it)) for it in items) / len(items)
        group_entries.append((avg_diff, gid, items))
    rng.shuffle(group_entries)
    group_entries.sort(key=lambda t: t[0])

    buckets = {p: [] for p in PARTITION_ORDER}
    used_groups: Set[str] = set()
    # Interleaved greedy fill by remaining deficit ratio across sorted difficulty groups
    for _diff, gid, items in group_entries:
        if gid in used_groups:
            continue
        deficits = [
            ( (PARTITION_SIZES[p] - len(buckets[p])) / PARTITION_SIZES[p], p )
            for p in PARTITION_ORDER
            if len(buckets[p]) < PARTITION_SIZES[p]
        ]
        if not deficits:
            break
        deficits.sort(key=lambda pair: (-pair[0], PARTITION_ORDER.index(pair[1])))
        chosen_p = deficits[0][1]
        need = PARTITION_SIZES[chosen_p] - len(buckets[chosen_p])
        buckets[chosen_p].extend(items[:need])
        used_groups.add(gid)

    for p, req in PARTITION_SIZES.items():
        if len(buckets[p]) != req:
            raise RuntimeError(f"Task {task_id} partition {p} has {len(buckets[p])} items, need {req}")
    return buckets


def _build_re3d_entity_groups(
    v3_groups: Set[str],
    v3_samples: Set[str],
    few_nerd_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 1. Unused Re3d documents (zero overlap with real-v3)
    re3d_root = V3_DATA_ROOT / "sources" / "re3d"
    for sdir in sorted(p for p in re3d_root.iterdir() if p.is_dir()):
        docs = {r["_id"]: r for r in _read_jsonl(sdir / "documents.json")}
        ents: Dict[str, List[dict]] = defaultdict(list)
        for e in _read_jsonl(sdir / "entities.json"):
            if e["type"] in ENTITY_TYPES:
                ents[e["documentId"]].append(e)
        for did, doc in sorted(docs.items()):
            gid = f"re3d:{sdir.name}:{did}"
            if gid in v3_groups:
                continue
            text = doc["text"]
            for begin, end in _sentence_spans(text):
                if not (40 <= end - begin <= 600):
                    continue
                sid = f"re3d:{sdir.name}:{did}:{begin}:{end}"
                if sid in v3_samples:
                    continue
                inc = [e for e in ents[did] if e["begin"] >= begin and e["end"] <= end]
                if not inc:
                    continue
                target = {k: [] for k in ENTITY_TYPES.values()}
                seen = {k: set() for k in ENTITY_TYPES.values()}
                for ent in sorted(inc, key=lambda r: (r["begin"], r["end"], r["type"])):
                    key = ENTITY_TYPES[ent["type"]]
                    val = text[ent["begin"] : ent["end"]].strip()
                    if val and val not in seen[key]:
                        seen[key].add(val)
                        target[key].append(val)
                groups[gid].append(
                    {
                        "task_id": "re3d_entity_extraction",
                        "sample_id": sid,
                        "group_id": gid,
                        "source": f"re3d/{sdir.name}",
                        "input_text": text[begin:end],
                        "target": target,
                    }
                )

    # 2. Few-NERD supervised test supplement (CC BY-SA 4.0), filtered for unambiguous date-free spans
    # Coarse tags: 4=location, 5=organization, 7=person; Fine tag: 42=other-currency (money)
    for row in _read_jsonl(few_nerd_path):
        tokens: List[str] = row["tokens"]
        coarse: List[int] = row["ner_tags"]
        fine: List[int] = row["fine_ner_tags"]
        if not (8 <= len(tokens) <= 55):
            continue
        sentence = " ".join(tokens)
        if _TEMPORAL_OR_AMBIGUOUS_RE.search(sentence):
            continue

        target = {k: [] for k in ("person", "organization", "location", "money", "temporal")}
        seen = {k: set() for k in target}
        n_tok = len(tokens)
        idx = 0
        while idx < n_tok:
            c_tag = coarse[idx]
            f_tag = fine[idx]
            mapped: str | None = None
            if c_tag == 7:
                mapped = "person"
            elif c_tag == 5:
                mapped = "organization"
            elif c_tag == 4:
                mapped = "location"
            elif f_tag == 42:
                mapped = "money"

            if mapped is None:
                idx += 1
                continue
            end_idx = idx + 1
            while end_idx < n_tok and (
                (mapped == "money" and fine[end_idx] == 42)
                or (mapped != "money" and coarse[end_idx] == c_tag)
            ):
                end_idx += 1
            span_text = " ".join(tokens[idx:end_idx]).strip()
            if span_text and span_text not in seen[mapped]:
                seen[mapped].add(span_text)
                target[mapped].append(span_text)
            idx = end_idx

        total_ents = sum(len(v) for v in target.values())
        if total_ents == 0:
            continue
        # Group consecutive 2-row document blocks in Few-NERD so related adjacent sentences never split
        block_id = int(row["row_idx"]) // 2
        gid = f"few_nerd:doc_block_{block_id:05d}"
        sid = f"few_nerd:row_{int(row['row_idx']):05d}"
        groups[gid].append(
            {
                "task_id": "re3d_entity_extraction",
                "sample_id": sid,
                "group_id": gid,
                "source": "DFKI-SLT/few-nerd",
                "input_text": sentence,
                "target": target,
            }
        )
    return groups


def _build_squad2_groups(v3_groups: Set[str], v3_samples: Set[str]) -> Dict[str, List[Dict[str, Any]]]:
    squad_path = V3_DATA_ROOT / "sources" / "squad2" / "dev-v2.0.json"
    payload = json.loads(squad_path.read_text(encoding="utf-8"))
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for article in payload["data"]:
        title = article["title"]
        for p_idx, para in enumerate(article["paragraphs"]):
            context = para["context"].strip()
            if not (160 <= len(context) <= 1600):
                continue
            ctx_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
            v3_gid = f"squad2:{ctx_hash}"
            if v3_gid in v3_groups:
                continue
            gid = f"squad2_v4:{title}:{p_idx}:{ctx_hash}"
            for qa in para["qas"]:
                qid = str(qa["id"])
                if f"squad2:{qid}" in v3_samples:
                    continue
                is_imp = bool(qa.get("is_impossible", False))
                answers = [] if is_imp else sorted({a["text"].strip() for a in qa.get("answers", []) if a["text"].strip()})
                if not is_imp and not answers:
                    continue
                groups[gid].append(
                    {
                        "task_id": "squad2_grounded_qa",
                        "sample_id": f"squad2_v4:{qid}",
                        "group_id": gid,
                        "source": "SQuAD2.0-dev",
                        "context": context,
                        "question": qa["question"].strip(),
                        "is_unanswerable": is_imp,
                        "target": answers,
                    }
                )
    return groups


def _build_banking77_groups(
    v3_groups: Set[str],
    v3_samples: Set[str],
    v3_labels: Set[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    b_path = V3_DATA_ROOT / "sources" / "banking77" / "test.csv"
    cats_path = V3_DATA_ROOT / "sources" / "banking77" / "categories.json"
    all_cats = json.loads(cats_path.read_text(encoding="utf-8"))
    allowed_cats = sorted(c for c in all_cats if c not in v3_labels)[:20]
    allowed_set = set(allowed_cats)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with b_path.open(encoding="utf-8") as handle:
        for row_idx, row in enumerate(csv.DictReader(handle)):
            cat = row["category"].strip()
            text = row["text"].strip()
            if cat not in allowed_set or not text:
                continue
            text_hash = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16]
            if any(text_hash in g for g in v3_groups) or any(text_hash in s for s in v3_samples):
                continue
            # Group by normalized 3-token prefix + category so near-duplicate utterance templates stay together
            norm_words = re.findall(r"[a-z0-9]+", text.lower())
            stem = "_".join(norm_words[:3]) if norm_words else f"r{row_idx}"
            gid = f"banking77_v4:{cat}:{stem}:{row_idx}"
            groups[gid].append(
                {
                    "task_id": "banking77_intent",
                    "sample_id": f"banking77_v4:{row_idx:04d}",
                    "group_id": gid,
                    "source": "Banking77-test",
                    "input_text": text,
                    "options": allowed_cats,
                    "target": cat,
                }
            )
    return groups, allowed_cats


def _build_gsm8k_groups(gsm_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(_read_jsonl(gsm_path)):
        q = str(row["question"]).strip()
        ans = str(row["answer"]).strip()
        m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", ans)
        if not m or not q:
            continue
        target_num = m.group(1).replace(",", "")
        q_hash = hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]
        gid = f"gsm8k:prob_{idx:04d}:{q_hash}"
        groups[gid].append(
            {
                "task_id": "gsm8k_math",
                "sample_id": f"gsm8k:{idx:04d}",
                "group_id": gid,
                "source": "GSM8K-test",
                "question": q,
                "target": target_num,
            }
        )
    return groups


def _extract_signature_hint(code_str: str) -> str | None:
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not funcs:
        return None
    fn = funcs[-1]
    args = [a.arg for a in fn.args.args]
    return f"def {fn.name}({', '.join(args)}):"


def _build_mbpp_groups(mbpp_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(mbpp_path):
        if str(row.get("test_setup_code", "")).strip():
            continue
        code = str(row["code"]).replace("\r\n", "\n")
        sig = _extract_signature_hint(code)
        if not sig:
            continue
        tests = [str(t).strip() for t in row.get("test_list", []) if str(t).strip()]
        if not tests:
            continue
        tid = int(row["task_id"])
        gid = f"mbpp:task_{tid:04d}"
        groups[gid].append(
            {
                "task_id": "mbpp_python",
                "sample_id": f"mbpp:{tid:04d}",
                "group_id": gid,
                "source": "MBPP-full",
                "prompt": str(row["text"]).strip(),
                "signature_hint": sig,
                "reference_code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                "target": tests,
            }
        )
    return groups


def _build_sciq_groups(sciq_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(_read_jsonl(sciq_path)):
        support = str(row.get("support", "")).strip()
        question = str(row.get("question", "")).strip()
        correct = str(row.get("correct_answer", "")).strip()
        dists = [
            str(row.get("distractor1", "")).strip(),
            str(row.get("distractor2", "")).strip(),
            str(row.get("distractor3", "")).strip(),
        ]
        if len(support) < 40 or not question or not correct or any(not d for d in dists):
            continue
        choices = [correct, *dists]
        if len({c.lower() for c in choices}) != 4:
            continue
        rng = random.Random(_stable_seed(f"sciq_opt:{idx}:{question}"))
        rng.shuffle(choices)
        letters = ("A", "B", "C", "D")
        opt_map = {letters[k]: choices[k] for k in range(4)}
        target_letter = next(k for k, v in opt_map.items() if v == correct)

        # Group by normalized first 60 chars of support passage so shared-concept questions never split
        stem_key = re.sub(r"\s+", " ", support.lower())[:60]
        stem_hash = hashlib.sha256(stem_key.encode("utf-8")).hexdigest()[:16]
        gid = f"sciq:concept_{stem_hash}"
        groups[gid].append(
            {
                "task_id": "sciq_verification",
                "sample_id": f"sciq:{row['split_origin']}:{int(row['row_idx']):04d}",
                "group_id": gid,
                "source": f"SciQ-{row['split_origin']}",
                "support": support,
                "question": question,
                "options": opt_map,
                "target": target_letter,
            }
        )
    return groups


def audit_real_v4_datasets(
    manifest: Mapping[str, Any],
    splits_dir: Path = V4_SPLITS_DIR,
) -> Dict[str, Any]:
    """Run automated license, hash, count, target, group-disjointness, and v3-quarantine audits."""
    v3_groups, v3_samples, v3_labels = load_v3_quarantine_sets()
    expected_tasks = (
        "re3d_entity_extraction",
        "squad2_grounded_qa",
        "banking77_intent",
        "gsm8k_math",
        "mbpp_python",
        "sciq_verification",
    )

    errors: List[str] = []
    for src_id, meta in manifest.get("sources", {}).items():
        if not meta.get("license") or not meta.get("revision") or not meta.get("repository"):
            errors.append(f"incomplete_source_license_or_revision:{src_id}")

    for rel_path, expected_sha in manifest.get("source_file_sha256", {}).items():
        full_p = ROOT / rel_path
        if not full_p.exists() or sha256_file(full_p) != expected_sha:
            errors.append(f"raw_source_hash_mismatch:{rel_path}")

    task_audits: Dict[str, Any] = {}
    global_sample_ids: Set[str] = set()

    for task_id in expected_tasks:
        seen_groups_by_part: Dict[str, Set[str]] = {}
        part_counts: Dict[str, int] = {}
        for part in PARTITION_ORDER:
            split_file = splits_dir / f"{task_id}_{part}.jsonl"
            rel_split = f"research/data/real_v4/splits/{split_file.name}"
            if not split_file.exists():
                errors.append(f"missing_split_file:{rel_split}")
                continue
            actual_sha = sha256_file(split_file)
            if actual_sha != manifest["split_file_sha256"].get(rel_split):
                errors.append(f"split_hash_mismatch:{rel_split}")

            rows = list(_read_jsonl(split_file))
            part_counts[part] = len(rows)
            if len(rows) != PARTITION_SIZES[part]:
                errors.append(f"count_mismatch:{task_id}:{part}:got={len(rows)},expected={PARTITION_SIZES[part]}")

            p_groups: Set[str] = set()
            for r in rows:
                sid = str(r.get("sample_id", ""))
                gid = str(r.get("group_id", ""))
                if not sid or not gid:
                    errors.append(f"missing_id:{task_id}:{part}")
                if sid in global_sample_ids:
                    errors.append(f"duplicate_sample_id:{sid}")
                global_sample_ids.add(sid)
                p_groups.add(gid)

                if gid in v3_groups or sid in v3_samples:
                    errors.append(f"v3_leakage:{sid}:{gid}")
                if task_id == "banking77_intent" and r.get("target") in v3_labels:
                    errors.append(f"v3_banking_label_leakage:{r.get('target')}")
                if "target" not in r or r["target"] is None or (
                    task_id != "squad2_grounded_qa" and r["target"] == ""
                ):
                    errors.append(f"empty_target:{sid}")
            seen_groups_by_part[part] = p_groups

        # Check pairwise group disjointness across ALL 5 partitions
        cross_overlaps = 0
        parts_list = list(PARTITION_ORDER)
        for i in range(len(parts_list)):
            for j in range(i + 1, len(parts_list)):
                inter = seen_groups_by_part.get(parts_list[i], set()) & seen_groups_by_part.get(parts_list[j], set())
                if inter:
                    cross_overlaps += len(inter)
                    errors.append(f"group_overlap:{task_id}:{parts_list[i]}:{parts_list[j]}:{len(inter)}")

        task_audits[task_id] = {
            "partition_counts": part_counts,
            "unique_groups_per_partition": {p: len(g) for p, g in seen_groups_by_part.items()},
            "cross_partition_group_overlaps": cross_overlaps,
        }

    report = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": "prompt-prism-real-v4-selection-stability-2026",
        "audit_passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "total_items_across_tasks": len(global_sample_ids),
        "tasks": task_audits,
    }
    report["audit_sha256"] = canonical_sha256(report)
    return report


def build_all_real_v4_datasets() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build all 6 group-disjoint Real-v4 task splits, manifest, and audit report."""
    cached_paths = ensure_cached_sources(V4_SOURCES_DIR)
    v3_groups, v3_samples, v3_labels = load_v3_quarantine_sets()

    re3d_groups = _build_re3d_entity_groups(v3_groups, v3_samples, cached_paths["few_nerd"])
    squad_groups = _build_squad2_groups(v3_groups, v3_samples)
    bank_groups, bank_labels = _build_banking77_groups(v3_groups, v3_samples, v3_labels)
    gsm_groups = _build_gsm8k_groups(cached_paths["gsm8k"])
    mbpp_groups = _build_mbpp_groups(cached_paths["mbpp"])
    sciq_groups = _build_sciq_groups(cached_paths["sciq"])

    allocated = {
        "re3d_entity_extraction": _allocate_groups_stratified(
            "re3d_entity_extraction", re3d_groups, lambda r: len(r["input_text"])
        ),
        "squad2_grounded_qa": _allocate_groups_stratified(
            "squad2_grounded_qa", squad_groups, lambda r: len(r["context"]) + 200 * int(r["is_unanswerable"])
        ),
        "banking77_intent": _allocate_groups_stratified(
            "banking77_intent", bank_groups, lambda r: len(r["input_text"])
        ),
        "gsm8k_math": _allocate_groups_stratified(
            "gsm8k_math", gsm_groups, lambda r: len(r["question"])
        ),
        "mbpp_python": _allocate_groups_stratified(
            "mbpp_python", mbpp_groups, lambda r: len(r["prompt"]) + 50 * len(r["target"])
        ),
        "sciq_verification": _allocate_groups_stratified(
            "sciq_verification", sciq_groups, lambda r: len(r["support"])
        ),
    }

    V4_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_hashes: Dict[str, str] = {}
    summaries: Dict[str, Any] = {}

    for task_id, part_map in sorted(allocated.items()):
        summaries[task_id] = {}
        for part in PARTITION_ORDER:
            rows = sorted(part_map[part], key=lambda r: str(r["sample_id"]))
            out_file = V4_SPLITS_DIR / f"{task_id}_{part}.jsonl"
            out_file.write_text(
                "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            rel = out_file.relative_to(ROOT).as_posix()
            split_hashes[rel] = sha256_file(out_file)
            summaries[task_id][part] = {
                "count": len(rows),
                "unique_groups": len({r["group_id"] for r in rows}),
            }

    # Hash all raw source files referenced across v3-cached and v4-cached sources
    raw_source_paths = [
        V3_DATA_ROOT / "sources" / "banking77" / "LICENSE",
        V3_DATA_ROOT / "sources" / "banking77" / "categories.json",
        V3_DATA_ROOT / "sources" / "banking77" / "test.csv",
        V3_DATA_ROOT / "sources" / "squad2" / "dev-v2.0.json",
        *sorted((V3_DATA_ROOT / "sources" / "re3d").rglob("*")),
        *sorted(cached_paths.values()),
    ]
    source_hashes = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in raw_source_paths
        if p.is_file()
    }

    manifest: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "dataset_version": "real-v4.0.0",
        "study_id": "prompt-prism-real-v4-selection-stability-2026",
        "selection_seed": SELECTION_SEED,
        "partition_sizes": PARTITION_SIZES,
        "banking77_v4_intent_labels": bank_labels,
        "sources": SOURCE_METADATA,
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "split_file_sha256": dict(sorted(split_hashes.items())),
        "summaries": summaries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit_report = audit_real_v4_datasets(manifest, V4_SPLITS_DIR)
    AUDIT_REPORT_PATH.write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit_report["audit_passed"]:
        raise RuntimeError(f"Real-v4 dataset audit failed: {audit_report['errors'][:5]}")
    return manifest, audit_report


if __name__ == "__main__":
    manifest_out, audit_out = build_all_real_v4_datasets()
    print(f"Built Real-v4 datasets: {audit_out['total_items_across_tasks']} items, audit_passed={audit_out['audit_passed']}")
