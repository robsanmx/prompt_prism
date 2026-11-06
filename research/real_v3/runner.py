"""Resumable prospective real-v3 Gemini experiment runner."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import pandas as pd

from research.experiments.providers import GeminiProviderAdapter, ModelRequest
from research.real_v3.analysis import (
    analyze_study,
    discovery_recommendations,
    summarize_operations,
)
from research.real_v3.datasets import DATA_ROOT, INTENT_LABELS, ROOT, build_all
from research.real_v3.design import (
    FACTOR_IDS,
    all_factorial_configs,
    balanced_random_designs,
    resolution_iv_run_ids,
)
from research.real_v3.locking import (
    assert_lock_matches,
    create_recommendation_lock,
    file_sha256,
)
from research.real_v3.metrics import (
    evaluate_entity_extraction,
    evaluate_grounded_qa,
    evaluate_intent,
)
from research.real_v3.prompts import render_prompt


EXPERIMENT_ID = "prompt-prism-real-v3-gemini-2026-09-26"
MODEL_ID = "gemini-2.5-flash"
RESULT_ROOT = ROOT / "research" / "results" / "real_gemini_v3"
PROTOCOL_ROOT = ROOT / "research" / "protocol" / "real_v3"
DISCOVERY_MANIFEST = RESULT_ROOT / "discovery_manifest.jsonl"
CONFIRMATION_MANIFEST = RESULT_ROOT / "confirmation_manifest.jsonl"
RECOMMENDATIONS = RESULT_ROOT / "recommendations.json"
RECOMMENDATION_LOCK = RESULT_ROOT / "recommendation_lock.json"
PROTOCOL_LOCK = PROTOCOL_ROOT / "protocol_lock.json"
DESIGN_REGISTRY = PROTOCOL_ROOT / "design_registry.json"
RANDOM_SEED = 20260926
BOOTSTRAP_ITERATIONS = 2000
PRIMARY_METRICS = {
    "schema_extraction": "entity_f1",
    "grounded_qa": "token_f1",
    "intent_classification": "accuracy",
}


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _protocol_inputs() -> list[Path]:
    return [
        PROTOCOL_ROOT / "preregistration.md",
        PROTOCOL_ROOT / "statistical_analysis_plan.md",
        DESIGN_REGISTRY,
        DATA_ROOT / "dataset_manifest.json",
        ROOT / "research" / "real_v3" / "analysis.py",
        ROOT / "research" / "real_v3" / "datasets.py",
        ROOT / "research" / "real_v3" / "design.py",
        ROOT / "research" / "real_v3" / "metrics.py",
        ROOT / "research" / "real_v3" / "prompts.py",
        ROOT / "research" / "real_v3" / "runner.py",
        ROOT / "research" / "experiments" / "providers.py",
    ]


def current_protocol_payload() -> dict:
    files = {str(path.relative_to(ROOT)): file_sha256(path) for path in _protocol_inputs()}
    return {
        "experiment_id": EXPERIMENT_ID,
        "files": files,
        "protocol_hash": _canonical_hash(files),
    }


def freeze_protocol() -> dict:
    payload = current_protocol_payload()
    payload["locally_frozen_at"] = datetime.now(timezone.utc).isoformat()
    PROTOCOL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_LOCK.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def freeze_design_registry() -> dict:
    configs = all_factorial_configs()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "factor_ids": FACTOR_IDS,
        "full_factorial_configs": {str(run): row for run, row in configs.items()},
        "resolution_iv_generators": ["E=ABC", "F=BCD"],
        "resolution_iv_run_ids": resolution_iv_run_ids(configs),
        "random_design_seed": RANDOM_SEED,
        "random_designs": balanced_random_designs(configs, count=20, seed=RANDOM_SEED),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": 42,
    }
    DESIGN_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    DESIGN_REGISTRY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _design_registry() -> dict:
    registry = json.loads(DESIGN_REGISTRY.read_text(encoding="utf-8"))
    configs = all_factorial_configs()
    expected = balanced_random_designs(configs, count=20, seed=RANDOM_SEED)
    if registry["random_designs"] != expected or registry["resolution_iv_run_ids"] != resolution_iv_run_ids(configs):
        raise ValueError("Frozen design registry does not match the registered design code")
    return registry


def assert_protocol_frozen() -> dict:
    frozen = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    current = current_protocol_payload()
    if frozen["protocol_hash"] != current["protocol_hash"] or frozen["files"] != current["files"]:
        raise ValueError("Current protocol/data/code does not match the pre-inference protocol lock")
    return frozen


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON on {path}:{number}") from exc
    return rows


def _load_items(task: str, split: str) -> list[dict]:
    return _load_jsonl(DATA_ROOT / "splits" / f"{task}_{split}.jsonl")


def _examples() -> dict:
    result = {}
    for task in PRIMARY_METRICS:
        rows = _load_items(task, "pilot")
        if task == "schema_extraction":
            result[task] = [(row, row["target"]) for row in rows[:2]]
        elif task == "grounded_qa":
            chosen = [row for row in rows if row["is_answerable"]][:2]
            result[task] = [(row, {"answer": row["answers"][0]}) for row in chosen]
        else:
            result[task] = [(row, {"intent": row["target"]}) for row in rows[:2]]
    return result


def _evaluate(task: str, response: str, item: dict) -> dict:
    if task == "schema_extraction":
        return evaluate_entity_extraction(response, item["target"])
    if task == "grounded_qa":
        return evaluate_grounded_qa(response, item["answers"], item["is_answerable"])
    return evaluate_intent(response, item["target"], INTENT_LABELS)


def _cache_key(task: str, split: str, run_id: int, item: dict, system: str, prompt: str) -> str:
    return _canonical_hash({
        "experiment_id": EXPERIMENT_ID,
        "task": task,
        "split": split,
        "run_id": run_id,
        "sample_id": item["sample_id"],
        "model": MODEL_ID,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_output_tokens": 512,
        "thinking_budget": 0,
        "system_prompt": system,
        "user_prompt": prompt,
    })


def _one_call(
    adapter: GeminiProviderAdapter,
    task: str,
    split: str,
    run_id: int,
    factors: dict,
    item: dict,
    examples: dict,
) -> dict:
    rendered = render_prompt(task, item, factors, examples, INTENT_LABELS)
    key = _cache_key(task, split, run_id, item, rendered.system_prompt, rendered.user_prompt)
    response = adapter.generate(ModelRequest(
        prompt=rendered.user_prompt,
        system_prompt=rendered.system_prompt or None,
        model_id=MODEL_ID,
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        seed=42,
        extra_params={"thinking_budget": 0},
    ))
    metrics = _evaluate(task, response.raw_text, item) if (
        not response.error and response.finish_reason == "STOP"
    ) else {
        PRIMARY_METRICS[task]: 0.0
    }
    score = float(metrics.get(PRIMARY_METRICS[task], 0.0))
    return {
        "schema_version": 3,
        "trial_id": key,
        "cache_key": key,
        "experiment_id": EXPERIMENT_ID,
        "task_id": task,
        "split": split,
        "run_id": run_id,
        "factor_levels": factors,
        **factors,
        "sample_id": item["sample_id"],
        "group_id": item["group_id"],
        "model_requested": MODEL_ID,
        "model_returned": response.model_id,
        "model_version": response.model_version,
        "provider": "google_gemini_generate_content",
        "parameters": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_output_tokens": 512,
            "thinking_budget": 0,
        },
        "system_prompt": rendered.system_prompt,
        "prompt_text": rendered.user_prompt,
        "prompt_hash": _canonical_hash({"system": rendered.system_prompt, "user": rendered.user_prompt}),
        "response_text": response.raw_text,
        "response_hash": _sha_text(response.raw_text),
        "metrics": metrics,
        "score": score,
        "request_id": response.request_id,
        "finish_reason": response.finish_reason,
        "tokens_input": response.tokens_input,
        "tokens_output": response.tokens_output,
        "tokens_thoughts": response.tokens_thoughts,
        "tokens_total": response.tokens_total,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "timestamp": response.timestamp,
        "retry_count": response.retry_count,
        "data_origin": response.data_origin,
        "error": response.error,
        "safety_ratings": response.safety_ratings,
        "prompt_feedback": response.prompt_feedback,
    }


def _planned_jobs(split: str) -> Iterable[tuple[str, int, dict, dict]]:
    configs = all_factorial_configs()
    for task in PRIMARY_METRICS:
        for item in _load_items(task, split):
            for run_id, factors in configs.items():
                yield task, run_id, factors, item


def run_split(split: str, workers: int) -> None:
    if split not in {"discovery", "confirmation"}:
        raise ValueError(split)
    frozen = assert_protocol_frozen()
    if split == "confirmation":
        assert_lock_matches(
            DISCOVERY_MANIFEST,
            RECOMMENDATIONS,
            RECOMMENDATION_LOCK,
            protocol_hash=frozen["protocol_hash"],
        )
    path = DISCOVERY_MANIFEST if split == "discovery" else CONFIRMATION_MANIFEST
    existing_rows = _load_jsonl(path)
    existing = {row["cache_key"] for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError(f"Duplicate cache keys in {path}")
    examples = _examples()
    jobs = []
    for task, run_id, factors, item in _planned_jobs(split):
        rendered = render_prompt(task, item, factors, examples, INTENT_LABELS)
        key = _cache_key(task, split, run_id, item, rendered.system_prompt, rendered.user_prompt)
        if key not in existing:
            jobs.append((task, run_id, factors, item))
    total = 3 * 80 * 64
    print(f"{split}: {len(existing)}/{total} recorded; {len(jobs)} requests remain", flush=True)
    if not jobs:
        return

    adapter = GeminiProviderAdapter(model_id=MODEL_ID, max_retries=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(existing)
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_one_call, adapter, task, split, run_id, factors, item, examples):
            (task, run_id, item["sample_id"])
            for task, run_id, factors, item in jobs
        }
        for future in as_completed(futures):
            task, run_id, sample_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                raise RuntimeError(f"Unrecorded call failure for {task}/{run_id}/{sample_id}") from exc
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = max(time.monotonic() - started, 0.001)
                rate = (completed - len(existing)) / elapsed
                print(
                    f"{split}: {completed}/{total} ({completed / total:.1%}), "
                    f"new rate {rate:.1f} calls/s, last={task}/run{run_id}",
                    flush=True,
                )


def _records_frame(rows: list[dict], task: str) -> pd.DataFrame:
    records = []
    for row in rows:
        if row["task_id"] == task:
            records.append({
                "task_id": task,
                "split": row["split"],
                "sample_id": row["sample_id"],
                "run_id": row["run_id"],
                "score": row["score"],
                **row["factor_levels"],
            })
    return pd.DataFrame(records)


def assert_complete(path: Path, split: str) -> list[dict]:
    rows = _load_jsonl(path)
    expected = 3 * 80 * 64
    keys = {row["cache_key"] for row in rows}
    cells = {(row["task_id"], row["sample_id"], row["run_id"]) for row in rows}
    if len(rows) != expected or len(keys) != expected or len(cells) != expected:
        raise ValueError(f"{path} is incomplete or contains duplicates: {len(rows)}/{expected}")
    if any(row["split"] != split or row.get("data_origin") != "real" for row in rows):
        raise ValueError(f"{path} has a split or data-origin violation")
    return rows


def lock_recommendations() -> dict:
    frozen = assert_protocol_frozen()
    rows = assert_complete(DISCOVERY_MANIFEST, "discovery")
    random_designs = _design_registry()["random_designs"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": frozen["protocol_hash"],
        "random_design_seed": RANDOM_SEED,
        "random_designs": random_designs,
        "tasks": {},
    }
    for task in PRIMARY_METRICS:
        payload["tasks"][task] = discovery_recommendations(
            _records_frame(rows, task), random_designs
        )
    RECOMMENDATIONS.parent.mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create_recommendation_lock(
        DISCOVERY_MANIFEST,
        RECOMMENDATIONS,
        RECOMMENDATION_LOCK,
        protocol_hash=frozen["protocol_hash"],
    )
    return payload


def analyze() -> dict:
    frozen = assert_protocol_frozen()
    assert_lock_matches(
        DISCOVERY_MANIFEST,
        RECOMMENDATIONS,
        RECOMMENDATION_LOCK,
        protocol_hash=frozen["protocol_hash"],
    )
    discovery = assert_complete(DISCOVERY_MANIFEST, "discovery")
    confirmation = assert_complete(CONFIRMATION_MANIFEST, "confirmation")
    records = discovery + confirmation
    recommendations = json.loads(RECOMMENDATIONS.read_text(encoding="utf-8"))
    frames = {
        task: pd.concat([
            _records_frame(discovery, task),
            _records_frame(confirmation, task),
        ], ignore_index=True)
        for task in PRIMARY_METRICS
    }
    summary = analyze_study(
        frames,
        recommendations,
        recommendations["random_designs"],
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        seed=42,
    )
    summary.update(summarize_operations(
        records,
        resolution_iv_run_ids(),
        recommendations["random_designs"],
    ))
    summary.update({
        "experiment_id": EXPERIMENT_ID,
        "model_requested": MODEL_ID,
        "model_versions_returned": sorted({row.get("model_version") for row in records if row.get("model_version")}),
        "protocol_hash": frozen["protocol_hash"],
        "discovery_manifest_sha256": file_sha256(DISCOVERY_MANIFEST),
        "confirmation_manifest_sha256": file_sha256(CONFIRMATION_MANIFEST),
        "recommendations_sha256": file_sha256(RECOMMENDATIONS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    flat_rows = []
    for task, row in summary["tasks"].items():
        flat_rows.append({"task_id": task, **{
            key: value for key, value in row.items() if not isinstance(value, (list, dict))
        }})
    pd.DataFrame(flat_rows).to_csv(RESULT_ROOT / "results.csv", index=False)
    return summary


def prepare() -> dict:
    if DISCOVERY_MANIFEST.exists() or CONFIRMATION_MANIFEST.exists():
        raise ValueError("Cannot refreeze protocol after inference manifests exist")
    build_all()
    freeze_design_registry()
    return freeze_protocol()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["prepare", "discovery", "lock", "confirmation", "analyze", "all"],
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
    elif args.command == "discovery":
        run_split("discovery", args.workers)
    elif args.command == "lock":
        print(json.dumps(lock_recommendations(), indent=2, sort_keys=True))
    elif args.command == "confirmation":
        run_split("confirmation", args.workers)
    elif args.command == "analyze":
        print(json.dumps(analyze(), indent=2, sort_keys=True))
    else:
        prepare()
        run_split("discovery", args.workers)
        lock_recommendations()
        run_split("confirmation", args.workers)
        print(json.dumps(analyze(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
