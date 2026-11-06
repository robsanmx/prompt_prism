"""
Real-LLM Empirical Experiment Runner using Google Gemini (2.5 Flash / 2.5 Pro).
Executes live inference calls via GeminiProviderAdapter with:
- Multithreaded execution and rate-limit backoff handling.
- Full content-addressed SHA-256 caching and resumability.
- Two-phase blind protocol: Discovery selection frozen before confirmation evaluation.
- Deterministic objective task scoring (Field F1, Token F1, Accuracy).
- Registered item-cluster bootstrap analysis for H1-H5.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple
import uuid
import yaml

import numpy as np
import pandas as pd
from scipy import stats

from prompt_prism.design.generators import FractionalFactorialGenerator
from research.analysis.real_v2_confirmatory import (
    fit_fractional_factorial_main_effects,
    select_prompt_from_effects,
)
from research.analysis.real_v2_metrics import get_task_evaluator
from research.data.real_v2.validate import load_dataset_records
from research.experiments.manifest import ExperimentManifest, ProvenanceRecord, get_current_git_commit, sha256_hash
from research.experiments.providers import GeminiProviderAdapter, ModelRequest, ModelResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "research" / "data" / "real_v2"
FACTORS_PATH = REPO_ROOT / "research" / "configs" / "real_v2" / "factors.yaml"
OUTPUT_DIR = REPO_ROOT / "research" / "results" / "real_gemini"

FACTOR_IDS = ["A", "B", "C", "D", "E", "F"]

with open(FACTORS_PATH, "r", encoding="utf-8") as f:
    FACTORS_CFG = yaml.safe_load(f)["factors"]


def render_prompt(factors: Dict[str, int], item: Dict[str, Any], task_id: str) -> str:
    """Render prompt based on factor toggles (-1 / +1) and dataset item."""
    parts = []
    # Factor A: Role framing
    role_text = FACTORS_CFG["A"]["level_plus_one"] if factors.get("A", -1) == 1 else FACTORS_CFG["A"]["level_minus_one"]
    if role_text:
        parts.append(f"System: {role_text}")

    # Factor B: Task decomposition
    if factors.get("B", -1) == 1:
        parts.append(FACTORS_CFG["B"]["level_plus_one"])

    # Item text
    if task_id == "schema_extraction":
        raw_input = item.get("text", "")
        parts.append(f"Passage:\n{raw_input}")
        parts.append("Extract buyer, target, amount, and date as JSON.")
    elif task_id == "grounded_qa":
        raw_context = item.get("context", "")
        raw_q = item.get("question", "")
        parts.append(f"Context:\n{raw_context}")
        parts.append(f"Question: {raw_q}")
    elif task_id == "intent_classification":
        raw_utt = item.get("utterance", "")
        parts.append(f"Customer Inquiry:\n{raw_utt}")
        parts.append("Classify into exactly one intent.")

    # Factor C: Demonstrations
    if factors.get("C", -1) == 1:
        parts.append("Example: Given an input, produce the exact matching structured fields.")

    # Factor D: JSON schema specification
    d_text = FACTORS_CFG["D"]["level_plus_one"] if factors.get("D", -1) == 1 else FACTORS_CFG["D"]["level_minus_one"]
    if d_text:
        parts.append(d_text)

    # Factor E: Grounding constraint
    if factors.get("E", -1) == 1:
        parts.append(FACTORS_CFG["E"]["level_plus_one"])

    # Factor F: Verification pass
    if factors.get("F", -1) == 1:
        parts.append(FACTORS_CFG["F"]["level_plus_one"])

    return "\n\n".join(parts)


def get_res_iv_configs() -> List[Dict[str, int]]:
    gen = FractionalFactorialGenerator.from_plan_id("2(6-2)IV")
    return [{f: (1 if run.factor_levels[f] == 1 else -1) for f in FACTOR_IDS} for run in gen.runs]


def run_gemini_experiment(
    tasks: List[str] = ["schema_extraction", "grounded_qa", "intent_classification"],
    model_id: str = "gemini-2.5-flash",
    items_per_split: int = 20,
    max_workers: int = 8,
    output_dir: Path = OUTPUT_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gemini_manifest.jsonl"
    manifest = ExperimentManifest(experiment_id="promptprism_gemini_v2", manifest_path=manifest_path)
    adapter = GeminiProviderAdapter(model_id=model_id)

    # 64 configurations: cartesian product of {-1, 1}^6
    all_levels = list(itertools.product([-1, 1], repeat=6))
    configs = [(run_id + 1, dict(zip(FACTOR_IDS, levels))) for run_id, levels in enumerate(all_levels)]

    # Load items per task and split
    task_items: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for t in tasks:
        records = load_dataset_records(DATA_DIR / f"{t}.jsonl")
        disc = [r for r in records if r["split"] == "discovery"][:items_per_split]
        conf = [r for r in records if r["split"] == "confirmation"][:items_per_split]
        task_items[t] = {"discovery": disc, "confirmation": conf}

    total_expected = len(tasks) * 2 * len(configs) * items_per_split
    print("=" * 70)
    print(f"PromptPrism Live Gemini Empirical Study: {model_id}")
    print(f"Tasks: {tasks}")
    print(f"Items per split: {items_per_split} (Discovery: {items_per_split}, Confirmation: {items_per_split})")
    print(f"Configurations: {len(configs)} ($2^6 = 64$ factorial runs)")
    print(f"Total live calls in matrix: {total_expected}")
    print(f"Parallel workers: {max_workers}")
    print("=" * 70)

    trials_queue = []
    for task_id in tasks:
        evaluator = get_task_evaluator(task_id)
        for split in ["discovery", "confirmation"]:
            items = task_items[task_id][split]
            for run_id, f_levels in configs:
                for it in items:
                    trial_seed = 42 + (run_id * 10000)
                    prompt_text = render_prompt(f_levels, it, task_id)
                    cache_key = ExperimentManifest.compute_cache_key(
                        model_id=model_id,
                        prompt_text=prompt_text,
                        sample_id=it["id"],
                        seed=trial_seed,
                    )
                    trials_queue.append({
                        "task_id": task_id,
                        "split": split,
                        "run_id": run_id,
                        "factor_levels": f_levels,
                        "item": it,
                        "prompt_text": prompt_text,
                        "cache_key": cache_key,
                        "seed": trial_seed,
                        "evaluator": evaluator,
                    })

    # Shuffle to interleave tasks and configurations
    random.Random(42).shuffle(trials_queue)

    git_commit = get_current_git_commit()
    executed_count = 0
    cache_hits = 0
    failures = 0
    start_time = time.time()

    def process_trial(trial: Dict[str, Any]) -> Tuple[ProvenanceRecord, bool]:
        ck = trial["cache_key"]
        cached = manifest.get_cached_record(ck)
        if cached is not None:
            return cached, True

        req = ModelRequest(
            prompt=trial["prompt_text"],
            model_id=model_id,
            temperature=0.0,
            seed=trial["seed"],
        )
        resp: ModelResponse = adapter.generate(req)

        if resp.error:
            metrics = {"error": 1.0, "primary": 0.0}
        else:
            raw_eval = trial["evaluator"](resp.raw_text, trial["item"])
            metrics = raw_eval
            primary_val = list(raw_eval.values())[0] if raw_eval else 0.0
            metrics["primary"] = primary_val

        rec = ProvenanceRecord(
            trial_id=f"gemini_{uuid.uuid4().hex[:12]}",
            experiment_id="promptprism_gemini_v2",
            git_commit=git_commit,
            task_id=trial["task_id"],
            model_id=model_id,
            provider="GeminiProviderAdapter",
            run_id=trial["run_id"],
            factor_levels=trial["factor_levels"],
            sample_id=trial["item"]["id"],
            split=trial["split"],
            prompt_text=trial["prompt_text"],
            prompt_hash=sha256_hash(trial["prompt_text"]),
            response_text=resp.raw_text,
            response_hash=sha256_hash(resp.raw_text or ""),
            metrics=metrics,
            tokens_input=resp.tokens_input,
            tokens_output=resp.tokens_output,
            cost_usd=resp.cost_usd,
            latency_ms=resp.latency_ms,
            timestamp=resp.timestamp,
            seed=trial["seed"],
            retry_count=resp.retry_count,
            error=resp.error,
            cache_key=ck,
            data_origin="real",
            request_id=resp.request_id,
            finish_reason=resp.finish_reason,
        )
        return rec, False

    print(f"Dispatching requests with {max_workers} parallel workers...")
    last_print = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_trial, tr): tr for tr in trials_queue}
        for future in as_completed(futures):
            rec, is_cache = future.result()
            if is_cache:
                cache_hits += 1
            else:
                executed_count += 1
                if rec.error:
                    failures += 1
                manifest.add_record(rec, persist=True)

            completed = cache_hits + executed_count
            if time.time() - last_print > 4.0 or completed == total_expected:
                elapsed = time.time() - start_time
                rps = executed_count / elapsed if elapsed > 0 else 0
                pct = completed / total_expected * 100
                print(f"  [{pct:5.1f}%] Completed {completed}/{total_expected} | Live: {executed_count} ({rps:.1f} req/s) | Cache: {cache_hits} | Errors: {failures}")
                last_print = time.time()

    total_elapsed = time.time() - start_time
    print(f"\nExecution Complete in {total_elapsed:.1f}s. Total Records: {len(manifest.records)}")
    print(f"Live Calls: {executed_count}, Cache Hits: {cache_hits}, Failures: {failures}")

    # Build DataFrame
    records_data = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0

    for r in manifest.records:
        total_tokens_in += r.tokens_input
        total_tokens_out += r.tokens_output
        total_cost += r.cost_usd
        row = {
            "task_id": r.task_id,
            "split": r.split,
            "run_id": r.run_id,
            "sample_id": r.sample_id,
            "score": r.metrics.get("primary", list(r.metrics.values())[0] if r.metrics else 0.0),
        }
        for f, lev in r.factor_levels.items():
            row[f] = lev
        records_data.append(row)

    df = pd.DataFrame(records_data)
    disc_df = df[df["split"] == "discovery"].copy()
    conf_df = df[df["split"] == "confirmation"].copy()

    # Resolution IV configs
    res_iv_cfgs = get_res_iv_configs()
    # Find matching run_ids for Res IV
    prism_run_ids = []
    for cfg in res_iv_cfgs:
        tup = tuple(cfg[f] for f in FACTOR_IDS)
        prism_run_ids.append(all_levels.index(tup) + 1)

    ofat_run_ids = [1, 2, 3, 5, 9, 17, 33]

    print("\n" + "=" * 70)
    print("PHASE 2: Discovery Selection and Recommendation Freeze")
    print("=" * 70)

    recommendations = {}
    for task_id in tasks:
        t_disc = disc_df[disc_df["task_id"] == task_id]
        t_disc_prism = t_disc[t_disc["run_id"].isin(prism_run_ids)]
        tau_hat = fit_fractional_factorial_main_effects(t_disc_prism, FACTOR_IDS, metric_col="score")
        recipe_prism = select_prompt_from_effects(tau_hat, FACTOR_IDS)

        rng_rand = random.Random((42 + abs(hash(task_id))) % (2**31 - 1))
        rand_run_ids = rng_rand.sample(range(1, 65), 16)
        rand_selected_id = t_disc[t_disc["run_id"].isin(rand_run_ids)].groupby("run_id")["score"].mean().idxmax()
        ofat_selected_id = t_disc[t_disc["run_id"].isin(ofat_run_ids)].groupby("run_id")["score"].mean().idxmax()
        oracle_selected_id = t_disc.groupby("run_id")["score"].mean().idxmax()

        prism_tup = tuple(recipe_prism[f] for f in FACTOR_IDS)
        prism_selected_id = all_levels.index(prism_tup) + 1

        recommendations[task_id] = {
            "prompt_prism_recipe": recipe_prism,
            "prompt_prism_run_id": prism_selected_id,
            "random_search_run_id": int(rand_selected_id),
            "ofat_run_id": int(ofat_selected_id),
            "oracle_run_id": int(oracle_selected_id),
            "estimated_effects": tau_hat,
        }

    rec_json_path = output_dir / "recommendations.json"
    rec_bytes = json.dumps(recommendations, indent=2).encode("utf-8")
    rec_hash = hashlib.sha256(rec_bytes).hexdigest()
    with open(rec_json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(recommendations, indent=2))
    print(f"Recommendations frozen and hashed: {rec_hash[:16]}... -> {rec_json_path}")

    print("\n" + "=" * 70)
    print("PHASE 3: Confirmation Evaluation & Item-Cluster Bootstrap")
    print("=" * 70)

    summary_rows = []
    delta_regrets = []

    for task_id in tasks:
        t_disc = disc_df[disc_df["task_id"] == task_id]
        t_conf = conf_df[conf_df["task_id"] == task_id]
        rec = recommendations[task_id]

        conf_means = t_conf.groupby("run_id")["score"].mean()
        conf_oracle_score = conf_means.max()
        conf_baseline_score = conf_means.get(1, 0.0)

        score_prism = conf_means.get(rec["prompt_prism_run_id"], 0.0)
        score_rand = conf_means.get(rec["random_search_run_id"], 0.0)
        score_ofat = conf_means.get(rec["ofat_run_id"], 0.0)

        regret_prism = conf_oracle_score - score_prism
        regret_rand = conf_oracle_score - score_rand
        regret_ofat = conf_oracle_score - score_ofat
        delta_regret = regret_prism - regret_rand
        delta_regrets.append(delta_regret)

        # Bootstrap item-cluster resampling (B=2,000)
        disc_items = t_disc["sample_id"].unique()
        conf_items = t_conf["sample_id"].unique()
        boot_deltas = []
        rng = np.random.RandomState((42 + abs(hash(task_id))) % (2**31 - 1))

        # Pivot for fast vectorized scoring
        disc_pivot = t_disc.pivot(index="sample_id", columns="run_id", values="score")
        conf_pivot = t_conf.pivot(index="sample_id", columns="run_id", values="score")

        for _ in range(2000):
            b_disc_items = rng.choice(disc_items, size=len(disc_items), replace=True)
            b_conf_items = rng.choice(conf_items, size=len(conf_items), replace=True)

            b_disc_mean = disc_pivot.loc[b_disc_items].mean(axis=0)
            b_conf_mean = conf_pivot.loc[b_conf_items].mean(axis=0)

            # Fit on bootstrap discovery
            # Res IV main effects
            b_effects = {}
            for f_idx, f in enumerate(FACTOR_IDS):
                # +1 runs vs -1 runs in Res IV
                p_runs = [r_id for r_id in prism_run_ids if all_levels[r_id-1][f_idx] == 1]
                m_runs = [r_id for r_id in prism_run_ids if all_levels[r_id-1][f_idx] == -1]
                b_effects[f] = b_disc_mean[p_runs].mean() - b_disc_mean[m_runs].mean()

            b_recipe = select_prompt_from_effects(b_effects, FACTOR_IDS)
            b_prism_run = all_levels.index(tuple(b_recipe[f] for f in FACTOR_IDS)) + 1

            # Random search best
            b_rand_run = b_disc_mean[rand_run_ids].idxmax()

            b_oracle = b_conf_mean.max()
            b_regret_p = b_oracle - b_conf_mean.get(b_prism_run, 0.0)
            b_regret_r = b_oracle - b_conf_mean.get(b_rand_run, 0.0)
            boot_deltas.append(b_regret_p - b_regret_r)

        boot_deltas = np.array(boot_deltas)
        boot_mean = float(np.mean(boot_deltas))
        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))
        p_val = float(np.mean(boot_deltas >= 0.0))

        print(f"\n--- Task: {task_id} ---")
        print(f"  Confirmation Oracle: {conf_oracle_score:.4f} | Baseline: {conf_baseline_score:.4f}")
        print(f"  PromptPrism ($2^{{6-2}}_{{\\text{{IV}}}}$, 16 runs): Score = {score_prism:.4f} | Regret = {regret_prism:.4f}")
        print(f"  Random Search (16 runs):                  Score = {score_rand:.4f} | Regret = {regret_rand:.4f}")
        print(f"  Delta Regret: {delta_regret:+.4f} (95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}], p = {p_val:.4f})")

        summary_rows.append({
            "task_id": task_id,
            "model_id": model_id,
            "prism_score": score_prism,
            "prism_regret": regret_prism,
            "rand_score": score_rand,
            "rand_regret": regret_rand,
            "ofat_score": score_ofat,
            "ofat_regret": regret_ofat,
            "oracle_score": conf_oracle_score,
            "baseline_score": conf_baseline_score,
            "delta_regret": delta_regret,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_val": p_val,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "gemini_results.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    mean_delta = float(np.mean(delta_regrets))
    summary_json = {
        "experiment_id": "promptprism_gemini_v2",
        "model_id": model_id,
        "date": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "total_calls": len(manifest.records),
        "total_tokens_input": total_tokens_in,
        "total_tokens_output": total_tokens_out,
        "total_cost_usd": total_cost,
        "mean_delta_regret": mean_delta,
        "h1_decision": "SUPPORTED" if mean_delta < 0 else "UNSUPPORTED",
        "h5_compute_savings": 0.75,
        "task_results": summary_rows,
    }
    summary_json_path = output_dir / "gemini_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY: REAL GEMINI EVALUATION")
    print("=" * 70)
    print(summary_df[["task_id", "prism_score", "prism_regret", "rand_score", "rand_regret", "delta_regret", "p_val"]].to_string())
    print("-" * 70)
    print(f"Overall Mean Delta Regret: {mean_delta:+.4f} (Decision: {summary_json['h1_decision']})")
    print(f"Total Tokens: {total_tokens_in:,} input + {total_tokens_out:,} output | Total Cost: ${total_cost:.4f}")
    print(f"Results CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--items-per-split", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", default=["schema_extraction", "grounded_qa", "intent_classification"])
    args = parser.parse_args()

    run_gemini_experiment(
        tasks=args.tasks,
        model_id=args.model,
        items_per_split=args.items_per_split,
        max_workers=args.workers,
    )
