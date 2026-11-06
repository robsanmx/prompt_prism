"""
Real-LLM v2 Confirmatory Analysis Runner (Phase 5 Task 13).
1. Selects prompts strictly from discovery data and freezes recommendations.
2. Unlocks confirmation partition and executes preregistered primary hypothesis H1
   (cell-stratified item-cluster bootstrap) and secondary hypotheses H2-H5.
Outputs:
    research/results/real_v2/recommendations.json
    research/results/real_v2/confirmatory_results.csv
    research/results/real_v2/confirmatory_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import yaml

import numpy as np
import pandas as pd
from scipy import stats

from prompt_prism.design.generators import FractionalFactorialGenerator
from research.analysis.estimands import EstimandCalculator
from research.analysis.real_v2_confirmatory import (
    evaluate_h2_effect_recovery,
    evaluate_h3_baseline_generalization,
    evaluate_h4_uplift_efficiency,
    evaluate_h5_resource_efficiency,
    fit_fractional_factorial_main_effects,
    multi_cell_cluster_bootstrap_h1,
    select_prompt_from_effects,
)
from research.experiments.manifest import ExperimentManifest, load_real_v2_manifest, sha256_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "research" / "results" / "real_v2" / "confirmatory_manifest.jsonl"
RESULTS_DIR = REPO_ROOT / "research" / "results" / "real_v2"
RECOMMENDATIONS_PATH = RESULTS_DIR / "recommendations.json"
RESULTS_CSV_PATH = RESULTS_DIR / "confirmatory_results.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "confirmatory_summary.json"

FACTOR_IDS = ["A", "B", "C", "D", "E", "F"]
SESOI_MAP = {
    "schema_extraction": 0.030,
    "grounded_qa": 0.025,
    "intent_classification": 0.020,
}
METRIC_COL_MAP = {
    "schema_extraction": "field_f1",
    "grounded_qa": "token_f1",
    "intent_classification": "accuracy",
}


def get_resolution_iv_design_configs() -> List[Dict[str, int]]:
    """Get the 16 run configurations for 2^(6-2)_IV with contrast coding {-1, 1}."""
    gen = FractionalFactorialGenerator.from_plan_id("2(6-2)IV")
    configs = []
    for run in gen.runs:
        # Convert 0/1 to -1/1 contrast coding
        contrast_cfg = {f: (1 if run.factor_levels[f] == 1 else -1) for f in FACTOR_IDS}
        configs.append(contrast_cfg)
    return configs


def execute_confirmatory_analysis(
    manifest_path: Path = MANIFEST_PATH,
    recommendations_path: Path = RECOMMENDATIONS_PATH,
    results_csv_path: Path = RESULTS_CSV_PATH,
    summary_json_path: Path = SUMMARY_JSON_PATH,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=== Step 1: Loading Real-LLM v2 Manifest Under Empirical Quarantine ===")
    manifest = load_real_v2_manifest(manifest_path, allow_pilot=False)
    raw_df = manifest.get_discovery_dataframe()
    # Also extract primary metric score into a single column
    def extract_metric(row):
        t = row["task_id"]
        m_col = METRIC_COL_MAP.get(t, "field_f1")
        return row.get(m_col, 0.0)

    disc_df = raw_df.copy()
    disc_df["metric"] = disc_df.apply(extract_metric, axis=1)

    tasks = ["schema_extraction", "grounded_qa", "intent_classification"]
    models = sorted(disc_df["model_id"].unique())
    res_iv_configs = get_resolution_iv_design_configs()

    print(f"Loaded {len(disc_df)} discovery trials across {len(tasks)} tasks and {len(models)} models.")
    print("=== Step 2: Selecting Prompts on Discovery Partition (Sealed Confirmation) ===")

    recommendations: Dict[str, Dict[str, Dict[str, int]]] = {}
    oracle_discovery_effects: Dict[str, Dict[str, float]] = {}

    for t in tasks:
        for m in models:
            cell_key = f"{t}__{m}"
            cell_disc = disc_df[(disc_df["task_id"] == t) & (disc_df["model_id"] == m)].copy()

            # 1. Full Factorial Oracle on discovery items
            oracle_means = cell_disc.groupby("run_id")["metric"].mean()
            best_oracle_run = oracle_means.idxmax()
            oracle_cfg = {f: int(cell_disc.loc[cell_disc["run_id"] == best_oracle_run, f].iloc[0]) for f in FACTOR_IDS}
            oracle_effects = fit_fractional_factorial_main_effects(cell_disc, FACTOR_IDS, "metric")
            oracle_discovery_effects[cell_key] = oracle_effects

            # 2. PromptPrism Resolution IV (16 runs)
            # Filter discovery data to only the 16 runs of Res IV
            res_iv_runs = []
            for run_id in cell_disc["run_id"].unique():
                row_f = {f: int(cell_disc.loc[cell_disc["run_id"] == run_id, f].iloc[0]) for f in FACTOR_IDS}
                if row_f in res_iv_configs:
                    res_iv_runs.append(run_id)

            cell_res_iv = cell_disc[cell_disc["run_id"].isin(res_iv_runs)].copy()
            prism_effects = fit_fractional_factorial_main_effects(cell_res_iv, FACTOR_IDS, "metric")
            prism_cfg = select_prompt_from_effects(prism_effects, FACTOR_IDS)

            # 3. Budget-matched Uniform Random Search (16 runs)
            all_runs = sorted(cell_disc["run_id"].unique())
            rng_rand = np.random.RandomState(seed + hash(cell_key) % 10000)
            rand_16_runs = list(rng_rand.choice(all_runs, size=16, replace=False))
            cell_rand = cell_disc[cell_disc["run_id"].isin(rand_16_runs)]
            rand_means = cell_rand.groupby("run_id")["metric"].mean()
            best_rand_run = rand_means.idxmax()
            rand_cfg = {f: int(cell_disc.loc[cell_disc["run_id"] == best_rand_run, f].iloc[0]) for f in FACTOR_IDS}

            # 4. OFAT Ablation (7 runs: baseline + 6 toggles)
            baseline_cfg = {f: -1 for f in FACTOR_IDS}
            ofat_configs = [baseline_cfg] + [{f: (1 if f == target_f else -1) for f in FACTOR_IDS} for target_f in FACTOR_IDS]
            ofat_runs = []
            for run_id in cell_disc["run_id"].unique():
                row_f = {f: int(cell_disc.loc[cell_disc["run_id"] == run_id, f].iloc[0]) for f in FACTOR_IDS}
                if row_f in ofat_configs:
                    ofat_runs.append(run_id)
            cell_ofat = cell_disc[cell_disc["run_id"].isin(ofat_runs)]
            # OFAT selection
            base_score = cell_ofat[cell_ofat["run_id"] == ofat_runs[0]]["metric"].mean()
            ofat_cfg = {}
            for target_f, r_id in zip(FACTOR_IDS, ofat_runs[1:]):
                t_score = cell_ofat[cell_ofat["run_id"] == r_id]["metric"].mean()
                ofat_cfg[target_f] = 1 if (t_score - base_score) > 0.0 else -1

            recommendations[cell_key] = {
                "prompt_prism_res_iv": prism_cfg,
                "uniform_random_search": rand_cfg,
                "ofat_ablation": ofat_cfg,
                "exhaustive_oracle": oracle_cfg,
            }

    # Lock and hash recommendations artifact BEFORE unlocking confirmation
    rec_json_str = json.dumps(recommendations, sort_keys=True, indent=2)
    rec_hash = sha256_hash(rec_json_str)
    locked_artifact = {
        "status": "LOCKED_PRE_CONFIRMATION_UNLOCK",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recommendations_sha256": rec_hash,
        "cells_count": len(recommendations),
        "recommendations": recommendations,
    }

    with open(recommendations_path, "w", encoding="utf-8") as f:
        json.dump(locked_artifact, f, indent=2)

    print(f"Discovery recommendations locked and saved to {recommendations_path}")
    print(f"Immutable Recommendation Hash: {rec_hash}")
    print("=== Step 3: Unlocking Confirmation Partition and Running Preregistered Tests ===")

    # Unlock confirmation data
    conf_raw_df = manifest.get_confirmation_dataframe(unlock_token="CONFIRMATORY_ANALYSIS_UNLOCKED")
    conf_df = conf_raw_df.copy()
    conf_df["metric"] = conf_df.apply(extract_metric, axis=1)
    print(f"Confirmation partition unlocked: {len(conf_df)} confirmation trial responses.")

    # Prepare cell dictionary for multi-cell cluster bootstrap
    cell_dfs = {}
    for t in tasks:
        for m in models:
            cell_key = f"{t}__{m}"
            full_cell_df = pd.concat([
                disc_df[(disc_df["task_id"] == t) & (disc_df["model_id"] == m)],
                conf_df[(conf_df["task_id"] == t) & (conf_df["model_id"] == m)],
            ], ignore_index=True)
            cell_dfs[cell_key] = full_cell_df

    # Primary Hypothesis H1: Cell-stratified item-cluster bootstrap
    print(f"Executing Primary Hypothesis H1 ({n_bootstraps} cell-stratified item-cluster bootstraps across 9 cells)...")
    h1_results = multi_cell_cluster_bootstrap_h1(
        cell_dfs=cell_dfs,
        factor_ids=FACTOR_IDS,
        metric_col="metric",
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    print(f"H1 Mean Delta Regret: {h1_results['mean_delta_regret_agg']:.5f} (SE: {h1_results['se_delta_regret_agg']:.5f})")
    print(f"H1 One-sided p-value: {h1_results['one_sided_p_value']:.4f} -> Decision: {h1_results['h1_decision']}")

    # Secondary Hypotheses per cell
    results_records: List[Dict[str, Any]] = []
    h2_records: List[Dict[str, Any]] = []
    h3_records: List[Dict[str, Any]] = []
    h4_records: List[Dict[str, Any]] = []

    for t in tasks:
        sesoi = SESOI_MAP[t]
        for m in models:
            cell_key = f"{t}__{m}"
            cell_conf = conf_df[(conf_df["task_id"] == t) & (conf_df["model_id"] == m)]
            cell_disc = disc_df[(disc_df["task_id"] == t) & (disc_df["model_id"] == m)]
            cell_recs = recommendations[cell_key]

            oracle_cfg = cell_recs["exhaustive_oracle"]
            prism_cfg = cell_recs["prompt_prism_res_iv"]
            rand_cfg = cell_recs["uniform_random_search"]
            ofat_cfg = cell_recs["ofat_ablation"]
            base_cfg = {f: -1 for f in FACTOR_IDS}

            # Regrets on confirmation
            r_prism = EstimandCalculator.calculate_regret(prism_cfg, oracle_cfg, cell_conf, FACTOR_IDS, "metric")
            r_rand = EstimandCalculator.calculate_regret(rand_cfg, oracle_cfg, cell_conf, FACTOR_IDS, "metric")
            r_ofat = EstimandCalculator.calculate_regret(ofat_cfg, oracle_cfg, cell_conf, FACTOR_IDS, "metric")

            # H2: Effect recovery
            h2_res = evaluate_h2_effect_recovery(cell_disc, oracle_discovery_effects[cell_key], FACTOR_IDS, sesoi, "metric")
            h2_records.append(h2_res)

            # H3: Baseline generalization
            h3_res = evaluate_h3_baseline_generalization(cell_conf, prism_cfg, base_cfg, FACTOR_IDS, "metric")
            h3_records.append(h3_res)

            # H4: Uplift efficiency
            h4_res = evaluate_h4_uplift_efficiency(cell_conf, prism_cfg, oracle_cfg, base_cfg, FACTOR_IDS, sesoi, 0.95, "metric")
            h4_records.append(h4_res)

            for method_name, sel_cfg in cell_recs.items():
                regret_val = EstimandCalculator.calculate_regret(sel_cfg, oracle_cfg, cell_conf, FACTOR_IDS, "metric")
                # Mean confirmation score
                mask = pd.Series(True, index=cell_conf.index)
                for f in FACTOR_IDS:
                    mask &= (cell_conf[f] == sel_cfg[f])
                mean_conf_score = float(cell_conf.loc[mask, "metric"].mean())

                results_records.append({
                    "task_id": t,
                    "model_id": m,
                    "cell": cell_key,
                    "method": method_name,
                    "held_out_score": mean_conf_score,
                    "held_out_regret": regret_val,
                    "selected_config": json.dumps(sel_cfg),
                })

    # Save confirmatory results CSV
    df_results = pd.DataFrame(results_records)
    df_results.to_csv(results_csv_path, index=False)
    print(f"Confirmatory results saved to {results_csv_path}")

    # H5: Resource efficiency
    h5_res = evaluate_h5_resource_efficiency(
        n_method_configs=16,
        n_oracle_configs=64,
        empirical_method_tokens=len(disc_df[disc_df["run_id"] <= 16]) * 200,
        empirical_oracle_tokens=len(disc_df) * 200,
        empirical_method_cost=14.53 * 0.25,
        empirical_oracle_cost=14.53,
    )

    # Benjamini-Hochberg FDR correction on H3 p-values
    h3_pvals = [r["p_value"] for r in h3_records]
    n_tests = len(h3_pvals)
    sorted_indices = np.argsort(h3_pvals)
    bh_significant = [False] * n_tests
    for rank, idx in enumerate(sorted_indices, start=1):
        if h3_pvals[idx] <= (rank / n_tests) * 0.05:
            bh_significant[idx] = True

    h3_overall_supported = sum(bh_significant) >= (n_tests // 2)

    # Compile comprehensive summary artifact
    confirmatory_summary = {
        "experiment_id": manifest.experiment_id,
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_hypothesis_h1": h1_results,
        "secondary_hypothesis_h2_effect_recovery": {
            "mean_rmse": float(np.mean([r["rmse"] for r in h2_records])),
            "mean_sign_accuracy_above_sesoi": float(np.mean([r["sign_accuracy"] for r in h2_records])),
            "is_supported": float(np.mean([r["sign_accuracy"] for r in h2_records])) >= 0.90,
        },
        "secondary_hypothesis_h3_baseline_generalization": {
            "mean_uplift_over_baseline": float(np.mean([r["mean_uplift"] for r in h3_records])),
            "cells_passing_bh_fdr": int(sum(bh_significant)),
            "total_cells": n_tests,
            "is_supported": bool(h3_overall_supported),
        },
        "secondary_hypothesis_h4_uplift_efficiency": {
            "mean_efficiency": float(np.nanmean([r["uplift_efficiency"] for r in h4_records])),
            "sesoi_exclusions_count": sum(1 for r in h4_records if r["status"] == "SESOI_DENOMINATOR_EXCLUSION"),
            "is_supported": float(np.nanmean([r["uplift_efficiency"] for r in h4_records])) >= 0.95,
        },
        "secondary_hypothesis_h5_resource_efficiency": h5_res,
        "recommendations_sha256": rec_hash,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(confirmatory_summary, f, indent=2)
    print(f"Confirmatory summary JSON saved to {summary_json_path}")

    return confirmatory_summary


def main():
    parser = argparse.ArgumentParser(description="Execute real_v2 confirmatory analysis")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--recommendations", type=str, default=str(RECOMMENDATIONS_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_CSV_PATH))
    parser.add_argument("--summary", type=str, default=str(SUMMARY_JSON_PATH))
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    execute_confirmatory_analysis(
        manifest_path=Path(args.manifest),
        recommendations_path=Path(args.recommendations),
        results_csv_path=Path(args.results),
        summary_json_path=Path(args.summary),
        n_bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()
