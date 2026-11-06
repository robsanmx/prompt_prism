"""
Batch Monte Carlo Simulation Runner for Study A.
Executes scenario grids across thousands of replications to evaluate
effect recovery, interval coverage, Type I error, power, and selection regret.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from research.analysis.estimands import EstimandCalculator, EvaluationMetrics
from research.simulation.dgp import DGPConfig, SyntheticDGP
from research.simulation.replay import MethodReplay


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "research" / "configs" / "simulation.yaml"
RESULTS_DIR = REPO_ROOT / "research" / "results"


def run_single_simulation_trial(
    scenario_id: str,
    base_config: DGPConfig,
    rep_idx: int,
    methods: List[str],
    delta_sesoi: float = 0.03,
) -> List[Dict[str, Any]]:
    """
    Run one iteration of a simulation scenario:
    1. Generate synthetic cube with seed = base_config.seed + rep_idx.
    2. Split items 50/50 into discovery and confirmation.
    3. Replay each method on discovery items only.
    4. Evaluate metrics against true oracle on confirmation items.
    """
    trial_seed = base_config.seed + rep_idx * 10007
    cfg = DGPConfig(
        scenario_id=base_config.scenario_id,
        k=base_config.k,
        factor_ids=list(base_config.factor_ids),
        num_items=base_config.num_items,
        num_replicates=base_config.num_replicates,
        baseline_intercept=base_config.baseline_intercept,
        main_effects=dict(base_config.main_effects),
        two_factor_interactions=dict(base_config.two_factor_interactions),
        item_sd=base_config.item_sd,
        noise_sd=base_config.noise_sd,
        outcome_type=base_config.outcome_type,
        judge_noise_sd=base_config.judge_noise_sd,
        judge_bias=base_config.judge_bias,
        missing_rate=base_config.missing_rate,
        seed=trial_seed,
    )

    cube = SyntheticDGP.generate_cube(cfg)
    full_df = cube.dataframe
    factor_ids = cfg.factor_ids

    # 50/50 split of benchmark items
    all_sample_ids = sorted(full_df["sample_id"].unique())
    n_disc = len(all_sample_ids) // 2
    disc_items = set(all_sample_ids[:n_disc])
    conf_items = set(all_sample_ids[n_disc:])

    discovery_df = full_df[full_df["sample_id"].isin(disc_items)].copy()
    confirmation_df = full_df[full_df["sample_id"].isin(conf_items)].copy()

    records = []

    for method_name in methods:
        try:
            if method_name == "prompt_prism_res_iv":
                res = MethodReplay.replay_prompt_prism(
                    discovery_df=discovery_df,
                    factor_ids=factor_ids,
                    plan_id="2(6-2)IV" if cfg.k == 6 else "2(5-1)V",
                )
            elif method_name == "prompt_prism_res_iii":
                res = MethodReplay.replay_prompt_prism(
                    discovery_df=discovery_df,
                    factor_ids=factor_ids,
                    plan_id="2(6-3)III" if cfg.k == 6 else "2(3-1)III",
                )
            elif method_name == "random_search":
                res = MethodReplay.replay_random_search(
                    discovery_df=discovery_df,
                    factor_ids=factor_ids,
                    budget_runs=16,
                    seed=trial_seed,
                )
            elif method_name == "random_linear_model":
                res = MethodReplay.replay_random_linear_model(
                    discovery_df=discovery_df,
                    factor_ids=factor_ids,
                    budget_runs=16,
                    seed=trial_seed,
                )
            elif method_name == "ofat_ablation":
                res = MethodReplay.replay_ofat(
                    discovery_df=discovery_df,
                    factor_ids=factor_ids,
                )
            elif method_name == "exhaustive_oracle":
                res = MethodReplay.replay_exhaustive_oracle(
                    discovery_df=discovery_df,
                    factor_ids=factor_ids,
                )
            else:
                continue

            eval_res = EstimandCalculator.evaluate_method_performance(
                result=res,
                oracle_marginal_effects=cube.oracle_marginal_effects,
                oracle_best_config=cube.oracle_best_config,
                oracle_baseline_config=cube.oracle_baseline_config,
                confirmation_df=confirmation_df,
                factor_ids=factor_ids,
                delta_sesoi=delta_sesoi,
            )

            records.append({
                "scenario_id": scenario_id,
                "replication": rep_idx,
                "method_id": eval_res.method_id,
                "budget_configs": res.budget_configs,
                "rmse": eval_res.rmse,
                "mae": eval_res.mae,
                "sign_accuracy_above_delta": eval_res.sign_accuracy_above_delta,
                "coverage_95": eval_res.coverage_95,
                "type_1_error": eval_res.type_1_error,
                "power_above_delta": eval_res.power_above_delta,
                "fdr": eval_res.fdr,
                "held_out_regret": eval_res.held_out_regret,
                "uplift_captured_ratio": eval_res.uplift_captured_ratio,
                "optimal_config_selected": int(eval_res.optimal_config_selected),
            })
        except Exception as e:
            # Record failure
            records.append({
                "scenario_id": scenario_id,
                "replication": rep_idx,
                "method_id": method_name,
                "budget_configs": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "sign_accuracy_above_delta": 0.0,
                "coverage_95": 0.0,
                "type_1_error": np.nan,
                "power_above_delta": 0.0,
                "fdr": np.nan,
                "held_out_regret": np.nan,
                "uplift_captured_ratio": 0.0,
                "optimal_config_selected": 0,
                "error": str(e),
            })

    return records


def run_monte_carlo(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Optional[Path] = None,
    num_replications: Optional[int] = None,
    scenarios_filter: Optional[List[str]] = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Execute Monte Carlo study across configured scenarios.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    reps = num_replications or cfg_yaml.get("replications_per_scenario", 2000)
    all_scenarios = cfg_yaml.get("scenarios", {})

    methods = [
        "prompt_prism_res_iv",
        "random_search",
        "random_linear_model",
        "ofat_ablation",
        "exhaustive_oracle",
    ]

    all_records = []
    start_time = time.time()

    for s_id, s_data in all_scenarios.items():
        if scenarios_filter and s_id not in scenarios_filter:
            continue

        print(f"--- Running Scenario: {s_id} ({reps} replications) ---")

        # Parse interactions
        two_fi = {}
        for k_str, val in s_data.get("two_factor_interactions", {}).items():
            parts = [p.strip() for p in k_str.split(",")]
            if len(parts) == 2:
                two_fi[(parts[0], parts[1])] = float(val)

        dgp_cfg = DGPConfig(
            scenario_id=s_id,
            k=s_data.get("k", 6),
            num_items=s_data.get("num_items", 80),
            num_replicates=s_data.get("num_replicates", 1),
            baseline_intercept=s_data.get("baseline_intercept", 0.60),
            main_effects={k: float(v) for k, v in s_data.get("main_effects", {}).items()},
            two_factor_interactions=two_fi,
            item_sd=s_data.get("item_sd", 0.10),
            noise_sd=s_data.get("noise_sd", 0.04),
            outcome_type=s_data.get("outcome_type", "bounded"),
            judge_noise_sd=s_data.get("judge_noise_sd", 0.0),
            judge_bias=s_data.get("judge_bias", 0.0),
            missing_rate=s_data.get("missing_rate", 0.0),
            seed=s_data.get("seed", 42),
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_single_simulation_trial,
                    s_id,
                    dgp_cfg,
                    r_idx,
                    methods,
                )
                for r_idx in range(reps)
            ]

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                trial_records = future.result()
                all_records.extend(trial_records)
                completed += 1
                if completed % max(1, reps // 10) == 0:
                    print(f"  Progress {s_id}: {completed}/{reps} trials complete")

    df_results = pd.DataFrame(all_records)
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds ({len(df_results)} total trial evaluations).")

    target_out = output_path or (RESULTS_DIR / "simulation_summary.csv")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(target_out, index=False)
    print(f"Results saved to {target_out}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Runner for PromptPrism")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_monte_carlo(
        config_path=Path(args.config),
        output_path=Path(args.output) if args.output else None,
        num_replications=args.replications,
        scenarios_filter=args.scenarios,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
