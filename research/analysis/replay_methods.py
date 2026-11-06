"""
Method Replay on Confirmatory Response Cubes (Phase 4 Task 13).
Replays PromptPrism Resolution IV and comparator algorithms on the discovery partition
to generate locked recommendations and calculate effect recovery before unlocking confirmation data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from research.analysis.estimands import EstimandCalculator
from research.experiments.manifest import ExperimentManifest, sha256_hash
from research.simulation.replay import MethodReplay


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIRMATORY_CFG_PATH = REPO_ROOT / "research" / "configs" / "confirmatory.yaml"
MANIFEST_PATH = REPO_ROOT / "research" / "results" / "confirmatory_manifest.jsonl"
RESULTS_DIR = REPO_ROOT / "research" / "results"


def replay_confirmatory_study(manifest_path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    print("=== Replaying Methods on Confirmatory Discovery Data (Task 13) ===")
    manifest = ExperimentManifest(experiment_id="confirmatory_study_v1", manifest_path=manifest_path)

    # Strictly load discovery dataframe only — confirmation remains sealed
    disc_df = manifest.get_discovery_dataframe()

    factor_ids = ["A", "B", "C", "D", "E", "F"]
    tasks = ["schema_extraction", "grounded_qa", "intent_classification"]

    target_metric_map = {
        "schema_extraction": "field_f1",
        "grounded_qa": "token_f1",
        "intent_classification": "macro_f1",
    }

    recommendations: Dict[str, Any] = {}
    effect_recovery_records: List[Dict[str, Any]] = []

    for task_id in tasks:
        metric_col = target_metric_map[task_id]
        task_disc_df = disc_df[disc_df["sample_id"].str.startswith(f"{task_id.split('_')[0]}_")].copy()
        if len(task_disc_df) == 0:
            # Fallback filter by trial id or structure
            task_disc_df = disc_df.copy()

        print(f"Replaying methods for Task: {task_id} on {len(task_disc_df)} discovery trials...")

        # 1. Full Factorial Oracle on discovery items
        res_oracle = MethodReplay.replay_exhaustive_oracle(
            discovery_df=task_disc_df,
            factor_ids=factor_ids,
            target_col=metric_col,
        )

        # 2. PromptPrism Resolution IV (16 runs)
        res_prism = MethodReplay.replay_prompt_prism(
            discovery_df=task_disc_df,
            factor_ids=factor_ids,
            plan_id="2(6-2)IV",
            target_col=metric_col,
        )

        # 3. Budget-matched Uniform Random Search (16 runs, seed=42)
        res_random = MethodReplay.replay_random_search(
            discovery_df=task_disc_df,
            factor_ids=factor_ids,
            budget_runs=16,
            seed=42,
            target_col=metric_col,
        )

        # 4. Budget-matched Random Linear Model (16 runs, seed=42)
        res_linear = MethodReplay.replay_random_linear_model(
            discovery_df=task_disc_df,
            factor_ids=factor_ids,
            budget_runs=16,
            seed=42,
            target_col=metric_col,
        )

        # 5. OFAT Ablation (7 runs)
        res_ofat = MethodReplay.replay_ofat(
            discovery_df=task_disc_df,
            factor_ids=factor_ids,
            target_col=metric_col,
        )

        task_recs = {
            "prompt_prism_res_iv": res_prism.selected_config,
            "uniform_random_search": res_random.selected_config,
            "random_linear_model": res_linear.selected_config,
            "ofat_ablation": res_ofat.selected_config,
            "exhaustive_oracle": res_oracle.selected_config,
        }
        recommendations[task_id] = task_recs

        # Record effect estimation recovery
        for f_id in factor_ids:
            oracle_eff = res_oracle.estimated_main_effects[f_id]
            prism_eff = res_prism.estimated_main_effects[f_id]
            rand_eff = res_random.estimated_main_effects[f_id]
            lin_eff = res_linear.estimated_main_effects[f_id]
            ofat_eff = res_ofat.estimated_main_effects[f_id]

            effect_recovery_records.append({
                "task_id": task_id,
                "factor_id": f_id,
                "oracle_main_effect": oracle_eff,
                "prompt_prism_effect": prism_eff,
                "prompt_prism_error": prism_eff - oracle_eff,
                "random_search_effect": rand_eff,
                "random_linear_effect": lin_eff,
                "ofat_effect": ofat_eff,
            })

    # Save frozen recommendations and hashes
    recs_path = RESULTS_DIR / "method_recommendations.json"
    recommendations_artifact = {
        "status": "LOCKED_PRE_CONFIRMATION_UNLOCK",
        "timestamp": "2026-09-26T20:53:00Z",
        "recommendations": recommendations,
        "recommendations_hash": sha256_hash(json.dumps(recommendations, sort_keys=True)),
    }
    with open(recs_path, "w", encoding="utf-8") as f:
        json.dump(recommendations_artifact, f, indent=2)
    print(f"Locked recommendations saved to {recs_path}")

    # Save effect recovery CSV
    recovery_csv = RESULTS_DIR / "effect_recovery.csv"
    df_recovery = pd.DataFrame(effect_recovery_records)
    df_recovery.to_csv(recovery_csv, index=False)
    print(f"Effect recovery comparisons saved to {recovery_csv}")

    return recommendations_artifact


if __name__ == "__main__":
    replay_confirmatory_study()
