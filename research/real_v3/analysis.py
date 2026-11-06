"""Registered discovery selection and held-out analysis for real-v3."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd

from research.real_v3.design import FACTOR_IDS, all_factorial_configs, resolution_iv_run_ids


def fit_main_effects(
    frame: pd.DataFrame,
    factor_ids: List[str] = FACTOR_IDS,
    metric_col: str = "score",
) -> Dict[str, float]:
    """Estimate level +1 minus level -1 marginal contrasts."""
    if frame.empty:
        raise ValueError("Cannot fit effects from an empty frame")
    effects = {}
    for factor in factor_ids:
        plus = frame.loc[frame[factor] == 1, metric_col]
        minus = frame.loc[frame[factor] == -1, metric_col]
        if plus.empty or minus.empty:
            raise ValueError(f"Factor {factor} does not contain both levels")
        effects[factor] = float(plus.mean() - minus.mean())
    return effects


def select_recipe(effects: Dict[str, float]) -> Dict[str, int]:
    return {factor: (1 if effects[factor] > 0 else -1) for factor in FACTOR_IDS}


def config_to_run(frame: pd.DataFrame, recipe: Dict[str, int]) -> int:
    mask = pd.Series(True, index=frame.index)
    for factor in FACTOR_IDS:
        mask &= frame[factor] == recipe[factor]
    runs = frame.loc[mask, "run_id"].unique()
    if len(runs) != 1:
        raise ValueError(f"Recipe maps to {len(runs)} run IDs instead of one")
    return int(runs[0])


def _mean_by_run(frame: pd.DataFrame, metric_col: str) -> pd.Series:
    return frame.groupby("run_id", sort=True)[metric_col].mean()


def _first_idxmax(series: pd.Series) -> int:
    maximum = series.max()
    return int(min(int(index) for index, value in series.items() if value == maximum))


def analyze_task(
    frame: pd.DataFrame,
    prism_run_id: int,
    random_designs: List[List[int]],
    metric_col: str = "score",
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict:
    """Analyze fixed discovery selections against untouched confirmation items."""
    discovery = frame.loc[frame["split"] == "discovery"]
    confirmation = frame.loc[frame["split"] == "confirmation"]
    if discovery.empty or confirmation.empty:
        raise ValueError("Both discovery and confirmation observations are required")

    disc_means = _mean_by_run(discovery, metric_col)
    conf_means = _mean_by_run(confirmation, metric_col)
    discovery_oracle = _first_idxmax(disc_means)
    confirmation_best = _first_idxmax(conf_means)
    random_winners = [_first_idxmax(disc_means.loc[design]) for design in random_designs]

    oracle_score = float(conf_means.loc[discovery_oracle])
    prism_score = float(conf_means.loc[prism_run_id])
    random_scores = [float(conf_means.loc[run]) for run in random_winners]
    random_score = float(np.mean(random_scores))

    conf_pivot = confirmation.pivot(index="sample_id", columns="run_id", values=metric_col)
    rng = np.random.default_rng(seed)
    boot_deltas = []
    for _ in range(bootstrap_iterations):
        sample = rng.integers(0, len(conf_pivot), size=len(conf_pivot))
        means = conf_pivot.to_numpy()[sample].mean(axis=0)
        by_run = dict(zip(conf_pivot.columns, means))
        boot_random = float(np.mean([by_run[run] for run in random_winners]))
        boot_deltas.append(boot_random - by_run[prism_run_id])
    boot = np.asarray(boot_deltas)

    return {
        "discovery_oracle_run_id": discovery_oracle,
        "confirmation_best_run_id": confirmation_best,
        "prism_run_id": int(prism_run_id),
        "random_winner_run_ids": random_winners,
        "oracle_score": oracle_score,
        "prism_score": prism_score,
        "random_score": random_score,
        "prism_regret": oracle_score - prism_score,
        "random_regret": oracle_score - random_score,
        "delta_regret": random_score - prism_score,
        "ci_95_delta": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "one_sided_p_value": float(np.mean(boot >= 0.0)),
    }


def discovery_recommendations(
    discovery: pd.DataFrame,
    random_designs: List[List[int]],
    metric_col: str = "score",
) -> dict:
    """Create every selection using discovery rows only."""
    configs = all_factorial_configs()
    fraction = resolution_iv_run_ids(configs)
    fraction_frame = discovery.loc[discovery["run_id"].isin(fraction)]
    effects = fit_main_effects(fraction_frame, metric_col=metric_col)
    recipe = select_recipe(effects)
    prism_run = next(run for run, row in configs.items() if row == recipe)
    means = _mean_by_run(discovery, metric_col)
    return {
        "fraction_run_ids": fraction,
        "effects": effects,
        "recipe": recipe,
        "prism_run_id": int(prism_run),
        "random_winner_run_ids": [_first_idxmax(means.loc[design]) for design in random_designs],
        "discovery_oracle_run_id": _first_idxmax(means),
        "discovery_mean_by_run": {str(int(run)): float(value) for run, value in means.items()},
    }


def _pivot(frame: pd.DataFrame, split: str, metric_col: str) -> pd.DataFrame:
    subset = frame.loc[frame["split"] == split]
    pivot = subset.pivot(index="sample_id", columns="run_id", values=metric_col).sort_index(axis=1)
    expected = list(range(1, 65))
    if list(pivot.columns) != expected or pivot.isna().any().any():
        raise ValueError(f"{split} results are not a complete 64-run item cube")
    return pivot


def _select_from_means(
    means: np.ndarray,
    configs: Dict[int, Dict[str, int]],
    fraction: List[int],
    random_designs: List[List[int]],
) -> tuple[int, List[int], int]:
    fraction_index = np.array([run - 1 for run in fraction])
    fraction_means = means[fraction_index]
    recipe = {}
    for factor in FACTOR_IDS:
        levels = np.array([configs[run][factor] for run in fraction])
        effect = fraction_means[levels == 1].mean() - fraction_means[levels == -1].mean()
        recipe[factor] = 1 if effect > 0 else -1
    prism = next(run for run, row in configs.items() if row == recipe)
    random_winners = []
    for design in random_designs:
        indices = np.array([run - 1 for run in design])
        random_winners.append(int(design[int(np.argmax(means[indices]))]))
    oracle = int(np.argmax(means)) + 1
    return int(prism), random_winners, oracle


def _bh_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    total = len(ordered)
    adjusted = {}
    running = 1.0
    for reverse_index in range(total - 1, -1, -1):
        task = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_values[task] * total / rank)
        adjusted[task] = float(min(1.0, running))
    return adjusted


def analyze_study(
    frames: Dict[str, pd.DataFrame],
    recommendations: dict,
    random_designs: List[List[int]],
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict:
    """Run the frozen selection-aware primary test and registered secondary tests."""
    configs = all_factorial_configs()
    fraction = resolution_iv_run_ids(configs)
    pivots = {
        task: (_pivot(frame, "discovery", "score"), _pivot(frame, "confirmation", "score"))
        for task, frame in frames.items()
    }

    task_results = {}
    for task, (discovery, confirmation) in pivots.items():
        disc_means = discovery.mean(axis=0).to_numpy()
        conf_means = confirmation.mean(axis=0).to_numpy()
        prism, random_winners, oracle = _select_from_means(
            disc_means, configs, fraction, random_designs
        )
        frozen = recommendations["tasks"][task]
        if prism != frozen["prism_run_id"] or random_winners != frozen["random_winner_run_ids"] or oracle != frozen["discovery_oracle_run_id"]:
            raise ValueError(f"Frozen discovery selections do not match recomputation for {task}")
        random_score = float(np.mean([conf_means[run - 1] for run in random_winners]))
        prism_score = float(conf_means[prism - 1])
        oracle_score = float(conf_means[oracle - 1])
        baseline_score = float(conf_means[0])
        denominator = oracle_score - baseline_score
        task_results[task] = {
            "prism_run_id": prism,
            "random_winner_run_ids": random_winners,
            "discovery_oracle_run_id": oracle,
            "confirmation_best_run_id_post_hoc": int(np.argmax(conf_means)) + 1,
            "prism_score": prism_score,
            "random_score": random_score,
            "oracle_score": oracle_score,
            "baseline_score": baseline_score,
            "prism_regret": oracle_score - prism_score,
            "random_regret": oracle_score - random_score,
            "delta_regret": random_score - prism_score,
            "fraction_of_oracle_uplift": (
                (prism_score - baseline_score) / denominator if denominator > 0 else None
            ),
        }

    rng = np.random.default_rng(seed)
    boot_aggregate = np.empty(bootstrap_iterations)
    for iteration in range(bootstrap_iterations):
        deltas = []
        for task, (discovery, confirmation) in pivots.items():
            disc_values = discovery.to_numpy()
            conf_values = confirmation.to_numpy()
            disc_sample = rng.integers(0, len(discovery), size=len(discovery))
            conf_sample = rng.integers(0, len(confirmation), size=len(confirmation))
            disc_means = disc_values[disc_sample].mean(axis=0)
            conf_means = conf_values[conf_sample].mean(axis=0)
            prism, random_winners, _ = _select_from_means(
                disc_means, configs, fraction, random_designs
            )
            random_score = np.mean([conf_means[run - 1] for run in random_winners])
            deltas.append(float(random_score - conf_means[prism - 1]))
        boot_aggregate[iteration] = float(np.mean(deltas))

    observed = float(np.mean([row["delta_regret"] for row in task_results.values()]))
    one_sided_p = float((1 + np.sum(boot_aggregate >= 0.0)) / (bootstrap_iterations + 1))

    effect_recovery = {}
    for task, frame in frames.items():
        discovery = frame.loc[frame["split"] == "discovery"]
        oracle_effects = fit_main_effects(discovery)
        prism_effects = fit_main_effects(discovery.loc[discovery["run_id"].isin(fraction)])
        random_effects = [
            fit_main_effects(discovery.loc[discovery["run_id"].isin(design)])
            for design in random_designs
        ]

        def compare(estimated: Dict[str, float]) -> tuple[float, float]:
            errors = [(estimated[factor] - oracle_effects[factor]) ** 2 for factor in FACTOR_IDS]
            signs = [np.sign(estimated[factor]) == np.sign(oracle_effects[factor]) for factor in FACTOR_IDS]
            return float(np.sqrt(np.mean(errors))), float(np.mean(signs))

        prism_rmse, prism_sign = compare(prism_effects)
        random_comparisons = [compare(value) for value in random_effects]
        effect_recovery[task] = {
            "oracle_effects": oracle_effects,
            "prism_effects": prism_effects,
            "prism_rmse": prism_rmse,
            "prism_sign_agreement": prism_sign,
            "random_mean_rmse": float(np.mean([value[0] for value in random_comparisons])),
            "random_mean_sign_agreement": float(np.mean([value[1] for value in random_comparisons])),
        }

    h3_raw = {}
    h3_rows = {}
    for task, (_, confirmation) in pivots.items():
        selected = task_results[task]["prism_run_id"]
        differences = confirmation[selected].to_numpy() - confirmation[1].to_numpy()
        task_rng = np.random.default_rng(seed + sum(ord(ch) for ch in task))
        boot = np.empty(bootstrap_iterations)
        for iteration in range(bootstrap_iterations):
            indices = task_rng.integers(0, len(differences), size=len(differences))
            boot[iteration] = differences[indices].mean()
        p_value = float(min(1.0, 2 * min(
            (1 + np.sum(boot <= 0)) / (bootstrap_iterations + 1),
            (1 + np.sum(boot >= 0)) / (bootstrap_iterations + 1),
        )))
        h3_raw[task] = p_value
        h3_rows[task] = {
            "mean_difference": float(differences.mean()),
            "ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "raw_p_value": p_value,
        }
    adjusted = _bh_adjust(h3_raw)
    for task in h3_rows:
        h3_rows[task]["bh_adjusted_p_value"] = adjusted[task]

    return {
        "primary": {
            "estimand": "mean(random confirmation score - Prism confirmation score) across tasks",
            "observed_delta": observed,
            "ci_95": [
                float(np.percentile(boot_aggregate, 2.5)),
                float(np.percentile(boot_aggregate, 97.5)),
            ],
            "one_sided_p_value": one_sided_p,
            "alpha": 0.05,
            "supports_h1": bool(observed < 0 and one_sided_p < 0.05),
            "bootstrap_iterations": bootstrap_iterations,
        },
        "tasks": task_results,
        "h2_effect_recovery": effect_recovery,
        "h3_vs_baseline": h3_rows,
    }


def summarize_operations(records: List[dict], fraction: List[int], random_designs: List[List[int]]) -> dict:
    """Summarize completeness, failures, and actual discovery screening resources."""
    expected_per_split = 3 * 64 * 80
    by_split = Counter(row["split"] for row in records)
    finish_reasons = Counter(str(row.get("finish_reason")) for row in records)
    errors = sum(bool(row.get("error")) for row in records)

    discovery = [row for row in records if row["split"] == "discovery"]

    def resource(rows: List[dict]) -> dict:
        return {
            "calls": len(rows),
            "tokens_input": int(sum(row.get("tokens_input", 0) for row in rows)),
            "tokens_output": int(sum(row.get("tokens_output", 0) for row in rows)),
            "tokens_thoughts": int(sum(row.get("tokens_thoughts", 0) for row in rows)),
            "latency_seconds_sum": float(sum(row.get("latency_ms", 0.0) for row in rows) / 1000),
            "cost_usd": float(sum(row.get("cost_usd", 0.0) for row in rows)),
        }

    prism_rows = [row for row in discovery if row["run_id"] in fraction]
    exhaustive = resource(discovery)
    random_resources = [
        resource([row for row in discovery if row["run_id"] in design])
        for design in random_designs
    ]
    fields = list(random_resources[0])
    return {
        "completeness": {
            "counts": dict(by_split),
            "expected_per_split": expected_per_split,
            "complete": by_split["discovery"] == expected_per_split and by_split["confirmation"] == expected_per_split,
            "errors": errors,
            "finish_reasons": dict(finish_reasons),
        },
        "resources": {
            "all_calls": resource(records),
            "prism_discovery_screen": resource(prism_rows),
            "mean_balanced_random_discovery_screen": {
                field: float(np.mean([row[field] for row in random_resources])) for field in fields
            },
            "exhaustive_discovery_cube": exhaustive,
            "prism_call_fraction_of_exhaustive": len(prism_rows) / len(discovery),
        },
    }
