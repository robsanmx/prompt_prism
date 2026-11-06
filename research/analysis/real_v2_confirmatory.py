"""
Primary and Secondary Confirmatory Analysis for Real-LLM v2 Study.
Implements:
- H1: Cell-stratified item-cluster bootstrap testing selection regret difference:
      Delta_Regret = Regret(PromptPrism) - Regret(BalancedRandom)
- H2: Effect recovery (marginal effect RMSE and sign accuracy above SESOI)
- H3: Baseline generalization (improvement over c0 on confirmation items)
- H4: Uplift efficiency (capturing >= 95% of oracle uplift; NA if denom <= SESOI)
- H5: Resource efficiency (calls, tokens, latency, cost savings)

Strictly consumes data loaded via load_real_v2_manifest.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from research.analysis.estimands import EstimandCalculator
from research.experiments.manifest import ExperimentManifest, load_real_v2_manifest

FACTOR_IDS = ["A", "B", "C", "D", "E", "F"]


def fit_fractional_factorial_main_effects(
    df: pd.DataFrame,
    factor_ids: List[str] = FACTOR_IDS,
    metric_col: str = "metric",
) -> Dict[str, float]:
    """
    Fits marginal main effects using contrast coding (-1/+1).
    tau_j = mean(Y | x_j = +1) - mean(Y | x_j = -1).
    """
    effects = {}
    for f in factor_ids:
        y_plus = df.loc[df[f] == 1, metric_col].mean()
        y_minus = df.loc[df[f] == -1, metric_col].mean()
        effects[f] = float(y_plus - y_minus)
    return effects


def select_prompt_from_effects(
    effects: Dict[str, float],
    factor_ids: List[str] = FACTOR_IDS,
) -> Dict[str, int]:
    """Select prompt configuration setting active components to +1 if tau_j > 0."""
    return {f: (1 if effects.get(f, 0.0) > 0.0 else -1) for f in factor_ids}


def cell_cluster_bootstrap_h1(
    task_df: pd.DataFrame,
    factor_ids: List[str] = FACTOR_IDS,
    metric_col: str = "metric",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes cell-stratified item-cluster bootstrap for Primary Hypothesis H1.
    Resamples items with replacement within cell, reruns discovery selection,
    and evaluates held-out regret on confirmation items using item-level pre-aggregation.
    """
    rng = np.random.RandomState(seed)

    disc_df = task_df[task_df["split"] == "discovery"]
    conf_df = task_df[task_df["split"] == "confirmation"]

    # Pivot to item x run_id matrix
    disc_pivot = disc_df.pivot_table(index="sample_id", columns="run_id", values=metric_col, aggfunc="mean")
    conf_pivot = conf_df.pivot_table(index="sample_id", columns="run_id", values=metric_col, aggfunc="mean")

    all_runs = sorted(disc_pivot.columns)
    run_cfg_map = {}
    for r in all_runs:
        row = disc_df[disc_df["run_id"] == r].iloc[0]
        run_cfg_map[r] = {f: int(row[f]) for f in factor_ids}
    cfg_to_run = {tuple(cfg[f] for f in factor_ids): r for r, cfg in run_cfg_map.items()}

    rng_perm = np.random.RandomState(seed + 101)
    random_16_runs = list(rng_perm.choice(all_runs, size=min(16, len(all_runs)), replace=False))

    # Precompute factor mask indices for fast effect estimation
    factor_plus_runs = {f: [r for r in all_runs if run_cfg_map[r][f] == 1] for f in factor_ids}
    factor_minus_runs = {f: [r for r in all_runs if run_cfg_map[r][f] == -1] for f in factor_ids}

    disc_mat = disc_pivot.to_numpy()
    conf_mat = conf_pivot.to_numpy()
    run_cols = list(disc_pivot.columns)
    run_idx_map = {r: i for i, r in enumerate(run_cols)}

    n_disc_items = len(disc_pivot)
    n_conf_items = len(conf_pivot)

    regret_prism_boot = []
    regret_random_boot = []
    delta_regret_boot = []

    for _ in range(n_bootstraps):
        s_disc = rng.randint(0, n_disc_items, size=n_disc_items)
        s_conf = rng.randint(0, n_conf_items, size=n_conf_items)

        mean_disc_runs = disc_mat[s_disc, :].mean(axis=0)
        mean_conf_runs = conf_mat[s_conf, :].mean(axis=0)

        # Oracle on discovery
        best_oracle_col_idx = np.argmax(mean_disc_runs)
        oracle_score_conf = mean_conf_runs[best_oracle_col_idx]

        # PromptPrism selection
        prism_cfg = {}
        for f in factor_ids:
            p_cols = [run_idx_map[r] for r in factor_plus_runs[f]]
            m_cols = [run_idx_map[r] for r in factor_minus_runs[f]]
            eff = float(mean_disc_runs[p_cols].mean() - mean_disc_runs[m_cols].mean())
            prism_cfg[f] = 1 if eff > 0 else -1

        prism_run = cfg_to_run.get(tuple(prism_cfg[f] for f in factor_ids), run_cols[0])
        prism_score_conf = mean_conf_runs[run_idx_map[prism_run]]

        # Random design selection
        rand_cols = [run_idx_map[r] for r in random_16_runs]
        best_rand_col_idx = rand_cols[np.argmax(mean_disc_runs[rand_cols])]
        rand_score_conf = mean_conf_runs[best_rand_col_idx]

        r_prism = oracle_score_conf - prism_score_conf
        r_rand = oracle_score_conf - rand_score_conf
        diff = r_prism - r_rand

        regret_prism_boot.append(r_prism)
        regret_random_boot.append(r_rand)
        delta_regret_boot.append(diff)

    delta_arr = np.array(delta_regret_boot)
    mean_delta = float(np.mean(delta_arr))
    se_delta = float(np.std(delta_arr, ddof=1))
    ci_lower = float(np.percentile(delta_arr, 2.5))
    ci_upper = float(np.percentile(delta_arr, 97.5))

    p_value = float(np.mean(delta_arr >= 0.0))
    is_supported = (mean_delta < 0.0) and (p_value < 0.05)

    return {
        "n_bootstraps": n_bootstraps,
        "mean_regret_prism": float(np.mean(regret_prism_boot)),
        "mean_regret_random": float(np.mean(regret_random_boot)),
        "mean_delta_regret": mean_delta,
        "se_delta_regret": se_delta,
        "ci_95_delta": (ci_lower, ci_upper),
        "one_sided_p_value": p_value,
        "h1_decision": "SUPPORTED" if is_supported else "UNSUPPORTED",
    }


def multi_cell_cluster_bootstrap_h1(
    cell_dfs: Dict[str, pd.DataFrame],
    factor_ids: List[str] = FACTOR_IDS,
    metric_col: str = "metric",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes cell-stratified item-cluster bootstrap across all task-model cells.
    Computes equal-cell aggregate contrast:
        Delta_agg = mean_c( Regret_c(PromptPrism) - Regret_c(BalancedRandom) ).
    """
    rng = np.random.RandomState(seed)
    cell_names = sorted(cell_dfs.keys())
    n_cells = len(cell_names)

    cell_matrices = {}
    for c_name, c_df in cell_dfs.items():
        disc_df = c_df[c_df["split"] == "discovery"]
        conf_df = c_df[c_df["split"] == "confirmation"]

        disc_pivot = disc_df.pivot_table(index="sample_id", columns="run_id", values=metric_col, aggfunc="mean")
        conf_pivot = conf_df.pivot_table(index="sample_id", columns="run_id", values=metric_col, aggfunc="mean")

        all_runs = sorted(disc_pivot.columns)
        run_cfg_map = {}
        for r in all_runs:
            row = disc_df[disc_df["run_id"] == r].iloc[0]
            run_cfg_map[r] = {f: int(row[f]) for f in factor_ids}
        cfg_to_run = {tuple(cfg[f] for f in factor_ids): r for r, cfg in run_cfg_map.items()}

        rng_perm = np.random.RandomState(seed + hash(c_name) % 10000)
        random_16_runs = list(rng_perm.choice(all_runs, size=min(16, len(all_runs)), replace=False))

        factor_plus_runs = {f: [r for r in all_runs if run_cfg_map[r][f] == 1] for f in factor_ids}
        factor_minus_runs = {f: [r for r in all_runs if run_cfg_map[r][f] == -1] for f in factor_ids}

        run_cols = list(disc_pivot.columns)
        run_idx_map = {r: i for i, r in enumerate(run_cols)}

        cell_matrices[c_name] = {
            "disc_mat": disc_pivot.to_numpy(),
            "conf_mat": conf_pivot.to_numpy(),
            "run_cols": run_cols,
            "run_idx_map": run_idx_map,
            "cfg_to_run": cfg_to_run,
            "random_16_runs": random_16_runs,
            "factor_plus_runs": factor_plus_runs,
            "factor_minus_runs": factor_minus_runs,
            "n_disc": len(disc_pivot),
            "n_conf": len(conf_pivot),
        }

    agg_deltas = []
    cell_deltas = {c: [] for c in cell_names}

    for _ in range(n_bootstraps):
        boot_deltas = []
        for c_name in cell_names:
            c_data = cell_matrices[c_name]
            s_disc = rng.randint(0, c_data["n_disc"], size=c_data["n_disc"])
            s_conf = rng.randint(0, c_data["n_conf"], size=c_data["n_conf"])

            mean_disc_runs = c_data["disc_mat"][s_disc, :].mean(axis=0)
            mean_conf_runs = c_data["conf_mat"][s_conf, :].mean(axis=0)

            # Oracle on discovery
            best_oracle_col_idx = np.argmax(mean_disc_runs)
            oracle_score_conf = mean_conf_runs[best_oracle_col_idx]

            # PromptPrism selection
            prism_cfg = {}
            for f in factor_ids:
                p_cols = [c_data["run_idx_map"][r] for r in c_data["factor_plus_runs"][f]]
                m_cols = [c_data["run_idx_map"][r] for r in c_data["factor_minus_runs"][f]]
                eff = float(mean_disc_runs[p_cols].mean() - mean_disc_runs[m_cols].mean())
                prism_cfg[f] = 1 if eff > 0 else -1

            prism_run = c_data["cfg_to_run"].get(tuple(prism_cfg[f] for f in factor_ids), c_data["run_cols"][0])
            prism_score_conf = mean_conf_runs[c_data["run_idx_map"][prism_run]]

            # Random design selection
            rand_cols = [c_data["run_idx_map"][r] for r in c_data["random_16_runs"]]
            best_rand_col_idx = rand_cols[np.argmax(mean_disc_runs[rand_cols])]
            rand_score_conf = mean_conf_runs[best_rand_col_idx]

            r_prism = oracle_score_conf - prism_score_conf
            r_rand = oracle_score_conf - rand_score_conf
            c_diff = r_prism - r_rand
            boot_deltas.append(c_diff)
            cell_deltas[c_name].append(c_diff)

        agg_deltas.append(float(np.mean(boot_deltas)))

    agg_arr = np.array(agg_deltas)
    mean_agg = float(np.mean(agg_arr))
    se_agg = float(np.std(agg_arr, ddof=1))
    ci_lower = float(np.percentile(agg_arr, 2.5))
    ci_upper = float(np.percentile(agg_arr, 97.5))
    p_value = float(np.mean(agg_arr >= 0.0))
    is_supported = (mean_agg < 0.0) and (p_value < 0.05)

    return {
        "n_cells": n_cells,
        "n_bootstraps": n_bootstraps,
        "mean_delta_regret_agg": mean_agg,
        "se_delta_regret_agg": se_agg,
        "ci_95_delta_agg": (ci_lower, ci_upper),
        "one_sided_p_value": p_value,
        "h1_decision": "SUPPORTED" if is_supported else "UNSUPPORTED",
    }


def evaluate_h2_effect_recovery(
    discovery_df: pd.DataFrame,
    oracle_effects: Dict[str, float],
    factor_ids: List[str] = FACTOR_IDS,
    sesoi: float = 0.02,
    metric_col: str = "metric",
) -> Dict[str, Any]:
    """
    H2: Effect recovery. Evaluates RMSE and sign recovery against oracle effects
    for factors whose oracle magnitude exceeds SESOI.
    """
    prism_effects = fit_fractional_factorial_main_effects(discovery_df, factor_ids, metric_col)

    f_eval = [f for f in factor_ids if abs(oracle_effects.get(f, 0.0)) >= sesoi]
    if len(f_eval) == 0:
        f_eval = factor_ids

    diffs = [prism_effects[f] - oracle_effects[f] for f in f_eval]
    rmse = float(np.sqrt(np.mean(np.array(diffs) ** 2)))

    sign_matches = [
        np.sign(prism_effects[f]) == np.sign(oracle_effects[f])
        for f in f_eval
        if abs(oracle_effects[f]) >= sesoi
    ]
    sign_accuracy = float(np.mean(sign_matches)) if len(sign_matches) > 0 else 1.0

    return {
        "rmse": rmse,
        "sign_accuracy": sign_accuracy,
        "evaluated_factors": f_eval,
        "prism_effects": prism_effects,
        "oracle_effects": oracle_effects,
    }


def evaluate_h3_baseline_generalization(
    conf_df: pd.DataFrame,
    selected_cfg: Dict[str, int],
    baseline_cfg: Optional[Dict[str, int]] = None,
    factor_ids: List[str] = FACTOR_IDS,
    metric_col: str = "metric",
    block_col: str = "sample_id",
) -> Dict[str, Any]:
    """
    H3: Baseline generalization. Tests whether the selected prompt improves over
    the all-zero/all-minus-1 baseline configuration on confirmation items.
    """
    if baseline_cfg is None:
        baseline_cfg = {f: -1 for f in factor_ids}

    # Extract selected configuration rows
    sel_mask = pd.Series(True, index=conf_df.index)
    base_mask = pd.Series(True, index=conf_df.index)
    for f in factor_ids:
        sel_mask &= (conf_df[f] == selected_cfg[f])
        base_mask &= (conf_df[f] == baseline_cfg[f])

    sel_df = conf_df[sel_mask]
    base_df = conf_df[base_mask]

    merged = pd.merge(
        sel_df[[block_col, metric_col]],
        base_df[[block_col, metric_col]],
        on=block_col,
        suffixes=("_sel", "_base"),
    )

    if len(merged) < 2:
        return {
            "mean_uplift": 0.0,
            "p_value": 1.0,
            "is_supported": False,
            "n_pairs": len(merged),
        }

    diffs = (merged[f"{metric_col}_sel"] - merged[f"{metric_col}_base"]).to_numpy(dtype=float)
    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))

    if se_diff > 1e-12:
        t_stat = mean_diff / se_diff
        df = len(diffs) - 1
        p_val = float(1.0 - stats.t.cdf(t_stat, df=df)) # One-sided test diff > 0
    else:
        p_val = 0.0 if mean_diff > 0 else 1.0

    is_supported = (mean_diff > 0.0) and (p_val < 0.05)
    return {
        "mean_uplift": mean_diff,
        "se_uplift": se_diff,
        "p_value": p_val,
        "is_supported": is_supported,
        "n_pairs": len(merged),
    }


def evaluate_h4_uplift_efficiency(
    conf_df: pd.DataFrame,
    selected_cfg: Dict[str, int],
    oracle_cfg: Dict[str, int],
    baseline_cfg: Optional[Dict[str, int]] = None,
    factor_ids: List[str] = FACTOR_IDS,
    sesoi: float = 0.02,
    target_efficiency: float = 0.95,
    metric_col: str = "metric",
) -> Dict[str, Any]:
    """
    H4: Uplift efficiency. Calculates ratio of selected prompt uplift to oracle uplift.
    If oracle uplift <= sesoi, returns NaN/exclusion per plan Section 6.2.
    """
    if baseline_cfg is None:
        baseline_cfg = {f: -1 for f in factor_ids}

    def _mean_score(cfg: Dict[str, int]) -> float:
        mask = pd.Series(True, index=conf_df.index)
        for f in factor_ids:
            mask &= (conf_df[f] == cfg[f])
        return float(conf_df.loc[mask, metric_col].mean())

    y_sel = _mean_score(selected_cfg)
    y_ora = _mean_score(oracle_cfg)
    y_base = _mean_score(baseline_cfg)

    oracle_uplift = y_ora - y_base
    selected_uplift = y_sel - y_base

    if oracle_uplift <= sesoi:
        return {
            "uplift_efficiency": float("nan"),
            "status": "SESOI_DENOMINATOR_EXCLUSION",
            "is_supported": None,
            "oracle_uplift": oracle_uplift,
            "selected_uplift": selected_uplift,
            "sesoi": sesoi,
        }

    efficiency = float(selected_uplift / oracle_uplift)
    is_supported = efficiency >= target_efficiency

    return {
        "uplift_efficiency": efficiency,
        "status": "VALID",
        "is_supported": is_supported,
        "oracle_uplift": oracle_uplift,
        "selected_uplift": selected_uplift,
        "sesoi": sesoi,
    }


def evaluate_h5_resource_efficiency(
    n_method_configs: int = 16,
    n_oracle_configs: int = 64,
    empirical_method_tokens: int = 0,
    empirical_oracle_tokens: int = 0,
    empirical_method_cost: float = 0.0,
    empirical_oracle_cost: float = 0.0,
) -> Dict[str, Any]:
    """
    H5: Resource efficiency.
    Method consumes 75% fewer configurations than 64-run exhaustive oracle.
    Token, call, time, and dollar savings are empirical measurements.
    """
    config_savings = (n_oracle_configs - n_method_configs) / n_oracle_configs
    token_savings = (
        (empirical_oracle_tokens - empirical_method_tokens) / empirical_oracle_tokens
        if empirical_oracle_tokens > 0 else 0.0
    )
    cost_savings = (
        (empirical_oracle_cost - empirical_method_cost) / empirical_oracle_cost
        if empirical_oracle_cost > 0 else 0.0
    )

    return {
        "config_savings": float(config_savings),
        "empirical_token_savings": float(token_savings),
        "empirical_cost_savings": float(cost_savings),
        "is_supported": config_savings >= 0.75,
    }
