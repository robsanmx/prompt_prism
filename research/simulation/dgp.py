"""
Data Generating Process (DGP) for Monte Carlo Statistical Validation (Study A).
Generates synthetic factorial prompt evaluation datasets with known ground truth
main effects, interactions, item variance, judge noise, and missingness.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class DGPConfig:
    """
    Configuration specifying the data-generating parameters for a synthetic scenario.
    """

    scenario_id: str = "default_additive"
    k: int = 6
    factor_ids: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E", "F"])
    num_items: int = 80
    num_replicates: int = 1
    baseline_intercept: float = 0.60
    main_effects: Dict[str, float] = field(default_factory=dict)
    two_factor_interactions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    higher_order_interactions: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    item_sd: float = 0.10
    noise_sd: float = 0.05
    heteroscedastic: bool = False
    outcome_type: str = "bounded"  # "continuous", "bounded", "binary"
    judge_noise_sd: float = 0.0
    judge_bias: float = 0.0
    missing_rate: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if len(self.factor_ids) != self.k:
            if self.k <= 26:
                alphabet = [chr(65 + i) for i in range(self.k)]
                self.factor_ids = alphabet
            else:
                self.factor_ids = [f"X{i+1}" for i in range(self.k)]


@dataclass
class SyntheticResponseCube:
    """
    Full response cube containing observed data and exact oracle estimands.
    """

    scenario_id: str
    config: DGPConfig
    dataframe: pd.DataFrame
    oracle_marginal_effects: Dict[str, float]
    oracle_best_config: Dict[str, int]
    oracle_baseline_config: Dict[str, int]
    oracle_uplift: float
    true_item_difficulties: np.ndarray


class SyntheticDGP:
    """
    Deterministic synthetic response generator.
    """

    @classmethod
    def generate_cube(cls, config: DGPConfig) -> SyntheticResponseCube:
        """
        Generate a complete 2^k full-factorial response cube across all items and replicates.
        """
        rng = np.random.default_rng(config.seed)
        k = config.k
        factor_ids = config.factor_ids
        num_items = config.num_items
        num_reps = config.num_replicates

        # 1. Generate item difficulties
        item_difficulties = rng.normal(0.0, config.item_sd, size=num_items)

        # 2. All 2^k combinations in standard Yates / lexicographical order
        all_combos = list(itertools.product([0, 1], repeat=k))
        num_configs = len(all_combos)

        records = []

        # Function to compute latent continuous score z
        def compute_latent(c_dict: Dict[str, int], item_eff: float) -> float:
            z = config.baseline_intercept + item_eff
            # Main effects (0/1 toggle)
            for f_id, f_val in c_dict.items():
                z += config.main_effects.get(f_id, 0.0) * f_val

            # 2-factor interactions
            for (f1, f2), coeff in config.two_factor_interactions.items():
                if f1 in c_dict and f2 in c_dict:
                    z += coeff * (c_dict[f1] * c_dict[f2])

            # Higher-order interactions
            for f_tuple, coeff in config.higher_order_interactions.items():
                prod = 1
                for f_id in f_tuple:
                    prod *= c_dict.get(f_id, 0)
                z += coeff * prod

            return z

        # Precompute expected (noise-free, item-averaged) scores for each config to establish oracles
        expected_scores = {}
        for combo in all_combos:
            c_dict = dict(zip(factor_ids, combo))
            # Average over population of items (item difficulty expectation = 0)
            z_mean = compute_latent(c_dict, 0.0)
            if config.outcome_type == "binary":
                # Logistic sigmoid
                expected_scores[combo] = 1.0 / (1.0 + np.exp(-z_mean))
            elif config.outcome_type == "bounded":
                expected_scores[combo] = float(np.clip(z_mean, 0.0, 1.0))
            else:
                expected_scores[combo] = float(z_mean)

        # Compute exact oracle marginal effects: mean[Y | X_j=1] - mean[Y | X_j=0]
        oracle_marginal_effects = {}
        for j_idx, f_id in enumerate(factor_ids):
            scores_1 = [score for combo, score in expected_scores.items() if combo[j_idx] == 1]
            scores_0 = [score for combo, score in expected_scores.items() if combo[j_idx] == 0]
            oracle_marginal_effects[f_id] = float(np.mean(scores_1) - np.mean(scores_0))

        # Best and baseline configurations
        best_combo = max(expected_scores.keys(), key=lambda c: expected_scores[c])
        baseline_combo = tuple([0] * k)
        oracle_best_config = dict(zip(factor_ids, best_combo))
        oracle_baseline_config = dict(zip(factor_ids, baseline_combo))
        oracle_uplift = expected_scores[best_combo] - expected_scores[baseline_combo]

        # 3. Simulate observations for each config, item, and replicate
        for run_id, combo in enumerate(all_combos):
            c_dict = dict(zip(factor_ids, combo))
            for item_idx in range(num_items):
                sample_id = f"item_{item_idx:04d}"
                item_eff = item_difficulties[item_idx]

                for rep in range(num_reps):
                    z = compute_latent(c_dict, item_eff)

                    # Determine observation noise
                    if config.heteroscedastic:
                        # Noise increases with more active factors
                        active_count = sum(combo)
                        current_noise_sd = config.noise_sd * (1.0 + 0.2 * active_count)
                    else:
                        current_noise_sd = config.noise_sd

                    # Generate observation
                    if config.outcome_type == "binary":
                        p = 1.0 / (1.0 + np.exp(-z))
                        observed = float(rng.binomial(1, p))
                    else:
                        noise = rng.normal(0.0, current_noise_sd)
                        observed = z + noise

                        # Correlated judge noise if enabled
                        if config.judge_noise_sd > 0.0:
                            j_noise = rng.normal(config.judge_bias, config.judge_noise_sd)
                            observed += j_noise

                        if config.outcome_type == "bounded":
                            observed = float(np.clip(observed, 0.0, 1.0))
                        else:
                            observed = float(observed)

                    # Missing data simulation
                    is_missing = False
                    error_msg = None
                    if config.missing_rate > 0.0 and rng.random() < config.missing_rate:
                        is_missing = True
                        observed = np.nan
                        error_msg = "Simulated API Timeout / Malformed Response"

                    rec = {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "replicate": rep,
                        **c_dict,
                        "metric": observed,
                        "error": error_msg,
                    }
                    records.append(rec)

        df = pd.DataFrame(records)

        return SyntheticResponseCube(
            scenario_id=config.scenario_id,
            config=config,
            dataframe=df,
            oracle_marginal_effects=oracle_marginal_effects,
            oracle_best_config=oracle_best_config,
            oracle_baseline_config=oracle_baseline_config,
            oracle_uplift=oracle_uplift,
            true_item_difficulties=item_difficulties,
        )
