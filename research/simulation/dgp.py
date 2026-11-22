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
    factor_ids: List[str] = field(
        default_factory=lambda: ["A", "B", "C", "D", "E", "F"]
    )
    num_items: int = 80
    num_replicates: int = 1
    baseline_intercept: float = 0.60
    main_effects: Dict[str, float] = field(default_factory=dict)
    two_factor_interactions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    higher_order_interactions: Dict[Tuple[str, ...], float] = field(
        default_factory=dict
    )
    item_sd: float = 0.10
    noise_sd: float = 0.05
    heteroscedastic: bool = False
    outcome_type: str = "bounded"  # "continuous", "bounded", "binary"
    judge_noise_sd: float = 0.0
    judge_bias: float = 0.0
    missing_rate: float = 0.0
    # Section 8.3 extensions (all default-off to preserve bit-identical Study A cubes):
    # factor_id -> (trigger_prob_p in [0, 1], targeted_effect_tau when triggered)
    conditional_effects: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    num_targeted_per_factor: int = 0
    contrast_heterogeneity_sd: float = 0.0
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
    coverage_map: Dict[str, Set[str]] = field(default_factory=dict)
    baseline_sample_ids: Set[str] = field(default_factory=set)
    oracle_targeted_effects: Dict[str, float] = field(default_factory=dict)


class SyntheticDGP:
    """
    Deterministic synthetic response generator.
    """

    @classmethod
    def generate_cube(cls, config: DGPConfig) -> SyntheticResponseCube:
        """
        Generate a complete 2^k full-factorial response cube across all items and replicates.
        """
        all_combos = list(itertools.product([0, 1], repeat=config.k))
        return cls._simulate_combos(config, all_combos)

    @classmethod
    def generate_design_dataset(
        cls,
        config: DGPConfig,
        combos: List[Tuple[int, ...]],
    ) -> SyntheticResponseCube:
        """
        Generate a synthetic dataset over a specific list of run factor-level tuples
        (e.g. a 16-run 2^(6-2)IV fractional factorial design), supporting conditional-rule
        trigger probabilities, per-factor targeted items, and item-by-factor heterogeneity.
        """
        return cls._simulate_combos(config, combos)

    @classmethod
    def _simulate_combos(
        cls, config: DGPConfig, combos: List[Tuple[int, ...]]
    ) -> SyntheticResponseCube:
        rng = np.random.default_rng(config.seed)
        k = config.k
        factor_ids = config.factor_ids
        num_items = config.num_items
        num_reps = config.num_replicates
        use_extensions = bool(
            config.conditional_effects
            or config.num_targeted_per_factor > 0
            or config.contrast_heterogeneity_sd > 0.0
        )

        # 1. Generate baseline item difficulties
        item_difficulties = rng.normal(0.0, config.item_sd, size=num_items)

        # Dedicated child RNG for Section 8.3 extensions so Study A RNG stream is untouched
        ext_rng = np.random.default_rng(config.seed + 10_000_003)

        # Build item catalog: baseline items + optional targeted items per factor
        baseline_sample_ids: Set[str] = {f"item_{i:04d}" for i in range(num_items)}
        coverage_map: Dict[str, Set[str]] = {f_id: set() for f_id in factor_ids}
        item_specs: List[Tuple[str, float, Dict[str, int], str]] = []
        # Each tuple: (sample_id, item_eff, trigger_mask_by_factor, kind)

        for item_idx in range(num_items):
            sample_id = f"item_{item_idx:04d}"
            trig_mask: Dict[str, int] = {}
            if config.conditional_effects:
                for f_id, (prob_p, _tau_t) in config.conditional_effects.items():
                    triggered = int(ext_rng.random() < float(prob_p))
                    trig_mask[f_id] = triggered
                    if triggered and f_id in coverage_map:
                        coverage_map[f_id].add(sample_id)
            item_specs.append(
                (sample_id, float(item_difficulties[item_idx]), trig_mask, "baseline")
            )

        if config.num_targeted_per_factor > 0:
            for f_id in factor_ids:
                targ_diffs = ext_rng.normal(
                    0.0, config.item_sd, size=config.num_targeted_per_factor
                )
                for t_idx in range(config.num_targeted_per_factor):
                    sample_id = f"targ_{f_id}_{t_idx:04d}"
                    trig_mask = {}
                    for cf_id, (prob_p, _tau_t) in config.conditional_effects.items():
                        if cf_id == f_id:
                            trig_mask[cf_id] = 1
                        else:
                            trig_mask[cf_id] = int(ext_rng.random() < float(prob_p))
                    coverage_map[f_id].add(sample_id)
                    item_specs.append(
                        (sample_id, float(targ_diffs[t_idx]), trig_mask, "targeted")
                    )

        # Optional per-item × per-factor contrast heterogeneity (models realistic paired contrast SDs)
        item_factor_het: Dict[Tuple[str, str], float] = {}
        if config.contrast_heterogeneity_sd > 0.0:
            for sample_id, _eff, _mask, _kind in item_specs:
                for f_id in factor_ids:
                    item_factor_het[(sample_id, f_id)] = float(
                        ext_rng.normal(0.0, config.contrast_heterogeneity_sd)
                    )

        # Function to compute latent continuous score z
        def compute_latent(
            c_dict: Dict[str, int],
            item_eff: float,
            *,
            trigger_mask: Optional[Dict[str, int]] = None,
            sample_id: Optional[str] = None,
            expected_conditional: bool = False,
        ) -> float:
            z = config.baseline_intercept + item_eff
            # Main effects (0/1 toggle)
            for f_id, f_val in c_dict.items():
                z += config.main_effects.get(f_id, 0.0) * f_val
                if (
                    f_val
                    and config.conditional_effects
                    and f_id in config.conditional_effects
                ):
                    prob_p, tau_t = config.conditional_effects[f_id]
                    if expected_conditional:
                        z += float(prob_p) * float(tau_t)
                    elif trigger_mask is not None and trigger_mask.get(f_id, 0):
                        z += float(tau_t)
                if f_val and sample_id is not None and item_factor_het:
                    z += item_factor_het.get((sample_id, f_id), 0.0)

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

        # Precompute expected (noise-free, item-averaged) scores on full 2^k to establish oracles
        all_combos = list(itertools.product([0, 1], repeat=k))
        expected_scores = {}
        for combo in all_combos:
            c_dict = dict(zip(factor_ids, combo))
            z_mean = compute_latent(c_dict, 0.0, expected_conditional=True)
            if config.outcome_type == "binary":
                expected_scores[combo] = 1.0 / (1.0 + np.exp(-z_mean))
            elif config.outcome_type == "bounded":
                expected_scores[combo] = float(np.clip(z_mean, 0.0, 1.0))
            else:
                expected_scores[combo] = float(z_mean)

        # Compute exact oracle marginal effects: mean[Y | X_j=1] - mean[Y | X_j=0]
        oracle_marginal_effects = {}
        oracle_targeted_effects = {}
        for j_idx, f_id in enumerate(factor_ids):
            scores_1 = [
                score for combo, score in expected_scores.items() if combo[j_idx] == 1
            ]
            scores_0 = [
                score for combo, score in expected_scores.items() if combo[j_idx] == 0
            ]
            oracle_marginal_effects[f_id] = float(np.mean(scores_1) - np.mean(scores_0))
            cond_tau = (
                float(config.conditional_effects[f_id][1])
                if f_id in config.conditional_effects
                else 0.0
            )
            oracle_targeted_effects[f_id] = float(
                config.main_effects.get(f_id, 0.0) + cond_tau
            )

        # Best and baseline configurations
        best_combo = max(expected_scores.keys(), key=lambda c: expected_scores[c])
        baseline_combo = tuple([0] * k)
        oracle_best_config = dict(zip(factor_ids, best_combo))
        oracle_baseline_config = dict(zip(factor_ids, baseline_combo))
        oracle_uplift = expected_scores[best_combo] - expected_scores[baseline_combo]

        # 3. Simulate observations for each config, item, and replicate
        records = []
        for run_id, combo in enumerate(combos):
            c_dict = dict(zip(factor_ids, combo))
            for sample_id, item_eff, trig_mask, item_kind in item_specs:
                for rep in range(num_reps):
                    z = compute_latent(
                        c_dict,
                        item_eff,
                        trigger_mask=trig_mask if use_extensions else None,
                        sample_id=sample_id if use_extensions else None,
                    )

                    # Determine observation noise
                    if config.heteroscedastic:
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

                        if config.judge_noise_sd > 0.0:
                            j_noise = rng.normal(
                                config.judge_bias, config.judge_noise_sd
                            )
                            observed += j_noise

                        if config.outcome_type == "bounded":
                            observed = float(np.clip(observed, 0.0, 1.0))
                        else:
                            observed = float(observed)

                    # Missing data simulation
                    error_msg = None
                    if config.missing_rate > 0.0 and rng.random() < config.missing_rate:
                        observed = np.nan
                        error_msg = "Simulated API Timeout / Malformed Response"

                    rec: Dict[str, Any] = {
                        "run_id": run_id,
                        "sample_id": sample_id,
                        "replicate": rep,
                        **c_dict,
                        "metric": observed,
                        "error": error_msg,
                    }
                    if use_extensions:
                        rec["kind"] = item_kind
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
            coverage_map=coverage_map,
            baseline_sample_ids=baseline_sample_ids,
            oracle_targeted_effects=oracle_targeted_effects,
        )
