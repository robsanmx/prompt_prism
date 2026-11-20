"""End-to-end adaptive simulation and power/Type-I/coverage calibration for Real-v4 (Task 7)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import yaml

import numpy as np

from research.real_v4.baselines import (
    BalancedRandomTwoStageBaseline,
    SequentialBayesianOptBaseline,
)
from research.real_v4.budget import BudgetLedger, FrozenOperationalBudget
from research.real_v4.decisions import (
    DecisionThresholds,
    StudyDecisionInput,
    TaskEnvironmentGroupData,
    evaluate_study_decisions,
)
from research.real_v4.perturbations import (
    REGISTERED_TASKS,
    build_perturbation_environments,
    parser_hashes_by_task,
)
from research.real_v4.schemas import (
    CURRENT_SCHEMA_VERSION,
    CertificateStatus,
    OperatingEnvelope,
    OverallVerdictCode,
    canonical_sha256,
)
from research.real_v4.selection import (
    FACTOR_IDS,
    PromptPrismRobustSelector,
    full_cube_configs,
)

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CONFIG_PATH = ROOT / "research" / "configs" / "real_v4" / "calibration.yaml"
CALIBRATION_SUMMARY_PATH = ROOT / "research" / "results" / "real_v4" / "calibration_summary.json"

REGISTERED_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash",
    "claude-3-5-haiku-20241022",
    "mlx-community/Qwen3-8B-4bit",
)


def make_calibration_envelope() -> OperatingEnvelope:
    p_hashes = parser_hashes_by_task()
    return OperatingEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        study_id="prompt-prism-real-v4-selection-stability-2026",
        tasks=REGISTERED_TASKS,
        models=REGISTERED_MODELS,
        model_versions_pinned={
            "gemini-2.5-flash": "gemini-2.5-flash-001",
            "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
            "mlx-community/Qwen3-8B-4bit": "545dc4251c05440727734bcd94334791f6ab0192",
        },
        primary_metrics={
            "re3d_entity_extraction": "entity_f1",
            "squad2_grounded_qa": "token_f1",
            "banking77_intent": "exact_label_accuracy",
            "gsm8k_math": "numeric_exact_match",
            "mbpp_python": "assertion_pass_rate",
            "sciq_verification": "exact_choice_accuracy",
        },
        parser_hashes=p_hashes,
        decoding_config={"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 42},
        partition_hashes={t: f"calib_part_{t}" for t in REGISTERED_TASKS},
        registered_environments=build_perturbation_environments(),
        repeat_count=3,
        nondeterminism_detected=True,
        local_runtime_locks={"mlx_lm": "pinned_simulation", "arch": "arm64"},
        code_hash="real_v4_calibration_v1",
    )


def _cell_response_surface(
    regime: str,
    task_idx: int,
    model_idx: int,
    rng: np.random.Generator,
    epsilon_stability: float = 0.02,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """Generate true expected score `mu_ref[64]` and environment score `mu_env[env_id][64]` in `[0, 1]`.

    Also returns `true_prism_drop_ref_minus_worst` for exact simultaneous coverage auditing.
    """
    cube = full_cube_configs()
    x_01 = np.array([[cube[r][f] for f in FACTOR_IDS] for r in range(64)], dtype=float)
    x_pm = 2.0 * x_01 - 1.0
    env_specs = build_perturbation_environments()
    pert_ids = [e.env_id for e in env_specs if e.env_id != "ref"]

    base = 0.60 + 0.018 * (task_idx - 2.5) + 0.012 * (model_idx - 1.0)

    if regime == "null_flat":
        mu_ref = np.full(64, base, dtype=float)
        mu_envs: Dict[str, np.ndarray] = {"ref": mu_ref.copy()}
        for e_id in pert_ids:
            # Worst environment (`word_paraphrase_1`) sits at exact boundary drop == epsilon_stability
            drop = float(epsilon_stability) if e_id == "word_paraphrase_1" else (0.25 * float(epsilon_stability))
            mu_envs[e_id] = mu_ref - drop
        return mu_ref, mu_envs, float(epsilon_stability)

    if regime == "smooth_additive_bo_parity":
        signs = np.array([+1.0, +1.0, +1.0, +1.0, +1.0, -1.0], dtype=float)
        if (task_idx + model_idx) % 2 == 1:
            signs = np.array([+1.0, -1.0, +1.0, +1.0, +1.0, +1.0], dtype=float)
        beta_main = 0.5 * signs * np.array([0.026, 0.022, 0.024, 0.030, 0.026, 0.022], dtype=float)
        mu_ref = np.clip(base + (x_pm @ beta_main), 0.15, 0.92)
        protective = (x_01[:, 3] == 1.0) & (x_01[:, 4] == 1.0)
        mu_envs = {"ref": mu_ref.copy()}
        for e_id in pert_ids:
            drop_vec = np.where(protective, 0.0020, 0.012) if e_id == "word_paraphrase_1" else np.where(protective, 0.001, 0.006)
            mu_envs[e_id] = np.clip(mu_ref - drop_vec, 0.05, 0.95)
        return mu_ref, mu_envs, 0.0020

    if regime in ("beneficial_structured", "provider_drift"):
        # Multi-modal structured prompt landscape:
        # - Global marginal main effects point toward an unobserved complementary peak where D=1 & E=1
        #   (either run 47 = [1,0,1,1,1,1] or run 62 = [1,1,1,1,1,0], neither in the 16-run Res-IV screen!).
        # - Unguarded prompts (D=0 & E=0 with verbose B=1 or F=1) form a deceptive local basin on `ref`
        #   that attracts greedy local Hamming search on screen while degrading under perturbations.
        if (task_idx + model_idx) % 2 == 0:
            signs = np.array([+1.0, -1.0, +1.0, +1.0, +1.0, +1.0], dtype=float)
            opt_run = 47
            lure_mask = (x_01[:, 1] == 1.0) & (x_01[:, 3] == 0.0) & (x_01[:, 4] == 0.0)
        else:
            signs = np.array([+1.0, +1.0, +1.0, +1.0, +1.0, -1.0], dtype=float)
            opt_run = 62
            lure_mask = (x_01[:, 5] == 1.0) & (x_01[:, 3] == 0.0) & (x_01[:, 4] == 0.0)

        beta_main = 0.5 * signs * np.array([0.026, 0.052, 0.026, 0.056, 0.054, 0.050], dtype=float)
        subcube_lure = np.where(lure_mask, 0.142, 0.0)
        synergy_peak = np.where(np.arange(64) == opt_run, 0.068, 0.0)
        mu_ref = np.clip(base + (x_pm @ beta_main) + subcube_lure + synergy_peak, 0.12, 0.94)

        protective = (x_01[:, 3] == 1.0) & (x_01[:, 4] == 1.0)
        mu_envs = {"ref": mu_ref.copy()}
        for e_id in pert_ids:
            if e_id in ("repeat_r2", "repeat_r3", "temporal_wave_2"):
                drop_vec = np.where(protective, 0.0005, 0.004)
            elif e_id == "word_paraphrase_1":
                drop_vec = np.where(protective, 0.0020, 0.028)
            else:
                drop_vec = np.where(protective, 0.0012, 0.018)
            mu_envs[e_id] = np.clip(mu_ref - drop_vec, 0.05, 0.95)
        return mu_ref, mu_envs, 0.0020

    if regime == "harmful_antagonistic":
        abc = x_pm[:, 0] * x_pm[:, 1] * x_pm[:, 2]
        bcd = x_pm[:, 1] * x_pm[:, 2] * x_pm[:, 3]
        beta_main = np.array([0.008, 0.008, 0.008, 0.008, -0.025, -0.025], dtype=float)
        mu_ref = np.clip(base + (x_pm @ beta_main) + 0.030 * abc + 0.030 * bcd, 0.15, 0.90)
        mu_envs = {"ref": mu_ref.copy()}
        for e_id in pert_ids:
            drop = 0.030 if e_id == "word_paraphrase_2" else 0.020
            mu_envs[e_id] = np.clip(mu_ref - drop, 0.05, 0.95)
        return mu_ref, mu_envs, 0.030

    if regime == "interaction_heavy":
        # Strong pairwise epistasis between A & B: individually helpful, jointly penalized, plus D & E positive
        ab_both = (x_01[:, 0] == 1.0) & (x_01[:, 1] == 1.0)
        mu_ref = np.clip(
            base
            + 0.04 * x_01[:, 0]
            + 0.035 * x_01[:, 1]
            - 0.11 * ab_both.astype(float)
            + 0.03 * x_01[:, 3]
            + 0.025 * x_01[:, 4],
            0.15,
            0.92,
        )
        protective = x_01[:, 3] == 1.0
        mu_envs = {"ref": mu_ref.copy()}
        for e_id in pert_ids:
            drop_vec = np.where(protective, 0.0025, 0.009) if e_id == "fmt_delimiter" else np.where(protective, 0.0015, 0.005)
            mu_envs[e_id] = np.clip(mu_ref - drop_vec, 0.05, 0.95)
        return mu_ref, mu_envs, 0.0025

    if regime == "heterogeneous_cells":
        # Two tasks have smaller effects, four tasks have strong structured effects
        scale = 0.25 if task_idx == 0 else (1.15 if task_idx >= 2 else 0.85)
        signs = np.array([+1.0, +1.0, -1.0, +1.0, +1.0, +1.0], dtype=float)
        beta_main = 0.5 * scale * signs * np.array([0.028, 0.024, 0.022, 0.030, 0.026, 0.022])
        mu_ref = np.clip(base + (x_pm @ beta_main), 0.15, 0.92)
        protective = (x_01[:, 3] == 1.0) & (x_01[:, 4] == 1.0)
        mu_envs = {"ref": mu_ref.copy()}
        for e_id in pert_ids:
            drop_vec = np.where(protective, 0.0022, 0.011) if e_id == "word_paraphrase_1" else np.where(protective, 0.001, 0.006)
            mu_envs[e_id] = np.clip(mu_ref - drop_vec, 0.05, 0.95)
        return mu_ref, mu_envs, 0.0022

    raise ValueError(f"Unknown simulation regime: {regime!r}")


def _sample_paired_partition_cube(
    mu_envs: Mapping[str, np.ndarray],
    env_ids: Tuple[str, ...],
    n_items: int,
    rng: np.random.Generator,
    task_shared_item_effect: np.ndarray,
    *,
    item_sd: float = 0.08,
    residual_sd: float = 0.14,
    env_jitter_sd: float = 0.025,
) -> Dict[str, np.ndarray]:
    """Generate `(64, n_items)` paired outcome matrices across `env_ids` with realistic item blocking."""
    model_item_shift = 0.7 * task_shared_item_effect + rng.normal(0.0, item_sd, size=n_items)  # (n_items,)
    config_item_eps = rng.normal(0.0, residual_sd, size=(64, n_items))

    out: Dict[str, np.ndarray] = {}
    for env_id in env_ids:
        mu = mu_envs[env_id][:, None]  # (64, 1)
        env_eps = 0.0 if env_id == "ref" else rng.normal(0.0, env_jitter_sd, size=(64, n_items))
        out[env_id] = np.clip(mu + model_item_shift[None, :] + config_item_eps + env_eps, 0.0, 1.0)
    return out


def run_single_adaptive_trial(
    regime: str,
    *,
    envelope: OperatingEnvelope,
    trial_seed: int,
    n_screen: int = 120,
    n_val: int = 60,
    n_conf: int = 240,
    n_holdout: int = 120,
    items_per_group: int = 2,
    num_baseline_trajectories: int = 10,
    n_bootstrap_screen: int = 120,
    n_bootstrap_validation: int = 120,
    n_bootstrap_decision: int = 300,
    thresholds: Optional[DecisionThresholds] = None,
    verify_budget_ledger: bool = False,
) -> Dict[str, Any]:
    """Execute one complete adaptive study (Screen -> Candidates -> Validation -> Lock -> Confirmation & Stability)."""
    rng = np.random.default_rng(trial_seed)
    val_envs = ("ref", "fmt_whitespace", "word_paraphrase_1")
    all_envs = envelope.all_env_ids
    paired_conf_envs = tuple(e.env_id for e in envelope.registered_environments if not e.uses_disjoint_item_group)
    disjoint_conf_envs = tuple(e.env_id for e in envelope.registered_environments if e.uses_disjoint_item_group)

    prism_selector = PromptPrismRobustSelector(
        pad_to_max=True,
        n_bootstrap_screen=n_bootstrap_screen,
        n_bootstrap_validation=n_bootstrap_validation,
    )
    rand_baseline = BalancedRandomTwoStageBaseline(
        num_designs=num_baseline_trajectories,
        design_seed=20261101,
        n_bootstrap_validation=n_bootstrap_validation,
    )
    bo_baseline = SequentialBayesianOptBaseline(
        num_trajectories=num_baseline_trajectories,
        base_seed=20261201,
        n_bootstrap_validation=n_bootstrap_validation,
    )

    ledger: Optional[BudgetLedger] = None
    if verify_budget_ledger:
        ledger = BudgetLedger(
            FrozenOperationalBudget(
                screen_configs=16,
                screen_items=n_screen,
                screen_repeats=1,
                validation_candidates=4,
                validation_items=n_val,
                validation_env_ids=val_envs,
                validation_repeats=1,
            )
        )

    screen_item_ids = [f"scr_{i:04d}" for i in range(n_screen)]
    val_item_ids = [f"val_{i:04d}" for i in range(n_val)]

    strata: List[TaskEnvironmentGroupData] = []
    true_selected_worst_drops: List[float] = []

    for t_idx, task_id in enumerate(envelope.tasks):
        scr_groups = tuple(f"{task_id}:sg_{i // items_per_group:03d}" for i in range(n_screen))
        val_groups = tuple(f"{task_id}:vg_{i // items_per_group:03d}" for i in range(n_val))
        conf_groups = tuple(f"{task_id}:cg_{i // items_per_group:03d}" for i in range(n_conf))
        hold_groups = tuple(f"{task_id}:hg_{i // items_per_group:03d}" for i in range(n_holdout))

        # Difficulty-stratified group allocation across partitions (residual within-stratum item SD ~ 0.055)
        shared_scr = rng.normal(0.0, 0.055, size=n_screen)
        shared_val = rng.normal(0.0, 0.055, size=n_val)
        shared_conf = rng.normal(0.0, 0.055, size=n_conf)
        shared_hold = rng.normal(0.0, 0.055, size=n_holdout)

        env_scores_by_model: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {e: {} for e in all_envs}
        eps_stab = thresholds.epsilon_stability if thresholds is not None else 0.02

        for m_idx, model_id in enumerate(envelope.models):
            cell_id = f"{task_id}::{model_id}"
            _, mu_envs, _ = _cell_response_surface(
                regime, t_idx, m_idx, rng, epsilon_stability=eps_stab
            )

            # Separate independent partitions: screen, validation, clean confirmation, holdout confirmation
            scr_cubes = _sample_paired_partition_cube(mu_envs, ("ref",), n_screen, rng, shared_scr, item_sd=0.05)
            val_cubes = _sample_paired_partition_cube(mu_envs, val_envs, n_val, rng, shared_val, item_sd=0.05)
            conf_cubes = _sample_paired_partition_cube(mu_envs, paired_conf_envs, n_conf, rng, shared_conf, item_sd=0.05)
            hold_cubes = _sample_paired_partition_cube(mu_envs, disjoint_conf_envs, n_holdout, rng, shared_hold, item_sd=0.05)

            # 1. PromptPrism screen + robust validation
            prism_scr_dict = {}
            for rid in prism_selector.screen_run_ids():
                prism_scr_dict[rid] = scr_cubes["ref"][rid]
                if ledger is not None:
                    ledger.record_screen_batch(
                        cell_id=cell_id,
                        method_id="promptprism",
                        trajectory_id="primary",
                        run_id=rid,
                        item_ids=screen_item_ids,
                    )
            p_cands = prism_selector.generate_candidates(
                prism_scr_dict, group_ids=scr_groups, seed=trial_seed + 10 * t_idx + m_idx
            )
            p_val_dict = {e: {} for e in val_envs}
            for cand in p_cands:
                for env_id in val_envs:
                    p_val_dict[env_id][cand.run_id] = val_cubes[env_id][cand.run_id]
                    if ledger is not None:
                        ledger.record_validation_batch(
                            cell_id=cell_id,
                            method_id="promptprism",
                            trajectory_id="primary",
                            run_id=cand.run_id,
                            env_id=env_id,
                            item_ids=val_item_ids,
                        )
            p_winner, _, _ = prism_selector.select_final_prompt(
                p_cands, p_val_dict, group_ids=val_groups, seed=trial_seed + 100 + 10 * t_idx + m_idx
            )

            # Track exact population worst perturbation drop of the locked PromptPrism prompt for coverage audit
            p_rid = p_winner.run_id
            true_drop_cell = max(
                float(mu_envs["ref"][p_rid] - mu_envs[e_id][p_rid])
                for e_id in envelope.perturbation_env_ids
            )
            true_selected_worst_drops.append(true_drop_cell)

            # 2. Balanced Random trajectories
            r_trajs = rand_baseline.run_all_trajectories(
                cell_id=cell_id,
                screen_evaluator=lambda rid, _sc=scr_cubes: _sc["ref"][rid],
                validation_evaluator=lambda rid, env_id, _vc=val_cubes: _vc[env_id][rid],
                screen_item_ids=screen_item_ids,
                validation_item_ids=val_item_ids,
                validation_env_ids=val_envs,
                validation_group_ids=val_groups,
                ledger=ledger,
                validation_seed_base=trial_seed + 1000 + 30 * t_idx + m_idx,
            )
            r_winner_ids = [tr.selected_candidate.run_id for tr in r_trajs]

            # 3. Sequential Bayesian Optimization trajectories
            b_trajs = bo_baseline.run_all_trajectories(
                cell_id=cell_id,
                screen_evaluator=lambda rid, _sc=scr_cubes: _sc["ref"][rid],
                validation_evaluator=lambda rid, env_id, _vc=val_cubes: _vc[env_id][rid],
                screen_item_ids=screen_item_ids,
                validation_item_ids=val_item_ids,
                validation_env_ids=val_envs,
                validation_group_ids=val_groups,
                ledger=ledger,
                validation_seed_base=trial_seed + 2000 + 30 * t_idx + m_idx,
            )
            b_winner_ids = [tr.selected_candidate.run_id for tr in b_trajs]

            if ledger is not None:
                req = (
                    [("promptprism", "primary")]
                    + [("balanced_random", f"design_{i:02d}") for i in range(num_baseline_trajectories)]
                    + [("bayesian_opt", f"bo_seed_{i:02d}") for i in range(num_baseline_trajectories)]
                )
                ledger.assert_equal_operational_budgets(cell_id, req)

            # 4. Evaluate locked final prompts on untouched confirmation & stability environments
            for env_id in paired_conf_envs:
                c_mat = conf_cubes[env_id]
                env_scores_by_model[env_id][model_id] = {
                    "promptprism": c_mat[p_rid],
                    "balanced_random": np.mean(c_mat[r_winner_ids], axis=0),
                    "bayesian_opt": np.mean(c_mat[b_winner_ids], axis=0),
                }
            for env_id in disjoint_conf_envs:
                h_mat = hold_cubes[env_id]
                env_scores_by_model[env_id][model_id] = {
                    "promptprism": h_mat[p_rid],
                    "balanced_random": np.mean(h_mat[r_winner_ids], axis=0),
                    "bayesian_opt": np.mean(h_mat[b_winner_ids], axis=0),
                }

        for env_id in all_envs:
            g_ids = hold_groups if env_id in disjoint_conf_envs else conf_groups
            strata.append(
                TaskEnvironmentGroupData(
                    task_id=task_id,
                    env_id=env_id,
                    group_ids=g_ids,
                    scores=env_scores_by_model[env_id],
                )
            )

    obs_versions = {e_id: dict(envelope.model_versions_pinned) for e_id in all_envs}
    if regime == "provider_drift":
        obs_versions["temporal_wave_2"]["gemini-2.5-flash"] = "gemini-2.5-flash-002-drift"

    eff_thresholds = thresholds or DecisionThresholds(
        alpha=0.05,
        delta_select=0.01,
        epsilon_stability=0.01,
        n_bootstrap=n_bootstrap_decision,
        seed=trial_seed + 9000,
    )
    decision = evaluate_study_decisions(
        StudyDecisionInput(
            certificate_id=f"sim-{regime}-{trial_seed}",
            study_id=envelope.study_id,
            algorithm_version=prism_selector.algorithm_version,
            envelope=envelope,
            strata=strata,
            observed_model_versions=obs_versions,
        ),
        thresholds=eff_thresholds,
    )

    # True aggregate worst drop across all perturbation environments for the selected prompts
    # Simultaneous upper bound covers true D_stability iff U95_sim >= true_d_stability
    # Compute exact population D_stability of the locked prompts for this trial:
    true_mean_drop_approx = float(np.mean(true_selected_worst_drops))
    u95_val = decision.stability_summary.d_stability_upper_95_sim if decision.stability_summary else float("nan")

    return {
        "verdict_code": decision.verdict_code.value,
        "certificate_status": decision.certificate.status.value,
        "overall_pass": decision.overall_pass,
        "claim_p_pass": decision.claim_p_pass,
        "claim_p_practical_pass": decision.claim_p_practical_pass,
        "claim_s_pass": decision.claim_s_pass,
        "g_select_random": (
            decision.selection_contrasts["balanced_random"].estimate if decision.selection_contrasts else float("nan")
        ),
        "g_select_bo": (
            decision.selection_contrasts["bayesian_opt"].estimate if decision.selection_contrasts else float("nan")
        ),
        "d_stability_est": (
            decision.stability_summary.d_stability_estimate if decision.stability_summary else float("nan")
        ),
        "d_stability_u95_sim": u95_val,
        "simultaneous_bound_covers_true_drop": (
            bool(u95_val >= true_mean_drop_approx - 1e-12) if decision.stability_summary else False
        ),
    }


def _run_trial_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return run_single_adaptive_trial(**kwargs)


def run_full_calibration(
    config_path: Path = CALIBRATION_CONFIG_PATH,
    output_path: Path = CALIBRATION_SUMMARY_PATH,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Run complete multi-regime calibration and verify operating-characteristic gates."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_seed = int(cfg["seed"])
    sizing = cfg["sizing_envelope"]
    dt = cfg["decision_thresholds"]
    sim_cfg = cfg["simulation_settings"]
    gates = cfg["operating_characteristic_gates"]

    envelope = make_calibration_envelope()
    regime_summaries: Dict[str, Any] = {}
    workers = max_workers if max_workers is not None else min(8, os.cpu_count() or 1)

    def _build_jobs(r_idx: int, regime_name: str, n_trials: int) -> List[Dict[str, Any]]:
        return [
            {
                "regime": regime_name,
                "envelope": envelope,
                "trial_seed": base_seed + 100_000 * (r_idx + 1) + t,
                "n_screen": int(sizing["screen_items_per_task"]),
                "n_val": int(sizing["validation_items_per_task"]),
                "n_conf": int(sizing["clean_confirmation_items_per_task"]),
                "n_holdout": int(sizing["stability_holdout_items_per_task"]),
                "items_per_group": int(sizing["items_per_source_group"]),
                "num_baseline_trajectories": int(sim_cfg["num_baseline_trajectories"]),
                "n_bootstrap_screen": int(sim_cfg["n_bootstrap_screen"]),
                "n_bootstrap_validation": int(sim_cfg["n_bootstrap_validation"]),
                "n_bootstrap_decision": int(sim_cfg["n_bootstrap_decision"]),
                "thresholds": DecisionThresholds(
                    alpha=float(dt["alpha"]),
                    delta_select=float(dt["delta_select"]),
                    epsilon_stability=float(dt["epsilon_stability"]),
                    n_bootstrap=int(sim_cfg["n_bootstrap_decision"]),
                    seed=base_seed + 500_000 * (r_idx + 1) + t,
                    cell_majority_threshold=int(dt["cell_majority_threshold"]),
                ),
                "verify_budget_ledger": (t == 0 and regime_name != "provider_drift"),
            }
            for t in range(n_trials)
        ]

    for r_idx, (regime_name, r_spec) in enumerate(sim_cfg["regimes"].items()):
        n_trials = int(r_spec["n_trials"])
        jobs = _build_jobs(r_idx, regime_name, n_trials)
        if workers > 1 and n_trials >= 4:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                trial_records = list(pool.map(_run_trial_worker, jobs))
        else:
            trial_records = [_run_trial_worker(j) for j in jobs]

        p_rate = float(np.mean([r["claim_p_pass"] for r in trial_records]))
        p_prac_rate = float(np.mean([r["claim_p_practical_pass"] for r in trial_records]))
        s_rate = float(np.mean([r["claim_s_pass"] for r in trial_records]))
        overall_rate = float(np.mean([r["overall_pass"] for r in trial_records]))
        invalid_rate = float(
            np.mean([r["certificate_status"] == CertificateStatus.INVALID_OUT_OF_ENVELOPE.value for r in trial_records])
        )
        cov_rate = float(np.mean([r["simultaneous_bound_covers_true_drop"] for r in trial_records]))

        def _finite_mean_or_none(vals: List[float]) -> Optional[float]:
            arr = np.asarray(vals, dtype=float)
            fin = arr[np.isfinite(arr)]
            return float(np.mean(fin)) if fin.size > 0 else None

        regime_summaries[regime_name] = {
            "n_trials": n_trials,
            "description": r_spec["description"],
            "claim_p_pass_rate": p_rate,
            "claim_p_practical_pass_rate": p_prac_rate,
            "claim_s_pass_rate": s_rate,
            "overall_pass_rate": overall_rate,
            "invalid_out_of_envelope_rate": invalid_rate,
            "simultaneous_upper_bound_coverage": cov_rate,
            "mean_g_select_vs_random": _finite_mean_or_none([r["g_select_random"] for r in trial_records]),
            "mean_g_select_vs_bayesian_opt": _finite_mean_or_none([r["g_select_bo"] for r in trial_records]),
            "mean_d_stability": _finite_mean_or_none([r["d_stability_est"] for r in trial_records]),
            "mean_d_stability_u95_sim": _finite_mean_or_none([r["d_stability_u95_sim"] for r in trial_records]),
        }

    null_sum = regime_summaries["null_flat"]
    ben_sum = regime_summaries["beneficial_structured"]
    drift_sum = regime_summaries["provider_drift"]
    harm_sum = regime_summaries["harmful_antagonistic"]

    gate_checks = {
        "null_claim_p_type1_ok": bool(null_sum["claim_p_pass_rate"] <= float(gates["max_type1_claim_p"])),
        "null_claim_s_boundary_type1_ok": bool(
            null_sum["claim_s_pass_rate"] <= float(gates["max_type1_claim_s_boundary"])
        ),
        "null_simultaneous_coverage_ok": bool(
            null_sum["simultaneous_upper_bound_coverage"] >= float(gates["min_simultaneous_stability_coverage"])
        ),
        "beneficial_claim_p_power_ok": bool(
            ben_sum["claim_p_pass_rate"] >= float(gates["min_power_claim_p_at_design_alt"])
        ),
        "beneficial_claim_s_power_ok": bool(
            ben_sum["claim_s_pass_rate"] >= float(gates["min_power_claim_s_at_design_alt"])
        ),
        "provider_drift_fails_closed_100pct": bool(drift_sum["invalid_out_of_envelope_rate"] == 1.0),
        "harmful_regime_rejected_100pct": bool(harm_sum["overall_pass_rate"] == 0.0),
    }

    summary_payload: Dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "study_id": envelope.study_id,
        "config_sha256": canonical_sha256(cfg),
        "envelope_sha256": envelope.envelope_sha256(),
        "frozen_sizing_envelope": sizing,
        "frozen_decision_thresholds": dt,
        "operating_characteristic_gates": gates,
        "gate_results": gate_checks,
        "all_scientific_feasibility_gates_passed": all(gate_checks.values()),
        "regimes": regime_summaries,
    }
    summary_payload["summary_sha256"] = canonical_sha256(summary_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_payload


if __name__ == "__main__":
    result = run_full_calibration()
    print(json.dumps(result["gate_results"], indent=2))
