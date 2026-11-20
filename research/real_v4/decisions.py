"""Frozen decision engine for Real-v4 Claim P (Selection) and Claim S (Bounded Stability).

Operates strictly on structured outcome fixtures or locked manifest projections without
coupling to live model execution, enabling synthetic boundary and coverage verification
before any prospective outcome exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.real_v4.certificate import StabilityCertificate, build_stability_certificate
from research.real_v4.schemas import (
    ALLOWED_CONCLUSION_MAP,
    CertificateStatus,
    ComparatorContrastSummary,
    OperatingEnvelope,
    OverallVerdictCode,
    StabilitySummary,
    SupportingDiagnostics,
)

PRIMARY_COMPARATORS: Tuple[str, str] = ("balanced_random", "bayesian_opt")
PROMPTPRISM_METHOD_ID: str = "promptprism"
BOUNDARY_TOL: float = 1e-12


@dataclass(frozen=True)
class DecisionThresholds:
    alpha: float = 0.05
    delta_select: float = 0.01
    epsilon_stability: float = 0.01
    n_bootstrap: int = 2000
    seed: int = 20261118
    cell_majority_threshold: int = 12

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 0.5):
            raise ValueError(f"alpha must lie in (0, 0.5), got {self.alpha}")
        if self.delta_select < 0.0:
            raise ValueError(f"delta_select must be >= 0, got {self.delta_select}")
        if self.epsilon_stability <= 0.0:
            raise ValueError(f"epsilon_stability must be > 0, got {self.epsilon_stability}")
        if self.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100, got {self.n_bootstrap}")


@dataclass(frozen=True)
class TaskEnvironmentGroupData:
    """Paired outcomes for one `(task_id, env_id)` stratum across all registered models and methods.

    `scores[model_id][method_id]` is a 1D float array of length `N_items` in `[0, 1]`,
    paired item-by-item with `group_ids` (length `N_items`).
    For environments sharing `uses_disjoint_item_group=False`, `group_ids` must match
    that task's `ref` `group_ids` in identical order so within-task resampling preserves
    exact item pairing across models, methods, and perturbations.
    """

    task_id: str
    env_id: str
    group_ids: Tuple[str, ...]
    scores: Mapping[str, Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class StudyDecisionInput:
    certificate_id: str
    study_id: str
    algorithm_version: str
    envelope: OperatingEnvelope
    strata: Sequence[TaskEnvironmentGroupData]
    observed_model_versions: Dict[str, Dict[str, str]]
    hash_mismatches: Tuple[str, ...] = ()
    extra_drift_events: Tuple[str, ...] = ()
    supporting_diagnostics: Optional[SupportingDiagnostics] = None
    comparators: Tuple[str, ...] = PRIMARY_COMPARATORS


@dataclass(frozen=True)
class StudyDecisionResult:
    verdict_code: OverallVerdictCode
    allowed_conclusion: str
    overall_pass: bool
    claim_p_pass: bool
    claim_p_practical_pass: bool
    claim_s_pass: bool
    cell_consistency_gate_pass: bool
    positive_cells_by_comparator: Dict[str, int]
    model_family_means_by_comparator: Dict[str, Dict[str, float]]
    selection_contrasts: Dict[str, ComparatorContrastSummary]
    stability_summary: Optional[StabilitySummary]
    certificate: StabilityCertificate
    supporting_diagnostics: Optional[SupportingDiagnostics] = field(default=None)


def _group_means_matrix(
    group_ids: Tuple[str, ...],
    models: Tuple[str, ...],
    methods: Tuple[str, ...],
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """Aggregate item-level scores into unique-group sum and count arrays for exact item-weighted bootstrap."""
    unique_groups: List[str] = []
    group_to_idx: Dict[str, int] = {}
    for gid in group_ids:
        if gid not in group_to_idx:
            group_to_idx[gid] = len(unique_groups)
            unique_groups.append(gid)
    n_groups = len(unique_groups)
    idx_arr = np.array([group_to_idx[g] for g in group_ids], dtype=int)

    sums: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models}
    counts: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in models}
    g_counts = np.bincount(idx_arr, minlength=n_groups).astype(float)

    for model_id in models:
        for method_id in methods:
            arr = np.asarray(scores[model_id][method_id], dtype=float)
            sums[model_id][method_id] = np.bincount(idx_arr, weights=arr, minlength=n_groups)
            counts[model_id][method_id] = g_counts
    return unique_groups, sums, counts


def evaluate_study_decisions(
    decision_input: StudyDecisionInput,
    thresholds: DecisionThresholds = DecisionThresholds(),
) -> StudyDecisionResult:
    """Evaluate Claim P, Claim S, and the overall Real-v4 verdict with fail-closed integrity."""
    env = decision_input.envelope
    tasks = env.tasks
    models = env.models
    comparators = decision_input.comparators
    all_methods = (PROMPTPRISM_METHOD_ID, *comparators)
    env_spec_map = {e.env_id: e for e in env.registered_environments}

    # Index strata by (task_id, env_id)
    strata_map: Dict[Tuple[str, str], TaskEnvironmentGroupData] = {}
    duplicate_strata: List[str] = []
    for item in decision_input.strata:
        key = (item.task_id, item.env_id)
        if key in strata_map:
            duplicate_strata.append(f"duplicate_stratum:{item.task_id}:{item.env_id}")
        strata_map[key] = item

    observed_envs = tuple(sorted({item.env_id for item in decision_input.strata}))
    missing_cells_count = 0
    incomplete_pairs_count = 0
    integrity_errors: List[str] = [*decision_input.extra_drift_events, *duplicate_strata]

    # Check completeness and pairing across every registered task, environment, model, and method
    for env_id in env.all_env_ids:
        spec = env_spec_map[env_id]
        for task_id in tasks:
            st = strata_map.get((task_id, env_id))
            if st is None:
                missing_cells_count += len(models)
                continue
            n_items = len(st.group_ids)
            if n_items == 0:
                missing_cells_count += len(models)
                continue
            if not spec.uses_disjoint_item_group and env_id != "ref":
                ref_st = strata_map.get((task_id, "ref"))
                if ref_st is not None and st.group_ids != ref_st.group_ids:
                    incomplete_pairs_count += 1
                    integrity_errors.append(f"group_alignment_mismatch:{task_id}:{env_id}")
            for model_id in models:
                if model_id not in st.scores:
                    missing_cells_count += 1
                    continue
                m_scores = st.scores[model_id]
                for method_id in all_methods:
                    if method_id not in m_scores:
                        incomplete_pairs_count += 1
                        continue
                    arr = np.asarray(m_scores[method_id], dtype=float)
                    if arr.ndim != 1 or len(arr) != n_items:
                        incomplete_pairs_count += 1
                        continue
                    if not np.all(np.isfinite(arr)) or np.any(arr < -1e-9) or np.any(arr > 1.0 + 1e-9):
                        incomplete_pairs_count += 1
                        integrity_errors.append(f"invalid_score_values:{task_id}:{env_id}:{model_id}:{method_id}")

    # Build preliminary certificate check to detect any envelope/version/missing breach
    prelim_cert = build_stability_certificate(
        certificate_id=decision_input.certificate_id,
        study_id=decision_input.study_id,
        algorithm_version=decision_input.algorithm_version,
        envelope=env,
        observed_environments=observed_envs,
        observed_model_versions=decision_input.observed_model_versions,
        claim_p_pass=False,
        claim_p_practical_pass=False,
        claim_s_statistical_pass=False,
        missing_cells_count=missing_cells_count,
        incomplete_pairs_count=incomplete_pairs_count,
        hash_mismatches=decision_input.hash_mismatches,
        extra_drift_events=tuple(integrity_errors),
        supporting_diagnostics=decision_input.supporting_diagnostics,
    )
    if prelim_cert.status == CertificateStatus.INVALID_OUT_OF_ENVELOPE:
        return StudyDecisionResult(
            verdict_code=OverallVerdictCode.INVALID_OUT_OF_ENVELOPE,
            allowed_conclusion=ALLOWED_CONCLUSION_MAP[OverallVerdictCode.INVALID_OUT_OF_ENVELOPE],
            overall_pass=False,
            claim_p_pass=False,
            claim_p_practical_pass=False,
            claim_s_pass=False,
            cell_consistency_gate_pass=False,
            positive_cells_by_comparator={b: 0 for b in comparators},
            model_family_means_by_comparator={b: {} for b in comparators},
            selection_contrasts={},
            stability_summary=None,
            certificate=prelim_cert,
            supporting_diagnostics=decision_input.supporting_diagnostics,
        )

    # All strata are complete and verified. Precompute group sums/counts per (task_id, env_id).
    grouped: Dict[Tuple[str, str], Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]] = {}
    for env_id in env.all_env_ids:
        for task_id in tasks:
            st = strata_map[(task_id, env_id)]
            grouped[(task_id, env_id)] = _group_means_matrix(st.group_ids, models, all_methods, st.scores)

    n_cells = len(tasks) * len(models)
    all_envs = env.all_env_ids
    pert_envs = env.perturbation_env_ids
    B = thresholds.n_bootstrap

    # Point estimates: equal weight across all |tasks| * |models| cells
    cell_means: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    env_method_means: Dict[str, Dict[str, float]] = {e: {} for e in all_envs}

    for env_id in all_envs:
        cell_vals_by_method: Dict[str, List[float]] = {m: [] for m in all_methods}
        for task_id in tasks:
            _, sums, counts = grouped[(task_id, env_id)]
            for model_id in models:
                c_dict: Dict[str, float] = {}
                for method_id in all_methods:
                    mu = float(sums[model_id][method_id].sum() / counts[model_id][method_id].sum())
                    c_dict[method_id] = mu
                    cell_vals_by_method[method_id].append(mu)
                cell_means[(task_id, model_id, env_id)] = c_dict
        for method_id in all_methods:
            env_method_means[env_id][method_id] = float(np.mean(cell_vals_by_method[method_id]))

    # Paired hierarchical group bootstrap within each task (carrying all models, methods, and paired envs)
    rng = np.random.default_rng(thresholds.seed)
    boot_env_method_sum = {
        e: {m: np.zeros(B, dtype=float) for m in all_methods}
        for e in all_envs
    }

    for task_id in tasks:
        ref_groups, _, _ = grouped[(task_id, "ref")]
        n_ref_g = len(ref_groups)
        ref_draws = rng.integers(0, n_ref_g, size=(B, n_ref_g))

        # Separate independent group draws for environments with disjoint item groups (e.g. item_holdout_group)
        disjoint_draws: Dict[str, np.ndarray] = {}
        for env_id in pert_envs:
            if env_spec_map[env_id].uses_disjoint_item_group:
                d_groups, _, _ = grouped[(task_id, env_id)]
                n_dg = len(d_groups)
                disjoint_draws[env_id] = rng.integers(0, n_dg, size=(B, n_dg))

        for env_id in all_envs:
            _, sums, counts = grouped[(task_id, env_id)]
            draws = disjoint_draws[env_id] if env_id in disjoint_draws else ref_draws
            for model_id in models:
                # Counts are identical across methods for a given (task, env, model)
                denom = counts[model_id][PROMPTPRISM_METHOD_ID][draws].sum(axis=1)
                for method_id in all_methods:
                    num = sums[model_id][method_id][draws].sum(axis=1)
                    boot_env_method_sum[env_id][method_id] += num / denom

    boot_env_method = {
        e: {m: boot_env_method_sum[e][m] / n_cells for m in all_methods}
        for e in all_envs
    }

    # 1. Evaluate Claim P: Selection Superiority on clean reference confirmation ('ref')
    selection_contrasts: Dict[str, ComparatorContrastSummary] = {}
    positive_cells_by_comp: Dict[str, int] = {}
    model_family_means_by_comp: Dict[str, Dict[str, float]] = {}

    for comp in comparators:
        est = float(env_method_means["ref"][PROMPTPRISM_METHOD_ID] - env_method_means["ref"][comp])
        boot_diff = boot_env_method["ref"][PROMPTPRISM_METHOD_ID] - boot_env_method["ref"][comp]
        se = float(np.std(boot_diff, ddof=1))
        if se < 1e-15:
            ci_low = est
            p_val = 0.0 if est > BOUNDARY_TOL else 1.0
        else:
            ci_low = float(np.percentile(boot_diff, 100.0 * thresholds.alpha))
            p_val = float((1.0 + np.sum(boot_diff <= BOUNDARY_TOL)) / (B + 1.0))

        passes_stat = bool(ci_low > BOUNDARY_TOL)
        passes_prac = bool(ci_low > thresholds.delta_select + BOUNDARY_TOL)
        selection_contrasts[comp] = ComparatorContrastSummary(
            comparator_id=comp,
            estimate=est,
            se=se,
            ci95_lower=ci_low,
            p_value_one_sided=p_val,
            practical_margin=thresholds.delta_select,
            passes_statistical_gate=passes_stat,
            passes_practical_gate=passes_prac,
        )

        # Cell consistency diagnostics
        pos_count = 0
        m_means: Dict[str, float] = {m: 0.0 for m in models}
        for task_id in tasks:
            for model_id in models:
                c_diff = (
                    cell_means[(task_id, model_id, "ref")][PROMPTPRISM_METHOD_ID]
                    - cell_means[(task_id, model_id, "ref")][comp]
                )
                if c_diff > 0.0:
                    pos_count += 1
                m_means[model_id] += c_diff / len(tasks)
        positive_cells_by_comp[comp] = pos_count
        model_family_means_by_comp[comp] = m_means

    claim_p_pass = all(c.passes_statistical_gate for c in selection_contrasts.values())
    claim_p_practical_pass = all(c.passes_practical_gate for c in selection_contrasts.values())
    cell_consistency_gate_pass = all(
        positive_cells_by_comp[b] >= thresholds.cell_majority_threshold
        and all(v >= 0.0 for v in model_family_means_by_comp[b].values())
        for b in comparators
    )

    # 2. Evaluate Claim S: Simultaneous Maximum-Drop Noninferiority + Worst-Environment Comparator Check
    ref_mean = float(env_method_means["ref"][PROMPTPRISM_METHOD_ID])
    per_env_means = {e: float(env_method_means[e][PROMPTPRISM_METHOD_ID]) for e in all_envs}
    per_env_drops = {e: float(ref_mean - env_method_means[e][PROMPTPRISM_METHOD_ID]) for e in pert_envs}
    worst_pert_env = max(pert_envs, key=lambda e: (per_env_drops[e], e))
    d_stab_est = float(per_env_drops[worst_pert_env])

    boot_drops = np.column_stack(
        [
            boot_env_method["ref"][PROMPTPRISM_METHOD_ID] - boot_env_method[e][PROMPTPRISM_METHOD_ID]
            for e in pert_envs
        ]
    )  # shape (B, |pert_envs|)
    d_hats = np.array([per_env_drops[e] for e in pert_envs], dtype=float)
    se_drops = np.std(boot_drops, axis=0, ddof=1)

    if np.all(se_drops < 1e-15):
        u95_sim = d_stab_est
    else:
        se_safe = np.maximum(se_drops, 1e-12)
        studentized_pivots = (boot_drops - d_hats[None, :]) / se_safe[None, :]
        t_max_boot = np.max(studentized_pivots, axis=1)
        q_sim = float(np.percentile(t_max_boot, 100.0 * (1.0 - thresholds.alpha)))
        # Zero-variance environments retain exact point drop
        eff_se = np.where(se_drops < 1e-15, 0.0, se_drops)
        u95_sim = float(np.max(d_hats + q_sim * eff_se))

    passes_abs_stability = bool(u95_sim < thresholds.epsilon_stability - BOUNDARY_TOL)

    # Worst-environment comparator noninferiority: G_robust,b = min_e mu_prism,e - min_e mu_b,e > -epsilon_stability
    prism_all_means = np.array([env_method_means[e][PROMPTPRISM_METHOD_ID] for e in all_envs], dtype=float)
    prism_boot_all = np.column_stack([boot_env_method[e][PROMPTPRISM_METHOD_ID] for e in all_envs])
    prism_min_est = float(np.min(prism_all_means))
    prism_min_boot = np.min(prism_boot_all, axis=1)

    comp_worst_checks: Dict[str, ComparatorContrastSummary] = {}
    for comp in comparators:
        comp_all_means = np.array([env_method_means[e][comp] for e in all_envs], dtype=float)
        comp_boot_all = np.column_stack([boot_env_method[e][comp] for e in all_envs])
        comp_min_est = float(np.min(comp_all_means))
        comp_min_boot = np.min(comp_boot_all, axis=1)

        g_rob_est = float(prism_min_est - comp_min_est)
        g_rob_boot = prism_min_boot - comp_min_boot
        se_rob = float(np.std(g_rob_boot, ddof=1))
        if se_rob < 1e-15:
            rob_ci_low = g_rob_est
            rob_p = 0.0 if g_rob_est > -thresholds.epsilon_stability + BOUNDARY_TOL else 1.0
        else:
            centered_rob = g_rob_boot - np.mean(g_rob_boot) + g_rob_est
            rob_ci_low = float(np.percentile(centered_rob, 100.0 * thresholds.alpha))
            rob_p = float((1.0 + np.sum(centered_rob <= -thresholds.epsilon_stability + BOUNDARY_TOL)) / (B + 1.0))

        passes_noninf = bool(rob_ci_low > -thresholds.epsilon_stability + BOUNDARY_TOL)
        passes_sup = bool(rob_ci_low > BOUNDARY_TOL)
        comp_worst_checks[comp] = ComparatorContrastSummary(
            comparator_id=comp,
            estimate=g_rob_est,
            se=se_rob,
            ci95_lower=rob_ci_low,
            p_value_one_sided=rob_p,
            practical_margin=thresholds.epsilon_stability,
            passes_statistical_gate=passes_noninf,
            passes_practical_gate=passes_sup,
        )

    passes_comp_worst = all(c.passes_statistical_gate for c in comp_worst_checks.values())
    claim_s_pass = bool(passes_abs_stability and passes_comp_worst)

    stability_summary = StabilitySummary(
        reference_mean=ref_mean,
        d_stability_estimate=d_stab_est,
        d_stability_upper_95_sim=u95_sim,
        epsilon_stability=thresholds.epsilon_stability,
        worst_perturbation_env_id=worst_pert_env,
        per_env_means=per_env_means,
        per_env_drops=per_env_drops,
        passes_absolute_noninferiority=passes_abs_stability,
        comparator_worst_env_checks=comp_worst_checks,
        passes_comparator_worst_env=passes_comp_worst,
    )

    certificate = build_stability_certificate(
        certificate_id=decision_input.certificate_id,
        study_id=decision_input.study_id,
        algorithm_version=decision_input.algorithm_version,
        envelope=env,
        observed_environments=observed_envs,
        observed_model_versions=decision_input.observed_model_versions,
        claim_p_pass=claim_p_pass,
        claim_p_practical_pass=claim_p_practical_pass,
        claim_s_statistical_pass=claim_s_pass,
        missing_cells_count=0,
        incomplete_pairs_count=0,
        hash_mismatches=decision_input.hash_mismatches,
        extra_drift_events=decision_input.extra_drift_events,
        selection_contrasts=selection_contrasts,
        stability_summary=stability_summary,
        supporting_diagnostics=decision_input.supporting_diagnostics,
    )

    return StudyDecisionResult(
        verdict_code=certificate.verdict_code,
        allowed_conclusion=certificate.allowed_conclusion,
        overall_pass=certificate.overall_study_pass,
        claim_p_pass=claim_p_pass,
        claim_p_practical_pass=claim_p_practical_pass,
        claim_s_pass=claim_s_pass,
        cell_consistency_gate_pass=cell_consistency_gate_pass,
        positive_cells_by_comparator=positive_cells_by_comp,
        model_family_means_by_comparator=model_family_means_by_comp,
        selection_contrasts=selection_contrasts,
        stability_summary=stability_summary,
        certificate=certificate,
        supporting_diagnostics=decision_input.supporting_diagnostics,
    )
