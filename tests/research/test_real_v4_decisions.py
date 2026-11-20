"""Hand-calculated boundary, four-outcome matrix, coverage, and mutation tests for Real-v4 decisions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from research.real_v4.decisions import (
    DecisionThresholds,
    StudyDecisionInput,
    TaskEnvironmentGroupData,
    evaluate_study_decisions,
)
from research.real_v4.schemas import (
    CertificateStatus,
    OverallVerdictCode,
    SupportingDiagnostics,
)
from tests.research.test_real_v4_schemas import make_sample_envelope


def _build_synthetic_strata(
    envelope,
    *,
    n_items: int = 80,
    prism_ref: float = 0.75,
    random_ref: float = 0.72,
    bo_ref: float = 0.725,
    prism_pert_drop: float = 0.002,
    worst_pert_env: str = "word_paraphrase_1",
    random_pert_drop: float = 0.008,
    bo_pert_drop: float = 0.007,
    noise_sd: float = 0.0,
    seed: int = 1234,
) -> List[TaskEnvironmentGroupData]:
    rng = np.random.default_rng(seed)
    strata: List[TaskEnvironmentGroupData] = []
    for task_id in envelope.tasks:
        ref_groups = tuple(f"{task_id}:g_{i // 2:03d}" for i in range(n_items))
        holdout_groups = tuple(
            f"{task_id}:holdout_{i // 2:03d}" for i in range(n_items)
        )
        # Shared item difficulty within task
        item_base = (
            rng.normal(0.0, noise_sd, size=n_items)
            if noise_sd > 0
            else np.zeros(n_items)
        )

        for spec in envelope.registered_environments:
            env_id = spec.env_id
            g_ids = holdout_groups if spec.uses_disjoint_item_group else ref_groups
            scores: Dict[str, Dict[str, np.ndarray]] = {}
            for model_id in envelope.models:
                if env_id == "ref":
                    p_mu = prism_ref
                    r_mu = random_ref
                    b_mu = bo_ref
                elif env_id == worst_pert_env:
                    p_mu = prism_ref - prism_pert_drop
                    r_mu = random_ref - random_pert_drop
                    b_mu = bo_ref - bo_pert_drop
                else:
                    p_mu = prism_ref - 0.5 * prism_pert_drop
                    r_mu = random_ref - 0.5 * random_pert_drop
                    b_mu = bo_ref - 0.5 * bo_pert_drop

                eps_p = rng.normal(0.0, noise_sd, size=n_items) if noise_sd > 0 else 0.0
                eps_r = rng.normal(0.0, noise_sd, size=n_items) if noise_sd > 0 else 0.0
                eps_b = rng.normal(0.0, noise_sd, size=n_items) if noise_sd > 0 else 0.0
                scores[model_id] = {
                    "promptprism": np.clip(p_mu + item_base + eps_p, 0.0, 1.0),
                    "balanced_random": np.clip(r_mu + item_base + eps_r, 0.0, 1.0),
                    "bayesian_opt": np.clip(b_mu + item_base + eps_b, 0.0, 1.0),
                }
            strata.append(
                TaskEnvironmentGroupData(
                    task_id=task_id,
                    env_id=env_id,
                    group_ids=g_ids,
                    scores=scores,
                )
            )
    return strata


def test_four_outcome_interpretation_matrix_and_exact_boundaries():
    env = make_sample_envelope()
    obs_versions = {e_id: dict(env.model_versions_pinned) for e_id in env.all_env_ids}
    thresholds = DecisionThresholds(
        alpha=0.05, delta_select=0.01, epsilon_stability=0.01, n_bootstrap=200, seed=42
    )

    # 1. Pass / Pass (noiseless hand-verifiable fixture)
    s_pp = _build_synthetic_strata(
        env,
        prism_ref=0.75,
        random_ref=0.73,
        bo_ref=0.735,
        prism_pert_drop=0.004,
        noise_sd=0.0,
    )
    res_pp = evaluate_study_decisions(
        StudyDecisionInput("cert-pp", env.study_id, "v4.0", env, s_pp, obs_versions),
        thresholds=thresholds,
    )
    assert res_pp.claim_p_pass is True
    assert res_pp.claim_p_practical_pass is True
    assert res_pp.claim_s_pass is True
    assert res_pp.overall_pass is True
    assert res_pp.verdict_code == OverallVerdictCode.PASS_SELECTION_AND_STABILITY
    assert res_pp.certificate.status == CertificateStatus.PASS
    assert np.isclose(res_pp.selection_contrasts["balanced_random"].estimate, 0.02)
    assert np.isclose(res_pp.selection_contrasts["bayesian_opt"].estimate, 0.015)
    assert np.isclose(res_pp.stability_summary.d_stability_estimate, 0.004)

    # 2. Pass / Fail (selection wins, stability drop exceeds 0.01)
    s_pf = _build_synthetic_strata(
        env,
        prism_ref=0.75,
        random_ref=0.73,
        bo_ref=0.735,
        prism_pert_drop=0.015,
        noise_sd=0.0,
    )
    res_pf = evaluate_study_decisions(
        StudyDecisionInput("cert-pf", env.study_id, "v4.0", env, s_pf, obs_versions),
        thresholds=thresholds,
    )
    assert res_pf.claim_p_pass is True
    assert res_pf.claim_s_pass is False
    assert res_pf.overall_pass is False
    assert res_pf.verdict_code == OverallVerdictCode.PASS_SELECTION_ONLY
    assert res_pf.certificate.status == CertificateStatus.FAIL

    # 3. Fail / Pass (selection loses against bayesian_opt, but prompt is stable and noninferior on worst env)
    s_fp = _build_synthetic_strata(
        env,
        prism_ref=0.735,
        random_ref=0.73,
        bo_ref=0.735,  # exact zero gain vs BO -> must FAIL Claim P!
        prism_pert_drop=0.002,
        bo_pert_drop=0.005,
        noise_sd=0.0,
    )
    res_fp = evaluate_study_decisions(
        StudyDecisionInput("cert-fp", env.study_id, "v4.0", env, s_fp, obs_versions),
        thresholds=thresholds,
    )
    assert res_fp.selection_contrasts["balanced_random"].passes_statistical_gate is True
    assert res_fp.selection_contrasts["bayesian_opt"].passes_statistical_gate is False
    assert res_fp.claim_p_pass is False
    assert res_fp.claim_s_pass is True
    assert res_fp.overall_pass is False
    assert res_fp.verdict_code == OverallVerdictCode.PASS_STABILITY_ONLY
    assert res_fp.certificate.status == CertificateStatus.PASS

    # 4. Fail / Fail (harmful selection AND unstable)
    s_ff = _build_synthetic_strata(
        env,
        prism_ref=0.70,
        random_ref=0.73,
        bo_ref=0.74,
        prism_pert_drop=0.025,
        noise_sd=0.0,
    )
    res_ff = evaluate_study_decisions(
        StudyDecisionInput("cert-ff", env.study_id, "v4.0", env, s_ff, obs_versions),
        thresholds=thresholds,
    )
    assert res_ff.claim_p_pass is False
    assert res_ff.claim_s_pass is False
    assert res_ff.overall_pass is False
    assert res_ff.verdict_code == OverallVerdictCode.FAIL_BOTH
    assert res_ff.certificate.status == CertificateStatus.FAIL

    # 5. Exact equality boundary checks: D_stability == epsilon_stability (0.01) MUST FAIL strict '<'
    s_exact_stab = _build_synthetic_strata(
        env,
        prism_ref=0.75,
        random_ref=0.72,
        bo_ref=0.72,
        prism_pert_drop=0.01,
        noise_sd=0.0,
    )
    res_exact_stab = evaluate_study_decisions(
        StudyDecisionInput(
            "cert-exact-stab", env.study_id, "v4.0", env, s_exact_stab, obs_versions
        ),
        thresholds=thresholds,
    )
    assert res_exact_stab.stability_summary.passes_absolute_noninferiority is False
    assert res_exact_stab.claim_s_pass is False

    # 6. Exact equality boundary on G_robust == -epsilon_stability (-0.01) MUST FAIL strict '>'
    s_exact_rob = _build_synthetic_strata(
        env,
        prism_ref=0.73,
        random_ref=0.738,
        bo_ref=0.738,
        prism_pert_drop=0.002,  # prism min = 0.728
        random_pert_drop=0.000,  # random min = 0.738 -> G_robust = -0.010 exact!
        bo_pert_drop=0.000,
        noise_sd=0.0,
    )
    res_exact_rob = evaluate_study_decisions(
        StudyDecisionInput(
            "cert-exact-rob", env.study_id, "v4.0", env, s_exact_rob, obs_versions
        ),
        thresholds=thresholds,
    )
    assert res_exact_rob.stability_summary.passes_absolute_noninferiority is True
    assert res_exact_rob.stability_summary.passes_comparator_worst_env is False
    assert res_exact_rob.claim_s_pass is False


def test_supporting_diagnostics_cannot_override_failed_primary_claims():
    env = make_sample_envelope()
    obs_versions = {e_id: dict(env.model_versions_pinned) for e_id in env.all_env_ids}
    thresholds = DecisionThresholds(n_bootstrap=200, seed=99)

    s_null = _build_synthetic_strata(
        env,
        prism_ref=0.72,
        random_ref=0.72,
        bo_ref=0.72,
        prism_pert_drop=0.02,
        noise_sd=0.0,
    )
    strong_supporting = SupportingDiagnostics(
        effect_recovery_rmse_prism=0.001,
        effect_recovery_rmse_random=0.050,
        effect_recovery_gain=0.049,
        identical_repeat_exact_match_rate=1.0,
        score_variance_across_repeats=0.0,
        selection_resample_agreement=1.0,
    )
    res = evaluate_study_decisions(
        StudyDecisionInput(
            "cert-no-rescue",
            env.study_id,
            "v4.0",
            env,
            s_null,
            obs_versions,
            supporting_diagnostics=strong_supporting,
        ),
        thresholds=thresholds,
    )
    assert res.claim_p_pass is False
    assert res.claim_s_pass is False
    assert res.overall_pass is False
    assert res.verdict_code == OverallVerdictCode.FAIL_BOTH
    assert res.supporting_diagnostics.effect_recovery_gain == pytest.approx(0.049)


def test_missing_environment_missing_cell_and_version_drift_fail_closed():
    env = make_sample_envelope()
    obs_versions = {e_id: dict(env.model_versions_pinned) for e_id in env.all_env_ids}
    thresholds = DecisionThresholds(n_bootstrap=200, seed=7)
    s_good = _build_synthetic_strata(
        env,
        prism_ref=0.78,
        random_ref=0.72,
        bo_ref=0.72,
        prism_pert_drop=0.001,
        noise_sd=0.0,
    )

    # Mutation 1: Drop one environment ('temporal_wave_2') -> must fail closed!
    s_dropped_env = [st for st in s_good if st.env_id != "temporal_wave_2"]
    res_drop_env = evaluate_study_decisions(
        StudyDecisionInput(
            "cert-drop-env", env.study_id, "v4.0", env, s_dropped_env, obs_versions
        ),
        thresholds=thresholds,
    )
    assert res_drop_env.verdict_code == OverallVerdictCode.INVALID_OUT_OF_ENVELOPE
    assert res_drop_env.certificate.status == CertificateStatus.INVALID_OUT_OF_ENVELOPE
    assert res_drop_env.overall_pass is False

    # Mutation 2: Drop one model from one cell -> must fail closed!
    first = s_good[0]
    partial_scores = {k: v for k, v in first.scores.items() if k != env.models[0]}
    s_missing_model = [
        TaskEnvironmentGroupData(
            first.task_id, first.env_id, first.group_ids, partial_scores
        ),
        *s_good[1:],
    ]
    res_missing_cell = evaluate_study_decisions(
        StudyDecisionInput(
            "cert-miss-cell", env.study_id, "v4.0", env, s_missing_model, obs_versions
        ),
        thresholds=thresholds,
    )
    assert (
        res_missing_cell.certificate.status == CertificateStatus.INVALID_OUT_OF_ENVELOPE
    )
    assert res_missing_cell.certificate.missing_cells_count >= 1

    # Mutation 3: Provider version drift in temporal_wave_2 -> must fail closed!
    drifted = {k: dict(v) for k, v in obs_versions.items()}
    drifted["temporal_wave_2"][
        "claude-3-5-haiku-20241022"
    ] = "claude-3-5-haiku-20250301"
    res_drift = evaluate_study_decisions(
        StudyDecisionInput("cert-drift", env.study_id, "v4.0", env, s_good, drifted),
        thresholds=thresholds,
    )
    assert res_drift.certificate.status == CertificateStatus.INVALID_OUT_OF_ENVELOPE
    assert res_drift.claim_s_pass is False


def test_monte_carlo_null_type1_and_simultaneous_noninferiority_coverage():
    env = make_sample_envelope()
    obs_versions = {e_id: dict(env.model_versions_pinned) for e_id in env.all_env_ids}

    n_trials = 60
    false_claim_p = 0
    false_claim_s_at_boundary = 0
    for trial in range(n_trials):
        # Null selection (prism == random == bo == 0.70) AND boundary instability (true drop == 0.01 == epsilon)
        strata = _build_synthetic_strata(
            env,
            n_items=60,
            prism_ref=0.70,
            random_ref=0.70,
            bo_ref=0.70,
            prism_pert_drop=0.01,  # right at noninferiority boundary H0: D_stability = epsilon_stability
            random_pert_drop=0.01,
            bo_pert_drop=0.01,
            noise_sd=0.04,
            seed=5000 + trial,
        )
        res = evaluate_study_decisions(
            StudyDecisionInput(
                f"cert-mc-{trial}", env.study_id, "v4.0", env, strata, obs_versions
            ),
            thresholds=DecisionThresholds(
                alpha=0.05,
                delta_select=0.01,
                epsilon_stability=0.01,
                n_bootstrap=200,
                seed=8000 + trial,
            ),
        )
        if res.claim_p_pass:
            false_claim_p += 1
        if res.claim_s_pass:
            false_claim_s_at_boundary += 1

    # Intersection-union across two comparators under global null should have <= 5% (+MC slack) Type-I rate
    assert false_claim_p / n_trials <= 0.09
    # Boundary noninferiority H0 (true drop == epsilon_stability on one environment) should reject <= ~10% w/ MC slack
    assert false_claim_s_at_boundary / n_trials <= 0.12
