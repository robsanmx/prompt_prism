"""
Experimental Two-Stage Prompt Selection and Bounded Envelope Certificate Contracts.

Public API Promotion Gate Status (Real-v4 Prospective Study):
- Registered promotion gate required BOTH Claim P (equal-budget selection superiority) and
  Claim S (8-environment simultaneous max-drop stability < 0.02) to pass.
- Audited Real-v4 verdict (`research/results/real_v4/summary.json`): `FAIL_BOTH`
  (Claim P: +0.22 pp vs balanced random [L95=-0.15 pp] and +0.14 pp vs Bayesian optimization
  [L95=-0.22 pp]; Claim S: full 8-environment U95_sim=0.0433 >= 0.02 due to unpaired item-holdout
  sampling variance and hosted HTTP 429 zero-scoring, while paired prompt/repeat/temporal
  perturbations satisfied U95_sim=0.0048 < 0.02; supporting 6-factor main-effect RMSE fell 35.38%).
- Consequently, `PROMOTED_TO_DEFAULT_PUBLIC_API` is `False`: existing `additive`, `screen_best`,
  and `interaction_2fi` workflows remain unchanged and are reported strictly as `predicted`
  unless validated on untouched confirmation data, while two-stage selection is exposed only
  via explicit `experimental_opt_in=True` without marketing a default stability assurance.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.models import (
    OperatingEnvelopeSpec,
    PromptEvidenceState,
    PromptSelectionReport,
    PromptStabilityCertificate,
)
from .optimizer import OptimalPromptRecommendation

PROMOTED_TO_DEFAULT_PUBLIC_API: bool = False
REAL_V4_REGISTERED_VERDICT: str = "FAIL_BOTH"
EXPERIMENTAL_TWO_STAGE_ALGORITHM_VERSION: str = (
    "promptprism_experimental_two_stage_v4.0.0"
)


def build_selection_report_from_recommendation(
    recommendation: OptimalPromptRecommendation,
    *,
    confirmation_scores: Optional[Sequence[float]] = None,
    confirmation_envelope: Optional[OperatingEnvelopeSpec] = None,
    stability_certificate: Optional[PromptStabilityCertificate] = None,
    request_envelope: Optional[OperatingEnvelopeSpec] = None,
) -> PromptSelectionReport:
    """Build an explicit `PromptSelectionReport` from an existing `OptimalPromptRecommendation`.

    Existing `additive`, `screen_best`, and `interaction_2fi` recommendations default strictly
    to `evidence_state="predicted"` (`can_display_passing_stability_badge=False`) and are never
    relabeled as confirmation-validated or stability-certified without untouched evidence and a
    matching operating envelope.
    """
    conf_mean: Optional[float] = None
    has_conf = False
    notes: List[str] = []

    if confirmation_scores is not None:
        arr = np.asarray(list(confirmation_scores), dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(
                "confirmation_scores must be a non-empty sequence of finite floats"
            )
        conf_mean = float(np.mean(arr))
        if confirmation_envelope is not None:
            has_conf = True
        else:
            notes.append(
                "Confirmation scores provided without OperatingEnvelopeSpec; retaining 'predicted' tier."
            )

    if stability_certificate is not None:
        eff_req_env = request_envelope or confirmation_envelope
        val_res = stability_certificate.validate_request(
            eff_req_env, has_confirmation_validation=has_conf, strict=False
        )
        if val_res.mismatch_reasons:
            notes.append(
                "Envelope validation diagnostics: "
                + ", ".join(val_res.mismatch_reasons)
            )
        return PromptSelectionReport(
            strategy=recommendation.strategy,
            evidence_state=val_res.evidence_state,
            experimental=stability_certificate.experimental,
            selected_factor_levels=dict(recommendation.optimal_factor_levels),
            predicted_or_screen_score=float(recommendation.predicted_optimal_score),
            confirmation_score=conf_mean,
            can_display_passing_stability_badge=val_res.can_display_passing_stability_badge,
            stability_badge_label=val_res.badge_label,
            envelope=eff_req_env or stability_certificate.envelope,
            certificate=stability_certificate,
            notes=notes,
        )

    if has_conf:
        return PromptSelectionReport(
            strategy=recommendation.strategy,
            evidence_state=PromptEvidenceState.CONFIRMATION_VALIDATED,
            experimental=False,
            selected_factor_levels=dict(recommendation.optimal_factor_levels),
            predicted_or_screen_score=float(recommendation.predicted_optimal_score),
            confirmation_score=conf_mean,
            can_display_passing_stability_badge=False,
            stability_badge_label="CONFIRMATION-VALIDATED ONLY (NO PERTURBATION CERTIFICATE)",
            envelope=confirmation_envelope,
            certificate=None,
            notes=notes,
        )

    notes.append(
        "Single-stage screen/model output only; evaluate on untouched confirmation items before deployment."
    )
    return PromptSelectionReport(
        strategy=recommendation.strategy,
        evidence_state=PromptEvidenceState.PREDICTED,
        experimental=False,
        selected_factor_levels=dict(recommendation.optimal_factor_levels),
        predicted_or_screen_score=float(recommendation.predicted_optimal_score),
        confirmation_score=conf_mean,
        can_display_passing_stability_badge=False,
        stability_badge_label="UNVERIFIED (SINGLE-STAGE PREDICTED ONLY)",
        envelope=confirmation_envelope,
        certificate=None,
        notes=notes,
    )


def evaluate_simultaneous_perturbation_certificate(
    *,
    envelope: OperatingEnvelopeSpec,
    reference_scores: Sequence[float],
    perturbation_scores_by_env: Mapping[str, Sequence[float]],
    group_ids: Optional[Sequence[str]] = None,
    epsilon_stability: float = 0.02,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 20261118,
    observed_model_versions_by_env: Optional[Mapping[str, str]] = None,
) -> PromptStabilityCertificate:
    """Compute a fail-closed `PromptStabilityCertificate` across registered perturbation environments."""
    if epsilon_stability <= 0.0:
        raise ValueError("epsilon_stability must be > 0")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")

    ref_arr = np.asarray(list(reference_scores), dtype=float)
    n_items = ref_arr.size
    expected_envs = sorted(set(envelope.perturbation_env_ids))

    missing_cells = 0
    drift_events: List[str] = []

    if n_items == 0 or not np.all(np.isfinite(ref_arr)):
        missing_cells += 1

    if not expected_envs:
        drift_events.append("no_registered_perturbation_environments")

    aligned_pert: Dict[str, np.ndarray] = {}
    for env_id in expected_envs:
        if env_id not in perturbation_scores_by_env:
            missing_cells += 1
            continue
        p_arr = np.asarray(list(perturbation_scores_by_env[env_id]), dtype=float)
        if p_arr.size != n_items or not np.all(np.isfinite(p_arr)):
            missing_cells += 1
            drift_events.append(f"unpaired_or_invalid_scores:{env_id}")
            continue
        aligned_pert[env_id] = p_arr

    if observed_model_versions_by_env is not None:
        for env_id, obs_ver in sorted(observed_model_versions_by_env.items()):
            if str(obs_ver) != envelope.model_version:
                drift_events.append(
                    f"model_version_drift:{env_id}:expected={envelope.model_version},observed={obs_ver}"
                )

    if missing_cells > 0 or drift_events:
        return PromptStabilityCertificate(
            experimental=True,
            envelope=envelope,
            status="INVALID/OUT_OF_ENVELOPE",
            reference_mean=(
                float(np.mean(ref_arr))
                if n_items > 0 and np.all(np.isfinite(ref_arr))
                else None
            ),
            d_stability_estimate=None,
            d_stability_upper_95_sim=None,
            epsilon_stability=epsilon_stability,
            missing_cells_count=missing_cells,
            drift_events=drift_events,
        )

    ref_mean = float(np.mean(ref_arr))
    drops = {
        e: float(ref_mean - float(np.mean(aligned_pert[e]))) for e in expected_envs
    }
    d_est = float(max(drops.values()))

    # Paired item-group bootstrap for simultaneous studentized maximum pivot
    if group_ids is None:
        g_lists = [np.array([i], dtype=int) for i in range(n_items)]
    else:
        if len(group_ids) != n_items:
            raise ValueError("group_ids length must match reference_scores length")
        buckets: Dict[str, List[int]] = {}
        for idx, g in enumerate(group_ids):
            buckets.setdefault(str(g), []).append(idx)
        g_lists = [np.array(v, dtype=int) for v in buckets.values()]

    rng = np.random.default_rng(seed)
    boot_drops = np.zeros((n_bootstrap, len(expected_envs)), dtype=float)
    n_g = len(g_lists)
    for b in range(n_bootstrap):
        picks = np.concatenate([g_lists[i] for i in rng.integers(0, n_g, size=n_g)])
        r_b = float(np.mean(ref_arr[picks]))
        for e_i, e_id in enumerate(expected_envs):
            boot_drops[b, e_i] = r_b - float(np.mean(aligned_pert[e_id][picks]))

    se_vec = np.std(boot_drops, axis=0, ddof=1)
    if np.all(se_vec < 1e-12):
        u95_sim = d_est
    else:
        se_safe = np.maximum(se_vec, 1e-12)
        d_vec = np.array([drops[e] for e in expected_envs], dtype=float)
        t_max = np.max((boot_drops - d_vec[None, :]) / se_safe[None, :], axis=1)
        q_alpha = float(np.percentile(t_max, 100.0 * (1.0 - alpha)))
        u95_sim = float(np.max(d_vec + q_alpha * se_vec))

    passes = bool(u95_sim < epsilon_stability - 1e-12)
    return PromptStabilityCertificate(
        experimental=True,
        envelope=envelope,
        status="PASS" if passes else "FAIL",
        reference_mean=ref_mean,
        d_stability_estimate=d_est,
        d_stability_upper_95_sim=u95_sim,
        epsilon_stability=epsilon_stability,
        missing_cells_count=0,
        drift_events=[],
    )


class ExperimentalTwoStageSelector:
    """Experimental opt-in two-stage prompt selector (screen -> multi-env simultaneous-LCB validation).

    Requires `experimental_opt_in=True` because the Real-v4 prospective promotion gate (`PASS_BOTH`)
    was not met (`FAIL_BOTH`).
    """

    def __init__(
        self,
        *,
        experimental_opt_in: bool = False,
        max_candidates: int = 4,
        alpha_validation: float = 0.05,
        n_bootstrap: int = 200,
        seed: int = 20261118,
    ) -> None:
        if not experimental_opt_in:
            raise ValueError(
                "ExperimentalTwoStageSelector requires `experimental_opt_in=True`. "
                "Prospective Real-v4 evaluation resulted in `FAIL_BOTH` (no statistically significant "
                "equal-budget selection gain over balanced random / Bayesian optimization, and full "
                "8-environment U95_sim=0.0433 >= 0.02), so this selector is not promoted as a default assurance."
            )
        if not (1 <= max_candidates <= 8):
            raise ValueError("max_candidates must be in 1..8")
        if n_bootstrap < 50:
            raise ValueError("n_bootstrap must be >= 50")
        self.max_candidates = max_candidates
        self.alpha_validation = alpha_validation
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def generate_candidates_from_screen(
        self,
        *,
        factor_ids: Sequence[str],
        run_factor_levels: Mapping[int, Mapping[str, int]],
        screen_scores_by_run: Mapping[int, Sequence[float]],
        maximize: bool = True,
    ) -> List[Tuple[int, Dict[str, int], str, float]]:
        """Generate up to `max_candidates` unique `(run_id, factor_levels, source_rule, screen_score)` tuples."""
        f_ids = list(factor_ids)
        k = len(f_ids)
        sign = 1.0 if maximize else -1.0

        run_means = {
            int(rid): float(np.mean(np.asarray(list(scores), dtype=float)))
            for rid, scores in sorted(screen_scores_by_run.items())
        }
        if not run_means:
            raise ValueError("screen_scores_by_run cannot be empty")

        # Build full binary grid if all levels are in {0, 1} and k <= 10
        observed_by_tuple = {
            tuple(int(run_factor_levels[rid][f]) for f in f_ids): int(rid)
            for rid in sorted(run_means)
        }
        best_obs_rid = max(sorted(run_means), key=lambda r: (sign * run_means[r], -r))

        # Fit main-effect linear model over {-1, +1}^k
        rids_sorted = sorted(run_means)
        x_pm = np.array(
            [
                [2.0 * float(run_factor_levels[r][f]) - 1.0 for f in f_ids]
                for r in rids_sorted
            ],
            dtype=float,
        )
        y_vec = np.array([sign * run_means[r] for r in rids_sorted], dtype=float)
        d_mat = np.column_stack([np.ones(len(rids_sorted)), x_pm])
        beta = np.linalg.pinv(d_mat) @ y_vec
        contrasts = 2.0 * beta[1:]  # main-effect contrasts

        # Candidate 1: main-effect sign composition (0 contrast -> 0)
        sign_tuple = tuple(1 if c > 1e-12 else 0 for c in contrasts)
        # Candidate 3: conservative thresholded recipe (retain only contrasts exceeding half median |contrast|)
        thresh = 0.5 * float(np.median(np.abs(contrasts))) if k > 0 else 0.0
        cons_tuple = tuple(1 if c > max(1e-12, thresh) else 0 for c in contrasts)
        # Candidate 4: flip weakest active factor from sign_tuple
        if k > 0:
            weak_idx = int(np.argmin(np.abs(contrasts)))
            flip_list = list(sign_tuple)
            flip_list[weak_idx] = 1 - flip_list[weak_idx]
            flip_tuple = tuple(flip_list)
        else:
            flip_tuple = sign_tuple

        def _tuple_to_rid_and_levels(
            tpl: Tuple[int, ...],
        ) -> Tuple[int, Dict[str, int], float]:
            lvls = {f: int(tpl[i]) for i, f in enumerate(f_ids)}
            if tpl in observed_by_tuple:
                rid = observed_by_tuple[tpl]
                sc = run_means[rid]
            else:
                # Deterministic synthetic cube index >= 1000 when unobserved in screen
                bit_id = sum(int(tpl[i]) << (k - 1 - i) for i in range(k))
                rid = 1000 + bit_id
                pred = float(
                    beta[0] + np.dot(2.0 * np.array(tpl, dtype=float) - 1.0, beta[1:])
                )
                sc = sign * pred
            return rid, lvls, sc

        raw_proposals = [
            (*_tuple_to_rid_and_levels(sign_tuple), "main_effect_sign"),
            (
                best_obs_rid,
                {f: int(run_factor_levels[best_obs_rid][f]) for f in f_ids},
                run_means[best_obs_rid],
                "screen_best",
            ),
            (*_tuple_to_rid_and_levels(cons_tuple), "conservative_threshold"),
            (*_tuple_to_rid_and_levels(flip_tuple), "weakest_factor_flip"),
        ]

        chosen: List[Tuple[int, Dict[str, int], str, float]] = []
        seen_tuples = set()
        for rid, lvls, sc, rule in raw_proposals:
            tpl = tuple(lvls[f] for f in f_ids)
            if tpl not in seen_tuples and len(chosen) < self.max_candidates:
                seen_tuples.add(tpl)
                chosen.append((rid, lvls, rule, sc))

        # Pad with remaining observed screen runs in descending score order if needed
        for rid in sorted(rids_sorted, key=lambda r: (-sign * run_means[r], r)):
            if len(chosen) >= self.max_candidates:
                break
            lvls = {f: int(run_factor_levels[rid][f]) for f in f_ids}
            tpl = tuple(lvls[f] for f in f_ids)
            if tpl not in seen_tuples:
                seen_tuples.add(tpl)
                chosen.append((rid, lvls, "screen_runner_up", run_means[rid]))

        return chosen

    def select_on_validation(
        self,
        *,
        candidates: Sequence[Tuple[int, Dict[str, int], str, float]],
        validation_scores_by_env: Mapping[str, Mapping[int, Sequence[float]]],
        group_ids: Optional[Sequence[str]] = None,
        maximize: bool = True,
        confirmation_scores: Optional[Sequence[float]] = None,
        confirmation_envelope: Optional[OperatingEnvelopeSpec] = None,
        stability_certificate: Optional[PromptStabilityCertificate] = None,
        request_envelope: Optional[OperatingEnvelopeSpec] = None,
    ) -> PromptSelectionReport:
        """Select the candidate maximizing simultaneous validation LCB across validation environments."""
        if not candidates:
            raise ValueError("candidates cannot be empty")
        if not validation_scores_by_env:
            raise ValueError("validation_scores_by_env cannot be empty")

        sign = 1.0 if maximize else -1.0
        env_ids = sorted(validation_scores_by_env.keys())
        first_rid = candidates[0][0]
        n_items = len(validation_scores_by_env[env_ids[0]][first_rid])

        if group_ids is None:
            g_lists = [np.array([i], dtype=int) for i in range(n_items)]
        else:
            buckets: Dict[str, List[int]] = {}
            for idx, g in enumerate(group_ids):
                buckets.setdefault(str(g), []).append(idx)
            g_lists = [np.array(v, dtype=int) for v in buckets.values()]

        rng = np.random.default_rng(self.seed)
        n_g = len(g_lists)
        boot_picks = [
            np.concatenate([g_lists[i] for i in rng.integers(0, n_g, size=n_g)])
            for _ in range(self.n_bootstrap)
        ]

        best_tuple: Optional[Tuple[float, float, int, Dict[str, int], str]] = None
        for rid, lvls, rule, _scr in candidates:
            env_arrs = {
                e: sign
                * np.asarray(list(validation_scores_by_env[e][rid]), dtype=float)
                for e in env_ids
            }
            env_means = np.array(
                [float(np.mean(env_arrs[e])) for e in env_ids], dtype=float
            )
            worst_mean = float(np.min(env_means))

            boot_mat = np.zeros((self.n_bootstrap, len(env_ids)), dtype=float)
            for b, picks in enumerate(boot_picks):
                for e_i, e in enumerate(env_ids):
                    boot_mat[b, e_i] = float(np.mean(env_arrs[e][picks]))
            se_vec = np.std(boot_mat, axis=0, ddof=1)
            if np.all(se_vec < 1e-12):
                lcb = worst_mean
            else:
                se_safe = np.maximum(se_vec, 1e-12)
                t_min = np.min(
                    (boot_mat - env_means[None, :]) / se_safe[None, :], axis=1
                )
                q_low = float(np.percentile(t_min, 100.0 * self.alpha_validation))
                lcb = float(np.min(env_means + q_low * se_vec))

            cand_key = (lcb, worst_mean, -int(rid), lvls, rule)
            if best_tuple is None or cand_key[:3] > best_tuple[:3]:
                best_tuple = (lcb, worst_mean, -int(rid), lvls, rule)

        assert best_tuple is not None
        win_lcb, win_worst, neg_rid, win_lvls, win_rule = best_tuple
        win_rid = -neg_rid
        reported_val_score = sign * win_worst

        conf_mean: Optional[float] = None
        has_conf = False
        notes = [
            f"Experimental opt-in two-stage selection (winner rule={win_rule}, validation LCB={sign * win_lcb:.4f}).",
            "Prospective Real-v4 promotion gate (`PASS_BOTH`) was not met (`FAIL_BOTH`); no universal assurance is claimed.",
        ]
        if confirmation_scores is not None:
            conf_arr = np.asarray(list(confirmation_scores), dtype=float)
            conf_mean = float(np.mean(conf_arr))
            has_conf = confirmation_envelope is not None

        if stability_certificate is not None:
            eff_env = request_envelope or confirmation_envelope
            val_res = stability_certificate.validate_request(
                eff_env, has_confirmation_validation=has_conf, strict=False
            )
            if val_res.mismatch_reasons:
                notes.append(
                    "Envelope validation diagnostics: "
                    + ", ".join(val_res.mismatch_reasons)
                )
            return PromptSelectionReport(
                strategy="experimental_two_stage",
                evidence_state=val_res.evidence_state,
                experimental=True,
                selected_factor_levels=dict(win_lvls),
                selected_run_id=win_rid,
                predicted_or_screen_score=reported_val_score,
                confirmation_score=conf_mean,
                can_display_passing_stability_badge=val_res.can_display_passing_stability_badge,
                stability_badge_label=val_res.badge_label,
                envelope=eff_env or stability_certificate.envelope,
                certificate=stability_certificate,
                notes=notes,
            )

        if has_conf:
            return PromptSelectionReport(
                strategy="experimental_two_stage",
                evidence_state=PromptEvidenceState.CONFIRMATION_VALIDATED,
                experimental=True,
                selected_factor_levels=dict(win_lvls),
                selected_run_id=win_rid,
                predicted_or_screen_score=reported_val_score,
                confirmation_score=conf_mean,
                can_display_passing_stability_badge=False,
                stability_badge_label="CONFIRMATION-VALIDATED ONLY (NO PERTURBATION CERTIFICATE)",
                envelope=confirmation_envelope,
                certificate=None,
                notes=notes,
            )

        return PromptSelectionReport(
            strategy="experimental_two_stage",
            evidence_state=PromptEvidenceState.PREDICTED,
            experimental=True,
            selected_factor_levels=dict(win_lvls),
            selected_run_id=win_rid,
            predicted_or_screen_score=reported_val_score,
            confirmation_score=conf_mean,
            can_display_passing_stability_badge=False,
            stability_badge_label="UNVERIFIED (NO CONFIRMATION OR STABILITY CERTIFICATE)",
            envelope=confirmation_envelope,
            certificate=None,
            notes=notes,
        )
