"""Fail-closed Bounded Stability Certificate contract for Real-v4."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import Field, model_validator

from research.real_v4.schemas import (
    ALLOWED_CONCLUSION_MAP,
    CURRENT_SCHEMA_VERSION,
    CertificateStatus,
    ComparatorContrastSummary,
    OperatingEnvelope,
    OverallVerdictCode,
    StabilitySummary,
    SupportingDiagnostics,
    _VersionedBaseModel,
    canonical_sha256,
)


class EnvelopeVerificationResult(_VersionedBaseModel):
    is_within_envelope: bool
    effective_status: CertificateStatus
    reasons: Tuple[str, ...] = ()


class StabilityCertificate(_VersionedBaseModel):
    certificate_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    algorithm_version: str = Field(min_length=1)
    status: CertificateStatus
    verdict_code: OverallVerdictCode
    allowed_conclusion: str = Field(min_length=1)
    claim_p_pass: bool
    claim_p_practical_pass: bool
    claim_s_pass: bool
    overall_study_pass: bool
    envelope: OperatingEnvelope
    envelope_sha256: str = Field(min_length=64, max_length=64)
    observed_environments: Tuple[str, ...]
    observed_model_versions: Dict[str, Dict[str, str]]
    missing_cells_count: int = Field(ge=0)
    incomplete_pairs_count: int = Field(ge=0)
    hash_mismatches: Tuple[str, ...] = ()
    drift_events: Tuple[str, ...] = ()
    selection_contrasts: Dict[str, ComparatorContrastSummary] = Field(default_factory=dict)
    stability_summary: Optional[StabilitySummary] = None
    supporting_diagnostics: Optional[SupportingDiagnostics] = None

    @model_validator(mode="after")
    def _enforce_fail_closed_invariants(self) -> "StabilityCertificate":
        expected_env_hash = self.envelope.envelope_sha256()
        if self.envelope_sha256 != expected_env_hash:
            raise ValueError(
                f"envelope_sha256 mismatch: expected {expected_env_hash}, got {self.envelope_sha256}"
            )

        registered_envs = set(self.envelope.all_env_ids)
        observed_envs = set(self.observed_environments)
        missing_envs = sorted(registered_envs - observed_envs)
        unregistered_envs = sorted(observed_envs - registered_envs)

        # Detect model version drift on any environment requiring identical model version
        detected_drift: List[str] = list(self.drift_events)
        env_spec_map = {e.env_id: e for e in self.envelope.registered_environments}
        for env_id, model_map in sorted(self.observed_model_versions.items()):
            spec = env_spec_map.get(env_id)
            if spec is not None and spec.requires_identical_model_version:
                for model_id, pinned_ver in sorted(self.envelope.model_versions_pinned.items()):
                    obs_ver = model_map.get(model_id)
                    if obs_ver != pinned_ver:
                        msg = f"{env_id}:{model_id}:expected={pinned_ver},observed={obs_ver}"
                        if msg not in detected_drift:
                            detected_drift.append(msg)

        has_envelope_breach = bool(
            missing_envs
            or unregistered_envs
            or self.missing_cells_count > 0
            or self.incomplete_pairs_count > 0
            or len(self.hash_mismatches) > 0
            or len(detected_drift) > 0
        )

        if has_envelope_breach:
            if self.status != CertificateStatus.INVALID_OUT_OF_ENVELOPE:
                raise ValueError(
                    "Certificate cannot be PASS or ordinary FAIL when envelope integrity is breached; "
                    "expected status='INVALID/OUT_OF_ENVELOPE'"
                )
            if self.claim_s_pass or self.overall_study_pass:
                raise ValueError("claim_s_pass and overall_study_pass must be False under envelope breach")
            if self.verdict_code != OverallVerdictCode.INVALID_OUT_OF_ENVELOPE:
                raise ValueError("verdict_code must be INVALID_OUT_OF_ENVELOPE when envelope is breached")

        if self.status == CertificateStatus.PASS:
            if not self.claim_s_pass:
                raise ValueError("Certificate status cannot be PASS when claim_s_pass is False")
            if self.stability_summary is None:
                raise ValueError("Certificate status PASS requires non-null stability_summary")
            if not (
                self.stability_summary.passes_absolute_noninferiority
                and self.stability_summary.passes_comparator_worst_env
            ):
                raise ValueError("Certificate status PASS requires both absolute and comparator stability gates")

        if self.overall_study_pass != (self.claim_p_pass and self.claim_s_pass and not has_envelope_breach):
            raise ValueError("overall_study_pass must equal (claim_p_pass and claim_s_pass and valid_envelope)")

        expected_conclusion = ALLOWED_CONCLUSION_MAP[self.verdict_code]
        if self.allowed_conclusion != expected_conclusion:
            raise ValueError(
                f"allowed_conclusion mismatch for {self.verdict_code}: "
                f"expected {expected_conclusion!r}, got {self.allowed_conclusion!r}"
            )
        return self

    def certificate_sha256(self) -> str:
        return canonical_sha256(self)


def build_stability_certificate(
    *,
    certificate_id: str,
    study_id: str,
    algorithm_version: str,
    envelope: OperatingEnvelope,
    observed_environments: Tuple[str, ...],
    observed_model_versions: Dict[str, Dict[str, str]],
    claim_p_pass: bool,
    claim_p_practical_pass: bool,
    claim_s_statistical_pass: bool,
    missing_cells_count: int = 0,
    incomplete_pairs_count: int = 0,
    hash_mismatches: Tuple[str, ...] = (),
    extra_drift_events: Tuple[str, ...] = (),
    selection_contrasts: Optional[Dict[str, ComparatorContrastSummary]] = None,
    stability_summary: Optional[StabilitySummary] = None,
    supporting_diagnostics: Optional[SupportingDiagnostics] = None,
    schema_version: str = CURRENT_SCHEMA_VERSION,
) -> StabilityCertificate:
    """Construct a fail-closed StabilityCertificate from audited study state."""
    registered_envs = set(envelope.all_env_ids)
    observed_set = set(observed_environments)
    missing_envs = sorted(registered_envs - observed_set)
    unregistered_envs = sorted(observed_set - registered_envs)

    drift_events: List[str] = list(extra_drift_events)
    env_spec_map = {e.env_id: e for e in envelope.registered_environments}
    for env_id in sorted(observed_set & registered_envs):
        spec = env_spec_map[env_id]
        if spec.requires_identical_model_version:
            obs_models = observed_model_versions.get(env_id, {})
            for model_id, pinned_ver in sorted(envelope.model_versions_pinned.items()):
                obs_ver = obs_models.get(model_id)
                if obs_ver != pinned_ver:
                    msg = f"{env_id}:{model_id}:expected={pinned_ver},observed={obs_ver}"
                    if msg not in drift_events:
                        drift_events.append(msg)

    for m_env in missing_envs:
        msg = f"missing_environment:{m_env}"
        if msg not in drift_events:
            drift_events.append(msg)
    for u_env in unregistered_envs:
        msg = f"unregistered_environment:{u_env}"
        if msg not in drift_events:
            drift_events.append(msg)

    has_breach = bool(
        missing_envs
        or unregistered_envs
        or missing_cells_count > 0
        or incomplete_pairs_count > 0
        or len(hash_mismatches) > 0
        or len(drift_events) > 0
    )

    if has_breach:
        status = CertificateStatus.INVALID_OUT_OF_ENVELOPE
        verdict_code = OverallVerdictCode.INVALID_OUT_OF_ENVELOPE
        effective_claim_s = False
        overall_pass = False
    else:
        effective_claim_s = bool(claim_s_statistical_pass)
        status = CertificateStatus.PASS if effective_claim_s else CertificateStatus.FAIL
        overall_pass = bool(claim_p_pass and effective_claim_s)
        if claim_p_pass and effective_claim_s:
            verdict_code = OverallVerdictCode.PASS_SELECTION_AND_STABILITY
        elif claim_p_pass and not effective_claim_s:
            verdict_code = OverallVerdictCode.PASS_SELECTION_ONLY
        elif not claim_p_pass and effective_claim_s:
            verdict_code = OverallVerdictCode.PASS_STABILITY_ONLY
        else:
            verdict_code = OverallVerdictCode.FAIL_BOTH

    return StabilityCertificate(
        schema_version=schema_version,
        certificate_id=certificate_id,
        study_id=study_id,
        algorithm_version=algorithm_version,
        status=status,
        verdict_code=verdict_code,
        allowed_conclusion=ALLOWED_CONCLUSION_MAP[verdict_code],
        claim_p_pass=bool(claim_p_pass),
        claim_p_practical_pass=bool(claim_p_practical_pass),
        claim_s_pass=effective_claim_s,
        overall_study_pass=overall_pass,
        envelope=envelope,
        envelope_sha256=envelope.envelope_sha256(),
        observed_environments=tuple(observed_environments),
        observed_model_versions=observed_model_versions,
        missing_cells_count=missing_cells_count,
        incomplete_pairs_count=incomplete_pairs_count,
        hash_mismatches=tuple(hash_mismatches),
        drift_events=tuple(drift_events),
        selection_contrasts=selection_contrasts or {},
        stability_summary=stability_summary,
        supporting_diagnostics=supporting_diagnostics,
    )


def verify_certificate_for_request(
    certificate: StabilityCertificate,
    *,
    task_id: str,
    model_id: str,
    model_version: str,
    parser_hash: str,
    algorithm_version: str,
    env_id: str = "ref",
    code_hash: Optional[str] = None,
) -> EnvelopeVerificationResult:
    """Verify at deployment/inspection time whether a request falls inside the certified envelope."""
    reasons: List[str] = []
    env = certificate.envelope

    if certificate.status != CertificateStatus.PASS:
        reasons.append(f"certificate_not_passing:{certificate.status.value}")
    if algorithm_version != certificate.algorithm_version:
        reasons.append(
            f"algorithm_version_mismatch:expected={certificate.algorithm_version},got={algorithm_version}"
        )
    if task_id not in env.tasks:
        reasons.append(f"unregistered_task:{task_id}")
    else:
        expected_parser = env.parser_hashes.get(task_id)
        if parser_hash != expected_parser:
            reasons.append(f"parser_hash_mismatch:{task_id}:expected={expected_parser},got={parser_hash}")
    if model_id not in env.models:
        reasons.append(f"unregistered_model:{model_id}")
    else:
        pinned = env.model_versions_pinned.get(model_id)
        if model_version != pinned:
            reasons.append(f"model_version_drift:{model_id}:expected={pinned},got={model_version}")
    if env_id not in env.all_env_ids:
        reasons.append(f"unregistered_environment:{env_id}")
    if code_hash is not None and code_hash != env.code_hash:
        reasons.append(f"code_hash_mismatch:expected={env.code_hash},got={code_hash}")

    if reasons:
        is_out = any(not r.startswith("certificate_not_passing:") for r in reasons)
        eff_status = (
            CertificateStatus.INVALID_OUT_OF_ENVELOPE
            if is_out
            else certificate.status
        )
        return EnvelopeVerificationResult(
            is_within_envelope=False,
            effective_status=eff_status,
            reasons=tuple(reasons),
        )

    return EnvelopeVerificationResult(
        is_within_envelope=True,
        effective_status=CertificateStatus.PASS,
        reasons=(),
    )
