"""Unit, golden-hash, protocol-lint, and fail-closed tests for Real-v4 schemas and certificates."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.real_v4.certificate import (
    StabilityCertificate,
    build_stability_certificate,
    verify_certificate_for_request,
)
from research.real_v4.schemas import (
    ALLOWED_CONCLUSION_MAP,
    CandidateProvenance,
    CertificateStatus,
    ComparatorContrastSummary,
    EnvironmentFamily,
    EnvironmentSpec,
    IncompatibleSchemaVersionError,
    LockedSelectionRecord,
    MethodBudgetSpec,
    OperatingEnvelope,
    OverallVerdictCode,
    ProtocolLintError,
    StabilitySummary,
    SupportingDiagnostics,
    canonical_json,
    lint_protocol_documents,
    validate_schema_version,
)

ROOT = Path(__file__).resolve().parents[2]
REAL_V4_PROTOCOL_DIR = ROOT / "research" / "protocol" / "real_v4"


def make_sample_envelope(
    *,
    schema_version: str = "1.0.0",
    nondeterminism_detected: bool = True,
    repeat_count: int = 3,
) -> OperatingEnvelope:
    tasks = (
        "re3d_entity_extraction",
        "squad2_grounded_qa",
        "banking77_intent",
        "gsm8k_math",
        "mbpp_python",
        "sciq_verification",
    )
    models = (
        "gemini-2.5-flash",
        "claude-3-5-haiku-20241022",
        "mlx-community/Qwen3-8B-4bit",
    )
    envs = (
        EnvironmentSpec(
            schema_version=schema_version,
            env_id="ref",
            family=EnvironmentFamily.REFERENCE,
            description="Clean reference confirmation",
            parent_env_id=None,
            transformation_hash="00000000ref00000",
        ),
        EnvironmentSpec(
            schema_version=schema_version,
            env_id="repeat_r2",
            family=EnvironmentFamily.REPEAT,
            description="Identical repeat 2",
            parent_env_id="ref",
            transformation_hash="11111111rep20000",
        ),
        EnvironmentSpec(
            schema_version=schema_version,
            env_id="fmt_whitespace",
            family=EnvironmentFamily.FORMATTING,
            description="Double newline section spacing",
            parent_env_id="ref",
            transformation_hash="22222222fmtws000",
        ),
        EnvironmentSpec(
            schema_version=schema_version,
            env_id="word_paraphrase_1",
            family=EnvironmentFamily.WORDING,
            description="Audited instruction paraphrase 1",
            parent_env_id="ref",
            transformation_hash="33333333word1000",
        ),
        EnvironmentSpec(
            schema_version=schema_version,
            env_id="item_holdout_group",
            family=EnvironmentFamily.ITEM_GROUP,
            description="Group-disjoint confirmation holdout",
            parent_env_id="ref",
            transformation_hash="44444444itemg000",
            uses_disjoint_item_group=True,
        ),
        EnvironmentSpec(
            schema_version=schema_version,
            env_id="temporal_wave_2",
            family=EnvironmentFamily.TEMPORAL,
            description="Second time wave",
            parent_env_id="ref",
            transformation_hash="55555555wave2000",
        ),
    )
    return OperatingEnvelope(
        schema_version=schema_version,
        study_id="prompt-prism-real-v4-selection-stability-2026",
        tasks=tasks,
        models=models,
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
        parser_hashes={t: f"parser_sha_{t[:8]}_v1" for t in tasks},
        decoding_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "seed": 42,
        },
        partition_hashes={t: f"part_sha_{t[:8]}_v1" for t in tasks},
        registered_environments=envs,
        repeat_count=repeat_count,
        nondeterminism_detected=nondeterminism_detected,
        local_runtime_locks={"mlx_lm": "0.24.0", "arch": "arm64"},
        code_hash="abcdef0123456789abcdef0123456789",
    )


def test_real_v4_protocol_documents_pass_lint():
    hashes = lint_protocol_documents(REAL_V4_PROTOCOL_DIR)
    assert set(hashes.keys()) == {
        "study_plan.md",
        "preregistration.md",
        "statistical_analysis_plan.md",
    }
    assert all(len(digest) == 64 for digest in hashes.values())


def test_protocol_linter_rejects_placeholders_and_missing_fields(tmp_path: Path):
    for name in ("study_plan.md", "preregistration.md", "statistical_analysis_plan.md"):
        (tmp_path / name).write_text(
            (REAL_V4_PROTOCOL_DIR / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    assert lint_protocol_documents(tmp_path)

    # Injecting TODO triggers ProtocolLintError
    (tmp_path / "study_plan.md").write_text(
        (tmp_path / "study_plan.md").read_text(encoding="utf-8")
        + "\nTODO: tune threshold later\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolLintError, match="placeholder"):
        lint_protocol_documents(tmp_path)

    # Removing required contract token triggers ProtocolLintError
    clean_prereg = (REAL_V4_PROTOCOL_DIR / "preregistration.md").read_text(
        encoding="utf-8"
    )
    (tmp_path / "study_plan.md").write_text(
        (REAL_V4_PROTOCOL_DIR / "study_plan.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "preregistration.md").write_text(
        clean_prereg.replace("bayesian_opt", "other_method"),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolLintError, match="bayesian_opt"):
        lint_protocol_documents(tmp_path)


def test_schema_version_forward_compatibility_and_major_rejection():
    assert validate_schema_version("1.0.0") == (1, 0, 0)
    assert validate_schema_version("1.4.2") == (1, 4, 2)
    env_minor = make_sample_envelope(schema_version="1.2.0")
    assert env_minor.schema_version == "1.2.0"

    for bad_ver in ("2.0.0", "0.9.1", "v1.0.0", "1.0", "unknown"):
        with pytest.raises(IncompatibleSchemaVersionError):
            validate_schema_version(bad_ver)
        with pytest.raises(IncompatibleSchemaVersionError):
            make_sample_envelope(schema_version=bad_ver)


def test_envelope_and_locked_selection_round_trip_and_golden_hash():
    env1 = make_sample_envelope()
    env2 = OperatingEnvelope.model_validate_json(env1.canonical_json())
    assert env1 == env2
    assert env1.envelope_sha256() == env2.envelope_sha256()
    assert canonical_json(env1) == canonical_json(env2)

    # Nondeterminism requires at least 3 repeats
    with pytest.raises(ValueError, match="repeat_count must be >= 3"):
        make_sample_envelope(nondeterminism_detected=True, repeat_count=2)

    cand0 = CandidateProvenance(
        run_id=5,
        factor_levels={"A": 0, "B": 0, "C": 0, "D": 1, "E": 0, "F": 1},
        source_rule="sign_composed",
        screen_score=0.75,
        rank_in_screen=1,
    )
    cand1 = CandidateProvenance(
        run_id=12,
        factor_levels={"A": 0, "B": 0, "C": 1, "D": 1, "E": 0, "F": 0},
        source_rule="screen_best",
        screen_score=0.74,
        rank_in_screen=2,
    )
    budget = MethodBudgetSpec(
        method_id="promptprism_robust_v4",
        cell_id="gsm8k_math::gemini-2.5-flash",
        screen_configs=16,
        screen_items=120,
        screen_repeats=1,
        validation_candidates=2,
        validation_items=60,
        validation_envs=3,
        validation_repeats=1,
        oracle_audit_calls=5760,
    )
    assert budget.total_screen_calls == 1920
    assert budget.total_validation_calls == 360
    assert budget.total_operational_discovery_calls == 2280

    rec = LockedSelectionRecord(
        study_id=env1.study_id,
        cell_id="gsm8k_math::gemini-2.5-flash",
        method_id="promptprism_robust_v4",
        trajectory_id="primary",
        algorithm_version="v4.0.0",
        candidates=(cand0, cand1),
        selected_run_id=5,
        selected_factor_levels=cand0.factor_levels,
        validation_robust_score=0.71,
        validation_worst_env_score=0.73,
        validation_env_scores={
            "ref": 0.76,
            "fmt_whitespace": 0.74,
            "word_paraphrase_1": 0.73,
        },
        budget=budget,
        data_hash="1234567890abcdef",
        model_hash="fedcba0987654321",
        screen_manifest_hash="aaaa1111bbbb2222",
        validation_manifest_hash="cccc3333dddd4444",
    )
    rec_rt = LockedSelectionRecord.model_validate_json(rec.canonical_json())
    assert rec_rt.canonical_sha256() == rec.canonical_sha256()

    # Selecting a run_id not in candidates is rejected
    with pytest.raises(ValueError, match="must belong to locked candidates"):
        LockedSelectionRecord(
            **{
                **rec.model_dump(),
                "selected_run_id": 63,
            }
        )


def test_certificate_fail_closed_states_and_runtime_request_verification():
    env = make_sample_envelope()
    obs_versions = {e_id: dict(env.model_versions_pinned) for e_id in env.all_env_ids}

    comp_ok = ComparatorContrastSummary(
        comparator_id="balanced_random",
        estimate=0.022,
        se=0.004,
        ci95_lower=0.015,
        p_value_one_sided=0.0005,
        practical_margin=0.01,
        passes_statistical_gate=True,
        passes_practical_gate=True,
    )
    comp_bo = ComparatorContrastSummary(
        comparator_id="bayesian_opt",
        estimate=0.018,
        se=0.004,
        ci95_lower=0.011,
        p_value_one_sided=0.001,
        practical_margin=0.01,
        passes_statistical_gate=True,
        passes_practical_gate=True,
    )
    stab_ok = StabilitySummary(
        reference_mean=0.78,
        d_stability_estimate=0.003,
        d_stability_upper_95_sim=0.007,
        epsilon_stability=0.01,
        worst_perturbation_env_id="word_paraphrase_1",
        per_env_means={e_id: 0.778 for e_id in env.all_env_ids},
        per_env_drops={e_id: 0.002 for e_id in env.perturbation_env_ids},
        passes_absolute_noninferiority=True,
        comparator_worst_env_checks={
            "balanced_random": comp_ok,
            "bayesian_opt": comp_bo,
        },
        passes_comparator_worst_env=True,
    )

    cert_pass = build_stability_certificate(
        certificate_id="cert-v4-001",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        observed_environments=env.all_env_ids,
        observed_model_versions=obs_versions,
        claim_p_pass=True,
        claim_p_practical_pass=True,
        claim_s_statistical_pass=True,
        selection_contrasts={"balanced_random": comp_ok, "bayesian_opt": comp_bo},
        stability_summary=stab_ok,
        supporting_diagnostics=SupportingDiagnostics(effect_recovery_gain=0.005),
    )
    assert cert_pass.status == CertificateStatus.PASS
    assert cert_pass.overall_study_pass is True
    assert (
        StabilityCertificate.model_validate_json(
            cert_pass.canonical_json()
        ).certificate_sha256()
        == cert_pass.certificate_sha256()
    )

    # Missing environment forces INVALID/OUT_OF_ENVELOPE
    cert_missing = build_stability_certificate(
        certificate_id="cert-v4-missing",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        observed_environments=env.all_env_ids[:-1],
        observed_model_versions=obs_versions,
        claim_p_pass=True,
        claim_p_practical_pass=True,
        claim_s_statistical_pass=True,
        stability_summary=stab_ok,
    )
    assert cert_missing.status == CertificateStatus.INVALID_OUT_OF_ENVELOPE
    assert cert_missing.claim_s_pass is False
    assert cert_missing.overall_study_pass is False

    # Provider model version drift in temporal_wave_2 forces INVALID/OUT_OF_ENVELOPE
    drifted_versions = {k: dict(v) for k, v in obs_versions.items()}
    drifted_versions["temporal_wave_2"][
        "gemini-2.5-flash"
    ] = "gemini-2.5-flash-002-silent-update"
    cert_drift = build_stability_certificate(
        certificate_id="cert-v4-drift",
        study_id=env.study_id,
        algorithm_version="v4.0.0",
        envelope=env,
        observed_environments=env.all_env_ids,
        observed_model_versions=drifted_versions,
        claim_p_pass=True,
        claim_p_practical_pass=True,
        claim_s_statistical_pass=True,
        stability_summary=stab_ok,
    )
    assert cert_drift.status == CertificateStatus.INVALID_OUT_OF_ENVELOPE
    assert any("gemini-2.5-flash" in ev for ev in cert_drift.drift_events)

    # Direct tampering to force status='PASS' on drifted versions raises ValueError
    with pytest.raises(ValueError, match="INVALID/OUT_OF_ENVELOPE"):
        StabilityCertificate(
            certificate_id="tampered",
            study_id=env.study_id,
            algorithm_version="v4.0.0",
            status=CertificateStatus.PASS,
            verdict_code=OverallVerdictCode.PASS_SELECTION_AND_STABILITY,
            allowed_conclusion=ALLOWED_CONCLUSION_MAP[
                OverallVerdictCode.PASS_SELECTION_AND_STABILITY
            ],
            claim_p_pass=True,
            claim_p_practical_pass=True,
            claim_s_pass=True,
            overall_study_pass=True,
            envelope=env,
            envelope_sha256=env.envelope_sha256(),
            observed_environments=env.all_env_ids,
            observed_model_versions=drifted_versions,
            missing_cells_count=0,
            incomplete_pairs_count=0,
            stability_summary=stab_ok,
        )

    # Runtime envelope verification passes on exact match and fails closed on any drift
    ok_check = verify_certificate_for_request(
        cert_pass,
        task_id="gsm8k_math",
        model_id="gemini-2.5-flash",
        model_version="gemini-2.5-flash-001",
        parser_hash=env.parser_hashes["gsm8k_math"],
        algorithm_version="v4.0.0",
        env_id="ref",
    )
    assert ok_check.is_within_envelope is True
    assert ok_check.effective_status == CertificateStatus.PASS

    bad_ver_check = verify_certificate_for_request(
        cert_pass,
        task_id="gsm8k_math",
        model_id="gemini-2.5-flash",
        model_version="gemini-2.5-flash-999",
        parser_hash=env.parser_hashes["gsm8k_math"],
        algorithm_version="v4.0.0",
        env_id="ref",
    )
    assert bad_ver_check.is_within_envelope is False
    assert bad_ver_check.effective_status == CertificateStatus.INVALID_OUT_OF_ENVELOPE
