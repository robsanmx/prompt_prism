"""
Tests for conditional public API selection reports and fail-closed stability envelope certificates (Task 17).
"""

from __future__ import annotations

import json

import pytest

from prompt_prism.analysis.selection import (
    PROMOTED_TO_DEFAULT_PUBLIC_API,
    REAL_V4_REGISTERED_VERDICT,
    ExperimentalTwoStageSelector,
    build_selection_report_from_recommendation,
    evaluate_simultaneous_perturbation_certificate,
)
from prompt_prism.core.factors import Factor, Level
from prompt_prism.core.models import (
    OperatingEnvelopeSpec,
    PromptEvidenceState,
    PromptSelectionReport,
    PromptStabilityCertificate,
)
from prompt_prism.evaluation.metrics import ExactMatch
from prompt_prism.experiment import Experiment


def _sample_envelope(**overrides) -> OperatingEnvelopeSpec:
    base = {
        "task_id": "gsm8k_math",
        "model_id": "gemini-2.5-flash",
        "model_version": "gemini-2.5-flash",
        "dataset_partition_hash": "sha256_part_001",
        "parser_hash": "sha256_parser_001",
        "perturbation_env_ids": ["fmt_whitespace", "word_paraphrase_1"],
        "algorithm_version": "promptprism_experimental_two_stage_v4.0.0",
        "decoding_config": {"temperature": 0.0, "max_output_tokens": 512, "seed": 42},
    }
    base.update(overrides)
    return OperatingEnvelopeSpec(**base)


def _build_small_experiment() -> Experiment:
    factors = [
        Factor(
            id="A",
            name="Role",
            levels=[
                Level(0, "Neutral", ""),
                Level(1, "Expert", "You are a math expert."),
            ],
        ),
        Factor(
            id="B",
            name="Contract",
            levels=[
                Level(0, "Plain", "Answer:"),
                Level(1, "Strict", "Return exact answer:"),
            ],
        ),
        Factor(
            id="C",
            name="Guardrail",
            levels=[Level(0, "None", ""), Level(1, "Check", "Verify arithmetic.")],
        ),
    ]
    return Experiment.from_factors(
        factors=factors,
        data_template="Question: {question}",
        metrics=[ExactMatch()],
        target_metric="exact_match",
    )


def test_promotion_gate_status_is_experimental_opt_in_only():
    assert PROMOTED_TO_DEFAULT_PUBLIC_API is False
    assert REAL_V4_REGISTERED_VERDICT == "FAIL_BOTH"
    with pytest.raises(ValueError, match="experimental_opt_in=True"):
        ExperimentalTwoStageSelector(experimental_opt_in=False)


def test_existing_strategies_remain_backward_compatible_and_default_to_predicted():
    exp = _build_small_experiment()
    dataset = [
        {"id": f"s{i}", "question": f"What is {i}+{i}?", "target": "42"}
        for i in range(12)
    ]

    def mock_llm(prompt: str) -> str:
        return "42" if "Return exact answer:" in prompt else "wrong"

    exp.run(dataset=dataset, client=mock_llm, max_workers=1)

    for strategy in ("additive", "screen_best", "interaction_2fi"):
        rep = exp.analyze(
            strategy=strategy,
            include_interactions=(strategy == "interaction_2fi"),
        )
        assert rep.optimal_recommendation is not None
        assert rep.optimal_recommendation.strategy == strategy

        sel_rep = exp.selection_report(report=rep, strategy=strategy)
        assert sel_rep.evidence_state == PromptEvidenceState.PREDICTED
        assert sel_rep.experimental is False
        assert sel_rep.can_display_passing_stability_badge is False
        assert "STABILITY-CERTIFIED" not in sel_rep.to_markdown()
        assert "`predicted`" in sel_rep.to_markdown()


def test_evidence_states_distinguish_predicted_confirmation_validated_and_stability_certified():
    exp = _build_small_experiment()
    dataset = [{"id": f"s{i}", "question": "Q", "target": "42"} for i in range(10)]
    exp.run(
        dataset=dataset,
        client=lambda p: "42" if "Return exact answer:" in p else "0",
        max_workers=1,
    )
    rep = exp.analyze(strategy="screen_best")
    assert rep.optimal_recommendation is not None

    env = _sample_envelope()

    # 1. Predicted only (no confirmation split)
    r_pred = build_selection_report_from_recommendation(rep.optimal_recommendation)
    assert r_pred.evidence_state == "predicted"
    assert r_pred.can_display_passing_stability_badge is False

    # 2. Confirmation-validated (disjoint confirmation scores + envelope, no stability certificate)
    r_conf = build_selection_report_from_recommendation(
        rep.optimal_recommendation,
        confirmation_scores=[1.0, 1.0, 0.9, 1.0],
        confirmation_envelope=env,
    )
    assert r_conf.evidence_state == "confirmation-validated"
    assert r_conf.confirmation_score == pytest.approx(0.975)
    assert r_conf.can_display_passing_stability_badge is False

    # 3. Stability-certified within matching operating envelope
    cert_pass = evaluate_simultaneous_perturbation_certificate(
        envelope=env,
        reference_scores=[0.90] * 40,
        perturbation_scores_by_env={
            "fmt_whitespace": [0.90] * 40,
            "word_paraphrase_1": [0.895] * 40,
        },
        epsilon_stability=0.02,
        n_bootstrap=200,
    )
    assert cert_pass.status == "PASS"
    assert cert_pass.is_passing_within_envelope is True

    r_cert = build_selection_report_from_recommendation(
        rep.optimal_recommendation,
        confirmation_scores=[0.90] * 40,
        confirmation_envelope=env,
        stability_certificate=cert_pass,
        request_envelope=env,
    )
    assert r_cert.evidence_state == "stability-certified"
    assert r_cert.can_display_passing_stability_badge is True
    assert "STABILITY-CERTIFIED WITHIN ENVELOPE" in r_cert.to_markdown()

    # JSON round-trip serialization check
    restored = PromptSelectionReport.model_validate(
        json.loads(r_cert.model_dump_json())
    )
    assert restored.evidence_state == "stability-certified"
    assert restored.certificate is not None
    assert restored.certificate.certificate_sha256() == cert_pass.certificate_sha256()


@pytest.mark.parametrize(
    "override_field,override_value",
    [
        ("model_id", "other-model"),
        ("model_version", "gemini-2.5-flash-drifted"),
        ("dataset_partition_hash", "tampered_data_hash"),
        ("parser_hash", "tampered_parser_hash"),
        ("perturbation_env_ids", ["fmt_whitespace"]),
        ("algorithm_version", "promptprism_v9.9.9"),
        ("decoding_config", {"temperature": 0.7, "max_output_tokens": 512, "seed": 42}),
    ],
)
def test_out_of_envelope_requests_refuse_validation_and_cannot_display_passing_badge(
    override_field, override_value
):
    env = _sample_envelope()
    cert = PromptStabilityCertificate(
        envelope=env,
        status="PASS",
        reference_mean=0.88,
        d_stability_estimate=0.002,
        d_stability_upper_95_sim=0.006,
        epsilon_stability=0.02,
    )
    assert cert.is_passing_within_envelope is True

    bad_req = _sample_envelope(**{override_field: override_value})
    val = cert.validate_request(bad_req, has_confirmation_validation=True, strict=False)
    assert val.valid is False
    assert val.can_display_passing_stability_badge is False
    assert val.evidence_state != "stability-certified"
    assert "OUT-OF-ENVELOPE" in val.badge_label

    with pytest.raises(ValueError, match="out of envelope"):
        cert.validate_request(bad_req, strict=True)


def test_experimental_two_stage_end_to_end_and_missing_env_fail_closed():
    exp = _build_small_experiment()
    dataset = [{"id": f"s{i}", "question": "Q", "target": "42"} for i in range(8)]
    exp.run(
        dataset=dataset,
        client=lambda p: "42" if "Return exact answer:" in p else "0",
        max_workers=1,
    )

    with pytest.raises(ValueError, match="experimental_opt_in=True"):
        exp.experimental_select_two_stage(
            validation_scores_by_env={}, experimental_opt_in=False
        )

    # Populate validation scores for all possible candidate run_ids (screen runs + synthetic >=1000)
    selector = ExperimentalTwoStageSelector(
        experimental_opt_in=True, max_candidates=4, n_bootstrap=80
    )
    df = exp.last_results.to_dataframe()
    run_levels = {
        int(r.run_id): dict(r.factor_levels) for r in exp.last_results.design.runs
    }
    scr_scores = {
        int(rid): grp["exact_match"].tolist() for rid, grp in df.groupby("run_id")
    }
    cands = selector.generate_candidates_from_screen(
        factor_ids=exp.factors.ids,
        run_factor_levels=run_levels,
        screen_scores_by_run=scr_scores,
    )

    val_scores = {
        env_id: {
            rid: ([0.95] * 10 if lvls["B"] == 1 else [0.20] * 10)
            for rid, lvls, _, _ in cands
        }
        for env_id in ("ref", "fmt_whitespace", "word_paraphrase_1")
    }
    env = _sample_envelope()

    # Missing perturbation environment in certificate evaluation fails closed
    bad_cert = evaluate_simultaneous_perturbation_certificate(
        envelope=env,
        reference_scores=[0.95] * 20,
        perturbation_scores_by_env={
            "fmt_whitespace": [0.95] * 20
        },  # missing word_paraphrase_1
        n_bootstrap=100,
    )
    assert bad_cert.status == "INVALID/OUT_OF_ENVELOPE"
    assert bad_cert.is_passing_within_envelope is False

    rep = exp.experimental_select_two_stage(
        validation_scores_by_env=val_scores,
        experimental_opt_in=True,
        confirmation_scores=[0.94] * 20,
        confirmation_envelope=env,
        stability_certificate=bad_cert,
        request_envelope=env,
    )
    assert rep.experimental is True
    assert rep.selected_factor_levels["B"] == 1
    assert rep.can_display_passing_stability_badge is False
    assert "STABILITY-CERTIFIED WITHIN ENVELOPE" not in rep.to_markdown()


def test_experiment_diagnose_auto_estimates_tokens_and_reports_noise_floor():
    exp = _build_small_experiment()
    dataset = [{"id": f"s{i}", "question": "Q", "target": "42"} for i in range(16)]
    exp.run(
        dataset=dataset,
        client=lambda p: "42" if "Return exact answer:" in p else "0",
        max_workers=1,
    )

    audit = exp.diagnose(n_sign_flip=200)
    assert audit.n_items == 16
    assert audit.noise_floor.small_sample_warning is True
    assert any(
        f.factor_id == "B" and f.effect_delta == pytest.approx(1.0)
        for f in audit.factors
    )
    assert "PromptPrism 2.0 Calibrated Attribution" in audit.to_markdown()
