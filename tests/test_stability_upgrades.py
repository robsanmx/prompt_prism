"""
Offline unit tests for PromptPrism stability upgrades:
1. Native multi-turn chat message composition & PromptTemplate.structured_chat (XML / brackets)
2. Factor.demo & FactorSet.from_candidate_demos in-domain demo factor builder
3. Stability-aware clause attribution (`env_col` / `ref_env`) in `diagnose()`
4. Repeat-noise floor, per-env SE, and precision-vs-genuine diagnostic notes in `certify`
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from prompt_prism import (
    Factor,
    FactorSet,
    OperatingEnvelopeSpec,
    PromptComposer,
    PromptTemplate,
    RunConfig,
    evaluate_simultaneous_perturbation_certificate,
    run_calibrated_attribution_audit,
)


def test_multi_turn_compose_messages_preserves_alternating_demo_turns():
    """Consecutive same-role sections merge, while alternating user/assistant demo turns stay intact."""
    fset = FactorSet.from_candidate_demos(
        demos=[
            {"query": "Where is my card?", "target": "card_arrival"},
            {"query": "ATM kept my card", "target": "card_swallowed"},
        ],
        input_fields=["query"],
        output_key="target",
        field_style="xml",
        output_prefix="FINAL_ANSWER: ",
        multi_turn=True,
        extra_factors=[
            Factor.binary(
                "strict_contract",
                level_0_content="Answer briefly.",
                level_1_content="Return ONLY FINAL_ANSWER: <label>.",
            )
        ],
    )
    tmpl = PromptTemplate.structured_chat(
        fset,
        system_prompt="Classify banking queries.",
        input_fields=["query"],
        field_style="xml",
    )
    composer = PromptComposer(tmpl, fset)

    # All ON -> system (merged), user(demo1), assistant(demo1), user(demo2), assistant(demo2), user(query)
    rc_all = RunConfig(
        run_id=1,
        factor_levels={"A": 1, "B": 1, "C": 1},
        factor_names={"strict_contract": 1, "demo_1": 1, "demo_2": 1},
    )
    msgs = composer.compose_messages(rc_all, data={"query": "Reset my PIN"})
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "Classify banking queries." in msgs[0]["content"]
    assert "Return ONLY FINAL_ANSWER: <label>." in msgs[0]["content"]
    assert "<query>Where is my card?</query>" == msgs[1]["content"]
    assert "FINAL_ANSWER: card_arrival" == msgs[2]["content"]
    assert "<query>ATM kept my card</query>" == msgs[3]["content"]
    assert "FINAL_ANSWER: card_swallowed" == msgs[4]["content"]
    assert "<query>Reset my PIN</query>" == msgs[5]["content"]

    # Demo 1 OFF, Demo 2 ON -> 4 messages
    rc_half = RunConfig(
        run_id=2,
        factor_levels={"A": 0, "B": 0, "C": 1},
        factor_names={"strict_contract": 0, "demo_1": 0, "demo_2": 1},
    )
    msgs_half = composer.compose_messages(rc_half, data={"query": "Reset my PIN"})
    assert [m["role"] for m in msgs_half] == ["system", "user", "assistant", "user"]
    assert "ATM kept my card" in msgs_half[1]["content"]

    # compose_text renders multi-turn demos cleanly without dict reprs
    txt = composer.compose_text(rc_half, data={"query": "Reset my PIN"})
    assert "User: <query>ATM kept my card</query>" in txt
    assert "Assistant: FINAL_ANSWER: card_swallowed" in txt
    assert "{'role'" not in txt


def test_structured_chat_brackets_and_tuple_demos():
    fset = FactorSet.from_candidate_demos(
        demos=[("2 + 2?", "4")],
        field_style="brackets",
        multi_turn=True,
    )
    tmpl = PromptTemplate.structured_chat(
        fset,
        system_prompt="Solve math.",
        input_fields=["question"],
        field_style="brackets",
    )
    rc = RunConfig(run_id=0, factor_levels={"A": 1}, factor_names={"demo_1": 1})
    msgs = PromptComposer(tmpl, fset).compose_messages(rc, data={"question": "3 + 5?"})
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "[[ ## question ## ]]\n3 + 5?" == msgs[3]["content"]

    with pytest.raises(ValueError, match="field_style"):
        PromptTemplate.structured_chat(fset, field_style="invalid")


def test_stability_aware_diagnose_flags_stabilizing_and_fragile_clauses():
    """`env_col` isolates clean effects on ref and flags STABILIZING vs. FRAGILE clauses."""
    rng = np.random.default_rng(7)
    rows = []
    # Full 2^3 factorial across 60 items x 2 envs (ref, fmt_delimiter)
    # A: neutral on clean ref (+0.00), STABILIZING under perturbation (-0.12 drop!) -> must be KEEP, never SAFE_TO_PRUNE
    # B: slightly positive on clean ref (+0.05), FRAGILE under perturbation (+0.15 extra drop!) -> KEEP + FRAGILE warning
    # C: truly dead on both clean ref and perturbation -> SAFE_TO_PRUNE + NEUTRAL
    combos = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    item_bases = 0.80 + rng.normal(0.0, 0.05, size=60)
    for run_id, (a, b, c) in enumerate(combos):
        for i in range(60):
            base = float(item_bases[i])
            ref_score = base + 0.05 * b + rng.normal(0.0, 0.004)
            # Perturbation drop: 0.14 when A=0, 0.02 when A=1; +0.15 extra drop when B=1
            drop = (0.14 - 0.12 * a) + 0.15 * b + rng.normal(0.0, 0.008)
            pert_score = ref_score - drop
            for env_id, sc in (("ref", ref_score), ("fmt_delimiter", pert_score)):
                rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": f"item_{i:03d}",
                        "env_id": env_id,
                        "A": a,
                        "B": b,
                        "C": c,
                        "score": sc,
                    }
                )
    df = pd.DataFrame(rows)
    audit = run_calibrated_attribution_audit(
        data=df,
        factor_cols=["A", "B", "C"],
        target_col="score",
        equivalence_margin=0.02,
        n_sign_flip=0,
        env_col="env_id",
        ref_env="ref",
    )
    by_f = {f.factor_id: f for f in audit.factors}

    # A is clean-neutral, but STABILIZING prevents false pruning!
    assert abs(by_f["A"].effect_delta) < 0.01
    assert by_f["A"].stability_drop_delta is not None
    assert by_f["A"].stability_drop_delta < -0.10
    assert by_f["A"].stability_verdict == "STABILIZING"
    assert by_f["A"].pruning_verdict == "KEEP"

    # B helps clean accuracy but is flagged FRAGILE with a warning
    assert by_f["B"].effect_delta > 0.04
    assert by_f["B"].stability_drop_delta is not None
    assert by_f["B"].stability_drop_delta > 0.10
    assert by_f["B"].stability_verdict == "FRAGILE"
    assert any("FRAGILE" in w and "'B'" in w for w in audit.warnings)

    # C is dead on both clean and perturbation -> SAFE_TO_PRUNE + NEUTRAL
    assert by_f["C"].stability_verdict == "NEUTRAL"
    assert by_f["C"].pruning_verdict == "SAFE_TO_PRUNE"

    md = audit.to_markdown()
    assert "Δ Drop" in md
    assert "`STABILIZING`" in md
    assert "`FRAGILE`" in md


def test_certificate_distinguishes_precision_limited_vs_genuine_drop():
    """Certificate reports worst_env_id, per_env_se, repeat_noise_se, recommended_min_n, and diagnostic_note."""
    env = OperatingEnvelopeSpec(
        task_id="test_task",
        model_id="test-model",
        model_version="v1",
        dataset_partition_hash="h1",
        parser_hash="p1",
        perturbation_env_ids=["repeat_r2", "fmt_delimiter"],
    )
    # Case 1: Genuine 5 pp drop -> FAIL with genuine-drop note
    ref = [1.0] * 100
    rep = [1.0] * 100
    delim_bad = [1.0] * 95 + [0.0] * 5
    cert_bad = evaluate_simultaneous_perturbation_certificate(
        envelope=env,
        reference_scores=ref,
        perturbation_scores_by_env={"repeat_r2": rep, "fmt_delimiter": delim_bad},
        epsilon_stability=0.02,
        n_bootstrap=200,
        seed=1,
    )
    assert cert_bad.status == "FAIL"
    assert cert_bad.worst_env_id == "fmt_delimiter"
    assert cert_bad.repeat_noise_se == pytest.approx(0.0)
    assert cert_bad.recommended_min_n is None
    assert cert_bad.diagnostic_note is not None
    assert "Genuine perturbation drop" in cert_bad.diagnostic_note

    # Case 2: Precision-limited FAIL (tiny 0.8 pp drop, noisy SE -> U95_sim > 0.02)
    rng = np.random.default_rng(123)
    ref2 = (rng.random(120) < 0.90).astype(float)
    # Flip 2 down and 1 up -> net drop ~0.83 pp, but SE > 0.012
    rep2 = ref2.copy()
    rep2[0] = 0.0 if ref2[0] == 1.0 else 1.0
    delim_noisy = ref2.copy()
    ones = np.where(ref2 == 1.0)[0]
    zeros = np.where(ref2 == 0.0)[0]
    delim_noisy[ones[:4]] = 0.0
    delim_noisy[zeros[:3]] = 1.0
    cert_prec = evaluate_simultaneous_perturbation_certificate(
        envelope=env,
        reference_scores=ref2,
        perturbation_scores_by_env={"repeat_r2": rep2, "fmt_delimiter": delim_noisy},
        epsilon_stability=0.02,
        n_bootstrap=300,
        seed=2,
    )
    assert cert_prec.status == "FAIL"
    assert cert_prec.d_stability_estimate is not None
    assert cert_prec.d_stability_estimate < 0.02
    assert cert_prec.repeat_noise_se is not None and cert_prec.repeat_noise_se > 0.0
    assert cert_prec.recommended_min_n is not None and cert_prec.recommended_min_n > 120
    assert cert_prec.diagnostic_note is not None
    assert "Precision-limited FAIL" in cert_prec.diagnostic_note


def test_cli_diagnose_env_col_and_certify_diagnostics(tmp_path):
    """CLI `diagnose --env-col` and `certify` emit stability attribution & diagnostic notes."""
    rng = np.random.default_rng(19)
    diag_rows = []
    item_bases = 0.85 + rng.normal(0.0, 0.04, size=40)
    for run_id, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        for i in range(40):
            ref_s = float(item_bases[i] + rng.normal(0.0, 0.003))
            drop = (0.12 - 0.10 * a) + rng.normal(0.0, 0.005)
            for env_id, sc in (("ref", ref_s), ("fmt_delimiter", ref_s - drop)):
                diag_rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": f"s_{i:03d}",
                        "env_id": env_id,
                        "A": a,
                        "B": b,
                        "score": sc,
                    }
                )
    diag_csv = tmp_path / "diag.csv"
    pd.DataFrame(diag_rows).to_csv(diag_csv, index=False)

    res_diag = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "diagnose",
            "--data",
            str(diag_csv),
            "--target",
            "score",
            "--factors",
            "A",
            "B",
            "--env-col",
            "env_id",
            "--ref-env",
            "ref",
        ],
        capture_output=True,
        text=True,
    )
    assert res_diag.returncode == 0, res_diag.stderr
    assert "Δ Drop" in res_diag.stdout
    assert "`STABILIZING`" in res_diag.stdout

    cert_rows = []
    for i in range(60):
        cert_rows.append({"sample_id": f"c_{i:03d}", "env_id": "ref", "score": 1.0})
        cert_rows.append(
            {"sample_id": f"c_{i:03d}", "env_id": "repeat_r2", "score": 1.0}
        )
        cert_rows.append(
            {
                "sample_id": f"c_{i:03d}",
                "env_id": "fmt_delimiter",
                "score": 0.0 if i < 3 else 1.0,
            }
        )
    cert_csv = tmp_path / "cert.csv"
    pd.DataFrame(cert_rows).to_csv(cert_csv, index=False)

    res_cert = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "certify",
            "--data",
            str(cert_csv),
            "--target",
            "score",
            "--task-id",
            "demo_task",
            "--model-id",
            "demo-model",
            "--model-version",
            "v1",
            "--dataset-hash",
            "h1",
            "--parser-hash",
            "p1",
            "--bootstrap",
            "200",
        ],
        capture_output=True,
        text=True,
    )
    assert res_cert.returncode == 0, res_cert.stderr
    assert "[fmt_delimiter]" in res_cert.stdout
    assert "Repeat Noise SE:" in res_cert.stdout
    assert "Genuine perturbation drop" in res_cert.stdout
