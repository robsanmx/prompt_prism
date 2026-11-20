"""
Unit tests for CLI commands.
"""

import subprocess
import sys


def test_cli_list_designs():
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "list-designs",
            "--factors",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "2(5-1)V" in res.stdout
    assert "2(5-2)III" in res.stdout


def test_cli_generate_design():
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "design",
            "--factors",
            "5",
            "--runs",
            "16",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "2(5-1)V" in res.stdout
    assert "combination" in res.stdout


def test_cli_alias():
    res = subprocess.run(
        [sys.executable, "-m", "prompt_prism.cli.main", "alias", "--plan", "2(5-1)V"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Plan: 2(5-1)V" in res.stdout
    assert "Defining Relation:" in res.stdout

    # Unknown plan exits non-zero
    res_err = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "alias",
            "--plan",
            "unknown_plan_xyz",
        ],
        capture_output=True,
        text=True,
    )
    assert res_err.returncode != 0
    assert (
        "not found in catalog" in res_err.stdout
        or "not found in catalog" in res_err.stderr
    )


def test_cli_list_metrics():
    res = subprocess.run(
        [sys.executable, "-m", "prompt_prism.cli.main", "list-metrics"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "exact_match" in res.stdout
    assert "faithfulness" in res.stdout
    assert "hallucination" in res.stdout

    # R32: the printed direction must follow DEEPEVAL_METRIC_SPECS, not a hardcoded string.
    # Every supported deepeval (>=4.2.0) metric is higher-is-better, so no line may claim
    # otherwise - this is what caught toxicity/bias/hallucination being described backwards.
    from prompt_prism.evaluation.deepeval_metrics import DEEPEVAL_METRIC_SPECS

    for kind, spec in DEEPEVAL_METRIC_SPECS.items():
        line = next(
            (
                ln
                for ln in res.stdout.splitlines()
                if ln.strip().startswith(f"• {kind} ")
            ),
            None,
        )
        if line is None:
            continue
        if spec["higher_is_better"]:
            assert (
                "lower is better" not in line.lower()
            ), f"{kind} is higher_is_better but list-metrics says lower is better: {line!r}"
        else:
            assert (
                "lower is better" in line.lower()
            ), f"{kind} is lower-is-better but list-metrics does not say so: {line!r}"


def _write_toxicity_csv(path):
    """A dataset where f1 strongly *raises* the metric and f2 is inert.

    Fixed seed so the significance verdicts are stable; the jitter keeps the residual
    variance non-zero so the F-test is well defined under sample_id blocking.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for run_id, (f1, f2) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)] * 2):
        for sample_id in range(6):
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "f1": f1,
                    "f2": f2,
                    "toxicity": 0.1 + 0.6 * f1 + float(rng.normal(0, 0.02)),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _effects_table_row(report_text, factor_id):
    """The report's main-effects table row for a factor, as rendered markdown."""
    prefix = f"| **{factor_id}**"
    rows = [ln for ln in report_text.splitlines() if ln.startswith(prefix)]
    assert len(rows) == 1, f"expected one {prefix} row in the report, found {len(rows)}"
    return rows[0]


def _section(stdout, heading):
    """The body of one `####` section of the recommendation markdown."""
    assert heading in stdout, f"no {heading!r} section in:\n{stdout}"
    return stdout.split(heading, 1)[1].split("####", 1)[0]


def test_cli_analyze_minimize_report_agrees_with_recommendation(tmp_path):
    """`analyze --minimize` must apply the direction to the effects, not just the optimizer.

    Regression: run_anova was called without `maximize`, so every FactorEffect kept the
    maximize-direction action_recommendation that the report table and Pareto chart
    render, while the optimizer flipped. The same report then called f1 harmful in its
    recommendation and told the reader to ENABLE it one table down.
    """
    data_p = tmp_path / "tox.csv"
    report_p = tmp_path / "report.md"
    _write_toxicity_csv(data_p)

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "analyze",
            "--data",
            str(data_p),
            "--target",
            "toxicity",
            "--minimize",
            "--output-report",
            str(report_p),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr

    # The recommendation calls the toxicity-raising factor harmful ...
    assert "**f1**" in _section(res.stdout, "Harmful Factors")

    # ... and the report's main-effects table must not contradict it.
    f1_row = _effects_table_row(report_p.read_text(), "f1")
    assert (
        "DISABLE" in f1_row
    ), f"report table contradicts the recommendation: {f1_row!r}"
    assert "ENABLE" not in f1_row


def test_cli_analyze_maximize_is_unaffected(tmp_path):
    """The default (maximize) direction keeps calling a score-raising factor a booster."""
    data_p = tmp_path / "score.csv"
    report_p = tmp_path / "report.md"
    _write_toxicity_csv(data_p)

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "analyze",
            "--data",
            str(data_p),
            "--target",
            "toxicity",
            "--output-report",
            str(report_p),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr

    assert "**f1**" in _section(res.stdout, "Significant Boosters")

    f1_row = _effects_table_row(report_p.read_text(), "f1")
    assert "ENABLE" in f1_row


def test_cli_version_reports_2_0():
    res = subprocess.run(
        [sys.executable, "-m", "prompt_prism.cli.main", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "2.0.0" in res.stdout


def test_cli_diagnose_calibrated_attribution_and_token_pruning(tmp_path):
    import json

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    rows = []
    # 120 paired items (standard N >= 100): A=+0.25 booster, B=-0.15 harmful, C=0.000 bloat (SAFE_TO_PRUNE)
    for run_id, (a, b, c) in enumerate(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
    ):
        for s in range(120):
            base = 0.50 + 0.05 * ((s % 5) - 2)
            y = np.clip(
                base + 0.25 * a - 0.15 * b + float(rng.normal(0.0, 0.005)), 0.0, 1.0
            )
            rows.append(
                {
                    "run_id": run_id,
                    "sample_id": f"item_{s:03d}",
                    "A": a,
                    "B": b,
                    "C": c,
                    "f1": y,
                }
            )

    csv_p = tmp_path / "screen.csv"
    md_p = tmp_path / "diag.md"
    json_p = tmp_path / "diag.json"
    pd.DataFrame(rows).to_csv(csv_p, index=False)

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "diagnose",
            "--data",
            str(csv_p),
            "--target",
            "f1",
            "--token-costs",
            "A=80,B=100,C=320",
            "--prune-margin",
            "0.01",
            "--sign-flip",
            "500",
            "--output-report",
            str(md_p),
            "--json-out",
            str(json_p),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "SAFE_TO_PRUNE" in res.stdout
    assert "KEEP" in res.stdout
    assert "REMOVE_HARMFUL" in res.stdout
    assert md_p.exists() and json_p.exists()

    payload = json.loads(json_p.read_text(encoding="utf-8"))
    verdicts = {f["factor_id"]: f["pruning_verdict"] for f in payload["factors"]}
    assert verdicts["A"] == "KEEP"
    assert verdicts["B"] == "REMOVE_HARMFUL"
    assert verdicts["C"] == "SAFE_TO_PRUNE"
    assert payload["estimated_prunable_tokens"] == 320
    assert payload["noise_floor"]["small_sample_warning"] is False


def test_cli_certify_and_verify_cert_fail_closed(tmp_path):
    import json

    import pandas as pd

    rows = []
    for s in range(40):
        rows.append(
            {
                "sample_id": f"s_{s:03d}",
                "group_id": f"g_{s // 2:02d}",
                "env_id": "ref",
                "score": 0.90,
                "returned_model_version": "gemini-2.5-flash",
            }
        )
        rows.append(
            {
                "sample_id": f"s_{s:03d}",
                "group_id": f"g_{s // 2:02d}",
                "env_id": "fmt_whitespace",
                "score": 0.898,
                "returned_model_version": "gemini-2.5-flash",
            }
        )
        rows.append(
            {
                "sample_id": f"s_{s:03d}",
                "group_id": f"g_{s // 2:02d}",
                "env_id": "word_paraphrase_1",
                "score": 0.895,
                "returned_model_version": "gemini-2.5-flash",
            }
        )

    csv_p = tmp_path / "stab.csv"
    cert_p = tmp_path / "cert.json"
    pd.DataFrame(rows).to_csv(csv_p, index=False)

    res_cert = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "certify",
            "--data",
            str(csv_p),
            "--target",
            "score",
            "--task-id",
            "banking77_intent",
            "--model-id",
            "gemini-2.5-flash",
            "--model-version",
            "gemini-2.5-flash",
            "--dataset-hash",
            "sha256_ds_01",
            "--parser-hash",
            "sha256_p_01",
            "--bootstrap",
            "200",
            "--output",
            str(cert_p),
            "--fail-on-error",
        ],
        capture_output=True,
        text=True,
    )
    assert res_cert.returncode == 0, res_cert.stderr
    assert "Status:              PASS" in res_cert.stdout
    assert cert_p.exists()

    # 1. Exact envelope match passes CI gate (exit code 0)
    res_ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "verify-cert",
            "--cert",
            str(cert_p),
            "--task-id",
            "banking77_intent",
            "--model-id",
            "gemini-2.5-flash",
            "--model-version",
            "gemini-2.5-flash",
            "--dataset-hash",
            "sha256_ds_01",
            "--parser-hash",
            "sha256_p_01",
            "--envs",
            "fmt_whitespace",
            "word_paraphrase_1",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res_ok.returncode == 0, res_ok.stderr
    v_ok = json.loads(res_ok.stdout)
    assert v_ok["valid"] is True
    assert v_ok["evidence_state"] == "stability-certified"
    assert v_ok["can_display_passing_stability_badge"] is True

    # 2. Provider model version swap fails closed (exit code 1, no passing badge)
    res_drift = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_prism.cli.main",
            "verify-cert",
            "--cert",
            str(cert_p),
            "--task-id",
            "banking77_intent",
            "--model-id",
            "gemini-2.5-flash",
            "--model-version",
            "gemini-2.5-flash-2026-09",
            "--dataset-hash",
            "sha256_ds_01",
            "--parser-hash",
            "sha256_p_01",
            "--envs",
            "fmt_whitespace",
            "word_paraphrase_1",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res_drift.returncode == 1
    v_drift = json.loads(res_drift.stdout)
    assert v_drift["valid"] is False
    assert v_drift["can_display_passing_stability_badge"] is False
    assert "OUT-OF-ENVELOPE" in v_drift["badge_label"]
