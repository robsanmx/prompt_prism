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
