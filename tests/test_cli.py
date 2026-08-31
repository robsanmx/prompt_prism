"""
Unit tests for CLI commands.
"""

import subprocess
import sys


def test_cli_list_designs():
    res = subprocess.run(
        [sys.executable, "-m", "prompt_prism.cli.main", "list-designs", "--factors", "5"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "2(5-1)V" in res.stdout
    assert "2(5-2)III" in res.stdout


def test_cli_generate_design():
    res = subprocess.run(
        [sys.executable, "-m", "prompt_prism.cli.main", "design", "--factors", "5", "--runs", "16"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "2(5-1)V" in res.stdout
    assert "combination" in res.stdout
