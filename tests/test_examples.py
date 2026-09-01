"""
Smoke tests ensuring all example scripts compile and run cleanly without network calls.
"""

import importlib.util
import py_compile
from pathlib import Path

import pytest

HAS_DEEPEVAL = importlib.util.find_spec("deepeval") is not None


def _strip_provider_keys(monkeypatch):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_examples_py_compile():
    examples_dir = Path(__file__).parent.parent / "examples"
    py_files = list(examples_dir.glob("*.py"))
    assert len(py_files) >= 3

    for py_file in py_files:
        py_compile.compile(str(py_file), doraise=True)


def test_prompt_optimization_tutorial_runs(tmp_path, monkeypatch):
    _strip_provider_keys(monkeypatch)
    repo_root = Path(__file__).parent.parent
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)
    from examples import prompt_optimization_tutorial

    prompt_optimization_tutorial.main()


def test_rag_prompt_optimization_runs(tmp_path, monkeypatch):
    _strip_provider_keys(monkeypatch)
    repo_root = Path(__file__).parent.parent
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)
    from examples import rag_prompt_optimization

    rag_prompt_optimization.main()


@pytest.mark.skipif(
    not HAS_DEEPEVAL,
    reason="example builds deepeval judge metrics; without the extra every score is NaN "
    "and ANOVA has no varying factor to fit (R21 skip-guard convention)",
)
def test_deepeval_golden_dataset_optimization_runs(tmp_path, monkeypatch):
    _strip_provider_keys(monkeypatch)
    repo_root = Path(__file__).parent.parent
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)
    from examples import deepeval_golden_dataset_optimization

    deepeval_golden_dataset_optimization.main()
