"""
Command-Line Interface (CLI) for prompt-prism / prism (v2.0).

Supports:
- Orthogonal design discovery & generation (`list-designs`, `alias`, `design`, `list-metrics`)
- Scored test set generation, human/agent review, freezing, and coverage stats (`tests`)
- Legacy & epistasis-aware screening analysis (`analyze`)
- v2.0 calibrated item-clustered attribution, noise-floor, TOST pruning, and coverage gate (`diagnose`)
- v2.0 fail-closed stability envelope certification & CI/CD verification (`certify`, `verify-cert`)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import __version__
from ..analysis.anova import ANOVAEngine
from ..analysis.diagnostics import (
    _normalize_id_collection,
    run_calibrated_attribution_audit,
)
from ..analysis.optimizer import OptimalPromptFinder
from ..analysis.selection import (
    EXPERIMENTAL_TWO_STAGE_ALGORITHM_VERSION,
    evaluate_simultaneous_perturbation_certificate,
)
from ..core.models import OperatingEnvelopeSpec, PromptStabilityCertificate
from ..design.aliasing import AliasStructure
from ..design.catalog import get_catalog_entry, list_available_plans
from ..design.generators import FractionalFactorialGenerator
from ..design.recommender import recommend_design
from ..reporting.reporter import AnalysisReport
from ..testgen.generator import TestSetGenerator
from ..testgen.loaders import (
    load_factors_file,
    load_seed_examples_file,
    load_template_file,
    resolve_cli_client,
)
from ..testgen.models import TestSet
from ..testgen.review import (
    compute_testset_stats,
    export_review,
    format_testset_stats,
    import_review,
)
from ..visualization.plots import generate_ascii_pareto


def _exit_error(message: str, code: int = 1) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _prepare_diagnose_testset(
    tests_dir: Path,
    df: pd.DataFrame,
    f_cols: List[str],
    sample_col: str,
    *,
    split: str,
    allow_unfrozen: bool,
    require_tag_agreement: bool,
    tok_costs: Optional[Dict[str, int]],
) -> Tuple[TestSet, Dict[str, Any], Optional[Dict[str, int]]]:
    """Load and validate a TestSet for `prism diagnose --tests`, resolving factor aliases."""
    try:
        ts = TestSet.load(tests_dir)
    except (FileNotFoundError, ValueError) as exc:
        _exit_error(str(exc))

    if not ts.is_frozen and not allow_unfrozen:
        _exit_error(
            f"TestSet at '{tests_dir}' is not frozen. "
            f"Run 'prism tests freeze {tests_dir}' first or pass --allow-unfrozen."
        )

    f_col_set = set(f_cols)
    alias_to_col: Dict[str, str] = {col: col for col in f_cols}
    for fid, fname in ts.manifest.factor_id_to_name.items():
        if fid in f_col_set and fname not in f_col_set:
            alias_to_col[fname] = fid
        elif fname in f_col_set and fid not in f_col_set:
            alias_to_col[fid] = fname

    try:
        raw_cov = ts.coverage(
            split=split,  # type: ignore[arg-type]
            require_tag_agreement=require_tag_agreement,
        )
        baseline_ids = ts.baseline_sample_ids(split=split)  # type: ignore[arg-type]
        judge_frac = ts.judge_scored_fraction(split=split)  # type: ignore[arg-type]
    except ValueError as exc:
        _exit_error(str(exc))

    resolved_cov: Dict[str, set] = {}
    for k, sids in raw_cov.items():
        target_key = alias_to_col.get(str(k), str(k))
        norm_ids = _normalize_id_collection(sids, param_name=f"coverage[{k!r}]")
        resolved_cov.setdefault(target_key, set()).update(norm_ids)

    resolved_tok_costs: Optional[Dict[str, int]] = None
    if tok_costs is not None:
        resolved_tok_costs = {
            alias_to_col.get(str(k), str(k)): int(v) for k, v in tok_costs.items()
        }

    # Helpful error if data only belongs to another split (e.g., confirmation data with default --split discovery)
    if sample_col in df.columns:
        expected_split_ids = {
            c.id
            for c in ts.cases_for_split(split=split, include_rejected=False)  # type: ignore[arg-type]
        }
        present_ids = set(df[sample_col].astype(str).unique())
        has_expected_overlap = bool(present_ids & expected_split_ids) and (
            baseline_ids is None or bool(present_ids & baseline_ids)
        )
        if expected_split_ids and not has_expected_overlap:
            splits_seen = sorted(
                {
                    c.split
                    for c in ts.cases
                    if c.id in present_ids and c.split is not None
                }
            )
            if splits_seen and split not in splits_seen:
                _exit_error(
                    f"No '{split}' items from '{tests_dir}' matched '{sample_col}' in data; "
                    f"data matches split(s) {splits_seen}. Pass --split {splits_seen[0]}."
                )

    gen_model = ts.manifest.generator.get("model_id")
    kwargs: Dict[str, Any] = {
        "coverage": resolved_cov,
        "baseline_sample_ids": baseline_ids,
        "coverage_source": ts.provenance,
        "test_set_sha256": ts.manifest.cases_sha256,
        "judge_scored_fraction": judge_frac,
        "generator_model_id": str(gen_model) if gen_model else None,
    }
    return ts, kwargs, resolved_tok_costs


def _handle_tests_command(
    args: argparse.Namespace,
    tests_parser: argparse.ArgumentParser,
    review_parser: argparse.ArgumentParser,
) -> None:
    """Dispatch `prism tests {generate,review,freeze,stats}`."""
    subcmd = args.tests_command
    if subcmd == "generate":
        out_dir_str = args.out or args.refresh_confirmation
        if not out_dir_str:
            _exit_error(
                "--out is required unless --refresh-confirmation <dir> is provided."
            )
        try:
            factors = load_factors_file(args.factors)
            template = load_template_file(args.template, factors=factors)
            seeds = (
                load_seed_examples_file(args.seed_examples)
                if args.seed_examples
                else None
            )
            gen_client = resolve_cli_client(
                client_spec=args.client,
                model=args.model,
                role="generator",
            )
            tagger_client = None
            if args.tagger_client or args.tagger_model:
                tagger_client = resolve_cli_client(
                    client_spec=args.tagger_client,
                    model=args.tagger_model,
                    role="tagger",
                )

            generator = TestSetGenerator(
                client=gen_client,
                tagger_client=tagger_client,
                n_baseline_discovery=args.n_baseline_discovery,
                n_baseline_confirmation=args.n_baseline_confirmation,
                n_per_factor_discovery=args.n_per_factor_discovery,
                n_per_factor_confirmation=args.n_per_factor_confirmation,
                max_oversample_factor=args.max_oversample_factor,
                seed=args.seed,
                judge_fallback=not args.no_judge_fallback,
            )

            if args.refresh_confirmation:
                existing = TestSet.load(Path(args.refresh_confirmation))
                ts = generator.refresh_confirmation(
                    existing,
                    factors,
                    template,
                    task_description=args.task,
                    seed_examples=seeds,
                    seed_offset=args.seed_offset,
                    freeze=not args.no_freeze,
                )
            else:
                ts = generator.generate(
                    factors,
                    template,
                    task_description=args.task,
                    seed_examples=seeds,
                    freeze=not args.no_freeze,
                )

            saved_dir = ts.save(Path(out_dir_str))
        except (
            FileNotFoundError,
            ValueError,
            TypeError,
            ImportError,
            AttributeError,
        ) as exc:
            _exit_error(str(exc))

        action_label = (
            "Refreshed confirmation split in"
            if args.refresh_confirmation
            else "Generated TestSet in"
        )
        print(f"\n{action_label} {saved_dir}\n")
        print(format_testset_stats(ts))
        print()

    elif subcmd == "review":
        rcmd = args.review_command
        if rcmd == "export":
            try:
                ts = TestSet.load(Path(args.test_dir))
                summary = export_review(
                    ts,
                    Path(args.out),
                    fmt=args.format,
                    split=args.split,
                    only_disagreed=args.only_disagreed,
                )
            except (FileNotFoundError, ValueError) as exc:
                _exit_error(str(exc))
            print(
                f"Exported {summary['exported']} case(s) "
                f"({summary['disagreed']} tagger-disagreed first) "
                f"to {summary['path']} [{summary['format']}]."
            )

        elif rcmd == "import":
            try:
                ts = TestSet.load(Path(args.test_dir))
                summary = import_review(
                    ts,
                    Path(args.review_file),
                    fmt=args.format,
                    freeze=not args.no_freeze,
                )
                ts.save(Path(args.test_dir))
            except (FileNotFoundError, ValueError) as exc:
                _exit_error(str(exc))
            state = "re-frozen" if summary["frozen"] else "saved as draft"
            print(
                f"Imported {summary['rows_read']} review row(s) [{summary['format']}] "
                f"({summary['updated']} updated: {summary['accepted']} accepted, "
                f"{summary['rejected']} rejected, {summary['pending']} pending); "
                f"TestSet {state} at {args.test_dir}."
            )
            print(f"cases_sha256:        {summary['cases_sha256']}")
            print(f"active_cases_sha256: {summary['active_cases_sha256']}")
        else:
            review_parser.print_help()

    elif subcmd == "freeze":
        try:
            ts = TestSet.load(Path(args.test_dir))
            ts.freeze(force=args.force)
            ts.save(Path(args.test_dir))
        except (FileNotFoundError, ValueError) as exc:
            _exit_error(str(exc))
        print(f"Frozen TestSet at {args.test_dir}")
        print(f"frozen_at:           {ts.manifest.frozen_at}")
        print(f"cases_sha256:        {ts.manifest.cases_sha256}")
        print(f"active_cases_sha256: {ts.manifest.active_cases_sha256}")

    elif subcmd == "stats":
        try:
            ts = TestSet.load(Path(args.test_dir))
        except (FileNotFoundError, ValueError) as exc:
            _exit_error(str(exc))
        if args.as_json:
            stats = compute_testset_stats(
                ts, min_targeted_items=args.min_targeted_items
            )
            print(json.dumps(stats, indent=2))
        else:
            print(
                "\n"
                + format_testset_stats(ts, min_targeted_items=args.min_targeted_items)
                + "\n"
            )
    else:
        tests_parser.print_help()


def _load_tabular(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: file '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _auto_detect_binary_factors(
    df: pd.DataFrame, target_col: str, extra_exclude: Optional[List[str]] = None
) -> List[str]:
    exclude = {
        target_col,
        "run_id",
        "sample_id",
        "group_id",
        "env_id",
        "trial_id",
        "error",
        "latency_ms",
        "combination",
        "targets_factors",
        "intended_factor",
        "tagger_factors",
        "tagger_agrees",
        "kind",
        "split",
        "source",
        "review",
    }
    if extra_exclude:
        exclude.update(x for x in extra_exclude if x)
    detected: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        non_null = df[c].dropna()
        if non_null.empty:
            continue
        try:
            unique_vals = set(non_null.unique())
        except TypeError:
            # Unhashable cells (e.g. list-valued metadata columns) are never binary factors
            continue
        if any(isinstance(v, (bool, np.bool_)) for v in unique_vals):
            continue
        if unique_vals.issubset({0, 1, 0.0, 1.0, -1, -1.0}):
            detected.append(c)
    return detected


def _parse_token_costs(spec: Optional[str]) -> Optional[Dict[str, int]]:
    if not spec:
        return None
    out: Dict[str, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(
                f"Error: invalid --token-costs entry '{pair}'. Expected KEY=INT (e.g. A=120,B=450).",
                file=sys.stderr,
            )
            sys.exit(1)
        k, v = pair.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out or None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="prompt-prism",
        description=(
            "PromptPrism 2.0: Prompt Diagnostics, Token Pruning, Calibrated Attribution, "
            "and Envelope-Bounded Stability Certification using Fractional Factorial DoE."
        ),
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: list-designs
    list_parser = subparsers.add_parser(
        "list-designs",
        help="List available standard fractional factorial designs (2^(k-p))",
    )
    list_parser.add_argument(
        "--factors", "-k", type=int, help="Filter by number of factors"
    )
    list_parser.add_argument(
        "--min-resolution",
        "-r",
        type=int,
        default=3,
        help="Minimum resolution (3, 4, 5, etc.)",
    )

    # Command: alias
    alias_parser = subparsers.add_parser(
        "alias", help="Display the alias structure and defining relation for a plan"
    )
    alias_parser.add_argument(
        "--plan",
        "-p",
        type=str,
        required=True,
        help="Plan ID (e.g. '2(7-3)IV' or '2(5-1)V')",
    )

    # Command: list-metrics
    subparsers.add_parser(
        "list-metrics",
        help="List built-in deterministic metrics and DeepEval LLM judge metrics",
    )

    # Command: design
    gen_parser = subparsers.add_parser(
        "design", help="Generate an orthogonal design matrix (CSV/JSON)"
    )
    gen_parser.add_argument(
        "--factors", "-k", type=int, required=True, help="Number of factors"
    )
    gen_parser.add_argument(
        "--runs", "-r", type=int, help="Maximum number of runs / prompt variants"
    )
    gen_parser.add_argument(
        "--plan", "-p", type=str, help="Specific plan ID (e.g. 2(5-1)V)"
    )
    gen_parser.add_argument("--output", "-o", type=str, help="Output CSV/JSON filepath")

    # Command: analyze (backward compatible, upgraded with evidence-tier & strategy options)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run ANOVA and Optimal Prompt analysis on experiment results CSV/JSONL (evidence tier: predicted)",
    )
    analyze_parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to experiment results CSV or JSONL",
    )
    analyze_parser.add_argument(
        "--target", "-t", type=str, required=True, help="Target metric column name"
    )
    analyze_parser.add_argument(
        "--factors",
        "-f",
        nargs="+",
        help="Factor column names (auto-detected if omitted)",
    )
    analyze_parser.add_argument(
        "--block-by",
        "-b",
        type=str,
        default="sample_id",
        help="Blocking column (default: sample_id)",
    )
    analyze_parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        help="Significance alpha threshold (default: 0.05)",
    )
    analyze_parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=("additive", "screen_best", "interaction_2fi", "auto"),
        default="additive",
        help="Selection strategy (default: additive)",
    )
    analyze_parser.add_argument(
        "--include-interactions",
        action="store_true",
        help="Fit 2-factor interaction terms (recommended on Resolution V+ designs)",
    )
    analyze_parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize target metric instead of maximizing (e.g. for latency or error rate)",
    )
    analyze_parser.add_argument(
        "--output-report",
        "-o",
        type=str,
        help="Save report to Markdown (.md) or HTML (.html)",
    )

    # Command: diagnose (NEW in v2.0 — calibrated item-clustered attribution, noise floor & TOST token pruning)
    diag_parser = subparsers.add_parser(
        "diagnose",
        help="[v2.0] Calibrated item-clustered contrast t-tests, noise floor, and TOST token-bloat pruning audit",
    )
    diag_parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to experiment results CSV or JSONL",
    )
    diag_parser.add_argument(
        "--target", "-t", type=str, required=True, help="Target metric column name"
    )
    diag_parser.add_argument(
        "--factors",
        "-f",
        nargs="+",
        help="Binary factor column names (auto-detected if omitted)",
    )
    diag_parser.add_argument(
        "--sample-col",
        type=str,
        default="sample_id",
        help="Paired item identifier column (default: sample_id)",
    )
    diag_parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        help="BH FDR and TOST significance threshold (default: 0.05)",
    )
    diag_parser.add_argument(
        "--prune-margin",
        type=float,
        default=0.01,
        help="Practical equivalence margin |delta| < margin for SAFE_TO_PRUNE TOST gate (default: 0.01)",
    )
    diag_parser.add_argument(
        "--min-targeted-items",
        type=int,
        default=20,
        help="Minimum paired targeted items required before SAFE_TO_PRUNE or REMOVE_HARMFUL when coverage is present (default: 20)",
    )
    diag_parser.add_argument(
        "--targeted-margin",
        type=float,
        default=None,
        help="One-sided non-inferiority margin on targeted items (default: max(2 * prune_margin, 0.08))",
    )
    diag_parser.add_argument(
        "--token-costs",
        type=str,
        help="Optional comma-separated token cost per factor (e.g. 'A=120,B=450,C=80')",
    )
    diag_parser.add_argument(
        "--sign-flip",
        type=int,
        default=2000,
        help="Rademacher sign-flip randomization draws (0 to disable; default: 2000)",
    )
    diag_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sign-flip randomization (default: 42)",
    )
    diag_parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimize target metric instead of maximizing",
    )
    diag_parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Path to a frozen TestSet directory (cases.jsonl + manifest.json) for coverage-gated attribution",
    )
    diag_parser.add_argument(
        "--split",
        type=str,
        choices=("discovery", "confirmation", "all"),
        default="discovery",
        help="TestSet split to evaluate when --tests is provided (default: discovery)",
    )
    diag_parser.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="Allow diagnosing with an unfrozen draft TestSet when --tests is provided",
    )
    diag_parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional evaluated model ID (surfaces generator-equals-model warning with --tests)",
    )
    diag_parser.add_argument(
        "--no-require-tag-agreement",
        action="store_true",
        help="Include tagger-disagreed targeted items in coverage even before human acceptance",
    )
    diag_parser.add_argument(
        "--env-col",
        type=str,
        default=None,
        help="Optional perturbation environment column for stability-aware drop attribution",
    )
    diag_parser.add_argument(
        "--ref-env",
        type=str,
        default="ref",
        help="Clean reference environment label when --env-col is set (default: ref)",
    )
    diag_parser.add_argument(
        "--output-report",
        "-o",
        type=str,
        help="Save Markdown audit report to filepath (.md)",
    )
    diag_parser.add_argument(
        "--json-out",
        type=str,
        help="Save machine-readable JSON audit to filepath (.json)",
    )

    # Command: certify (NEW in v2.0 — generate fail-closed stability envelope certificate)
    cert_parser = subparsers.add_parser(
        "certify",
        help="[v2.0] Issue a fail-closed OperatingEnvelopeSpec stability certificate from paired perturbation results",
    )
    cert_parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to paired confirmation/perturbation CSV or JSONL",
    )
    cert_parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help="Score column name (values in [0, 1])",
    )
    cert_parser.add_argument(
        "--task-id", type=str, required=True, help="Pinned task identifier"
    )
    cert_parser.add_argument(
        "--model-id", type=str, required=True, help="Pinned model identifier"
    )
    cert_parser.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Pinned provider version or local weight commit hash",
    )
    cert_parser.add_argument(
        "--dataset-hash",
        type=str,
        required=True,
        help="SHA-256 hash of the confirmation item partition",
    )
    cert_parser.add_argument(
        "--parser-hash",
        type=str,
        required=True,
        help="SHA-256 hash of the deterministic metric/parser",
    )
    cert_parser.add_argument(
        "--env-col",
        type=str,
        default="env_id",
        help="Environment column name (default: env_id)",
    )
    cert_parser.add_argument(
        "--ref-env",
        type=str,
        default="ref",
        help="Clean reference environment label (default: ref)",
    )
    cert_parser.add_argument(
        "--envs",
        nargs="+",
        help="Expected perturbation environment IDs (defaults to all non-ref env_ids present in data)",
    )
    cert_parser.add_argument(
        "--sample-col",
        type=str,
        default="sample_id",
        help="Item identifier column (default: sample_id)",
    )
    cert_parser.add_argument(
        "--group-col",
        type=str,
        default="group_id",
        help="Optional cluster/group column for paired group bootstrap (default: group_id if present)",
    )
    cert_parser.add_argument(
        "--version-col",
        type=str,
        default="returned_model_version",
        help="Optional column of observed provider model versions to audit for drift",
    )
    cert_parser.add_argument(
        "--algorithm-version",
        type=str,
        default=EXPERIMENTAL_TWO_STAGE_ALGORITHM_VERSION,
        help="Pinned selector/algorithm version string",
    )
    cert_parser.add_argument(
        "--decoding-json",
        type=str,
        help='Optional JSON object of decoding settings (default: {"temperature":0.0,"max_output_tokens":512,"seed":42})',
    )
    cert_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="Maximum allowable simultaneous drop margin epsilon_stability (default: 0.02)",
    )
    cert_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="One-sided confidence level alpha (default: 0.05)",
    )
    cert_parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Paired bootstrap replicates (default: 1000)",
    )
    cert_parser.add_argument(
        "--seed", type=int, default=20261118, help="Bootstrap RNG seed"
    )
    cert_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save machine-readable stability certificate JSON to filepath",
    )
    cert_parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with code 2 if certificate status is not PASS (useful in CI/CD gates)",
    )

    # Command: verify-cert (NEW in v2.0 — CI/CD fail-closed envelope verification gate)
    verify_parser = subparsers.add_parser(
        "verify-cert",
        help="[v2.0] Fail-closed CI/CD gate verifying a PromptStabilityCertificate against a live deployment envelope",
    )
    verify_parser.add_argument(
        "--cert",
        "-c",
        type=str,
        required=True,
        help="Path to stability certificate JSON file",
    )
    verify_parser.add_argument(
        "--task-id", type=str, required=True, help="Live task identifier"
    )
    verify_parser.add_argument(
        "--model-id", type=str, required=True, help="Live model identifier"
    )
    verify_parser.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Live provider version or local weight commit hash",
    )
    verify_parser.add_argument(
        "--dataset-hash",
        type=str,
        required=True,
        help="Live confirmation dataset partition hash",
    )
    verify_parser.add_argument(
        "--parser-hash",
        type=str,
        required=True,
        help="Live deterministic metric/parser hash",
    )
    verify_parser.add_argument(
        "--envs",
        nargs="+",
        required=True,
        help="Required perturbation environment IDs covered by deployment gate",
    )
    verify_parser.add_argument(
        "--algorithm-version",
        type=str,
        default=EXPERIMENTAL_TWO_STAGE_ALGORITHM_VERSION,
        help="Live algorithm version string",
    )
    verify_parser.add_argument(
        "--decoding-json",
        type=str,
        help="Optional JSON object of live decoding config",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print validation verdict as machine-readable JSON",
    )

    # Command: tests (NEW in v2.1 — scored test set generation, review, freezing & stats)
    tests_parser = subparsers.add_parser(
        "tests",
        help="[v2.1] Generate, review, freeze, and inspect scored TestSets for coverage-gated diagnostics",
    )
    tests_sub = tests_parser.add_subparsers(
        dest="tests_command", help="TestSet commands"
    )

    # tests generate
    tgen_parser = tests_sub.add_parser(
        "generate",
        help="Generate (or refresh confirmation split of) a scored, coverage-tagged TestSet",
    )
    tgen_parser.add_argument(
        "--factors",
        "-f",
        type=str,
        required=True,
        help="Path to factors YAML or JSON file",
    )
    tgen_parser.add_argument(
        "--template",
        "-t",
        type=str,
        required=True,
        help="Path to prompt template YAML or JSON file",
    )
    tgen_parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Natural-language task description for test generation",
    )
    tgen_parser.add_argument(
        "--seed-examples",
        type=str,
        default=None,
        help="Optional seed examples file (.jsonl, .json, .yaml, or .csv)",
    )
    tgen_parser.add_argument(
        "--n-baseline-discovery",
        type=int,
        default=100,
        help="Target post-filter baseline items in discovery split (default: 100)",
    )
    tgen_parser.add_argument(
        "--n-baseline-confirmation",
        type=int,
        default=40,
        help="Target post-filter baseline items in confirmation split (default: 40)",
    )
    tgen_parser.add_argument(
        "--n-per-factor-discovery",
        type=int,
        default=35,
        help="Target post-filter targeted items per factor in discovery split (default: 35)",
    )
    tgen_parser.add_argument(
        "--n-per-factor-confirmation",
        type=int,
        default=20,
        help="Target post-filter targeted items per factor in confirmation split (default: 20)",
    )
    tgen_parser.add_argument(
        "--max-oversample-factor",
        type=float,
        default=2.0,
        help="Bounded oversampling multiplier for post-filter target counts (default: 2.0)",
    )
    tgen_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation prompts and stratified splitting (default: 0)",
    )
    tgen_parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help="Seed offset used with --refresh-confirmation (default: 1)",
    )
    tgen_parser.add_argument(
        "--no-judge-fallback",
        action="store_true",
        help="Forbid llm_judge fallback checks (deterministic checks only)",
    )
    tgen_parser.add_argument(
        "--client",
        type=str,
        default=None,
        help="Generator client reference 'module.path:callable' (or LLMClient instance)",
    )
    tgen_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Generator model name via LiteLLM (requires prompt-prism[cloud])",
    )
    tgen_parser.add_argument(
        "--tagger-client",
        type=str,
        default=None,
        help="Optional separate blind tagger client 'module.path:callable'",
    )
    tgen_parser.add_argument(
        "--tagger-model",
        type=str,
        default=None,
        help="Optional separate blind tagger model via LiteLLM (requires prompt-prism[cloud])",
    )
    tgen_parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Leave the generated TestSet unfrozen (draft) for human review before freezing",
    )
    tgen_parser.add_argument(
        "--refresh-confirmation",
        type=str,
        default=None,
        metavar="DIR",
        help="Regenerate only the confirmation split of existing TestSet DIR while preserving discovery",
    )
    tgen_parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Output TestSet directory (defaults to --refresh-confirmation DIR when refreshing)",
    )

    # tests review {export, import}
    review_parser = tests_sub.add_parser(
        "review",
        help="Export or import human/agent review decisions (CSV or Markdown)",
    )
    review_sub = review_parser.add_subparsers(
        dest="review_command", help="Review commands"
    )

    rev_exp = review_sub.add_parser(
        "export",
        help="Export TestSet cases for review (tagger-disagreed items sorted first)",
    )
    rev_exp.add_argument("test_dir", type=str, help="Path to TestSet directory")
    rev_exp.add_argument(
        "--format",
        type=str,
        choices=("csv", "md"),
        default=None,
        help="Export format ('csv' or 'md'; inferred from --out extension if omitted)",
    )
    rev_exp.add_argument(
        "--out",
        "-o",
        type=str,
        required=True,
        help="Output review file path (e.g. review.csv or review.md)",
    )
    rev_exp.add_argument(
        "--split",
        type=str,
        choices=("all", "discovery", "confirmation"),
        default="all",
        help="Which split to export (default: all)",
    )
    rev_exp.add_argument(
        "--only-disagreed",
        action="store_true",
        help="Export only tagger-disagreed and untagged targeted items",
    )

    rev_imp = review_sub.add_parser(
        "import",
        help="Import accepted/rejected review decisions (auto-detects CSV/MD) and re-freeze",
    )
    rev_imp.add_argument("test_dir", type=str, help="Path to TestSet directory")
    rev_imp.add_argument(
        "review_file", type=str, help="Edited review file (.csv or .md)"
    )
    rev_imp.add_argument(
        "--format",
        type=str,
        choices=("csv", "md"),
        default=None,
        help="Override review file format detection ('csv' or 'md')",
    )
    rev_imp.add_argument(
        "--no-freeze",
        action="store_true",
        help="Save review decisions as draft without re-freezing",
    )

    # tests freeze
    freeze_parser = tests_sub.add_parser(
        "freeze",
        help="Freeze a draft TestSet with canonical-JSON SHA-256 hashes",
    )
    freeze_parser.add_argument("test_dir", type=str, help="Path to TestSet directory")
    freeze_parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly re-freeze an already-frozen TestSet",
    )

    # tests stats
    stats_parser = tests_sub.add_parser(
        "stats",
        help="Show per-split, per-factor, clause-role, and judge-share TestSet stats",
    )
    stats_parser.add_argument("test_dir", type=str, help="Path to TestSet directory")
    stats_parser.add_argument(
        "--min-targeted-items",
        type=int,
        default=20,
        help="Coverage gate threshold used for thin-factor warnings (default: 20)",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print TestSet statistics as machine-readable JSON",
    )

    args = parser.parse_args(argv)

    if args.command == "list-designs":
        plans = list_available_plans(num_factors=args.factors)
        plans = [p for p in plans if p["resolution"] >= args.min_resolution]
        print(
            f"\n{'Plan ID':<15} {'Factors':<8} {'Runs':<6} {'Resolution':<12} {'Generators'}"
        )
        print("-" * 75)
        for p in plans:
            gens = ", ".join(p["generators"])
            print(
                f"{p['plan_id']:<15} {p['num_factors']:<8} {p['runs']:<6} Res {p['resolution']:<8} {gens}"
            )
        print(
            f"\nShowing {len(plans)} orthogonal plans. Use 'prompt-prism alias --plan <PLAN_ID>' for alias details.\n"
        )

    elif args.command == "alias":
        plan_id = args.plan.strip()
        entry = get_catalog_entry(plan_id)
        if not entry:
            print(
                f"Error: Plan ID '{plan_id}' not found in catalog. Use 'prompt-prism list-designs' to see supported plans."
            )
            sys.exit(1)

        alias_struct = AliasStructure(generators=entry["generators"])
        factors = list(entry["factors"])
        all_aliases = alias_struct.get_all_aliases(factors=factors, max_order=2)

        print("\n==================================================================")
        print(
            f" Plan: {plan_id} (Factors: {entry['num_factors']}, Runs: {entry['runs']}, Resolution: Res {entry['resolution']})"
        )
        print("==================================================================")
        print(f"Defining Relation: I = {' = '.join(alias_struct.defining_relation)}")
        print(f"Generators:        {', '.join(entry['generators'])}\n")
        print("Aliasing Structure (Confounded Effects up to order 2):")
        print("-" * 65)
        if all_aliases:
            for eff, aliases in sorted(all_aliases.items()):
                alias_str = " + ".join(aliases)
                print(f"  [{eff:<6}] confounded with: {alias_str}")
        else:
            print(
                "  (No main effects or 2-factor interactions confounded with each other)"
            )
        print(f"\nSummary:\n{alias_struct.summary()}\n")

    elif args.command == "list-metrics":
        has_deepeval = importlib.util.find_spec("deepeval") is not None

        print("\n" + "=" * 75)
        print(f" 📊 PROMPTPRISM v{__version__} EVALUATION METRICS SUITE")
        print("=" * 75)

        print("\n1. ⚡ Built-in Deterministic Metrics (Zero-Cost / Offline / Fast)")
        print("   --------------------------------------------------------------")
        print(
            "   • exact_match           : Exact normalized text or categorical value match [0, 1]"
        )
        print(
            "   • f1_score              : Token-level Precision, Recall, and F1 text overlap [0, 1]"
        )
        print(
            "   • json_validity         : Validates JSON syntax and optional required schema keys"
        )
        print(
            "   • attribute_overlap     : Jaccard / F1 across extracted key-value dictionary attributes"
        )
        print(
            "   • levenshtein_sim       : Normalized character-level Levenshtein similarity [0, 1]"
        )
        print("   • regex_match           : Regular expression pattern match [0, 1]")
        print(
            "   • check_suite           : Per-item TestSet CheckSpec suite (deterministic + optional judge)"
        )
        print(
            "   • custom_metric         : User-defined Python callable: fn(pred, target, context)"
        )

        print("\n2. 🎯 DeepEval Reference-Based Metrics (Uses Golden Dataset Targets)")
        print("   -----------------------------------------------------------------")
        print(
            "   • g_eval                : Semantic truth alignment against golden expected targets"
        )
        print(
            "   • contextual_recall     : Measures % of golden target facts retrieved in context"
        )
        print(
            "   • contextual_precision  : Evaluates ranking precision of retrieved context vs golden target"
        )
        print(
            "   • json_correctness      : Semantic LLM validation of structured JSON output"
        )

        print("\n3. 🔍 DeepEval Reference-Free Metrics (No Golden Dataset Required)")
        print("   ---------------------------------------------------------------")
        print(
            "   • answer_relevancy      : Measures if LLM output directly answers the user prompt"
        )
        print(
            "   • faithfulness          : Verifies if output is factually grounded in retrieval context"
        )
        print(
            "   • hallucination         : Groundedness against source context; 1.0 = no fabricated claims"
        )
        print(
            "   • toxicity              : Safety audit; 1.0 = free of toxic, offensive, or harmful speech"
        )
        print(
            "   • bias                  : Impartiality; 1.0 = free of demographic, gender, or ideological bias"
        )
        print(
            "   • contextual_relevancy  : Evaluates relevance of retrieved context chunks to query"
        )
        print(
            "   • summarization         : Measures key points retention in generated summary"
        )

        status = (
            "✅ AVAILABLE (deepeval installed)"
            if has_deepeval
            else "⚠️ REQUIRES: pip install prompt-prism[deepeval]"
        )
        print(f"\nDeepEval Status: {status}")
        print(
            "Judge Cache:     ✅ SQLite JudgeCache built-in (zero repeated judge API costs)\n"
        )

    elif args.command == "design":
        if args.plan:
            design = FractionalFactorialGenerator.from_plan_id(args.plan)
        else:
            design = recommend_design(factors=args.factors, max_runs=args.runs)

        df = design.to_dataframe()
        print(
            f"\nGenerated Design: {design.plan_id} ({len(design.runs)} runs, {len(design.factor_ids)} factors, Resolution: {design.resolution})\n"
        )
        print(df.to_string(index=False))

        if args.output:
            out_p = Path(args.output)
            if out_p.suffix.lower() == ".json":
                out_p.write_text(json.dumps(design.model_dump(), indent=2))
            else:
                df.to_csv(out_p, index=False)
            print(f"\nSaved design to {args.output}\n")

    elif args.command == "analyze":
        df = _load_tabular(Path(args.data))
        if args.target not in df.columns:
            print(
                f"Error: target column '{args.target}' not found in dataset. Columns: {list(df.columns)}"
            )
            sys.exit(1)

        f_cols = args.factors or _auto_detect_binary_factors(
            df, args.target, extra_exclude=[args.block_by]
        )
        block_col = (
            args.block_by if (args.block_by and args.block_by in df.columns) else None
        )
        # Direction reaches both run_anova and find_optimal_prompt so main-effects table
        # actions and optimizer recommendations stay consistent.
        maximize = not args.minimize
        anova_res = ANOVAEngine.run_anova(
            data=df,
            factor_cols=f_cols,
            target_col=args.target,
            block_col=block_col,
            include_interactions=args.include_interactions,
            alpha=args.alpha,
            maximize=maximize,
        )
        opt_rec = OptimalPromptFinder.find_optimal_prompt(
            anova_res, maximize=maximize, strategy=args.strategy
        )

        print("\n" + generate_ascii_pareto(anova_res) + "\n")
        print(opt_rec.summary_markdown)
        print(
            "\n[Evidence Tier: predicted] Single-stage screen output only (not confirmation-validated "
            "or stability-certified). Run `prompt-prism diagnose` for calibrated item-clustered t-tests "
            "& TOST token-bloat pruning, and `prompt-prism certify` on untouched perturbation data."
        )

        if args.output_report:
            rep = AnalysisReport(anova_result=anova_res, optimal_recommendation=opt_rec)
            if args.output_report.endswith(".html"):
                rep.to_html(args.output_report)
            else:
                rep.to_markdown(args.output_report)
            print(f"\nSaved report to {args.output_report}\n")

    elif args.command == "diagnose":
        df = _load_tabular(Path(args.data))
        if args.target not in df.columns:
            print(
                f"Error: target column '{args.target}' not found in dataset. Columns: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

        extra_excl = [args.sample_col]
        if args.env_col:
            extra_excl.append(args.env_col)
        f_cols = args.factors or _auto_detect_binary_factors(
            df, args.target, extra_exclude=extra_excl
        )
        if not f_cols:
            print(
                "Error: no binary factor columns detected. Pass --factors explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

        tok_costs = _parse_token_costs(args.token_costs)
        loaded_ts: Optional[TestSet] = None
        ts_kwargs: Dict[str, Any] = {}
        if args.tests:
            loaded_ts, ts_kwargs, tok_costs = _prepare_diagnose_testset(
                Path(args.tests),
                df,
                f_cols,
                args.sample_col,
                split=args.split,
                allow_unfrozen=args.allow_unfrozen,
                require_tag_agreement=not args.no_require_tag_agreement,
                tok_costs=tok_costs,
            )

        try:
            audit = run_calibrated_attribution_audit(
                data=df,
                factor_cols=f_cols,
                target_col=args.target,
                sample_col=args.sample_col,
                alpha=args.alpha,
                maximize=not args.minimize,
                equivalence_margin=args.prune_margin,
                token_costs=tok_costs,
                n_sign_flip=args.sign_flip,
                seed=args.seed,
                min_targeted_items=args.min_targeted_items,
                targeted_margin=args.targeted_margin,
                model_id=args.model_id,
                env_col=args.env_col,
                ref_env=args.ref_env,
                **ts_kwargs,
            )
        except ValueError as exc:
            _exit_error(str(exc))

        if (
            loaded_ts is not None
            and args.sample_col in df.columns
            and loaded_ts.detect_split_leakage(df[args.sample_col].astype(str).unique())
        ):
            audit.warnings.append(
                "Split leakage detected: evaluated results mix both 'discovery' and "
                "'confirmation' items from the TestSet."
            )

        md = audit.to_markdown()
        print("\n" + md + "\n")

        if args.output_report:
            Path(args.output_report).write_text(md + "\n", encoding="utf-8")
            print(f"Saved calibrated audit report to {args.output_report}")
        if args.json_out:
            Path(args.json_out).write_text(
                audit.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved calibrated audit JSON to {args.json_out}")

    elif args.command == "certify":
        df = _load_tabular(Path(args.data))
        for req_col in (args.target, args.env_col, args.sample_col):
            if req_col not in df.columns:
                print(
                    f"Error: required column '{req_col}' not found in dataset. Columns: {list(df.columns)}",
                    file=sys.stderr,
                )
                sys.exit(1)

        ref_df = df[df[args.env_col].astype(str) == args.ref_env].copy()
        if ref_df.empty:
            print(
                f"Error: reference environment '{args.ref_env}' not found in column '{args.env_col}'.",
                file=sys.stderr,
            )
            sys.exit(1)

        ref_df = ref_df.sort_values(args.sample_col)
        ref_samples = ref_df[args.sample_col].astype(str).tolist()
        ref_scores = ref_df[args.target].astype(float).tolist()
        group_ids = (
            ref_df[args.group_col].astype(str).tolist()
            if (args.group_col and args.group_col in ref_df.columns)
            else None
        )

        all_other_envs = sorted(
            {
                str(e)
                for e in df[args.env_col].dropna().unique()
                if str(e) != args.ref_env
            }
        )
        pert_envs = list(args.envs) if args.envs else all_other_envs

        pert_scores: Dict[str, List[float]] = {}
        obs_versions: Dict[str, str] = {}
        has_ver_col = args.version_col in df.columns
        if has_ver_col:
            ref_vers = sorted(
                {str(v) for v in ref_df[args.version_col].dropna().unique()}
            )
            if ref_vers:
                obs_versions[args.ref_env] = next(
                    (v for v in ref_vers if v != args.model_version), ref_vers[0]
                )

        for e_id in pert_envs:
            edf = df[df[args.env_col].astype(str) == e_id]
            if edf.empty:
                continue
            by_sample = dict(
                zip(edf[args.sample_col].astype(str), edf[args.target].astype(float))
            )
            if all(s in by_sample for s in ref_samples):
                pert_scores[e_id] = [by_sample[s] for s in ref_samples]
            else:
                # Pass mismatched-length list so evaluate_simultaneous_perturbation_certificate fails closed
                pert_scores[e_id] = edf[args.target].astype(float).tolist()
            if has_ver_col:
                e_vers = sorted(
                    {str(v) for v in edf[args.version_col].dropna().unique()}
                )
                if e_vers:
                    obs_versions[e_id] = next(
                        (v for v in e_vers if v != args.model_version), e_vers[0]
                    )

        decoding_cfg: Dict[str, Any] = (
            json.loads(args.decoding_json)
            if args.decoding_json
            else {"temperature": 0.0, "max_output_tokens": 512, "seed": 42}
        )
        envelope = OperatingEnvelopeSpec(
            task_id=args.task_id,
            model_id=args.model_id,
            model_version=args.model_version,
            dataset_partition_hash=args.dataset_hash,
            parser_hash=args.parser_hash,
            perturbation_env_ids=pert_envs,
            algorithm_version=args.algorithm_version,
            decoding_config=decoding_cfg,
        )
        cert = evaluate_simultaneous_perturbation_certificate(
            envelope=envelope,
            reference_scores=ref_scores,
            perturbation_scores_by_env=pert_scores,
            group_ids=group_ids,
            epsilon_stability=args.epsilon,
            alpha=args.alpha,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            observed_model_versions_by_env=obs_versions or None,
        )

        print("\n=== PromptPrism 2.0 Stability Envelope Certificate ===")
        print(f"Status:              {cert.status}")
        print(
            f"Task / Model:        {envelope.task_id} / {envelope.model_id}@{envelope.model_version}"
        )
        print(
            f"Perturbation Envs:   {', '.join(envelope.perturbation_env_ids) or '(none)'}"
        )
        if cert.reference_mean is not None:
            print(f"Reference Mean:      {cert.reference_mean:.4f}")
        if (
            cert.d_stability_estimate is not None
            and cert.d_stability_upper_95_sim is not None
        ):
            worst_str = f" [{cert.worst_env_id}]" if cert.worst_env_id else ""
            print(
                f"Max Drop / U95_sim:  {cert.d_stability_estimate:+.4f}{worst_str} / "
                f"{cert.d_stability_upper_95_sim:.4f} (margin < {cert.epsilon_stability:.4f})"
            )
        if cert.repeat_noise_se is not None:
            print(f"Repeat Noise SE:     {cert.repeat_noise_se:.4f}")
        if cert.diagnostic_note:
            print(f"Diagnostic:          {cert.diagnostic_note}")
        if cert.drift_events:
            print(f"Drift / Integrity:   {'; '.join(cert.drift_events)}")
        print(f"Envelope SHA-256:    {envelope.envelope_sha256()}")
        print(f"Certificate SHA-256: {cert.certificate_sha256()}\n")

        if args.output:
            Path(args.output).write_text(
                cert.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            print(f"Saved stability certificate to {args.output}")

        if args.fail_on_error and not cert.is_passing_within_envelope:
            sys.exit(2)

    elif args.command == "verify-cert":
        cert_p = Path(args.cert)
        if not cert_p.exists():
            print(f"Error: certificate file '{args.cert}' not found.", file=sys.stderr)
            sys.exit(1)
        cert = PromptStabilityCertificate.model_validate_json(
            cert_p.read_text(encoding="utf-8")
        )
        decoding_cfg = (
            json.loads(args.decoding_json)
            if args.decoding_json
            else {"temperature": 0.0, "max_output_tokens": 512, "seed": 42}
        )
        req_env = OperatingEnvelopeSpec(
            task_id=args.task_id,
            model_id=args.model_id,
            model_version=args.model_version,
            dataset_partition_hash=args.dataset_hash,
            parser_hash=args.parser_hash,
            perturbation_env_ids=list(args.envs),
            algorithm_version=args.algorithm_version,
            decoding_config=decoding_cfg,
        )
        res = cert.validate_request(
            req_env, has_confirmation_validation=True, strict=False
        )
        if args.as_json:
            print(res.model_dump_json(indent=2))
        else:
            print("\n=== PromptPrism 2.0 Envelope Gate Verification ===")
            print(f"Valid:               {res.valid}")
            print(f"Evidence State:      {res.evidence_state}")
            print(f"Stability Badge:     {res.badge_label}")
            if res.mismatch_reasons:
                print(f"Refusal Reasons:     {'; '.join(res.mismatch_reasons)}")
            print()

        if not (res.valid and res.can_display_passing_stability_badge):
            sys.exit(1)

    elif args.command == "tests":
        _handle_tests_command(args, tests_parser, review_parser)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
