"""
Command-Line Interface (CLI) for prompt-prism / prism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from ..analysis.anova import ANOVAEngine
from ..analysis.optimizer import OptimalPromptFinder
from ..design.aliasing import AliasStructure
from ..design.catalog import CATALOG_DESIGNS, get_catalog_entry, list_available_plans
from ..design.generators import FractionalFactorialGenerator, PlackettBurmanGenerator
from ..design.recommender import recommend_design
from ..reporting.reporter import AnalysisReport
from ..visualization.plots import generate_ascii_pareto


def main():
    parser = argparse.ArgumentParser(
        prog="prompt-prism",
        description="PromptPrism: Statistical Prompt Optimization & Factorial Analysis for LLMs using Fractional Factorial DoE and ANOVA.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: list-designs
    list_parser = subparsers.add_parser("list-designs", help="List available standard fractional factorial designs (2^(k-p))")
    list_parser.add_argument("--factors", "-k", type=int, help="Filter by number of factors")
    list_parser.add_argument("--min-resolution", "-r", type=int, default=3, help="Minimum resolution (3, 4, 5, etc.)")

    # Command: alias
    alias_parser = subparsers.add_parser("alias", help="Display the alias structure and defining relation for a plan")
    alias_parser.add_argument("--plan", "-p", type=str, required=True, help="Plan ID (e.g. '2(7-3)IV' or '2(5-1)V')")

    # Command: list-metrics
    subparsers.add_parser("list-metrics", help="List built-in deterministic metrics and DeepEval LLM judge metrics")

    # Command: design
    gen_parser = subparsers.add_parser("design", help="Generate an orthogonal design matrix (CSV/JSON)")
    gen_parser.add_argument("--factors", "-k", type=int, required=True, help="Number of factors")
    gen_parser.add_argument("--runs", "-r", type=int, help="Maximum number of runs / prompt variants")
    gen_parser.add_argument("--plan", "-p", type=str, help="Specific plan ID (e.g. 2(5-1)V)")
    gen_parser.add_argument("--output", "-o", type=str, help="Output CSV/JSON filepath")

    # Command: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run ANOVA and Optimal Prompt analysis on experiment results CSV")
    analyze_parser.add_argument("--data", "-d", type=str, required=True, help="Path to experiment results CSV")
    analyze_parser.add_argument("--target", "-t", type=str, required=True, help="Target metric column name")
    analyze_parser.add_argument("--factors", "-f", nargs="+", help="Factor column names (auto-detected if omitted)")
    analyze_parser.add_argument("--block-by", "-b", type=str, default="sample_id", help="Blocking column (default: sample_id)")
    analyze_parser.add_argument("--alpha", "-a", type=float, default=0.05, help="Significance alpha threshold (default: 0.05)")
    analyze_parser.add_argument("--output-report", "-o", type=str, help="Save report to Markdown (.md) or HTML (.html)")

    args = parser.parse_args()

    if args.command == "list-designs":
        plans = list_available_plans(num_factors=args.factors)
        plans = [p for p in plans if p["resolution"] >= args.min_resolution]
        print(f"\n{'Plan ID':<15} {'Factors':<8} {'Runs':<6} {'Resolution':<12} {'Generators'}")
        print("-" * 75)
        for p in plans:
            gens = ", ".join(p["generators"])
            print(f"{p['plan_id']:<15} {p['num_factors']:<8} {p['runs']:<6} Res {p['resolution']:<8} {gens}")
        print(f"\nShowing {len(plans)} orthogonal plans. Use 'prompt-prism alias --plan <PLAN_ID>' for alias details.\n")

    elif args.command == "alias":
        plan_id = args.plan.strip()
        entry = get_catalog_entry(plan_id)
        if not entry:
            print(f"Error: Plan ID '{plan_id}' not found in catalog. Use 'prompt-prism list-designs' to see supported plans.")
            sys.exit(1)

        alias_struct = AliasStructure(generators=entry["generators"])
        factors = list(entry["factors"])
        all_aliases = alias_struct.get_all_aliases(factors=factors, max_order=2)

        print(f"\n==================================================================")
        print(f" Plan: {plan_id} (Factors: {entry['num_factors']}, Runs: {entry['runs']}, Resolution: Res {entry['resolution']})")
        print(f"==================================================================")
        print(f"Defining Relation: I = {' = '.join(alias_struct.defining_relation)}")
        print(f"Generators:        {', '.join(entry['generators'])}\n")
        print("Aliasing Structure (Confounded Effects up to order 2):")
        print("-" * 65)
        if all_aliases:
            for eff, aliases in sorted(all_aliases.items()):
                alias_str = " + ".join(aliases)
                print(f"  [{eff:<6}] confounded with: {alias_str}")
        else:
            print("  (No main effects or 2-factor interactions confounded with each other)")
        print(f"\nSummary:\n{alias_struct.summary()}\n")

    elif args.command == "list-metrics":
        has_deepeval = False
        try:
            import deepeval
            has_deepeval = True
        except ImportError:
            has_deepeval = False

        print("\n" + "=" * 75)
        print(" 📊 PROMPTPRISM EVALUATION METRICS SUITE")
        print("=" * 75)

        print("\n1. ⚡ Built-in Deterministic Metrics (Zero-Cost / Offline / Fast)")
        print("   --------------------------------------------------------------")
        print("   • exact_match           : Exact normalized text or categorical value match [0, 1]")
        print("   • f1_score              : Token-level Precision, Recall, and F1 text overlap [0, 1]")
        print("   • json_validity         : Validates JSON syntax and optional required schema keys")
        print("   • attribute_overlap     : Jaccard / F1 across extracted key-value dictionary attributes")
        print("   • levenshtein_sim       : Normalized character-level Levenshtein similarity [0, 1]")
        print("   • regex_match           : Regular expression pattern match [0, 1]")
        print("   • custom_metric         : User-defined Python callable: fn(pred, target, context)")

        print("\n2. 🎯 DeepEval Reference-Based Metrics (Uses Golden Dataset Targets)")
        print("   -----------------------------------------------------------------")
        print("   • g_eval                : Semantic truth alignment against golden expected targets")
        print("   • contextual_recall     : Measures % of golden target facts retrieved in context")
        print("   • contextual_precision  : Evaluates ranking precision of retrieved context vs golden target")
        print("   • json_correctness      : Semantic LLM validation of structured JSON output")

        print("\n3. 🔍 DeepEval Reference-Free Metrics (No Golden Dataset Required)")
        print("   ---------------------------------------------------------------")
        print("   • answer_relevancy      : Measures if LLM output directly answers the user prompt")
        print("   • faithfulness          : Verifies if output is factually grounded in retrieval context")
        print("   • hallucination         : Detects fabricated claims not present in source context")
        print("   • toxicity              : Safety audit detecting toxic, offensive, or harmful speech")
        print("   • bias                  : Detects demographic, gender, or ideological bias")
        print("   • contextual_relevancy  : Evaluates relevance of retrieved context chunks to query")
        print("   • summarization         : Measures key points retention in generated summary")

        status = "✅ AVAILABLE (deepeval installed)" if has_deepeval else "⚠️ REQUIRES: pip install prompt-prism[deepeval]"
        print(f"\nDeepEval Status: {status}")
        print("Judge Cache:     ✅ SQLite JudgeCache built-in (zero repeated judge API costs)\n")

    elif args.command == "design":
        if args.plan:
            design = FractionalFactorialGenerator.from_plan_id(args.plan)
        else:
            design = recommend_design(factors=args.factors, max_runs=args.runs)

        df = design.to_dataframe()
        print(f"\nGenerated Design: {design.plan_id} ({len(design.runs)} runs, {len(design.factor_ids)} factors, Resolution: {design.resolution})\n")
        print(df.to_string(index=False))

        if args.output:
            out_p = Path(args.output)
            if out_p.suffix.lower() == ".json":
                out_p.write_text(json.dumps(design.model_dump(), indent=2))
            else:
                df.to_csv(out_p, index=False)
            print(f"\nSaved design to {args.output}\n")

    elif args.command == "analyze":
        data_p = Path(args.data)
        if not data_p.exists():
            print(f"Error: file '{args.data}' not found.")
            sys.exit(1)

        df = pd.read_csv(data_p)
        if args.target not in df.columns:
            print(f"Error: target column '{args.target}' not found in dataset. Columns: {list(df.columns)}")
            sys.exit(1)

        if args.factors:
            f_cols = args.factors
        else:
            exclude = {args.target, "run_id", "sample_id", "trial_id", "error", "latency_ms", "combination"}
            if args.block_by:
                exclude.add(args.block_by)
            f_cols = [c for c in df.columns if c not in exclude and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]

        block_col = args.block_by if (args.block_by and args.block_by in df.columns) else None
        anova_res = ANOVAEngine.run_anova(data=df, factor_cols=f_cols, target_col=args.target, block_col=block_col, alpha=args.alpha)
        opt_rec = OptimalPromptFinder.find_optimal_prompt(anova_res)

        print("\n" + generate_ascii_pareto(anova_res) + "\n")
        print(opt_rec.summary_markdown)

        if args.output_report:
            rep = AnalysisReport(anova_result=anova_res, optimal_recommendation=opt_rec)
            if args.output_report.endswith(".html"):
                rep.to_html(args.output_report)
            else:
                rep.to_markdown(args.output_report)
            print(f"\nSaved report to {args.output_report}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
