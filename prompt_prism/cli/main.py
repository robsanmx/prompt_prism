"""
Command-Line Interface (CLI) for prompt-prism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from ..analysis.anova import ANOVAEngine
from ..analysis.optimizer import OptimalPromptFinder
from ..design.catalog import CATALOG_DESIGNS, list_available_plans
from ..design.generators import FractionalFactorialGenerator, PlackettBurmanGenerator
from ..design.recommender import recommend_design
from ..reporting.reporter import AnalysisReport
from ..visualization.plots import generate_ascii_pareto


def main():
    parser = argparse.ArgumentParser(
        prog="prompt-prism",
        description="Prompt Optimization using Fractional Factorial Design of Experiments (DoE) & ANOVA",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: list-designs
    list_parser = subparsers.add_parser("list-designs", help="List available standard fractional factorial designs")
    list_parser.add_argument("--factors", "-k", type=int, help="Filter by number of factors")

    # Command: generate-design
    gen_parser = subparsers.add_parser("design", help="Generate a design matrix")
    gen_parser.add_argument("--factors", "-k", type=int, required=True, help="Number of factors")
    gen_parser.add_argument("--runs", "-r", type=int, help="Maximum number of runs / prompt variants")
    gen_parser.add_argument("--plan", "-p", type=str, help="Specific plan ID (e.g. 2(5-1)V)")
    gen_parser.add_argument("--output", "-o", type=str, help="Output CSV/JSON filepath")

    # Command: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run ANOVA on experiment results CSV")
    analyze_parser.add_argument("--data", "-d", type=str, required=True, help="Path to experiment results CSV")
    analyze_parser.add_argument("--target", "-t", type=str, required=True, help="Target metric column name")
    analyze_parser.add_argument("--factors", "-f", nargs="+", help="Factor column names (auto-detected if omitted)")
    analyze_parser.add_argument("--output-report", "-o", type=str, help="Save report to Markdown or HTML")

    args = parser.parse_args()

    if args.command == "list-designs":
        plans = list_available_plans(num_factors=args.factors)
        print(f"\n{'Plan ID':<15} {'Factors':<8} {'Runs':<6} {'Resolution':<12} {'Generators'}")
        print("-" * 65)
        for p in plans:
            gens = ", ".join(p["generators"])
            print(f"{p['plan_id']:<15} {p['num_factors']:<8} {p['runs']:<6} Res {p['resolution']:<8} {gens}")
        print()

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
            f_cols = [c for c in df.columns if c not in exclude and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]

        anova_res = ANOVAEngine.run_anova(data=df, factor_cols=f_cols, target_col=args.target)
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
