"""
Section 8.3: Monte Carlo Simulation Calibration for the Coverage-Aware Pruning Gate.

Extends Study A's SyntheticDGP with conditional-rule clauses affecting a fraction
p in {0.02, 0.05, 0.20, 1.00} of baseline items across realistic paired contrast SDs
(0.08 - 0.15) in a 16-run 2^(6-2)IV fractional factorial design.

Measures:
1. False-prune rate: probability a real conditional rule is mislabeled `SAFE_TO_PRUNE`
   WITHOUT the coverage gate (baseline-only) vs. WITH the coverage gate.
2. Dead-clause pruning power: probability a true dead clause (`tau_F = 0, tau_T = 0`)
   is labeled `SAFE_TO_PRUNE` across `(n_per_factor_discovery, targeted_margin, contrast_sd)`
   (both conditional on baseline TOST passing and overall).

Usage:
    python -m research.simulation.coverage_calibration --reps 150 --out research/results/coverage_calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tabulate import tabulate

from prompt_prism.analysis.diagnostics import run_calibrated_attribution_audit
from prompt_prism.design.generators import FractionalFactorialGenerator
from research.simulation.dgp import DGPConfig, SyntheticDGP


def _design_combos_6f() -> List[Tuple[int, ...]]:
    design = FractionalFactorialGenerator.from_plan_id("2(6-2)IV")
    fids = list(design.factor_ids)
    return [tuple(int(run.factor_levels[fid]) for fid in fids) for run in design.runs]


def run_false_prune_comparison(
    *,
    p_values: Sequence[float] = (0.02, 0.05, 0.20, 1.00),
    contrast_sds: Sequence[float] = (0.08, 0.11, 0.15),
    targeted_effects: Sequence[float] = (0.15, 0.25),
    n_baseline: int = 100,
    n_targeted: int = 35,
    equivalence_margin: float = 0.03,
    targeted_margin: float = 0.08,
    min_targeted_items: int = 20,
    alpha: float = 0.05,
    reps: int = 120,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """Compare false-prune rates on a real conditional rule (`C`) with vs. without the coverage gate."""
    combos = _design_combos_6f()
    rows: List[Dict[str, Any]] = []
    run_noise = 0.02

    for tau_t in targeted_effects:
        for sd in contrast_sds:
            het_sd = float(np.sqrt(max(0.0, sd**2 - (run_noise / 2.0) ** 2)))
            for p_trig in p_values:
                fp_no_gate = 0
                fp_with_gate = 0
                keep_with_gate = 0

                for r in range(reps):
                    cfg = DGPConfig(
                        scenario_id=f"fp_p{p_trig}_sd{sd}_tau{tau_t}_r{r}",
                        k=6,
                        factor_ids=["A", "B", "C", "D", "E", "F"],
                        num_items=n_baseline,
                        num_replicates=1,
                        baseline_intercept=0.55,
                        main_effects={"A": 0.08, "B": -0.08, "F": 0.06},
                        conditional_effects={"C": (p_trig, tau_t), "D": (p_trig, 0.0)},
                        num_targeted_per_factor=n_targeted,
                        item_sd=0.08,
                        noise_sd=run_noise,
                        contrast_heterogeneity_sd=het_sd,
                        outcome_type="continuous",
                        seed=seed
                        + r * 31
                        + int(round(p_trig * 1000))
                        + int(round(sd * 100)) * 7,
                    )
                    cube = SyntheticDGP.generate_design_dataset(cfg, combos)
                    df = cube.dataframe
                    df_base = df[df["sample_id"].isin(cube.baseline_sample_ids)]

                    # 1. Unguarded baseline-only audit (coverage=None)
                    audit_ung = run_calibrated_attribution_audit(
                        data=df_base,
                        factor_cols=cfg.factor_ids,
                        target_col="metric",
                        alpha=alpha,
                        equivalence_margin=equivalence_margin,
                        n_sign_flip=0,
                    )
                    c_ung = next(d for d in audit_ung.factors if d.factor_id == "C")
                    if c_ung.pruning_verdict == "SAFE_TO_PRUNE":
                        fp_no_gate += 1

                    # 2. Coverage-gated audit (baseline F + targeted T)
                    audit_gated = run_calibrated_attribution_audit(
                        data=df,
                        factor_cols=cfg.factor_ids,
                        target_col="metric",
                        alpha=alpha,
                        equivalence_margin=equivalence_margin,
                        coverage=cube.coverage_map,
                        baseline_sample_ids=cube.baseline_sample_ids,
                        min_targeted_items=min_targeted_items,
                        targeted_margin=targeted_margin,
                        n_sign_flip=0,
                    )
                    c_gated = next(d for d in audit_gated.factors if d.factor_id == "C")
                    if c_gated.pruning_verdict == "SAFE_TO_PRUNE":
                        fp_with_gate += 1
                    elif c_gated.pruning_verdict in {"KEEP", "KEEP_TARGETED"}:
                        keep_with_gate += 1

                rows.append(
                    {
                        "tau_targeted": tau_t,
                        "contrast_sd": sd,
                        "trigger_p": p_trig,
                        "n_baseline": n_baseline,
                        "n_targeted": n_targeted,
                        "equivalence_margin": equivalence_margin,
                        "targeted_margin": targeted_margin,
                        "reps": reps,
                        "false_prune_no_gate": fp_no_gate / reps,
                        "false_prune_with_gate": fp_with_gate / reps,
                        "keep_rate_with_gate": keep_with_gate / reps,
                    }
                )
    return rows


def run_dead_clause_power_grid(
    *,
    n_targeted_grid: Sequence[int] = (20, 30, 35, 50),
    targeted_margin_grid: Sequence[float] = (0.05, 0.06, 0.08),
    contrast_sds: Sequence[float] = (0.08, 0.11, 0.15),
    n_baseline: int = 100,
    equivalence_margin: float = 0.03,
    min_targeted_items: int = 20,
    alpha: float = 0.05,
    reps: int = 150,
    seed: int = 4096,
) -> List[Dict[str, Any]]:
    """Measure dead-clause pruning power (overall and targeted NI gate) across (n_T, targeted_margin, sd)."""
    combos = _design_combos_6f()
    rows: List[Dict[str, Any]] = []
    run_noise = 0.02

    for sd in contrast_sds:
        het_sd = float(np.sqrt(max(0.0, sd**2 - (run_noise / 2.0) ** 2)))
        for n_t in n_targeted_grid:
            # Generate `reps` cubes once per (sd, n_t) and evaluate every targeted_margin on the same draws
            cubes = []
            for r in range(reps):
                cfg = DGPConfig(
                    scenario_id=f"dead_sd{sd}_nt{n_t}_r{r}",
                    k=6,
                    factor_ids=["A", "B", "C", "D", "E", "F"],
                    num_items=n_baseline,
                    num_replicates=1,
                    baseline_intercept=0.55,
                    main_effects={"A": 0.08, "B": -0.08, "F": 0.06},
                    conditional_effects={"C": (0.05, 0.25), "D": (0.05, 0.0)},
                    num_targeted_per_factor=n_t,
                    item_sd=0.08,
                    noise_sd=run_noise,
                    contrast_heterogeneity_sd=het_sd,
                    outcome_type="continuous",
                    seed=seed + r * 37 + n_t * 13 + int(round(sd * 100)) * 5,
                )
                cubes.append(SyntheticDGP.generate_design_dataset(cfg, combos))

            for m_t in targeted_margin_grid:
                ni_pass_count = 0
                f_tost_pass_count = 0
                safe_prune_count = 0
                fp_c_count = 0

                for cube in cubes:
                    audit = run_calibrated_attribution_audit(
                        data=cube.dataframe,
                        factor_cols=cube.config.factor_ids,
                        target_col="metric",
                        alpha=alpha,
                        equivalence_margin=equivalence_margin,
                        coverage=cube.coverage_map,
                        baseline_sample_ids=cube.baseline_sample_ids,
                        min_targeted_items=min_targeted_items,
                        targeted_margin=m_t,
                        n_sign_flip=0,
                    )
                    by_f = {d.factor_id: d for d in audit.factors}
                    d_dead = by_f["D"]
                    c_real = by_f["C"]

                    if d_dead.tost_p_value < alpha:
                        f_tost_pass_count += 1
                    if (
                        d_dead.p_noninferiority_targeted is not None
                        and d_dead.p_noninferiority_targeted < alpha
                    ):
                        ni_pass_count += 1
                    if d_dead.pruning_verdict == "SAFE_TO_PRUNE":
                        safe_prune_count += 1
                    if c_real.pruning_verdict == "SAFE_TO_PRUNE":
                        fp_c_count += 1

                rows.append(
                    {
                        "contrast_sd": sd,
                        "n_per_factor_discovery": n_t,
                        "targeted_margin": m_t,
                        "n_baseline": n_baseline,
                        "equivalence_margin": equivalence_margin,
                        "reps": reps,
                        "targeted_ni_power": ni_pass_count / reps,
                        "baseline_tost_power": f_tost_pass_count / reps,
                        "joint_prune_power": safe_prune_count / reps,
                        "real_rule_false_prune_rate": fp_c_count / reps,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Section 8.3 Monte Carlo coverage-gate calibration study."
    )
    parser.add_argument("--reps", type=int, default=120, help="Replications per cell")
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.03,
        help="Baseline TOST equivalence margin (default: 0.03)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path for calibration results",
    )
    args = parser.parse_args()

    fp_rows = run_false_prune_comparison(
        reps=args.reps,
        equivalence_margin=args.equivalence_margin,
    )
    power_rows = run_dead_clause_power_grid(
        reps=args.reps,
        equivalence_margin=args.equivalence_margin,
    )

    print("\n=== 1. False-Prune Rate: Unguarded Baseline vs. Coverage Gate ===")
    print(
        tabulate(
            [
                [
                    r["tau_targeted"],
                    r["contrast_sd"],
                    r["trigger_p"],
                    f"{r['false_prune_no_gate']:.1%}",
                    f"{r['false_prune_with_gate']:.1%}",
                    f"{r['keep_rate_with_gate']:.1%}",
                ]
                for r in fp_rows
            ],
            headers=[
                "tau_T",
                "Contrast SD",
                "p",
                "FP (No Gate)",
                "FP (Gated)",
                "Keep (Gated)",
            ],
            tablefmt="github",
        )
    )

    print("\n=== 2. Dead-Clause Pruning Power across (n_T, targeted_margin, SD) ===")
    print(
        tabulate(
            [
                [
                    r["contrast_sd"],
                    r["n_per_factor_discovery"],
                    r["targeted_margin"],
                    f"{r['targeted_ni_power']:.1%}",
                    f"{r['baseline_tost_power']:.1%}",
                    f"{r['joint_prune_power']:.1%}",
                    f"{r['real_rule_false_prune_rate']:.1%}",
                ]
                for r in power_rows
            ],
            headers=[
                "Contrast SD",
                "n_T",
                "targeted_margin",
                "NI Power (T)",
                "TOST Power (F)",
                "Joint Prune Power",
                "FP (Real Rule)",
            ],
            tablefmt="github",
        )
    )

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps(
                {
                    "reps": args.reps,
                    "equivalence_margin": args.equivalence_margin,
                    "false_prune_comparison": fp_rows,
                    "dead_clause_power_grid": power_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved calibration results to {out_p}")


if __name__ == "__main__":
    main()
