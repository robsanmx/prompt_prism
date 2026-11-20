"""
Mixed-Level Categorical Axis Audit, Alphabetical Optimal Designs (D-/A-Optimal),
and PromptEval Rasch/Covariate Comparators on PromptEval MMLU (Phase 0 Task R2).

Treats the 100-template FormatSpread MMLU catalog directly as an incomplete
4 x 9 x 11 = 396 mixed-level factorial grid over `op` (4 levels), `space` (9 levels),
and `sep` (11 levels), rather than only as five post-hoc binary contrasts.

Key components:
1. Source integrity verification via `verify_source_hashes()` and complete catalog audit
   across 100 catalog rows, 93 unique outer template strings, and 89 observed
   `(op, space, sep)` combinations (documenting 1-way and 2-way marginals, missing
   cells, and the number-theoretic + empirical proof that no textbook complete
   mixed-level orthogonal array OA(N, 4^1 9^1 11^1) with N < 396 exists or can be
   subsetted from the catalog).
2. Outcome-blind, rotationally invariant normalized Helmert categorical contrast bases:
   - `coarsened_main_13p` (13 params) and `coarsened_inter_15p` (15 params) at B = 16,
   - `additive_22p` (1 + 3 + 8 + 10 = 22 params, full rank = 22) at B in {24, 32},
   - `additive22_plus_inter_28p` (28 params, full additive + coarsened axis interactions) at B = 32,
   - `binary_5bit_16p` (16 params, 5-bit main + 2FI basis projected onto all 89 cells).
3. Strictly outcome-blind subset selection and multi-prompt estimation under identical
   operational discovery query budgets Q = B * N_disc:
   - Multi-start Woodbury Fedorov exchange for **D-optimal** (max log det(X_S^T X_S))
     and **A-optimal** (min tr((X_S^T X_S)^-1)) subsets using ONLY metadata X,
   - 200 seeded **well-conditioned random** full-rank subsets (with explicit kappa cap),
   - 200 seeded **full-rank random** subsets,
   - 200 seeded **unconditioned random** subsets (recording exact rank-deficiency rates),
   - **5-bit 2^(5-1)_V** half-fractions (at B = 16) and **full 32-cell binary cube** (at B = 32),
   - **PromptEval** estimators (Maia Polo et al., NeurIPS 2024): exact Schur-complement
     Newton 1PL Rasch + discrete FormatSpread covariate logistic model (`PE-discrete`)
     on B-template subsets and sparse B-per-item across-catalog schedules, plus one-hot
     1PL Rasch anchor model (`PE-Rasch`), strictly fitted on Q = B * N_disc discovery queries.
4. Evaluation across all 855 MMLU model-task cubes (Split 0 and 20 random 50/50 item splits)
   and 15 pooled MMLU models with 1,000-draw two-way model-task cluster bootstrap 95% CIs.
"""

from __future__ import annotations

import itertools
import json
import math
import pickle
import platform
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import qr

from research.hf_multi_prompt.run_experiments import (
    DATA_DIR,
    MMLU_MODELS,
    RESULTS_DIR,
    SEEDS,
    build_cube_design_matrices,
    build_mmlu_mapping_table,
    concatenate_task_discovery_items,
    enumerate_fractional_cosets,
    generate_item_splits,
    select_with_tie_policy,
    sha256_file,
    sha256_text,
    two_way_cluster_bootstrap_ci,
    verify_source_hashes,
)

MIXED_LEVEL_SEEDS: Dict[str, int] = {
    "fedorov_restarts": 20261118,
    "rand_fullrank_16": 20261161,
    "rand_wellcond_16": 20261162,
    "rand_uncond_16": 20261163,
    "rand_fullrank_24": 20261164,
    "rand_wellcond_24": 20261165,
    "rand_uncond_24": 20261166,
    "rand_fullrank_32": 20261167,
    "rand_wellcond_32": 20261168,
    "rand_uncond_32": 20261169,
    "pe_sparse_16": 20261171,
    "pe_sparse_24": 20261172,
    "pe_sparse_32": 20261173,
    "item_splits": SEEDS["item_splits"],
    "cluster_bootstrap": SEEDS["cluster_bootstrap"],
    "tie_break": SEEDS["tie_break"],
}

WELL_CONDITIONED_CAPS: Dict[Tuple[str, int], float] = {
    ("coarsened_main_13p", 16): 6.5,
    ("coarsened_inter_15p", 16): 12.0,
    ("additive_22p", 24): 11.5,
    ("additive_22p", 32): 5.5,
    ("additive22_plus_inter_28p", 32): 11.5,
}


# ==============================================================================
# 1. Mixed-Level Catalog Construction and Completeness / OA Non-Existence Audit
# ==============================================================================

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _lcm_many(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out = _lcm(out, int(v))
    return out


def map_space_coarse5(sp: str) -> str:
    """Deterministic outcome-blind grouping of 9 FormatSpread `space` levels into 5 structural classes."""
    if sp == "":
        return "0_empty"
    if sp in (" ", "  "):
        return "1_inline_ws"
    if sp in (", ", " , "):
        return "2_inline_comma"
    if sp in ("\n", "\n ", " \n"):
        return "3_newline_ws"
    if sp == "; \n":
        return "4_semicolon_newline"
    raise ValueError(f"Unknown FormatSpread space level: {sp!r}")


def map_space_coarse3(sp: str) -> str:
    """Deterministic outcome-blind grouping of 9 `space` levels into 3 layout classes."""
    if "\n" in sp:
        return "2_newline"
    if "," in sp:
        return "1_inline_comma"
    return "0_inline_ws"


def map_sep_coarse6(sep: str) -> str:
    """Deterministic outcome-blind grouping of 11 FormatSpread `sep` levels into 6 delimiter classes."""
    if sep in (":", "."):
        return "0_colon_period"
    if sep in ("'", '"'):
        return "1_quotes"
    if sep in ("-", "!"):
        return "2_dash_bang"
    if sep == "<sep>":
        return "3_xml_tag"
    if sep in ("|", "]"):
        return "4_pipe_bracket"
    if sep in ("/", "\\"):
        return "5_slashes"
    raise ValueError(f"Unknown FormatSpread sep level: {sep!r}")


def map_sep_coarse4(sep: str) -> str:
    """Deterministic 4-class delimiter partition matching the (D, E) 2-bit split on `sep`."""
    if sep in (":", "."):
        return "0_clause_punct"
    if sep in ("'", '"', "-", "!"):
        return "1_quote_dash_bang"
    if sep in ("<sep>", "|", "]"):
        return "2_tag_pipe_bracket"
    if sep in ("/", "\\"):
        return "3_slashes"
    raise ValueError(f"Unknown FormatSpread sep level: {sep!r}")


def load_mmlu_catalog_tables(
    data_dir: Path = DATA_DIR,
) -> Dict[str, Any]:
    """
    Load and cross-verify the 100-row MMLU template catalog, 93 unique outer template strings,
    and 89 observed `(op, space, sep)` combinations.
    """
    meta_path = data_dir / "templates" / "mmlu_templates_metadata.json"
    meta = json.loads(meta_path.read_text())
    csv_paths = sorted((data_dir / "templates" / "MMLU").glob("*.csv"))
    if len(csv_paths) != 57:
        raise RuntimeError(f"Expected 57 MMLU task CSVs, found {len(csv_paths)}")

    ref_df = pd.read_csv(csv_paths[0])
    ref_templates = ref_df["prompt template"].astype(str).tolist()
    all_csvs_identical = True
    for p in csv_paths[1:]:
        cur_templates = pd.read_csv(p)["prompt template"].astype(str).tolist()
        if cur_templates != ref_templates:
            all_csvs_identical = False
            break
    if not all_csvs_identical:
        raise RuntimeError("MMLU subject CSVs do not share identical 100-row template ordering")

    df_map, leg_map, can_map = build_mmlu_mapping_table(ref_df, meta)

    rows_100: List[Dict[str, Any]] = []
    for idx, r in ref_df.iterrows():
        tmpl = str(r["prompt template"])
        m = meta[tmpl]
        op, sp, sep = str(m["op"]), str(m["space"]), str(m["sep"])
        A = 1 if op in ("lambda x: x", "lambda x: x.title()") else 0
        B = 1 if op in ("lambda x: x.upper()", "lambda x: x.title()") else 0
        C = 1 if "\n" in sp else 0
        D = 1 if sep in (":", ".", "'", '"', "-", "!") else 0
        E = 1 if sep in (":", ".", "<sep>", "|", "]") else 0
        cell_tuple = (A, B, C, D, E)
        rows_100.append(
            {
                "matrix_row_index": int(idx),
                "template": tmpl,
                "template_sha256": sha256_text(tmpl),
                "op": op,
                "space": sp,
                "space_repr": repr(sp),
                "sep": sep,
                "space_coarse5": map_space_coarse5(sp),
                "space_coarse3": map_space_coarse3(sp),
                "sep_coarse6": map_sep_coarse6(sep),
                "sep_coarse4": map_sep_coarse4(sep),
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "E": E,
                "binary_cell_tuple": cell_tuple,
                "binary_cell_id": "".join(str(x) for x in cell_tuple),
            }
        )
    df_100 = pd.DataFrame(rows_100)

    # Deduplicated 93 unique outer template strings (ordered canonically by sha256, then row index)
    df_93 = (
        df_100.sort_values(["template_sha256", "matrix_row_index"])
        .drop_duplicates(subset=["template_sha256"], keep="first")
        .sort_values(["op", "space", "sep", "template_sha256", "matrix_row_index"])
        .reset_index(drop=True)
    )
    df_93["unique_template_index"] = np.arange(len(df_93), dtype=int)

    # 89 unique observed (op, space, sep) combinations, ordered canonically by (op, space, sep)
    # Representative single template (`rep_row_index`) is the lowest (template_sha256, matrix_row_index)
    # within that combination, which is strictly invariant to catalog row permutations AND guarantees
    # that all 32 `canonical_first` binary-cube representatives are an exact subset of `df_89`!
    combo_records: List[Dict[str, Any]] = []
    sorted_for_89 = df_100.sort_values(
        ["op", "space", "sep", "template_sha256", "matrix_row_index"]
    )
    for combo_idx, ((op, sp, sep), grp) in enumerate(
        sorted_for_89.groupby(["op", "space", "sep"], sort=True)
    ):
        first = grp.iloc[0]
        all_rows = grp.sort_values("matrix_row_index")["matrix_row_index"].astype(int).tolist()
        uniq_rows = (
            grp.drop_duplicates(subset=["template_sha256"], keep="first")["matrix_row_index"]
            .astype(int)
            .tolist()
        )
        combo_records.append(
            {
                "combo_index": int(combo_idx),
                "op": str(op),
                "space": str(sp),
                "space_repr": repr(str(sp)),
                "sep": str(sep),
                "space_coarse5": map_space_coarse5(str(sp)),
                "space_coarse3": map_space_coarse3(str(sp)),
                "sep_coarse6": map_sep_coarse6(str(sep)),
                "sep_coarse4": map_sep_coarse4(str(sep)),
                "A": int(first["A"]),
                "B": int(first["B"]),
                "C": int(first["C"]),
                "D": int(first["D"]),
                "E": int(first["E"]),
                "binary_cell_tuple": first["binary_cell_tuple"],
                "binary_cell_id": str(first["binary_cell_id"]),
                "rep_row_index": int(first["matrix_row_index"]),
                "rep_template_sha256": str(first["template_sha256"]),
                "all_row_indices": all_rows,
                "unique_row_indices": uniq_rows,
                "n_catalog_rows": int(len(all_rows)),
                "n_unique_templates": int(len(uniq_rows)),
            }
        )
    df_89 = pd.DataFrame(combo_records)

    return {
        "df_100": df_100,
        "df_93": df_93,
        "df_89": df_89,
        "df_binary_map": df_map,
        "legacy_binary_map": leg_map,
        "canonical_binary_map": can_map,
        "all_57_csvs_identical": all_csvs_identical,
    }


def audit_mmlu_mixed_level_catalog(
    catalog_bundle: Dict[str, Any],
    Ys: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform a complete structural, combinatorial, and number-theoretic audit of the
    MMLU FormatSpread 4 x 9 x 11 mixed-level template catalog.
    """
    df_100: pd.DataFrame = catalog_bundle["df_100"]
    df_93: pd.DataFrame = catalog_bundle["df_93"]
    df_89: pd.DataFrame = catalog_bundle["df_89"]

    ops = sorted(df_89["op"].unique().tolist())
    spaces = sorted(df_89["space"].unique().tolist())
    seps = sorted(df_89["sep"].unique().tolist())

    n_op, n_space, n_sep = len(ops), len(spaces), len(seps)
    full_grid_size = n_op * n_space * n_sep
    if (n_op, n_space, n_sep, full_grid_size) != (4, 9, 11, 396):
        raise RuntimeError(
            f"Unexpected mixed-level dimensions: {(n_op, n_space, n_sep)} != (4, 9, 11)"
        )
    if len(df_100) != 100 or len(df_93) != 93 or len(df_89) != 89:
        raise RuntimeError(
            f"Unexpected catalog counts: rows={len(df_100)}, uniq={len(df_93)}, combos={len(df_89)}"
        )

    observed_3way = set(zip(df_89["op"], df_89["space"], df_89["sep"]))
    all_3way = list(itertools.product(ops, spaces, seps))
    missing_3way = [
        {"op": o, "space_repr": repr(sp), "sep": s}
        for (o, sp, s) in all_3way
        if (o, sp, s) not in observed_3way
    ]

    # 2-way projections
    obs_op_sp = set(zip(df_89["op"], df_89["space"]))
    miss_op_sp = [
        {"op": o, "space_repr": repr(sp)}
        for o, sp in itertools.product(ops, spaces)
        if (o, sp) not in obs_op_sp
    ]
    obs_op_sep = set(zip(df_89["op"], df_89["sep"]))
    miss_op_sep = [
        {"op": o, "sep": s}
        for o, s in itertools.product(ops, seps)
        if (o, s) not in obs_op_sep
    ]
    obs_sp_sep = set(zip(df_89["space"], df_89["sep"]))
    miss_sp_sep = [
        {"space_repr": repr(sp), "sep": s}
        for sp, s in itertools.product(spaces, seps)
        if (sp, s) not in obs_sp_sep
    ]

    def _marginals(df_sub: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        return {
            "op": {str(k): int(v) for k, v in df_sub["op"].value_counts().sort_index().items()},
            "space_repr": {
                str(k): int(v)
                for k, v in df_sub["space_repr"].value_counts().sort_index().items()
            },
            "sep": {str(k): int(v) for k, v in df_sub["sep"].value_counts().sort_index().items()},
        }

    # Audit duplicate outer template strings and multi-row combinations
    dup_groups = (
        df_100[df_100.duplicated("template_sha256", keep=False)]
        .groupby("template_sha256")["matrix_row_index"]
        .apply(lambda s: sorted(int(x) for x in s))
        .tolist()
    )
    dup_groups.sort()

    dup_outcome_diff_cubes = None
    if Ys is not None and "MMLU" in Ys:
        differing_pairs = 0
        for grp_rows in dup_groups:
            r0, r1 = grp_rows[0], grp_rows[1]
            has_diff = any(
                not np.array_equal(Ys["MMLU"][t][m_i][r0], Ys["MMLU"][t][m_i][r1])
                for t in Ys["MMLU"]
                for m_i in range(len(MMLU_MODELS))
            )
            if has_diff:
                differing_pairs += 1
        dup_outcome_diff_cubes = {
            "n_duplicate_outer_string_pairs": len(dup_groups),
            "n_pairs_with_distinct_item_outcomes_in_Ys": differing_pairs,
            "explanation": (
                "Outer prompt template strings in abstract_algebra.csv collapse `{choices}` "
                "formatting; FormatSpread also varies internal choice-enumeration formatting, "
                "so rows with identical outer strings still represent distinct prompt executions in Ys.pickle."
            ),
        }

    multi_row_combos = [
        {
            "combo_index": int(r["combo_index"]),
            "op": str(r["op"]),
            "space_repr": str(r["space_repr"]),
            "sep": str(r["sep"]),
            "n_catalog_rows": int(r["n_catalog_rows"]),
            "n_unique_templates": int(r["n_unique_templates"]),
            "all_row_indices": list(r["all_row_indices"]),
        }
        for _, r in df_89[df_89["n_catalog_rows"] > 1].iterrows()
    ]

    # Mathematical proof of OA(N, 4^1 9^1 11^1) non-existence for N < 396
    lcm_t1 = _lcm_many([n_op, n_space, n_sep])
    lcm_t2 = _lcm_many([n_op * n_space, n_op * n_sep, n_space * n_sep])

    return {
        "axes": {
            "op_levels": ops,
            "space_levels_repr": [repr(s) for s in spaces],
            "sep_levels": seps,
            "n_op": n_op,
            "n_space": n_space,
            "n_sep": n_sep,
        },
        "counts": {
            "full_cartesian_grid": full_grid_size,
            "catalog_rows": int(len(df_100)),
            "unique_outer_template_strings": int(len(df_93)),
            "observed_axis_combinations": int(len(df_89)),
            "missing_axis_combinations": int(full_grid_size - len(df_89)),
            "observed_fraction": float(len(df_89) / full_grid_size),
            "missing_fraction": float((full_grid_size - len(df_89)) / full_grid_size),
        },
        "pairwise_coverage": {
            "op_x_space_observed": int(len(obs_op_sp)),
            "op_x_space_total": int(n_op * n_space),
            "op_x_space_missing_pairs": miss_op_sp,
            "op_x_sep_observed": int(len(obs_op_sep)),
            "op_x_sep_total": int(n_op * n_sep),
            "op_x_sep_missing_pairs": miss_op_sep,
            "space_x_sep_observed": int(len(obs_sp_sep)),
            "space_x_sep_total": int(n_space * n_sep),
            "space_x_sep_missing_count": int(len(miss_sp_sep)),
            "space_x_sep_missing_pairs": miss_sp_sep,
        },
        "marginal_counts": {
            "catalog_100_rows": _marginals(df_100),
            "unique_93_templates": _marginals(df_93),
            "observed_89_combinations": _marginals(df_89),
        },
        "duplicate_and_multi_row_audit": {
            "duplicate_outer_template_row_pairs": dup_groups,
            "duplicate_outcome_check": dup_outcome_diff_cubes,
            "multi_row_axis_combinations": multi_row_combos,
        },
        "orthogonal_array_impossibility_proof": {
            "pairwise_coprime_levels": bool(
                math.gcd(4, 9) == 1 and math.gcd(4, 11) == 1 and math.gcd(9, 11) == 1
            ),
            "strength_1_min_balanced_runs_lcm_4_9_11": int(lcm_t1),
            "strength_2_min_oa_runs_lcm_36_44_99": int(lcm_t2),
            "min_observed_marginal_count_89": int(
                min(df_89["sep"].value_counts().min(), df_89["space"].value_counts().min(), df_89["op"].value_counts().min())
            ),
            "summary_reason": (
                "Because 4, 9, and 11 are pairwise coprime, any strength-2 orthogonal array "
                "OA(N, 4^1 9^1 11^1) requires lcm(4*9, 4*11, 9*11) = lcm(36, 44, 99) = 396 | N "
                "(and even strength-1 equal marginal replication requires lcm(4, 9, 11) = 396 | N). "
                "No fractional OA with N < 396 exists mathematically, and the PromptEval MMLU catalog "
                "observes only 89 of 396 cells (missing 307 three-way cells, 2/36 op*space pairs, "
                "3/44 op*sep pairs, and 31/99 space*sep pairs, with sep=']' observed only 5 times)."
            ),
        },
        "missing_3way_combinations_count": len(missing_3way),
    }


# ==============================================================================
# 2. Normalized Helmert Categorical Contrast Parameterizations & Rank Safeguards
# ==============================================================================

def helmert_contrast_matrix(n_levels: int) -> np.ndarray:
    """
    Construct an (L, L-1) zero-sum orthogonal Helmert contrast matrix H satisfying:
      1^T H = 0   and   (1 / L) H^T H = I_{L-1}.
    On a complete Cartesian product grid, appending an intercept column yields
    (1 / N_grid) X^T X = I_p (condition number 1.0), and makes D-optimality,
    A-optimality, condition number, and Euclidean contrast RMSE strictly invariant
    to rotational contrast reparameterizations within each factor.
    """
    if n_levels < 2:
        raise ValueError(f"At least 2 levels required for Helmert contrasts, got {n_levels}")
    H = np.zeros((n_levels, n_levels - 1), dtype=float)
    for j in range(n_levels - 1):
        H[: j + 1, j] = -1.0
        H[j + 1, j] = float(j + 1)
        H[:, j] *= math.sqrt(n_levels / float(np.sum(H[:, j] ** 2)))
    return H


def validate_mixed_level_design(
    X_sub: np.ndarray,
    design_id: str,
    max_condition_number: Optional[float] = None,
) -> Tuple[int, float]:
    """Validate full column rank and optional condition-number bound before OLS fitting."""
    if X_sub.ndim != 2:
        raise ValueError(f"Design {design_id!r} must be 2D, got shape {X_sub.shape}")
    n_runs, n_params = X_sub.shape
    if n_runs < n_params:
        raise ValueError(
            f"Design {design_id!r} has fewer runs ({n_runs}) than parameters ({n_params})"
        )
    rk = int(np.linalg.matrix_rank(X_sub))
    if rk < n_params:
        raise ValueError(
            f"Design {design_id!r} is rank deficient (aliased): rank={rk}, required={n_params}"
        )
    cond = float(np.linalg.cond(X_sub))
    if max_condition_number is not None and cond > max_condition_number:
        raise ValueError(
            f"Design {design_id!r} exceeds maximum condition number: {cond:.4f} > {max_condition_number:.4f}"
        )
    return rk, cond


def fit_mixed_level_ols(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    design_id: str,
    max_condition_number: Optional[float] = None,
) -> np.ndarray:
    """Fit mixed-level categorical OLS only after explicit full-rank validation."""
    validate_mixed_level_design(X_sub, design_id=design_id, max_condition_number=max_condition_number)
    beta, *_ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
    return beta


def build_additive_mixed_level_matrix(
    df_axes: pd.DataFrame,
    factor_cols: Sequence[str] = ("op", "space", "sep"),
    level_orders: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build `[1, H_{F1}, H_{F2}, ...]` additive normalized Helmert design matrix for arbitrary
    categorical factor columns. Purely a function of categorical metadata (strictly outcome-blind).
    """
    n_rows = len(df_axes)
    blocks = [np.ones((n_rows, 1), dtype=float)]
    col_slices: Dict[str, Tuple[int, int]] = {"intercept": (0, 1)}
    orders_used: Dict[str, List[Any]] = {}
    cur = 1

    for col in factor_cols:
        levels = (
            list(level_orders[col])
            if (level_orders is not None and col in level_orders)
            else sorted(df_axes[col].unique().tolist())
        )
        orders_used[col] = levels
        idx_map = {lv: i for i, lv in enumerate(levels)}
        try:
            codes = [idx_map[val] for val in df_axes[col].tolist()]
        except KeyError as exc:
            raise ValueError(f"Unseen level {exc} in column {col!r}") from exc
        H = helmert_contrast_matrix(len(levels))
        blocks.append(H[codes])
        col_slices[col] = (cur, cur + len(levels) - 1)
        cur += len(levels) - 1

    X = np.hstack(blocks)
    return X, {
        "n_rows": n_rows,
        "n_parameters": int(X.shape[1]),
        "factor_cols": list(factor_cols),
        "column_slices": col_slices,
        "level_orders": orders_used,
    }


def build_all_catalog_bases(
    df_cat: pd.DataFrame,
    catalog_label: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Construct all planned outcome-blind categorical and binary contrast bases on `df_cat`
    (applicable to `df_89`, `df_93`, and `df_100`).
    """
    n = len(df_cat)

    # 1. Full 22-parameter additive basis: 1 + op(3) + space(8) + sep(10) = 22
    X_22, info_22 = build_additive_mixed_level_matrix(df_cat, ("op", "space", "sep"))
    rk_22, cond_22 = validate_mixed_level_design(X_22, f"{catalog_label}_additive_22p")

    # 2. Coarsened 13-parameter additive basis for B=16: 1 + op(3) + space_coarse5(4) + sep_coarse6(5) = 13
    X_13, info_13 = build_additive_mixed_level_matrix(
        df_cat, ("op", "space_coarse5", "sep_coarse6")
    )
    rk_13, cond_13 = validate_mixed_level_design(X_13, f"{catalog_label}_coarsened_main_13p")

    # 3. Coarsened 15-parameter axis-interaction basis for B=16:
    #    1 + op(3) + space_coarse3(2) + sep_coarse4(3) + op*newline(3) + sep_coarse4*newline(3) = 15
    X_c9, info_c9 = build_additive_mixed_level_matrix(
        df_cat, ("op", "space_coarse3", "sep_coarse4")
    )
    nl_sign = np.where(df_cat["C"].to_numpy(dtype=int) == 1, 1.0, -1.0)[:, None]
    s_op = info_c9["column_slices"]["op"]
    s_sep4 = info_c9["column_slices"]["sep_coarse4"]
    X_op_block = X_c9[:, s_op[0] : s_op[1]]
    X_sep4_block = X_c9[:, s_sep4[0] : s_sep4[1]]
    X_15 = np.hstack([X_c9, X_op_block * nl_sign, X_sep4_block * nl_sign])
    rk_15, cond_15 = validate_mixed_level_design(X_15, f"{catalog_label}_coarsened_inter_15p")
    slices_15 = dict(info_c9["column_slices"])
    slices_15["op_x_newline"] = (9, 12)
    slices_15["sep4_x_newline"] = (12, 15)

    # 4. Full 22-parameter additive + 6 coarsened cross-axis interactions = 28 parameters (for B=32)
    X_28 = np.hstack([X_22, X_op_block * nl_sign, X_sep4_block * nl_sign])
    rk_28, cond_28 = validate_mixed_level_design(
        X_28, f"{catalog_label}_additive22_plus_inter_28p"
    )
    slices_28 = dict(info_22["column_slices"])
    slices_28["op_x_newline"] = (22, 25)
    slices_28["sep4_x_newline"] = (25, 28)

    # 5. 5-bit binary [1, X_main, X_2FI] 16-parameter basis evaluated on every catalog entry
    X_01 = df_cat[["A", "B", "C", "D", "E"]].to_numpy(dtype=int)
    X_pm = (2 * X_01 - 1).astype(float)
    pairs = list(itertools.combinations(range(5), 2))
    X_2fi = np.column_stack([X_pm[:, i] * X_pm[:, j] for i, j in pairs])
    X_5bit_16 = np.column_stack([np.ones(n, dtype=float), X_pm, X_2fi])
    rk_5bit, cond_5bit = validate_mixed_level_design(
        X_5bit_16, f"{catalog_label}_binary_5bit_16p"
    )

    # Quantify subspace projection: fraction of 21-df additive mixed-level variance spanned by 5-bit 16p basis
    P_5bit = X_5bit_16 @ np.linalg.pinv(X_5bit_16)
    X_22_nonint = X_22[:, 1:]
    proj_fro_sq = float(np.sum((P_5bit @ X_22_nonint) ** 2))
    tot_fro_sq = float(np.sum(X_22_nonint ** 2))

    return {
        "coarsened_main_13p": {
            "X": X_13,
            "n_parameters": 13,
            "rank": rk_13,
            "condition_number": cond_13,
            "column_slices": info_13["column_slices"],
            "description": "Intercept + op(3) + space_coarse5(4) + sep_coarse6(5)",
        },
        "coarsened_inter_15p": {
            "X": X_15,
            "n_parameters": 15,
            "rank": rk_15,
            "condition_number": cond_15,
            "column_slices": slices_15,
            "description": "Intercept + op(3) + space_coarse3(2) + sep_coarse4(3) + op*newline(3) + sep4*newline(3)",
        },
        "binary_5bit_16p": {
            "X": X_5bit_16,
            "n_parameters": 16,
            "rank": rk_5bit,
            "condition_number": cond_5bit,
            "column_slices": {"intercept": (0, 1), "main_5bit": (1, 6), "two_fi_5bit": (6, 16)},
            "additive_21df_energy_captured_fraction": float(proj_fro_sq / tot_fro_sq),
            "description": "5-bit binary [1, main(5), 2FI(10)] basis (8 additive df + 8 partial cross-axis products)",
        },
        "additive_22p": {
            "X": X_22,
            "n_parameters": 22,
            "rank": rk_22,
            "condition_number": cond_22,
            "column_slices": info_22["column_slices"],
            "description": "Full mixed-level additive basis: Intercept + op(3) + space(8) + sep(10)",
        },
        "additive22_plus_inter_28p": {
            "X": X_28,
            "n_parameters": 28,
            "rank": rk_28,
            "condition_number": cond_28,
            "column_slices": slices_28,
            "description": "Full 22p additive basis + op*newline(3) + sep4*newline(3) interactions",
        },
    }


# ==============================================================================
# 3. Outcome-Blind Subset Selection: Woodbury Fedorov D-/A-Optimal & Random Draws
# ==============================================================================

def compute_design_criteria(X_sub: np.ndarray) -> Dict[str, Any]:
    """Compute rank, condition number, log det(X_S^T X_S), and tr((X_S^T X_S)^-1) from X_sub ONLY."""
    n_runs, n_params = X_sub.shape
    rk = int(np.linalg.matrix_rank(X_sub))
    is_full = bool(rk == n_params and n_runs >= n_params)
    if not is_full:
        return {
            "n_runs": int(n_runs),
            "n_parameters": int(n_params),
            "rank": rk,
            "is_full_rank": False,
            "condition_number": float("inf"),
            "logdet_xtx": float("-inf"),
            "trace_inv_xtx": float("inf"),
        }
    M = X_sub.T @ X_sub
    cond = float(np.linalg.cond(X_sub))
    sign, logdet = np.linalg.slogdet(M)
    tr_inv = float(np.trace(np.linalg.inv(M)))
    return {
        "n_runs": int(n_runs),
        "n_parameters": int(n_params),
        "rank": rk,
        "is_full_rank": True,
        "condition_number": cond,
        "logdet_xtx": float(logdet) if sign > 0 else float("-inf"),
        "trace_inv_xtx": tr_inv,
    }


def select_optimal_subset_fedorov(
    X: np.ndarray,
    subset_size: int,
    criterion: str = "D",
    n_restarts: int = 100,
    seed: int = MIXED_LEVEL_SEEDS["fedorov_restarts"],
    extra_starts: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Strictly outcome-blind multi-start Fedorov single-point exchange algorithm using
    exact rank-2 Sherman-Morrison-Woodbury determinant and trace updates.

    - `criterion="D"`: maximizes `log det(X_S^T X_S)`
    - `criterion="A"`: minimizes `tr((X_S^T X_S)^-1)`

    Depends ONLY on the design matrix `X` (zero outcome input).
    """
    if criterion not in ("D", "A"):
        raise ValueError(f"Unsupported optimality criterion: {criterion!r}")
    N, p = X.shape
    if subset_size < p or subset_size > N:
        raise ValueError(
            f"Invalid subset_size={subset_size} for X of shape {X.shape} (requires {p} <= B <= {N})"
        )
    if int(np.linalg.matrix_rank(X)) < p:
        raise ValueError(f"Candidate pool X of shape {X.shape} is itself rank deficient")

    rng = np.random.default_rng(seed)

    # Start 0 & Start 1: Deterministic QR column-pivoting + greedy D- and A-augmentation
    # Including both deterministic greedy initializations guarantees A-optimal never
    # lands in a worse trace basin than the deterministic D-start (and vice versa),
    # while remaining strictly a function of X alone (100% outcome-blind).
    _, _, piv = qr(X.T, pivoting=True)
    base_piv = [int(x) for x in piv[:p]]

    starts: List[List[int]] = []
    seen_starts = set()
    for init_crit in ("D", "A"):
        init_s = list(base_piv)
        rem = [i for i in range(N) if i not in init_s]
        while len(init_s) < subset_size:
            M = X[init_s].T @ X[init_s]
            V = np.linalg.inv(M)
            Xr = X[rem]
            d_out = np.sum((Xr @ V) * Xr, axis=1)
            if init_crit == "D":
                j_best = rem[int(np.argmax(1.0 + d_out))]
            else:
                V2 = V @ V
                h_out = np.sum((Xr @ V2) * Xr, axis=1)
                j_best = rem[int(np.argmin(-h_out / (1.0 + d_out)))]
            init_s.append(int(j_best))
            rem.remove(int(j_best))
        s_sorted = sorted(init_s)
        key = tuple(s_sorted)
        if key not in seen_starts:
            seen_starts.add(key)
            starts.append(s_sorted)

    attempts = 0
    max_attempts = max(5000, 250 * n_restarts)
    target_random = n_restarts + len(starts)
    while len(starts) < target_random and attempts < max_attempts:
        attempts += 1
        cand = sorted(int(x) for x in rng.choice(N, size=subset_size, replace=False))
        key = tuple(cand)
        if key in seen_starts:
            continue
        if int(np.linalg.matrix_rank(X[cand])) == p:
            seen_starts.add(key)
            starts.append(cand)

    if extra_starts is not None:
        for ext in extra_starts:
            s_ext = sorted(int(x) for x in ext)
            key = tuple(s_ext)
            if key not in seen_starts and int(np.linalg.matrix_rank(X[s_ext])) == p:
                seen_starts.add(key)
                starts.append(s_ext)

    best_s: Optional[List[int]] = None
    best_val = float("-inf") if criterion == "D" else float("inf")

    for s0 in starts:
        S = list(s0)
        while True:
            in_mask = np.zeros(N, dtype=bool)
            in_mask[S] = True
            out = np.flatnonzero(~in_mask).tolist()
            if not out:
                break

            Xs = X[S]
            Xo = X[out]
            M = Xs.T @ Xs
            V = np.linalg.inv(M)
            G_io = (Xs @ V) @ Xo.T                       # (B, N-B)
            d_in = np.sum((Xs @ V) * Xs, axis=1)         # (B,)
            d_out = np.sum((Xo @ V) * Xo, axis=1)        # (N-B,)
            delta_D = (1.0 + d_out[None, :]) * (1.0 - d_in[:, None]) + G_io ** 2

            if criterion == "D":
                idx_flat = int(np.argmax(delta_D))
                r_i, c_j = divmod(idx_flat, len(out))
                if delta_D[r_i, c_j] <= 1.0 + 1e-10:
                    break
            else:
                V2 = V @ V
                H_io = (Xs @ V2) @ Xo.T
                h_in = np.sum((Xs @ V2) * Xs, axis=1)
                h_out = np.sum((Xo @ V2) * Xo, axis=1)
                valid = delta_D > 1e-10
                num = (
                    (1.0 + d_out[None, :]) * h_in[:, None]
                    - (1.0 - d_in[:, None]) * h_out[None, :]
                    - 2.0 * G_io * H_io
                )
                delta_A = np.where(valid, num / np.maximum(delta_D, 1e-15), np.inf)
                idx_flat = int(np.argmin(delta_A))
                r_i, c_j = divmod(idx_flat, len(out))
                if delta_A[r_i, c_j] >= -1e-10:
                    break

            S[r_i] = int(out[c_j])
            S.sort()

        crit = compute_design_criteria(X[S])
        val = crit["logdet_xtx"] if criterion == "D" else crit["trace_inv_xtx"]
        if criterion == "D":
            if val > best_val + 1e-10 or (
                abs(val - best_val) <= 1e-10 and (best_s is None or S < best_s)
            ):
                best_val = val
                best_s = list(S)
        else:
            if val < best_val - 1e-10 or (
                abs(val - best_val) <= 1e-10 and (best_s is None or S < best_s)
            ):
                best_val = val
                best_s = list(S)

    assert best_s is not None
    final_crit = compute_design_criteria(X[best_s])
    return {
        "indices": best_s,
        "criterion": criterion,
        "n_restarts_used": len(starts),
        "seed": seed,
        **final_crit,
    }


def select_paired_d_and_a_optimal(
    X: np.ndarray,
    subset_size: int,
    n_restarts: int = 100,
    seed: int = MIXED_LEVEL_SEEDS["fedorov_restarts"],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Outcome-blind paired D-optimal and A-optimal Fedorov optimization with cross-polishing
    between the two criteria's optima (using ONLY `X`), guaranteeing `logdet(D) >= logdet(A)`
    and `trace_inv(A) <= trace_inv(D)`.
    """
    d_init = select_optimal_subset_fedorov(X, subset_size, "D", n_restarts=n_restarts, seed=seed)
    a_init = select_optimal_subset_fedorov(
        X, subset_size, "A", n_restarts=n_restarts, seed=seed, extra_starts=[d_init["indices"]]
    )
    d_final = select_optimal_subset_fedorov(
        X, subset_size, "D", n_restarts=0, seed=seed, extra_starts=[d_init["indices"], a_init["indices"]]
    )
    d_final["n_restarts_used"] = d_init["n_restarts_used"]
    a_final = select_optimal_subset_fedorov(
        X, subset_size, "A", n_restarts=0, seed=seed, extra_starts=[a_init["indices"], d_final["indices"]]
    )
    a_final["n_restarts_used"] = a_init["n_restarts_used"]
    return d_final, a_final


def generate_mixed_level_random_subsets(
    X: np.ndarray,
    subset_size: int,
    num_subsets: int = 200,
    seed: int = 20261161,
    require_full_rank: bool = True,
    max_condition_number: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Outcome-blind random subset generator over a catalog design matrix `X`:
    - `require_full_rank=False, max_condition_number=None`: unconditioned uniform random draws
    - `require_full_rank=True, max_condition_number=None`: full-rank conditioned draws
    - `require_full_rank=True, max_condition_number=cap`: well-conditioned full-rank draws
    """
    N, p = X.shape
    rng = np.random.default_rng(seed)
    accepted: List[List[int]] = []
    seen = set()

    total_unique_checked = 0
    rejected_rank_def = 0
    rejected_ill_cond = 0
    accepted_full_rank = 0
    conds: List[float] = []
    logdets: List[float] = []
    tr_invs: List[float] = []
    ranks: List[int] = []

    max_draws = 500000
    draws = 0
    while len(accepted) < num_subsets and draws < max_draws:
        draws += 1
        cand = tuple(sorted(int(x) for x in rng.choice(N, size=subset_size, replace=False)))
        if cand in seen:
            continue
        seen.add(cand)
        total_unique_checked += 1

        crit = compute_design_criteria(X[list(cand)])
        if require_full_rank and not crit["is_full_rank"]:
            rejected_rank_def += 1
            continue
        if max_condition_number is not None and crit["condition_number"] > max_condition_number:
            rejected_ill_cond += 1
            continue

        accepted.append(list(cand))
        ranks.append(crit["rank"])
        if crit["is_full_rank"]:
            accepted_full_rank += 1
            conds.append(crit["condition_number"])
            logdets.append(crit["logdet_xtx"])
            tr_invs.append(crit["trace_inv_xtx"])

    if len(accepted) < num_subsets:
        raise RuntimeError(
            f"Could only generate {len(accepted)}/{num_subsets} subsets after {draws} draws "
            f"(B={subset_size}, p={p}, cap={max_condition_number})"
        )

    cond_arr = np.asarray(conds, dtype=float) if conds else np.array([float("inf")])
    ld_arr = np.asarray(logdets, dtype=float) if logdets else np.array([float("-inf")])
    tr_arr = np.asarray(tr_invs, dtype=float) if tr_invs else np.array([float("inf")])

    return {
        "subsets": accepted,
        "seed": seed,
        "subset_size": subset_size,
        "num_subsets": len(accepted),
        "total_draws": draws,
        "total_unique_checked": total_unique_checked,
        "rejected_rank_deficient": rejected_rank_def,
        "rejected_ill_conditioned": rejected_ill_cond,
        "empirical_full_rank_rate_among_checked": float(
            (total_unique_checked - rejected_rank_def) / max(1, total_unique_checked)
        ),
        "accepted_full_rank_rate": float(accepted_full_rank / len(accepted)),
        "max_condition_number_cap": max_condition_number,
        "rank_mean": float(np.mean(ranks)),
        "condition_number_mean": float(np.mean(cond_arr)) if accepted_full_rank == len(accepted) else float("inf"),
        "condition_number_median": float(np.median(cond_arr)) if accepted_full_rank == len(accepted) else float("inf"),
        "condition_number_min": float(np.min(cond_arr)) if accepted_full_rank == len(accepted) else float("inf"),
        "condition_number_max": float(np.max(cond_arr)) if accepted_full_rank == len(accepted) else float("inf"),
        "logdet_xtx_mean": float(np.mean(ld_arr)) if accepted_full_rank == len(accepted) else float("-inf"),
        "trace_inv_xtx_mean": float(np.mean(tr_arr)) if accepted_full_rank == len(accepted) else float("inf"),
    }


def extract_binary_5bit_subsets_on_89(
    catalog_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map the 32-cell 5-bit binary cube (`canonical_first` and `legacy_first`) and both
    16-run cosets of 2^(5-1)_V directly onto the 89 observed `(op, space, sep)` combinations
    and the 100-row catalog.
    """
    df_89: pd.DataFrame = catalog_bundle["df_89"]
    can_map: Dict[Tuple[int, ...], List[int]] = catalog_bundle["canonical_binary_map"]
    leg_map: Dict[Tuple[int, ...], List[int]] = catalog_bundle["legacy_binary_map"]

    all_cells = sorted(can_map.keys())
    rep_row_to_combo = {int(r["rep_row_index"]): int(r["combo_index"]) for _, r in df_89.iterrows()}
    row_to_combo = {}
    for _, r in df_89.iterrows():
        for m_row in r["all_row_indices"]:
            row_to_combo[int(m_row)] = int(r["combo_index"])

    can_cell_combo_idx = [rep_row_to_combo[can_map[c][0]] for c in all_cells]
    leg_cell_combo_idx = [row_to_combo[leg_map[c][0]] for c in all_cells]
    leg_cell_row_idx = [int(leg_map[c][0]) for c in all_cells]
    can_cell_row_idx = [int(can_map[c][0]) for c in all_cells]

    cosets = enumerate_fractional_cosets()
    pos16_cells = cosets["res5_cosets"]["2(5-1)V_coset_pos"]["indices"]
    neg16_cells = cosets["res5_cosets"]["2(5-1)V_coset_neg"]["indices"]

    return {
        "canonical_cube32_combo_indices": sorted(can_cell_combo_idx),
        "canonical_cube32_ordered_by_cell": can_cell_combo_idx,
        "canonical_cube32_row_indices": sorted(can_cell_row_idx),
        "canonical_res5_pos16_combo_indices": sorted(can_cell_combo_idx[i] for i in pos16_cells),
        "canonical_res5_neg16_combo_indices": sorted(can_cell_combo_idx[i] for i in neg16_cells),
        "legacy_cube32_combo_indices": sorted(leg_cell_combo_idx),
        "legacy_cube32_row_indices": sorted(leg_cell_row_idx),
    }


# ==============================================================================
# 4. PromptEval Estimators (Maia Polo et al., NeurIPS 2024) under Equal Budget
# ==============================================================================

def generate_prompteval_sparse_mask(
    n_templates: int,
    n_items: int,
    budget_per_item: int,
    seed: int,
) -> np.ndarray:
    """
    Generate an outcome-blind binary observation mask M in {0, 1}^{n_templates x n_items}
    with EXACT operational budget `M.sum() == budget_per_item * n_items`:
    - Every item column evaluates EXACTLY `budget_per_item` distinct templates (`M[:, i].sum() == B`).
    - Every template row is evaluated on `floor(B * n_items / U)` or `ceil(B * n_items / U)` items.
    """
    if not (1 <= budget_per_item <= n_templates):
        raise ValueError(
            f"budget_per_item={budget_per_item} must satisfy 1 <= B <= n_templates={n_templates}"
        )
    rng = np.random.default_rng(seed)
    perm_u = rng.permutation(n_templates)
    perm_i = rng.permutation(n_items)

    mask = np.zeros((n_templates, n_items), dtype=float)
    offsets = np.arange(budget_per_item, dtype=int)
    for t in range(n_items):
        col = int(perm_i[t])
        rows = perm_u[(t * budget_per_item + offsets) % n_templates]
        mask[rows, col] = 1.0
    return mask


def fit_prompteval_rasch_covariate(
    Y_obs: np.ndarray,
    mask_obs: Optional[np.ndarray],
    X_cov: np.ndarray,
    X_pred: Optional[np.ndarray] = None,
    lam_beta: float = 0.05,
    lam_delta: float = 0.05,
    max_iter: int = 12,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Fit Maia Polo et al. (2024) `PromptEval` 1PL Rasch / logistic covariate model:
      P(Y_{u, i} = 1) = sigmoid(X_{u, .} beta - delta_i)
    strictly on observed discovery entries (`mask_obs == 1`, or complete `Y_obs` when
    `mask_obs is None`) using exact block-Schur Newton-Raphson with backtracking safeguard.

    When `X_cov = I_U` (with `intercept_unpenalized=False`), this reduces to the one-hot
    1PL Rasch anchor model (`PE-Rasch`). When `X_cov` is a discrete FormatSpread feature
    matrix (`[1, contrasts]`), this implements `PE-discrete` and predicts both observed and
    unseen held-out templates via `X_pred`.
    """
    U, N = Y_obs.shape
    p = X_cov.shape[1]
    if X_cov.shape[0] != U:
        raise ValueError(f"Row mismatch between Y_obs ({U}) and X_cov ({X_cov.shape[0]})")
    if X_pred is None:
        X_pred = X_cov

    is_identity_cov = bool(p == U and np.allclose(X_cov, np.eye(U)))
    has_intercept = (not is_identity_cov) and bool(np.allclose(X_cov[:, 0], 1.0))
    pen_beta = np.full(p, float(lam_beta), dtype=float)
    if has_intercept:
        # Small anchor ridge fixes additive shift gauge between beta_0 and mean(delta)
        pen_beta[0] = float(lam_delta)

    beta = np.zeros(p, dtype=float)
    delta = np.zeros(N, dtype=float)

    if mask_obs is None:
        r_u = Y_obs.sum(axis=1)
        c_i = Y_obs.sum(axis=0)
    else:
        YM = Y_obs * mask_obs
        r_u = YM.sum(axis=1)
        c_i = YM.sum(axis=0)

    Xt_ru = r_u if is_identity_cov else (X_cov.T @ r_u)

    def _neg_loglik(b: np.ndarray, d: np.ndarray) -> float:
        theta = b if is_identity_cov else (X_cov @ b)
        eta = np.clip(theta[:, None] - d[None, :], -25.0, 25.0)
        softplus = np.logaddexp(0.0, eta)
        if mask_obs is None:
            nll = float(np.sum(softplus) - b @ Xt_ru + d @ c_i)
        else:
            nll = float(np.sum(softplus * mask_obs) - b @ Xt_ru + d @ c_i)
        reg = 0.5 * float(np.sum(pen_beta * (b ** 2)) + lam_delta * np.sum(d ** 2))
        return nll + reg

    cur_obj = _neg_loglik(beta, delta)
    n_iters = 0

    for it in range(max_iter):
        n_iters = it + 1
        theta = beta if is_identity_cov else (X_cov @ beta)
        eta = np.clip(theta[:, None] - delta[None, :], -25.0, 25.0)
        P_raw = 1.0 / (1.0 + np.exp(-eta))
        if mask_obs is None:
            P = P_raw
            W = P_raw * (1.0 - P_raw)
        else:
            P = P_raw * mask_obs
            W = P_raw * (1.0 - P_raw) * mask_obs

        diff_u = P.sum(axis=1) - r_u
        g_beta = (diff_u if is_identity_cov else (X_cov.T @ diff_u)) + pen_beta * beta
        g_delta = (c_i - P.sum(axis=0)) + lam_delta * delta

        w_u = W.sum(axis=1)
        h_dd = W.sum(axis=0) + lam_delta
        inv_hdd = 1.0 / h_dd

        if is_identity_cov:
            S_schur = np.diag(w_u + pen_beta) - (W * inv_hdd[None, :]) @ W.T
            rhs = -(g_beta + W @ (inv_hdd * g_delta))
            d_beta = np.linalg.solve(S_schur, rhs)
            d_delta = -inv_hdd * (g_delta - W.T @ d_beta)
        else:
            H_bb = (X_cov.T * w_u[None, :]) @ X_cov + np.diag(pen_beta)
            H_bd = -(X_cov.T @ W)
            S_schur = H_bb - (H_bd * inv_hdd[None, :]) @ H_bd.T
            rhs = -(g_beta - H_bd @ (inv_hdd * g_delta))
            d_beta = np.linalg.solve(S_schur, rhs)
            d_delta = -inv_hdd * (g_delta + H_bd.T @ d_beta)

        step = 1.0
        while step >= 1.0 / 32.0:
            cand_b = beta + step * d_beta
            cand_d = delta + step * d_delta
            cand_obj = _neg_loglik(cand_b, cand_d)
            if cand_obj <= cur_obj + 1e-10:
                beta, delta, cur_obj = cand_b, cand_d, cand_obj
                break
            step *= 0.5
        else:
            beta = beta + step * d_beta
            delta = delta + step * d_delta
            cur_obj = _neg_loglik(beta, delta)

        if max(float(np.max(np.abs(step * d_beta))), float(np.max(np.abs(step * d_delta)))) < tol:
            break

    theta_pred = beta if (is_identity_cov and X_pred is X_cov) else (X_pred @ beta)
    eta_pred = np.clip(theta_pred[:, None] - delta[None, :], -25.0, 25.0)
    pred_probs_matrix = 1.0 / (1.0 + np.exp(-eta_pred))
    pred_acc = pred_probs_matrix.mean(axis=1)
    return {
        "pred_acc": pred_acc,
        "pred_probs_matrix": pred_probs_matrix,
        "beta": beta,
        "delta": delta,
        "n_iters": n_iters,
    }


# ==============================================================================
# 5. Catalog Matrix Extraction & Multi-Budget Cohort Evaluation
# ==============================================================================

def extract_catalog_item_matrix(
    raw_100_mat: np.ndarray,
    catalog_bundle: Dict[str, Any],
    template_agg: str = "canonical_first_89",
) -> np.ndarray:
    """
    Extract the (U, N_items) response matrix for the specified template catalog granularity:
    - `canonical_first_89`: (89, N) single canonical template per observed `(op, space, sep)` combination
    - `cell_mean_89`: (89, N) row-averaged stratum mean per `(op, space, sep)` combination
    - `unique_93`: (93, N) deduplicated outer prompt-template strings
    - `catalog_100`: (100, N) all 100 raw catalog rows
    """
    if template_agg == "canonical_first_89":
        rep_idx = catalog_bundle["df_89"]["rep_row_index"].to_numpy(dtype=int)
        return raw_100_mat[rep_idx].astype(float)
    elif template_agg == "cell_mean_89":
        df_89 = catalog_bundle["df_89"]
        return np.array(
            [raw_100_mat[r["all_row_indices"]].mean(axis=0) for _, r in df_89.iterrows()],
            dtype=float,
        )
    elif template_agg == "unique_93":
        idx_93 = catalog_bundle["df_93"]["matrix_row_index"].to_numpy(dtype=int)
        return raw_100_mat[idx_93].astype(float)
    elif template_agg == "catalog_100":
        return raw_100_mat.astype(float)
    else:
        raise ValueError(f"Unknown template_agg: {template_agg!r}")


def select_all_outcome_blind_designs(
    catalog_bundle: Dict[str, Any],
    bases_89: Dict[str, Dict[str, Any]],
    seeds: Dict[str, int] = MIXED_LEVEL_SEEDS,
    n_random_draws: int = 200,
    n_fedorov_restarts: int = 100,
) -> Dict[str, Any]:
    """
    Construct EVERY deterministic and seeded-random design strictly from metadata X,
    before touching any outcome array Y.
    """
    X13 = bases_89["coarsened_main_13p"]["X"]
    X15 = bases_89["coarsened_inter_15p"]["X"]
    X22 = bases_89["additive_22p"]["X"]
    X28 = bases_89["additive22_plus_inter_28p"]["X"]
    X5b = bases_89["binary_5bit_16p"]["X"]

    bin_info = extract_binary_5bit_subsets_on_89(catalog_bundle)

    d16_13, a16_13 = select_paired_d_and_a_optimal(
        X13, 16, n_restarts=n_fedorov_restarts, seed=seeds["fedorov_restarts"]
    )
    d16_15, a16_15 = select_paired_d_and_a_optimal(
        X15, 16, n_restarts=n_fedorov_restarts, seed=seeds["fedorov_restarts"] + 1
    )
    d24_22, a24_22 = select_paired_d_and_a_optimal(
        X22, 24, n_restarts=n_fedorov_restarts, seed=seeds["fedorov_restarts"] + 2
    )
    d32_22, a32_22 = select_paired_d_and_a_optimal(
        X22, 32, n_restarts=n_fedorov_restarts, seed=seeds["fedorov_restarts"] + 3
    )
    d32_28, a32_28 = select_paired_d_and_a_optimal(
        X28, 32, n_restarts=n_fedorov_restarts, seed=seeds["fedorov_restarts"] + 4
    )

    designs: Dict[str, Any] = {
        "B16": {
            "d_opt_13p": d16_13,
            "a_opt_13p": a16_13,
            "d_opt_15p": d16_15,
            "a_opt_15p": a16_15,
            "rand_wellcond_13p": generate_mixed_level_random_subsets(
                X13, 16, num_subsets=n_random_draws, seed=seeds["rand_wellcond_16"],
                require_full_rank=True, max_condition_number=WELL_CONDITIONED_CAPS[("coarsened_main_13p", 16)],
            ),
            "rand_fullrank_13p": generate_mixed_level_random_subsets(
                X13, 16, num_subsets=n_random_draws, seed=seeds["rand_fullrank_16"],
                require_full_rank=True, max_condition_number=None,
            ),
            "rand_uncond_16": generate_mixed_level_random_subsets(
                X13, 16, num_subsets=n_random_draws, seed=seeds["rand_uncond_16"],
                require_full_rank=False, max_condition_number=None,
            ),
            "binary_res5_pos16": {
                "indices": bin_info["canonical_res5_pos16_combo_indices"],
                "criteria_5bit_16p": compute_design_criteria(
                    X5b[bin_info["canonical_res5_pos16_combo_indices"]]
                ),
                "criteria_coarse_13p": compute_design_criteria(
                    X13[bin_info["canonical_res5_pos16_combo_indices"]]
                ),
            },
            "binary_res5_neg16": {
                "indices": bin_info["canonical_res5_neg16_combo_indices"],
                "criteria_5bit_16p": compute_design_criteria(
                    X5b[bin_info["canonical_res5_neg16_combo_indices"]]
                ),
                "criteria_coarse_13p": compute_design_criteria(
                    X13[bin_info["canonical_res5_neg16_combo_indices"]]
                ),
            },
        },
        "B24": {
            "d_opt_22p": d24_22,
            "a_opt_22p": a24_22,
            "rand_wellcond_22p": generate_mixed_level_random_subsets(
                X22, 24, num_subsets=n_random_draws, seed=seeds["rand_wellcond_24"],
                require_full_rank=True, max_condition_number=WELL_CONDITIONED_CAPS[("additive_22p", 24)],
            ),
            "rand_fullrank_22p": generate_mixed_level_random_subsets(
                X22, 24, num_subsets=n_random_draws, seed=seeds["rand_fullrank_24"],
                require_full_rank=True, max_condition_number=None,
            ),
            "rand_uncond_24": generate_mixed_level_random_subsets(
                X22, 24, num_subsets=n_random_draws, seed=seeds["rand_uncond_24"],
                require_full_rank=False, max_condition_number=None,
            ),
        },
        "B32": {
            "d_opt_22p": d32_22,
            "a_opt_22p": a32_22,
            "d_opt_28p": d32_28,
            "a_opt_28p": a32_28,
            "rand_wellcond_22p": generate_mixed_level_random_subsets(
                X22, 32, num_subsets=n_random_draws, seed=seeds["rand_wellcond_32"],
                require_full_rank=True, max_condition_number=WELL_CONDITIONED_CAPS[("additive_22p", 32)],
            ),
            "rand_fullrank_22p": generate_mixed_level_random_subsets(
                X22, 32, num_subsets=n_random_draws, seed=seeds["rand_fullrank_32"],
                require_full_rank=True, max_condition_number=None,
            ),
            "rand_uncond_32": generate_mixed_level_random_subsets(
                X22, 32, num_subsets=n_random_draws, seed=seeds["rand_uncond_32"],
                require_full_rank=False, max_condition_number=None,
            ),
            "binary_cube32_canonical": {
                "indices": bin_info["canonical_cube32_combo_indices"],
                "criteria_5bit_16p": compute_design_criteria(
                    X5b[bin_info["canonical_cube32_combo_indices"]]
                ),
                "criteria_additive_22p": compute_design_criteria(
                    X22[bin_info["canonical_cube32_combo_indices"]]
                ),
            },
            "binary_cube32_legacy_audit": {
                "combo_indices_with_duplicates": bin_info["legacy_cube32_combo_indices"],
                "row_indices": bin_info["legacy_cube32_row_indices"],
                "criteria_additive_22p_on_89_projection": compute_design_criteria(
                    X22[bin_info["legacy_cube32_combo_indices"]]
                ),
            },
        },
    }
    return designs


def _axis_rmse(
    tau_hat: np.ndarray,
    tau_true: np.ndarray,
    col_slice_in_X: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Compute RMSE on an axis slice `(start, end)` in X (shifted by -1 for intercept-free tau)."""
    if col_slice_in_X is None:
        return np.full(tau_hat.shape[1:], np.nan, dtype=float)
    a, b = col_slice_in_X[0] - 1, col_slice_in_X[1] - 1
    return np.sqrt(np.mean((tau_hat[a:b] - tau_true[a:b]) ** 2, axis=0))


def _evaluate_subset_predictor_vectorized(
    Y_d: np.ndarray,                # (U, S_splits)
    Y_c: np.ndarray,                # (U, S_splits)
    X_target: np.ndarray,           # (U, p_target) ground-truth contrast basis
    target_slices: Dict[str, Tuple[int, int]],
    P_target_full: np.ndarray,      # (p_target, U) pinv(X_target)
    subsets: List[List[int]],       # list of D subsets of size B
    X_fit: Optional[np.ndarray],    # (U, p_fit) basis fitted on subset (None for ScreenBest-only / flat mean)
    mode: str = "ols",              # "ols", "flat_mean", "screen_only"
) -> Dict[str, np.ndarray]:
    """
    Fast vectorized evaluation across `S_splits` splits and `D` subsets of size `B` for:
    - Target categorical contrast RMSE (vs discovery truth & confirmation truth, plus op/space/sep slices)
    - Sign agreement with discovery truth
    - Held-out (U - B) unseen combination accuracy prediction RMSE (on confirmation & discovery items)
    - All-U template accuracy prediction RMSE on confirmation items
    - `ScreenBest`, `ModelBest`, and `ModelBestInScreen` confirmation accuracy (`mean_tied` and `first_index`)
    """
    U, S_splits = Y_d.shape
    D = len(subsets)
    B = len(subsets[0])
    sub_arr = np.asarray(subsets, dtype=int)  # (D, B)

    # Full-catalog ground-truth contrasts in X_target
    tau_true_d = 2.0 * (P_target_full @ Y_d)[1:, :]  # (p_target-1, S)
    tau_true_c = 2.0 * (P_target_full @ Y_c)[1:, :]  # (p_target-1, S)

    # Precompute out-of-sample index array (D, U - B)
    out_list = []
    for s in subsets:
        s_set = set(s)
        out_list.append([u for u in range(U) if u not in s_set])
    out_arr = np.asarray(out_list, dtype=int)  # (D, U - B)

    Y_d_in = Y_d[sub_arr, :]  # (D, B, S)
    s_range = np.arange(S_splits)

    # ScreenBest (empirical argmax inside S)
    sb_conf_mt, _, _ = select_with_tie_policy(Y_d_in, Y_c, sub_arr, tie_policy="mean_tied")
    sb_first_idx = sub_arr[np.arange(D)[:, None], np.argmax(Y_d_in, axis=1)]
    sb_conf_fi = Y_c[sb_first_idx, s_range[None, :]]

    if mode == "screen_only":
        nan_s = np.full(S_splits, np.nan, dtype=float)
        return {
            "contrast_rmse_disc": nan_s,
            "contrast_rmse_conf": nan_s,
            "contrast_sign_disc": nan_s,
            "contrast_rmse_op_disc": nan_s,
            "contrast_rmse_space_disc": nan_s,
            "contrast_rmse_sep_disc": nan_s,
            "heldout_pred_rmse_conf": nan_s,
            "heldout_pred_rmse_disc": nan_s,
            "all_pred_rmse_conf": nan_s,
            "conf_screen_best": sb_conf_mt.mean(axis=0),
            "conf_screen_best_fi": sb_conf_fi.mean(axis=0),
            "conf_model_best": nan_s,
            "conf_model_best_fi": nan_s,
            "conf_model_best_in_screen": nan_s,
        }

    if mode == "flat_mean":
        mean_in = Y_d_in.mean(axis=1, keepdims=True)  # (D, 1, S)
        Y_hat = np.broadcast_to(mean_in, (D, U, S_splits)).copy()
    elif mode == "ols":
        assert X_fit is not None
        P_fits = np.array([np.linalg.pinv(X_fit[s]) for s in subsets], dtype=float)  # (D, p_fit, B)
        betas = np.einsum("dpb,dbs->dps", P_fits, Y_d_in)                            # (D, p_fit, S)
        Y_hat = np.einsum("up,dps->dus", X_fit, betas)                               # (D, U, S)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Project predicted surface Y_hat onto target basis X_target: tau_hat = 2 * (P_target_full @ Y_hat)[1:]
    # Note: when X_fit == X_target and mode == "ols", P_target_full @ X_target == I, so tau_hat == 2 * betas[:, 1:]!
    tau_hat = 2.0 * np.einsum("pu,dus->dps", P_target_full[1:, :], Y_hat)            # (D, p_t-1, S)

    c_rmse_d = np.sqrt(np.mean((tau_hat - tau_true_d[None, :, :]) ** 2, axis=1)).mean(axis=0)
    c_rmse_c = np.sqrt(np.mean((tau_hat - tau_true_c[None, :, :]) ** 2, axis=1)).mean(axis=0)
    c_sign_d = np.mean(np.sign(tau_hat) == np.sign(tau_true_d[None, :, :]), axis=1).mean(axis=0)

    def _slice_err(col_name: str) -> np.ndarray:
        if col_name not in target_slices:
            return np.full(S_splits, np.nan, dtype=float)
        sl = target_slices[col_name]
        a, b = sl[0] - 1, sl[1] - 1
        return np.sqrt(
            np.mean((tau_hat[:, a:b, :] - tau_true_d[None, a:b, :]) ** 2, axis=1)
        ).mean(axis=0)

    # Held-out (U - B) prediction error
    Y_hat_out = np.take_along_axis(Y_hat, out_arr[:, :, None], axis=1)               # (D, U-B, S)
    Y_c_out = Y_c[out_arr, :]                                                        # (D, U-B, S)
    Y_d_out = Y_d[out_arr, :]                                                        # (D, U-B, S)

    ho_rmse_c = np.sqrt(np.mean((Y_hat_out - Y_c_out) ** 2, axis=1)).mean(axis=0)
    ho_rmse_d = np.sqrt(np.mean((Y_hat_out - Y_d_out) ** 2, axis=1)).mean(axis=0)
    all_rmse_c = np.sqrt(np.mean((Y_hat - Y_c[None, :, :]) ** 2, axis=1)).mean(axis=0)

    # ModelBest across all U templates & ModelBestInScreen across observed B templates
    mb_conf_mt, _, _ = select_with_tie_policy(Y_hat, Y_c, None, tie_policy="mean_tied")
    mb_conf_fi = Y_c[np.argmax(Y_hat, axis=1), s_range[None, :]]

    Y_hat_in = np.take_along_axis(Y_hat, sub_arr[:, :, None], axis=1)
    mb_in_mt, _, _ = select_with_tie_policy(Y_hat_in, Y_c, sub_arr, tie_policy="mean_tied")

    return {
        "contrast_rmse_disc": c_rmse_d,
        "contrast_rmse_conf": c_rmse_c,
        "contrast_sign_disc": c_sign_d,
        "contrast_rmse_op_disc": _slice_err("op"),
        "contrast_rmse_space_disc": _slice_err("space") if "space" in target_slices else _slice_err("space_coarse5"),
        "contrast_rmse_sep_disc": _slice_err("sep") if "sep" in target_slices else _slice_err("sep_coarse6"),
        "heldout_pred_rmse_conf": ho_rmse_c,
        "heldout_pred_rmse_disc": ho_rmse_d,
        "all_pred_rmse_conf": all_rmse_c,
        "conf_screen_best": sb_conf_mt.mean(axis=0),
        "conf_screen_best_fi": sb_conf_fi.mean(axis=0),
        "conf_model_best": mb_conf_mt.mean(axis=0),
        "conf_model_best_fi": mb_conf_fi.mean(axis=0),
        "conf_model_best_in_screen": mb_in_mt.mean(axis=0),
    }


def _evaluate_prompteval_single_split(
    Y_item_disc: np.ndarray,        # (U, N_disc) binary/float
    y_disc_full: np.ndarray,        # (U,)
    y_conf_full: np.ndarray,        # (U,)
    X_target: np.ndarray,           # (U, p_target)
    target_slices: Dict[str, Tuple[int, int]],
    P_target_full: np.ndarray,      # (p_target, U)
    X_cov: np.ndarray,              # (U, p_cov) covariates for PE
    subset_indices: Optional[List[int]],
    sparse_mask: Optional[np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate PromptEval (`PE-discrete` on subset or sparse schedule, or `PE-Rasch` 1PL one-hot
    when `X_cov = I_U`) strictly fitted on `Q = B * N_disc` discovery item evaluations.
    """
    U, N_disc = Y_item_disc.shape
    tau_true_d = 2.0 * (P_target_full @ y_disc_full)[1:]
    tau_true_c = 2.0 * (P_target_full @ y_conf_full)[1:]

    if subset_indices is not None:
        sub_arr = np.asarray(subset_indices, dtype=int)
        out_arr = np.asarray([u for u in range(U) if u not in set(subset_indices)], dtype=int)
        fit = fit_prompteval_rasch_covariate(
            Y_obs=Y_item_disc[sub_arr, :],
            mask_obs=None,
            X_cov=X_cov[sub_arr, :],
            X_pred=X_cov,
        )
        y_hat = fit["pred_acc"]
        sb_conf, _, _ = select_with_tie_policy(y_disc_full[sub_arr], y_conf_full, sub_arr, tie_policy="mean_tied")
        sb_conf_fi = float(y_conf_full[sub_arr[int(np.argmax(y_disc_full[sub_arr]))]])
        mb_in_conf, _, _ = select_with_tie_policy(y_hat[sub_arr], y_conf_full, sub_arr, tie_policy="mean_tied")
        ho_rmse_c = float(np.sqrt(np.mean((y_hat[out_arr] - y_conf_full[out_arr]) ** 2)))
        ho_rmse_d = float(np.sqrt(np.mean((y_hat[out_arr] - y_disc_full[out_arr]) ** 2)))
    else:
        assert sparse_mask is not None
        fit = fit_prompteval_rasch_covariate(
            Y_obs=Y_item_disc,
            mask_obs=sparse_mask,
            X_cov=X_cov,
            X_pred=X_cov,
        )
        y_hat = fit["pred_acc"]
        # Empirical per-template accuracy on observed sparse items
        obs_counts = np.maximum(1.0, sparse_mask.sum(axis=1))
        y_emp_sparse = (Y_item_disc * sparse_mask).sum(axis=1) / obs_counts
        sb_conf, _, _ = select_with_tie_policy(y_emp_sparse, y_conf_full, None, tie_policy="mean_tied")
        sb_conf_fi = float(y_conf_full[int(np.argmax(y_emp_sparse))])
        mb_in_conf, _, _ = select_with_tie_policy(y_hat, y_conf_full, None, tie_policy="mean_tied")
        # In sparse across-89 sampling, every template has ~B*N_disc/89 observed items;
        # heldout_pred_rmse_disc measures unobserved cell-item mean imputation on the remaining (1 - B/89) items!
        unobs_mask = 1.0 - sparse_mask
        unobs_counts = np.maximum(1.0, unobs_mask.sum(axis=1))
        y_unobs_disc = (Y_item_disc * unobs_mask).sum(axis=1) / unobs_counts
        y_hat_unobs_disc = (fit["pred_probs_matrix"] * unobs_mask).sum(axis=1) / unobs_counts
        ho_rmse_c = float("nan")
        ho_rmse_d = float(np.sqrt(np.mean((y_hat_unobs_disc - y_unobs_disc) ** 2)))

    tau_hat = 2.0 * (P_target_full[1:, :] @ y_hat)
    c_rmse_d = float(np.sqrt(np.mean((tau_hat - tau_true_d) ** 2)))
    c_rmse_c = float(np.sqrt(np.mean((tau_hat - tau_true_c) ** 2)))
    c_sign_d = float(np.mean(np.sign(tau_hat) == np.sign(tau_true_d)))

    def _sl_rmse(name: str) -> float:
        if name not in target_slices:
            return float("nan")
        a, b = target_slices[name][0] - 1, target_slices[name][1] - 1
        return float(np.sqrt(np.mean((tau_hat[a:b] - tau_true_d[a:b]) ** 2)))

    all_rmse_c = float(np.sqrt(np.mean((y_hat - y_conf_full) ** 2)))
    mb_conf, _, _ = select_with_tie_policy(y_hat, y_conf_full, None, tie_policy="mean_tied")
    mb_conf_fi = float(y_conf_full[int(np.argmax(y_hat))])

    return {
        "contrast_rmse_disc": c_rmse_d,
        "contrast_rmse_conf": c_rmse_c,
        "contrast_sign_disc": c_sign_d,
        "contrast_rmse_op_disc": _sl_rmse("op"),
        "contrast_rmse_space_disc": _sl_rmse("space") if "space" in target_slices else _sl_rmse("space_coarse5"),
        "contrast_rmse_sep_disc": _sl_rmse("sep") if "sep" in target_slices else _sl_rmse("sep_coarse6"),
        "heldout_pred_rmse_conf": ho_rmse_c,
        "heldout_pred_rmse_disc": ho_rmse_d,
        "all_pred_rmse_conf": all_rmse_c,
        "conf_screen_best": float(sb_conf),
        "conf_screen_best_fi": sb_conf_fi,
        "conf_model_best": float(mb_conf),
        "conf_model_best_fi": mb_conf_fi,
        "conf_model_best_in_screen": float(mb_in_conf),
    }


def run_mixed_level_study(
    Ys: Optional[Dict[str, Any]] = None,
    seeds: Dict[str, int] = MIXED_LEVEL_SEEDS,
    n_random_draws: int = 200,
    n_fedorov_restarts: int = 100,
    n_pe_rand_draws_split0: int = 10,
    n_splits: int = 20,
    n_boot: int = 1000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete Phase 0 Task R2 mixed-level study on MMLU across:
    - Primary `canonical_first_89` (Split 0 and 20 seeded random item splits, 855 cubes + 15 pooled models)
    - Stratum sensitivity `cell_mean_89` (Split 0 and 20 splits)
    - Catalog sensitivities `unique_93` and `catalog_100` (Split 0, 855 cubes + 15 pooled models)
    """
    source_hashes = verify_source_hashes()
    if Ys is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with (DATA_DIR / "Ys.pickle").open("rb") as f:
                Ys = pickle.load(f)

    catalog_bundle = load_mmlu_catalog_tables(DATA_DIR)
    catalog_audit = audit_mmlu_mixed_level_catalog(catalog_bundle, Ys=Ys)

    bases_89 = build_all_catalog_bases(catalog_bundle["df_89"], "combinations_89")
    bases_93 = build_all_catalog_bases(catalog_bundle["df_93"], "unique_templates_93")
    bases_100 = build_all_catalog_bases(catalog_bundle["df_100"], "catalog_rows_100")

    # Select all outcome-blind designs BEFORE accessing Ys outcomes
    designs_89 = select_all_outcome_blind_designs(
        catalog_bundle,
        bases_89,
        seeds=seeds,
        n_random_draws=n_random_draws,
        n_fedorov_restarts=n_fedorov_restarts,
    )

    tasks = sorted(Ys["MMLU"].keys())
    n_models = len(MMLU_MODELS)
    n_tasks = len(tasks)
    n_splits_total = n_splits + 1

    task_splits = {
        t: generate_item_splits(Ys["MMLU"][t][0].shape[1], n_splits=n_splits, seed=seeds["item_splits"])
        for t in tasks
    }

    # Pre-generate outcome-blind sparse PromptEval masks per task & split!
    # For each task t, split s in 0..n_splits, and budget B in {16, 24, 32}, mask depends ONLY on (89, N_disc, B, seed)
    sparse_masks_task: Dict[Tuple[int, str, int], np.ndarray] = {}
    for B, seed_key in [(16, "pe_sparse_16"), (24, "pe_sparse_24"), (32, "pe_sparse_32")]:
        base_seed = seeds[seed_key]
        for t_i, t in enumerate(tasks):
            d_masks, _, _ = task_splits[t]
            for s_i in range(n_splits_total):
                n_disc_s = int(np.sum(d_masks[:, s_i] > 0))
                mask_seed = base_seed + t_i * 1000 + s_i
                m_obs = generate_prompteval_sparse_mask(89, n_disc_s, B, seed=mask_seed)
                assert int(m_obs.sum()) == B * n_disc_s
                sparse_masks_task[(B, t, s_i)] = m_obs

    # Define specifications evaluated on the primary 89-combination universe
    X13 = bases_89["coarsened_main_13p"]["X"]
    P13 = np.linalg.pinv(X13)
    sl13 = bases_89["coarsened_main_13p"]["column_slices"]

    X15 = bases_89["coarsened_inter_15p"]["X"]
    P15 = np.linalg.pinv(X15)
    sl15 = bases_89["coarsened_inter_15p"]["column_slices"]

    X22 = bases_89["additive_22p"]["X"]
    P22 = np.linalg.pinv(X22)
    sl22 = bases_89["additive_22p"]["column_slices"]

    X28 = bases_89["additive22_plus_inter_28p"]["X"]
    P28 = np.linalg.pinv(X28)
    sl28 = bases_89["additive22_plus_inter_28p"]["column_slices"]

    X5b = bases_89["binary_5bit_16p"]["X"]
    I89 = np.eye(89, dtype=float)

    ols_method_specs: List[Dict[str, Any]] = [
        # === Budget B = 16 (target basis: coarsened_main_13p, plus coarsened_inter_15p) ===
        {
            "key": "B16_d_opt_13p", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "d_optimal_ols", "family": "optimal_design",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X13, "mode": "ols",
            "subsets": [designs_89["B16"]["d_opt_13p"]["indices"]],
            "meta": designs_89["B16"]["d_opt_13p"],
        },
        {
            "key": "B16_a_opt_13p", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "a_optimal_ols", "family": "optimal_design",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X13, "mode": "ols",
            "subsets": [designs_89["B16"]["a_opt_13p"]["indices"]],
            "meta": designs_89["B16"]["a_opt_13p"],
        },
        {
            "key": "B16_rand_wellcond_13p", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "random_well_conditioned_ols", "family": "random_well_conditioned",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X13, "mode": "ols",
            "subsets": designs_89["B16"]["rand_wellcond_13p"]["subsets"],
            "meta": designs_89["B16"]["rand_wellcond_13p"],
        },
        {
            "key": "B16_rand_fullrank_13p", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "random_full_rank_ols", "family": "random_full_rank",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X13, "mode": "ols",
            "subsets": designs_89["B16"]["rand_fullrank_13p"]["subsets"],
            "meta": designs_89["B16"]["rand_fullrank_13p"],
        },
        {
            "key": "B16_rand_uncond", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "random_unconditioned", "family": "random_unconditioned",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": None, "mode": "screen_only",
            "subsets": designs_89["B16"]["rand_uncond_16"]["subsets"],
            "meta": designs_89["B16"]["rand_uncond_16"],
        },
        {
            "key": "B16_binary_res5_pos", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 16,
            "method_id": "binary_2_5_1_V_pos", "family": "binary_factorial",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X5b, "mode": "ols",
            "subsets": [designs_89["B16"]["binary_res5_pos16"]["indices"]],
            "meta": designs_89["B16"]["binary_res5_pos16"]["criteria_5bit_16p"],
        },
        {
            "key": "B16_binary_res5_neg", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 16,
            "method_id": "binary_2_5_1_V_neg", "family": "binary_factorial",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X5b, "mode": "ols",
            "subsets": [designs_89["B16"]["binary_res5_neg16"]["indices"]],
            "meta": designs_89["B16"]["binary_res5_neg16"]["criteria_5bit_16p"],
        },
        {
            "key": "B16_binary_res5_avg", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 16,
            "method_id": "binary_2_5_1_V_coset_avg", "family": "binary_factorial",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": X5b, "mode": "ols",
            "subsets": [
                designs_89["B16"]["binary_res5_pos16"]["indices"],
                designs_89["B16"]["binary_res5_neg16"]["indices"],
            ],
            "meta": designs_89["B16"]["binary_res5_neg16"]["criteria_5bit_16p"],
        },
        {
            "key": "B16_obs_mean_dopt", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 1,
            "method_id": "observed_mean_predictor", "family": "baseline",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_fit": None, "mode": "flat_mean",
            "subsets": [designs_89["B16"]["d_opt_13p"]["indices"]],
            "meta": designs_89["B16"]["d_opt_13p"],
        },
        # Coarsened axis-interaction basis (15p) at B=16
        {
            "key": "B16_d_opt_15p", "budget": 16, "basis_id": "coarsened_inter_15p", "n_params": 15,
            "method_id": "d_optimal_ols", "family": "optimal_design",
            "X_target": X15, "P_target": P15, "slices": sl15, "X_fit": X15, "mode": "ols",
            "subsets": [designs_89["B16"]["d_opt_15p"]["indices"]],
            "meta": designs_89["B16"]["d_opt_15p"],
        },
        {
            "key": "B16_a_opt_15p", "budget": 16, "basis_id": "coarsened_inter_15p", "n_params": 15,
            "method_id": "a_optimal_ols", "family": "optimal_design",
            "X_target": X15, "P_target": P15, "slices": sl15, "X_fit": X15, "mode": "ols",
            "subsets": [designs_89["B16"]["a_opt_15p"]["indices"]],
            "meta": designs_89["B16"]["a_opt_15p"],
        },
        # === Budget B = 24 (full 22-parameter additive categorical model) ===
        {
            "key": "B24_d_opt_22p", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "d_optimal_ols", "family": "optimal_design",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": [designs_89["B24"]["d_opt_22p"]["indices"]],
            "meta": designs_89["B24"]["d_opt_22p"],
        },
        {
            "key": "B24_a_opt_22p", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "a_optimal_ols", "family": "optimal_design",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": [designs_89["B24"]["a_opt_22p"]["indices"]],
            "meta": designs_89["B24"]["a_opt_22p"],
        },
        {
            "key": "B24_rand_wellcond_22p", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "random_well_conditioned_ols", "family": "random_well_conditioned",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": designs_89["B24"]["rand_wellcond_22p"]["subsets"],
            "meta": designs_89["B24"]["rand_wellcond_22p"],
        },
        {
            "key": "B24_rand_fullrank_22p", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "random_full_rank_ols", "family": "random_full_rank",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": designs_89["B24"]["rand_fullrank_22p"]["subsets"],
            "meta": designs_89["B24"]["rand_fullrank_22p"],
        },
        {
            "key": "B24_rand_uncond", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "random_unconditioned", "family": "random_unconditioned",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": None, "mode": "screen_only",
            "subsets": designs_89["B24"]["rand_uncond_24"]["subsets"],
            "meta": designs_89["B24"]["rand_uncond_24"],
        },
        {
            "key": "B24_obs_mean_dopt", "budget": 24, "basis_id": "additive_22p", "n_params": 1,
            "method_id": "observed_mean_predictor", "family": "baseline",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": None, "mode": "flat_mean",
            "subsets": [designs_89["B24"]["d_opt_22p"]["indices"]],
            "meta": designs_89["B24"]["d_opt_22p"],
        },
        # === Budget B = 32 (full 22-parameter additive categorical model + 5-bit cube + 28p inter) ===
        {
            "key": "B32_d_opt_22p", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "d_optimal_ols", "family": "optimal_design",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": [designs_89["B32"]["d_opt_22p"]["indices"]],
            "meta": designs_89["B32"]["d_opt_22p"],
        },
        {
            "key": "B32_a_opt_22p", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "a_optimal_ols", "family": "optimal_design",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": [designs_89["B32"]["a_opt_22p"]["indices"]],
            "meta": designs_89["B32"]["a_opt_22p"],
        },
        {
            "key": "B32_rand_wellcond_22p", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "random_well_conditioned_ols", "family": "random_well_conditioned",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": designs_89["B32"]["rand_wellcond_22p"]["subsets"],
            "meta": designs_89["B32"]["rand_wellcond_22p"],
        },
        {
            "key": "B32_rand_fullrank_22p", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "random_full_rank_ols", "family": "random_full_rank",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": designs_89["B32"]["rand_fullrank_22p"]["subsets"],
            "meta": designs_89["B32"]["rand_fullrank_22p"],
        },
        {
            "key": "B32_rand_uncond", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "random_unconditioned", "family": "random_unconditioned",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": None, "mode": "screen_only",
            "subsets": designs_89["B32"]["rand_uncond_32"]["subsets"],
            "meta": designs_89["B32"]["rand_uncond_32"],
        },
        {
            "key": "B32_binary_cube32_5bit", "budget": 32, "basis_id": "additive_22p", "n_params": 16,
            "method_id": "binary_cube32_5bit_model", "family": "binary_factorial",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X5b, "mode": "ols",
            "subsets": [designs_89["B32"]["binary_cube32_canonical"]["indices"]],
            "meta": designs_89["B32"]["binary_cube32_canonical"]["criteria_5bit_16p"],
        },
        {
            "key": "B32_binary_cube32_ols22", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "binary_cube32_direct_ols22", "family": "binary_factorial",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": X22, "mode": "ols",
            "subsets": [designs_89["B32"]["binary_cube32_canonical"]["indices"]],
            "meta": designs_89["B32"]["binary_cube32_canonical"]["criteria_additive_22p"],
        },
        {
            "key": "B32_obs_mean_dopt", "budget": 32, "basis_id": "additive_22p", "n_params": 1,
            "method_id": "observed_mean_predictor", "family": "baseline",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_fit": None, "mode": "flat_mean",
            "subsets": [designs_89["B32"]["d_opt_22p"]["indices"]],
            "meta": designs_89["B32"]["d_opt_22p"],
        },
        # 28p additive + coarsened axis-interaction basis at B=32
        {
            "key": "B32_d_opt_28p", "budget": 32, "basis_id": "additive22_plus_inter_28p", "n_params": 28,
            "method_id": "d_optimal_ols", "family": "optimal_design",
            "X_target": X28, "P_target": P28, "slices": sl28, "X_fit": X28, "mode": "ols",
            "subsets": [designs_89["B32"]["d_opt_28p"]["indices"]],
            "meta": designs_89["B32"]["d_opt_28p"],
        },
        {
            "key": "B32_a_opt_28p", "budget": 32, "basis_id": "additive22_plus_inter_28p", "n_params": 28,
            "method_id": "a_optimal_ols", "family": "optimal_design",
            "X_target": X28, "P_target": P28, "slices": sl28, "X_fit": X28, "mode": "ols",
            "subsets": [designs_89["B32"]["a_opt_28p"]["indices"]],
            "meta": designs_89["B32"]["a_opt_28p"],
        },
    ]

    pe_method_specs: List[Dict[str, Any]] = [
        {
            "key": "B16_pe_dopt_subset", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "prompteval_discrete_dopt_subset", "family": "prompteval",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_cov": X13,
            "mode": "dopt_subset", "subset": designs_89["B16"]["d_opt_13p"]["indices"],
            "meta": designs_89["B16"]["d_opt_13p"],
        },
        {
            "key": "B16_pe_rand_subset", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 13,
            "method_id": "prompteval_discrete_rand_subset", "family": "prompteval",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_cov": X13,
            "mode": "rand_subset", "rand_subsets": designs_89["B16"]["rand_fullrank_13p"]["subsets"][:n_pe_rand_draws_split0],
            "meta": designs_89["B16"]["rand_fullrank_13p"],
        },
        {
            "key": "B16_pe_sparse_disc", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 22,
            "method_id": "prompteval_discrete_sparse", "family": "prompteval",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_cov": X22,
            "mode": "sparse", "meta": compute_design_criteria(X22),
        },
        {
            "key": "B16_pe_sparse_1pl", "budget": 16, "basis_id": "coarsened_main_13p", "n_params": 89,
            "method_id": "prompteval_1pl_rasch_sparse", "family": "prompteval",
            "X_target": X13, "P_target": P13, "slices": sl13, "X_cov": I89,
            "mode": "sparse", "meta": {"rank": 89, "is_full_rank": True, "condition_number": 1.0, "logdet_xtx": 0.0, "trace_inv_xtx": 89.0},
        },
        {
            "key": "B24_pe_dopt_subset", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "prompteval_discrete_dopt_subset", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": X22,
            "mode": "dopt_subset", "subset": designs_89["B24"]["d_opt_22p"]["indices"],
            "meta": designs_89["B24"]["d_opt_22p"],
        },
        {
            "key": "B24_pe_rand_subset", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "prompteval_discrete_rand_subset", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": X22,
            "mode": "rand_subset", "rand_subsets": designs_89["B24"]["rand_fullrank_22p"]["subsets"][:n_pe_rand_draws_split0],
            "meta": designs_89["B24"]["rand_fullrank_22p"],
        },
        {
            "key": "B24_pe_sparse_disc", "budget": 24, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "prompteval_discrete_sparse", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": X22,
            "mode": "sparse", "meta": compute_design_criteria(X22),
        },
        {
            "key": "B24_pe_sparse_1pl", "budget": 24, "basis_id": "additive_22p", "n_params": 89,
            "method_id": "prompteval_1pl_rasch_sparse", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": I89,
            "mode": "sparse", "meta": {"rank": 89, "is_full_rank": True, "condition_number": 1.0, "logdet_xtx": 0.0, "trace_inv_xtx": 89.0},
        },
        {
            "key": "B32_pe_dopt_subset", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "prompteval_discrete_dopt_subset", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": X22,
            "mode": "dopt_subset", "subset": designs_89["B32"]["d_opt_22p"]["indices"],
            "meta": designs_89["B32"]["d_opt_22p"],
        },
        {
            "key": "B32_pe_rand_subset", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "prompteval_discrete_rand_subset", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": X22,
            "mode": "rand_subset", "rand_subsets": designs_89["B32"]["rand_fullrank_22p"]["subsets"][:n_pe_rand_draws_split0],
            "meta": designs_89["B32"]["rand_fullrank_22p"],
        },
        {
            "key": "B32_pe_sparse_disc", "budget": 32, "basis_id": "additive_22p", "n_params": 22,
            "method_id": "prompteval_discrete_sparse", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": X22,
            "mode": "sparse", "meta": compute_design_criteria(X22),
        },
        {
            "key": "B32_pe_sparse_1pl", "budget": 32, "basis_id": "additive_22p", "n_params": 89,
            "method_id": "prompteval_1pl_rasch_sparse", "family": "prompteval",
            "X_target": X22, "P_target": P22, "slices": sl22, "X_cov": I89,
            "mode": "sparse", "meta": {"rank": 89, "is_full_rank": True, "condition_number": 1.0, "logdet_xtx": 0.0, "trace_inv_xtx": 89.0},
        },
    ]

    metric_names = [
        "contrast_rmse_disc", "contrast_rmse_conf", "contrast_sign_disc",
        "contrast_rmse_op_disc", "contrast_rmse_space_disc", "contrast_rmse_sep_disc",
        "heldout_pred_rmse_conf", "heldout_pred_rmse_disc", "all_pred_rmse_conf",
        "conf_screen_best", "conf_screen_best_fi", "conf_model_best",
        "conf_model_best_fi", "conf_model_best_in_screen",
    ]

    csv_rows: List[Dict[str, Any]] = []
    primary_panel_store: Dict[str, Dict[str, np.ndarray]] = {}

    for template_agg in ("canonical_first_89", "cell_mean_89"):
        run_pe = template_agg == "canonical_first_89"  # PromptEval requires binary item responses
        active_specs = ols_method_specs + (pe_method_specs if run_pe else [])

        cube_metrics: Dict[str, Dict[str, np.ndarray]] = {
            sp["key"]: {mn: np.full((n_models, n_tasks, n_splits_total), np.nan, dtype=float) for mn in metric_names}
            for sp in active_specs
        }
        pool_metrics: Dict[str, Dict[str, np.ndarray]] = {
            sp["key"]: {mn: np.full(n_models, np.nan, dtype=float) for mn in metric_names}
            for sp in active_specs
        }
        oracle_cube = np.zeros((n_models, n_tasks, n_splits_total), dtype=float)
        mean_cube = np.zeros((n_models, n_tasks, n_splits_total), dtype=float)
        oracle_pool = np.zeros(n_models, dtype=float)
        mean_pool = np.zeros(n_models, dtype=float)

        for m_idx, _ in enumerate(MMLU_MODELS):
            task_mats: List[np.ndarray] = []
            task_conf_mats: List[np.ndarray] = []

            for t_idx, t in enumerate(tasks):
                Y_u_item = extract_catalog_item_matrix(Ys["MMLU"][t][m_idx], catalog_bundle, template_agg)
                task_mats.append(Y_u_item)
                task_conf_mats.append(Y_u_item[:, 1::2])

                d_masks, c_masks, _ = task_splits[t]
                d_bin = (d_masks > 0).astype(float)
                c_bin = (c_masks > 0).astype(float)
                Y_d = (Y_u_item @ d_bin) / d_bin.sum(axis=0)[None, :]
                Y_c = (Y_u_item @ c_bin) / c_bin.sum(axis=0)[None, :]
                Y_d[:, 0] = Y_u_item[:, ::2].mean(axis=1)
                Y_c[:, 0] = Y_u_item[:, 1::2].mean(axis=1)

                orc_c, _, _ = select_with_tie_policy(Y_d, Y_c, None, tie_policy="mean_tied")
                oracle_cube[m_idx, t_idx] = orc_c
                mean_cube[m_idx, t_idx] = Y_c.mean(axis=0)

                for sp in ols_method_specs:
                    res = _evaluate_subset_predictor_vectorized(
                        Y_d, Y_c, sp["X_target"], sp["slices"], sp["P_target"],
                        sp["subsets"], sp["X_fit"], mode=sp["mode"],
                    )
                    for mn in metric_names:
                        cube_metrics[sp["key"]][mn][m_idx, t_idx] = res[mn]

                if run_pe:
                    for sp in pe_method_specs:
                        for s_i in range(n_splits_total):
                            if s_i == 0:
                                Y_item_d = Y_u_item[:, ::2]
                            else:
                                d_idx = np.flatnonzero(d_masks[:, s_i] > 0)
                                Y_item_d = Y_u_item[:, d_idx]

                            if sp["mode"] == "dopt_subset":
                                pe_res = _evaluate_prompteval_single_split(
                                    Y_item_d, Y_d[:, s_i], Y_c[:, s_i],
                                    sp["X_target"], sp["slices"], sp["P_target"],
                                    sp["X_cov"], sp["subset"], None,
                                )
                            elif sp["mode"] == "rand_subset":
                                # Evaluate full ensemble on Split 0, and deterministic draw s_i % K on repeated splits
                                subs_to_run = sp["rand_subsets"] if s_i == 0 else [sp["rand_subsets"][s_i % len(sp["rand_subsets"])]]
                                sub_dicts = [
                                    _evaluate_prompteval_single_split(
                                        Y_item_d, Y_d[:, s_i], Y_c[:, s_i],
                                        sp["X_target"], sp["slices"], sp["P_target"],
                                        sp["X_cov"], s_cand, None,
                                    )
                                    for s_cand in subs_to_run
                                ]
                                pe_res = {mn: float(np.mean([d[mn] for d in sub_dicts])) for mn in metric_names}
                            else:
                                m_sparse = sparse_masks_task[(sp["budget"], t, s_i)]
                                pe_res = _evaluate_prompteval_single_split(
                                    Y_item_d, Y_d[:, s_i], Y_c[:, s_i],
                                    sp["X_target"], sp["slices"], sp["P_target"],
                                    sp["X_cov"], None, m_sparse,
                                )
                            for mn in metric_names:
                                cube_metrics[sp["key"]][mn][m_idx, t_idx, s_i] = pe_res[mn]

            # Pooled 14,042-item model evaluation (within-task even/odd split preserved!)
            Y_pool_disc_items = concatenate_task_discovery_items(task_mats)
            Y_pool_conf_items = np.hstack(task_conf_mats)
            y_pool_d = Y_pool_disc_items.mean(axis=1)[:, None]
            y_pool_c = Y_pool_conf_items.mean(axis=1)[:, None]

            orc_p, _, _ = select_with_tie_policy(y_pool_d, y_pool_c, None, tie_policy="mean_tied")
            oracle_pool[m_idx] = float(np.squeeze(orc_p))
            mean_pool[m_idx] = float(y_pool_c.mean())

            for sp in ols_method_specs:
                res_p = _evaluate_subset_predictor_vectorized(
                    y_pool_d, y_pool_c, sp["X_target"], sp["slices"], sp["P_target"],
                    sp["subsets"], sp["X_fit"], mode=sp["mode"],
                )
                for mn in metric_names:
                    pool_metrics[sp["key"]][mn][m_idx] = float(res_p[mn][0])

            if run_pe:
                for sp in pe_method_specs:
                    if sp["mode"] == "dopt_subset":
                        pe_p = _evaluate_prompteval_single_split(
                            Y_pool_disc_items, y_pool_d[:, 0], y_pool_c[:, 0],
                            sp["X_target"], sp["slices"], sp["P_target"],
                            sp["X_cov"], sp["subset"], None,
                        )
                    elif sp["mode"] == "rand_subset":
                        sub_dicts_p = [
                            _evaluate_prompteval_single_split(
                                Y_pool_disc_items, y_pool_d[:, 0], y_pool_c[:, 0],
                                sp["X_target"], sp["slices"], sp["P_target"],
                                sp["X_cov"], s_cand, None,
                            )
                            for s_cand in sp["rand_subsets"]
                        ]
                        pe_p = {mn: float(np.mean([d[mn] for d in sub_dicts_p])) for mn in metric_names}
                    else:
                        pool_mask = generate_prompteval_sparse_mask(
                            89, Y_pool_disc_items.shape[1], sp["budget"],
                            seed=seeds[f"pe_sparse_{sp['budget']}"] + 99999,
                        )
                        pe_p = _evaluate_prompteval_single_split(
                            Y_pool_disc_items, y_pool_d[:, 0], y_pool_c[:, 0],
                            sp["X_target"], sp["slices"], sp["P_target"],
                            sp["X_cov"], None, pool_mask,
                        )
                    for mn in metric_names:
                        pool_metrics[sp["key"]][mn][m_idx] = pe_p[mn]

        if template_agg == "canonical_first_89":
            primary_panel_store["cube_metrics"] = cube_metrics
            primary_panel_store["pool_metrics"] = pool_metrics
            primary_panel_store["oracle_cube"] = oracle_cube
            primary_panel_store["mean_cube"] = mean_cube
            primary_panel_store["oracle_pool"] = oracle_pool
            primary_panel_store["mean_pool"] = mean_pool

        # Emit summary rows for CSV
        for sp in active_specs:
            meta_d = sp["meta"]
            rk_mean = float(meta_d.get("rank_mean", meta_d.get("rank", np.nan)))
            fr_rate = float(meta_d.get("accepted_full_rank_rate", 1.0 if meta_d.get("is_full_rank", True) else 0.0))
            cd_mean = float(meta_d.get("condition_number_mean", meta_d.get("condition_number", np.nan)))
            cd_med = float(meta_d.get("condition_number_median", meta_d.get("condition_number", np.nan)))
            ld_mean = float(meta_d.get("logdet_xtx_mean", meta_d.get("logdet_xtx", np.nan)))
            tr_mean = float(meta_d.get("trace_inv_xtx_mean", meta_d.get("trace_inv_xtx", np.nan)))
            n_des = int(meta_d.get("num_subsets", len(sp.get("subsets", [1]))))

            cm = cube_metrics[sp["key"]]
            for split_label, sl_slice in [("split0_even_odd", slice(0, 1)), ("mean_20_random_splits", slice(1, n_splits_total))]:
                sb_mat = cm["conf_screen_best"][:, :, sl_slice].mean(axis=2)
                mb_mat = cm["conf_model_best"][:, :, sl_slice].mean(axis=2)
                orc_mat = oracle_cube[:, :, sl_slice].mean(axis=2)
                mn_mat = mean_cube[:, :, sl_slice].mean(axis=2)

                ci_sb = two_way_cluster_bootstrap_ci(sb_mat, n_boot=n_boot, seed=seeds["cluster_bootstrap"])
                if np.isfinite(mb_mat).all():
                    ci_mb = two_way_cluster_bootstrap_ci(mb_mat, n_boot=n_boot, seed=seeds["cluster_bootstrap"])
                else:
                    ci_mb = {"ci_low": np.nan, "ci_high": np.nan}

                csv_rows.append(
                    {
                        "catalog_granularity": "combinations_89",
                        "template_agg": template_agg,
                        "evaluation_level": "task_855_cubes",
                        "split_regime": split_label,
                        "budget": sp["budget"],
                        "basis_id": sp["basis_id"],
                        "n_parameters": sp["n_params"],
                        "method_id": sp["method_id"],
                        "method_family": sp["family"],
                        "outcome_blind_design": True,
                        "query_budget_per_item_equiv": sp["budget"],
                        "n_designs_evaluated": n_des,
                        "matrix_rank_mean": rk_mean,
                        "is_full_rank_rate": fr_rate,
                        "condition_number_mean": cd_mean,
                        "condition_number_median": cd_med,
                        "logdet_xtx_mean": ld_mean,
                        "trace_inv_xtx_mean": tr_mean,
                        "contrast_rmse_vs_disc_truth": float(cm["contrast_rmse_disc"][:, :, sl_slice].mean()),
                        "contrast_rmse_vs_conf_truth": float(cm["contrast_rmse_conf"][:, :, sl_slice].mean()),
                        "contrast_sign_agreement_disc": float(cm["contrast_sign_disc"][:, :, sl_slice].mean()),
                        "contrast_rmse_op_disc": float(cm["contrast_rmse_op_disc"][:, :, sl_slice].mean()),
                        "contrast_rmse_space_disc": float(cm["contrast_rmse_space_disc"][:, :, sl_slice].mean()),
                        "contrast_rmse_sep_disc": float(cm["contrast_rmse_sep_disc"][:, :, sl_slice].mean()),
                        "heldout_pred_rmse_conf": float(cm["heldout_pred_rmse_conf"][:, :, sl_slice].mean()),
                        "heldout_pred_rmse_disc": float(cm["heldout_pred_rmse_disc"][:, :, sl_slice].mean()),
                        "all_templates_pred_rmse_conf": float(cm["all_pred_rmse_conf"][:, :, sl_slice].mean()),
                        "conf_acc_screen_best": float(sb_mat.mean()),
                        "conf_acc_model_best": float(mb_mat.mean()),
                        "conf_acc_model_best_in_screen": float(cm["conf_model_best_in_screen"][:, :, sl_slice].mean()),
                        "conf_acc_screen_best_first_index": float(cm["conf_screen_best_fi"][:, :, sl_slice].mean()),
                        "conf_acc_model_best_first_index": float(cm["conf_model_best_fi"][:, :, sl_slice].mean()),
                        "conf_acc_oracle_full": float(orc_mat.mean()),
                        "conf_acc_catalog_mean": float(mn_mat.mean()),
                        "regret_screen_best_vs_oracle": float(orc_mat.mean() - sb_mat.mean()),
                        "regret_model_best_vs_oracle": float(orc_mat.mean() - mb_mat.mean()),
                        "ci95_low_conf_screen_best": float(ci_sb["ci_low"]),
                        "ci95_high_conf_screen_best": float(ci_sb["ci_high"]),
                        "ci95_low_conf_model_best": float(ci_mb["ci_low"]),
                        "ci95_high_conf_model_best": float(ci_mb["ci_high"]),
                    }
                )

            pm = pool_metrics[sp["key"]]
            csv_rows.append(
                {
                    "catalog_granularity": "combinations_89",
                    "template_agg": template_agg,
                    "evaluation_level": "pooled_15_models",
                    "split_regime": "split0_even_odd",
                    "budget": sp["budget"],
                    "basis_id": sp["basis_id"],
                    "n_parameters": sp["n_params"],
                    "method_id": sp["method_id"],
                    "method_family": sp["family"],
                    "outcome_blind_design": True,
                    "query_budget_per_item_equiv": sp["budget"],
                    "n_designs_evaluated": n_des,
                    "matrix_rank_mean": rk_mean,
                    "is_full_rank_rate": fr_rate,
                    "condition_number_mean": cd_mean,
                    "condition_number_median": cd_med,
                    "logdet_xtx_mean": ld_mean,
                    "trace_inv_xtx_mean": tr_mean,
                    "contrast_rmse_vs_disc_truth": float(pm["contrast_rmse_disc"].mean()),
                    "contrast_rmse_vs_conf_truth": float(pm["contrast_rmse_conf"].mean()),
                    "contrast_sign_agreement_disc": float(pm["contrast_sign_disc"].mean()),
                    "contrast_rmse_op_disc": float(pm["contrast_rmse_op_disc"].mean()),
                    "contrast_rmse_space_disc": float(pm["contrast_rmse_space_disc"].mean()),
                    "contrast_rmse_sep_disc": float(pm["contrast_rmse_sep_disc"].mean()),
                    "heldout_pred_rmse_conf": float(pm["heldout_pred_rmse_conf"].mean()),
                    "heldout_pred_rmse_disc": float(pm["heldout_pred_rmse_disc"].mean()),
                    "all_templates_pred_rmse_conf": float(pm["all_pred_rmse_conf"].mean()),
                    "conf_acc_screen_best": float(pm["conf_screen_best"].mean()),
                    "conf_acc_model_best": float(pm["conf_model_best"].mean()),
                    "conf_acc_model_best_in_screen": float(pm["conf_model_best_in_screen"].mean()),
                    "conf_acc_screen_best_first_index": float(pm["conf_screen_best_fi"].mean()),
                    "conf_acc_model_best_first_index": float(pm["conf_model_best_fi"].mean()),
                    "conf_acc_oracle_full": float(oracle_pool.mean()),
                    "conf_acc_catalog_mean": float(mean_pool.mean()),
                    "regret_screen_best_vs_oracle": float(oracle_pool.mean() - pm["conf_screen_best"].mean()),
                    "regret_model_best_vs_oracle": float(oracle_pool.mean() - pm["conf_model_best"].mean()),
                    "ci95_low_conf_screen_best": np.nan,
                    "ci95_high_conf_screen_best": np.nan,
                    "ci95_low_conf_model_best": np.nan,
                    "ci95_high_conf_model_best": np.nan,
                }
            )

    # Row-level 100-catalog and deduplicated 93-template sensitivities (Split 0, B in {24, 32} on 22p)
    for gran_label, t_agg, b_dict in [
        ("unique_templates_93", "unique_93", bases_93),
        ("catalog_rows_100", "catalog_100", bases_100),
    ]:
        X_g = b_dict["additive_22p"]["X"]
        P_g = np.linalg.pinv(X_g)
        sl_g = b_dict["additive_22p"]["column_slices"]
        for B_g, seed_off in [(24, 100), (32, 200)]:
            d_opt_g, a_opt_g = select_paired_d_and_a_optimal(
                X_g, B_g, n_restarts=n_fedorov_restarts, seed=seeds["fedorov_restarts"] + seed_off
            )
            r_fr_g = generate_mixed_level_random_subsets(
                X_g, B_g, num_subsets=n_random_draws, seed=seeds[f"rand_fullrank_{B_g}"] + seed_off,
                require_full_rank=True, max_condition_number=None,
            )
            for m_id_g, fam_g, subs_g, meta_g in [
                ("d_optimal_ols", "optimal_design", [d_opt_g["indices"]], d_opt_g),
                ("a_optimal_ols", "optimal_design", [a_opt_g["indices"]], a_opt_g),
                ("random_full_rank_ols", "random_full_rank", r_fr_g["subsets"], r_fr_g),
            ]:
                c_acc_mat = np.zeros((n_models, n_tasks), dtype=float)
                m_acc_mat = np.zeros((n_models, n_tasks), dtype=float)
                orc_g_mat = np.zeros((n_models, n_tasks), dtype=float)
                mn_g_mat = np.zeros((n_models, n_tasks), dtype=float)
                agg_vals = {mn: [] for mn in metric_names}
                pool_vals = {mn: [] for mn in metric_names}

                for m_idx in range(n_models):
                    t_mats_g = []
                    t_conf_g = []
                    for t_idx, t in enumerate(tasks):
                        Y_g = extract_catalog_item_matrix(Ys["MMLU"][t][m_idx], catalog_bundle, t_agg)
                        t_mats_g.append(Y_g)
                        t_conf_g.append(Y_g[:, 1::2])
                        yd0 = Y_g[:, ::2].mean(axis=1)[:, None]
                        yc0 = Y_g[:, 1::2].mean(axis=1)[:, None]
                        orc0, _, _ = select_with_tie_policy(yd0, yc0, None, tie_policy="mean_tied")
                        orc_g_mat[m_idx, t_idx] = float(np.squeeze(orc0))
                        mn_g_mat[m_idx, t_idx] = float(yc0.mean())
                        rg = _evaluate_subset_predictor_vectorized(
                            yd0, yc0, X_g, sl_g, P_g, subs_g, X_g, mode="ols"
                        )
                        c_acc_mat[m_idx, t_idx] = float(rg["conf_screen_best"][0])
                        m_acc_mat[m_idx, t_idx] = float(rg["conf_model_best"][0])
                        for mn in metric_names:
                            agg_vals[mn].append(float(rg[mn][0]))

                    yp_d = concatenate_task_discovery_items(t_mats_g).mean(axis=1)[:, None]
                    yp_c = np.hstack(t_conf_g).mean(axis=1)[:, None]
                    rg_p = _evaluate_subset_predictor_vectorized(
                        yp_d, yp_c, X_g, sl_g, P_g, subs_g, X_g, mode="ols"
                    )
                    for mn in metric_names:
                        pool_vals[mn].append(float(rg_p[mn][0]))

                ci_sb_g = two_way_cluster_bootstrap_ci(c_acc_mat, n_boot=n_boot, seed=seeds["cluster_bootstrap"])
                ci_mb_g = two_way_cluster_bootstrap_ci(m_acc_mat, n_boot=n_boot, seed=seeds["cluster_bootstrap"])
                rk_m = float(meta_g.get("rank_mean", meta_g.get("rank", 22)))
                cd_m = float(meta_g.get("condition_number_mean", meta_g.get("condition_number", np.nan)))
                cd_md = float(meta_g.get("condition_number_median", meta_g.get("condition_number", np.nan)))
                ld_m = float(meta_g.get("logdet_xtx_mean", meta_g.get("logdet_xtx", np.nan)))
                tr_m = float(meta_g.get("trace_inv_xtx_mean", meta_g.get("trace_inv_xtx", np.nan)))

                csv_rows.append(
                    {
                        "catalog_granularity": gran_label,
                        "template_agg": t_agg,
                        "evaluation_level": "task_855_cubes",
                        "split_regime": "split0_even_odd",
                        "budget": B_g,
                        "basis_id": "additive_22p",
                        "n_parameters": 22,
                        "method_id": m_id_g,
                        "method_family": fam_g,
                        "outcome_blind_design": True,
                        "query_budget_per_item_equiv": B_g,
                        "n_designs_evaluated": len(subs_g),
                        "matrix_rank_mean": rk_m,
                        "is_full_rank_rate": 1.0,
                        "condition_number_mean": cd_m,
                        "condition_number_median": cd_md,
                        "logdet_xtx_mean": ld_m,
                        "trace_inv_xtx_mean": tr_m,
                        "contrast_rmse_vs_disc_truth": float(np.mean(agg_vals["contrast_rmse_disc"])),
                        "contrast_rmse_vs_conf_truth": float(np.mean(agg_vals["contrast_rmse_conf"])),
                        "contrast_sign_agreement_disc": float(np.mean(agg_vals["contrast_sign_disc"])),
                        "contrast_rmse_op_disc": float(np.mean(agg_vals["contrast_rmse_op_disc"])),
                        "contrast_rmse_space_disc": float(np.mean(agg_vals["contrast_rmse_space_disc"])),
                        "contrast_rmse_sep_disc": float(np.mean(agg_vals["contrast_rmse_sep_disc"])),
                        "heldout_pred_rmse_conf": float(np.mean(agg_vals["heldout_pred_rmse_conf"])),
                        "heldout_pred_rmse_disc": float(np.mean(agg_vals["heldout_pred_rmse_disc"])),
                        "all_templates_pred_rmse_conf": float(np.mean(agg_vals["all_pred_rmse_conf"])),
                        "conf_acc_screen_best": float(c_acc_mat.mean()),
                        "conf_acc_model_best": float(m_acc_mat.mean()),
                        "conf_acc_model_best_in_screen": float(np.mean(agg_vals["conf_model_best_in_screen"])),
                        "conf_acc_screen_best_first_index": float(np.mean(agg_vals["conf_screen_best_fi"])),
                        "conf_acc_model_best_first_index": float(np.mean(agg_vals["conf_model_best_fi"])),
                        "conf_acc_oracle_full": float(orc_g_mat.mean()),
                        "conf_acc_catalog_mean": float(mn_g_mat.mean()),
                        "regret_screen_best_vs_oracle": float(orc_g_mat.mean() - c_acc_mat.mean()),
                        "regret_model_best_vs_oracle": float(orc_g_mat.mean() - m_acc_mat.mean()),
                        "ci95_low_conf_screen_best": float(ci_sb_g["ci_low"]),
                        "ci95_high_conf_screen_best": float(ci_sb_g["ci_high"]),
                        "ci95_low_conf_model_best": float(ci_mb_g["ci_low"]),
                        "ci95_high_conf_model_best": float(ci_mb_g["ci_high"]),
                    }
                )
                csv_rows.append(
                    {
                        "catalog_granularity": gran_label,
                        "template_agg": t_agg,
                        "evaluation_level": "pooled_15_models",
                        "split_regime": "split0_even_odd",
                        "budget": B_g,
                        "basis_id": "additive_22p",
                        "n_parameters": 22,
                        "method_id": m_id_g,
                        "method_family": fam_g,
                        "outcome_blind_design": True,
                        "query_budget_per_item_equiv": B_g,
                        "n_designs_evaluated": len(subs_g),
                        "matrix_rank_mean": rk_m,
                        "is_full_rank_rate": 1.0,
                        "condition_number_mean": cd_m,
                        "condition_number_median": cd_md,
                        "logdet_xtx_mean": ld_m,
                        "trace_inv_xtx_mean": tr_m,
                        "contrast_rmse_vs_disc_truth": float(np.mean(pool_vals["contrast_rmse_disc"])),
                        "contrast_rmse_vs_conf_truth": float(np.mean(pool_vals["contrast_rmse_conf"])),
                        "contrast_sign_agreement_disc": float(np.mean(pool_vals["contrast_sign_disc"])),
                        "contrast_rmse_op_disc": float(np.mean(pool_vals["contrast_rmse_op_disc"])),
                        "contrast_rmse_space_disc": float(np.mean(pool_vals["contrast_rmse_space_disc"])),
                        "contrast_rmse_sep_disc": float(np.mean(pool_vals["contrast_rmse_sep_disc"])),
                        "heldout_pred_rmse_conf": float(np.mean(pool_vals["heldout_pred_rmse_conf"])),
                        "heldout_pred_rmse_disc": float(np.mean(pool_vals["heldout_pred_rmse_disc"])),
                        "all_templates_pred_rmse_conf": float(np.mean(pool_vals["all_pred_rmse_conf"])),
                        "conf_acc_screen_best": float(np.mean(pool_vals["conf_screen_best"])),
                        "conf_acc_model_best": float(np.mean(pool_vals["conf_model_best"])),
                        "conf_acc_model_best_in_screen": float(np.mean(pool_vals["conf_model_best_in_screen"])),
                        "conf_acc_screen_best_first_index": float(np.mean(pool_vals["conf_screen_best_fi"])),
                        "conf_acc_model_best_first_index": float(np.mean(pool_vals["conf_model_best_fi"])),
                        "conf_acc_oracle_full": float("nan"),
                        "conf_acc_catalog_mean": float("nan"),
                        "regret_screen_best_vs_oracle": float("nan"),
                        "regret_model_best_vs_oracle": float("nan"),
                        "ci95_low_conf_screen_best": np.nan,
                        "ci95_high_conf_screen_best": np.nan,
                        "ci95_low_conf_model_best": np.nan,
                        "ci95_high_conf_model_best": np.nan,
                    }
                )

    df_results = pd.DataFrame(csv_rows)

    # Compute paired two-way cluster bootstrap comparisons on `canonical_first_89` across 20 random splits
    cm_p = primary_panel_store["cube_metrics"]
    pm_p = primary_panel_store["pool_metrics"]

    def _paired_diff_20splits(key_a: str, metric_a: str, key_b: str, metric_b: str) -> Dict[str, Any]:
        mat_a = cm_p[key_a][metric_a][:, :, 1:].mean(axis=2)
        mat_b = cm_p[key_b][metric_b][:, :, 1:].mean(axis=2)
        diff = mat_a - mat_b
        ci = two_way_cluster_bootstrap_ci(diff, n_boot=n_boot, seed=seeds["cluster_bootstrap"])
        return {
            "mean_a": float(mat_a.mean()),
            "mean_b": float(mat_b.mean()),
            "diff_a_minus_b": float(ci["estimate"]),
            "diff_pp": float(ci["estimate"] * 100.0),
            "ci95_low": float(ci["ci_low"]),
            "ci95_high": float(ci["ci_high"]),
            "ci95_low_pp": float(ci["ci_low"] * 100.0),
            "ci95_high_pp": float(ci["ci_high"] * 100.0),
            "se": float(ci["se"]),
            "cube_win_count_a_gt_b": int(np.sum(mat_a > mat_b + 1e-12)),
            "cube_win_count_a_lt_b": int(np.sum(mat_a < mat_b - 1e-12)),
        }

    def _attribution_ratio(key_opt: str, key_comp: str) -> Dict[str, Any]:
        task_opt_20 = cm_p[key_opt]["contrast_rmse_disc"][:, :, 1:].mean(axis=2)
        task_cmp_20 = cm_p[key_comp]["contrast_rmse_disc"][:, :, 1:].mean(axis=2)
        pool_opt = pm_p[key_opt]["contrast_rmse_disc"]
        pool_cmp = pm_p[key_comp]["contrast_rmse_disc"]
        diff_ci = two_way_cluster_bootstrap_ci(
            task_cmp_20 - task_opt_20, n_boot=n_boot, seed=seeds["cluster_bootstrap"]
        )
        return {
            "task_20splits_rmse_opt": float(task_opt_20.mean()),
            "task_20splits_rmse_comp": float(task_cmp_20.mean()),
            "task_20splits_ratio_comp_over_opt": float(task_cmp_20.mean() / task_opt_20.mean()),
            "task_cube_wins_opt": int(np.sum(task_opt_20 < task_cmp_20 - 1e-12)),
            "task_rmse_reduction_ci95": [float(diff_ci["ci_low"]), float(diff_ci["ci_high"])],
            "pooled_rmse_opt": float(pool_opt.mean()),
            "pooled_rmse_comp": float(pool_cmp.mean()),
            "pooled_ratio_comp_over_opt": float(pool_cmp.mean() / pool_opt.mean()),
            "pooled_model_wins_opt": int(np.sum(pool_opt < pool_cmp - 1e-12)),
        }

    def _strip_X(b_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        return {
            k: {kk: vv for kk, vv in v.items() if kk != "X"}
            for k, v in b_info.items()
        }

    def _summarize_design_entry(d_entry: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d_entry.items() if k != "subsets"}

    summary: Dict[str, Any] = {
        "task_id": "Phase0_Task_R2_Mixed_Level_Analysis",
        "source_hashes": source_hashes,
        "seeds": seeds,
        "well_conditioned_caps": {f"{k[0]}_B{k[1]}": v for k, v in WELL_CONDITIONED_CAPS.items()},
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
        },
        "catalog_audit": catalog_audit,
        "basis_diagnostics": {
            "combinations_89": _strip_X(bases_89),
            "unique_templates_93": _strip_X(bases_93),
            "catalog_rows_100": _strip_X(bases_100),
        },
        "selected_designs_89": {
            budget_key: {
                d_key: _summarize_design_entry(d_val)
                for d_key, d_val in b_dict.items()
            }
            for budget_key, b_dict in designs_89.items()
        },
        "headline_comparisons": {
            "attribution_contrast_rmse_ratios": {
                "B16_aopt_vs_rand_fullrank_13p": _attribution_ratio("B16_aopt_13p" if "B16_aopt_13p" in cm_p else "B16_a_opt_13p", "B16_rand_fullrank_13p"),
                "B16_dopt_vs_rand_fullrank_13p": _attribution_ratio("B16_d_opt_13p", "B16_rand_fullrank_13p"),
                "B16_aopt_vs_rand_wellcond_13p": _attribution_ratio("B16_a_opt_13p", "B16_rand_wellcond_13p"),
                "B16_aopt_vs_binary_2_5_1_V_avg": _attribution_ratio("B16_a_opt_13p", "B16_binary_res5_avg"),
                "B16_aopt_vs_pe_dopt_subset": _attribution_ratio("B16_a_opt_13p", "B16_pe_dopt_subset"),
                "B24_aopt_vs_rand_fullrank_22p": _attribution_ratio("B24_a_opt_22p", "B24_rand_fullrank_22p"),
                "B24_dopt_vs_rand_fullrank_22p": _attribution_ratio("B24_d_opt_22p", "B24_rand_fullrank_22p"),
                "B24_aopt_vs_rand_wellcond_22p": _attribution_ratio("B24_a_opt_22p", "B24_rand_wellcond_22p"),
                "B32_aopt_vs_rand_fullrank_22p": _attribution_ratio("B32_a_opt_22p", "B32_rand_fullrank_22p"),
                "B32_dopt_vs_rand_fullrank_22p": _attribution_ratio("B32_d_opt_22p", "B32_rand_fullrank_22p"),
                "B32_aopt_vs_rand_wellcond_22p": _attribution_ratio("B32_a_opt_22p", "B32_rand_wellcond_22p"),
                "B32_aopt_vs_binary_cube32_5bit": _attribution_ratio("B32_a_opt_22p", "B32_binary_cube32_5bit"),
                "B32_aopt_vs_binary_cube32_ols22": _attribution_ratio("B32_a_opt_22p", "B32_binary_cube32_ols22"),
            },
            "heldout_prediction_rmse_paired_20splits": {
                "B16_dopt_vs_rand_fullrank_conf": _paired_diff_20splits("B16_d_opt_13p", "heldout_pred_rmse_conf", "B16_rand_fullrank_13p", "heldout_pred_rmse_conf"),
                "B16_pe_dopt_vs_dopt_ols_conf": _paired_diff_20splits("B16_pe_dopt_subset", "heldout_pred_rmse_conf", "B16_d_opt_13p", "heldout_pred_rmse_conf"),
                "B16_dopt_vs_binary_res5_conf": _paired_diff_20splits("B16_d_opt_13p", "heldout_pred_rmse_conf", "B16_binary_res5_avg", "heldout_pred_rmse_conf"),
                "B24_dopt_vs_rand_fullrank_conf": _paired_diff_20splits("B24_d_opt_22p", "heldout_pred_rmse_conf", "B24_rand_fullrank_22p", "heldout_pred_rmse_conf"),
                "B24_pe_dopt_vs_dopt_ols_conf": _paired_diff_20splits("B24_pe_dopt_subset", "heldout_pred_rmse_conf", "B24_d_opt_22p", "heldout_pred_rmse_conf"),
                "B32_dopt_vs_rand_fullrank_conf": _paired_diff_20splits("B32_d_opt_22p", "heldout_pred_rmse_conf", "B32_rand_fullrank_22p", "heldout_pred_rmse_conf"),
                "B32_dopt_vs_binary_cube32_5bit_conf": _paired_diff_20splits("B32_d_opt_22p", "heldout_pred_rmse_conf", "B32_binary_cube32_5bit", "heldout_pred_rmse_conf"),
                "B32_pe_dopt_vs_dopt_ols_conf": _paired_diff_20splits("B32_pe_dopt_subset", "heldout_pred_rmse_conf", "B32_d_opt_22p", "heldout_pred_rmse_conf"),
            },
            "selection_confirmation_accuracy_paired_20splits": {
                "B16_dopt_screenbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B16_d_opt_13p", "conf_screen_best", "B16_rand_uncond", "conf_screen_best"),
                "B16_dopt_screenbest_vs_binary_res5_screenbest": _paired_diff_20splits("B16_d_opt_13p", "conf_screen_best", "B16_binary_res5_avg", "conf_screen_best"),
                "B16_pe_sparse_modelbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B16_pe_sparse_disc", "conf_model_best", "B16_rand_uncond", "conf_screen_best"),
                "B16_pe_dopt_modelbest_vs_dopt_screenbest": _paired_diff_20splits("B16_pe_dopt_subset", "conf_model_best", "B16_d_opt_13p", "conf_screen_best"),
                "B16_dopt_ols_modelbest_vs_dopt_screenbest": _paired_diff_20splits("B16_d_opt_13p", "conf_model_best", "B16_d_opt_13p", "conf_screen_best"),
                "B24_dopt_screenbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B24_d_opt_22p", "conf_screen_best", "B24_rand_uncond", "conf_screen_best"),
                "B24_aopt_screenbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B24_a_opt_22p", "conf_screen_best", "B24_rand_uncond", "conf_screen_best"),
                "B24_pe_sparse_modelbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B24_pe_sparse_disc", "conf_model_best", "B24_rand_uncond", "conf_screen_best"),
                "B24_pe_dopt_modelbest_vs_dopt_screenbest": _paired_diff_20splits("B24_pe_dopt_subset", "conf_model_best", "B24_d_opt_22p", "conf_screen_best"),
                "B32_dopt_screenbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B32_d_opt_22p", "conf_screen_best", "B32_rand_uncond", "conf_screen_best"),
                "B32_dopt_screenbest_vs_binary_cube32_screenbest": _paired_diff_20splits("B32_d_opt_22p", "conf_screen_best", "B32_binary_cube32_5bit", "conf_screen_best"),
                "B32_pe_sparse_modelbest_vs_rand_uncond_screenbest": _paired_diff_20splits("B32_pe_sparse_disc", "conf_model_best", "B32_rand_uncond", "conf_screen_best"),
                "B32_pe_dopt_modelbest_vs_dopt_screenbest": _paired_diff_20splits("B32_pe_dopt_subset", "conf_model_best", "B32_d_opt_22p", "conf_screen_best"),
            },
        },
    }
    return df_results, summary


def write_mixed_level_artifacts(
    df_results: pd.DataFrame,
    summary: Dict[str, Any],
    csv_path: Path = RESULTS_DIR / "mixed_level_results.csv",
    json_path: Path = RESULTS_DIR / "mixed_level_summary.json",
) -> Dict[str, str]:
    """Write deterministic CSV and JSON artifacts with SHA-256 hashes."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(csv_path, index=False, float_format="%.10f")
    csv_sha = sha256_file(csv_path)

    summary_out = dict(summary)
    summary_out["artifact_hashes"] = {
        "mixed_level_results_csv_sha256": csv_sha,
    }
    json_text = json.dumps(summary_out, indent=2, sort_keys=True) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    json_sha = sha256_file(json_path)
    return {
        "mixed_level_results_csv_sha256": csv_sha,
        "mixed_level_summary_json_sha256": json_sha,
    }


def main() -> None:
    df_results, summary = run_mixed_level_study()
    hashes = write_mixed_level_artifacts(df_results, summary)
    print("Mixed-level analysis complete.")
    print("Rows in mixed_level_results.csv:", len(df_results))
    print("Artifact SHA-256 hashes:", json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
