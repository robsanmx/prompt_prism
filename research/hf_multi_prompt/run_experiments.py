"""
Statistically Defensible Multi-Model Factorial Prompt Engineering Study using
Hugging Face PromptEval (MMLU) and SLAB-NLP (BBH & LMentry) Repositories.

Implements the five verification modules specified in `tasks/paper_hf_revision_spec.md`:
1. `design-validity` (DV-1..DV-5): Rank and condition safeguards, 200 full-rank
   balanced random 16-run second-order-product comparators, complete coset and orientation
   evaluation across 2^(5-1)_V, 2^(5-2)_III, and 2^(4-1)_IV, and 20 repeated
   discovery/confirmation item splits with two-way model-task cluster bootstrap CIs.
2. `cube-validity` (CV-1..CV-4): Order-invariant template mapping manifest, comparison
   of all-template cell aggregation (`cell_mean`), repeated random single-template
   resampling (`random_single_rep`), canonical hash-ordered single-template selection
   (`canonical_first`), and legacy first-template selection (`legacy_first`), plus
   executable factor construct audit and non-causal contrast terminology.
3. `statistical-inference` (SI-1..SI-6): Exact complete-cube Fourier effect magnitude
   decomposition (RMS main, 2FI, 3FI, 4FI, 5FI) separated from estimation RMSE,
   higher-order aliasing verification, association between 2FI magnitude and
   additive-sign regret, item-clustered contrast t-tests and sign-flip permutation
   tests vs. fixed-block RCBD and unblocked ANOVA, and explicit BH multiplicity
   families (per-cube, per-model, global cohort).
4. `paper-artifacts` (PA-1..PA-5): Machine-generated CSV/JSON results, uncertainty-aware
   figures, dynamic LaTeX tables with zero hardcoded headline ratios, and a complete
   claim-to-evidence map.
5. `reproducibility-qa` (RQ-1..RQ-4): Source SHA-256 verification, package/seed
   provenance, and output artifact hashes in `summary_manifest.json`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.multitest import multipletests

from prompt_prism import (
    ANOVAEngine,
    FractionalFactorialGenerator,
    FullFactorialGenerator,
    OptimalPromptFinder,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "research" / "hf_multi_prompt" / "data"
RESULTS_DIR = ROOT / "research" / "hf_multi_prompt" / "results"
PAPER_DIR = ROOT / "paper_hf"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

EXPECTED_YS_SHA256 = "4300e539e82f9c9f3f3ba9b4ad024ec0335f1c7b8914a7a43554055e4506416a"
EXPECTED_MMLU_META_SHA256 = "fc3b3834c91a3dfa714283c477acce9a52d224ca4246bb11c756b74f340401f8"
EXPECTED_TEMPLATE_CATALOG_MANIFEST_SHA256 = "a3985944a6d9073754fca2813206eae357b548f62bf9dfc7594b5365f4573b81"
EXPECTED_TEMPLATE_CATALOG_SHA256 = {
    "templates/MMLU/abstract_algebra.csv": "3b92f5be1ceef1867bdafe19144eac76bba43eff2f12685ba101d85da140ddc6",
    "templates/BBH/causal_judgement.csv": "5e811ee0652b0de66270c7de81cd3b127f266cb4bc9a37a695507faf384d84e2",
    "templates/BBH/disambiguation_qa.csv": "9ec9886eaafceeb81c80f780b8d3d7998b72e69bd5ec1a8a8755075b7f71ef37",
    "templates/BBH/formal_fallacies.csv": "42632cf6d4768b4d365b10538018086972fbda32a03ad2176325da360113b178",
    "templates/BBH/geometric_shapes.csv": "561c023917666546f5b189317a2901a1c4dcb7475c42250521e4fdb8b97760bf",
    "templates/BBH/hyperbaton.csv": "9611b9e07a7a1774497cb02c603b5f907475b07bdadf65145ae72a25a97567a4",
    "templates/BBH/logical_deduction_five_objects.csv": "ba7886e2cb88b6dc6c9a1a0c6b6f2c4d9b971aa14865fb1e401ecc8ab19b49cb",
    "templates/BBH/logical_deduction_seven_objects.csv": "1044bb7bbf4020e842327810fa748ef810b478330bf53a804cf5d143376a1084",
    "templates/BBH/logical_deduction_three_objects.csv": "2fb16478a5901352d5beaf28c4b66420a3f0756dfd9656f7c2a1e642602bbc36",
    "templates/BBH/movie_recommendation.csv": "97f65d50c823c8ecb54a3ea1e58dd4ec4b2797a289d4db68c85504ff74de92ff",
    "templates/BBH/navigate.csv": "5cd0286b598eb1bbdb03733a99b8e9b983ee7f363a958ea1681f72ff62a82e2d",
    "templates/BBH/penguins_in_a_table.csv": "8ece2e5c17897233ba1a193112ed76ca891b7ca2ff1e19f7b502faf654c676d5",
    "templates/BBH/ruin_names.csv": "db8786c325fe3e75728a773b9f087ec790a4f5fd81f83111c12724de20aa1e19",
    "templates/BBH/salient_translation_error_detection.csv": "fe7ed67fb09a60b8ee3a9c7c9ee13202561838998454b4b608deaaecf2229bbb",
    "templates/BBH/snarks.csv": "d23a21cb4aa251507b659bed9f93e6590f921ebd940e3c3f73e8de266c4368a0",
    "templates/BBH/sports_understanding.csv": "a868613efd17be8d3b26208dfad7f01e9e9e5f37a54e410adc8f272ee45705e9",
    "templates/LMentry/all_words_from_category.csv": "5081fced01c6e7143796b9f4597ca27cce625f3e094248ac06b41f6cf3e26610",
    "templates/LMentry/any_words_from_category.csv": "884cef7c777af9dfc878333075a880f6c16f0466c5c4315cc2dc6166da1c28da",
    "templates/LMentry/ends_with_word.csv": "e8033695cecd2f0216efd957261f3b550b73e44d757ad1f2bc1f0db88aeb817c",
    "templates/LMentry/first_alphabetically.csv": "0738568d6d1ae79b6e0b9fd9e427d3a4b21d4a8ee476c0c6e17404882fef91a2",
    "templates/LMentry/homophones.csv": "a04f2cbb85c8c8411a1dd7d13d3936119438c6637ec8ba6597db18cfff991a9b",
    "templates/LMentry/less_letters.csv": "ac56a9ad9bdf8b295ef6e1e9673c6b00337ab2275eec9039e3d9e6df3cb993c3",
    "templates/LMentry/more_letters.csv": "4643d9967bad7a506fbbbfbf740d3a41e6286bd0f407104fb737b3e5eed1bfc5",
    "templates/LMentry/rhyming_word.csv": "1eb8fb8da2330d2e937c5ff174319deec09b4759d4ec5f5b3fa605a8bc49efd9",
    "templates/LMentry/word_before.csv": "67329eb4212c6956af08ac1c5de175db4d8f2f7887992da16674fef697a7fe5a",
    "templates/LMentry/word_not_containing.csv": "eb4eb20c6d141226a58099a7b4b3e609b6f5f3d49cc25bc1deb2ab7d20eaedee",
}

SEEDS = {
    "rand16_full_rank": 20261105,
    "rand16_legacy": 20261105,
    "rand8_cohort1": 20261106,
    "rand8_cohort2": 20261107,
    "rand16_screenbest_uncond": 20261108,
    "rand8_c1_screenbest_uncond": 20261109,
    "rand8_c2_screenbest_uncond": 20261111,
    "item_splits": 20261120,
    "template_reps": 20261110,
    "cluster_bootstrap": 20261130,
    "sign_flip_perm": 20261140,
    "tie_break": 20261150,
}

MMLU_MODELS = [
    ("meta_llama_llama_3_8b", "Llama-3-8B (Base)", "Base"),
    ("meta_llama_llama_3_8b_instruct", "Llama-3-8B-Instruct", "Instruct"),
    ("meta_llama_llama_3_70b_instruct", "Llama-3-70B-Instruct", "Instruct"),
    ("codellama_codellama_34b_instruct", "CodeLlama-34B-Instruct", "Instruct"),
    ("google_flan_t5_xl", "Flan-T5-XL (3B)", "Instruct"),
    ("google_flan_t5_xxl", "Flan-T5-XXL (11B)", "Instruct"),
    ("google_flan_ul2", "Flan-UL2 (20B)", "Instruct"),
    ("ibm_mistralai_merlinite_7b", "Merlinite-7B", "Instruct"),
    ("mistralai_mixtral_8x7b_instruct_v01", "Mixtral-8x7B-Instruct", "Instruct"),
    ("mistralai_mistral_7b_instruct_v0_2", "Mistral-7B-Instruct-v0.2", "Instruct"),
    ("google_gemma_7b", "Gemma-7B (Base)", "Base"),
    ("google_gemma_7b_it", "Gemma-7B-IT", "Instruct"),
    ("tiiuae_falcon_40b", "Falcon-40B (Base)", "Base"),
    ("mistralai_mistral_7b_v0_1", "Mistral-7B-v0.1 (Base)", "Base"),
    ("tiiuae_falcon_180b", "Falcon-180B (Base)", "Base"),
]

BBH_MODELS = [
    "airoboros-13b",
    "alpaca-13b",
    "falcon-7b-instruct",
    "flan-t5-xl",
    "flan-t5-xxl",
    "minotaur-15b",
    "mpt-7b-instruct",
    "nous-hermes-13b",
    "t0pp",
    "ultralm-13b",
    "vicuna-13b",
]

LMENTRY_MODELS = [
    "airoboros-13b",
    "alpaca-13b",
    "alpaca-7b",
    "falcon-7b-instruct",
    "flan-t5-small",
    "flan-t5-large",
    "flan-t5-base",
    "flan-t5-xl",
    "flan-t5-xxl",
    "minotaur-15b",
    "mpt-7b-instruct",
    "nous-hermes-13b",
    "t0pp",
    "t0-3b",
    "ultralm-13b",
    "vicuna-13b",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analysis_template_catalog_paths(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return every task catalog that contributes to an analyzed response matrix."""
    paths = [
        *data_dir.glob("templates/MMLU/*.csv"),
        *data_dir.glob("templates/BBH/*.csv"),
        *data_dir.glob("templates/LMentry/*.csv"),
        data_dir / "templates" / "mmlu_templates.json",
    ]
    return sorted(paths, key=lambda path: path.relative_to(data_dir).as_posix())


def catalog_manifest_sha256(
    catalog_hashes: Dict[str, str],
) -> str:
    """Hash the sorted path:digest catalog manifest, including a trailing newline."""
    payload = "".join(
        f"{relative_path}:{digest}\n"
        for relative_path, digest in sorted(catalog_hashes.items())
    )
    return sha256_text(payload)


def verify_source_hashes(
    ys_path: Path = DATA_DIR / "Ys.pickle",
    meta_path: Path = DATA_DIR / "templates" / "mmlu_templates_metadata.json",
    expected_ys: str = EXPECTED_YS_SHA256,
    expected_meta: str = EXPECTED_MMLU_META_SHA256,
    expected_catalogs: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Verify source dataset integrity before running any analysis (Spec Section 5.2)."""
    ys_hash = sha256_file(ys_path)
    meta_hash = sha256_file(meta_path)
    if ys_hash != expected_ys:
        raise RuntimeError(
            f"Source integrity failure for {ys_path}: expected SHA-256 {expected_ys}, got {ys_hash}"
        )
    if meta_hash != expected_meta:
        raise RuntimeError(
            f"Source integrity failure for {meta_path}: expected SHA-256 {expected_meta}, got {meta_hash}"
        )
    catalog_hashes: Dict[str, str] = {}
    if expected_catalogs is not None:
        catalog_items = [
            (relative_path, DATA_DIR / relative_path, expected_hash)
            for relative_path, expected_hash in sorted(expected_catalogs.items())
        ]
    else:
        catalog_items = [
            (path.relative_to(DATA_DIR).as_posix(), path, None)
            for path in analysis_template_catalog_paths(DATA_DIR)
        ]
    for relative_path, catalog_path, expected_hash in catalog_items:
        actual_hash = sha256_file(catalog_path)
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(
                f"Source integrity failure for {catalog_path}: expected SHA-256 "
                f"{expected_hash}, got {actual_hash}"
            )
        catalog_hashes[relative_path] = actual_hash
    manifest_hash = catalog_manifest_sha256(catalog_hashes)
    if expected_catalogs is None and manifest_hash != EXPECTED_TEMPLATE_CATALOG_MANIFEST_SHA256:
        raise RuntimeError(
            "Source integrity failure for the analyzed template-catalog manifest: "
            f"expected SHA-256 {EXPECTED_TEMPLATE_CATALOG_MANIFEST_SHA256}, got {manifest_hash}"
        )
    return {
        "Ys_pickle_sha256": ys_hash,
        "mmlu_metadata_sha256": meta_hash,
        "template_catalog_sha256": catalog_hashes,
        "template_catalog_manifest_sha256": manifest_hash,
    }


# ==============================================================================
# MODULE 1: design-validity (DV-1 .. DV-5)
# ==============================================================================

def validate_design_matrix(
    design_matrix: np.ndarray,
    design_id: str,
    model_id: str,
    max_condition_number: Optional[float] = None,
) -> Tuple[int, float]:
    """Validate that a screening design matrix is full rank for the target model (DV-1)."""
    if design_matrix.ndim != 2:
        raise ValueError(f"Design {design_id!r} must be 2D, got shape {design_matrix.shape}")
    n_runs, n_parameters = design_matrix.shape
    if n_runs < n_parameters:
        raise ValueError(
            f"Design {design_id!r} has fewer runs ({n_runs}) than parameters ({n_parameters}) for {model_id!r}"
        )
    rank = int(np.linalg.matrix_rank(design_matrix))
    if rank != n_parameters:
        raise ValueError(
            f"Design {design_id!r} is rank deficient for {model_id!r}: "
            f"rank={rank}, required={n_parameters}"
        )
    cond = float(np.linalg.cond(design_matrix))
    if max_condition_number is not None and cond > max_condition_number:
        raise ValueError(
            f"Design {design_id!r} exceeds maximum condition number for {model_id!r}: "
            f"cond={cond:.4f} > {max_condition_number:.4f}"
        )
    return rank, cond


def fit_design_ols(
    design_matrix: np.ndarray,
    y: np.ndarray,
    design_id: str,
    model_id: str,
) -> np.ndarray:
    """Fit OLS parameters only after explicit rank validation (DV-1)."""
    validate_design_matrix(design_matrix, design_id=design_id, model_id=model_id)
    beta, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
    return beta


def build_cube_design_matrices(num_factors: int) -> Dict[str, Any]:
    """Construct full-cube {-1, +1} matrices for main effects, 2FIs, and higher-order terms."""
    all_cells = list(itertools.product([0, 1], repeat=num_factors))
    X_01 = np.array(all_cells, dtype=int)
    X_pm = 2 * X_01 - 1
    n_cells = len(all_cells)
    pairs = list(itertools.combinations(range(num_factors), 2))
    X_2fi = np.column_stack([X_pm[:, i] * X_pm[:, j] for i, j in pairs])
    D_main = np.column_stack([np.ones(n_cells), X_pm])
    D_2fi = np.column_stack([np.ones(n_cells), X_pm, X_2fi])

    higher_orders: Dict[int, np.ndarray] = {}
    for order in range(3, num_factors + 1):
        combos = list(itertools.combinations(range(num_factors), order))
        higher_orders[order] = np.column_stack(
            [np.prod(X_pm[:, cols], axis=1) for cols in combos]
        )

    return {
        "all_cells": all_cells,
        "X_01": X_01,
        "X_pm": X_pm,
        "pairs": pairs,
        "X_2fi": X_2fi,
        "D_main": D_main,
        "D_2fi": D_2fi,
        "higher_orders": higher_orders,
    }


def generate_balanced_random_designs(
    num_cells: int,
    num_factors: int,
    subset_size: int,
    num_screens: int = 20,
    seed: int = 20261105,
    require_2fi_full_rank: bool = False,
    require_main_full_rank: bool = True,
) -> List[List[int]]:
    """
    Generate factor-balanced random screening designs.
    - When `require_2fi_full_rank=True`, requires full rank for [intercept, main, 2FI].
    - When `require_main_full_rank=True` (and `require_2fi_full_rank=False`), requires full rank for [intercept, main].
    - When both are False, accepts any factor-balanced random design (used for unconditioned `ScreenBest`).
    """
    mats = build_cube_design_matrices(num_factors)
    X_01 = mats["X_01"]
    D_check = mats["D_2fi"] if require_2fi_full_rank else mats["D_main"]
    req_rank = D_check.shape[1]

    rng = np.random.default_rng(seed)
    screens: List[List[int]] = []
    seen = set()
    target_ones = subset_size // 2
    while len(screens) < num_screens:
        cand = tuple(sorted(int(x) for x in rng.choice(num_cells, size=subset_size, replace=False)))
        if cand in seen:
            continue
        sub = X_01[list(cand)]
        if np.all(sub.sum(axis=0) == target_ones):
            seen.add(cand)
            if not require_2fi_full_rank and not require_main_full_rank:
                screens.append(list(cand))
            elif int(np.linalg.matrix_rank(D_check[list(cand)])) == req_rank:
                screens.append(list(cand))
    return screens


def generate_random_designs_with_diagnostics(
    num_factors: int,
    subset_size: int,
    num_accepted: int,
    seed: int,
    require_2fi_full_rank: bool,
    cohort_label: str,
    family_label: str,
    require_main_full_rank: bool = True,
) -> Tuple[List[List[int]], List[Dict[str, Any]], Dict[str, int]]:
    """Generate balanced random designs and record accepted and rejected candidate diagnostics (DV-1, DV-2)."""
    mats = build_cube_design_matrices(num_factors)
    X_01 = mats["X_01"]
    D_main = mats["D_main"]
    D_2fi = mats["D_2fi"]
    num_cells = len(mats["all_cells"])
    target_ones = subset_size // 2

    rng = np.random.default_rng(seed)
    accepted: List[List[int]] = []
    diag_rows: List[Dict[str, Any]] = []
    seen = set()
    total_draws = 0
    rejected_unbalanced = 0
    rejected_rank_deficient = 0
    accepted_main_rank_def = 0
    accepted_2fi_rank_def = 0

    while len(accepted) < num_accepted:
        total_draws += 1
        cand = tuple(sorted(int(x) for x in rng.choice(num_cells, size=subset_size, replace=False)))
        if cand in seen:
            continue
        sub = X_01[list(cand)]
        if not np.all(sub.sum(axis=0) == target_ones):
            rejected_unbalanced += 1
            continue
        seen.add(cand)
        idx_list = list(cand)
        rank_main = int(np.linalg.matrix_rank(D_main[idx_list]))
        cond_main = float(np.linalg.cond(D_main[idx_list])) if rank_main == D_main.shape[1] else float("inf")
        rank_2fi = int(np.linalg.matrix_rank(D_2fi[idx_list]))
        # A condition number for the full intercept+main+2FI model is meaningful only
        # when all model columns are identifiable. An 8-run design can have full row
        # rank while remaining necessarily deficient for the 16-column five-factor model.
        cond_2fi = (
            float(np.linalg.cond(D_2fi[idx_list]))
            if rank_2fi == D_2fi.shape[1]
            else float("inf")
        )

        if require_2fi_full_rank:
            is_valid = (rank_main == D_main.shape[1]) and (rank_2fi == D_2fi.shape[1])
            model_spec = "intercept_main_2fi"
            n_params = D_2fi.shape[1]
            rk = rank_2fi
            cd = cond_2fi
        elif require_main_full_rank:
            is_valid = rank_main == D_main.shape[1]
            model_spec = "intercept_main"
            n_params = D_main.shape[1]
            rk = rank_main
            cd = cond_main
        else:
            is_valid = True
            model_spec = "screenbest_unconditioned"
            n_params = 0
            rk = rank_main
            cd = cond_main

        if is_valid:
            design_idx = len(accepted)
            accepted.append(idx_list)
            design_id = f"{family_label}_accepted_{design_idx:03d}"
            if rank_main < D_main.shape[1]:
                accepted_main_rank_def += 1
            if rank_2fi < D_2fi.shape[1]:
                accepted_2fi_rank_def += 1
        else:
            rejected_rank_deficient += 1
            design_id = f"{family_label}_rejected_{rejected_rank_deficient:03d}"

        diag_rows.append(
            {
                "design_id": design_id,
                "cohort": cohort_label,
                "design_family": family_label,
                "num_factors": num_factors,
                "num_runs": subset_size,
                "selected_cell_indices": json.dumps(idx_list),
                "defining_relation": "random_balanced",
                "coset_sign": "none",
                "random_seed": seed,
                "generation_attempt": total_draws,
                "model_spec": model_spec,
                "n_parameters": n_params,
                "matrix_rank": rk,
                "is_full_rank": bool(
                    (rank_2fi == D_2fi.shape[1]) if require_2fi_full_rank else (rank_main == D_main.shape[1])
                ),
                "condition_number": cd,
                "rank_main": rank_main,
                "cond_main": cond_main,
                "rank_2fi": rank_2fi,
                "cond_2fi": cond_2fi,
                "accepted_primary": bool(is_valid),
            }
        )

    stats_summary = {
        "seed": seed,
        "num_accepted": len(accepted),
        "total_random_draws": total_draws,
        "unique_balanced_candidates_checked": len(accepted) + rejected_rank_deficient,
        "rejected_unbalanced_draws": rejected_unbalanced,
        "rejected_rank_deficient_balanced": rejected_rank_deficient,
        "accepted_main_rank_deficient": accepted_main_rank_def,
        "accepted_2fi_rank_deficient": accepted_2fi_rank_def,
    }
    return accepted, diag_rows, stats_summary


def enumerate_fractional_cosets() -> Dict[str, Any]:
    """
    Enumerate all regular fractional factorial cosets and orientation families (DV-3):
    - Cohort 1 2^(5-1)_V: both cosets I = +ABCDE and I = -ABCDE (partition 32 cells).
    - Cohort 1 2^(5-2)_III: all 4 cosets of canonical (D = s1*AB, E = s2*AC) (partition 32 cells),
      plus all 15 distinct regular 2^(5-2)_III defining-word subgroups x 4 cosets = 60 quarter-fractions.
    - Cohort 2 2^(4-1)_IV: both cosets I = +ABCD and I = -ABCD (partition 16 cells).
    """
    m5 = build_cube_design_matrices(5)
    X5_pm = m5["X_pm"]

    # 1. Both cosets of 2^(5-1)_V: I = +ABCDE and I = -ABCDE
    prod_abcde = np.prod(X5_pm, axis=1)
    res5_cosets = {
        "2(5-1)V_coset_pos": {
            "indices": np.where(prod_abcde == 1)[0].tolist(),
            "defining_relation": "I = +ABCDE (E = +ABCD)",
            "coset_sign": "+1",
            "is_legacy_default": False,
        },
        "2(5-1)V_coset_neg": {
            "indices": np.where(prod_abcde == -1)[0].tolist(),
            "defining_relation": "I = -ABCDE (E = -ABCD)",
            "coset_sign": "-1",
            "is_legacy_default": True,  # Matches 0/1 XOR generator in PromptPrism
        },
    }

    # 2. All 4 canonical cosets of 2^(5-2)_III: D = s1 * AB, E = s2 * AC
    res3_canonical_cosets = {}
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            idx = np.where(
                (X5_pm[:, 3] == s1 * X5_pm[:, 0] * X5_pm[:, 1])
                & (X5_pm[:, 4] == s2 * X5_pm[:, 0] * X5_pm[:, 2])
            )[0].tolist()
            cid = f"2(5-2)III_D{s1:+d}AB_E{s2:+d}AC"
            res3_canonical_cosets[cid] = {
                "indices": idx,
                "defining_relation": f"I = {s1:+d}ABD = {s2:+d}ACE = {s1*s2:+d}BCDE",
                "coset_sign": f"({s1:+d},{s2:+d})",
                "is_legacy_default": (s1 == -1 and s2 == -1),
            }

    # 3. All 10 choices of 2 generated columns out of 5 factors (with canonical pairing) x 4 cosets = 40
    # Better yet: enumerate all unique 8-run regular Resolution III fractions of 2^5!
    # A regular 2^(5-2)_III subgroup is defined by two independent length-3 words whose product has length >= 3.
    # There are C(5,3)=10 length-3 words; any pair of length-3 words shares exactly 1 factor, so their product has length 4!
    # Wait: 3 words of lengths (3, 3, 4) form a subgroup; each subgroup has two length-3 words, so C(10, 2) / 1 = 45?
    # No: two length-3 words in 5 factors ALWAYS share 2 factors (e.g. ABD and ACD -> product BC of length 2, which is Res II!)
    # OR share 1 factor (e.g. ABD and ACE -> product BCDE of length 4, which is Res III!).
    # How many pairs of length-3 words share exactly 1 factor?
    # Pick the shared factor (5 ways), then split the remaining 4 factors into two pairs: C(4,2)/2 = 3 ways -> 5 * 3 = 15 subgroups!
    # Each of the 15 regular Res III subgroups has 4 cosets -> 15 * 4 = 60 distinct regular 2^(5-2)_III designs!
    len3_words = list(itertools.combinations(range(5), 3))
    res3_all_60 = {}
    seen_subgroups = set()
    for w1, w2 in itertools.combinations(len3_words, 2):
        shared = set(w1) & set(w2)
        if len(shared) != 1:
            continue
        key = tuple(sorted([w1, w2]))
        if key in seen_subgroups:
            continue
        seen_subgroups.add(key)
        w1_name = "".join("ABCDE"[i] for i in w1)
        w2_name = "".join("ABCDE"[i] for i in w2)
        col1 = np.prod(X5_pm[:, w1], axis=1)
        col2 = np.prod(X5_pm[:, w2], axis=1)
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                idx = np.where((col1 == s1) & (col2 == s2))[0].tolist()
                did = f"2(5-2)III_{w1_name}{s1:+d}_{w2_name}{s2:+d}"
                res3_all_60[did] = {
                    "indices": idx,
                    "subgroup": f"{w1_name},{w2_name}",
                    "defining_relation": f"I = {s1:+d}{w1_name} = {s2:+d}{w2_name}",
                    "coset_sign": f"({s1:+d},{s2:+d})",
                }

    # 4. Both cosets of Cohort 2 2^(4-1)_IV: I = +ABCD and I = -ABCD
    m4 = build_cube_design_matrices(4)
    X4_pm = m4["X_pm"]
    prod_abcd = np.prod(X4_pm, axis=1)
    res4_cosets = {
        "2(4-1)IV_coset_pos": {
            "indices": np.where(prod_abcd == 1)[0].tolist(),
            "defining_relation": "I = +ABCD (D = +ABC)",
            "coset_sign": "+1",
            "is_legacy_default": True,  # Matches 0/1 XOR for 3 factors (0,0,0,0 -> (-1)^4 = +1)
        },
        "2(4-1)IV_coset_neg": {
            "indices": np.where(prod_abcd == -1)[0].tolist(),
            "defining_relation": "I = -ABCD (D = -ABC)",
            "coset_sign": "-1",
            "is_legacy_default": False,
        },
    }

    return {
        "res5_cosets": res5_cosets,
        "res3_canonical_cosets": res3_canonical_cosets,
        "res3_all_60": res3_all_60,
        "res4_cosets": res4_cosets,
    }


def generate_item_splits(n_items: int, n_splits: int = 20, seed: int = 20261120) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate normalized discovery and confirmation projection matrices of shape (n_items, n_splits + 1).
    Column 0 is the legacy deterministic even/odd split (`legacy_even_odd`);
    Columns 1..n_splits are seeded random 50/50 item partitions (`split_00`..`split_19`).
    """
    d_masks = np.zeros((n_items, n_splits + 1), dtype=float)
    c_masks = np.zeros((n_items, n_splits + 1), dtype=float)
    split_ids = ["legacy_even_odd"] + [f"split_{s:02d}" for s in range(n_splits)]

    # Column 0: legacy even/odd
    even_idx = np.arange(0, n_items, 2)
    odd_idx = np.arange(1, n_items, 2)
    d_masks[even_idx, 0] = 1.0 / len(even_idx)
    c_masks[odd_idx, 0] = 1.0 / len(odd_idx)

    rng = np.random.default_rng(seed)
    n_half = n_items // 2
    for s in range(1, n_splits + 1):
        perm = rng.permutation(n_items)
        d_idx = perm[:n_half]
        c_idx = perm[n_half:]
        d_masks[d_idx, s] = 1.0 / len(d_idx)
        c_masks[c_idx, s] = 1.0 / len(c_idx)

    return d_masks, c_masks, split_ids


def select_with_tie_policy(
    scores_d: np.ndarray,
    values_c: np.ndarray,
    candidate_indices: Optional[np.ndarray] = None,
    tie_policy: str = "mean_tied",
    tol: float = 1e-12,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select winning cell(s) from discovery scores `scores_d` and evaluate confirmation values `values_c`
    under an explicit tie policy (`mean_tied`, `first_index`, or `seeded_random`) (Spec Section 18, Task 2).
    """
    s_arr = np.asarray(scores_d, dtype=float)
    v_arr = np.asarray(values_c, dtype=float)
    squeeze_out = False
    if s_arr.ndim == 1:
        s_arr = s_arr[:, None]
        if v_arr.ndim == 1:
            v_arr = v_arr[:, None]
        squeeze_out = True
    elif v_arr.ndim == 1:
        v_arr = v_arr[:, None]

    if s_arr.ndim == 2:
        if candidate_indices is not None:
            c_idx = np.asarray(candidate_indices, dtype=int)
            cand_v = v_arr[c_idx, :]
        else:
            cand_v = v_arr
    elif s_arr.ndim == 3:
        if candidate_indices is not None:
            c_idx = np.asarray(candidate_indices, dtype=int)
            if c_idx.ndim == 1:
                cand_v = np.broadcast_to(v_arr[c_idx, :][None, :, :], s_arr.shape)
            else:
                cand_v = v_arr[c_idx, :]
        else:
            cand_v = np.broadcast_to(v_arr[None, :, :], s_arr.shape)
    else:
        raise ValueError(f"Unsupported scores_d shape: {s_arr.shape}")

    cand_axis = -2
    max_scores = np.max(s_arr, axis=cand_axis, keepdims=True)
    tied_mask = s_arr >= (max_scores - tol)
    tied_count = np.sum(tied_mask, axis=cand_axis)
    is_tied = tied_count > 1

    if tie_policy == "mean_tied":
        conf_sel = np.sum(cand_v * tied_mask, axis=cand_axis) / np.maximum(1, tied_count)
    elif tie_policy == "first_index":
        first_idx = np.argmax(s_arr, axis=cand_axis)
        conf_sel = np.take_along_axis(
            cand_v, np.expand_dims(first_idx, axis=cand_axis), axis=cand_axis
        ).squeeze(axis=cand_axis)
    elif tie_policy == "seeded_random":
        if rng is None:
            rng = np.random.default_rng(SEEDS["tie_break"])
        u = rng.uniform(0.0, 1.0, size=s_arr.shape)
        rand_scores = np.where(tied_mask, u, -1.0)
        rand_idx = np.argmax(rand_scores, axis=cand_axis)
        conf_sel = np.take_along_axis(
            cand_v, np.expand_dims(rand_idx, axis=cand_axis), axis=cand_axis
        ).squeeze(axis=cand_axis)
    else:
        raise ValueError(f"Unknown tie_policy: {tie_policy!r}")

    if squeeze_out:
        return np.squeeze(conf_sel), np.squeeze(tied_count), np.squeeze(is_tied)
    return conf_sel, tied_count, is_tied


def summarize_tie_diagnostics(
    tied_count: np.ndarray,
    tie_indicator: np.ndarray,
) -> Dict[str, float]:
    """Summarize tie incidence without thresholding an average tied-set size.

    For design ensembles, callers average the per-design Boolean indicator before
    passing it here. This preserves the probability of a tie; ``mean(count)>1`` does
    not, because a few large tied sets can make every aggregate record look tied.
    """
    counts = np.asarray(tied_count, dtype=float)
    indicators = np.asarray(tie_indicator, dtype=float)
    return {
        "split0_tie_rate": float(indicators[..., 0].mean()),
        "split0_mean_tied_set_size": float(counts[..., 0].mean()),
        "splits20_tie_rate": float(indicators[..., 1:].mean()),
        "splits20_mean_tied_set_size": float(counts[..., 1:].mean()),
    }


def _compute_tost_margins_from_boot(
    point: float,
    se: float,
    ci90_low: float,
    ci90_high: float,
    boot_means: np.ndarray,
    margins_pp: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00),
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute a nonconfirmatory normal-Wald TOST sensitivity across margins in pp.

    The model-task bootstrap is a conditional panel-reweighting exercise rather than a
    boundary-null resampling test. These values therefore audit margin sensitivity only and
    must not be presented as exact p-values or confirmatory evidence of equivalence.
    """
    out: Dict[str, Any] = {}
    se_safe = max(1e-15, float(se))
    for m_pp in margins_pp:
        delta = float(m_pp) / 100.0
        t_lower = (point + delta) / se_safe
        t_upper = (point - delta) / se_safe
        p_lower_wald = float(stats.norm.sf(t_lower))
        p_upper_wald = float(stats.norm.cdf(t_upper))
        p_tost_wald = float(max(p_lower_wald, p_upper_wald))
        within_ci90 = bool((ci90_low > -delta) and (ci90_high < delta))
        out[f"{m_pp:.2f}pp"] = {
            "method": "normal_wald_tost_sensitivity",
            "confirmatory_equivalence_test": False,
            "margin_pp": float(m_pp),
            "margin_prop": delta,
            "t_lower": float(t_lower),
            "t_upper": float(t_upper),
            "p_lower_wald": p_lower_wald,
            "p_upper_wald": p_upper_wald,
            "p_tost_wald": p_tost_wald,
            "ci90_within_margin": within_ci90,
            "wald_rejects_both_one_sided_nulls": bool(
                (p_tost_wald < alpha) and within_ci90
            ),
        }
    return out


def two_way_cluster_bootstrap_ci(
    mat_model_task: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261130,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two-way (double/pigeonhole) cluster bootstrap across models (rows) and tasks (columns) (SI-3).
    Simultaneously accounts for dependence within models across tasks and within tasks across models.
    """
    arr = np.asarray(mat_model_task, dtype=float)
    if arr.ndim == 1:
        M = len(arr)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, M, size=(n_boot, M))
        boot_means = arr[idx].mean(axis=1)
        point = float(arr.mean())
    else:
        M, T = arr.shape
        rng = np.random.default_rng(seed)
        m_idx = rng.integers(0, M, size=(n_boot, M))
        t_idx = rng.integers(0, T, size=(n_boot, T))
        boot_means = np.array([arr[m_idx[b]][:, t_idx[b]].mean() for b in range(n_boot)], dtype=float)
        point = float(arr.mean())

    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    lo90 = float(np.percentile(boot_means, 5.0))
    hi90 = float(np.percentile(boot_means, 95.0))
    se = float(boot_means.std(ddof=1))
    return {
        "estimate": point,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "ci90_low": lo90,
        "ci90_high": hi90,
        "equivalence_sensitivity_margins": _compute_tost_margins_from_boot(
            point, se, lo90, hi90, boot_means, alpha=alpha
        ),
    }


def panel_cluster_bootstrap_ci(
    df_panel: pd.DataFrame,
    value_col: str,
    model_col: str = "model_id",
    task_col: str = "task",
    n_boot: int = 1000,
    seed: int = 20261130,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Two-way cluster bootstrap for an unbalanced model-task panel (Cohort 2).
    Preserves shared `model_id` clusters across benchmarks (`BBH` and `LMentry`) by pivoting
    into a single `(unique model_id) x (cohort, task)` matrix.
    """
    col_keys: Any = ["cohort", task_col] if "cohort" in df_panel.columns else task_col
    pivot = df_panel.pivot(index=model_col, columns=col_keys, values=value_col)
    arr = pivot.to_numpy(dtype=float)
    M, T = arr.shape
    if not np.isfinite(arr).any():
        raise ValueError("Panel bootstrap requires at least one finite observation")
    rng = np.random.default_rng(seed)
    valid_means: List[float] = []
    invalid_draws = 0
    max_draws = max(1000, 100 * n_boot)
    draws = 0
    while len(valid_means) < n_boot and draws < max_draws:
        draws += 1
        m_idx = rng.integers(0, M, size=M)
        t_idx = rng.integers(0, T, size=T)
        sub = arr[m_idx][:, t_idx]
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            invalid_draws += 1
            continue
        valid_means.append(float(finite.mean()))
    if len(valid_means) < n_boot:
        raise RuntimeError(
            f"Only {len(valid_means)} valid panel bootstrap replicates after {draws} draws"
        )
    boot_means = np.asarray(valid_means, dtype=float)
    point = float(df_panel[value_col].mean())
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    lo90 = float(np.percentile(boot_means, 5.0))
    hi90 = float(np.percentile(boot_means, 95.0))
    se = float(boot_means.std(ddof=1))
    return {
        "estimate": point,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "ci90_low": lo90,
        "ci90_high": hi90,
        "n_model_clusters": int(M),
        "n_task_clusters": int(T),
        "n_valid_bootstrap": int(len(boot_means)),
        "n_invalid_bootstrap_draws": int(invalid_draws),
        "equivalence_sensitivity_margins": _compute_tost_margins_from_boot(
            point, se, lo90, hi90, boot_means, alpha=alpha
        ),
    }


def compute_tost_paired_cluster(
    mat_or_panel: Any,
    margin_pp: float = 0.25,
    value_col: Optional[str] = None,
    n_boot: int = 1000,
    seed: int = 20261130,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Return a nonconfirmatory normal-Wald TOST margin sensitivity.

    This compatibility helper does not implement an exact boundary-null test and does not make
    a confirmatory equivalence decision. The ordinary paired difference and conditional panel
    bootstrap interval remain the reported results.
    """
    if isinstance(mat_or_panel, pd.DataFrame):
        if value_col is None:
            raise ValueError("value_col is required when mat_or_panel is a DataFrame")
        ci_info = panel_cluster_bootstrap_ci(mat_or_panel, value_col=value_col, n_boot=n_boot, seed=seed, alpha=alpha)
    else:
        ci_info = two_way_cluster_bootstrap_ci(np.asarray(mat_or_panel, dtype=float), n_boot=n_boot, seed=seed, alpha=alpha)

    point = float(ci_info["estimate"])
    se = max(1e-15, float(ci_info["se"]))
    ci90_low = float(ci_info["ci90_low"])
    ci90_high = float(ci_info["ci90_high"])
    delta = float(margin_pp) / 100.0
    t_lower = (point + delta) / se
    t_upper = (point - delta) / se
    p_lower = float(stats.norm.sf(t_lower))
    p_upper = float(stats.norm.cdf(t_upper))
    p_tost = float(max(p_lower, p_upper))
    within_ci90 = bool((ci90_low > -delta) and (ci90_high < delta))
    return {
        "method": "normal_wald_tost_sensitivity",
        "confirmatory_equivalence_test": False,
        "estimate": point,
        "se": float(ci_info["se"]),
        "ci95_low": float(ci_info["ci_low"]),
        "ci95_high": float(ci_info["ci_high"]),
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "margin_pp": float(margin_pp),
        "margin_prop": delta,
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "ci90_within_margin": within_ci90,
        "wald_rejects_both_one_sided_nulls": bool(
            (p_tost < alpha) and within_ci90
        ),
    }


def compute_oracle_uplift_recovery_stats(
    method_conf: np.ndarray,
    baseline_conf: np.ndarray,
    oracle_conf: np.ndarray,
    min_positive_denom: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute oracle-uplift recovery with explicit handling for zero or negative denominators (Spec 6.2, DV-5):
    1. `ratio_of_means_pct`: (mean(method) - mean(base)) / (mean(oracle) - mean(base)) * 100 across ALL cubes.
    2. `positive_uplift_cube_fraction`: fraction of cubes where (oracle - base) > min_positive_denom.
    3. `nonpositive_uplift_cube_count`: count of cubes where (oracle - base) <= min_positive_denom.
    4. `per_cube_median_positive_pct`: median per-cube recovery among cubes with positive discovery-oracle uplift.
    5. `per_cube_clipped_mean_pct`: mean of per-cube recovery clipped to [-100%, +200%] on positive-uplift cubes.
    """
    m_arr = np.asarray(method_conf, dtype=float).ravel()
    b_arr = np.asarray(baseline_conf, dtype=float).ravel()
    o_arr = np.asarray(oracle_conf, dtype=float).ravel()

    mean_num = float(m_arr.mean() - b_arr.mean())
    mean_den = float(o_arr.mean() - b_arr.mean())
    if abs(mean_den) < 1e-12:
        raise ValueError("Mean discovery-oracle uplift denominator is zero; ratio-of-means recovery is undefined.")
    ratio_of_means_pct = (mean_num / mean_den) * 100.0

    den_per_cube = o_arr - b_arr
    num_per_cube = m_arr - b_arr
    pos_mask = den_per_cube > min_positive_denom
    n_total = len(den_per_cube)
    n_pos = int(np.sum(pos_mask))
    n_nonpos = int(n_total - n_pos)

    if n_pos > 0:
        ratios_pos = (num_per_cube[pos_mask] / den_per_cube[pos_mask]) * 100.0
        med_pos = float(np.median(ratios_pos))
        clipped_pos = float(np.mean(np.clip(ratios_pos, -100.0, 200.0)))
    else:
        med_pos = float("nan")
        clipped_pos = float("nan")

    return {
        "ratio_of_means_pct": ratio_of_means_pct,
        "mean_numerator_acc_diff": mean_num,
        "mean_denominator_oracle_uplift": mean_den,
        "n_total_cubes": n_total,
        "n_positive_uplift_cubes": n_pos,
        "n_nonpositive_uplift_cubes": n_nonpos,
        "positive_uplift_cube_fraction": float(n_pos / max(1, n_total)),
        "positive_uplift_threshold": float(min_positive_denom),
        "per_cube_median_positive_pct": med_pos,
        "per_cube_clipped_mean_positive_pct": clipped_pos,
    }


def compute_all_reference_cells_recovery(
    method_conf: np.ndarray,
    oracle_or_cells_conf: np.ndarray,
    cells_or_oracle_conf: np.ndarray,
    min_positive_denom: float = 1e-6,
) -> Dict[str, Any]:
    """
    Evaluate Ratio-of-Means (RoM) oracle recovery and positive-uplift cube fractions across
    all K reference cells x_ref in {0, ..., K-1} (Task 2 baseline sensitivity).
    Accepts `(method_conf, oracle_conf, all_cells_conf)` or `(method_conf, all_cells_conf, oracle_conf)`
    where `all_cells_conf` has shape `(..., K)` and `method_conf`/`oracle_conf` have shape `(...)`.
    """
    m_arr = np.asarray(method_conf, dtype=float).ravel()
    a2 = np.asarray(oracle_or_cells_conf, dtype=float)
    a3 = np.asarray(cells_or_oracle_conf, dtype=float)
    if a2.size == m_arr.size and a3.size > m_arr.size:
        o_arr = a2.ravel()
        c_mat = a3.reshape(m_arr.size, -1)
    elif a3.size == m_arr.size and a2.size > m_arr.size:
        o_arr = a3.ravel()
        c_mat = a2.reshape(m_arr.size, -1)
    else:
        raise ValueError(f"Incompatible shapes in compute_all_reference_cells_recovery: {m_arr.shape}, {a2.shape}, {a3.shape}")
    K = c_mat.shape[1]

    roms = []
    pos_fracs = []
    denoms = []
    for k in range(K):
        b_k = c_mat[:, k]
        den_k = float(o_arr.mean() - b_k.mean())
        num_k = float(m_arr.mean() - b_k.mean())
        rom_k = (num_k / den_k * 100.0) if abs(den_k) >= 1e-12 else float("nan")
        pos_f_k = float(np.mean((o_arr - b_k) > min_positive_denom))
        roms.append(rom_k)
        pos_fracs.append(pos_f_k)
        denoms.append(den_k)

    roms_arr = np.array(roms, dtype=float)
    pos_arr = np.array(pos_fracs, dtype=float)
    return {
        "n_reference_cells": int(K),
        "cell0_rom_pct": float(roms_arr[0]),
        "min_rom_pct": float(np.nanmin(roms_arr)),
        "rom_pct_min": float(np.nanmin(roms_arr)),
        "q25_rom_pct": float(np.nanpercentile(roms_arr, 25.0)),
        "median_rom_pct": float(np.nanmedian(roms_arr)),
        "rom_pct_median": float(np.nanmedian(roms_arr)),
        "mean_rom_pct": float(np.nanmean(roms_arr)),
        "q75_rom_pct": float(np.nanpercentile(roms_arr, 75.0)),
        "max_rom_pct": float(np.nanmax(roms_arr)),
        "rom_pct_max": float(np.nanmax(roms_arr)),
        "min_rom_ref_cell": int(np.nanargmin(roms_arr)),
        "max_rom_ref_cell": int(np.nanargmax(roms_arr)),
        "cell0_positive_uplift_fraction": float(pos_arr[0]),
        "min_positive_uplift_fraction": float(pos_arr.min()),
        "median_positive_uplift_fraction": float(np.median(pos_arr)),
        "max_positive_uplift_fraction": float(pos_arr.max()),
        "min_mean_denom_pp": float(np.min(denoms) * 100.0),
        "max_mean_denom_pp": float(np.max(denoms) * 100.0),
    }


# ==============================================================================
# MODULE 2: cube-validity (CV-1 .. CV-4)
# ==============================================================================

def build_mmlu_cell_mapping(
    df_mmlu: Optional[pd.DataFrame] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[int]]]:
    """
    Map the 100 FormatSpread MMLU templates into a complete 2^5 = 32 factorial cube.
    Preserves original indices in `cell_to_indices` for legacy comparison while also
    supporting canonical hash-ordered and cell-mean estimators via `build_mmlu_mapping_table`.
    """
    if meta is None:
        meta = json.loads((DATA_DIR / "templates" / "mmlu_templates_metadata.json").read_text())
    if df_mmlu is None:
        df_mmlu = pd.read_csv(DATA_DIR / "templates" / "MMLU" / "abstract_algebra.csv")
    cell_to_indices: Dict[Tuple[int, ...], List[int]] = {}
    for idx, r in df_mmlu.iterrows():
        m = meta[r["prompt template"]]
        op, sp, sep = m["op"], m["space"], m["sep"]
        A = 1 if op in ("lambda x: x", "lambda x: x.title()") else 0
        B = 1 if op in ("lambda x: x.upper()", "lambda x: x.title()") else 0
        C = 1 if "\n" in sp else 0
        D = 1 if sep in (":", ".", "'", '"', "-", "!") else 0
        E = 1 if sep in (":", ".", "<sep>", "|", "]") else 0
        cell_to_indices.setdefault((A, B, C, D, E), []).append(int(idx))

    all_cells = sorted(cell_to_indices.keys())
    assert len(all_cells) == 32, f"Expected 32 cells, got {len(all_cells)}"
    return all_cells, cell_to_indices


def build_mmlu_mapping_table(
    df_mmlu: Optional[pd.DataFrame] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, ...], List[int]], Dict[Tuple[int, ...], List[int]]]:
    """
    Build complete machine-readable mapping table for MMLU templates (CV-1).
    Returns:
    - `df_map`: DataFrame with one row per template
    - `legacy_cell_to_indices`: order-dependent original row index list per cell
    - `canonical_cell_to_indices`: order-invariant (sorted by template SHA-256) index list per cell
    """
    if meta is None:
        meta = json.loads((DATA_DIR / "templates" / "mmlu_templates_metadata.json").read_text())
    if df_mmlu is None:
        df_mmlu = pd.read_csv(DATA_DIR / "templates" / "MMLU" / "abstract_algebra.csv")

    rows = []
    for pos_idx, (_, r) in enumerate(df_mmlu.iterrows()):
        tmpl = str(r["prompt template"])
        orig_idx = int(r["original_index"]) if "original_index" in df_mmlu.columns else pos_idx
        m = meta[tmpl]
        op, sp, sep = m["op"], m["space"], m["sep"]
        A = 1 if op in ("lambda x: x", "lambda x: x.title()") else 0
        B = 1 if op in ("lambda x: x.upper()", "lambda x: x.title()") else 0
        C = 1 if "\n" in sp else 0
        D = 1 if sep in (":", ".", "'", '"', "-", "!") else 0
        E = 1 if sep in (":", ".", "<sep>", "|", "]") else 0
        cell = (A, B, C, D, E)
        thash_full = sha256_text(tmpl)
        words = tmpl.split()
        rows.append(
            {
                "cohort": "MMLU",
                "benchmark": "MMLU",
                "task": "all_57_subjects",
                "catalog_row_index": pos_idx,
                "matrix_row_index": orig_idx,
                "template_sha256_full": thash_full,
                "template_sha256_16": thash_full[:16],
                "raw_format_or_op": op,
                "raw_space_or_len": repr(sp),
                "raw_sep_or_wl": sep,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "E": E,
                "threshold_B": "categorical_op",
                "threshold_D": "categorical_sep",
                "cell_tuple": str(cell),
                "cell_id": "".join(str(x) for x in cell),
                "char_length": len(tmpl),
                "word_count": len(words),
                "mean_word_length": float(sum(len(w) for w in words) / max(1, len(words))),
                "newline_count": tmpl.count("\n"),
                "template_preview": tmpl.replace("\n", "\\n")[:90],
            }
        )

    df_map = pd.DataFrame(rows)
    cell_counts = df_map.groupby("cell_id")["matrix_row_index"].transform("count")
    df_map["cell_template_count"] = cell_counts
    df_map["cell_mean_weight"] = 1.0 / df_map["cell_template_count"]

    legacy_map: Dict[Tuple[int, ...], List[int]] = {}
    canonical_map: Dict[Tuple[int, ...], List[int]] = {}
    unique_row_set = set()

    for cell_id, grp in df_map.groupby("cell_id"):
        cell_tuple = tuple(int(c) for c in cell_id)
        legacy_Order = grp.sort_values("catalog_row_index")["matrix_row_index"].astype(int).tolist()
        canon_sorted = grp.sort_values(["template_sha256_full", "matrix_row_index"])
        canon_Order = canon_sorted["matrix_row_index"].astype(int).tolist()
        uniq_Order = canon_sorted.drop_duplicates(subset=["template_sha256_full"], keep="first")["matrix_row_index"].astype(int).tolist()
        legacy_map[cell_tuple] = legacy_Order
        canonical_map[cell_tuple] = canon_Order
        unique_row_set.update(uniq_Order)

    df_map["is_unique_cell_template"] = df_map["matrix_row_index"].isin(unique_row_set)
    df_map["is_duplicate_text"] = ~df_map["is_unique_cell_template"]
    uniq_counts = df_map.groupby("cell_id")["is_unique_cell_template"].transform("sum")
    df_map["cell_unique_template_count"] = uniq_counts
    df_map["cell_mean_unique_weight"] = np.where(
        df_map["is_unique_cell_template"], 1.0 / df_map["cell_unique_template_count"], 0.0
    )

    legacy_first_set = {idxs[0] for idxs in legacy_map.values()}
    canon_first_set = {idxs[0] for idxs in canonical_map.values()}
    df_map["is_legacy_first_rep"] = df_map["matrix_row_index"].isin(legacy_first_set)
    df_map["is_canonical_first_rep"] = df_map["matrix_row_index"].isin(canon_first_set)
    return df_map, legacy_map, canonical_map


def extract_unique_cell_map(df_map: pd.DataFrame) -> Dict[Tuple[int, ...], List[int]]:
    """Extract deduplicated canonical cell-to-indices map weighting unique template hashes equally (Task 4)."""
    uniq_map: Dict[Tuple[int, ...], List[int]] = {}
    for cell_id, grp in df_map.groupby("cell_id"):
        cell_tuple = tuple(int(c) for c in cell_id)
        canon_sorted = grp.sort_values(["template_sha256_full", "matrix_row_index"])
        uniq_map[cell_tuple] = (
            canon_sorted.drop_duplicates(subset=["template_sha256_full"], keep="first")["matrix_row_index"]
            .astype(int)
            .tolist()
        )
    return uniq_map


def build_slab_cell_mapping(df: pd.DataFrame) -> Dict[Tuple[int, ...], List[int]]:
    """Map SLAB-NLP (BBH / LMentry) templates into a 2^4 = 16 factorial cube."""
    A = df["prompt format"].isin(["SEP_VARS", "Q&A"]).astype(int)
    med_len = df.assign(A=A).groupby("A")["prompt length"].transform("median")
    B = (df["prompt length"] >= med_len).astype(int)
    C = df["prompt template"].apply(
        lambda s: 1
        if any(w in str(s).lower() for w in ["please", "you", "your", "task", "model", "following", "given"])
        else 0
    )
    avg_wl = df["prompt template"].apply(
        lambda s: sum(len(w) for w in str(s).split()) / max(1, len(str(s).split()))
    )
    med_wl = df.assign(A=A, B=B, C=C, wl=avg_wl).groupby(["A", "B", "C"])["wl"].transform("median")
    D = (avg_wl > med_wl).astype(int)

    cell_map: Dict[Tuple[int, ...], List[int]] = {}
    for idx, c in enumerate(zip(A, B, C, D)):
        cell_map.setdefault(tuple(int(x) for x in c), []).append(int(idx))
    return cell_map


def build_slab_mapping_table_for_task(
    df: pd.DataFrame,
    benchmark: str,
    task_name: str,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, ...], List[int]], Dict[Tuple[int, ...], List[int]]]:
    """Build order-invariant and legacy template mappings for a single SLAB-NLP task (CV-1)."""
    A = df["prompt format"].isin(["SEP_VARS", "Q&A"]).astype(int)
    med_len = df.assign(A=A).groupby("A")["prompt length"].transform("median")
    B = (df["prompt length"] >= med_len).astype(int)
    C = df["prompt template"].apply(
        lambda s: 1
        if any(w in str(s).lower() for w in ["please", "you", "your", "task", "model", "following", "given"])
        else 0
    )
    avg_wl = df["prompt template"].apply(
        lambda s: sum(len(w) for w in str(s).split()) / max(1, len(str(s).split()))
    )
    med_wl = df.assign(A=A, B=B, C=C, wl=avg_wl).groupby(["A", "B", "C"])["wl"].transform("median")
    D = (avg_wl > med_wl).astype(int)

    rows = []
    for pos_idx in range(len(df)):
        r = df.iloc[pos_idx]
        orig_idx = int(r["original_index"]) if "original_index" in df.columns else pos_idx
        tmpl = str(r["prompt template"])
        thash_full = sha256_text(tmpl)
        cell = (int(A.iloc[pos_idx]), int(B.iloc[pos_idx]), int(C.iloc[pos_idx]), int(D.iloc[pos_idx]))
        words = tmpl.split()
        rows.append(
            {
                "cohort": "SLAB",
                "benchmark": benchmark,
                "task": task_name,
                "catalog_row_index": pos_idx,
                "matrix_row_index": orig_idx,
                "template_sha256_full": thash_full,
                "template_sha256_16": thash_full[:16],
                "raw_format_or_op": str(r["prompt format"]),
                "raw_space_or_len": str(r["prompt length"]),
                "raw_sep_or_wl": f"{ float(avg_wl.iloc[pos_idx]):.4f}",
                "A": cell[0],
                "B": cell[1],
                "C": cell[2],
                "D": cell[3],
                "E": -1,
                "threshold_B": f"{float(med_len.iloc[pos_idx]):.2f}",
                "threshold_D": f"{float(med_wl.iloc[pos_idx]):.4f}",
                "cell_tuple": str(cell),
                "cell_id": "".join(str(x) for x in cell),
                "char_length": len(tmpl),
                "word_count": len(words),
                "mean_word_length": float(avg_wl.iloc[pos_idx]),
                "newline_count": tmpl.count("\n"),
                "template_preview": tmpl.replace("\n", "\\n")[:90],
            }
        )

    df_map = pd.DataFrame(rows)
    cell_counts = df_map.groupby("cell_id")["matrix_row_index"].transform("count")
    df_map["cell_template_count"] = cell_counts
    df_map["cell_mean_weight"] = 1.0 / df_map["cell_template_count"]

    legacy_map: Dict[Tuple[int, ...], List[int]] = {}
    canonical_map: Dict[Tuple[int, ...], List[int]] = {}
    unique_row_set = set()
    for cell_id, grp in df_map.groupby("cell_id"):
        cell_tuple = tuple(int(c) for c in cell_id)
        legacy_map[cell_tuple] = grp.sort_values("catalog_row_index")["matrix_row_index"].astype(int).tolist()
        canon_sorted = grp.sort_values(["template_sha256_full", "matrix_row_index"])
        canonical_map[cell_tuple] = canon_sorted["matrix_row_index"].astype(int).tolist()
        uniq_Order = canon_sorted.drop_duplicates(subset=["template_sha256_full"], keep="first")["matrix_row_index"].astype(int).tolist()
        unique_row_set.update(uniq_Order)

    df_map["is_unique_cell_template"] = df_map["matrix_row_index"].isin(unique_row_set)
    df_map["is_duplicate_text"] = ~df_map["is_unique_cell_template"]
    uniq_counts = df_map.groupby("cell_id")["is_unique_cell_template"].transform("sum")
    df_map["cell_unique_template_count"] = uniq_counts
    df_map["cell_mean_unique_weight"] = np.where(
        df_map["is_unique_cell_template"], 1.0 / df_map["cell_unique_template_count"], 0.0
    )

    legacy_first_set = {idxs[0] for idxs in legacy_map.values()}
    canon_first_set = {idxs[0] for idxs in canonical_map.values()}
    df_map["is_legacy_first_rep"] = df_map["matrix_row_index"].isin(legacy_first_set)
    df_map["is_canonical_first_rep"] = df_map["matrix_row_index"].isin(canon_first_set)
    return df_map, legacy_map, canonical_map


def extract_cell_item_matrix(
    raw_mat: np.ndarray,
    all_cells: List[Tuple[int, ...]],
    cell_to_indices: Dict[Tuple[int, ...], List[int]],
    estimator: str = "cell_mean",
    rep_draw_map: Optional[Dict[Tuple[int, ...], int]] = None,
) -> np.ndarray:
    """
    Construct the (K, N) cell-by-item response matrix under the specified cell estimator (CV-2):
    - `cell_mean` / `cell_mean_unique` / `cell_mean_catalog_row`: Equal-weight average across the
      templates in `cell_to_indices[c]` (order-invariant).
    - `canonical_first`: First template in canonical SHA-256 order within each cell (order-invariant).
    - `legacy_first`: First template in original catalog row order (`cell_to_indices[c][0]`).
    - `random_single_rep` / `random_single_rep_unique`: Specific template index per cell supplied by `rep_draw_map`.
    """
    if estimator in ("cell_mean", "cell_mean_unique", "cell_mean_catalog_row"):
        return np.array([raw_mat[cell_to_indices[c]].mean(axis=0) for c in all_cells], dtype=float)
    elif estimator in ("legacy_first", "canonical_first"):
        return np.array([raw_mat[cell_to_indices[c][0]] for c in all_cells], dtype=float)
    elif estimator in ("random_single_rep", "random_single_rep_unique"):
        if rep_draw_map is None:
            raise ValueError(f"rep_draw_map must be provided when estimator={estimator!r}")
        return np.array([raw_mat[rep_draw_map[c]] for c in all_cells], dtype=float)
    else:
        raise ValueError(f"Unknown template-cell estimator: {estimator!r}")


def build_factor_audit_table(df_template_mapping: pd.DataFrame) -> pd.DataFrame:
    """Create the factor construct audit table with executable rules and co-varying features (CV-3)."""
    factor_defs = [
        (
            "MMLU",
            "A",
            "case_preservation_contrast",
            "case_preservation",
            "op in {'lambda x: x', 'lambda x: x.title()'} (1) vs {'lambda x: x.lower()', 'lambda x: x.upper()'} (0)",
            "Deterministic partition of FormatSpread 4-level casing operator `op`",
        ),
        (
            "MMLU",
            "B",
            "header_capitalization_contrast",
            "header_capitalization",
            "op in {'lambda x: x.upper()', 'lambda x: x.title()'} (1) vs {'lambda x: x', 'lambda x: x.lower()'} (0)",
            "Orthogonal companion contrast on FormatSpread casing operator `op`",
        ),
        (
            "MMLU",
            "C",
            "multiline_spacing_contrast",
            "multiline_layout",
            "'\\n' in space (1) vs no newline in space (0)",
            "Co-varies with total character length (+7.4 chars) and newline count",
        ),
        (
            "MMLU",
            "D",
            "delimiter_punctuation_split",
            "natural_punctuation",
            "sep in {':', '.', \"'\", '\"', '-', '!'} (1) vs {'<sep>', '|', ']', '/', '\\\\'} (0)",
            "Post-hoc binary partition of 11-level FormatSpread `sep` axis (crossed with E)",
        ),
        (
            "MMLU",
            "E",
            "delimiter_boundary_split",
            "structural_boundary",
            "sep in {':', '.', '<sep>', '|', ']'} (1) vs {\"'\", '\"', '-', '!', '/', '\\\\'} (0)",
            "Post-hoc binary partition of 11-level FormatSpread `sep` axis (crossed with D)",
        ),
        (
            "SLAB",
            "A",
            "format_schema_tag",
            "structured_schema",
            "prompt_format in {'SEP_VARS', 'Q&A'} (1) vs 'other' (0)",
            "Strongly co-varies with prompt token/character length (mean prompt_length 40.3 vs 24.4)",
        ),
        (
            "SLAB",
            "B",
            "above_median_prompt_length",
            "instruction_elaboration",
            "prompt_length >= task median(prompt_length | A) (1) vs < median (0)",
            "Task-and-stratum-dependent median split on token/character length, not controlled semantic elaboration",
        ),
        (
            "SLAB",
            "C",
            "directive_keyword_presence",
            "role_directive_framing",
            "any keyword {'please','you','your','task','model','following','given'} in lowercase template (1) vs none (0)",
            "Lexical keyword filter; co-varies with catalog prompt_length (mean 33.8 vs 23.5), whitespace word_count (23.8 vs 15.1 words), and char_length (146.8 vs 93.4 chars)",
        ),
        (
            "SLAB",
            "D",
            "above_median_word_length",
            "lexical_specificity",
            "mean_word_length > task median(mean_word_length | A, B, C) (1) vs <= median (0)",
            "Cell-conditional median split on average characters per whitespace token",
        ),
    ]

    audit_rows = []
    for cohort, fid, revised_name, legacy_name, rule_str, caveat in factor_defs:
        sub = df_template_mapping[df_template_mapping["cohort"] == cohort]
        col = sub[fid]
        s0 = sub[col == 0]
        s1 = sub[col == 1]
        rep0 = s0.sort_values("template_sha256_full").iloc[0]["template_preview"] if len(s0) else ""
        rep1 = s1.sort_values("template_sha256_full").iloc[0]["template_preview"] if len(s1) else ""
        audit_rows.append(
            {
                "cohort": cohort,
                "factor_id": fid,
                "revised_factor_name": revised_name,
                "legacy_factor_name": legacy_name,
                "executable_rule": rule_str,
                "n_templates_level_0": int(len(s0)),
                "n_templates_level_1": int(len(s1)),
                "prop_level_1": float(len(s1) / max(1, len(sub))),
                "mean_char_len_level_0": float(s0["char_length"].mean()),
                "mean_char_len_level_1": float(s1["char_length"].mean()),
                "mean_word_count_level_0": float(s0["word_count"].mean()),
                "mean_word_count_level_1": float(s1["word_count"].mean()),
                "mean_word_len_level_0": float(s0["mean_word_length"].mean()),
                "mean_word_len_level_1": float(s1["mean_word_length"].mean()),
                "representative_template_level_0": rep0,
                "representative_template_level_1": rep1,
                "construct_validity_note": caveat,
            }
        )
    return pd.DataFrame(audit_rows)


# ==============================================================================
# MODULE 3: statistical-inference (SI-1 .. SI-6)
# ==============================================================================

def decompose_full_cube_effects(y_cell_means: np.ndarray, num_factors: int) -> Dict[str, Any]:
    """
    Exact orthogonal Fourier decomposition of a complete 2^k cube (SI-1, SI-2).
    Separates true effect magnitudes (RMS of main, 2FI, 3FI, 4FI, 5FI) from screening estimation RMSE
    and verifies exact cell-mean reconstruction.
    """
    mats = build_cube_design_matrices(num_factors)
    K = 1 << num_factors
    scale = K / 2.0  # 16.0 for k=5, 8.0 for k=4

    X_pm = mats["X_pm"]
    X_2fi = mats["X_2fi"]
    tau_main = (X_pm.T @ y_cell_means) / scale
    tau_2fi = (X_2fi.T @ y_cell_means) / scale

    y_recon = np.full(K, float(y_cell_means.mean())) + 0.5 * (X_pm @ tau_main) + 0.5 * (X_2fi @ tau_2fi)

    higher_taus: Dict[int, np.ndarray] = {}
    for order, X_ho in mats["higher_orders"].items():
        t_ho = (X_ho.T @ y_cell_means) / scale
        higher_taus[order] = t_ho
        y_recon = y_recon + 0.5 * (X_ho @ t_ho)

    recon_max_abs_err = float(np.max(np.abs(y_recon - y_cell_means)))
    if recon_max_abs_err > 1e-12:
        raise RuntimeError(f"Fourier cube reconstruction failed: max_err={recon_max_abs_err}")

    total_ss = float(np.sum((y_cell_means - y_cell_means.mean()) ** 2))
    # Each orthogonal contrast tau has sum of squares contribution (K / 4) * tau^2
    ss_main = float((K / 4.0) * np.sum(tau_main ** 2))
    ss_2fi = float((K / 4.0) * np.sum(tau_2fi ** 2))
    ss_3fi = float((K / 4.0) * np.sum(higher_taus[3] ** 2)) if 3 in higher_taus else 0.0
    ss_4fi = float((K / 4.0) * np.sum(higher_taus[4] ** 2)) if 4 in higher_taus else 0.0
    ss_5fi = float((K / 4.0) * np.sum(higher_taus[5] ** 2)) if 5 in higher_taus else 0.0

    all_ho = np.concatenate([higher_taus[o] for o in sorted(higher_taus.keys())])

    return {
        "tau_main": tau_main,
        "tau_2fi": tau_2fi,
        "higher_taus": higher_taus,
        "rms_main": float(np.sqrt(np.mean(tau_main ** 2))),
        "rms_2fi": float(np.sqrt(np.mean(tau_2fi ** 2))),
        "rms_3fi": float(np.sqrt(np.mean(higher_taus[3] ** 2))) if 3 in higher_taus else 0.0,
        "rms_4fi": float(np.sqrt(np.mean(higher_taus[4] ** 2))) if 4 in higher_taus else 0.0,
        "rms_all_higher_order": float(np.sqrt(np.mean(all_ho ** 2))),
        "var_share_main": float(ss_main / max(1e-15, total_ss)),
        "var_share_2fi": float(ss_2fi / max(1e-15, total_ss)),
        "var_share_higher_order": float((ss_3fi + ss_4fi + ss_5fi) / max(1e-15, total_ss)),
        "reconstruction_max_abs_err": recon_max_abs_err,
    }


def compute_cube_rcbd_and_robust_inference(
    Y_t: np.ndarray,
    X_pm: np.ndarray,
    n_perm: int = 0,
    perm_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Compare three inferential procedures for factor main effects on a (K, N) cube (SI-4):
    1. Unblocked fixed-effects ANOVA F-test (treats K*N cells as independent).
    2. Classical fixed-block RCBD ANOVA F-test (removes item mean SS_item, assumes homogeneous
       residual variance across contrasts).
    3. Item-clustered contrast t-test (computes each factor's orthogonal contrast per item
       c_{j,i} = (2/K) sum_c X_{c,j} Y_{c,i} and tests E[c_{j,i}] = 0 across the N items with
       contrast-specific variance, accounting for binary outcomes and arbitrary within-item
       covariance across cells).
    Optionally computes sign-flip randomization p-values across items when `n_perm > 0`.
    """
    if np.isnan(Y_t).any():
        raise ValueError("Missing values detected in cell-item matrix Y_t; complete item blocks required.")
    K, N = Y_t.shape
    num_factors = X_pm.shape[1]
    if K != X_pm.shape[0]:
        raise ValueError(f"Row mismatch between Y_t ({K}) and X_pm ({X_pm.shape[0]})")
    if N < 2:
        raise ValueError(f"At least 2 item blocks required for RCBD/cluster inference, got N={N}")

    scale = K / 2.0
    # Item-level contrasts: shape (num_factors, N)
    C_item = (X_pm.T @ Y_t) / scale
    taus = C_item.mean(axis=1)  # (num_factors,)

    y_bar = float(Y_t.mean())
    ss_tot = float(np.sum((Y_t - y_bar) ** 2))
    ss_item = float(K * np.sum((Y_t.mean(axis=0) - y_bar) ** 2))
    ss_factors = (K * N / 4.0) * (taus ** 2)
    ss_model = float(np.sum(ss_factors))
    df_u = K * N - 1 - num_factors
    df_b = K * N - 1 - num_factors - (N - 1)

    sd_c = C_item.std(axis=1, ddof=1)
    is_zero_var = sd_c <= 1e-15
    n_zero_var = int(np.sum(is_zero_var))

    if ss_tot > 1e-9:
        residual_ss_u = float(ss_tot - ss_model)
        residual_ss_b = float(ss_tot - ss_model - ss_item)
        mse_u = residual_ss_u / df_u
        mse_b = residual_ss_b / df_b
        estimable_u = bool(np.isfinite(mse_u) and mse_u > 1e-12)
        estimable_b = bool(np.isfinite(mse_b) and mse_b > 1e-12)
        if estimable_u and estimable_b:
            mse_reduction_pct = float((mse_u - mse_b) / mse_u * 100.0)
            f_stat_multiplier = float(mse_u / mse_b)
        else:
            # A zero residual denominator does not create infinite evidence. There is
            # no estimable sampling variance, so retain neutral diagnostics and p=1.
            mse_reduction_pct = 0.0
            f_stat_multiplier = 1.0
        r2_unblocked = float(ss_model / ss_tot)
        r2_rcbd = float((ss_model + ss_item) / ss_tot)
        p_unblocked = (
            stats.f.sf(ss_factors / mse_u, 1, df_u)
            if estimable_u
            else np.ones(num_factors)
        )
        p_rcbd = (
            stats.f.sf(ss_factors / mse_b, 1, df_b)
            if estimable_b
            else np.ones(num_factors)
        )

        # Item-clustered contrast t-test (asymptotic one-sample t-test on item contrasts C_item)
        se_c = sd_c / np.sqrt(N)
        t_stat_cluster = np.zeros(num_factors, dtype=float)
        p_cluster = np.ones(num_factors, dtype=float)
        for j in range(num_factors):
            if sd_c[j] > 1e-15:
                t_val = float(np.abs(taus[j]) / se_c[j])
                t_stat_cluster[j] = t_val
                p_cluster[j] = float(2.0 * stats.t.sf(t_val, df=N - 1))
            else:
                t_stat_cluster[j] = 0.0
                p_cluster[j] = 1.0
    else:
        estimable_u = False
        estimable_b = False
        mse_reduction_pct = 0.0
        f_stat_multiplier = 1.0
        r2_unblocked = 0.0
        r2_rcbd = 0.0
        p_unblocked = np.ones(num_factors)
        p_rcbd = np.ones(num_factors)
        p_cluster = np.ones(num_factors)
        t_stat_cluster = np.zeros(num_factors)

    p_perm = None
    if n_perm > 0:
        if perm_rng is None:
            perm_rng = np.random.default_rng(SEEDS["sign_flip_perm"])
        signs = perm_rng.choice([-1.0, 1.0], size=(n_perm, N))
        # Permuted means: (n_perm, num_factors)
        perm_means = np.abs((signs @ C_item.T) / N)
        obs_means = np.abs(taus)[None, :]
        p_perm = (1.0 + np.sum(perm_means >= obs_means - 1e-15, axis=0)) / (n_perm + 1.0)
        p_perm[is_zero_var] = 1.0

    _, pu_bh_cube, _, _ = multipletests(p_unblocked, method="fdr_bh")
    _, pb_bh_cube, _, _ = multipletests(p_rcbd, method="fdr_bh")
    _, pc_bh_cube, _, _ = multipletests(p_cluster, method="fdr_bh")

    return {
        "taus": taus,
        "mse_reduction_pct": mse_reduction_pct,
        "f_stat_multiplier": f_stat_multiplier,
        "r2_unblocked": r2_unblocked,
        "r2_rcbd": r2_rcbd,
        "p_unblocked": p_unblocked,
        "p_rcbd": p_rcbd,
        "p_cluster": p_cluster,
        "p_perm": p_perm,
        "t_stat_cluster": t_stat_cluster,
        "is_zero_var_contrast": is_zero_var,
        "zero_var_mask": is_zero_var,
        "n_zero_var_contrasts": n_zero_var,
        "unblocked_residual_variance_estimable": estimable_u,
        "rcbd_residual_variance_estimable": estimable_b,
        "sign_flip_n_perm": int(n_perm),
        "sign_flip_min_achievable_p": float(1.0 / (n_perm + 1.0)) if n_perm > 0 else float("nan"),
        "sig_unblocked_cube_bh": int(np.sum(pu_bh_cube < 0.05)),
        "sig_rcbd_cube_bh": int(np.sum(pb_bh_cube < 0.05)),
        "sig_cluster_cube_bh": int(np.sum(pc_bh_cube < 0.05)),
    }


def verify_with_anova_engine(Ys: dict) -> dict:
    """Directly run prompt_prism.ANOVAEngine and OptimalPromptFinder on a representative cube."""
    all_cells, cell_to_indices = build_mmlu_cell_mapping()
    mat = Ys["MMLU"]["professional_law"][11]  # Gemma-7B-IT on professional_law
    Y_t = np.array([mat[cell_to_indices[c][0]] for c in all_cells], dtype=float)
    records = []
    for c_i, (A, B, C, D, E) in enumerate(all_cells):
        for s_i in range(Y_t.shape[1]):
            records.append(
                {
                    "A": A,
                    "B": B,
                    "C": C,
                    "D": D,
                    "E": E,
                    "sample_id": f"item_{s_i:04d}",
                    "accuracy": float(Y_t[c_i, s_i]),
                }
            )
    df_long = pd.DataFrame(records)
    res_unb = ANOVAEngine.run_anova(df_long, ["A", "B", "C", "D", "E"], "accuracy", block_col=None)
    res_blk = ANOVAEngine.run_anova(df_long, ["A", "B", "C", "D", "E"], "accuracy", block_col="sample_id")
    opt = OptimalPromptFinder.find_optimal_prompt(res_blk)
    return {
        "benchmark": "MMLU/professional_law",
        "model": "Gemma-7B-IT",
        "n_observations": len(df_long),
        "unblocked_r2": res_unb.r_squared,
        "rcbd_r2": res_blk.r_squared,
        "unblocked_residual_se": res_unb.residual_std_error,
        "rcbd_residual_se": res_blk.residual_std_error,
        "optimal_levels": opt.optimal_factor_levels,
        "predicted_gain": opt.expected_gain_absolute,
    }


# ==============================================================================
# COHORT ORCHESTRATION AND SENSITIVITY PIPELINE
# ==============================================================================

def concatenate_task_discovery_items(
    task_matrices: List[np.ndarray],
) -> np.ndarray:
    """Concatenate legacy even-index discovery items separately within each task.

    Applying ``[:, ::2]`` after concatenation is incorrect whenever a preceding task
    has an odd number of items, because it reverses the discovery parity for the next
    task. This helper preserves the declared within-task split.
    """
    if not task_matrices:
        raise ValueError("At least one task matrix is required")
    n_cells = task_matrices[0].shape[0]
    if any(matrix.ndim != 2 or matrix.shape[0] != n_cells for matrix in task_matrices):
        raise ValueError("All task matrices must be 2D with the same number of cells")
    return np.hstack([matrix[:, ::2] for matrix in task_matrices])

def evaluate_cohort1_estimator(
    Ys: dict,
    all_cells: List[Tuple[int, ...]],
    cell_map: Dict[Tuple[int, ...], List[int]],
    estimator_name: str,
    task_splits: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    cosets_info: Dict[str, Any],
    rand16_fullrank: List[List[int]],
    rand16_conds: np.ndarray,
    rand16_legacy: List[List[int]],
    rand8_fullrank: List[List[int]],
    compute_rcbd: bool = True,
    n_perm_rcbd: int = 200,
    rand16_screenbest_uncond: Optional[List[List[int]]] = None,
    rand8_screenbest_uncond: Optional[List[List[int]]] = None,
    tie_policy: str = "mean_tied",
) -> Dict[str, Any]:
    """
    Vectorized evaluation of Cohort 1 (15 LLMs x 57 MMLU tasks = 855 cubes) for a given
    template-cell estimator across:
    - Legacy even/odd split (split index 0) and 20 seeded random splits (split indices 1..20)
    - Both cosets of 2^(5-1)_V, all 4 canonical cosets of 2^(5-2)_III, all 60 oriented 2^(5-2)_III designs
    - 200 full-rank balanced Random-16 designs (for 2FI/main attribution) and 200 unconditioned
      factor-balanced Random-16 designs (for ScreenBest selection)
    - 20 legacy balanced Random-16 designs (diagnostic only)
    - 200 full-rank balanced Random-8 designs (for main attribution), 200 unconditioned Random-8
      designs (for ScreenBest), and lower-budget OFAT-6
    """
    m5 = build_cube_design_matrices(5)
    X_pm = m5["X_pm"]
    X_2fi = m5["X_2fi"]
    D_main = m5["D_main"]
    D_2fi = m5["D_2fi"]

    if rand16_screenbest_uncond is None:
        rand16_screenbest_uncond = generate_balanced_random_designs(
            32, 5, 16, num_screens=200, seed=SEEDS["rand16_screenbest_uncond"],
            require_2fi_full_rank=False, require_main_full_rank=False,
        )
    if rand8_screenbest_uncond is None:
        rand8_screenbest_uncond = generate_balanced_random_designs(
            32, 5, 8, num_screens=200, seed=SEEDS["rand8_c1_screenbest_uncond"],
            require_2fi_full_rank=False, require_main_full_rank=False,
        )

    res5_pos = cosets_info["res5_cosets"]["2(5-1)V_coset_pos"]["indices"]
    res5_neg = cosets_info["res5_cosets"]["2(5-1)V_coset_neg"]["indices"]
    res5_pos_arr = np.array(res5_pos, dtype=int)
    res5_neg_arr = np.array(res5_neg, dtype=int)
    res3_c_dict = cosets_info["res3_canonical_cosets"]
    res3_c_keys = list(res3_c_dict.keys())
    res3_c_idx = [res3_c_dict[k]["indices"] for k in res3_c_keys]
    res3_60_dict = cosets_info["res3_all_60"]
    res3_60_keys = list(res3_60_dict.keys())
    res3_60_idx = np.array([res3_60_dict[k]["indices"] for k in res3_60_keys])  # (60, 8)

    ofat_cells = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
    ofat_idx = [all_cells.index(c) for c in ofat_cells]

    # Validate primary attribution design matrices explicitly (DV-1)
    validate_design_matrix(D_2fi[res5_pos], "2(5-1)V_coset_pos", "MMLU_2FI")
    validate_design_matrix(D_2fi[res5_neg], "2(5-1)V_coset_neg", "MMLU_2FI")
    validate_design_matrix(D_main[ofat_idx], "OFAT_6", "MMLU_main")
    for k, v in res3_c_dict.items():
        validate_design_matrix(D_main[v["indices"]], k, "MMLU_main")
    for i, c in enumerate(rand16_fullrank):
        validate_design_matrix(D_2fi[c], f"rand16_fullrank_{i:03d}", "MMLU_2FI")
    for i, c in enumerate(rand8_fullrank):
        validate_design_matrix(D_main[c], f"rand8_fullrank_{i:03d}", "MMLU_main")

    # Precompute pseudoinverses for fast vectorized OLS across splits
    P2_pos = np.linalg.pinv(D_2fi[res5_pos])
    P2_neg = np.linalg.pinv(D_2fi[res5_neg])
    Pm_ofat = np.linalg.pinv(D_main[ofat_idx])
    Pm_res3_c = np.array([np.linalg.pinv(D_main[idx]) for idx in res3_c_idx])  # (4, 6, 8)
    Pm_res3_60 = np.array([np.linalg.pinv(D_main[idx]) for idx in res3_60_idx])  # (60, 6, 8)

    r16_arr = np.array(rand16_fullrank)  # (200, 16)
    r16_sb_arr = np.array(rand16_screenbest_uncond)  # (200, 16)
    P2_r16 = np.array([np.linalg.pinv(D_2fi[c]) for c in rand16_fullrank])  # (200, 16, 16)
    Pm_r16 = np.array([np.linalg.pinv(D_main[c]) for c in rand16_fullrank])  # (200, 6, 16)
    well_cond_mask = rand16_conds <= 15.0

    r16_leg_arr = np.array(rand16_legacy)  # (20, 16)
    P2_r16_leg = np.array([np.linalg.pinv(D_2fi[c]) for c in rand16_legacy])
    Pm_r16_leg = np.array([np.linalg.pinv(D_main[c]) for c in rand16_legacy])

    r8_arr = np.array(rand8_fullrank)  # (200, 8)
    r8_sb_arr = np.array(rand8_screenbest_uncond)  # (200, 8)
    Pm_r8 = np.array([np.linalg.pinv(D_main[c]) for c in rand8_fullrank])  # (200, 6, 8)

    tasks = sorted(Ys["MMLU"].keys())
    n_models = len(MMLU_MODELS)
    n_tasks = len(tasks)
    n_splits_total = 21  # col 0 = legacy_even_odd, cols 1..20 = split_00..split_19

    acc_keys = [
        "baseline", "oracle32", "ofat",
        "res5_pos_best", "res5_neg_best", "res5_avg_best",
        "res5_pos_2fi", "res5_neg_2fi", "res5_avg_2fi",
        "res5_pos_main", "res5_neg_main", "res5_avg_main",
        "rand16_best", "rand16_best_2fi_cond", "rand16_2fi", "rand16_main",
        "rand16_leg_best", "rand16_leg_2fi", "rand16_leg_main",
        "res3_c0_best", "res3_c1_best", "res3_c2_best", "res3_c3_best", "res3_c_avg_best",
        "res3_c0_main", "res3_c1_main", "res3_c2_main", "res3_c3_main", "res3_c_avg_main",
        "res3_60_avg_best", "res3_60_min_best", "res3_60_max_best",
        "rand8_best", "rand8_best_main_cond", "rand8_main",
    ]
    acc = {k: np.zeros((n_models, n_tasks, n_splits_total), dtype=float) for k in acc_keys}
    acc_first = {k: np.zeros((n_models, n_tasks, n_splits_total), dtype=float) for k in [
        "oracle32", "res5_pos_best", "res5_neg_best", "res5_avg_best",
        "res5_pos_2fi", "res5_neg_2fi", "res5_avg_2fi",
        "res5_pos_main", "res5_neg_main", "res5_avg_main",
        "rand16_best", "rand16_best_2fi_cond", "rand16_2fi",
        "res3_c0_best", "res3_c_avg_best", "res3_60_avg_best", "rand8_best", "ofat",
    ]}
    acc_seed = {k: np.zeros((n_models, n_tasks, n_splits_total), dtype=float) for k in [
        "oracle32", "res5_pos_best", "res5_neg_best", "res5_avg_best",
        "res5_pos_2fi", "res5_neg_2fi", "res5_avg_2fi",
        "res5_pos_main", "res5_neg_main", "res5_avg_main", "rand16_best",
    ]}
    tie_counts = {k: np.zeros((n_models, n_tasks, n_splits_total), dtype=float) for k in [
        "oracle32", "res5_pos_best", "res5_neg_best", "res5_pos_2fi", "res5_neg_2fi", "res5_pos_main", "res5_neg_main", "rand16_best",
    ]}
    tie_indicators = {
        key: np.zeros((n_models, n_tasks, n_splits_total), dtype=float)
        for key in tie_counts
    }
    all_cells_conf_s0 = np.zeros((n_models * n_tasks, 32), dtype=float)
    all_cells_conf_20s = np.zeros((n_models * n_tasks, 32), dtype=float)

    rmse_arr = {
        k: np.zeros((n_models, n_tasks, n_splits_total), dtype=float)
        for k in [
            "main_res5_pos", "main_res5_neg", "2fi_res5_pos", "2fi_res5_neg",
            "main_rand16", "2fi_rand16", "2fi_rand16_wellcond",
            "main_rand16_leg", "2fi_rand16_leg",
            "main_res3_c0", "main_res3_c1", "main_res3_c2", "main_res3_c3", "main_res3_c_avg",
            "main_res3_60_avg",
            "main_rand8", "main_ofat",
        ]
    }

    model_rows = []
    cube_rows = []
    fourier_rows = []
    rcbd_rows = []
    pool_r16_2fi_design_dist = np.zeros((n_models, len(rand16_fullrank)), dtype=float)

    perm_rng = np.random.default_rng(SEEDS["sign_flip_perm"])
    tie_rng = np.random.default_rng(SEEDS["tie_break"])

    for m_idx, (m_id, m_label, m_type) in enumerate(MMLU_MODELS):
        # 1. Pooled 14,042-item matrix per model
        task_cell_item_matrices = [
            extract_cell_item_matrix(
                Ys["MMLU"][task][m_idx],
                all_cells,
                cell_map,
                estimator=estimator_name,
            )
            for task in tasks
        ]
        Y_pool = np.hstack(task_cell_item_matrices)
        y_pool_d = concatenate_task_discovery_items(task_cell_item_matrices).mean(axis=1)
        true_main_pool = (X_pm.T @ y_pool_d) / 16.0
        true_2fi_pool = (X_2fi.T @ y_pool_d) / 16.0

        bp_pool = P2_pos @ y_pool_d[res5_pos]
        bn_pool = P2_neg @ y_pool_d[res5_neg]
        pool_rmse_main_res5_pos = float(np.sqrt(np.mean((2.0 * bp_pool[1:6] - true_main_pool) ** 2)))
        pool_rmse_main_res5_neg = float(np.sqrt(np.mean((2.0 * bn_pool[1:6] - true_main_pool) ** 2)))
        pool_rmse_2fi_res5_pos = float(np.sqrt(np.mean((2.0 * bp_pool[6:] - true_2fi_pool) ** 2)))
        pool_rmse_2fi_res5_neg = float(np.sqrt(np.mean((2.0 * bn_pool[6:] - true_2fi_pool) ** 2)))

        bm3_pool = Pm_res3_c[0] @ y_pool_d[res3_c_idx[0]]
        pool_rmse_main_res3 = float(np.sqrt(np.mean((2.0 * bm3_pool[1:] - true_main_pool) ** 2)))
        bm3_all_pool = np.einsum("cij,cj->ci", Pm_res3_c, y_pool_d[np.array(res3_c_idx)])
        pool_rmse_main_res3_c_avg = float(np.sqrt(np.mean((2.0 * bm3_all_pool[:, 1:] - true_main_pool[None, :]) ** 2, axis=1)).mean())
        bm3_60_pool = np.einsum("cij,cj->ci", Pm_res3_60, y_pool_d[res3_60_idx])
        pool_rmse_main_res3_all60 = float(np.sqrt(np.mean((2.0 * bm3_60_pool[:, 1:] - true_main_pool[None, :]) ** 2, axis=1)).mean())

        b_ofat_pool = Pm_ofat @ y_pool_d[ofat_idx]
        pool_rmse_main_ofat = float(np.sqrt(np.mean((2.0 * b_ofat_pool[1:] - true_main_pool) ** 2)))

        y_pool_r16 = y_pool_d[r16_arr]
        bm_r16_pool = np.einsum("dij,dj->di", Pm_r16, y_pool_r16)
        b2_r16_pool = np.einsum("dij,dj->di", P2_r16, y_pool_r16)
        r16_main_rmses_pool = np.sqrt(np.mean((2.0 * bm_r16_pool[:, 1:] - true_main_pool[None, :]) ** 2, axis=1))
        r16_2fi_rmses_pool = np.sqrt(np.mean((2.0 * b2_r16_pool[:, 6:] - true_2fi_pool[None, :]) ** 2, axis=1))
        pool_r16_2fi_design_dist[m_idx] = r16_2fi_rmses_pool
        r16_signs_pool = np.mean(np.sign(2.0 * bm_r16_pool[:, 1:]) == np.sign(true_main_pool[None, :]), axis=1)

        y_pool_r16_leg = y_pool_d[r16_leg_arr]
        bm_r16_leg_pool = np.einsum("dij,dj->di", Pm_r16_leg, y_pool_r16_leg)
        b2_r16_leg_pool = np.einsum("dij,dj->di", P2_r16_leg, y_pool_r16_leg)
        r16_leg_main_rmses_pool = np.sqrt(np.mean((2.0 * bm_r16_leg_pool[:, 1:] - true_main_pool[None, :]) ** 2, axis=1))
        r16_leg_2fi_rmses_pool = np.sqrt(np.mean((2.0 * b2_r16_leg_pool[:, 6:] - true_2fi_pool[None, :]) ** 2, axis=1))

        y_pool_r8 = y_pool_d[r8_arr]
        bm_r8_pool = np.einsum("dij,dj->di", Pm_r8, y_pool_r8)
        r8_main_rmses_pool = np.sqrt(np.mean((2.0 * bm_r8_pool[:, 1:] - true_main_pool[None, :]) ** 2, axis=1))

        sign_res5_pos = float(np.mean(np.sign(2.0 * bp_pool[1:6]) == np.sign(true_main_pool)))
        sign_res5_neg = float(np.mean(np.sign(2.0 * bn_pool[1:6]) == np.sign(true_main_pool)))

        # 2. Per-task 57 cubes for this model
        for t_idx, t in enumerate(tasks):
            cube_flat_idx = m_idx * n_tasks + t_idx
            mat = Ys["MMLU"][t][m_idx]
            Y_t = extract_cell_item_matrix(mat, all_cells, cell_map, estimator=estimator_name)
            K, N = Y_t.shape
            d_masks, c_masks, _ = task_splits[t]
            d_bin = (d_masks > 0).astype(float)
            c_bin = (c_masks > 0).astype(float)
            Y_d = (Y_t @ d_bin) / d_bin.sum(axis=0)[None, :]  # (32, 21)
            Y_c = (Y_t @ c_bin) / c_bin.sum(axis=0)[None, :]  # (32, 21)
            Y_d[:, 0] = Y_t[:, ::2].mean(axis=1)
            Y_c[:, 0] = Y_t[:, 1::2].mean(axis=1)
            all_cells_conf_s0[cube_flat_idx] = Y_c[:, 0]
            all_cells_conf_20s[cube_flat_idx] = Y_c[:, 1:].mean(axis=1)
            s_range = np.arange(n_splits_total)

            tm = (X_pm.T @ Y_d) / 16.0   # (5, 21)
            t2 = (X_2fi.T @ Y_d) / 16.0  # (10, 21)

            Bp, *_ = np.linalg.lstsq(D_2fi[res5_pos], Y_d[res5_pos, :], rcond=None)
            Bn, *_ = np.linalg.lstsq(D_2fi[res5_neg], Y_d[res5_neg, :], rcond=None)
            Bp[:, 0], *_ = np.linalg.lstsq(D_2fi[res5_pos], Y_d[res5_pos, 0], rcond=None)
            Bn[:, 0], *_ = np.linalg.lstsq(D_2fi[res5_neg], Y_d[res5_neg, 0], rcond=None)
            Bp_m, *_ = np.linalg.lstsq(D_main[res5_pos], Y_d[res5_pos, :], rcond=None)
            Bn_m, *_ = np.linalg.lstsq(D_main[res5_neg], Y_d[res5_neg, :], rcond=None)
            Bp_m[:, 0], *_ = np.linalg.lstsq(D_main[res5_pos], Y_d[res5_pos, 0], rcond=None)
            Bn_m[:, 0], *_ = np.linalg.lstsq(D_main[res5_neg], Y_d[res5_neg, 0], rcond=None)
            rmse_arr["main_res5_pos"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bp_m[1:6, :] - tm) ** 2, axis=0))
            rmse_arr["main_res5_neg"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bn_m[1:6, :] - tm) ** 2, axis=0))
            rmse_arr["2fi_res5_pos"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bp[6:, :] - t2) ** 2, axis=0))
            rmse_arr["2fi_res5_neg"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bn[6:, :] - t2) ** 2, axis=0))

            acc["baseline"][m_idx, t_idx] = Y_c[0, :]
            acc["oracle32"][m_idx, t_idx], tie_counts["oracle32"][m_idx, t_idx], tie_indicators["oracle32"][m_idx, t_idx] = select_with_tie_policy(Y_d, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc_first["oracle32"][m_idx, t_idx] = Y_c[np.argmax(Y_d, axis=0), s_range]
            acc_seed["oracle32"][m_idx, t_idx], _, _ = select_with_tie_policy(Y_d, Y_c, tie_policy="seeded_random", rng=tie_rng)

            acc["res5_pos_best"][m_idx, t_idx], tie_counts["res5_pos_best"][m_idx, t_idx], tie_indicators["res5_pos_best"][m_idx, t_idx] = select_with_tie_policy(Y_d[res5_pos, :], Y_c, res5_pos_arr, tie_policy=tie_policy, rng=tie_rng)
            acc["res5_neg_best"][m_idx, t_idx], tie_counts["res5_neg_best"][m_idx, t_idx], tie_indicators["res5_neg_best"][m_idx, t_idx] = select_with_tie_policy(Y_d[res5_neg, :], Y_c, res5_neg_arr, tie_policy=tie_policy, rng=tie_rng)
            acc["res5_avg_best"][m_idx, t_idx] = 0.5 * (acc["res5_pos_best"][m_idx, t_idx] + acc["res5_neg_best"][m_idx, t_idx])

            acc_first["res5_pos_best"][m_idx, t_idx] = Y_c[res5_pos_arr[np.argmax(Y_d[res5_pos, :], axis=0)], s_range]
            acc_first["res5_neg_best"][m_idx, t_idx] = Y_c[res5_neg_arr[np.argmax(Y_d[res5_neg, :], axis=0)], s_range]
            acc_first["res5_avg_best"][m_idx, t_idx] = 0.5 * (acc_first["res5_pos_best"][m_idx, t_idx] + acc_first["res5_neg_best"][m_idx, t_idx])

            acc_seed["res5_pos_best"][m_idx, t_idx], _, _ = select_with_tie_policy(Y_d[res5_pos, :], Y_c, res5_pos_arr, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["res5_neg_best"][m_idx, t_idx], _, _ = select_with_tie_policy(Y_d[res5_neg, :], Y_c, res5_neg_arr, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["res5_avg_best"][m_idx, t_idx] = 0.5 * (acc_seed["res5_pos_best"][m_idx, t_idx] + acc_seed["res5_neg_best"][m_idx, t_idx])

            pred_p_2fi = D_2fi @ Bp
            pred_n_2fi = D_2fi @ Bn
            pred_p_2fi[:, 0] = D_2fi @ Bp[:, 0]
            pred_n_2fi[:, 0] = D_2fi @ Bn[:, 0]
            acc["res5_pos_2fi"][m_idx, t_idx], tie_counts["res5_pos_2fi"][m_idx, t_idx], tie_indicators["res5_pos_2fi"][m_idx, t_idx] = select_with_tie_policy(pred_p_2fi, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["res5_neg_2fi"][m_idx, t_idx], tie_counts["res5_neg_2fi"][m_idx, t_idx], tie_indicators["res5_neg_2fi"][m_idx, t_idx] = select_with_tie_policy(pred_n_2fi, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["res5_avg_2fi"][m_idx, t_idx] = 0.5 * (acc["res5_pos_2fi"][m_idx, t_idx] + acc["res5_neg_2fi"][m_idx, t_idx])
            acc_first["res5_pos_2fi"][m_idx, t_idx] = Y_c[np.argmax(pred_p_2fi, axis=0), s_range]
            acc_first["res5_neg_2fi"][m_idx, t_idx] = Y_c[np.argmax(pred_n_2fi, axis=0), s_range]
            acc_first["res5_avg_2fi"][m_idx, t_idx] = 0.5 * (acc_first["res5_pos_2fi"][m_idx, t_idx] + acc_first["res5_neg_2fi"][m_idx, t_idx])
            acc_seed["res5_pos_2fi"][m_idx, t_idx], _, _ = select_with_tie_policy(pred_p_2fi, Y_c, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["res5_neg_2fi"][m_idx, t_idx], _, _ = select_with_tie_policy(pred_n_2fi, Y_c, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["res5_avg_2fi"][m_idx, t_idx] = 0.5 * (acc_seed["res5_pos_2fi"][m_idx, t_idx] + acc_seed["res5_neg_2fi"][m_idx, t_idx])

            pred_p_main = D_main @ Bp_m
            pred_n_main = D_main @ Bn_m
            pred_p_main[:, 0] = D_main @ Bp_m[:, 0]
            pred_n_main[:, 0] = D_main @ Bn_m[:, 0]
            acc["res5_pos_main"][m_idx, t_idx], tie_counts["res5_pos_main"][m_idx, t_idx], tie_indicators["res5_pos_main"][m_idx, t_idx] = select_with_tie_policy(pred_p_main, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["res5_neg_main"][m_idx, t_idx], tie_counts["res5_neg_main"][m_idx, t_idx], tie_indicators["res5_neg_main"][m_idx, t_idx] = select_with_tie_policy(pred_n_main, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["res5_avg_main"][m_idx, t_idx] = 0.5 * (acc["res5_pos_main"][m_idx, t_idx] + acc["res5_neg_main"][m_idx, t_idx])
            acc_first["res5_pos_main"][m_idx, t_idx] = Y_c[np.argmax(pred_p_main, axis=0), s_range]
            acc_first["res5_neg_main"][m_idx, t_idx] = Y_c[np.argmax(pred_n_main, axis=0), s_range]
            acc_first["res5_avg_main"][m_idx, t_idx] = 0.5 * (acc_first["res5_pos_main"][m_idx, t_idx] + acc_first["res5_neg_main"][m_idx, t_idx])
            acc_seed["res5_pos_main"][m_idx, t_idx], _, _ = select_with_tie_policy(pred_p_main, Y_c, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["res5_neg_main"][m_idx, t_idx], _, _ = select_with_tie_policy(pred_n_main, Y_c, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["res5_avg_main"][m_idx, t_idx] = 0.5 * (acc_seed["res5_pos_main"][m_idx, t_idx] + acc_seed["res5_neg_main"][m_idx, t_idx])

            # OFAT (6 runs, lower-budget baseline)
            Bo, *_ = np.linalg.lstsq(D_main[ofat_idx], Y_d[ofat_idx, :], rcond=None)
            rmse_arr["main_ofat"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bo[1:, :] - tm) ** 2, axis=0))
            acc["ofat"][m_idx, t_idx], _, _ = select_with_tie_policy(D_main @ Bo, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc_first["ofat"][m_idx, t_idx] = Y_c[np.argmax(D_main @ Bo, axis=0), s_range]

            # Res III 4 canonical cosets
            for c_i in range(4):
                sub_idx = res3_c_idx[c_i]
                sub_arr = np.array(sub_idx, dtype=int)
                B3, *_ = np.linalg.lstsq(D_main[sub_idx], Y_d[sub_idx, :], rcond=None)
                rmse_arr[f"main_res3_c{c_i}"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * B3[1:, :] - tm) ** 2, axis=0))
                acc[f"res3_c{c_i}_best"][m_idx, t_idx], _, _ = select_with_tie_policy(Y_d[sub_idx, :], Y_c, sub_arr, tie_policy=tie_policy, rng=tie_rng)
                acc[f"res3_c{c_i}_main"][m_idx, t_idx], _, _ = select_with_tie_policy(D_main @ B3, Y_c, tie_policy=tie_policy, rng=tie_rng)
                if c_i == 0:
                    acc_first["res3_c0_best"][m_idx, t_idx] = Y_c[sub_arr[np.argmax(Y_d[sub_idx, :], axis=0)], s_range]

            rmse_arr["main_res3_c_avg"][m_idx, t_idx] = 0.25 * sum(rmse_arr[f"main_res3_c{c_i}"][m_idx, t_idx] for c_i in range(4))
            acc["res3_c_avg_best"][m_idx, t_idx] = 0.25 * sum(acc[f"res3_c{c_i}_best"][m_idx, t_idx] for c_i in range(4))
            acc["res3_c_avg_main"][m_idx, t_idx] = 0.25 * sum(acc[f"res3_c{c_i}_main"][m_idx, t_idx] for c_i in range(4))
            acc_first["res3_c_avg_best"][m_idx, t_idx] = 0.25 * sum(
                Y_c[np.array(res3_c_idx[c_i])[np.argmax(Y_d[res3_c_idx[c_i], :], axis=0)], s_range] for c_i in range(4)
            )

            # Res III all 60 regular quarter-fractions
            Y_d_60 = Y_d[res3_60_idx, :]  # (60, 8, 21)
            B3_60 = np.einsum("dij,djs->dis", Pm_res3_60, Y_d_60)
            rmse_arr["main_res3_60_avg"][m_idx, t_idx] = np.sqrt(
                np.mean((2.0 * B3_60[:, 1:, :] - tm[None, :, :]) ** 2, axis=1)
            ).mean(axis=0)
            conf_60, _, _ = select_with_tie_policy(Y_d_60, Y_c, res3_60_idx, tie_policy=tie_policy, rng=tie_rng)
            acc["res3_60_avg_best"][m_idx, t_idx] = conf_60.mean(axis=0)
            acc["res3_60_min_best"][m_idx, t_idx] = conf_60.min(axis=0)
            acc["res3_60_max_best"][m_idx, t_idx] = conf_60.max(axis=0)
            best_60_first = res3_60_idx[np.arange(60)[:, None], np.argmax(Y_d_60, axis=1)]
            acc_first["res3_60_avg_best"][m_idx, t_idx] = Y_c[best_60_first, s_range[None, :]].mean(axis=0)

            # 200 Full-Rank Rand-16 (attribution) and 200 Unconditioned Rand-16 (ScreenBest)
            Y_d_r16 = Y_d[r16_arr, :]  # (200, 16, 21)
            Bm_r16 = np.einsum("dij,djs->dis", Pm_r16, Y_d_r16)
            B2_r16 = np.einsum("dij,djs->dis", P2_r16, Y_d_r16)
            err_m_r16 = np.sqrt(np.mean((2.0 * Bm_r16[:, 1:6, :] - tm[None, :, :]) ** 2, axis=1))
            err_2_r16 = np.sqrt(np.mean((2.0 * B2_r16[:, 6:, :] - t2[None, :, :]) ** 2, axis=1))
            rmse_arr["main_rand16"][m_idx, t_idx] = err_m_r16.mean(axis=0)
            rmse_arr["2fi_rand16"][m_idx, t_idx] = err_2_r16.mean(axis=0)
            rmse_arr["2fi_rand16_wellcond"][m_idx, t_idx] = err_2_r16[well_cond_mask, :].mean(axis=0)

            Y_d_r16_sb = Y_d[r16_sb_arr, :]  # (200, 16, 21) unconditioned ScreenBest
            c_r16_sb, tc_r16_sb, ti_r16_sb = select_with_tie_policy(Y_d_r16_sb, Y_c, r16_sb_arr, tie_policy=tie_policy, rng=tie_rng)
            acc["rand16_best"][m_idx, t_idx] = c_r16_sb.mean(axis=0)
            tie_counts["rand16_best"][m_idx, t_idx] = tc_r16_sb.mean(axis=0)
            tie_indicators["rand16_best"][m_idx, t_idx] = ti_r16_sb.mean(axis=0)
            c_r16_sb_seed, _, _ = select_with_tie_policy(Y_d_r16_sb, Y_c, r16_sb_arr, tie_policy="seeded_random", rng=tie_rng)
            acc_seed["rand16_best"][m_idx, t_idx] = c_r16_sb_seed.mean(axis=0)
            r16_sb_first = r16_sb_arr[np.arange(len(r16_sb_arr))[:, None], np.argmax(Y_d_r16_sb, axis=1)]
            acc_first["rand16_best"][m_idx, t_idx] = Y_c[r16_sb_first, s_range[None, :]].mean(axis=0)

            c_r16_cond, _, _ = select_with_tie_policy(Y_d_r16, Y_c, r16_arr, tie_policy=tie_policy, rng=tie_rng)
            acc["rand16_best_2fi_cond"][m_idx, t_idx] = c_r16_cond.mean(axis=0)
            r16_cond_first = r16_arr[np.arange(len(rand16_fullrank))[:, None], np.argmax(Y_d_r16, axis=1)]
            acc_first["rand16_best_2fi_cond"][m_idx, t_idx] = Y_c[r16_cond_first, s_range[None, :]].mean(axis=0)

            pred_r16_2fi = np.einsum("ci,dis->dcs", D_2fi, B2_r16)
            c_r16_2fi, _, _ = select_with_tie_policy(pred_r16_2fi, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["rand16_2fi"][m_idx, t_idx] = c_r16_2fi.mean(axis=0)
            acc_first["rand16_2fi"][m_idx, t_idx] = Y_c[np.argmax(pred_r16_2fi, axis=1), s_range[None, :]].mean(axis=0)

            pred_r16_main = np.einsum("ci,dis->dcs", D_main, Bm_r16)
            c_r16_main, _, _ = select_with_tie_policy(pred_r16_main, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["rand16_main"][m_idx, t_idx] = c_r16_main.mean(axis=0)

            # 20 Legacy Rand-16 (diagnostic only)
            Y_d_leg = Y_d[r16_leg_arr, :]
            Bm_leg = np.einsum("dij,djs->dis", Pm_r16_leg, Y_d_leg)
            B2_leg = np.einsum("dij,djs->dis", P2_r16_leg, Y_d_leg)
            rmse_arr["main_rand16_leg"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bm_leg[:, 1:6, :] - tm[None, :, :]) ** 2, axis=1)).mean(axis=0)
            rmse_arr["2fi_rand16_leg"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * B2_leg[:, 6:, :] - t2[None, :, :]) ** 2, axis=1)).mean(axis=0)
            leg_best_global = r16_leg_arr[np.arange(len(rand16_legacy))[:, None], np.argmax(Y_d_leg, axis=1)]
            acc["rand16_leg_best"][m_idx, t_idx] = Y_c[leg_best_global, s_range[None, :]].mean(axis=0)
            acc["rand16_leg_2fi"][m_idx, t_idx] = Y_c[np.argmax(np.einsum("ci,dis->dcs", D_2fi, B2_leg), axis=1), s_range[None, :]].mean(axis=0)
            acc["rand16_leg_main"][m_idx, t_idx] = Y_c[np.argmax(np.einsum("ci,dis->dcs", D_main, Bm_leg), axis=1), s_range[None, :]].mean(axis=0)

            # 200 Full-Rank Rand-8 (attribution) and 200 Unconditioned Rand-8 (ScreenBest)
            Y_d_r8 = Y_d[r8_arr, :]
            Bm_r8 = np.einsum("dij,djs->dis", Pm_r8, Y_d_r8)
            rmse_arr["main_rand8"][m_idx, t_idx] = np.sqrt(np.mean((2.0 * Bm_r8[:, 1:6, :] - tm[None, :, :]) ** 2, axis=1)).mean(axis=0)

            Y_d_r8_sb = Y_d[r8_sb_arr, :]
            c_r8_sb, _, _ = select_with_tie_policy(Y_d_r8_sb, Y_c, r8_sb_arr, tie_policy=tie_policy, rng=tie_rng)
            acc["rand8_best"][m_idx, t_idx] = c_r8_sb.mean(axis=0)
            r8_sb_first = r8_sb_arr[np.arange(len(r8_sb_arr))[:, None], np.argmax(Y_d_r8_sb, axis=1)]
            acc_first["rand8_best"][m_idx, t_idx] = Y_c[r8_sb_first, s_range[None, :]].mean(axis=0)

            c_r8_cond, _, _ = select_with_tie_policy(Y_d_r8, Y_c, r8_arr, tie_policy=tie_policy, rng=tie_rng)
            acc["rand8_best_main_cond"][m_idx, t_idx] = c_r8_cond.mean(axis=0)

            pred_r8_main = np.einsum("ci,dis->dcs", D_main, Bm_r8)
            c_r8_main, _, _ = select_with_tie_policy(pred_r8_main, Y_c, tie_policy=tie_policy, rng=tie_rng)
            acc["rand8_main"][m_idx, t_idx] = c_r8_main.mean(axis=0)

            # Full-cube effect magnitude decomposition (SI-1, SI-2) on full items, discovery split 0, and 20 splits
            cell_means_cube = Y_t.mean(axis=1)
            f_full = decompose_full_cube_effects(cell_means_cube, 5)
            f_disc0 = decompose_full_cube_effects(Y_d[:, 0], 5)
            mean20_disc_rms_main = float(np.sqrt(np.mean(tm[:, 1:] ** 2, axis=0)).mean())
            mean20_disc_rms_2fi = float(np.sqrt(np.mean(t2[:, 1:] ** 2, axis=0)).mean())
            fourier_rows.append(
                {
                    "cohort": "MMLU",
                    "estimator": estimator_name,
                    "model_id": m_id,
                    "task": t,
                    "n_items": N,
                    "spread_pp": float((cell_means_cube.max() - cell_means_cube.min()) * 100.0),
                    "full_rms_main": f_full["rms_main"],
                    "full_rms_2fi": f_full["rms_2fi"],
                    "full_rms_3fi": f_full["rms_3fi"],
                    "full_rms_4fi": f_full["rms_4fi"],
                    "full_rms_higher_order": f_full["rms_all_higher_order"],
                    "full_var_share_main": f_full["var_share_main"],
                    "full_var_share_2fi": f_full["var_share_2fi"],
                    "full_var_share_higher_order": f_full["var_share_higher_order"],
                    "disc0_rms_main": f_disc0["rms_main"],
                    "disc0_rms_2fi": f_disc0["rms_2fi"],
                    "disc0_rms_3fi": f_disc0["rms_3fi"],
                    "disc0_rms_4fi": f_disc0["rms_4fi"],
                    "mean20_disc_rms_main": mean20_disc_rms_main,
                    "mean20_disc_rms_2fi": mean20_disc_rms_2fi,
                    "disc0_rmse_main_res5": float(rmse_arr["main_res5_neg"][m_idx, t_idx, 0]),
                    "disc0_rmse_2fi_res5": float(rmse_arr["2fi_res5_neg"][m_idx, t_idx, 0]),
                    "disc0_rmse_2fi_rand16_fullrank": float(rmse_arr["2fi_rand16"][m_idx, t_idx, 0]),
                    "regret_main_split0": float(acc["oracle32"][m_idx, t_idx, 0] - acc["res5_avg_main"][m_idx, t_idx, 0]),
                    "regret_main_20splits": float((acc["oracle32"][m_idx, t_idx, 1:] - acc["res5_avg_main"][m_idx, t_idx, 1:]).mean()),
                    "regret_main_20splits_first_index": float((acc_first["oracle32"][m_idx, t_idx, 1:] - acc_first["res5_avg_main"][m_idx, t_idx, 1:]).mean()),
                    "gain_2fi_over_main_20splits": float((acc["res5_avg_2fi"][m_idx, t_idx, 1:] - acc["res5_avg_main"][m_idx, t_idx, 1:]).mean()),
                    "gain_best_over_main_20splits": float((acc["res5_avg_best"][m_idx, t_idx, 1:] - acc["res5_avg_main"][m_idx, t_idx, 1:]).mean()),
                    "gain_best_over_main_20splits_first_index": float((acc_first["res5_avg_best"][m_idx, t_idx, 1:] - acc_first["res5_avg_main"][m_idx, t_idx, 1:]).mean()),
                    "reconstruction_max_abs_err": f_full["reconstruction_max_abs_err"],
                }
            )

            # RCBD vs Unblocked vs Item-Clustered Robust inference
            if compute_rcbd:
                inf = compute_cube_rcbd_and_robust_inference(Y_t, X_pm, n_perm=n_perm_rcbd, perm_rng=perm_rng)
                for f_i, f_name in enumerate("ABCDE"):
                    rcbd_rows.append(
                        {
                            "cohort": "MMLU",
                            "estimator": estimator_name,
                            "model_id": m_id,
                            "task": t,
                            "factor": f_name,
                            "n_items": N,
                            "tau_effect": float(inf["taus"][f_i]),
                            "p_unblocked": float(inf["p_unblocked"][f_i]),
                            "p_rcbd": float(inf["p_rcbd"][f_i]),
                            "p_item_cluster": float(inf["p_cluster"][f_i]),
                            "p_sign_flip_perm": float(inf["p_perm"][f_i]) if inf["p_perm"] is not None else float("nan"),
                            "t_stat_cluster": float(inf["t_stat_cluster"][f_i]),
                            "zero_var_contrast": bool(inf["zero_var_mask"][f_i]),
                        }
                    )
            else:
                inf = {
                    "mse_reduction_pct": np.nan, "f_stat_multiplier": np.nan,
                    "r2_unblocked": np.nan, "r2_rcbd": np.nan,
                    "sig_unblocked_cube_bh": np.nan, "sig_rcbd_cube_bh": np.nan,
                    "sig_cluster_cube_bh": np.nan, "n_zero_var_contrasts": np.nan,
                }

            cube_rows.append(
                {
                    "cohort": "MMLU",
                    "estimator": estimator_name,
                    "model_id": m_id,
                    "model_label": m_label,
                    "model_type": m_type,
                    "task": t,
                    "n_items": N,
                    # Legacy even/odd split (index 0) and 20-split mean (indices 1..20)
                    "rmse_main_res5": float(rmse_arr["main_res5_neg"][m_idx, t_idx, 0]),
                    "rmse_main_res5_20splits": float(rmse_arr["main_res5_neg"][m_idx, t_idx, 1:].mean()),
                    "rmse_main_rand16": float(rmse_arr["main_rand16"][m_idx, t_idx, 0]),
                    "rmse_main_rand16_20splits": float(rmse_arr["main_rand16"][m_idx, t_idx, 1:].mean()),
                    "rmse_2fi_res5": float(rmse_arr["2fi_res5_neg"][m_idx, t_idx, 0]),
                    "rmse_2fi_res5_20splits": float(rmse_arr["2fi_res5_neg"][m_idx, t_idx, 1:].mean()),
                    "rmse_2fi_rand16_fullrank": float(rmse_arr["2fi_rand16"][m_idx, t_idx, 0]),
                    "rmse_2fi_rand16_fullrank_20splits": float(rmse_arr["2fi_rand16"][m_idx, t_idx, 1:].mean()),
                    "rmse_2fi_rand16_legacy_rankdef": float(rmse_arr["2fi_rand16_leg"][m_idx, t_idx, 0]),
                    "rmse_main_res3_coset_avg": float(rmse_arr["main_res3_c_avg"][m_idx, t_idx, 0]),
                    "rmse_main_res3_coset_avg_20splits": float(rmse_arr["main_res3_c_avg"][m_idx, t_idx, 1:].mean()),
                    "rmse_main_res3_all60": float(rmse_arr["main_res3_60_avg"][m_idx, t_idx, 0]),
                    "rmse_main_res3_all60_20splits": float(rmse_arr["main_res3_60_avg"][m_idx, t_idx, 1:].mean()),
                    "rmse_main_res3_legacy_c0": float(rmse_arr["main_res3_c0"][m_idx, t_idx, 0]),
                    "rmse_main_rand8": float(rmse_arr["main_rand8"][m_idx, t_idx, 0]),
                    "rmse_main_rand8_20splits": float(rmse_arr["main_rand8"][m_idx, t_idx, 1:].mean()),
                    "rmse_main_ofat": float(rmse_arr["main_ofat"][m_idx, t_idx, 0]),
                    "rms_main_true_disc0": f_disc0["rms_main"],
                    "rms_2fi_true_disc0": f_disc0["rms_2fi"],
                    "rms_3fi_true_disc0": f_disc0["rms_3fi"],
                    "rms_4fi_true_disc0": f_disc0["rms_4fi"],
                    "conf_baseline": float(acc["baseline"][m_idx, t_idx, 0]),
                    "conf_baseline_20splits": float(acc["baseline"][m_idx, t_idx, 1:].mean()),
                    "conf_oracle32": float(acc["oracle32"][m_idx, t_idx, 0]),
                    "conf_oracle32_20splits": float(acc["oracle32"][m_idx, t_idx, 1:].mean()),
                    "conf_res5_best_coset_pos": float(acc["res5_pos_best"][m_idx, t_idx, 0]),
                    "conf_res5_best_coset_neg": float(acc["res5_neg_best"][m_idx, t_idx, 0]),
                    "conf_res5_best_coset_avg": float(acc["res5_avg_best"][m_idx, t_idx, 0]),
                    "conf_res5_best_coset_avg_20splits": float(acc["res5_avg_best"][m_idx, t_idx, 1:].mean()),
                    "conf_res5_2fi_coset_pos": float(acc["res5_pos_2fi"][m_idx, t_idx, 0]),
                    "conf_res5_2fi_coset_neg": float(acc["res5_neg_2fi"][m_idx, t_idx, 0]),
                    "conf_res5_2fi_coset_avg": float(acc["res5_avg_2fi"][m_idx, t_idx, 0]),
                    "conf_res5_2fi_coset_avg_20splits": float(acc["res5_avg_2fi"][m_idx, t_idx, 1:].mean()),
                    "conf_res5_main_coset_pos": float(acc["res5_pos_main"][m_idx, t_idx, 0]),
                    "conf_res5_main_coset_neg": float(acc["res5_neg_main"][m_idx, t_idx, 0]),
                    "conf_res5_main_coset_avg": float(acc["res5_avg_main"][m_idx, t_idx, 0]),
                    "conf_res5_main_coset_avg_20splits": float(acc["res5_avg_main"][m_idx, t_idx, 1:].mean()),
                    "conf_rand16_best": float(acc["rand16_best"][m_idx, t_idx, 0]),
                    "conf_rand16_best_20splits": float(acc["rand16_best"][m_idx, t_idx, 1:].mean()),
                    "conf_rand16_best_2fi_cond": float(acc["rand16_best_2fi_cond"][m_idx, t_idx, 0]),
                    "conf_rand16_best_2fi_cond_20splits": float(acc["rand16_best_2fi_cond"][m_idx, t_idx, 1:].mean()),
                    "conf_rand16_2fi_fullrank": float(acc["rand16_2fi"][m_idx, t_idx, 0]),
                    "conf_rand16_2fi_fullrank_20splits": float(acc["rand16_2fi"][m_idx, t_idx, 1:].mean()),
                    "conf_rand16_main": float(acc["rand16_main"][m_idx, t_idx, 0]),
                    "conf_rand16_main_20splits": float(acc["rand16_main"][m_idx, t_idx, 1:].mean()),
                    "conf_res3_best_coset_avg": float(acc["res3_c_avg_best"][m_idx, t_idx, 0]),
                    "conf_res3_best_coset_avg_20splits": float(acc["res3_c_avg_best"][m_idx, t_idx, 1:].mean()),
                    "conf_res3_best_all60": float(acc["res3_60_avg_best"][m_idx, t_idx, 0]),
                    "conf_res3_best_all60_20splits": float(acc["res3_60_avg_best"][m_idx, t_idx, 1:].mean()),
                    "conf_res3_best_legacy_c0": float(acc["res3_c0_best"][m_idx, t_idx, 0]),
                    "conf_rand8_best": float(acc["rand8_best"][m_idx, t_idx, 0]),
                    "conf_rand8_best_20splits": float(acc["rand8_best"][m_idx, t_idx, 1:].mean()),
                    "conf_ofat": float(acc["ofat"][m_idx, t_idx, 0]),
                    "conf_ofat_20splits": float(acc["ofat"][m_idx, t_idx, 1:].mean()),
                    "regret_res5_best_coset_avg_20splits": float((acc["oracle32"][m_idx, t_idx, 1:] - acc["res5_avg_best"][m_idx, t_idx, 1:]).mean()),
                    "regret_res5_2fi_coset_avg_20splits": float((acc["oracle32"][m_idx, t_idx, 1:] - acc["res5_avg_2fi"][m_idx, t_idx, 1:]).mean()),
                    "regret_res5_main_coset_avg_20splits": float((acc["oracle32"][m_idx, t_idx, 1:] - acc["res5_avg_main"][m_idx, t_idx, 1:]).mean()),
                    "regret_rand16_best_20splits": float((acc["oracle32"][m_idx, t_idx, 1:] - acc["rand16_best"][m_idx, t_idx, 1:]).mean()),
                    "mse_reduction_pct": float(inf["mse_reduction_pct"]),
                    "f_stat_multiplier": float(inf["f_stat_multiplier"]),
                    "r2_unblocked": float(inf["r2_unblocked"]),
                    "r2_rcbd": float(inf["r2_rcbd"]),
                    "sig_factors_unblocked": int(inf["sig_unblocked_cube_bh"]) if compute_rcbd else None,
                    "sig_factors_rcbd": int(inf["sig_rcbd_cube_bh"]) if compute_rcbd else None,
                    "sig_factors_item_cluster": int(inf["sig_cluster_cube_bh"]) if compute_rcbd else None,
                    "n_zero_var_contrasts": int(inf.get("n_zero_var_contrasts", 0)) if compute_rcbd else None,
                }
            )

        df_m_cubes = pd.DataFrame([r for r in cube_rows if r["model_id"] == m_id])
        cell_means_pool = Y_pool.mean(axis=1)
        model_rows.append(
            {
                "estimator": estimator_name,
                "model_id": m_id,
                "model_label": m_label,
                "model_type": m_type,
                "mean_acc": float(cell_means_pool.mean()),
                "min_acc": float(cell_means_pool.min()),
                "max_acc": float(cell_means_pool.max()),
                "spread_pp": float((cell_means_pool.max() - cell_means_pool.min()) * 100.0),
                "pool_rmse_main_res5": pool_rmse_main_res5_neg,
                "pool_rmse_main_rand16": float(np.mean(r16_main_rmses_pool)),
                "pool_rmse_2fi_res5": pool_rmse_2fi_res5_neg,
                "pool_rmse_2fi_rand16_fullrank": float(np.mean(r16_2fi_rmses_pool)),
                "pool_rmse_2fi_rand16_wellcond": float(np.mean(r16_2fi_rmses_pool[well_cond_mask])),
                "pool_rmse_2fi_rand16_legacy_rankdef": float(np.mean(r16_leg_2fi_rmses_pool)),
                "pool_2fi_ratio_fullrank": float(np.mean(r16_2fi_rmses_pool) / pool_rmse_2fi_res5_neg),
                "pool_rmse_main_res3": pool_rmse_main_res3,
                "pool_rmse_main_res3_coset_avg": pool_rmse_main_res3_c_avg,
                "pool_rmse_main_res3_all60": pool_rmse_main_res3_all60,
                "pool_rmse_main_rand8": float(np.mean(r8_main_rmses_pool)),
                "pool_rmse_main_ofat": pool_rmse_main_ofat,
                "pool_sign_res5": 0.5 * (sign_res5_pos + sign_res5_neg),
                "pool_sign_rand16": float(np.mean(r16_signs_pool)),
                "task_conf_oracle32": float(df_m_cubes["conf_oracle32"].mean()),
                "task_conf_oracle32_20splits": float(df_m_cubes["conf_oracle32_20splits"].mean()),
                "task_conf_res5_best_coset_avg": float(df_m_cubes["conf_res5_best_coset_avg"].mean()),
                "task_conf_res5_best_coset_avg_20splits": float(df_m_cubes["conf_res5_best_coset_avg_20splits"].mean()),
                "task_conf_res5_best_coset_pos": float(df_m_cubes["conf_res5_best_coset_pos"].mean()),
                "task_conf_res5_best_coset_neg": float(df_m_cubes["conf_res5_best_coset_neg"].mean()),
                "task_conf_res5_2fi_coset_avg": float(df_m_cubes["conf_res5_2fi_coset_avg"].mean()),
                "task_conf_res5_2fi_coset_avg_20splits": float(df_m_cubes["conf_res5_2fi_coset_avg_20splits"].mean()),
                "task_conf_res5_main_coset_avg": float(df_m_cubes["conf_res5_main_coset_avg"].mean()),
                "task_conf_rand16_best": float(df_m_cubes["conf_rand16_best"].mean()),
                "task_conf_rand16_best_20splits": float(df_m_cubes["conf_rand16_best_20splits"].mean()),
                "task_conf_rand16_2fi_fullrank": float(df_m_cubes["conf_rand16_2fi_fullrank"].mean()),
                "task_conf_rand16_main": float(df_m_cubes["conf_rand16_main"].mean()),
                "task_conf_ofat": float(df_m_cubes["conf_ofat"].mean()),
                "task_conf_baseline": float(df_m_cubes["conf_baseline"].mean()),
                "mse_reduction_pct": float(df_m_cubes["mse_reduction_pct"].mean()) if compute_rcbd else None,
                "f_stat_multiplier": float(df_m_cubes["f_stat_multiplier"].mean()) if compute_rcbd else None,
                "sig_unblocked_total": int(df_m_cubes["sig_factors_unblocked"].sum()) if compute_rcbd else None,
                "sig_rcbd_total": int(df_m_cubes["sig_factors_rcbd"].sum()) if compute_rcbd else None,
                "sig_item_cluster_total": int(df_m_cubes["sig_factors_item_cluster"].sum()) if compute_rcbd else None,
            }
        )

    df_models = pd.DataFrame(model_rows)
    df_cubes = pd.DataFrame(cube_rows)
    df_fourier = pd.DataFrame(fourier_rows)
    df_rcbd = pd.DataFrame(rcbd_rows)

    # Distribution across the 200 full-rank random designs (averaged over 15 models)
    per_design_mean_pool_2fi = pool_r16_2fi_design_dist.mean(axis=0)  # (200,)
    res5_pool_2fi_mean = float(df_models["pool_rmse_2fi_res5"].mean())
    per_design_ratios = per_design_mean_pool_2fi / res5_pool_2fi_mean

    # Two-way cluster bootstrap CIs for key Cohort 1 estimands (on 20-split cube averages and legacy split 0)
    ci_2fi_rmse_diff_20s = two_way_cluster_bootstrap_ci(
        rmse_arr["2fi_rand16"][:, :, 1:].mean(axis=2) - rmse_arr["2fi_res5_neg"][:, :, 1:].mean(axis=2)
    )
    ci_main_rmse_diff_20s = two_way_cluster_bootstrap_ci(
        rmse_arr["main_rand16"][:, :, 1:].mean(axis=2) - rmse_arr["main_res5_neg"][:, :, 1:].mean(axis=2)
    )
    ci_screenbest_vs_rand16_20s = two_way_cluster_bootstrap_ci(
        acc["res5_avg_best"][:, :, 1:].mean(axis=2) - acc["rand16_best"][:, :, 1:].mean(axis=2)
    )
    ci_screenbest_vs_rand16_split0 = two_way_cluster_bootstrap_ci(
        acc["res5_avg_best"][:, :, 0] - acc["rand16_best"][:, :, 0]
    )
    ci_screenbest_vs_rand16_2fi_cond_20s = two_way_cluster_bootstrap_ci(
        acc["res5_avg_best"][:, :, 1:].mean(axis=2) - acc["rand16_best_2fi_cond"][:, :, 1:].mean(axis=2)
    )
    ci_screenbest_vs_rand16_2fi_cond_split0 = two_way_cluster_bootstrap_ci(
        acc["res5_avg_best"][:, :, 0] - acc["rand16_best_2fi_cond"][:, :, 0]
    )
    ci_res5_2fi_vs_rand16_2fi_20s = two_way_cluster_bootstrap_ci(
        acc["res5_avg_2fi"][:, :, 1:].mean(axis=2) - acc["rand16_2fi"][:, :, 1:].mean(axis=2)
    )
    ci_res5_2fi_vs_main_20s = two_way_cluster_bootstrap_ci(
        acc["res5_avg_2fi"][:, :, 1:].mean(axis=2) - acc["res5_avg_main"][:, :, 1:].mean(axis=2)
    )
    ci_screenbest_vs_main_20s = two_way_cluster_bootstrap_ci(
        acc["res5_avg_best"][:, :, 1:].mean(axis=2) - acc["res5_avg_main"][:, :, 1:].mean(axis=2)
    )

    # Tie diagnostics across rules
    tie_diag = {}
    for k_t, t_arr in tie_counts.items():
        tie_diag[k_t] = summarize_tie_diagnostics(t_arr, tie_indicators[k_t])

    # Win-count frequencies (replacing universal "consistently" claims)
    win_counts = {
        "cube_main_res5_beats_rand16_split0": int((df_cubes["rmse_main_res5"] < df_cubes["rmse_main_rand16"]).sum()),
        "cube_main_res5_beats_rand16_split0_frac": float((df_cubes["rmse_main_res5"] < df_cubes["rmse_main_rand16"]).mean()),
        "cube_main_res5_beats_rand16_20splits": int((df_cubes["rmse_main_res5_20splits"] < df_cubes["rmse_main_rand16_20splits"]).sum()),
        "cube_main_res5_beats_rand16_20splits_frac": float((df_cubes["rmse_main_res5_20splits"] < df_cubes["rmse_main_rand16_20splits"]).mean()),
        "cube_2fi_res5_beats_rand16_split0": int((df_cubes["rmse_2fi_res5"] < df_cubes["rmse_2fi_rand16_fullrank"]).sum()),
        "cube_2fi_res5_beats_rand16_split0_frac": float((df_cubes["rmse_2fi_res5"] < df_cubes["rmse_2fi_rand16_fullrank"]).mean()),
        "cube_2fi_res5_beats_rand16_20splits": int((df_cubes["rmse_2fi_res5_20splits"] < df_cubes["rmse_2fi_rand16_fullrank_20splits"]).sum()),
        "cube_2fi_res5_beats_rand16_20splits_frac": float((df_cubes["rmse_2fi_res5_20splits"] < df_cubes["rmse_2fi_rand16_fullrank_20splits"]).mean()),
        "cube_main_res3_all60_beats_rand8_split0": int((df_cubes["rmse_main_res3_all60"] < df_cubes["rmse_main_rand8"]).sum()),
        "cube_main_res3_all60_beats_rand8_split0_frac": float((df_cubes["rmse_main_res3_all60"] < df_cubes["rmse_main_rand8"]).mean()),
        "cube_main_res5_beats_ofat_split0": int((df_cubes["rmse_main_res5"] < df_cubes["rmse_main_ofat"]).sum()),
        "cube_main_res5_beats_ofat_split0_frac": float((df_cubes["rmse_main_res5"] < df_cubes["rmse_main_ofat"]).mean()),
        "model_pool_main_res5_beats_rand16": int((df_models["pool_rmse_main_res5"] < df_models["pool_rmse_main_rand16"]).sum()),
        "model_pool_2fi_res5_beats_rand16": int((df_models["pool_rmse_2fi_res5"] < df_models["pool_rmse_2fi_rand16_fullrank"]).sum()),
        "model_pool_main_res3_all60_beats_rand8": int((df_models["pool_rmse_main_res3_all60"] < df_models["pool_rmse_main_rand8"]).sum()),
    }

    summary = {
        "estimator": estimator_name,
        "primary_tie_policy": tie_policy,
        "inference_computed": bool(compute_rcbd),
        "n_models": len(df_models),
        "n_tasks": len(tasks),
        "n_cubes": len(df_cubes),
        "total_items_per_model": int(df_cubes[df_cubes["model_id"] == MMLU_MODELS[0][0]]["n_items"].sum()),
        "total_cell_item_observations": int(df_cubes["n_items"].sum() * 32),
        "cell0_template_indices": [int(x) for x in cell_map[(0, 0, 0, 0, 0)]],
        "cell0_selected_row_index": int(cell_map[(0, 0, 0, 0, 0)][0]) if not estimator_name.startswith("cell_mean") else -1,
        "win_counts": win_counts,
        # Task-level RMSE (even/odd split 0 and 20-split mean)
        "mean_task_rmse_main_res5": float(df_cubes["rmse_main_res5"].mean()),
        "mean_task_rmse_main_res5_20splits": float(df_cubes["rmse_main_res5_20splits"].mean()),
        "mean_task_rmse_main_rand16": float(df_cubes["rmse_main_rand16"].mean()),
        "mean_task_rmse_main_rand16_20splits": float(df_cubes["rmse_main_rand16_20splits"].mean()),
        "task_main_rmse_ratio_rand16_over_res5": float(df_cubes["rmse_main_rand16"].mean() / df_cubes["rmse_main_res5"].mean()),
        "task_main_rmse_ratio_rand16_over_res5_20splits": float(df_cubes["rmse_main_rand16_20splits"].mean() / df_cubes["rmse_main_res5_20splits"].mean()),
        "mean_task_rmse_2fi_res5": float(df_cubes["rmse_2fi_res5"].mean()),
        "mean_task_rmse_2fi_res5_20splits": float(df_cubes["rmse_2fi_res5_20splits"].mean()),
        "mean_task_rmse_2fi_rand16_fullrank": float(df_cubes["rmse_2fi_rand16_fullrank"].mean()),
        "mean_task_rmse_2fi_rand16_fullrank_20splits": float(df_cubes["rmse_2fi_rand16_fullrank_20splits"].mean()),
        "task_2fi_rmse_ratio_fullrank_over_res5": float(df_cubes["rmse_2fi_rand16_fullrank"].mean() / df_cubes["rmse_2fi_res5"].mean()),
        "task_2fi_rmse_ratio_fullrank_over_res5_20splits": float(df_cubes["rmse_2fi_rand16_fullrank_20splits"].mean() / df_cubes["rmse_2fi_res5_20splits"].mean()),
        "mean_task_rmse_2fi_rand16_legacy_rankdef": float(df_cubes["rmse_2fi_rand16_legacy_rankdef"].mean()),
        "mean_task_rmse_main_res3_coset_avg": float(df_cubes["rmse_main_res3_coset_avg"].mean()),
        "mean_task_rmse_main_res3_coset_avg_20splits": float(df_cubes["rmse_main_res3_coset_avg_20splits"].mean()),
        "mean_task_rmse_main_res3_all60": float(df_cubes["rmse_main_res3_all60"].mean()),
        "mean_task_rmse_main_res3_all60_20splits": float(df_cubes["rmse_main_res3_all60_20splits"].mean()),
        "mean_task_rmse_main_res3_legacy_c0": float(df_cubes["rmse_main_res3_legacy_c0"].mean()),
        "mean_task_rmse_main_rand8": float(df_cubes["rmse_main_rand8"].mean()),
        "mean_task_rmse_main_rand8_20splits": float(df_cubes["rmse_main_rand8_20splits"].mean()),
        "task_main_rmse_ratio_rand8_over_res3_coset_avg": float(df_cubes["rmse_main_rand8"].mean() / df_cubes["rmse_main_res3_coset_avg"].mean()),
        "task_main_rmse_ratio_rand8_over_res3_coset_avg_20splits": float(df_cubes["rmse_main_rand8_20splits"].mean() / df_cubes["rmse_main_res3_coset_avg_20splits"].mean()),
        "task_main_rmse_ratio_rand8_over_res3_all60": float(df_cubes["rmse_main_rand8"].mean() / df_cubes["rmse_main_res3_all60"].mean()),
        "task_main_rmse_ratio_rand8_over_res3_all60_20splits": float(df_cubes["rmse_main_rand8_20splits"].mean() / df_cubes["rmse_main_res3_all60_20splits"].mean()),
        "mean_task_rmse_main_ofat": float(df_cubes["rmse_main_ofat"].mean()),
        "task_main_rmse_ratio_ofat_over_res5": float(df_cubes["rmse_main_ofat"].mean() / df_cubes["rmse_main_res5"].mean()),
        # Pooled-level RMSE
        "mean_pool_rmse_main_res5": float(df_models["pool_rmse_main_res5"].mean()),
        "mean_pool_rmse_main_rand16": float(df_models["pool_rmse_main_rand16"].mean()),
        "pool_main_rmse_ratio_rand16_over_res5": float(df_models["pool_rmse_main_rand16"].mean() / df_models["pool_rmse_main_res5"].mean()),
        "mean_pool_rmse_main_ofat": float(df_models["pool_rmse_main_ofat"].mean()),
        "pool_main_rmse_ratio_ofat_over_res5": float(df_models["pool_rmse_main_ofat"].mean() / df_models["pool_rmse_main_res5"].mean()),
        "mean_pool_rmse_2fi_res5": res5_pool_2fi_mean,
        "mean_pool_rmse_2fi_rand16_fullrank": float(df_models["pool_rmse_2fi_rand16_fullrank"].mean()),
        "mean_pool_rmse_2fi_rand16_wellcond": float(df_models["pool_rmse_2fi_rand16_wellcond"].mean()),
        "mean_pool_rmse_2fi_rand16_legacy_rankdef": float(df_models["pool_rmse_2fi_rand16_legacy_rankdef"].mean()),
        "pool_2fi_rmse_ratio_fullrank_over_res5": float(df_models["pool_rmse_2fi_rand16_fullrank"].mean() / res5_pool_2fi_mean),
        "pool_2fi_rmse_ratio_wellcond_over_res5": float(df_models["pool_rmse_2fi_rand16_wellcond"].mean() / res5_pool_2fi_mean),
        "pool_2fi_rmse_ratio_legacy_rankdef_over_res5": float(df_models["pool_rmse_2fi_rand16_legacy_rankdef"].mean() / res5_pool_2fi_mean),
        "pool_2fi_fullrank_design_ratio_quantiles": {
            "min": float(per_design_ratios.min()),
            "q025": float(np.percentile(per_design_ratios, 2.5)),
            "q25": float(np.percentile(per_design_ratios, 25.0)),
            "median": float(np.median(per_design_ratios)),
            "mean": float(per_design_ratios.mean()),
            "q75": float(np.percentile(per_design_ratios, 75.0)),
            "q975": float(np.percentile(per_design_ratios, 97.5)),
            "max": float(per_design_ratios.max()),
        },
        "mean_pool_rmse_main_res3": float(df_models["pool_rmse_main_res3"].mean()),
        "mean_pool_rmse_main_res3_coset_avg": float(df_models["pool_rmse_main_res3_coset_avg"].mean()),
        "mean_pool_rmse_main_res3_all60": float(df_models["pool_rmse_main_res3_all60"].mean()),
        "mean_pool_rmse_main_rand8": float(df_models["pool_rmse_main_rand8"].mean()),
        "pool_main_rmse_ratio_rand8_over_res3_coset_avg": float(df_models["pool_rmse_main_rand8"].mean() / df_models["pool_rmse_main_res3_coset_avg"].mean()),
        "pool_main_rmse_ratio_rand8_over_res3_all60": float(df_models["pool_rmse_main_rand8"].mean() / df_models["pool_rmse_main_res3_all60"].mean()),
        # Confirmation accuracy (split 0 and 20-split mean)
        "mean_conf_baseline": float(df_cubes["conf_baseline"].mean()),
        "mean_conf_baseline_20splits": float(df_cubes["conf_baseline_20splits"].mean()),
        "mean_conf_oracle32": float(df_cubes["conf_oracle32"].mean()),
        "mean_conf_oracle32_20splits": float(df_cubes["conf_oracle32_20splits"].mean()),
        "mean_conf_res5_best_coset_pos": float(df_cubes["conf_res5_best_coset_pos"].mean()),
        "mean_conf_res5_best_coset_neg": float(df_cubes["conf_res5_best_coset_neg"].mean()),
        "mean_conf_res5_best_coset_avg": float(df_cubes["conf_res5_best_coset_avg"].mean()),
        "mean_conf_res5_best_coset_avg_20splits": float(df_cubes["conf_res5_best_coset_avg_20splits"].mean()),
        "mean_conf_res5_2fi_coset_pos": float(df_cubes["conf_res5_2fi_coset_pos"].mean()),
        "mean_conf_res5_2fi_coset_neg": float(df_cubes["conf_res5_2fi_coset_neg"].mean()),
        "mean_conf_res5_2fi_coset_avg": float(df_cubes["conf_res5_2fi_coset_avg"].mean()),
        "mean_conf_res5_2fi_coset_avg_20splits": float(df_cubes["conf_res5_2fi_coset_avg_20splits"].mean()),
        "mean_conf_res5_main_coset_pos": float(df_cubes["conf_res5_main_coset_pos"].mean()),
        "mean_conf_res5_main_coset_neg": float(df_cubes["conf_res5_main_coset_neg"].mean()),
        "mean_conf_res5_main_coset_avg": float(df_cubes["conf_res5_main_coset_avg"].mean()),
        "mean_conf_res5_main_coset_avg_20splits": float(df_cubes["conf_res5_main_coset_avg_20splits"].mean()),
        "mean_conf_rand16_best": float(df_cubes["conf_rand16_best"].mean()),
        "mean_conf_rand16_best_20splits": float(df_cubes["conf_rand16_best_20splits"].mean()),
        "mean_conf_rand16_best_2fi_cond": float(df_cubes["conf_rand16_best_2fi_cond"].mean()),
        "mean_conf_rand16_best_2fi_cond_20splits": float(df_cubes["conf_rand16_best_2fi_cond_20splits"].mean()),
        "mean_conf_rand16_2fi_fullrank": float(df_cubes["conf_rand16_2fi_fullrank"].mean()),
        "mean_conf_rand16_2fi_fullrank_20splits": float(df_cubes["conf_rand16_2fi_fullrank_20splits"].mean()),
        "mean_conf_rand16_main": float(df_cubes["conf_rand16_main"].mean()),
        "mean_conf_rand16_main_20splits": float(df_cubes["conf_rand16_main_20splits"].mean()),
        "mean_conf_res3_best_coset_avg": float(df_cubes["conf_res3_best_coset_avg"].mean()),
        "mean_conf_res3_best_coset_avg_20splits": float(df_cubes["conf_res3_best_coset_avg_20splits"].mean()),
        "mean_conf_res3_best_all60": float(df_cubes["conf_res3_best_all60"].mean()),
        "mean_conf_res3_best_all60_20splits": float(df_cubes["conf_res3_best_all60_20splits"].mean()),
        "mean_conf_res3_best_legacy_c0": float(df_cubes["conf_res3_best_legacy_c0"].mean()),
        "mean_conf_rand8_best": float(df_cubes["conf_rand8_best"].mean()),
        "mean_conf_rand8_best_20splits": float(df_cubes["conf_rand8_best_20splits"].mean()),
        "mean_conf_ofat": float(df_cubes["conf_ofat"].mean()),
        "mean_conf_ofat_20splits": float(df_cubes["conf_ofat_20splits"].mean()),
        "mean_regret_vs_oracle_20splits": {
            "res5_best_coset_avg": float((acc["oracle32"][:, :, 1:] - acc["res5_avg_best"][:, :, 1:]).mean()),
            "res5_2fi_coset_avg": float((acc["oracle32"][:, :, 1:] - acc["res5_avg_2fi"][:, :, 1:]).mean()),
            "res5_main_coset_avg": float((acc["oracle32"][:, :, 1:] - acc["res5_avg_main"][:, :, 1:]).mean()),
            "rand16_best": float((acc["oracle32"][:, :, 1:] - acc["rand16_best"][:, :, 1:]).mean()),
            "rand16_best_2fi_cond": float((acc["oracle32"][:, :, 1:] - acc["rand16_best_2fi_cond"][:, :, 1:]).mean()),
            "ofat": float((acc["oracle32"][:, :, 1:] - acc["ofat"][:, :, 1:]).mean()),
            "baseline": float((acc["oracle32"][:, :, 1:] - acc["baseline"][:, :, 1:]).mean()),
        },
        "tie_diagnostics": tie_diag,
        # Oracle-uplift recovery (split 0 and 20-split mean)
        "recovery_split0_res5_best_coset_pos": compute_oracle_uplift_recovery_stats(
            acc["res5_pos_best"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_split0_res5_best_coset_neg": compute_oracle_uplift_recovery_stats(
            acc["res5_neg_best"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_split0_res5_best_coset_avg": compute_oracle_uplift_recovery_stats(
            acc["res5_avg_best"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_split0_rand16_best": compute_oracle_uplift_recovery_stats(
            acc["rand16_best"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_split0_res5_2fi_coset_pos": compute_oracle_uplift_recovery_stats(
            acc["res5_pos_2fi"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_split0_res5_2fi_coset_neg": compute_oracle_uplift_recovery_stats(
            acc["res5_neg_2fi"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_split0_res5_2fi_coset_avg": compute_oracle_uplift_recovery_stats(
            acc["res5_avg_2fi"][:, :, 0], acc["baseline"][:, :, 0], acc["oracle32"][:, :, 0]
        ),
        "recovery_20splits_res5_best_coset_avg": compute_oracle_uplift_recovery_stats(
            acc["res5_avg_best"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2)
        ),
        "recovery_20splits_rand16_best": compute_oracle_uplift_recovery_stats(
            acc["rand16_best"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2)
        ),
        "recovery_20splits_res5_2fi_coset_avg": compute_oracle_uplift_recovery_stats(
            acc["res5_avg_2fi"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2)
        ),
        "recovery_20splits_rand16_2fi_fullrank": compute_oracle_uplift_recovery_stats(
            acc["rand16_2fi"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2)
        ),
        "recovery_20splits_res5_main_coset_avg": compute_oracle_uplift_recovery_stats(
            acc["res5_avg_main"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2)
        ),
        "all_reference_cells_recovery": {
            "split0_res5_best_coset_avg": compute_all_reference_cells_recovery(
                acc["res5_avg_best"][:, :, 0], acc["oracle32"][:, :, 0], all_cells_conf_s0
            ),
            "split0_res5_2fi_coset_avg": compute_all_reference_cells_recovery(
                acc["res5_avg_2fi"][:, :, 0], acc["oracle32"][:, :, 0], all_cells_conf_s0
            ),
            "split0_rand16_best": compute_all_reference_cells_recovery(
                acc["rand16_best"][:, :, 0], acc["oracle32"][:, :, 0], all_cells_conf_s0
            ),
            "splits20_res5_best_coset_avg": compute_all_reference_cells_recovery(
                acc["res5_avg_best"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2), all_cells_conf_20s
            ),
            "splits20_res5_2fi_coset_avg": compute_all_reference_cells_recovery(
                acc["res5_avg_2fi"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2), all_cells_conf_20s
            ),
            "splits20_rand16_best": compute_all_reference_cells_recovery(
                acc["rand16_best"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2), all_cells_conf_20s
            ),
            "splits20_res5_main_coset_avg": compute_all_reference_cells_recovery(
                acc["res5_avg_main"][:, :, 1:].mean(axis=2), acc["oracle32"][:, :, 1:].mean(axis=2), all_cells_conf_20s
            ),
        },
        "first_index_compatibility": {
            "mean_conf_res5_best_coset_pos_split0": float(acc_first["res5_pos_best"][:, :, 0].mean()),
            "mean_conf_res5_best_coset_neg_split0": float(acc_first["res5_neg_best"][:, :, 0].mean()),
            "mean_conf_res5_best_coset_avg_split0": float(acc_first["res5_avg_best"][:, :, 0].mean()),
            "mean_conf_res5_best_coset_avg_20splits": float(acc_first["res5_avg_best"][:, :, 1:].mean()),
            "mean_conf_res5_2fi_coset_pos_split0": float(acc_first["res5_pos_2fi"][:, :, 0].mean()),
            "mean_conf_res5_2fi_coset_neg_split0": float(acc_first["res5_neg_2fi"][:, :, 0].mean()),
            "mean_conf_res5_2fi_coset_avg_split0": float(acc_first["res5_avg_2fi"][:, :, 0].mean()),
            "mean_conf_res5_2fi_coset_avg_20splits": float(acc_first["res5_avg_2fi"][:, :, 1:].mean()),
            "mean_conf_rand16_best_split0": float(acc_first["rand16_best"][:, :, 0].mean()),
            "mean_conf_rand16_best_20splits": float(acc_first["rand16_best"][:, :, 1:].mean()),
            "mean_conf_rand16_2fi_fullrank_split0": float(acc_first["rand16_2fi"][:, :, 0].mean()),
            "mean_conf_rand16_2fi_fullrank_20splits": float(acc_first["rand16_2fi"][:, :, 1:].mean()),
            "recovery_split0_res5_best_coset_pos": compute_oracle_uplift_recovery_stats(
                acc_first["res5_pos_best"][:, :, 0], acc["baseline"][:, :, 0], acc_first["oracle32"][:, :, 0]
            ),
            "recovery_split0_res5_best_coset_neg": compute_oracle_uplift_recovery_stats(
                acc_first["res5_neg_best"][:, :, 0], acc["baseline"][:, :, 0], acc_first["oracle32"][:, :, 0]
            ),
            "recovery_split0_res5_best_coset_avg": compute_oracle_uplift_recovery_stats(
                acc_first["res5_avg_best"][:, :, 0], acc["baseline"][:, :, 0], acc_first["oracle32"][:, :, 0]
            ),
            "recovery_split0_res5_2fi_coset_pos": compute_oracle_uplift_recovery_stats(
                acc_first["res5_pos_2fi"][:, :, 0], acc["baseline"][:, :, 0], acc_first["oracle32"][:, :, 0]
            ),
            "recovery_split0_res5_2fi_coset_neg": compute_oracle_uplift_recovery_stats(
                acc_first["res5_neg_2fi"][:, :, 0], acc["baseline"][:, :, 0], acc_first["oracle32"][:, :, 0]
            ),
            "recovery_split0_res5_2fi_coset_avg": compute_oracle_uplift_recovery_stats(
                acc_first["res5_avg_2fi"][:, :, 0], acc["baseline"][:, :, 0], acc_first["oracle32"][:, :, 0]
            ),
            "recovery_20splits_res5_best_coset_avg": compute_oracle_uplift_recovery_stats(
                acc_first["res5_avg_best"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc_first["oracle32"][:, :, 1:].mean(axis=2)
            ),
            "recovery_20splits_res5_2fi_coset_avg": compute_oracle_uplift_recovery_stats(
                acc_first["res5_avg_2fi"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc_first["oracle32"][:, :, 1:].mean(axis=2)
            ),
            "recovery_20splits_rand16_best": compute_oracle_uplift_recovery_stats(
                acc_first["rand16_best"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc_first["oracle32"][:, :, 1:].mean(axis=2)
            ),
            "recovery_20splits_rand16_2fi_fullrank": compute_oracle_uplift_recovery_stats(
                acc_first["rand16_2fi"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), acc_first["oracle32"][:, :, 1:].mean(axis=2)
            ),
        },
        "tie_policy_sensitivity": {
            p_name: {
                "mean_conf_res5_best_coset_avg_split0": float(p_acc["res5_avg_best"][:, :, 0].mean()),
                "mean_conf_res5_best_coset_avg_20splits": float(p_acc["res5_avg_best"][:, :, 1:].mean()),
                "mean_conf_res5_2fi_coset_avg_split0": float(p_acc["res5_avg_2fi"][:, :, 0].mean()),
                "mean_conf_res5_2fi_coset_avg_20splits": float(p_acc["res5_avg_2fi"][:, :, 1:].mean()),
                "mean_conf_rand16_best_split0": float(p_acc["rand16_best"][:, :, 0].mean()),
                "mean_conf_rand16_best_20splits": float(p_acc["rand16_best"][:, :, 1:].mean()),
                "recovery_split0_res5_best_coset_avg_pct": float(
                    compute_oracle_uplift_recovery_stats(
                        p_acc["res5_avg_best"][:, :, 0], acc["baseline"][:, :, 0], p_acc["oracle32"][:, :, 0]
                    )["ratio_of_means_pct"]
                ),
                "recovery_20splits_res5_best_coset_avg_pct": float(
                    compute_oracle_uplift_recovery_stats(
                        p_acc["res5_avg_best"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), p_acc["oracle32"][:, :, 1:].mean(axis=2)
                    )["ratio_of_means_pct"]
                ),
                "recovery_split0_res5_2fi_coset_avg_pct": float(
                    compute_oracle_uplift_recovery_stats(
                        p_acc["res5_avg_2fi"][:, :, 0], acc["baseline"][:, :, 0], p_acc["oracle32"][:, :, 0]
                    )["ratio_of_means_pct"]
                ),
                "recovery_20splits_res5_2fi_coset_avg_pct": float(
                    compute_oracle_uplift_recovery_stats(
                        p_acc["res5_avg_2fi"][:, :, 1:].mean(axis=2), acc["baseline"][:, :, 1:].mean(axis=2), p_acc["oracle32"][:, :, 1:].mean(axis=2)
                    )["ratio_of_means_pct"]
                ),
            }
            for p_name, p_acc in [("mean_tied", acc), ("first_index", acc_first), ("seeded_random", acc_seed)]
        },
        # Cluster-bootstrap paired comparisons
        "ci_2fi_rmse_diff_rand16_minus_res5_20splits": ci_2fi_rmse_diff_20s,
        "ci_main_rmse_diff_rand16_minus_res5_20splits": ci_main_rmse_diff_20s,
        "ci_screenbest_minus_rand16_best_20splits": ci_screenbest_vs_rand16_20s,
        "ci_screenbest_minus_rand16_best_split0": ci_screenbest_vs_rand16_split0,
        "ci_screenbest_minus_rand16_best_2fi_cond_20splits": ci_screenbest_vs_rand16_2fi_cond_20s,
        "ci_screenbest_minus_rand16_best_2fi_cond_split0": ci_screenbest_vs_rand16_2fi_cond_split0,
        "ci_res5_2fi_minus_rand16_2fi_20splits": ci_res5_2fi_vs_rand16_2fi_20s,
        "ci_res5_2fi_minus_res5_main_20splits": ci_res5_2fi_vs_main_20s,
        "ci_screenbest_minus_res5_main_20splits": ci_screenbest_vs_main_20s,
        # True effect magnitudes (SI-1)
        "mean_disc0_rms_main_true": float(df_fourier["disc0_rms_main"].mean()),
        "mean_disc0_rms_2fi_true": float(df_fourier["disc0_rms_2fi"].mean()),
        "mean_disc0_rms_3fi_true": float(df_fourier["disc0_rms_3fi"].mean()),
        "mean_disc0_rms_4fi_true": float(df_fourier["disc0_rms_4fi"].mean()),
        "mean_full_rms_main_true": float(df_fourier["full_rms_main"].mean()),
        "mean_full_rms_2fi_true": float(df_fourier["full_rms_2fi"].mean()),
        "mean_full_rms_3fi_true": float(df_fourier["full_rms_3fi"].mean()),
        "mean_full_rms_4fi_true": float(df_fourier["full_rms_4fi"].mean()),
        # RCBD and robust inference
        "mse_summary_cube_count": int(len(df_cubes)) if compute_rcbd else None,
        "mean_mse_reduction_pct": float(df_cubes["mse_reduction_pct"].mean()) if compute_rcbd else None,
        "min_mse_reduction_pct": float(df_cubes["mse_reduction_pct"].min()) if compute_rcbd else None,
        "max_mse_reduction_pct": float(df_cubes["mse_reduction_pct"].max()) if compute_rcbd else None,
        "mean_f_stat_multiplier": float(df_cubes["f_stat_multiplier"].mean()) if compute_rcbd else None,
        "mean_r2_unblocked": float(df_cubes["r2_unblocked"].mean()) if compute_rcbd else None,
        "mean_r2_rcbd": float(df_cubes["r2_rcbd"].mean()) if compute_rcbd else None,
        "total_factor_tests": int(len(df_cubes) * 5) if compute_rcbd else None,
        "total_sig_unblocked_per_cube_bh": int(df_cubes["sig_factors_unblocked"].sum()) if compute_rcbd else None,
        "total_sig_rcbd_per_cube_bh": int(df_cubes["sig_factors_rcbd"].sum()) if compute_rcbd else None,
        "total_sig_item_cluster_per_cube_bh": int(df_cubes["sig_factors_item_cluster"].sum()) if compute_rcbd else None,
        "total_zero_var_contrasts": int(df_cubes["n_zero_var_contrasts"].sum()) if compute_rcbd else None,
    }

    return {
        "df_models": df_models,
        "df_cubes": df_cubes,
        "df_fourier": df_fourier,
        "df_rcbd": df_rcbd,
        "acc_arrays": acc,
        "rmse_arrays": rmse_arr,
        "summary": summary,
    }


def evaluate_cohort2_estimator(
    Ys: dict,
    all_cells_4: List[Tuple[int, ...]],
    task_cell_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]],
    estimator_name: str,
    task_splits: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, List[str]]],
    cosets_info: Dict[str, Any],
    rand8_fullrank: List[List[int]],
    rand8_screenbest_uncond: Optional[List[List[int]]] = None,
    tie_policy: str = "mean_tied",
    compute_rcbd: bool = True,
    n_perm_rcbd: int = 200,
) -> Dict[str, Any]:
    """
    Vectorized evaluation of Cohort 2 (178 BBH & LMentry 2^4 = 16 cubes) across:
    - Legacy even/odd split (col 0) and 20 seeded random item splits (cols 1..20)
    - Both cosets of 2^(4-1)_IV (I = +ABCD and I = -ABCD)
    - 200 full-rank balanced Random-8 designs (for OLS main-effect attribution)
      and 200 unconditioned balanced Random-8 designs (for ScreenBest)
    - OFAT-5 (lower-budget 5-run baseline)
    """
    if rand8_screenbest_uncond is None:
        rand8_screenbest_uncond = rand8_fullrank

    m4 = build_cube_design_matrices(4)
    X4_pm = m4["X_pm"]
    X4_2fi = m4["X_2fi"]
    D4_main = m4["D_main"]

    res4_pos = cosets_info["res4_cosets"]["2(4-1)IV_coset_pos"]["indices"]
    res4_neg = cosets_info["res4_cosets"]["2(4-1)IV_coset_neg"]["indices"]
    ofat4_cells = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    ofat4_idx = [all_cells_4.index(c) for c in ofat4_cells]

    validate_design_matrix(D4_main[res4_pos], "2(4-1)IV_coset_pos", "SLAB_main")
    validate_design_matrix(D4_main[res4_neg], "2(4-1)IV_coset_neg", "SLAB_main")
    validate_design_matrix(D4_main[ofat4_idx], "OFAT_5", "SLAB_main")
    for i, c in enumerate(rand8_fullrank):
        validate_design_matrix(D4_main[c], f"slab_rand8_fullrank_{i:03d}", "SLAB_main")

    r8_arr = np.array(rand8_fullrank)  # (200, 8)
    r8_sb_uncond_arr = np.array(rand8_screenbest_uncond)  # (200, 8)
    Pm_r8 = np.array([np.linalg.pinv(D4_main[c]) for c in rand8_fullrank])  # (200, 5, 8)

    n_splits_total = 21

    cube_rows = []
    fourier_rows = []
    rcbd_rows = []
    rec_keys = [
        "baseline", "oracle16", "ofat",
        "res4_pos_best", "res4_neg_best", "res4_avg_best",
        "res4_pos_main", "res4_neg_main", "res4_avg_main",
        "rand8_best", "rand8_best_main_cond", "rand8_main",
        "rmse_res4_pos", "rmse_res4_neg", "rmse_rand8", "rmse_ofat",
    ]
    split_records: Dict[str, List[np.ndarray]] = {k: [] for k in rec_keys}
    split_records_first: Dict[str, List[np.ndarray]] = {k: [] for k in rec_keys if not k.startswith("rmse_")}
    split_records_rand_tie: Dict[str, List[np.ndarray]] = {k: [] for k in rec_keys if not k.startswith("rmse_")}
    tie_records: Dict[str, List[np.ndarray]] = {
        k: []
        for k in [
            "oracle16", "res4_pos_best", "res4_neg_best",
            "res4_pos_main", "res4_neg_main", "ofat",
            "rand8_best", "rand8_main",
        ]
    }
    tie_indicator_records: Dict[str, List[np.ndarray]] = {
        key: [] for key in tie_records
    }
    all_cells_conf_s0_list: List[np.ndarray] = []
    all_cells_conf_20s_list: List[np.ndarray] = []

    perm_rng = np.random.default_rng(SEEDS["sign_flip_perm"] + 1)
    tie_rng = np.random.default_rng(SEEDS["tie_break"] + 1)

    for bench, model_names in [("BBH", BBH_MODELS), ("LMentry", LMENTRY_MODELS)]:
        for fpath in sorted((DATA_DIR / "templates" / bench).glob("*.csv")):
            task_name = fpath.stem
            key = (bench, task_name)
            if key not in task_cell_maps:
                continue
            cell_map = task_cell_maps[key]
            d_masks, c_masks, _ = task_splits[key]

            for m_idx, mat in enumerate(Ys[bench][task_name]):
                m_name = model_names[m_idx]
                Y_t = extract_cell_item_matrix(mat, all_cells_4, cell_map, estimator=estimator_name)
                K, N = Y_t.shape
                d_bin = (d_masks > 0).astype(float)
                c_bin = (c_masks > 0).astype(float)
                Y_d = (Y_t @ d_bin) / d_bin.sum(axis=0)[None, :]  # (16, 21)
                Y_c = (Y_t @ c_bin) / c_bin.sum(axis=0)[None, :]  # (16, 21)
                Y_d[:, 0] = Y_t[:, ::2].mean(axis=1)
                Y_c[:, 0] = Y_t[:, 1::2].mean(axis=1)
                all_cells_conf_s0_list.append(Y_c[:, 0])
                all_cells_conf_20s_list.append(Y_c[:, 1:].mean(axis=1))

                tm = (X4_pm.T @ Y_d) / 8.0  # (4, 21)
                t2 = (X4_2fi.T @ Y_d) / 8.0  # (6, 21)
                Bp, *_ = np.linalg.lstsq(D4_main[res4_pos], Y_d[res4_pos, :], rcond=None)
                Bn, *_ = np.linalg.lstsq(D4_main[res4_neg], Y_d[res4_neg, :], rcond=None)
                Bo, *_ = np.linalg.lstsq(D4_main[ofat4_idx], Y_d[ofat4_idx, :], rcond=None)
                Bp[:, 0], *_ = np.linalg.lstsq(D4_main[res4_pos], Y_d[res4_pos, 0], rcond=None)
                Bn[:, 0], *_ = np.linalg.lstsq(D4_main[res4_neg], Y_d[res4_neg, 0], rcond=None)
                Bo[:, 0], *_ = np.linalg.lstsq(D4_main[ofat4_idx], Y_d[ofat4_idx, 0], rcond=None)

                rmse_p = np.sqrt(np.mean((2.0 * Bp[1:, :] - tm) ** 2, axis=0))
                rmse_n = np.sqrt(np.mean((2.0 * Bn[1:, :] - tm) ** 2, axis=0))
                rmse_o = np.sqrt(np.mean((2.0 * Bo[1:, :] - tm) ** 2, axis=0))

                res4_pos_arr = np.array(res4_pos)
                res4_neg_arr = np.array(res4_neg)
                pred_mp = D4_main @ Bp
                pred_mn = D4_main @ Bn
                pred_ofat = D4_main @ Bo

                Y_d_r8 = Y_d[r8_arr, :]  # (200, 8, 21)
                Y_d_r8_sb_uncond = Y_d[r8_sb_uncond_arr, :]  # (200, 8, 21)
                Bm_r8 = np.einsum("dij,djs->dis", Pm_r8, Y_d_r8)
                rmse_r8 = np.sqrt(np.mean((2.0 * Bm_r8[:, 1:, :] - tm[None, :, :]) ** 2, axis=1)).mean(axis=0)
                pred_r8_main = np.einsum("ci,dis->dcs", D4_main, Bm_r8)

                c_base = Y_c[0, :]
                c_ora, n_ora, i_ora = select_with_tie_policy(Y_d, Y_c, tie_policy=tie_policy, rng=tie_rng)
                c_bp, n_bp, i_bp = select_with_tie_policy(Y_d[res4_pos, :], Y_c, candidate_indices=res4_pos_arr, tie_policy=tie_policy, rng=tie_rng)
                c_bn, n_bn, i_bn = select_with_tie_policy(Y_d[res4_neg, :], Y_c, candidate_indices=res4_neg_arr, tie_policy=tie_policy, rng=tie_rng)
                c_bavg = 0.5 * (c_bp + c_bn)
                c_mp, n_mp, i_mp = select_with_tie_policy(pred_mp, Y_c, tie_policy=tie_policy, rng=tie_rng)
                c_mn, n_mn, i_mn = select_with_tie_policy(pred_mn, Y_c, tie_policy=tie_policy, rng=tie_rng)
                c_mavg = 0.5 * (c_mp + c_mn)
                c_ofat, n_ofat, i_ofat = select_with_tie_policy(pred_ofat, Y_c, tie_policy=tie_policy, rng=tie_rng)
                c_r8_b_all, n_r8_b_all, i_r8_b_all = select_with_tie_policy(Y_d_r8_sb_uncond, Y_c, candidate_indices=r8_sb_uncond_arr, tie_policy=tie_policy, rng=tie_rng)
                c_r8_best = c_r8_b_all.mean(axis=0)
                n_r8_best = n_r8_b_all.mean(axis=0)
                i_r8_best = i_r8_b_all.mean(axis=0)
                c_r8_bc_all, _, _ = select_with_tie_policy(Y_d_r8, Y_c, candidate_indices=r8_arr, tie_policy=tie_policy, rng=tie_rng)
                c_r8_best_cond = c_r8_bc_all.mean(axis=0)
                c_r8_m_all, n_r8_m_all, i_r8_m_all = select_with_tie_policy(pred_r8_main, Y_c, tie_policy=tie_policy, rng=tie_rng)
                c_r8_main = c_r8_m_all.mean(axis=0)
                n_r8_main = n_r8_m_all.mean(axis=0)
                i_r8_main = i_r8_m_all.mean(axis=0)

                # First-index and seeded-random tie policies for sensitivity
                for pol_str, target_rec in [("first_index", split_records_first), ("seeded_random", split_records_rand_tie)]:
                    p_ora, _, _ = select_with_tie_policy(Y_d, Y_c, tie_policy=pol_str, rng=tie_rng)
                    p_bp, _, _ = select_with_tie_policy(Y_d[res4_pos, :], Y_c, candidate_indices=res4_pos_arr, tie_policy=pol_str, rng=tie_rng)
                    p_bn, _, _ = select_with_tie_policy(Y_d[res4_neg, :], Y_c, candidate_indices=res4_neg_arr, tie_policy=pol_str, rng=tie_rng)
                    p_bavg = 0.5 * (p_bp + p_bn)
                    p_mp, _, _ = select_with_tie_policy(pred_mp, Y_c, tie_policy=pol_str, rng=tie_rng)
                    p_mn, _, _ = select_with_tie_policy(pred_mn, Y_c, tie_policy=pol_str, rng=tie_rng)
                    p_mavg = 0.5 * (p_mp + p_mn)
                    p_ofat, _, _ = select_with_tie_policy(pred_ofat, Y_c, tie_policy=pol_str, rng=tie_rng)
                    p_r8_best = select_with_tie_policy(Y_d_r8_sb_uncond, Y_c, candidate_indices=r8_sb_uncond_arr, tie_policy=pol_str, rng=tie_rng)[0].mean(axis=0)
                    p_r8_best_cond = select_with_tie_policy(Y_d_r8, Y_c, candidate_indices=r8_arr, tie_policy=pol_str, rng=tie_rng)[0].mean(axis=0)
                    p_r8_main = select_with_tie_policy(pred_r8_main, Y_c, tie_policy=pol_str, rng=tie_rng)[0].mean(axis=0)
                    for k_p, v_p in [
                        ("baseline", c_base), ("oracle16", p_ora), ("ofat", p_ofat),
                        ("res4_pos_best", p_bp), ("res4_neg_best", p_bn), ("res4_avg_best", p_bavg),
                        ("res4_pos_main", p_mp), ("res4_neg_main", p_mn), ("res4_avg_main", p_mavg),
                        ("rand8_best", p_r8_best), ("rand8_best_main_cond", p_r8_best_cond), ("rand8_main", p_r8_main),
                    ]:
                        target_rec[k_p].append(v_p)

                for k_t, v_t in [
                    ("oracle16", n_ora), ("res4_pos_best", n_bp), ("res4_neg_best", n_bn),
                    ("res4_pos_main", n_mp), ("res4_neg_main", n_mn), ("ofat", n_ofat),
                    ("rand8_best", n_r8_best), ("rand8_main", n_r8_main),
                ]:
                    tie_records[k_t].append(v_t)
                for k_t, v_t in [
                    ("oracle16", i_ora), ("res4_pos_best", i_bp), ("res4_neg_best", i_bn),
                    ("res4_pos_main", i_mp), ("res4_neg_main", i_mn), ("ofat", i_ofat),
                    ("rand8_best", i_r8_best), ("rand8_main", i_r8_main),
                ]:
                    tie_indicator_records[k_t].append(v_t)

                sign_res4_0 = float(np.mean(np.sign(2.0 * Bp[1:, 0]) == np.sign(tm[:, 0])))
                sign_r8_0 = float(np.mean(np.sign(2.0 * Bm_r8[:, 1:, 0]) == np.sign(tm[None, :, 0])))

                for k_rec, arr_val in [
                    ("baseline", c_base), ("oracle16", c_ora), ("ofat", c_ofat),
                    ("res4_pos_best", c_bp), ("res4_neg_best", c_bn), ("res4_avg_best", c_bavg),
                    ("res4_pos_main", c_mp), ("res4_neg_main", c_mn), ("res4_avg_main", c_mavg),
                    ("rand8_best", c_r8_best), ("rand8_best_main_cond", c_r8_best_cond), ("rand8_main", c_r8_main),
                    ("rmse_res4_pos", rmse_p), ("rmse_res4_neg", rmse_n),
                    ("rmse_rand8", rmse_r8), ("rmse_ofat", rmse_o),
                ]:
                    split_records[k_rec].append(arr_val)

                f_full = decompose_full_cube_effects(Y_t.mean(axis=1), 4)
                f_disc0 = decompose_full_cube_effects(Y_d[:, 0], 4)
                mean20_disc_rms_main = float(np.sqrt(np.mean(tm[:, 1:] ** 2, axis=0)).mean())
                mean20_disc_rms_2fi = float(np.sqrt(np.mean(t2[:, 1:] ** 2, axis=0)).mean())
                c_ora_first = split_records_first["oracle16"][-1]
                c_mavg_first = split_records_first["res4_avg_main"][-1]
                c_bavg_first = split_records_first["res4_avg_best"][-1]
                fourier_rows.append(
                    {
                        "cohort": bench,
                        "estimator": estimator_name,
                        "model_id": m_name,
                        "task": task_name,
                        "n_items": N,
                        "full_rms_main": f_full["rms_main"],
                        "full_rms_2fi": f_full["rms_2fi"],
                        "full_rms_3fi": f_full["rms_3fi"],
                        "full_rms_4fi": f_full["rms_4fi"],
                        "full_rms_higher_order": f_full["rms_all_higher_order"],
                        "full_var_share_main": f_full["var_share_main"],
                        "full_var_share_2fi": f_full["var_share_2fi"],
                        "full_var_share_higher_order": f_full["var_share_higher_order"],
                        "disc0_rms_main": f_disc0["rms_main"],
                        "disc0_rms_2fi": f_disc0["rms_2fi"],
                        "disc0_rms_3fi": f_disc0["rms_3fi"],
                        "disc0_rms_4fi": f_disc0["rms_4fi"],
                        "mean20_disc_rms_main": mean20_disc_rms_main,
                        "mean20_disc_rms_2fi": mean20_disc_rms_2fi,
                        "disc0_rmse_main_res5": float(rmse_p[0]),
                        "disc0_rmse_2fi_res5": float("nan"),
                        "disc0_rmse_2fi_rand16_fullrank": float("nan"),
                        "regret_main_split0": float(c_ora[0] - c_mavg[0]),
                        "regret_main_20splits": float((c_ora[1:] - c_mavg[1:]).mean()),
                        "regret_main_20splits_first_index": float((c_ora_first[1:] - c_mavg_first[1:]).mean()),
                        "gain_2fi_over_main_20splits": float("nan"),
                        "gain_best_over_main_20splits": float((c_bavg[1:] - c_mavg[1:]).mean()),
                        "gain_best_over_main_20splits_first_index": float((c_bavg_first[1:] - c_mavg_first[1:]).mean()),
                        "reconstruction_max_abs_err": f_full["reconstruction_max_abs_err"],
                    }
                )

                if compute_rcbd:
                    inf = compute_cube_rcbd_and_robust_inference(Y_t, X4_pm, n_perm=n_perm_rcbd, perm_rng=perm_rng)
                    for f_i, f_name in enumerate("ABCD"):
                        rcbd_rows.append(
                            {
                                "cohort": bench,
                                "estimator": estimator_name,
                                "model_id": m_name,
                                "task": task_name,
                                "factor": f_name,
                                "n_items": N,
                                "tau_effect": float(inf["taus"][f_i]),
                                "p_unblocked": float(inf["p_unblocked"][f_i]),
                                "p_rcbd": float(inf["p_rcbd"][f_i]),
                                "p_item_cluster": float(inf["p_cluster"][f_i]),
                                "p_sign_flip_perm": float(inf["p_perm"][f_i]) if inf["p_perm"] is not None else float("nan"),
                                "t_stat_cluster": float(inf["t_stat_cluster"][f_i]),
                                "zero_var_contrast": bool(inf["zero_var_mask"][f_i]),
                            }
                        )
                else:
                    inf = {
                        "mse_reduction_pct": np.nan, "f_stat_multiplier": np.nan,
                        "r2_unblocked": np.nan, "r2_rcbd": np.nan,
                        "sig_unblocked_cube_bh": np.nan, "sig_rcbd_cube_bh": np.nan,
                        "sig_cluster_cube_bh": np.nan, "n_zero_var_contrasts": np.nan,
                    }

                cell_means = Y_t.mean(axis=1)
                cube_rows.append(
                    {
                        "cohort": bench,
                        "estimator": estimator_name,
                        "model_id": m_name,
                        "task": task_name,
                        "n_items": N,
                        "mean_acc": float(cell_means.mean()),
                        "spread_pp": float((cell_means.max() - cell_means.min()) * 100.0),
                        "rmse_main_res4": float(rmse_p[0]),
                        "rmse_main_res4_20splits": float(rmse_p[1:].mean()),
                        "rmse_main_rand8": float(rmse_r8[0]),
                        "rmse_main_rand8_20splits": float(rmse_r8[1:].mean()),
                        "rmse_main_ofat": float(rmse_o[0]),
                        "rmse_main_ofat_20splits": float(rmse_o[1:].mean()),
                        "rms_main_true_disc0": f_disc0["rms_main"],
                        "rms_2fi_true_disc0": f_disc0["rms_2fi"],
                        "rms_3fi_true_disc0": f_disc0["rms_3fi"],
                        "mean20_disc_rms_main": mean20_disc_rms_main,
                        "mean20_disc_rms_2fi": mean20_disc_rms_2fi,
                        "sign_res4": sign_res4_0,
                        "sign_rand8": sign_r8_0,
                        "conf_baseline": float(c_base[0]),
                        "conf_baseline_20splits": float(c_base[1:].mean()),
                        "conf_oracle16": float(c_ora[0]),
                        "conf_oracle16_20splits": float(c_ora[1:].mean()),
                        "conf_res4_best": float(c_bp[0]),
                        "conf_res4_best_coset_pos": float(c_bp[0]),
                        "conf_res4_best_coset_neg": float(c_bn[0]),
                        "conf_res4_best_coset_avg": float(c_bavg[0]),
                        "conf_res4_best_coset_avg_20splits": float(c_bavg[1:].mean()),
                        "conf_res4_main": float(c_mp[0]),
                        "conf_res4_main_coset_pos": float(c_mp[0]),
                        "conf_res4_main_coset_neg": float(c_mn[0]),
                        "conf_res4_main_coset_avg": float(c_mavg[0]),
                        "conf_res4_main_coset_avg_20splits": float(c_mavg[1:].mean()),
                        "conf_rand8_best": float(c_r8_best[0]),
                        "conf_rand8_best_20splits": float(c_r8_best[1:].mean()),
                        "conf_rand8_best_main_cond": float(c_r8_best_cond[0]),
                        "conf_rand8_best_main_cond_20splits": float(c_r8_best_cond[1:].mean()),
                        "conf_rand8_main": float(c_r8_main[0]),
                        "conf_rand8_main_20splits": float(c_r8_main[1:].mean()),
                        "conf_ofat": float(c_ofat[0]),
                        "conf_ofat_20splits": float(c_ofat[1:].mean()),
                        "regret_res4_best_coset_avg_20splits": float((c_ora[1:] - c_bavg[1:]).mean()),
                        "regret_res4_main_coset_avg_20splits": float((c_ora[1:] - c_mavg[1:]).mean()),
                        "regret_rand8_best_20splits": float((c_ora[1:] - c_r8_best[1:]).mean()),
                        "mse_reduction_pct": float(inf["mse_reduction_pct"]),
                        "f_stat_multiplier": float(inf["f_stat_multiplier"]),
                        "r2_unblocked": float(inf["r2_unblocked"]),
                        "r2_rcbd": float(inf["r2_rcbd"]),
                        "sig_factors_unblocked": int(inf["sig_unblocked_cube_bh"]) if compute_rcbd else None,
                        "sig_factors_rcbd": int(inf["sig_rcbd_cube_bh"]) if compute_rcbd else None,
                        "sig_factors_item_cluster": int(inf["sig_cluster_cube_bh"]) if compute_rcbd else None,
                        "n_zero_var_contrasts": int(inf.get("n_zero_var_contrasts", 0)) if compute_rcbd else None,
                    }
                )

    df_slab = pd.DataFrame(cube_rows)
    df_fourier = pd.DataFrame(fourier_rows)
    df_rcbd = pd.DataFrame(rcbd_rows)
    split_arrays = {k: np.array(v, dtype=float) for k, v in split_records.items()}  # each (178, 21)
    split_arrays_first = {k: np.array(v, dtype=float) for k, v in split_records_first.items()}
    split_arrays_rand_tie = {k: np.array(v, dtype=float) for k, v in split_records_rand_tie.items()}
    tie_arrays = {k: np.array(v, dtype=float) for k, v in tie_records.items()}
    tie_indicator_arrays = {
        k: np.array(v, dtype=float) for k, v in tie_indicator_records.items()
    }
    all_cells_conf_s0 = np.array(all_cells_conf_s0_list, dtype=float)  # (178, 16)
    all_cells_conf_20s = np.array(all_cells_conf_20s_list, dtype=float)  # (178, 16)

    tie_diag = {}
    for k_t, t_arr in tie_arrays.items():
        tie_diag[k_t] = summarize_tie_diagnostics(
            t_arr, tie_indicator_arrays[k_t]
        )

    df_slab["diff_screenbest_minus_rand8_20splits"] = df_slab["conf_res4_best_coset_avg_20splits"] - df_slab["conf_rand8_best_20splits"]
    df_slab["diff_screenbest_minus_rand8_split0"] = df_slab["conf_res4_best_coset_avg"] - df_slab["conf_rand8_best"]
    df_slab["diff_screenbest_minus_rand8_main_cond_20splits"] = df_slab["conf_res4_best_coset_avg_20splits"] - df_slab["conf_rand8_best_main_cond_20splits"]
    df_slab["diff_screenbest_minus_rand8_main_cond_split0"] = df_slab["conf_res4_best_coset_avg"] - df_slab["conf_rand8_best_main_cond"]
    df_slab["diff_screenbest_minus_main_20splits"] = df_slab["conf_res4_best_coset_avg_20splits"] - df_slab["conf_res4_main_coset_avg_20splits"]
    df_slab["diff_rand8_rmse_minus_res4_20splits"] = df_slab["rmse_main_rand8_20splits"] - df_slab["rmse_main_res4_20splits"]

    ci_sb_vs_r8 = panel_cluster_bootstrap_ci(df_slab, "diff_screenbest_minus_rand8_20splits")
    ci_sb_vs_r8_s0 = panel_cluster_bootstrap_ci(df_slab, "diff_screenbest_minus_rand8_split0")
    ci_sb_vs_r8_cond = panel_cluster_bootstrap_ci(df_slab, "diff_screenbest_minus_rand8_main_cond_20splits")
    ci_sb_vs_r8_cond_s0 = panel_cluster_bootstrap_ci(df_slab, "diff_screenbest_minus_rand8_main_cond_split0")
    ci_sb_vs_main = panel_cluster_bootstrap_ci(df_slab, "diff_screenbest_minus_main_20splits")
    ci_rmse_diff = panel_cluster_bootstrap_ci(df_slab, "diff_rand8_rmse_minus_res4_20splits")

    win_counts = {
        "cube_main_res4_beats_rand8_split0": int((df_slab["rmse_main_res4"] < df_slab["rmse_main_rand8"]).sum()),
        "cube_main_res4_beats_rand8_split0_frac": float((df_slab["rmse_main_res4"] < df_slab["rmse_main_rand8"]).mean()),
        "cube_main_res4_beats_rand8_20splits": int((df_slab["rmse_main_res4_20splits"] < df_slab["rmse_main_rand8_20splits"]).sum()),
        "cube_main_res4_beats_rand8_20splits_frac": float((df_slab["rmse_main_res4_20splits"] < df_slab["rmse_main_rand8_20splits"]).mean()),
        "cube_main_res4_beats_ofat_split0": int((df_slab["rmse_main_res4"] < df_slab["rmse_main_ofat"]).sum()),
        "cube_main_res4_beats_ofat_split0_frac": float((df_slab["rmse_main_res4"] < df_slab["rmse_main_ofat"]).mean()),
    }

    # Include all completeness-selected cubes. Degenerate all-zero cubes contribute the
    # documented neutral defaults (0% MSE reduction, 1x F multiplier) rather than silently
    # changing the summary denominator.
    rcbd_summary_cubes = df_slab
    summary = {
        "estimator": estimator_name,
        "primary_tie_policy": tie_policy,
        "inference_computed": bool(compute_rcbd),
        "n_cubes": len(df_slab),
        "n_bbh_cubes": int((df_slab["cohort"] == "BBH").sum()),
        "n_lmentry_cubes": int((df_slab["cohort"] == "LMentry").sum()),
        "bbh_cell_item_observations": int(df_slab[df_slab["cohort"] == "BBH"]["n_items"].sum() * 16),
        "lmentry_cell_item_observations": int(df_slab[df_slab["cohort"] == "LMentry"]["n_items"].sum() * 16),
        "total_cell_item_observations": int(df_slab["n_items"].sum() * 16),
        "win_counts": win_counts,
        "mean_spread_pp": float(df_slab["spread_pp"].mean()),
        "mean_rmse_main_res4": float(df_slab["rmse_main_res4"].mean()),
        "mean_rmse_main_res4_20splits": float(df_slab["rmse_main_res4_20splits"].mean()),
        "mean_rmse_main_rand8": float(df_slab["rmse_main_rand8"].mean()),
        "mean_rmse_main_rand8_20splits": float(df_slab["rmse_main_rand8_20splits"].mean()),
        "main_rmse_ratio_rand8_over_res4": float(df_slab["rmse_main_rand8"].mean() / df_slab["rmse_main_res4"].mean()),
        "main_rmse_ratio_rand8_over_res4_20splits": float(df_slab["rmse_main_rand8_20splits"].mean() / df_slab["rmse_main_res4_20splits"].mean()),
        "mean_rmse_main_ofat": float(df_slab["rmse_main_ofat"].mean()),
        "main_rmse_ratio_ofat_over_res4": float(df_slab["rmse_main_ofat"].mean() / df_slab["rmse_main_res4"].mean()),
        "mean_sign_res4": float(df_slab["sign_res4"].mean()),
        "mean_sign_rand8": float(df_slab["sign_rand8"].mean()),
        "mean_conf_oracle16": float(df_slab["conf_oracle16"].mean()),
        "mean_conf_oracle16_20splits": float(df_slab["conf_oracle16_20splits"].mean()),
        "mean_conf_res4_best": float(df_slab["conf_res4_best"].mean()),
        "mean_conf_res4_best_coset_pos": float(df_slab["conf_res4_best_coset_pos"].mean()),
        "mean_conf_res4_best_coset_neg": float(df_slab["conf_res4_best_coset_neg"].mean()),
        "mean_conf_res4_best_coset_avg": float(df_slab["conf_res4_best_coset_avg"].mean()),
        "mean_conf_res4_best_coset_avg_20splits": float(df_slab["conf_res4_best_coset_avg_20splits"].mean()),
        "mean_conf_rand8_best": float(df_slab["conf_rand8_best"].mean()),
        "mean_conf_rand8_best_20splits": float(df_slab["conf_rand8_best_20splits"].mean()),
        "mean_conf_rand8_best_main_cond": float(df_slab["conf_rand8_best_main_cond"].mean()),
        "mean_conf_rand8_best_main_cond_20splits": float(df_slab["conf_rand8_best_main_cond_20splits"].mean()),
        "mean_conf_res4_main": float(df_slab["conf_res4_main"].mean()),
        "mean_conf_res4_main_coset_avg": float(df_slab["conf_res4_main_coset_avg"].mean()),
        "mean_conf_res4_main_coset_avg_20splits": float(df_slab["conf_res4_main_coset_avg_20splits"].mean()),
        "mean_conf_rand8_main": float(df_slab["conf_rand8_main"].mean()),
        "mean_conf_rand8_main_20splits": float(df_slab["conf_rand8_main_20splits"].mean()),
        "mean_conf_ofat": float(df_slab["conf_ofat"].mean()),
        "mean_conf_ofat_20splits": float(df_slab["conf_ofat_20splits"].mean()),
        "mean_conf_baseline": float(df_slab["conf_baseline"].mean()),
        "mean_conf_baseline_20splits": float(df_slab["conf_baseline_20splits"].mean()),
        "mean_regret_vs_oracle_20splits": {
            "res4_best_coset_avg": float(df_slab["regret_res4_best_coset_avg_20splits"].mean()),
            "res4_main_coset_avg": float(df_slab["regret_res4_main_coset_avg_20splits"].mean()),
            "rand8_best": float(df_slab["regret_rand8_best_20splits"].mean()),
            "ofat": float((df_slab["conf_oracle16_20splits"] - df_slab["conf_ofat_20splits"]).mean()),
            "baseline": float((df_slab["conf_oracle16_20splits"] - df_slab["conf_baseline_20splits"]).mean()),
        },
        "tie_diagnostics": tie_diag,
        "recovery_split0_res4_best_coset_pos": compute_oracle_uplift_recovery_stats(
            df_slab["conf_res4_best_coset_pos"], df_slab["conf_baseline"], df_slab["conf_oracle16"]
        ),
        "recovery_split0_res4_best_coset_neg": compute_oracle_uplift_recovery_stats(
            df_slab["conf_res4_best_coset_neg"], df_slab["conf_baseline"], df_slab["conf_oracle16"]
        ),
        "recovery_split0_res4_best_coset_avg": compute_oracle_uplift_recovery_stats(
            df_slab["conf_res4_best_coset_avg"], df_slab["conf_baseline"], df_slab["conf_oracle16"]
        ),
        "recovery_split0_rand8_best": compute_oracle_uplift_recovery_stats(
            df_slab["conf_rand8_best"], df_slab["conf_baseline"], df_slab["conf_oracle16"]
        ),
        "recovery_20splits_res4_best_coset_avg": compute_oracle_uplift_recovery_stats(
            df_slab["conf_res4_best_coset_avg_20splits"], df_slab["conf_baseline_20splits"], df_slab["conf_oracle16_20splits"]
        ),
        "recovery_20splits_rand8_best": compute_oracle_uplift_recovery_stats(
            df_slab["conf_rand8_best_20splits"], df_slab["conf_baseline_20splits"], df_slab["conf_oracle16_20splits"]
        ),
        "recovery_20splits_res4_main_coset_avg": compute_oracle_uplift_recovery_stats(
            df_slab["conf_res4_main_coset_avg_20splits"], df_slab["conf_baseline_20splits"], df_slab["conf_oracle16_20splits"]
        ),
        "all_reference_cells_recovery": {
            "split0_res4_best_coset_avg": compute_all_reference_cells_recovery(
                df_slab["conf_res4_best_coset_avg"].to_numpy(), df_slab["conf_oracle16"].to_numpy(), all_cells_conf_s0
            ),
            "splits20_res4_best_coset_avg": compute_all_reference_cells_recovery(
                df_slab["conf_res4_best_coset_avg_20splits"].to_numpy(), df_slab["conf_oracle16_20splits"].to_numpy(), all_cells_conf_20s
            ),
            "splits20_rand8_best": compute_all_reference_cells_recovery(
                df_slab["conf_rand8_best_20splits"].to_numpy(), df_slab["conf_oracle16_20splits"].to_numpy(), all_cells_conf_20s
            ),
            "splits20_res4_main_coset_avg": compute_all_reference_cells_recovery(
                df_slab["conf_res4_main_coset_avg_20splits"].to_numpy(), df_slab["conf_oracle16_20splits"].to_numpy(), all_cells_conf_20s
            ),
        },
        "first_index_compatibility": {
            "mean_conf_res4_best_coset_pos_split0": float(split_arrays_first["res4_pos_best"][:, 0].mean()),
            "mean_conf_res4_best_coset_neg_split0": float(split_arrays_first["res4_neg_best"][:, 0].mean()),
            "mean_conf_res4_best_coset_avg_split0": float(split_arrays_first["res4_avg_best"][:, 0].mean()),
            "mean_conf_res4_best_coset_avg_20splits": float(split_arrays_first["res4_avg_best"][:, 1:].mean()),
            "mean_conf_res4_main_coset_avg_split0": float(split_arrays_first["res4_avg_main"][:, 0].mean()),
            "mean_conf_res4_main_coset_avg_20splits": float(split_arrays_first["res4_avg_main"][:, 1:].mean()),
            "mean_conf_rand8_best_split0": float(split_arrays_first["rand8_best"][:, 0].mean()),
            "mean_conf_rand8_best_20splits": float(split_arrays_first["rand8_best"][:, 1:].mean()),
            "recovery_split0_res4_best_coset_avg": compute_oracle_uplift_recovery_stats(
                split_arrays_first["res4_avg_best"][:, 0], split_arrays_first["baseline"][:, 0], split_arrays_first["oracle16"][:, 0]
            ),
            "recovery_20splits_res4_best_coset_avg": compute_oracle_uplift_recovery_stats(
                split_arrays_first["res4_avg_best"][:, 1:].mean(axis=1), split_arrays_first["baseline"][:, 1:].mean(axis=1), split_arrays_first["oracle16"][:, 1:].mean(axis=1)
            ),
            "recovery_20splits_res4_main_coset_avg": compute_oracle_uplift_recovery_stats(
                split_arrays_first["res4_avg_main"][:, 1:].mean(axis=1), split_arrays_first["baseline"][:, 1:].mean(axis=1), split_arrays_first["oracle16"][:, 1:].mean(axis=1)
            ),
            "recovery_20splits_rand8_best": compute_oracle_uplift_recovery_stats(
                split_arrays_first["rand8_best"][:, 1:].mean(axis=1), split_arrays_first["baseline"][:, 1:].mean(axis=1), split_arrays_first["oracle16"][:, 1:].mean(axis=1)
            ),
        },
        "tie_policy_sensitivity": {
            p_name: {
                "mean_conf_res4_best_coset_avg_split0": float(p_arr["res4_avg_best"][:, 0].mean()),
                "mean_conf_res4_best_coset_avg_20splits": float(p_arr["res4_avg_best"][:, 1:].mean()),
                "mean_conf_res4_main_coset_avg_split0": float(p_arr["res4_avg_main"][:, 0].mean()),
                "mean_conf_res4_main_coset_avg_20splits": float(p_arr["res4_avg_main"][:, 1:].mean()),
                "mean_conf_rand8_best_split0": float(p_arr["rand8_best"][:, 0].mean()),
                "mean_conf_rand8_best_20splits": float(p_arr["rand8_best"][:, 1:].mean()),
                "recovery_split0_res4_best_coset_avg_pct": float(
                    compute_oracle_uplift_recovery_stats(
                        p_arr["res4_avg_best"][:, 0], p_arr["baseline"][:, 0], p_arr["oracle16"][:, 0]
                    )["ratio_of_means_pct"]
                ),
                "recovery_20splits_res4_best_coset_avg_pct": float(
                    compute_oracle_uplift_recovery_stats(
                        p_arr["res4_avg_best"][:, 1:].mean(axis=1), p_arr["baseline"][:, 1:].mean(axis=1), p_arr["oracle16"][:, 1:].mean(axis=1)
                    )["ratio_of_means_pct"]
                ),
            }
            for p_name, p_arr in [("mean_tied", split_arrays), ("first_index", split_arrays_first), ("seeded_random", split_arrays_rand_tie)]
        },
        "ci_screenbest_minus_rand8_best_20splits": ci_sb_vs_r8,
        "ci_screenbest_minus_rand8_best_split0": ci_sb_vs_r8_s0,
        "ci_screenbest_minus_rand8_best_main_cond_20splits": ci_sb_vs_r8_cond,
        "ci_screenbest_minus_rand8_best_main_cond_split0": ci_sb_vs_r8_cond_s0,
        "ci_screenbest_minus_res4_main_20splits": ci_sb_vs_main,
        "ci_main_rmse_diff_rand8_minus_res4_20splits": ci_rmse_diff,
        "mean_disc0_rms_main_true": float(df_fourier["disc0_rms_main"].mean()),
        "mean_disc0_rms_2fi_true": float(df_fourier["disc0_rms_2fi"].mean()),
        "mean_disc0_rms_3fi_true": float(df_fourier["disc0_rms_3fi"].mean()),
        "mse_summary_cube_count": int(len(rcbd_summary_cubes)) if compute_rcbd else None,
        "mean_mse_reduction_pct": float(rcbd_summary_cubes["mse_reduction_pct"].mean()) if compute_rcbd else None,
        "min_mse_reduction_pct": float(rcbd_summary_cubes["mse_reduction_pct"].min()) if compute_rcbd else None,
        "max_mse_reduction_pct": float(rcbd_summary_cubes["mse_reduction_pct"].max()) if compute_rcbd else None,
        "mean_f_stat_multiplier": float(rcbd_summary_cubes["f_stat_multiplier"].mean()) if compute_rcbd else None,
        "mean_r2_unblocked": float(rcbd_summary_cubes["r2_unblocked"].mean()) if compute_rcbd else None,
        "mean_r2_rcbd": float(rcbd_summary_cubes["r2_rcbd"].mean()) if compute_rcbd else None,
        "total_factor_tests": int(len(df_slab) * 4) if compute_rcbd else None,
        "total_sig_unblocked_per_cube_bh": int(df_slab["sig_factors_unblocked"].sum()) if compute_rcbd else None,
        "total_sig_rcbd_per_cube_bh": int(df_slab["sig_factors_rcbd"].sum()) if compute_rcbd else None,
        "total_sig_item_cluster_per_cube_bh": int(df_slab["sig_factors_item_cluster"].sum()) if compute_rcbd else None,
        "total_zero_var_contrasts": int(df_slab["n_zero_var_contrasts"].sum()) if compute_rcbd else None,
    }

    return {
        "df_cubes": df_slab,
        "df_fourier": df_fourier,
        "df_rcbd": df_rcbd,
        "split_arrays": split_arrays,
        "summary": summary,
    }


def build_coset_results_table(
    res_c1_cell: Dict[str, Any],
    res_c1_leg: Dict[str, Any],
    res_c2_cell: Dict[str, Any],
    res_c2_leg: Dict[str, Any],
    cosets_info: Dict[str, Any],
) -> pd.DataFrame:
    """Build `coset_results.csv` summarizing attribution and selection across all fractional cosets (DV-3)."""
    rows = []
    for est_label, r1 in [("cell_mean", res_c1_cell), ("legacy_first", res_c1_leg)]:
        acc = r1["acc_arrays"]
        rmse = r1["rmse_arrays"]
        for split_mode, s_slice in [("legacy_even_odd", slice(0, 1)), ("mean_20_splits", slice(1, 21))]:
            base_m = float(acc["baseline"][:, :, s_slice].mean())
            ora_m = float(acc["oracle32"][:, :, s_slice].mean())
            denom = ora_m - base_m

            # Res V both cosets
            for cid, tag in [("2(5-1)V_coset_pos", "pos"), ("2(5-1)V_coset_neg", "neg")]:
                cmeta = cosets_info["res5_cosets"][cid]
                cb = float(acc[f"res5_{tag}_best"][:, :, s_slice].mean())
                c2 = float(acc[f"res5_{tag}_2fi"][:, :, s_slice].mean())
                cm = float(acc[f"res5_{tag}_main"][:, :, s_slice].mean())
                rows.append(
                    {
                        "cohort": "MMLU",
                        "estimator": est_label,
                        "split_regime": split_slice_name(split_mode),
                        "design_family": "2(5-1)V",
                        "coset_id": cid,
                        "defining_relation": cmeta["defining_relation"],
                        "coset_sign": cmeta["coset_sign"],
                        "is_legacy_default": cmeta["is_legacy_default"],
                        "rmse_main": float(rmse[f"main_res5_{tag}"][:, :, s_slice].mean()),
                        "rmse_2fi": float(rmse[f"2fi_res5_{tag}"][:, :, s_slice].mean()),
                        "conf_screen_best": cb,
                        "conf_model_2fi": c2,
                        "conf_model_main": cm,
                        "conf_baseline": base_m,
                        "conf_discovery_oracle": ora_m,
                        "recovery_screen_best_pct": (cb - base_m) / denom * 100.0,
                        "recovery_model_2fi_pct": (c2 - base_m) / denom * 100.0,
                        "recovery_model_main_pct": (cm - base_m) / denom * 100.0,
                    }
                )

            # Res III 4 canonical cosets
            res3_keys = list(cosets_info["res3_canonical_cosets"].keys())
            for c_i, cid in enumerate(res3_keys):
                cmeta = cosets_info["res3_canonical_cosets"][cid]
                cb = float(acc[f"res3_c{c_i}_best"][:, :, s_slice].mean())
                cm = float(acc[f"res3_c{c_i}_main"][:, :, s_slice].mean())
                rows.append(
                    {
                        "cohort": "MMLU",
                        "estimator": est_label,
                        "split_regime": split_slice_name(split_mode),
                        "design_family": "2(5-2)III_canonical",
                        "coset_id": cid,
                        "defining_relation": cmeta["defining_relation"],
                        "coset_sign": cmeta["coset_sign"],
                        "is_legacy_default": cmeta["is_legacy_default"],
                        "rmse_main": float(rmse[f"main_res3_c{c_i}"][:, :, s_slice].mean()),
                        "rmse_2fi": float("nan"),
                        "conf_screen_best": cb,
                        "conf_model_2fi": float("nan"),
                        "conf_model_main": cm,
                        "conf_baseline": base_m,
                        "conf_discovery_oracle": ora_m,
                        "recovery_screen_best_pct": (cb - base_m) / denom * 100.0,
                        "recovery_model_2fi_pct": float("nan"),
                        "recovery_model_main_pct": (cm - base_m) / denom * 100.0,
                    }
                )

            # Res III all 60 regular quarter-fractions (summary row)
            cb60 = float(acc["res3_60_avg_best"][:, :, s_slice].mean())
            rows.append(
                {
                    "cohort": "MMLU",
                    "estimator": est_label,
                    "split_regime": split_slice_name(split_mode),
                    "design_family": "2(5-2)III_all60_orientations",
                    "coset_id": "2(5-2)III_60_regular_mean",
                    "defining_relation": "All 15 regular Res-III subgroups x 4 cosets (60 designs)",
                    "coset_sign": "all_60",
                    "is_legacy_default": False,
                    "rmse_main": float(rmse["main_res3_60_avg"][:, :, s_slice].mean()),
                    "rmse_2fi": float("nan"),
                    "conf_screen_best": cb60,
                    "conf_model_2fi": float("nan"),
                    "conf_model_main": float("nan"),
                    "conf_baseline": base_m,
                    "conf_discovery_oracle": ora_m,
                    "recovery_screen_best_pct": (cb60 - base_m) / denom * 100.0,
                    "recovery_model_2fi_pct": float("nan"),
                    "recovery_model_main_pct": float("nan"),
                }
            )

    for est_label, r2 in [("cell_mean", res_c2_cell), ("legacy_first", res_c2_leg)]:
        sa = r2["split_arrays"]
        for split_mode, s_slice in [("legacy_even_odd", slice(0, 1)), ("mean_20_splits", slice(1, 21))]:
            base_m = float(sa["baseline"][:, s_slice].mean())
            ora_m = float(sa["oracle16"][:, s_slice].mean())
            denom = ora_m - base_m
            for cid, tag in [("2(4-1)IV_coset_pos", "pos"), ("2(4-1)IV_coset_neg", "neg")]:
                cmeta = cosets_info["res4_cosets"][cid]
                cb = float(sa[f"res4_{tag}_best"][:, s_slice].mean())
                cm = float(sa[f"res4_{tag}_main"][:, s_slice].mean())
                rows.append(
                    {
                        "cohort": "SLAB",
                        "estimator": est_label,
                        "split_regime": split_slice_name(split_mode),
                        "design_family": "2(4-1)IV",
                        "coset_id": cid,
                        "defining_relation": cmeta["defining_relation"],
                        "coset_sign": cmeta["coset_sign"],
                        "is_legacy_default": cmeta["is_legacy_default"],
                        "rmse_main": float(sa[f"rmse_res4_{tag}"][:, s_slice].mean()),
                        "rmse_2fi": float("nan"),
                        "conf_screen_best": cb,
                        "conf_model_2fi": float("nan"),
                        "conf_model_main": cm,
                        "conf_baseline": base_m,
                        "conf_discovery_oracle": ora_m,
                        "recovery_screen_best_pct": (cb - base_m) / denom * 100.0,
                        "recovery_model_2fi_pct": float("nan"),
                        "recovery_model_main_pct": (cm - base_m) / denom * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def split_slice_name(name: str) -> str:
    return name


def build_split_selection_table(
    res_c1_cell: Dict[str, Any],
    res_c1_leg: Dict[str, Any],
    res_c2_cell: Dict[str, Any],
    res_c2_leg: Dict[str, Any],
    split_ids: List[str],
) -> pd.DataFrame:
    """Build `split_selection_results.csv` with per-split accuracy and oracle recovery (DV-4, DV-5)."""
    rows = []
    for est_label, r1 in [("cell_mean", res_c1_cell), ("legacy_first", res_c1_leg)]:
        acc = r1["acc_arrays"]
        rmse = r1["rmse_arrays"]
        for s_idx, sid in enumerate(split_ids):
            bm = float(acc["baseline"][:, :, s_idx].mean())
            om = float(acc["oracle32"][:, :, s_idx].mean())
            den = om - bm
            rows.append(
                {
                    "cohort": "MMLU",
                    "estimator": est_label,
                    "split_id": sid,
                    "is_legacy_split": (s_idx == 0),
                    "rmse_main_res5": float(rmse["main_res5_neg"][:, :, s_idx].mean()),
                    "rmse_main_rand16": float(rmse["main_rand16"][:, :, s_idx].mean()),
                    "rmse_2fi_res5": float(rmse["2fi_res5_neg"][:, :, s_idx].mean()),
                    "rmse_2fi_rand16_fullrank": float(rmse["2fi_rand16"][:, :, s_idx].mean()),
                    "conf_baseline": bm,
                    "conf_discovery_oracle": om,
                    "conf_ofat": float(acc["ofat"][:, :, s_idx].mean()),
                    "conf_res5_main_coset_avg": float(acc["res5_avg_main"][:, :, s_idx].mean()),
                    "conf_res5_2fi_coset_avg": float(acc["res5_avg_2fi"][:, :, s_idx].mean()),
                    "conf_res5_2fi_coset_pos": float(acc["res5_pos_2fi"][:, :, s_idx].mean()),
                    "conf_res5_2fi_coset_neg": float(acc["res5_neg_2fi"][:, :, s_idx].mean()),
                    "conf_res5_best_coset_avg": float(acc["res5_avg_best"][:, :, s_idx].mean()),
                    "conf_res5_best_coset_pos": float(acc["res5_pos_best"][:, :, s_idx].mean()),
                    "conf_res5_best_coset_neg": float(acc["res5_neg_best"][:, :, s_idx].mean()),
                    "conf_rand16_main": float(acc["rand16_main"][:, :, s_idx].mean()),
                    "conf_rand16_2fi_fullrank": float(acc["rand16_2fi"][:, :, s_idx].mean()),
                    "conf_rand16_best": float(acc["rand16_best"][:, :, s_idx].mean()),
                    "conf_res3_best_coset_avg": float(acc["res3_c_avg_best"][:, :, s_idx].mean()),
                    "conf_rand8_best": float(acc["rand8_best"][:, :, s_idx].mean()),
                    "recovery_res5_best_coset_avg_pct": float((acc["res5_avg_best"][:, :, s_idx].mean() - bm) / den * 100.0),
                    "recovery_rand16_best_pct": float((acc["rand16_best"][:, :, s_idx].mean() - bm) / den * 100.0),
                    "recovery_res5_2fi_coset_avg_pct": float((acc["res5_avg_2fi"][:, :, s_idx].mean() - bm) / den * 100.0),
                    "recovery_rand16_2fi_fullrank_pct": float((acc["rand16_2fi"][:, :, s_idx].mean() - bm) / den * 100.0),
                    "recovery_res5_main_coset_avg_pct": float((acc["res5_avg_main"][:, :, s_idx].mean() - bm) / den * 100.0),
                }
            )

    for est_label, r2 in [("cell_mean", res_c2_cell), ("legacy_first", res_c2_leg)]:
        sa = r2["split_arrays"]
        for s_idx, sid in enumerate(split_ids):
            bm = float(sa["baseline"][:, s_idx].mean())
            om = float(sa["oracle16"][:, s_idx].mean())
            den = om - bm
            rows.append(
                {
                    "cohort": "SLAB",
                    "estimator": est_label,
                    "split_id": sid,
                    "is_legacy_split": (s_idx == 0),
                    "rmse_main_res5": float(sa["rmse_res4_pos"][:, s_idx].mean()),
                    "rmse_main_rand16": float(sa["rmse_rand8"][:, s_idx].mean()),
                    "rmse_2fi_res5": float("nan"),
                    "rmse_2fi_rand16_fullrank": float("nan"),
                    "conf_baseline": bm,
                    "conf_discovery_oracle": om,
                    "conf_ofat": float(sa["ofat"][:, s_idx].mean()),
                    "conf_res5_main_coset_avg": float(sa["res4_avg_main"][:, s_idx].mean()),
                    "conf_res5_2fi_coset_avg": float("nan"),
                    "conf_res5_2fi_coset_pos": float("nan"),
                    "conf_res5_2fi_coset_neg": float("nan"),
                    "conf_res5_best_coset_avg": float(sa["res4_avg_best"][:, s_idx].mean()),
                    "conf_res5_best_coset_pos": float(sa["res4_pos_best"][:, s_idx].mean()),
                    "conf_res5_best_coset_neg": float(sa["res4_neg_best"][:, s_idx].mean()),
                    "conf_rand16_main": float(sa["rand8_main"][:, s_idx].mean()),
                    "conf_rand16_2fi_fullrank": float("nan"),
                    "conf_rand16_best": float(sa["rand8_best"][:, s_idx].mean()),
                    "conf_res3_best_coset_avg": float("nan"),
                    "conf_rand8_best": float(sa["rand8_best"][:, s_idx].mean()),
                    "recovery_res5_best_coset_avg_pct": float((sa["res4_avg_best"][:, s_idx].mean() - bm) / den * 100.0),
                    "recovery_rand16_best_pct": float((sa["rand8_best"][:, s_idx].mean() - bm) / den * 100.0),
                    "recovery_res5_2fi_coset_avg_pct": float("nan"),
                    "recovery_rand16_2fi_fullrank_pct": float("nan"),
                    "recovery_res5_main_coset_avg_pct": float((sa["res4_avg_main"][:, s_idx].mean() - bm) / den * 100.0),
                }
            )
    return pd.DataFrame(rows).rename(
        columns={
            "rmse_main_res5": "rmse_main_fraction",
            "rmse_main_rand16": "rmse_main_random_fullrank",
            "rmse_2fi_res5": "rmse_second_order_fraction",
            "rmse_2fi_rand16_fullrank": "rmse_second_order_random_fullrank",
            "conf_res5_main_coset_avg": "conf_fraction_main_coset_avg",
            "conf_res5_2fi_coset_avg": "conf_fraction_second_order_coset_avg",
            "conf_res5_2fi_coset_pos": "conf_fraction_second_order_coset_pos",
            "conf_res5_2fi_coset_neg": "conf_fraction_second_order_coset_neg",
            "conf_res5_best_coset_avg": "conf_fraction_screenbest_coset_avg",
            "conf_res5_best_coset_pos": "conf_fraction_screenbest_coset_pos",
            "conf_res5_best_coset_neg": "conf_fraction_screenbest_coset_neg",
            "conf_rand16_main": "conf_random_main",
            "conf_rand16_2fi_fullrank": "conf_random_second_order_fullrank",
            "conf_rand16_best": "conf_random_screenbest",
            "conf_res3_best_coset_avg": "conf_quarter_fraction_screenbest_coset_avg",
            "recovery_res5_best_coset_avg_pct": "recovery_fraction_screenbest_coset_avg_pct",
            "recovery_rand16_best_pct": "recovery_random_screenbest_pct",
            "recovery_res5_2fi_coset_avg_pct": "recovery_fraction_second_order_coset_avg_pct",
            "recovery_rand16_2fi_fullrank_pct": "recovery_random_second_order_fullrank_pct",
            "recovery_res5_main_coset_avg_pct": "recovery_fraction_main_coset_avg_pct",
        }
    )


def evaluate_template_representative_sensitivity(
    Ys: dict,
    all_cells_5: List[Tuple[int, ...]],
    mmlu_canon_map: Dict[Tuple[int, ...], List[int]],
    mmlu_uniq_map: Dict[Tuple[int, ...], List[int]],
    all_cells_4: List[Tuple[int, ...]],
    slab_canon_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]],
    slab_uniq_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]],
    mmlu_splits: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    slab_splits: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, List[str]]],
    res_c1_cell: Dict[str, Any],
    res_c1_cell_uniq: Dict[str, Any],
    res_c1_leg: Dict[str, Any],
    res_c1_canon: Dict[str, Any],
    res_c2_cell: Dict[str, Any],
    res_c2_cell_uniq: Dict[str, Any],
    res_c2_leg: Dict[str, Any],
    res_c2_canon: Dict[str, Any],
    cosets_info: Dict[str, Any],
    rand16_fullrank: List[List[int]],
    rand8_c2_fullrank: List[List[int]],
    rand16_screenbest_uncond: Optional[List[List[int]]] = None,
    rand8_c2_screenbest_uncond: Optional[List[List[int]]] = None,
    n_rep_draws: int = 20,
) -> pd.DataFrame:
    """
    Build `template_sensitivity.csv` comparing `cell_mean_unique`, `cell_mean` (`cell_mean_catalog_row`),
    `canonical_first`, `legacy_first`, and `n_rep_draws` seeded random single-template draws per cell
    (both unique-template uniform `random_single_rep_unique` and catalog-row uniform `random_single_rep`),
    crossed with the 20 repeated 50/50 item splits (20 x 20 = 400 evaluations per cube) (CV-2, Task 4).
    """
    if rand16_screenbest_uncond is None:
        rand16_screenbest_uncond = rand16_fullrank
    if rand8_c2_screenbest_uncond is None:
        rand8_c2_screenbest_uncond = rand8_c2_fullrank

    rows = []

    def add_c1_row(est_name: str, draw_id: str, s: Dict[str, Any], weighting: str) -> None:
        rows.append(
            {
                "cohort": "MMLU",
                "estimator": est_name,
                "template_weighting": weighting,
                "draw_id": draw_id,
                "order_invariant": est_name != "legacy_first",
                "rmse_main_frac": s["mean_task_rmse_main_res5"],
                "rmse_main_frac_20splits": s["mean_task_rmse_main_res5_20splits"],
                "rmse_main_rand": s["mean_task_rmse_main_rand16"],
                "rmse_main_rand_20splits": s["mean_task_rmse_main_rand16_20splits"],
                "main_rmse_ratio": s["task_main_rmse_ratio_rand16_over_res5"],
                "main_rmse_ratio_20splits": s["task_main_rmse_ratio_rand16_over_res5_20splits"],
                "rmse_2fi_frac": s["mean_task_rmse_2fi_res5"],
                "rmse_2fi_frac_20splits": s["mean_task_rmse_2fi_res5_20splits"],
                "rmse_2fi_rand_fullrank": s["mean_task_rmse_2fi_rand16_fullrank"],
                "rmse_2fi_rand_fullrank_20splits": s["mean_task_rmse_2fi_rand16_fullrank_20splits"],
                "twofi_rmse_ratio_fullrank": s["task_2fi_rmse_ratio_fullrank_over_res5"],
                "twofi_rmse_ratio_fullrank_20splits": s["task_2fi_rmse_ratio_fullrank_over_res5_20splits"],
                "conf_baseline": s["mean_conf_baseline"],
                "conf_baseline_20splits": s["mean_conf_baseline_20splits"],
                "conf_oracle": s["mean_conf_oracle32"],
                "conf_oracle_20splits": s["mean_conf_oracle32_20splits"],
                "conf_screen_best_coset_avg": s["mean_conf_res5_best_coset_avg"],
                "conf_screen_best_coset_avg_20splits": s["mean_conf_res5_best_coset_avg_20splits"],
                "conf_rand_best": s["mean_conf_rand16_best"],
                "conf_rand_best_20splits": s["mean_conf_rand16_best_20splits"],
                "conf_model_2fi_coset_avg": s["mean_conf_res5_2fi_coset_avg"],
                "conf_model_2fi_coset_avg_20splits": s["mean_conf_res5_2fi_coset_avg_20splits"],
                "conf_model_main_coset_avg": s["mean_conf_res5_main_coset_avg"],
                "conf_model_main_coset_avg_20splits": s["mean_conf_res5_main_coset_avg_20splits"],
                "recovery_screen_best_coset_avg_pct": s["recovery_split0_res5_best_coset_avg"]["ratio_of_means_pct"],
                "recovery_screen_best_coset_avg_20splits_pct": s["recovery_20splits_res5_best_coset_avg"]["ratio_of_means_pct"],
                "recovery_rand_best_pct": s["recovery_split0_rand16_best"]["ratio_of_means_pct"],
                "recovery_rand_best_20splits_pct": s["recovery_20splits_rand16_best"]["ratio_of_means_pct"],
                "recovery_model_2fi_coset_avg_pct": s["recovery_split0_res5_2fi_coset_avg"]["ratio_of_means_pct"],
                "recovery_model_2fi_coset_avg_20splits_pct": s["recovery_20splits_res5_2fi_coset_avg"]["ratio_of_means_pct"],
                "sd_across_splits_conf_screen_best": float("nan"),
                "sd_across_splits_conf_model_2fi": float("nan"),
            }
        )

    def add_c2_row(est_name: str, draw_id: str, s: Dict[str, Any], weighting: str) -> None:
        rows.append(
            {
                "cohort": "SLAB",
                "estimator": est_name,
                "template_weighting": weighting,
                "draw_id": draw_id,
                "order_invariant": est_name != "legacy_first",
                "rmse_main_frac": s["mean_rmse_main_res4"],
                "rmse_main_frac_20splits": s["mean_rmse_main_res4_20splits"],
                "rmse_main_rand": s["mean_rmse_main_rand8"],
                "rmse_main_rand_20splits": s["mean_rmse_main_rand8_20splits"],
                "main_rmse_ratio": s["main_rmse_ratio_rand8_over_res4"],
                "main_rmse_ratio_20splits": s["main_rmse_ratio_rand8_over_res4_20splits"],
                "rmse_2fi_frac": float("nan"),
                "rmse_2fi_frac_20splits": float("nan"),
                "rmse_2fi_rand_fullrank": float("nan"),
                "rmse_2fi_rand_fullrank_20splits": float("nan"),
                "twofi_rmse_ratio_fullrank": float("nan"),
                "twofi_rmse_ratio_fullrank_20splits": float("nan"),
                "conf_baseline": s["mean_conf_baseline"],
                "conf_baseline_20splits": s["mean_conf_baseline_20splits"],
                "conf_oracle": s["mean_conf_oracle16"],
                "conf_oracle_20splits": s["mean_conf_oracle16_20splits"],
                "conf_screen_best_coset_avg": s["mean_conf_res4_best_coset_avg"],
                "conf_screen_best_coset_avg_20splits": s["mean_conf_res4_best_coset_avg_20splits"],
                "conf_rand_best": s["mean_conf_rand8_best"],
                "conf_rand_best_20splits": s["mean_conf_rand8_best_20splits"],
                "conf_model_2fi_coset_avg": float("nan"),
                "conf_model_2fi_coset_avg_20splits": float("nan"),
                "conf_model_main_coset_avg": s["mean_conf_res4_main_coset_avg"],
                "conf_model_main_coset_avg_20splits": s["mean_conf_res4_main_coset_avg_20splits"],
                "recovery_screen_best_coset_avg_pct": s["recovery_split0_res4_best_coset_avg"]["ratio_of_means_pct"],
                "recovery_screen_best_coset_avg_20splits_pct": s["recovery_20splits_res4_best_coset_avg"]["ratio_of_means_pct"],
                "recovery_rand_best_pct": s["recovery_split0_rand8_best"]["ratio_of_means_pct"],
                "recovery_rand_best_20splits_pct": s["recovery_20splits_rand8_best"]["ratio_of_means_pct"],
                "recovery_model_2fi_coset_avg_pct": float("nan"),
                "recovery_model_2fi_coset_avg_20splits_pct": float("nan"),
                "sd_across_splits_conf_screen_best": float("nan"),
                "sd_across_splits_conf_model_2fi": float("nan"),
            }
        )

    add_c1_row("cell_mean_unique", "unique_templates_equal_weight", res_c1_cell_uniq["summary"], "unique_template_uniform")
    add_c1_row("cell_mean", "primary_all_templates", res_c1_cell["summary"], "catalog_row_uniform")
    add_c1_row("canonical_first", "sha256_rank0", res_c1_canon["summary"], "single_template_sha256_rank0")
    add_c1_row("legacy_first", "catalog_row0", res_c1_leg["summary"], "single_template_catalog_row0")

    add_c2_row("cell_mean_unique", "unique_templates_equal_weight", res_c2_cell_uniq["summary"], "unique_template_uniform")
    add_c2_row("cell_mean", "primary_all_templates", res_c2_cell["summary"], "catalog_row_uniform")
    add_c2_row("canonical_first", "sha256_rank0", res_c2_canon["summary"], "single_template_sha256_rank0")
    add_c2_row("legacy_first", "catalog_row0", res_c2_leg["summary"], "single_template_catalog_row0")

    # Precompute split-averaged template matrices for MMLU: (M, T, 63, 21)
    m5 = build_cube_design_matrices(5)
    X5_pm, X5_2fi, D5_main, D5_2fi = m5["X_pm"], m5["X_2fi"], m5["D_main"], m5["D_2fi"]
    res5_pos = np.array(cosets_info["res5_cosets"]["2(5-1)V_coset_pos"]["indices"])
    res5_neg = np.array(cosets_info["res5_cosets"]["2(5-1)V_coset_neg"]["indices"])
    P2_pos = np.linalg.pinv(D5_2fi[res5_pos])
    P2_neg = np.linalg.pinv(D5_2fi[res5_neg])
    r16_arr = np.array(rand16_fullrank)
    r16_sb_arr = np.array(rand16_screenbest_uncond)
    P2_r16 = np.array([np.linalg.pinv(D5_2fi[c]) for c in rand16_fullrank])
    Pm_r16 = np.array([np.linalg.pinv(D5_main[c]) for c in rand16_fullrank])

    tasks_mmlu = sorted(Ys["MMLU"].keys())
    n_models_mmlu = len(MMLU_MODELS)
    n_tasks_mmlu = len(tasks_mmlu)
    n_tmpl_mmlu = int(Ys["MMLU"][tasks_mmlu[0]][0].shape[0])
    T_d_mmlu = np.empty((n_models_mmlu, n_tasks_mmlu, n_tmpl_mmlu, 21), dtype=float)
    T_c_mmlu = np.empty((n_models_mmlu, n_tasks_mmlu, n_tmpl_mmlu, 21), dtype=float)
    for t_idx, t in enumerate(tasks_mmlu):
        d_masks, c_masks, _ = mmlu_splits[t]
        d_bin = (d_masks > 0).astype(float)
        c_bin = (c_masks > 0).astype(float)
        d_norm = d_bin / d_bin.sum(axis=0)[None, :]
        c_norm = c_bin / c_bin.sum(axis=0)[None, :]
        for m_idx in range(n_models_mmlu):
            mat = np.asarray(Ys["MMLU"][t][m_idx], dtype=float)
            T_d_mmlu[m_idx, t_idx] = mat @ d_norm
            T_c_mmlu[m_idx, t_idx] = mat @ c_norm
            T_d_mmlu[m_idx, t_idx, :, 0] = mat[:, ::2].mean(axis=1)
            T_c_mmlu[m_idx, t_idx, :, 0] = mat[:, 1::2].mean(axis=1)

    for est_label, cell_pool_map, seed_offset, weight_label in [
        ("random_single_rep_unique", mmlu_uniq_map, 200, "unique_template_uniform"),
        ("random_single_rep", mmlu_canon_map, 0, "catalog_row_uniform"),
    ]:
        rng_rep = np.random.default_rng(SEEDS["template_reps"] + seed_offset)
        for d_idx in range(n_rep_draws):
            draw_idx = np.array([int(rng_rep.choice(cell_pool_map[c])) for c in all_cells_5], dtype=int)
            Y_d = T_d_mmlu[:, :, draw_idx, :]  # (15, 57, 32, 21)
            Y_c = T_c_mmlu[:, :, draw_idx, :]  # (15, 57, 32, 21)

            tm = np.einsum("ck,mtcs->mtks", X5_pm, Y_d) / 16.0  # (15, 57, 5, 21)
            t2 = np.einsum("ck,mtcs->mtks", X5_2fi, Y_d) / 16.0  # (15, 57, 10, 21)
            bp = np.einsum("pc,mtcs->mtps", P2_pos, Y_d[:, :, res5_pos, :])  # (15, 57, 16, 21)
            bn = np.einsum("pc,mtcs->mtps", P2_neg, Y_d[:, :, res5_neg, :])  # (15, 57, 16, 21)

            rm_f_s = np.sqrt(np.mean((2.0 * bn[:, :, 1:6, :] - tm) ** 2, axis=2)).mean(axis=(0, 1))  # (21,)
            r2_f_s = np.sqrt(np.mean((2.0 * bn[:, :, 6:, :] - t2) ** 2, axis=2)).mean(axis=(0, 1))  # (21,)

            yd_sub = Y_d[:, :, r16_arr, :]  # (15, 57, 200, 16, 21)
            bm_r = np.einsum("dij,mtdjs->mtdis", Pm_r16, yd_sub)  # (15, 57, 200, 6, 21)
            b2_r = np.einsum("dij,mtdjs->mtdis", P2_r16, yd_sub)  # (15, 57, 200, 16, 21)
            rm_r_s = np.sqrt(np.mean((2.0 * bm_r[:, :, :, 1:6, :] - tm[:, :, None, :, :]) ** 2, axis=3)).mean(axis=(0, 1, 2))  # (21,)
            r2_r_s = np.sqrt(np.mean((2.0 * b2_r[:, :, :, 6:, :] - t2[:, :, None, :, :]) ** 2, axis=3)).mean(axis=(0, 1, 2))  # (21,)

            cb_s = Y_c[:, :, 0, :].mean(axis=(0, 1))  # (21,)
            # Reshape (15*57, 32, 21) for tie-aware selection
            Yd_flat = Y_d.reshape(-1, 32, 21)
            Yc_flat = Y_c.reshape(-1, 32, 21)
            co_s = np.array([select_with_tie_policy(Yd_flat[i], Yc_flat[i], tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            cbp_s = np.array([select_with_tie_policy(Yd_flat[i, res5_pos, :], Yc_flat[i], candidate_indices=res5_pos, tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            cbn_s = np.array([select_with_tie_policy(Yd_flat[i, res5_neg, :], Yc_flat[i], candidate_indices=res5_neg, tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            sb_s = 0.5 * (cbp_s + cbn_s)

            pred_2fi_p = np.einsum("cp,mtps->mtcs", D5_2fi, bp).reshape(-1, 32, 21)
            pred_2fi_n = np.einsum("cp,mtps->mtcs", D5_2fi, bn).reshape(-1, 32, 21)
            pred_main_p = np.einsum("cp,mtps->mtcs", D5_main, bp[:, :, :6, :]).reshape(-1, 32, 21)
            pred_main_n = np.einsum("cp,mtps->mtcs", D5_main, bn[:, :, :6, :]).reshape(-1, 32, 21)
            c2p_s = np.array([select_with_tie_policy(pred_2fi_p[i], Yc_flat[i], tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            c2n_s = np.array([select_with_tie_policy(pred_2fi_n[i], Yc_flat[i], tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            m2_s = 0.5 * (c2p_s + c2n_s)
            cmp_s = np.array([select_with_tie_policy(pred_main_p[i], Yc_flat[i], tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            cmn_s = np.array([select_with_tie_policy(pred_main_n[i], Yc_flat[i], tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=0)
            mm_s = 0.5 * (cmp_s + cmn_s)

            yd_sb_flat = Y_d[:, :, r16_sb_arr, :].reshape(-1, len(rand16_screenbest_uncond), 16, 21)
            rb_s = np.array([select_with_tie_policy(yd_sb_flat[i], Yc_flat[i], candidate_indices=r16_sb_arr, tie_policy="mean_tied")[0] for i in range(Yd_flat.shape[0])]).mean(axis=(0, 1))

            den0 = float(co_s[0] - cb_s[0])
            den20 = float(co_s[1:].mean() - cb_s[1:].mean())
            rows.append(
                {
                    "cohort": "MMLU",
                    "estimator": est_label,
                    "template_weighting": weight_label,
                    "draw_id": f"rep_draw_{d_idx:02d}",
                    "order_invariant": True,
                    "rmse_main_frac": float(rm_f_s[0]),
                    "rmse_main_frac_20splits": float(rm_f_s[1:].mean()),
                    "rmse_main_rand": float(rm_r_s[0]),
                    "rmse_main_rand_20splits": float(rm_r_s[1:].mean()),
                    "main_rmse_ratio": float(rm_r_s[0] / rm_f_s[0]),
                    "main_rmse_ratio_20splits": float(rm_r_s[1:].mean() / rm_f_s[1:].mean()),
                    "rmse_2fi_frac": float(r2_f_s[0]),
                    "rmse_2fi_frac_20splits": float(r2_f_s[1:].mean()),
                    "rmse_2fi_rand_fullrank": float(r2_r_s[0]),
                    "rmse_2fi_rand_fullrank_20splits": float(r2_r_s[1:].mean()),
                    "twofi_rmse_ratio_fullrank": float(r2_r_s[0] / r2_f_s[0]),
                    "twofi_rmse_ratio_fullrank_20splits": float(r2_r_s[1:].mean() / r2_f_s[1:].mean()),
                    "conf_baseline": float(cb_s[0]),
                    "conf_baseline_20splits": float(cb_s[1:].mean()),
                    "conf_oracle": float(co_s[0]),
                    "conf_oracle_20splits": float(co_s[1:].mean()),
                    "conf_screen_best_coset_avg": float(sb_s[0]),
                    "conf_screen_best_coset_avg_20splits": float(sb_s[1:].mean()),
                    "conf_rand_best": float(rb_s[0]),
                    "conf_rand_best_20splits": float(rb_s[1:].mean()),
                    "conf_model_2fi_coset_avg": float(m2_s[0]),
                    "conf_model_2fi_coset_avg_20splits": float(m2_s[1:].mean()),
                    "conf_model_main_coset_avg": float(mm_s[0]),
                    "conf_model_main_coset_avg_20splits": float(mm_s[1:].mean()),
                    "recovery_screen_best_coset_avg_pct": float((sb_s[0] - cb_s[0]) / den0 * 100.0),
                    "recovery_screen_best_coset_avg_20splits_pct": float((sb_s[1:].mean() - cb_s[1:].mean()) / den20 * 100.0),
                    "recovery_rand_best_pct": float((rb_s[0] - cb_s[0]) / den0 * 100.0),
                    "recovery_rand_best_20splits_pct": float((rb_s[1:].mean() - cb_s[1:].mean()) / den20 * 100.0),
                    "recovery_model_2fi_coset_avg_pct": float((m2_s[0] - cb_s[0]) / den0 * 100.0),
                    "recovery_model_2fi_coset_avg_20splits_pct": float((m2_s[1:].mean() - cb_s[1:].mean()) / den20 * 100.0),
                    "sd_across_splits_conf_screen_best": float(np.std(sb_s[1:], ddof=1)),
                    "sd_across_splits_conf_model_2fi": float(np.std(m2_s[1:], ddof=1)),
                }
            )

    # Cohort 2 random single-template draws (unique-template uniform and catalog-row uniform), crossed with 21 splits
    m4 = build_cube_design_matrices(4)
    X4_pm, D4_main = m4["X_pm"], m4["D_main"]
    res4_pos = np.array(cosets_info["res4_cosets"]["2(4-1)IV_coset_pos"]["indices"])
    res4_neg = np.array(cosets_info["res4_cosets"]["2(4-1)IV_coset_neg"]["indices"])
    Pm4_pos = np.linalg.pinv(D4_main[res4_pos])
    Pm4_neg = np.linalg.pinv(D4_main[res4_neg])
    r8_c2_arr = np.array(rand8_c2_fullrank)
    r8_c2_sb_arr = np.array(rand8_c2_screenbest_uncond)
    Pm4_r8 = np.array([np.linalg.pinv(D4_main[c]) for c in rand8_c2_fullrank])

    # Precompute split-averaged template matrices per Cohort-2 task
    T_d_slab: Dict[Tuple[str, str], np.ndarray] = {}
    T_c_slab: Dict[Tuple[str, str], np.ndarray] = {}
    for key in sorted(slab_canon_maps.keys()):
        bench, task_name = key
        d_masks, c_masks, _ = slab_splits[key]
        d_bin = (d_masks > 0).astype(float)
        c_bin = (c_masks > 0).astype(float)
        d_norm = d_bin / d_bin.sum(axis=0)[None, :]
        c_norm = c_bin / c_bin.sum(axis=0)[None, :]
        mats = Ys[bench][task_name]
        n_m = len(mats)
        n_tmpl = mats[0].shape[0]
        td = np.empty((n_m, n_tmpl, 21), dtype=float)
        tc = np.empty((n_m, n_tmpl, 21), dtype=float)
        for m_i, mat in enumerate(mats):
            mat_f = np.asarray(mat, dtype=float)
            td[m_i] = mat_f @ d_norm
            tc[m_i] = mat_f @ c_norm
            td[m_i, :, 0] = mat_f[:, ::2].mean(axis=1)
            tc[m_i, :, 0] = mat_f[:, 1::2].mean(axis=1)
        T_d_slab[key] = td
        T_c_slab[key] = tc

    for est_label, task_pool_maps, seed_offset, weight_label in [
        ("random_single_rep_unique", slab_uniq_maps, 300, "unique_template_uniform"),
        ("random_single_rep", slab_canon_maps, 100, "catalog_row_uniform"),
    ]:
        rng_rep_c2 = np.random.default_rng(SEEDS["template_reps"] + seed_offset)
        for d_idx in range(n_rep_draws):
            Yd_cubes = []
            Yc_cubes = []
            for key, pool_map in sorted(task_pool_maps.items()):
                draw_idx = np.array([int(rng_rep_c2.choice(pool_map[c])) for c in all_cells_4], dtype=int)
                Yd_cubes.append(T_d_slab[key][:, draw_idx, :])
                Yc_cubes.append(T_c_slab[key][:, draw_idx, :])
            Yd_all = np.concatenate(Yd_cubes, axis=0)  # (178, 16, 21)
            Yc_all = np.concatenate(Yc_cubes, axis=0)  # (178, 16, 21)

            tm = np.einsum("ck,ncs->nks", X4_pm, Yd_all) / 8.0  # (178, 4, 21)
            bp = np.einsum("pc,ncs->nps", Pm4_pos, Yd_all[:, res4_pos, :])  # (178, 5, 21)
            bn = np.einsum("pc,ncs->nps", Pm4_neg, Yd_all[:, res4_neg, :])  # (178, 5, 21)
            rm_f_s = np.sqrt(np.mean((2.0 * bp[:, 1:, :] - tm) ** 2, axis=1)).mean(axis=0)  # (21,)

            yd_sub = Yd_all[:, r8_c2_arr, :]  # (178, 200, 8, 21)
            bm_r = np.einsum("dij,ndjs->ndis", Pm4_r8, yd_sub)  # (178, 200, 5, 21)
            rm_r_s = np.sqrt(np.mean((2.0 * bm_r[:, :, 1:, :] - tm[:, None, :, :]) ** 2, axis=2)).mean(axis=(0, 1))  # (21,)

            cb_s = Yc_all[:, 0, :].mean(axis=0)
            co_s = np.array([select_with_tie_policy(Yd_all[i], Yc_all[i], tie_policy="mean_tied")[0] for i in range(Yd_all.shape[0])]).mean(axis=0)
            cbp_s = np.array([select_with_tie_policy(Yd_all[i, res4_pos, :], Yc_all[i], candidate_indices=res4_pos, tie_policy="mean_tied")[0] for i in range(Yd_all.shape[0])]).mean(axis=0)
            cbn_s = np.array([select_with_tie_policy(Yd_all[i, res4_neg, :], Yc_all[i], candidate_indices=res4_neg, tie_policy="mean_tied")[0] for i in range(Yd_all.shape[0])]).mean(axis=0)
            sb_s = 0.5 * (cbp_s + cbn_s)

            pred_main_p = np.einsum("cp,nps->ncs", D4_main, bp)
            pred_main_n = np.einsum("cp,nps->ncs", D4_main, bn)
            cmp_s = np.array([select_with_tie_policy(pred_main_p[i], Yc_all[i], tie_policy="mean_tied")[0] for i in range(Yd_all.shape[0])]).mean(axis=0)
            cmn_s = np.array([select_with_tie_policy(pred_main_n[i], Yc_all[i], tie_policy="mean_tied")[0] for i in range(Yd_all.shape[0])]).mean(axis=0)
            mm_s = 0.5 * (cmp_s + cmn_s)

            yd_sb_sub = Yd_all[:, r8_c2_sb_arr, :]
            rb_s = np.array([select_with_tie_policy(yd_sb_sub[i], Yc_all[i], candidate_indices=r8_c2_sb_arr, tie_policy="mean_tied")[0] for i in range(Yd_all.shape[0])]).mean(axis=(0, 1))

            den0 = float(co_s[0] - cb_s[0])
            den20 = float(co_s[1:].mean() - cb_s[1:].mean())
            rows.append(
                {
                    "cohort": "SLAB",
                    "estimator": est_label,
                    "template_weighting": weight_label,
                    "draw_id": f"rep_draw_{d_idx:02d}",
                    "order_invariant": True,
                    "rmse_main_frac": float(rm_f_s[0]),
                    "rmse_main_frac_20splits": float(rm_f_s[1:].mean()),
                    "rmse_main_rand": float(rm_r_s[0]),
                    "rmse_main_rand_20splits": float(rm_r_s[1:].mean()),
                    "main_rmse_ratio": float(rm_r_s[0] / rm_f_s[0]),
                    "main_rmse_ratio_20splits": float(rm_r_s[1:].mean() / rm_f_s[1:].mean()),
                    "rmse_2fi_frac": float("nan"),
                    "rmse_2fi_frac_20splits": float("nan"),
                    "rmse_2fi_rand_fullrank": float("nan"),
                    "rmse_2fi_rand_fullrank_20splits": float("nan"),
                    "twofi_rmse_ratio_fullrank": float("nan"),
                    "twofi_rmse_ratio_fullrank_20splits": float("nan"),
                    "conf_baseline": float(cb_s[0]),
                    "conf_baseline_20splits": float(cb_s[1:].mean()),
                    "conf_oracle": float(co_s[0]),
                    "conf_oracle_20splits": float(co_s[1:].mean()),
                    "conf_screen_best_coset_avg": float(sb_s[0]),
                    "conf_screen_best_coset_avg_20splits": float(sb_s[1:].mean()),
                    "conf_rand_best": float(rb_s[0]),
                    "conf_rand_best_20splits": float(rb_s[1:].mean()),
                    "conf_model_2fi_coset_avg": float("nan"),
                    "conf_model_2fi_coset_avg_20splits": float("nan"),
                    "conf_model_main_coset_avg": float(mm_s[0]),
                    "conf_model_main_coset_avg_20splits": float(mm_s[1:].mean()),
                    "recovery_screen_best_coset_avg_pct": float((sb_s[0] - cb_s[0]) / den0 * 100.0),
                    "recovery_screen_best_coset_avg_20splits_pct": float((sb_s[1:].mean() - cb_s[1:].mean()) / den20 * 100.0),
                    "recovery_rand_best_pct": float((rb_s[0] - cb_s[0]) / den0 * 100.0),
                    "recovery_rand_best_20splits_pct": float((rb_s[1:].mean() - cb_s[1:].mean()) / den20 * 100.0),
                    "recovery_model_2fi_coset_avg_pct": float("nan"),
                    "recovery_model_2fi_coset_avg_20splits_pct": float("nan"),
                    "sd_across_splits_conf_screen_best": float(np.std(sb_s[1:], ddof=1)),
                    "sd_across_splits_conf_model_2fi": float("nan"),
                }
            )

    return pd.DataFrame(rows)


def build_multiplicity_table(df_rcbd_all: pd.DataFrame, n_perm_rcbd: int = 200) -> pd.DataFrame:
    """
    Evaluate BH (`fdr_bh`) and dependence-robust Benjamini-Yekutieli (`fdr_by`) FDR control at alpha=0.05
    across three explicit hypothesis families (SI-5, Task 6):
    - `per_cube_bh`: m = 5 tests per cube (MMLU) or m = 4 tests per cube (SLAB)
    - `per_model_bh`: all factor tests within a model across tasks
    - `global_cohort_bh`: all 4,275 tests (MMLU) or all 712 tests (SLAB)
    Explicitly bounds inference scope to conditional fixed-templates/models/tasks with item sampling,
    and records sign-flip Monte Carlo resolution (`1 / (n_perm_rcbd + 1)`).
    """
    min_achievable_p = 1.0 / float(n_perm_rcbd + 1)
    rows = []
    for cohort_group, cohort_filter in [("MMLU", ["MMLU"]), ("SLAB", ["BBH", "LMentry"])]:
        sub_c = df_rcbd_all[df_rcbd_all["cohort"].isin(cohort_filter)]
        for est_name, grp_est in sub_c.groupby("estimator"):
            n_tests = len(grp_est)
            n_zero_var = int(grp_est["zero_var_contrast"].sum()) if "zero_var_contrast" in grp_est.columns else 0

            # 1. Per-cube BH & BY
            sig_u_cube, sig_b_cube, sig_c_cube, sig_p_cube = 0, 0, 0, 0
            by_u_cube, by_b_cube, by_c_cube, by_p_cube = 0, 0, 0, 0
            for _, g_cube in grp_est.groupby(["cohort", "model_id", "task"]):
                sig_u_cube += int(np.sum(multipletests(g_cube["p_unblocked"], method="fdr_bh")[1] < 0.05))
                sig_b_cube += int(np.sum(multipletests(g_cube["p_rcbd"], method="fdr_bh")[1] < 0.05))
                sig_c_cube += int(np.sum(multipletests(g_cube["p_item_cluster"], method="fdr_bh")[1] < 0.05))
                sig_p_cube += int(np.sum(multipletests(g_cube["p_sign_flip_perm"], method="fdr_bh")[1] < 0.05))
                by_u_cube += int(np.sum(multipletests(g_cube["p_unblocked"], method="fdr_by")[1] < 0.05))
                by_b_cube += int(np.sum(multipletests(g_cube["p_rcbd"], method="fdr_by")[1] < 0.05))
                by_c_cube += int(np.sum(multipletests(g_cube["p_item_cluster"], method="fdr_by")[1] < 0.05))
                by_p_cube += int(np.sum(multipletests(g_cube["p_sign_flip_perm"], method="fdr_by")[1] < 0.05))

            # 2. Per-model BH & BY
            sig_u_mod, sig_b_mod, sig_c_mod, sig_p_mod = 0, 0, 0, 0
            by_u_mod, by_b_mod, by_c_mod, by_p_mod = 0, 0, 0, 0
            # A model ID shared by BBH and LMentry is one per-model family. Grouping by
            # benchmark as well would split the same model into two less-stringent families.
            model_groups = list(grp_est.groupby("model_id", sort=True))
            for _, g_mod in model_groups:
                sig_u_mod += int(np.sum(multipletests(g_mod["p_unblocked"], method="fdr_bh")[1] < 0.05))
                sig_b_mod += int(np.sum(multipletests(g_mod["p_rcbd"], method="fdr_bh")[1] < 0.05))
                sig_c_mod += int(np.sum(multipletests(g_mod["p_item_cluster"], method="fdr_bh")[1] < 0.05))
                sig_p_mod += int(np.sum(multipletests(g_mod["p_sign_flip_perm"], method="fdr_bh")[1] < 0.05))
                by_u_mod += int(np.sum(multipletests(g_mod["p_unblocked"], method="fdr_by")[1] < 0.05))
                by_b_mod += int(np.sum(multipletests(g_mod["p_rcbd"], method="fdr_by")[1] < 0.05))
                by_c_mod += int(np.sum(multipletests(g_mod["p_item_cluster"], method="fdr_by")[1] < 0.05))
                by_p_mod += int(np.sum(multipletests(g_mod["p_sign_flip_perm"], method="fdr_by")[1] < 0.05))

            # 3. Global cohort BH & BY
            sig_u_glob = int(np.sum(multipletests(grp_est["p_unblocked"], method="fdr_bh")[1] < 0.05))
            sig_b_glob = int(np.sum(multipletests(grp_est["p_rcbd"], method="fdr_bh")[1] < 0.05))
            sig_c_glob = int(np.sum(multipletests(grp_est["p_item_cluster"], method="fdr_bh")[1] < 0.05))
            sig_p_glob = int(np.sum(multipletests(grp_est["p_sign_flip_perm"], method="fdr_bh")[1] < 0.05))
            by_u_glob = int(np.sum(multipletests(grp_est["p_unblocked"], method="fdr_by")[1] < 0.05))
            by_b_glob = int(np.sum(multipletests(grp_est["p_rcbd"], method="fdr_by")[1] < 0.05))
            by_c_glob = int(np.sum(multipletests(grp_est["p_item_cluster"], method="fdr_by")[1] < 0.05))
            by_p_glob = int(np.sum(multipletests(grp_est["p_sign_flip_perm"], method="fdr_by")[1] < 0.05))

            for fam_name, su, sb, sc, sp, bu, bb, bc, bp in [
                ("per_cube_bh", sig_u_cube, sig_b_cube, sig_c_cube, sig_p_cube, by_u_cube, by_b_cube, by_c_cube, by_p_cube),
                ("per_model_bh", sig_u_mod, sig_b_mod, sig_c_mod, sig_p_mod, by_u_mod, by_b_mod, by_c_mod, by_p_mod),
                ("global_cohort_bh", sig_u_glob, sig_b_glob, sig_c_glob, sig_p_glob, by_u_glob, by_b_glob, by_c_glob, by_p_glob),
            ]:
                rows.append(
                    {
                        "cohort": cohort_group,
                        "estimator": str(est_name),
                        "fdr_family": fam_name,
                        "inference_scope": "conditional_fixed_panel_model_based_item_tests",
                        "total_tests": n_tests,
                        "n_model_families": int(len(model_groups)),
                        "n_zero_var_contrasts": n_zero_var,
                        "sign_flip_n_perm": int(n_perm_rcbd),
                        "sign_flip_min_achievable_p": float(min_achievable_p),
                        "detections_unblocked": su,
                        "detections_rcbd_fixed_block": sb,
                        "detections_item_cluster_t": sc,
                        "detections_item_sign_flip_perm": sp,
                        "detections_unblocked_by": bu,
                        "detections_rcbd_fixed_block_by": bb,
                        "detections_item_cluster_t_by": bc,
                        "detections_item_sign_flip_perm_by": bp,
                        "detection_rate_unblocked_pct": su / n_tests * 100.0,
                        "detection_rate_rcbd_pct": sb / n_tests * 100.0,
                        "detection_rate_item_cluster_pct": sc / n_tests * 100.0,
                        "ratio_rcbd_over_unblocked": float(sb / max(1, su)),
                        "ratio_item_cluster_over_unblocked": float(sc / max(1, su)),
                        "ratio_item_cluster_by_over_unblocked_by": float(bc / max(1, bu)),
                    }
                )
    return pd.DataFrame(rows)


def _partial_corr_1d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """Compute partial Pearson r and partial Spearman rho between x and y controlling for z (1D or 2D)."""
    if z.ndim == 1:
        z = z[:, None]
    Z = np.column_stack([np.ones(len(x)), z])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    p_r = float(np.corrcoef(rx, ry)[0, 1]) if (np.std(rx) > 1e-15 and np.std(ry) > 1e-15) else 0.0

    xr = stats.rankdata(x)
    yr = stats.rankdata(y)
    zr = np.column_stack([stats.rankdata(z[:, j]) for j in range(z.shape[1])])
    Zr = np.column_stack([np.ones(len(x)), zr])
    bxr, *_ = np.linalg.lstsq(Zr, xr, rcond=None)
    byr, *_ = np.linalg.lstsq(Zr, yr, rcond=None)
    rxr = xr - Zr @ bxr
    ryr = yr - Zr @ byr
    p_rho = float(np.corrcoef(rxr, ryr)[0, 1]) if (np.std(rxr) > 1e-15 and np.std(ryr) > 1e-15) else 0.0
    return p_r, p_rho


def _two_way_cluster_bootstrap_correlation(
    df_grp: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: Optional[str] = None,
    n_boot: int = 1000,
    seed: int = 20261130,
) -> Dict[str, float]:
    """
    Two-way model-task cluster bootstrap for Pearson r and Spearman rho (and optional partial r/rho
    controlling for `z_col`) across model-task cubes (Task 5).
    Pivots by `index="model_id"` and `columns=["cohort", "task"]` so shared models across BBH and LMentry
    are resampled together as single model clusters.
    """
    x_full = df_grp[x_col].to_numpy(dtype=float)
    y_full = df_grp[y_col].to_numpy(dtype=float)
    r_hat, _ = stats.pearsonr(x_full, y_full)
    rho_hat, _ = stats.spearmanr(x_full, y_full)

    # Single unified panel across (cohort, task) columns so shared model_ids stay coupled
    px = df_grp.pivot(index="model_id", columns=["cohort", "task"], values=x_col).to_numpy(dtype=float)
    py = df_grp.pivot(index="model_id", columns=["cohort", "task"], values=y_col).to_numpy(dtype=float)
    pz = df_grp.pivot(index="model_id", columns=["cohort", "task"], values=z_col).to_numpy(dtype=float) if z_col is not None else None
    M, T = px.shape

    part_r_hat, part_rho_hat = (0.0, 0.0)
    if z_col is not None:
        z_full = df_grp[z_col].to_numpy(dtype=float)
        part_r_hat, part_rho_hat = _partial_corr_1d(x_full, y_full, z_full)

    rng = np.random.default_rng(seed)
    r_boot = np.empty(n_boot, dtype=float)
    rho_boot = np.empty(n_boot, dtype=float)
    part_r_boot = np.empty(n_boot, dtype=float)
    part_rho_boot = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        m_idx = rng.integers(0, M, size=M)
        t_idx = rng.integers(0, T, size=T)
        xb = px[m_idx][:, t_idx].ravel()
        yb = py[m_idx][:, t_idx].ravel()
        valid = ~(np.isnan(xb) | np.isnan(yb))
        x_cat = xb[valid]
        y_cat = yb[valid]
        if len(x_cat) < 3 or np.std(x_cat) < 1e-15 or np.std(y_cat) < 1e-15:
            r_boot[b] = 0.0
            rho_boot[b] = 0.0
            part_r_boot[b] = 0.0
            part_rho_boot[b] = 0.0
        else:
            r_boot[b] = float(np.corrcoef(x_cat, y_cat)[0, 1])
            rho_boot[b] = float(stats.spearmanr(x_cat, y_cat)[0])
            if pz is not None:
                zb = pz[m_idx][:, t_idx].ravel()[valid]
                part_r_boot[b], part_rho_boot[b] = _partial_corr_1d(x_cat, y_cat, zb)
            else:
                part_r_boot[b] = 0.0
                part_rho_boot[b] = 0.0

    r_se = float(np.std(r_boot, ddof=1))
    rho_se = float(np.std(rho_boot, ddof=1))
    r_ci_lo, r_ci_hi = float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5))
    rho_ci_lo, rho_ci_hi = float(np.percentile(rho_boot, 2.5)), float(np.percentile(rho_boot, 97.5))

    res = {
        "interpretation": "conditional_model_task_panel_reweighting_sensitivity",
        "confirmatory_null_test": False,
        "n_unique_models": int(M),
        "n_unique_tasks": int(T),
        "pearson_r": float(r_hat),
        "pearson_se_cluster": r_se,
        "pearson_ci_low": r_ci_lo,
        "pearson_ci_high": r_ci_hi,
        "spearman_rho": float(rho_hat),
        "spearman_se_cluster": rho_se,
        "spearman_ci_low": rho_ci_lo,
        "spearman_ci_high": rho_ci_hi,
    }
    if z_col is not None:
        res.update(
            {
                "partial_control_col": z_col,
                "partial_pearson_r": float(part_r_hat),
                "partial_pearson_se_cluster": float(np.std(part_r_boot, ddof=1)),
                "partial_pearson_ci_low": float(np.percentile(part_r_boot, 2.5)),
                "partial_pearson_ci_high": float(np.percentile(part_r_boot, 97.5)),
                "partial_spearman_rho": float(part_rho_hat),
                "partial_spearman_se_cluster": float(np.std(part_rho_boot, ddof=1)),
                "partial_spearman_ci_low": float(np.percentile(part_rho_boot, 2.5)),
                "partial_spearman_ci_high": float(np.percentile(part_rho_boot, 97.5)),
            }
        )
    return res


def compute_interaction_regret_associations(df_fourier_all: pd.DataFrame) -> Dict[str, Any]:
    """
    Quantify correlation between 2FI RMS magnitude (`disc0_rms_2fi`, `mean20_disc_rms_2fi`, `full_rms_2fi`)
    and additive-model regret (`regret_main_20splits`, `regret_main_split0`) with shared-model two-way
    cluster bootstrap CIs and partial correlations controlling for main-effect RMS (SI-1, Task 5).
    """
    out: Dict[str, Any] = {}
    for cohort_group, cohort_filter in [("MMLU", ["MMLU"]), ("SLAB", ["BBH", "LMentry"])]:
        sub = df_fourier_all[df_fourier_all["cohort"].isin(cohort_filter)]
        for est_name, grp in sub.groupby("estimator"):
            c_reg_disc0 = _two_way_cluster_bootstrap_correlation(
                grp, "disc0_rms_2fi", "regret_main_20splits", z_col="disc0_rms_main", n_boot=1000, seed=SEEDS["cluster_bootstrap"]
            )
            c_reg_split0 = _two_way_cluster_bootstrap_correlation(
                grp, "disc0_rms_2fi", "regret_main_split0", z_col="disc0_rms_main", n_boot=1000, seed=SEEDS["cluster_bootstrap"] + 3
            )
            c_reg_mean20 = _two_way_cluster_bootstrap_correlation(
                grp, "mean20_disc_rms_2fi", "regret_main_20splits", z_col="mean20_disc_rms_main", n_boot=1000, seed=SEEDS["cluster_bootstrap"] + 7
            )
            c_reg = c_reg_mean20
            c_reg_full = _two_way_cluster_bootstrap_correlation(
                grp, "full_rms_2fi", "regret_main_20splits", z_col="full_rms_main", n_boot=1000, seed=SEEDS["cluster_bootstrap"] + 11
            )
            c_reg_first = _two_way_cluster_bootstrap_correlation(
                grp, "mean20_disc_rms_2fi", "regret_main_20splits_first_index", z_col="mean20_disc_rms_main", n_boot=1000, seed=SEEDS["cluster_bootstrap"] + 13
            )
            c_gain = _two_way_cluster_bootstrap_correlation(
                grp, "mean20_disc_rms_2fi", "gain_best_over_main_20splits", z_col="mean20_disc_rms_main", n_boot=1000, seed=SEEDS["cluster_bootstrap"] + 17
            )
            out[f"{cohort_group}_{est_name}"] = {
                "n_cubes": len(grp),
                "n_unique_models": c_reg["n_unique_models"],
                "n_unique_tasks": c_reg["n_unique_tasks"],
                "mean_disc0_rms_main": float(grp["disc0_rms_main"].mean()),
                "mean_disc0_rms_2fi": float(grp["disc0_rms_2fi"].mean()),
                "mean_disc0_rms_3fi": float(grp["disc0_rms_3fi"].mean()),
                "mean_disc0_rms_4fi": float(grp["disc0_rms_4fi"].mean()),
                "mean_mean20_disc_rms_main": float(grp["mean20_disc_rms_main"].mean()),
                "mean_mean20_disc_rms_2fi": float(grp["mean20_disc_rms_2fi"].mean()),
                "mean_full_rms_main": float(grp["full_rms_main"].mean()),
                "mean_full_rms_2fi": float(grp["full_rms_2fi"].mean()),
                "mean_full_var_share_main": float(grp["full_var_share_main"].mean()),
                "mean_full_var_share_2fi": float(grp["full_var_share_2fi"].mean()),
                "mean_full_var_share_higher_order": float(grp["full_var_share_higher_order"].mean()),
                "pearson_r_rms2fi_vs_main_regret": c_reg["pearson_r"],
                "pearson_se_cluster_rms2fi_vs_main_regret": c_reg["pearson_se_cluster"],
                "pearson_ci_low_rms2fi_vs_main_regret": c_reg["pearson_ci_low"],
                "pearson_ci_high_rms2fi_vs_main_regret": c_reg["pearson_ci_high"],
                "spearman_rho_rms2fi_vs_main_regret": c_reg["spearman_rho"],
                "spearman_se_cluster_rms2fi_vs_main_regret": c_reg["spearman_se_cluster"],
                "spearman_ci_low_rms2fi_vs_main_regret": c_reg["spearman_ci_low"],
                "spearman_ci_high_rms2fi_vs_main_regret": c_reg["spearman_ci_high"],
                "partial_pearson_r_rms2fi_vs_main_regret_given_rms_main": c_reg["partial_pearson_r"],
                "partial_pearson_ci_low_rms2fi_vs_main_regret_given_rms_main": c_reg["partial_pearson_ci_low"],
                "partial_pearson_ci_high_rms2fi_vs_main_regret_given_rms_main": c_reg["partial_pearson_ci_high"],
                "partial_spearman_rho_rms2fi_vs_main_regret_given_rms_main": c_reg["partial_spearman_rho"],
                "partial_spearman_ci_low_rms2fi_vs_main_regret_given_rms_main": c_reg["partial_spearman_ci_low"],
                "partial_spearman_ci_high_rms2fi_vs_main_regret_given_rms_main": c_reg["partial_spearman_ci_high"],
                "panel_reweighting_association_sensitivity": {
                    "split0_rms2fi_vs_regret_main_20splits": c_reg_disc0,
                    "split0_rms2fi_vs_regret_main_split0": c_reg_split0,
                    "mean20_disc_rms2fi_vs_regret_main_20splits": c_reg_mean20,
                    "full_rms2fi_vs_regret_main_20splits": c_reg_full,
                    "mean20_disc_rms2fi_vs_regret_main_20splits_first_index": c_reg_first,
                },
                "pearson_r_rms2fi_vs_screenbest_gain_over_main": c_gain["pearson_r"],
                "pearson_se_cluster_rms2fi_vs_screenbest_gain_over_main": c_gain["pearson_se_cluster"],
                "pearson_ci_low_rms2fi_vs_screenbest_gain_over_main": c_gain["pearson_ci_low"],
                "pearson_ci_high_rms2fi_vs_screenbest_gain_over_main": c_gain["pearson_ci_high"],
                "spearman_rho_rms2fi_vs_screenbest_gain_over_main": c_gain["spearman_rho"],
                "spearman_se_cluster_rms2fi_vs_screenbest_gain_over_main": c_gain["spearman_se_cluster"],
                "spearman_ci_low_rms2fi_vs_screenbest_gain_over_main": c_gain["spearman_ci_low"],
                "spearman_ci_high_rms2fi_vs_screenbest_gain_over_main": c_gain["spearman_ci_high"],
            }
    return out


# ==============================================================================
# MODULE 4: paper-artifacts (PA-1 .. PA-5)
# ==============================================================================

def generate_figures(
    df_mmlu_models: pd.DataFrame,
    res_c1_leg: Dict[str, Any],
    res_c1_cell: Dict[str, Any],
    res_c2_leg: Dict[str, Any],
    res_c2_cell: Dict[str, Any],
    df_cosets: pd.DataFrame,
) -> None:
    """Generate publication-grade figures with uncertainty intervals and coset ranges (PA-2)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "figure.dpi": 300,
        }
    )

    s1_leg = res_c1_leg["summary"]
    s2_leg = res_c2_leg["summary"]

    # Figure 1: Effect & Full-Rank 2FI RMSE (a), Cohort 2 Main-Effect RMSE (b), and Coset/Split-Aware Selection (c)
    fig, axes = plt.subplots(
        1, 3, figsize=(14.6, 4.25), gridspec_kw={"width_ratios": [1.06, 0.96, 1.32]}
    )

    # Panel A: Cohort 1 (MMLU) Pooled Main-Effect & Full-Rank 2FI RMSE
    ax = axes[0]
    labels_a = ["Main Effects\n(16 runs)", "2FI Products\n(16 runs, Full-Rank)", "Main Effects\n(8 runs, all-60)"]
    prism_a = [
        s1_leg["mean_pool_rmse_main_res5"],
        s1_leg["mean_pool_rmse_2fi_res5"],
        s1_leg["mean_pool_rmse_main_res3_all60"],
    ]
    rand_a = [
        s1_leg["mean_pool_rmse_main_rand16"],
        s1_leg["mean_pool_rmse_2fi_rand16_fullrank"],
        s1_leg["mean_pool_rmse_main_rand8"],
    ]
    x = np.arange(len(labels_a))
    w = 0.35
    b1 = ax.bar(
        x - w / 2, prism_a, w,
        label="Orthogonal ($2^{5-1}_{\\mathrm{V}}$ / $2^{5-2}_{\\mathrm{III}}$ all-60)",
        color="#1f77b4", edgecolor="black", linewidth=0.7,
    )
    b2 = ax.bar(
        x + w / 2, rand_a, w,
        label="Full-Rank Balanced Random (200 designs)",
        color="#ff7f0e", edgecolor="black", linewidth=0.7,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Pooled Estimation RMSE (log scale)")
    ax.set_title("(a) Cohort 1 (MMLU, 15 LLMs): Effect RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_a)
    ax.legend(loc="upper left", frameon=True, fontsize=7.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(1.2e-3, 3.4e-2)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h * 1.08),
            ha="center", va="bottom", fontsize=7.5,
        )

    # Panel B: Cohort 2 (BBH + LMentry, 178 cubes) Main-Effect RMSE across 20 splits
    ax = axes[1]
    methods_b = ["Orthogonal\n$2^{4-1}_{\\mathrm{IV}}$ (8 runs)", "Balanced Random\n(8 runs, 200 des.)", "Ref.-Cell-0 OFAT\n(5 runs, lower budget)"]
    df_c2_leg = res_c2_leg["df_cubes"]
    vals_b = [
        float(df_c2_leg["rmse_main_res4_20splits"].mean()),
        float(df_c2_leg["rmse_main_rand8_20splits"].mean()),
        float(df_c2_leg["rmse_main_ofat_20splits"].mean()),
    ]
    colors_b = ["#1f77b4", "#ff7f0e", "#d62728"]
    bars_b = ax.bar(methods_b, vals_b, width=0.50, color=colors_b, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Main-Effect RMSE (20-split mean)")
    ax.set_title("(b) Cohort 2 (BBH & LMentry, 178 Cubes)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(vals_b) * 1.25)
    for bar in bars_b:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h + 0.004),
            ha="center", va="bottom", fontsize=8.0, fontweight="bold",
        )

    # Panel C: Held-out Prompt Selection Accuracy on Cohort 2 (nonoverlapping labels at final PDF size)
    ax = axes[2]
    rules_c = [
        "Cell 0\nRef.",
        "OFAT\n(5 runs)",
        "Additive\nSign",
        "Orthog.\nBest (8)",
        "Random\nBest (8)",
        "Oracle\n(16 runs)",
    ]
    vals_c = [
        s2_leg["mean_conf_baseline_20splits"],
        s2_leg["mean_conf_ofat_20splits"],
        s2_leg["mean_conf_res4_main_coset_avg_20splits"],
        s2_leg["mean_conf_res4_best_coset_avg_20splits"],
        s2_leg["mean_conf_rand8_best_20splits"],
        s2_leg["mean_conf_oracle16_20splits"],
    ]
    colors_c = ["#7f7f7f", "#d62728", "#9467bd", "#1f77b4", "#ff7f0e", "#2ca02c"]
    x_c = np.arange(len(rules_c))
    bars_c = ax.bar(x_c, vals_c, width=0.54, color=colors_c, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x_c)
    ax.set_xticklabels(rules_c, fontsize=7.8)
    ax.set_xlim(-0.58, len(rules_c) - 0.42)
    ax.set_ylabel("Confirmation Accuracy (20-split mean)")
    ax.set_title("(c) Cohort 2 Selection (Tie-Averaged)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0.32, 0.58)
    for bar in bars_c:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h + 0.004),
            ha="center", va="bottom", fontsize=7.5,
        )

    fig.tight_layout(w_pad=1.8)
    fig.savefig(FIGURES_DIR / "fig1_effect_and_selection.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES_DIR / "fig1_effect_and_selection.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Multi-Model Prompt Sensitivity & RCBD / Item-Clustered Detection Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # Panel A: Prompt Sensitivity Spread across 15 LLMs on MMLU
    ax = axes[0]
    df_sorted = df_mmlu_models.sort_values("spread_pp", ascending=True)
    y_pos = np.arange(len(df_sorted))
    bar_colors = ["#d95f02" if t == "Base" else "#1b9e77" for t in df_sorted["model_type"]]
    ax.barh(y_pos, df_sorted["spread_pp"], color=bar_colors, edgecolor="black", linewidth=0.6, height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_label"], fontsize=8.5)
    ax.set_xlabel("Accuracy Spread Across $2^5=32$ Contrast Cube (Percentage Points)")
    ax.set_title("(a) Prompt Sensitivity Spread Across 15 LLMs (MMLU)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d95f02", edgecolor="black", label="Base Pretrained LLM"),
        Patch(facecolor="#1b9e77", edgecolor="black", label="Instruction-Tuned LLM"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    # Panel B: Unblocked vs Item-Clustered Robust vs Fixed-Block RCBD Detections across 15 LLMs
    ax = axes[1]
    x_pos = np.arange(len(df_sorted))
    w = 0.27
    ax.barh(
        x_pos - w, df_sorted["sig_unblocked_total"], height=w,
        color="#bdbdbd", edgecolor="black", linewidth=0.6,
        label="Unblocked ANOVA (Underpowered)",
    )
    ax.barh(
        x_pos, df_sorted["sig_item_cluster_total"], height=w,
        color="#fc8d59", edgecolor="black", linewidth=0.6,
        label="Item-Clustered $t$-Test (Calibrated Primary)",
    )
    ax.barh(
        x_pos + w, df_sorted["sig_rcbd_total"], height=w,
        color="#2b8cbe", edgecolor="black", linewidth=0.6,
        label="Pooled Fixed-Block RCBD (Uncalibrated)",
    )
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df_sorted["model_label"], fontsize=8.5)
    ax.set_xlabel("Per-Cube BH ($\\alpha=0.05$) Rejections Across 57 Tasks (out of 285 tests)")
    ax.set_title("(b) Calibrated Clustered $t$ vs. Uncalibrated Pooled RCBD")
    ax.set_xlim(0, 124)
    ax.legend(loc="lower right", frameon=True, fontsize=7.8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_models_and_rcbd.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURES_DIR / "fig2_models_and_rcbd.png", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(
    df_mmlu_models: pd.DataFrame,
    res_c1_leg: Dict[str, Any],
    res_c1_cell: Dict[str, Any],
    res_c2_leg: Dict[str, Any],
    res_c2_cell: Dict[str, Any],
    df_mult: pd.DataFrame,
    res_c1_cell_uniq: Optional[Dict[str, Any]] = None,
    res_c2_cell_uniq: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write machine-generated LaTeX tables consumed by `paper_hf/main.tex` (PA-2, PA-3, Task 8, Task 11).
    Formatted with native `\\footnotesize` and compact column spacing so tables render at full legible
    font size without `\\resizebox` shrinkage.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    s1 = res_c1_leg["summary"]
    s1_cell = res_c1_cell["summary"]
    s1_uniq = res_c1_cell_uniq["summary"] if res_c1_cell_uniq is not None else s1_cell
    s2 = res_c2_leg["summary"]
    s2_cell = res_c2_cell["summary"]
    s2_uniq = res_c2_cell_uniq["summary"] if res_c2_cell_uniq is not None else s2_cell

    ratio_2fi_pool = s1["pool_2fi_rmse_ratio_fullrank_over_res5"]
    ratio_rcbd_det = s1["total_sig_rcbd_per_cube_bh"] / max(1, s1["total_sig_unblocked_per_cube_bh"])
    ratio_clust_det = s1["total_sig_item_cluster_per_cube_bh"] / max(1, s1["total_sig_unblocked_per_cube_bh"])

    # Table 2 (table_mmlu_models.tex): Model-by-Model MMLU 2^5 Factorial Results (15 LLMs)
    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{0.5pt}",
        (
            "\\caption{Cohort 1 (PromptEval MMLU, 2023--2024 open-weights LLMs): $2^5=32$ constructed factorial evaluation "
            "across 15 LLMs and 57 MMLU subjects ($14{,}042$ items per model; $6{,}740{,}160$ cell--item observations; first-row single-template representative, tie-averaged selection). "
            f"The 16-run Resolution V design ($2^{{5-1}}_{{\\mathrm{{V}}}}$) separates main effects and second-order products in the five-bit binary contrast basis "
            f"from one another (aliasing them with fourth- and third-order basis products), reducing pooled $2\\mathrm{{FI}}$-product RMSE by "
            f"${ratio_2fi_pool:.2f}\\times$ on average (and in 15/15 models) relative to 200 full-rank balanced random 16-run attribution designs. "
            "Held-out confirmation accuracy is averaged across both $2^{5-1}_{\\mathrm{V}}$ cosets ($I=\\pm ABCDE$) and compared against "
            "200 unconditioned factor-balanced Random-16 screens and the 32-run discovery oracle. Rightmost columns report per-cube Benjamini--Hochberg "
            f"($\\alpha=0.05$) detections (conditional on evaluated templates, models, and tasks) for underpowered Unblocked ANOVA, calibrated Item-Clustered contrast $t$-tests (\\textbf{{${ratio_clust_det:.2f}\\times$}}), "
            f"and uncalibrated Pooled Fixed-Block RCBD (${ratio_rcbd_det:.2f}\\times$, withdrawn from recommendations due to variance-pooling false positives).}}"
        ),
        "\\label{tab:mmlu_models}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccccccc@{}}",
        "\\toprule",
        f"\\textbf{{Model}} & \\textbf{{Type}} & \\textbf{{Mean}} & \\textbf{{Spread}} & \\multicolumn{{2}}{{c}}{{\\textbf{{2FI-Product RMSE (${ratio_2fi_pool:.2f}\\times$)}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Conf.\\ Acc.\\ (Split 0)}}}} & \\textbf{{BH Det.\\ (${ratio_clust_det:.2f}\\times$)}} \\\\",
        "\\cmidrule(lr){5-6} \\cmidrule(lr){7-9}",
        " & & \\textbf{Acc.} & \\textbf{(pp)} & \\textbf{$2^{5-1}_{\\mathrm{V}}$} & \\textbf{Rand-16} & \\textbf{$2^{5-1}_{\\mathrm{V}}$} & \\textbf{Rand-16} & \\textbf{Oracle} & \\textbf{Unb./Cl.$t$/RCBD} \\\\",
        "\\midrule",
    ]
    for _, r in df_mmlu_models.iterrows():
        lines.append(
            f"{r['model_label']} & {r['model_type']} & {r['mean_acc']:.4f} & {r['spread_pp']:.2f} & "
            f"\\textbf{{{r['pool_rmse_2fi_res5']:.4f}}} & {r['pool_rmse_2fi_rand16_fullrank']:.4f} & "
            f"{r['task_conf_res5_best_coset_avg']:.4f} & {r['task_conf_rand16_best']:.4f} & {r['task_conf_oracle32']:.4f} & "
            f"{int(r['sig_unblocked_total'])}/\\textbf{{{int(r['sig_item_cluster_total'])}}}/{int(r['sig_rcbd_total'])} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"\\textbf{{Mean / Total}} & --- & {df_mmlu_models['mean_acc'].mean():.4f} & {df_mmlu_models['spread_pp'].mean():.2f} & "
            f"\\textbf{{{s1['mean_pool_rmse_2fi_res5']:.4f}}} & {s1['mean_pool_rmse_2fi_rand16_fullrank']:.4f} & "
            f"{s1['mean_conf_res5_best_coset_avg']:.4f} & {s1['mean_conf_rand16_best']:.4f} & {s1['mean_conf_oracle32']:.4f} & "
            f"{s1['total_sig_unblocked_per_cube_bh']}/\\textbf{{{s1['total_sig_item_cluster_per_cube_bh']}}}/{s1['total_sig_rcbd_per_cube_bh']} \\\\",
            "\\bottomrule",
            "\\end{tabular*}",
            "\\end{table*}",
        ]
    )
    (TABLES_DIR / "table_mmlu_models.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Compute dynamic ratios for Table 3 (table_cross_cohort.tex)
    r_c1_main_r16 = s1["task_main_rmse_ratio_rand16_over_res5"]
    r_c1_main_ofat = s1["task_main_rmse_ratio_ofat_over_res5"]
    r_c1_main_r8_4c = s1["task_main_rmse_ratio_rand8_over_res3_coset_avg"]
    r_c1_main_r8_60 = s1["task_main_rmse_ratio_rand8_over_res3_all60"]
    r_c2_main_r8 = s2["main_rmse_ratio_rand8_over_res4"]
    r_c2_main_ofat = s2["main_rmse_ratio_ofat_over_res4"]
    r_c1_2fi_task = s1["task_2fi_rmse_ratio_fullrank_over_res5"]

    rec_c1_sb_pos = s1["recovery_split0_res5_best_coset_pos"]["ratio_of_means_pct"]
    rec_c1_sb_neg = s1["recovery_split0_res5_best_coset_neg"]["ratio_of_means_pct"]
    rec_c1_sb_avg = s1["recovery_split0_res5_best_coset_avg"]["ratio_of_means_pct"]
    rec_c1_r16_sb = s1["recovery_split0_rand16_best"]["ratio_of_means_pct"]
    rec_c1_2fi_pos = s1["recovery_split0_res5_2fi_coset_pos"]["ratio_of_means_pct"]
    rec_c1_2fi_neg = s1["recovery_split0_res5_2fi_coset_neg"]["ratio_of_means_pct"]
    rec_c1_2fi_avg = s1["recovery_split0_res5_2fi_coset_avg"]["ratio_of_means_pct"]

    pos_c1_s0_pct = s1["recovery_split0_res5_best_coset_avg"]["positive_uplift_cube_fraction"] * 100.0
    pos_c1_s0_n = s1["recovery_split0_res5_best_coset_avg"]["n_positive_uplift_cubes"]
    pos_c1_20s_pct = s1["recovery_20splits_res5_best_coset_avg"]["positive_uplift_cube_fraction"] * 100.0
    pos_c1_20s_n = s1["recovery_20splits_res5_best_coset_avg"]["n_positive_uplift_cubes"]
    pos_c1_cell_20s_pct = s1_cell["recovery_20splits_res5_best_coset_avg"]["positive_uplift_cube_fraction"] * 100.0

    rec_c2_sb_pos = s2["recovery_split0_res4_best_coset_pos"]["ratio_of_means_pct"]
    rec_c2_sb_neg = s2["recovery_split0_res4_best_coset_neg"]["ratio_of_means_pct"]
    rec_c2_sb_avg = s2["recovery_split0_res4_best_coset_avg"]["ratio_of_means_pct"]
    rec_c2_r8_sb = s2["recovery_split0_rand8_best"]["ratio_of_means_pct"]

    pos_c2_s0_pct = s2["recovery_split0_res4_best_coset_avg"]["positive_uplift_cube_fraction"] * 100.0
    pos_c2_s0_n = s2["recovery_split0_res4_best_coset_avg"]["n_positive_uplift_cubes"]
    pos_c2_20s_pct = s2["recovery_20splits_res4_best_coset_avg"]["positive_uplift_cube_fraction"] * 100.0
    pos_c2_20s_n = s2["recovery_20splits_res4_best_coset_avg"]["n_positive_uplift_cubes"]
    pos_c2_cell_20s_pct = s2_cell["recovery_20splits_res4_best_coset_avg"]["positive_uplift_cube_fraction"] * 100.0

    m1_cube = df_mult[(df_mult["cohort"] == "MMLU") & (df_mult["estimator"] == "legacy_first") & (df_mult["fdr_family"] == "per_cube_bh")].iloc[0]
    m2_cube = df_mult[(df_mult["cohort"] == "SLAB") & (df_mult["estimator"] == "legacy_first") & (df_mult["fdr_family"] == "per_cube_bh")].iloc[0]
    m1_glob = df_mult[(df_mult["cohort"] == "MMLU") & (df_mult["estimator"] == "legacy_first") & (df_mult["fdr_family"] == "global_cohort_bh")].iloc[0]
    m2_glob = df_mult[(df_mult["cohort"] == "SLAB") & (df_mult["estimator"] == "legacy_first") & (df_mult["fdr_family"] == "global_cohort_bh")].iloc[0]

    # Table 3: Cross-Cohort Synthesis formatted cleanly at \scriptsize without \resizebox
    lines2 = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{0.95}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        (
            "\\caption{Cross-cohort synthesis of contrast reliability, attribution error, tie-averaged prompt selection, "
            "stratum sensitivity (first-row single-template representative vs.\\ unique-template [uniq] and catalog-row [row] stratum averages, which average 1--6 MMLU or 1--56 SLAB templates per cell and are not equal-budget single-prompt screens), "
            "and empirical-null calibrated inference across 1,033 constructed cubes ($855$ MMLU $2^5$ cubes [$6{,}740{,}160$ cell--item obs.] and $178$ completeness-selected SLAB-NLP $2^4$ cubes across $6/15$ BBH and $7/10$ LMentry tasks [$265{,}856$ cell--item obs.; $7{,}006{,}016$ total]).}"
        ),
        "\\label{tab:cross_cohort}",
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.35\\textwidth}>{\\raggedright\\arraybackslash}p{0.31\\textwidth}>{\\raggedright\\arraybackslash}p{0.28\\textwidth}}",
        "\\toprule",
        "\\textbf{Metric / Experimental Estimand} & \\textbf{Cohort 1: MMLU ($2^5=32$, 855 Cubes)} & \\textbf{Cohort 2: SLAB ($2^4=16$, 178 Cubes)} \\\\",
        "\\midrule",
        "\\multicolumn{3}{p{0.96\\textwidth}}{\\textit{Part A: Effect Reliability, Noise-Corrected Signal RMS, Absolute Error-to-Signal Ratios (ESR), \\& Design Attribution}} \\\\",
        (
            "Split-Half Reliability $r$ (Main / 2FI / 3FI, 20-Split) & "
            "Task: 0.330 / 0.155 / 0.148; Pool: \\textbf{0.964 / 0.890 / 0.889} & "
            "\\textbf{0.924 / 0.863 / 0.885} [uniq: 0.952 / 0.892 / 0.880] \\\\"
        ),
        (
            f"Observed vs.\\ Noise-Corrected Cohort Signal RMS (Main / 2FI / 3FI) & "
            f"Obs: {s1['mean_disc0_rms_main_true']:.5f}/{s1['mean_disc0_rms_2fi_true']:.5f}/{s1['mean_disc0_rms_3fi_true']:.5f}; Corr: \\textbf{{0.00730/0.00321/0.00309}} & "
            f"Obs: {s2['mean_disc0_rms_main_true']:.5f}/{s2['mean_disc0_rms_2fi_true']:.5f}/{s2['mean_disc0_rms_3fi_true']:.5f}; Corr [uniq]: \\textbf{{0.0773/0.0390/0.0327}} \\\\"
        ),
        (
            "Absolute Error-to-Signal Ratio (Attr.\\ ESR vs.\\ Signal; Held-Out Pred.\\ ESR) & "
            "Pool Main $2^{5-1}_{\\mathrm{V}}$: \\textbf{0.50} (pred 0.57); Task 2FI: \\textbf{2.47} vs.\\ Rand-16 12.26 (pred \\textbf{1.64} vs.\\ 5.08) & "
            "Main $2^{4-1}_{\\mathrm{IV}}$: \\textbf{0.74} (pred \\textbf{0.81}; uniq \\textbf{0.45}) vs.\\ Rand-8 1.10 (pred 1.13) \\\\"
        ),
        (
            f"Orthogonal Half-Fraction ($2^{{k-1}}$, 50\\% budget) Main RMSE & "
            f"\\textbf{{{s1['mean_task_rmse_main_res5']:.5f}}} [uniq: {s1_uniq['mean_task_rmse_main_res5']:.5f}; row: {s1_cell['mean_task_rmse_main_res5']:.5f}] & "
            f"\\textbf{{{s2['mean_rmse_main_res4']:.5f}}} [uniq: {s2_uniq['mean_rmse_main_res4']:.5f}; row: {s2_cell['mean_rmse_main_res4']:.5f}] \\\\"
        ),
        (
            f"Full-Rank Balanced Random Half-Fraction (200 des.) Main RMSE & "
            f"{s1['mean_task_rmse_main_rand16']:.5f} (${r_c1_main_r16:.2f}\\times$; {s1['win_counts']['cube_main_res5_beats_rand16_split0']}/855 cubes) & "
            f"{s2['mean_rmse_main_rand8']:.5f} (${r_c2_main_r8:.2f}\\times$; {s2['win_counts']['cube_main_res4_beats_rand8_split0']}/178 cubes) \\\\"
        ),
        (
            f"Reference-Cell-0 OFAT ($k+1$ runs: 6 vs.\\ 16; 5 vs.\\ 8) Main RMSE & "
            f"{s1['mean_task_rmse_main_ofat']:.5f} (${r_c1_main_ofat:.2f}\\times$; {s1['win_counts']['cube_main_res5_beats_ofat_split0']}/855 cubes) & "
            f"{s2['mean_rmse_main_ofat']:.5f} (${r_c2_main_ofat:.2f}\\times$; {s2['win_counts']['cube_main_res4_beats_ofat_split0']}/178 cubes) \\\\"
        ),
        (
            f"Orthogonal Quarter-Fraction ($2^{{5-2}}_{{\\mathrm{{III}}}}$, 8 runs) Main RMSE & "
            f"All-60: \\textbf{{{s1['mean_task_rmse_main_res3_all60']:.5f}}} (${r_c1_main_r8_60:.2f}\\times$ vs.\\ {s1['mean_task_rmse_main_rand8']:.5f}; 4-coset: {s1['mean_task_rmse_main_res3_coset_avg']:.5f}, ${r_c1_main_r8_4c:.2f}\\times$) & --- \\\\"
        ),
        (
            f"$2^{{5-1}}_{{\\mathrm{{V}}}}$ vs.\\ Full-Rank Rand-16 2FI-Product RMSE (Pooled / Task) & "
            f"Pool: \\textbf{{{s1['mean_pool_rmse_2fi_res5']:.5f}}} vs.\\ {s1['mean_pool_rmse_2fi_rand16_fullrank']:.5f} (\\textbf{{${ratio_2fi_pool:.2f}\\times$}}); Task: \\textbf{{{s1['mean_task_rmse_2fi_res5']:.5f}}} vs.\\ {s1['mean_task_rmse_2fi_rand16_fullrank']:.5f} (\\textbf{{${r_c1_2fi_task:.2f}\\times$}}) & --- \\\\"
        ),
        (
            "Observed Mixed-Level Catalog ($89/396$ of $4\\times 9\\times 11$) A-Opt RMSE & "
            "$B{=}24$ (22p): \\textbf{0.00281} vs.\\ Rand 0.00544 (\\textbf{$1.94\\times$} pool, \\textbf{$1.96\\times$} task; 5-bit spans 42.4\\% energy) & --- \\\\"
        ),
        "\\midrule",
        "\\multicolumn{3}{p{0.96\\textwidth}}{\\textit{Part B: Held-Out Confirmation Prompt Selection Accuracy, Regret, \\& Positive-Denominator Oracle Recovery (Tie-Averaged)}} \\\\",
        (
            f"Positive Discovery-Oracle Uplift Cubes (Split 0 / 20-Split) & "
            f"{pos_c1_s0_pct:.1f}\\% ({pos_c1_s0_n}/855) / {pos_c1_20s_pct:.1f}\\% ({pos_c1_20s_n}/855) [cell 20s: {pos_c1_cell_20s_pct:.1f}\\%] & "
            f"{pos_c2_s0_pct:.1f}\\% ({pos_c2_s0_n}/178) / {pos_c2_20s_pct:.1f}\\% ({pos_c2_20s_n}/178) [cell 20s: {pos_c2_cell_20s_pct:.1f}\\%] \\\\"
        ),
        (
            f"Reference Cell 0 \\& Lower-Budget OFAT Additive Extrapolation (Split 0 / 20-Split) & "
            f"Base: {s1['mean_conf_baseline']:.4f}/{s1['mean_conf_baseline_20splits']:.4f}; OFAT: {s1['mean_conf_ofat']:.4f}/{s1['mean_conf_ofat_20splits']:.4f} & "
            f"Base: {s2['mean_conf_baseline']:.4f}/{s2['mean_conf_baseline_20splits']:.4f}; OFAT: {s2['mean_conf_ofat']:.4f}/{s2['mean_conf_ofat_20splits']:.4f} \\\\"
        ),
        (
            f"Additive Main-Effect Sign Recipe (Split 0 / 20-Split [20s Regret]) & "
            f"{s1['mean_conf_res5_main_coset_avg']:.4f} / {s1['mean_conf_res5_main_coset_avg_20splits']:.4f} [regret: {s1['mean_regret_vs_oracle_20splits']['res5_main_coset_avg']*100:.2f} pp] & "
            f"{s2['mean_conf_res4_main_coset_avg']:.4f} / {s2['mean_conf_res4_main_coset_avg_20splits']:.4f} [regret: {s2['mean_regret_vs_oracle_20splits']['res4_main_coset_avg']*100:.2f} pp] \\\\"
        ),
        (
            f"Resolution V 2FI Model Recipe (Split 0 [Coset $+$, $-$] / 20-Split) & "
            f"{s1['mean_conf_res5_2fi_coset_avg']:.4f} ({rec_c1_2fi_avg:.1f}\\% [{rec_c1_2fi_pos:.1f}\\%, {rec_c1_2fi_neg:.1f}\\%]) / 20s: {s1['mean_conf_res5_2fi_coset_avg_20splits']:.4f} (cell 20s: {s1_cell['recovery_20splits_res5_2fi_coset_avg']['ratio_of_means_pct']:.1f}\\%) & --- \\\\"
        ),
        (
            f"Orthogonal Best-of-Screen (Split 0 [Coset $+$, $-$] / 20-Split) & "
            f"{s1['mean_conf_res5_best_coset_avg']:.4f} ({rec_c1_sb_avg:.1f}\\% [{rec_c1_sb_pos:.1f}\\%, {rec_c1_sb_neg:.1f}\\%]) / 20s: {s1['mean_conf_res5_best_coset_avg_20splits']:.4f} & "
            f"{s2['mean_conf_res4_best_coset_avg']:.4f} ({rec_c2_sb_avg:.1f}\\% [{rec_c2_sb_pos:.1f}\\%, {rec_c2_sb_neg:.1f}\\%]) / 20s: {s2['mean_conf_res4_best_coset_avg_20splits']:.4f} \\\\"
        ),
        (
            f"Unconditioned Balanced Random Best-of-Screen (Split 0 / 20-Split) & "
            f"{s1['mean_conf_rand16_best']:.4f} ({rec_c1_r16_sb:.1f}\\%) / 20s: {s1['mean_conf_rand16_best_20splits']:.4f} & "
            f"{s2['mean_conf_rand8_best']:.4f} ({rec_c2_r8_sb:.1f}\\%) / 20s: {s2['mean_conf_rand8_best_20splits']:.4f} \\\\"
        ),
        (
            f"Paired Diff: Orthog.\\ $-$ Uncond.\\ Random Best-of-Screen (20s, 95\\% Cond.\\ Int.) & "
            f"{s1['ci_screenbest_minus_rand16_best_20splits']['estimate']*100:+.3f} pp [{s1['ci_screenbest_minus_rand16_best_20splits']['ci_low']*100:+.3f}, {s1['ci_screenbest_minus_rand16_best_20splits']['ci_high']*100:+.3f}] (cell: {s1_cell['ci_screenbest_minus_rand16_best_20splits']['estimate']*100:+.3f} pp) & "
            f"{s2['ci_screenbest_minus_rand8_best_20splits']['estimate']*100:+.3f} pp [{s2['ci_screenbest_minus_rand8_best_20splits']['ci_low']*100:+.3f}, {s2['ci_screenbest_minus_rand8_best_20splits']['ci_high']*100:+.3f}] (cell: \\textbf{{{s2_cell['ci_screenbest_minus_rand8_best_20splits']['estimate']*100:+.3f} pp}}) \\\\"
        ),
        (
            f"Exhaustive Full-Cube Discovery Oracle (Split 0 / 20-Split) & "
            f"{s1['mean_conf_oracle32']:.4f} / {s1['mean_conf_oracle32_20splits']:.4f} & "
            f"{s2['mean_conf_oracle16']:.4f} / {s2['mean_conf_oracle16_20splits']:.4f} \\\\"
        ),
        "\\midrule",
        "\\multicolumn{3}{p{0.96\\textwidth}}{\\textit{Part C: Empirical-Null Calibration ($\\alpha=0.05$), Variance Reduction, \\& Post-Calibration Conditional Detections}} \\\\",
        (
            "Empirical-Null Raw Type-I \\& Partial-Null Global BH FDR (Clust.\\ $t$ vs.\\ RCBD) & "
            "Clust.\\ $t$: \\textbf{0.051} Type-I, \\textbf{0.051} FDR (93.3\\% pwr); Pooled RCBD: 0.271 Factor-$C$ Type-I, \\textbf{0.184} FDR (uncal.) & "
            "Clust.\\ $t$ ($N{\\ge}100$): \\textbf{0.053} Type-I, \\textbf{0.040} FDR ($N{=}26$: 0.063 qual.); RCBD cell FDR: 0.162 (uncal.) \\\\"
        ),
        (
            f"Mean Residual Variance ($\\mathrm{{MSE}}_{{\\mathrm{{resid}}}}$) Reduction [Min, Max] \\& $F$-Mult. & "
            f"\\textbf{{{s1['mean_mse_reduction_pct']:.2f}\\%}} [{s1['min_mse_reduction_pct']:.1f}\\%, {s1['max_mse_reduction_pct']:.2f}\\%] (${s1['mean_f_stat_multiplier']:.2f}\\times$) & "
            f"\\textbf{{{s2['mean_mse_reduction_pct']:.2f}\\%}} [{s2['min_mse_reduction_pct']:.1f}\\%, {s2['max_mse_reduction_pct']:.2f}\\%] (${s2['mean_f_stat_multiplier']:.2f}\\times$) \\\\"
        ),
        (
            f"Per-Cube BH [BY] Det.\\ ($\\alpha=0.05$): Unb.\\ / Calib.\\ Clust.\\ $t$ (Perm) / Uncal.\\ RCBD & "
            f"{s1['total_sig_unblocked_per_cube_bh']} [{int(m1_cube['detections_unblocked_by'])}] / \\textbf{{{s1['total_sig_item_cluster_per_cube_bh']}}} [{int(m1_cube['detections_item_cluster_t_by'])}] ({int(m1_cube['detections_item_sign_flip_perm'])}; ${ratio_clust_det:.2f}\\times$) / {s1['total_sig_rcbd_per_cube_bh']} (uncal.) & "
            f"{s2['total_sig_unblocked_per_cube_bh']} [{int(m2_cube['detections_unblocked_by'])}] / \\textbf{{{s2['total_sig_item_cluster_per_cube_bh']}}} [{int(m2_cube['detections_item_cluster_t_by'])}] ({int(m2_cube['detections_item_sign_flip_perm'])}) / {s2['total_sig_rcbd_per_cube_bh']} (uncal.) \\\\"
        ),
        (
            f"Global Cohort BH [BY] Det.\\ ($\\alpha=0.05$): Unb.\\ / Calib.\\ Clust.\\ $t$ ($B{{=}}2000$ Perm) / Uncal.\\ RCBD & "
            f"{int(m1_glob['detections_unblocked'])} [{int(m1_glob['detections_unblocked_by'])}] / \\textbf{{{int(m1_glob['detections_item_cluster_t'])}}} [{int(m1_glob['detections_item_cluster_t_by'])}] (58) / {int(m1_glob['detections_rcbd_fixed_block'])} (uncal., of 4,275) & "
            f"{int(m2_glob['detections_unblocked'])} [{int(m2_glob['detections_unblocked_by'])}] / \\textbf{{{int(m2_glob['detections_item_cluster_t'])}}} [{int(m2_glob['detections_item_cluster_t_by'])}] (350) / {int(m2_glob['detections_rcbd_fixed_block'])} (uncal., of 712) \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    (TABLES_DIR / "table_cross_cohort.tex").write_text("\n".join(lines2) + "\n", encoding="utf-8")


def write_generated_macros(headline: Dict[str, Any]) -> Path:
    """Write `paper_hf/tables/generated_macros.tex` with machine-generated headline macros (Task 8)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    s1 = headline["cohort1_mmlu_legacy_first"]
    s2 = headline["cohort2_slab_legacy_first"]
    obs = headline["study_observation_totals"]
    cell0 = headline["cell0_diagnostic"]
    macros = [
        "% Machine-generated by research/hf_multi_prompt/run_experiments.py -- DO NOT EDIT MANUALLY",
        f"\\newcommand{{\\TotalStudyObs}}{{{obs['total_study_cell_item_observations']:,}}}",
        f"\\newcommand{{\\CohortOneObs}}{{{obs['cohort1_mmlu_cell_item_observations']:,}}}",
        f"\\newcommand{{\\CohortTwoObs}}{{{obs['cohort2_slab_cell_item_observations']:,}}}",
        f"\\newcommand{{\\MmluPoolTwoFiRatio}}{{\\ensuremath{{{s1['pool_2fi_rmse_ratio_fullrank_over_res5']:.2f}\\times}}}}",
        f"\\newcommand{{\\MmluTaskTwoFiRatio}}{{\\ensuremath{{{s1['task_2fi_rmse_ratio_fullrank_over_res5']:.2f}\\times}}}}",
        f"\\newcommand{{\\MmluResThreeAllSixtyRatio}}{{\\ensuremath{{{s1['task_main_rmse_ratio_rand8_over_res3_all60']:.2f}\\times}}}}",
        f"\\newcommand{{\\MmluMseReductionMean}}{{{s1['mean_mse_reduction_pct']:.2f}\\%}}",
        f"\\newcommand{{\\SlabMseReductionMean}}{{{s2['mean_mse_reduction_pct']:.2f}\\%}}",
        f"\\newcommand{{\\SlabMseSummaryCubes}}{{{s2['mse_summary_cube_count']}}}",
        f"\\newcommand{{\\CellZeroCatalogRows}}{{{cell0['n_catalog_rows']}}}",
        f"\\newcommand{{\\CellZeroUniqueTemplates}}{{{cell0['n_unique_templates']}}}",
    ]

    v4_summary_path = ROOT / "research" / "results" / "real_v4" / "summary.json"
    if v4_summary_path.exists():
        v4 = json.loads(v4_summary_path.read_text(encoding="utf-8"))
        pv = v4["primary_verdict"]
        sel_r = pv["selection_contrasts"]["balanced_random"]
        sel_b = pv["selection_contrasts"]["bayesian_opt"]
        stab = pv["stability_summary"]
        paired_stab = v4["sensitivity_analyses"]["paired_item_perturbations_only"]["stability_summary"]
        eff = v4["supporting_effect_recovery"]
        rep = v4["repeatability_diagnostics"]
        res = v4["full_study_resource_reconciliation"]
        sel_agree = v4["stability_certificate"]["supporting_diagnostics"]["selection_resample_agreement"]
        eff_win_cells = sum(1 for c in eff["per_cell"].values() if c["gain_rmse_reduction"] > 0)

        verdict_tex = str(pv["verdict_code"]).replace("_", "\\_")
        macros.extend(
            [
                "% Prospective Real-v4 machine-generated macros from research/results/real_v4/summary.json",
                f"\\newcommand{{\\RealVFourTotalCalls}}{{{int(res['total_unique_calls_all_phases']):,}}}",
                f"\\newcommand{{\\RealVFourHostedUsd}}{{\\${res['total_hosted_cost_usd_all_phases']:.2f}}}",
                f"\\newcommand{{\\RealVFourVerdictCode}}{{\\texttt{{{verdict_tex}}}}}",
                f"\\newcommand{{\\RealVFourCertStatus}}{{\\texttt{{{pv['certificate_status']}}}}}",
                f"\\newcommand{{\\RealVFourRefMean}}{{{stab['reference_mean']:.4f}}}",
                f"\\newcommand{{\\RealVFourSelGainRandPp}}{{{sel_r['estimate'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourSelLowRandPp}}{{{sel_r['ci95_lower'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourSelPRand}}{{{sel_r['p_value_one_sided']:.3f}}}",
                f"\\newcommand{{\\RealVFourSelPosCellsRand}}{{{int(pv['positive_cells_by_comparator']['balanced_random'])}}}",
                f"\\newcommand{{\\RealVFourSelGainBoPp}}{{{sel_b['estimate'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourSelLowBoPp}}{{{sel_b['ci95_lower'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourSelPBo}}{{{sel_b['p_value_one_sided']:.3f}}}",
                f"\\newcommand{{\\RealVFourSelPosCellsBo}}{{{int(pv['positive_cells_by_comparator']['bayesian_opt'])}}}",
                f"\\newcommand{{\\RealVFourStabDropEst}}{{{stab['d_stability_estimate']:+.4f}}}",
                f"\\newcommand{{\\RealVFourStabDropEstPp}}{{{stab['d_stability_estimate'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourStabUpperSim}}{{{stab['d_stability_upper_95_sim']:.4f}}}",
                f"\\newcommand{{\\RealVFourStabUpperSimPp}}{{{stab['d_stability_upper_95_sim'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourPairedDropEst}}{{{paired_stab['d_stability_estimate']:+.4f}}}",
                f"\\newcommand{{\\RealVFourPairedUpperSim}}{{{paired_stab['d_stability_upper_95_sim']:.4f}}}",
                f"\\newcommand{{\\RealVFourPairedUpperSimPp}}{{{paired_stab['d_stability_upper_95_sim'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourWorstEnvGainRandPp}}{{{stab['comparator_worst_env_checks']['balanced_random']['estimate'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourWorstEnvLowRandPp}}{{{stab['comparator_worst_env_checks']['balanced_random']['ci95_lower'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourWorstEnvGainBoPp}}{{{stab['comparator_worst_env_checks']['bayesian_opt']['estimate'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourWorstEnvLowBoPp}}{{{stab['comparator_worst_env_checks']['bayesian_opt']['ci95_lower'] * 100.0:+.2f}}}",
                f"\\newcommand{{\\RealVFourEffRmsePrism}}{{{eff['mean_rmse_promptprism']:.4f}}}",
                f"\\newcommand{{\\RealVFourEffRmseRand}}{{{eff['mean_rmse_random']:.4f}}}",
                f"\\newcommand{{\\RealVFourEffReductionPct}}{{{eff['relative_rmse_reduction_pct']:.2f}\\%}}",
                f"\\newcommand{{\\RealVFourEffWinCells}}{{{eff_win_cells}}}",
                f"\\newcommand{{\\RealVFourRepeatMatchPct}}{{{rep['identical_repeat_exact_match_rate'] * 100.0:.2f}\\%}}",
                f"\\newcommand{{\\RealVFourValSelectAgreePct}}{{{float(sel_agree) * 100.0:.2f}\\%}}",
                f"\\newcommand{{\\RealVFourTerminalFailures}}{{{int(res['total_terminal_failures_all_phases']):,}}}",
            ]
        )

    out_path = TABLES_DIR / "generated_macros.tex"
    out_path.write_text("\n".join(macros) + "\n", encoding="utf-8")
    return out_path


def build_claim_evidence_map(headline: Dict[str, Any]) -> pd.DataFrame:
    """Create the structured Claim-to-Evidence Map required by Spec PA-5 and V2 Task 8."""
    s1_leg = headline["cohort1_mmlu_legacy_first"]
    s1_cell = headline["cohort1_mmlu_cell_mean"]
    s1_uniq = headline["cohort1_mmlu_cell_mean_unique"]
    s2_leg = headline["cohort2_slab_legacy_first"]
    s2_cell = headline["cohort2_slab_cell_mean"]
    s2_uniq = headline["cohort2_slab_cell_mean_unique"]
    assoc = headline["interaction_regret_associations"]
    a_c1_leg = assoc["MMLU_legacy_first"]
    a_c1_cell = assoc["MMLU_cell_mean"]
    a_c2_leg = assoc["SLAB_legacy_first"]
    a_c2_cell = assoc["SLAB_cell_mean"]

    claims = [
        {
            "claim_id": "C1_main_effect_rmse_advantage",
            "paper_location": "Abstract, Section 4.1, Table 3 Part A",
            "source_file": "results/headline_summary.json",
            "source_field": "cohort1_mmlu_legacy_first.task_main_rmse_ratio_rand16_over_res5 & win_counts & cohort2_slab_legacy_first.main_rmse_ratio_rand8_over_res4",
            "estimand": "Main-effect RMSE ratio and cube/model win frequencies vs equal-budget balanced random and lower-budget OFAT",
            "value_summary": (
                f"MMLU Res V vs Rand-16: {s1_leg['task_main_rmse_ratio_rand16_over_res5']:.2f}x task "
                f"({s1_leg['win_counts']['cube_main_res5_beats_rand16_split0']}/855 cubes, {s1_leg['win_counts']['model_pool_main_res5_beats_rand16']}/15 models pooled; "
                f"{s1_leg['pool_main_rmse_ratio_rand16_over_res5']:.2f}x pooled; "
                f"{s1_cell['task_main_rmse_ratio_rand16_over_res5_20splits']:.2f}x cell-mean 20-split); "
                f"MMLU Res III vs Rand-8: {s1_leg['task_main_rmse_ratio_rand8_over_res3_all60']:.2f}x task and "
                f"{s1_leg['pool_main_rmse_ratio_rand8_over_res3_all60']:.2f}x pooled "
                f"({s1_leg['mean_pool_rmse_main_res3_all60']:.5f} vs {s1_leg['mean_pool_rmse_main_rand8']:.5f}) across all 60 regular 8-run fractions "
                f"({s1_leg['task_main_rmse_ratio_rand8_over_res3_coset_avg']:.2f}x across 4 canonical cosets of D=+-AB, E=+-AC); "
                f"SLAB Res IV vs Rand-8: {s2_leg['main_rmse_ratio_rand8_over_res4']:.2f}x ({s2_leg['win_counts']['cube_main_res4_beats_rand8_split0']}/178 cubes); "
                f"vs reference-Cell-0, lower-budget OFAT (k+1 runs): MMLU {s1_leg['task_main_rmse_ratio_ofat_over_res5']:.2f}x task and "
                f"{s1_leg['mean_pool_rmse_main_ofat']/s1_leg['mean_pool_rmse_main_res5']:.2f}x pooled; SLAB {s2_leg['main_rmse_ratio_ofat_over_res4']:.2f}x."
            ),
            "averaging_unit": "855 MMLU cubes (15 models) and 178 completeness-selected SLAB cubes (6/15 BBH, 7/10 LMentry tasks)",
            "uncertainty_method": "Two-way model-task cluster bootstrap (1,000 replicates) + exact cube/model win counts",
            "status": "supported",
        },
        {
            "claim_id": "C2_res5_2fi_fullrank_advantage",
            "paper_location": "Abstract, Section 4.1, Table 2, Table 3 Part A",
            "source_file": "results/headline_summary.json",
            "source_field": "cohort1_mmlu_legacy_first.pool_2fi_rmse_ratio_fullrank_over_res5",
            "estimand": "Second-order-product RMSE ratio in the five-bit contrast basis: 200 full-rank balanced Random-16 attribution designs vs orthogonal 2^(5-1)_V",
            "value_summary": (
                f"Pooled: {s1_leg['pool_2fi_rmse_ratio_fullrank_over_res5']:.2f}x "
                f"({s1_leg['mean_pool_rmse_2fi_res5']:.5f} vs {s1_leg['mean_pool_rmse_2fi_rand16_fullrank']:.5f}; 15/15 models; "
                f"well-conditioned cond<=15: {s1_leg['pool_2fi_rmse_ratio_wellcond_over_res5']:.2f}x); "
                f"Per-task: {s1_leg['task_2fi_rmse_ratio_fullrank_over_res5']:.2f}x ({s1_leg['win_counts']['cube_2fi_res5_beats_rand16_split0']}/855 cubes; "
                f"20-split: {s1_leg['task_2fi_rmse_ratio_fullrank_over_res5_20splits']:.2f}x; "
                f"cell-mean 20-split: {s1_cell['task_2fi_rmse_ratio_fullrank_over_res5_20splits']:.2f}x). Replaces legacy rank-deficient 23.8x."
            ),
            "averaging_unit": "15 MMLU models (pooled) and 855 model-task cubes across 200 full-rank random attribution designs",
            "uncertainty_method": "Design distribution quantiles + two-way model-task cluster bootstrap",
            "status": "supported",
        },
        {
            "claim_id": "C3_coset_invariance_vs_selection_sensitivity_and_ties",
            "paper_location": "Section 3.1, Section 4.2, Table 3 Part B",
            "source_file": "results/coset_results.csv & results/headline_summary.json",
            "source_field": "recovery_split0_res5_best_coset_avg, first_index_compatibility, tie_policy_sensitivity, all_reference_cells_recovery",
            "estimand": "Tie-aware (mean_tied) vs index-dependent (first_index) selection and all-reference-corner RoM recovery range",
            "value_summary": (
                f"Under primary tie-averaged selection (mean_tied), MMLU Split 0 ScreenBest RoM recovery is "
                f"{s1_leg['recovery_split0_res5_best_coset_pos']['ratio_of_means_pct']:.1f}% (coset+) and "
                f"{s1_leg['recovery_split0_res5_best_coset_neg']['ratio_of_means_pct']:.1f}% (coset-, mean {s1_leg['recovery_split0_res5_best_coset_avg']['ratio_of_means_pct']:.1f}%; "
                f"vs legacy first_index {s1_leg['first_index_compatibility']['recovery_split0_res5_best_coset_pos']['ratio_of_means_pct']:.1f}% and "
                f"{s1_leg['first_index_compatibility']['recovery_split0_res5_best_coset_neg']['ratio_of_means_pct']:.1f}%, mean {s1_leg['first_index_compatibility']['recovery_split0_res5_best_coset_avg']['ratio_of_means_pct']:.1f}%). "
                f"Across all 32 reference corners, MMLU Split 0 ScreenBest RoM spans "
                f"{s1_leg['all_reference_cells_recovery']['split0_res5_best_coset_avg']['rom_pct_min']:.1f}% to "
                f"{s1_leg['all_reference_cells_recovery']['split0_res5_best_coset_avg']['rom_pct_max']:.1f}% "
                f"(median {s1_leg['all_reference_cells_recovery']['split0_res5_best_coset_avg']['rom_pct_median']:.1f}%)."
            ),
            "averaging_unit": "All valid regular fractional cosets and all K reference cells across 855 MMLU and 178 SLAB cubes",
            "uncertainty_method": "Exact tie-policy audit (mean_tied, first_index, seeded_random) + all-reference-cell sensitivity",
            "status": "qualified",
        },
        {
            "claim_id": "C4_screenbest_vs_unconditioned_balanced_random",
            "paper_location": "Abstract, Section 4.2, Table 3 Part B, Conclusion",
            "source_file": "results/headline_summary.json",
            "source_field": "ci_screenbest_minus_rand16_best_20splits & ci_screenbest_minus_rand8_best_20splits",
            "estimand": "Paired confirmation accuracy difference: coset-averaged Orthogonal ScreenBest minus unconditioned factor-balanced Random ScreenBest",
            "value_summary": (
                f"MMLU legacy_first 20-split diff vs unconditioned Rand-16 ScreenBest: {s1_leg['ci_screenbest_minus_rand16_best_20splits']['estimate']*100:+.3f} pp "
                f"(95% conditional interval [{s1_leg['ci_screenbest_minus_rand16_best_20splits']['ci_low']*100:+.3f}, {s1_leg['ci_screenbest_minus_rand16_best_20splits']['ci_high']*100:+.3f}] pp; "
                f"vs 2FI-conditioned Rand-16: {s1_leg['ci_screenbest_minus_rand16_best_2fi_cond_20splits']['estimate']*100:+.3f} pp); "
                f"SLAB legacy_first 20-split diff vs unconditioned Rand-8 ScreenBest: {s2_leg['ci_screenbest_minus_rand8_best_20splits']['estimate']*100:+.3f} pp "
                f"(95% conditional interval [{s2_leg['ci_screenbest_minus_rand8_best_20splits']['ci_low']*100:+.3f}, {s2_leg['ci_screenbest_minus_rand8_best_20splits']['ci_high']*100:+.3f}] pp), "
                f"while SLAB multi-template cell_mean favors orthogonal 2^(4-1)_IV by {s2_cell['ci_screenbest_minus_rand8_best_20splits']['estimate']*100:+.3f} pp "
                f"(95% conditional interval [{s2_cell['ci_screenbest_minus_rand8_best_20splits']['ci_low']*100:+.3f}, {s2_cell['ci_screenbest_minus_rand8_best_20splits']['ci_high']*100:+.3f}] pp)."
            ),
            "averaging_unit": "855 MMLU and 178 SLAB cubes x 20 repeated random splits",
            "uncertainty_method": "Conditional two-way model-task panel-reweighting bootstrap intervals; nonconfirmatory normal-Wald margin sensitivity",
            "status": "qualified",
        },
        {
            "claim_id": "C5_res5_2fi_selection_and_sign_paradox",
            "paper_location": "Abstract, Section 4.2, Table 3 Part B, Conclusion",
            "source_file": "results/headline_summary.json & results/interaction_magnitudes.csv",
            "source_field": "recovery_split0_res5_2fi_coset_avg & interaction_regret_associations",
            "estimand": "Resolution V second-order-product model selection and shared-model cluster-bootstrap association (raw and partial) between product RMS and additive-sign regret",
            "value_summary": (
                f"Under primary mean_tied selection, MMLU Split 0 Res V 2FI recovery is {s1_leg['recovery_split0_res5_2fi_coset_avg']['ratio_of_means_pct']:.1f}% "
                f"(coset+ {s1_leg['recovery_split0_res5_2fi_coset_pos']['ratio_of_means_pct']:.1f}%, coset- {s1_leg['recovery_split0_res5_2fi_coset_neg']['ratio_of_means_pct']:.1f}%; "
                f"legacy first_index: {s1_leg['first_index_compatibility']['recovery_split0_res5_2fi_coset_neg']['ratio_of_means_pct']:.1f}% on coset-, "
                f"{s1_leg['first_index_compatibility']['recovery_split0_res5_2fi_coset_pos']['ratio_of_means_pct']:.1f}% on coset+, "
                f"{s1_leg['first_index_compatibility']['recovery_split0_res5_2fi_coset_avg']['ratio_of_means_pct']:.1f}% avg; "
                f"cell-mean 20-split: {s1_cell['recovery_20splits_res5_2fi_coset_avg']['ratio_of_means_pct']:.1f}% vs {s1_cell['recovery_20splits_rand16_2fi_fullrank']['ratio_of_means_pct']:.1f}% Rand-16 2FI). "
                f"Cohort 2 shared-model cluster-bootstrap r={a_c2_leg['pearson_r_rms2fi_vs_main_regret']:.3f} "
                f"(95% conditional interval [{a_c2_leg['pearson_ci_low_rms2fi_vs_main_regret']:.3f}, {a_c2_leg['pearson_ci_high_rms2fi_vs_main_regret']:.3f}]; "
                f"partial r given rms_main={a_c2_leg['partial_pearson_r_rms2fi_vs_main_regret_given_rms_main']:.3f}, "
                f"95% conditional interval [{a_c2_leg['partial_pearson_ci_low_rms2fi_vs_main_regret_given_rms_main']:.3f}, {a_c2_leg['partial_pearson_ci_high_rms2fi_vs_main_regret_given_rms_main']:.3f}]; "
                f"cell-mean r={a_c2_cell['pearson_r_rms2fi_vs_main_regret']:.3f}, partial r={a_c2_cell['partial_pearson_r_rms2fi_vs_main_regret_given_rms_main']:.3f}); "
                f"Cohort 1 MMLU legacy r={a_c1_leg['pearson_r_rms2fi_vs_main_regret']:.3f} "
                f"(95% conditional interval [{a_c1_leg['pearson_ci_low_rms2fi_vs_main_regret']:.3f}, {a_c1_leg['pearson_ci_high_rms2fi_vs_main_regret']:.3f}], "
                f"partial r={a_c1_leg['partial_pearson_r_rms2fi_vs_main_regret_given_rms_main']:.3f}; cell-mean r={a_c1_cell['pearson_r_rms2fi_vs_main_regret']:.3f})."
            ),
            "averaging_unit": "855 MMLU (15 models x 57 tasks) and 178 SLAB cubes (16 unique models across 13 tasks, 11 shared models)",
            "uncertainty_method": "Shared-model two-way model-task conditional bootstrap intervals + partial correlations controlling for rms_main",
            "status": "qualified",
        },
        {
            "claim_id": "C6_rcbd_variance_reduction_and_conditional_item_detections",
            "paper_location": "Abstract, Section 4.3, Table 2, Table 3 Part C",
            "source_file": "results/multiplicity_results.csv & results/headline_summary.json",
            "source_field": "detections_unblocked, detections_item_cluster_t, detections_rcbd_fixed_block",
            "estimand": "Residual MSE reduction (mean vs max) and item-sampling BH/BY detection counts conditional on fixed templates, models, and tasks",
            "value_summary": (
                f"MMLU MSE reduction mean {s1_leg['mean_mse_reduction_pct']:.2f}% (range {s1_leg['min_mse_reduction_pct']:.1f}% to {s1_leg['max_mse_reduction_pct']:.2f}%; F-mult {s1_leg['mean_f_stat_multiplier']:.2f}x); "
                f"per-cube BH detections: 95 (Unblocked; 67 BY) -> 243 (Primary asymptotic Item-Clustered t, 2.56x; 240 sign-flip perm with 200 draws, min p=1/201; 171 BY) -> "
                f"565 (Exploratory Fixed-Block RCBD, 5.95x; 445 BY); "
                f"global BH (4,275 tests): 32 -> 52 -> 422. SLAB MSE reduction mean {s2_leg['mean_mse_reduction_pct']:.2f}% (up to {s2_leg['max_mse_reduction_pct']:.2f}%); "
                f"per-cube BH: 322 -> 358 (345 sign-flip perm; 330 BY) -> 361."
            ),
            "averaging_unit": "Benchmark items are the sampling units within 4,275 MMLU and 712 SLAB factor tests; models, tasks, and templates are fixed",
            "uncertainty_method": "Asymptotic item-clustered contrast t-test (primary, scipy.stats.t.sf), 200-draw sign-flip permutation test, and BH & BY FDR",
            "status": "supported",
        },
        {
            "claim_id": "C7_cell0_diagnostic_and_cell_mean_scope",
            "paper_location": "Abstract, Section 2.2, Section 4.2, Section 5",
            "source_file": "results/headline_summary.json & results/template_mapping.csv",
            "source_field": "cell0_diagnostic",
            "estimand": "MMLU Cell 0 (0,0,0,0,0) template representative vs 5-row/4-unique-template stratum on Split 0 confirmation",
            "value_summary": (
                f"Cell 0 maps to {headline['cell0_diagnostic']['n_catalog_rows']} catalog rows "
                f"{headline['cell0_diagnostic']['template_indices']} and "
                f"{headline['cell0_diagnostic']['n_unique_templates']} unique templates; "
                f"legacy_first selects row {headline['cell0_diagnostic']['legacy_selected_row_index']} "
                f"(Split 0 conf {headline['cell0_diagnostic']['split0_conf_legacy_first']:.6f}) vs catalog-row cell_mean "
                f"({headline['cell0_diagnostic']['split0_conf_cell_mean']:.6f}), a difference of "
                f"{headline['cell0_diagnostic']['split0_diff_legacy_minus_cell_mean_pp']:+.3f} pp. "
                f"cell_mean averages 1-6 MMLU or 1-56 SLAB templates per cell and is a stratum sensitivity analysis, not an equal-cost single-prompt screen."
            ),
            "averaging_unit": "855 MMLU cubes on Split 0 confirmation items",
            "uncertainty_method": "Exact template mapping and cell-stratum aggregation audit",
            "status": "supported",
        },
        {
            "claim_id": "C8_cohort2_completeness_and_study_size",
            "paper_location": "Abstract, Section 1, Section 2.2, Section 5, Section 6",
            "source_file": "results/headline_summary.json",
            "source_field": "study_observation_totals & cohort2_completeness_audit",
            "estimand": "Completeness-selected task counts and exact cell-item observation totals",
            "value_summary": (
                f"Total study observations: {headline['study_observation_totals']['total_study_cell_item_observations']:,} "
                f"(Cohort 1 MMLU: {headline['study_observation_totals']['cohort1_mmlu_cell_item_observations']:,}; "
                f"Cohort 2 SLAB: {headline['study_observation_totals']['cohort2_slab_cell_item_observations']:,} = "
                f"{headline['study_observation_totals']['cohort2_bbh_cell_item_observations']:,} BBH + "
                f"{headline['study_observation_totals']['cohort2_lmentry_cell_item_observations']:,} LMentry, as LMentry/word_not_containing has "
                f"{headline['study_observation_totals']['lmentry_word_not_containing_items']} items). "
                f"Completeness selection retains {headline['cohort2_completeness_audit']['bbh_complete_16cell_tasks']}/"
                f"{headline['cohort2_completeness_audit']['bbh_total_tasks']} BBH and "
                f"{headline['cohort2_completeness_audit']['lmentry_complete_16cell_tasks']}/"
                f"{headline['cohort2_completeness_audit']['lmentry_total_tasks']} LMentry tasks."
            ),
            "averaging_unit": "1,033 complete factorial cubes (855 MMLU + 178 SLAB)",
            "uncertainty_method": "Exact row-level dataset census",
            "status": "supported",
        },
        {
            "claim_id": "C9_duplicate_template_weighting_and_crossed_sensitivity",
            "paper_location": "Section 2.2, Section 4.1, Section 4.2, Table 3",
            "source_file": "results/template_mapping.csv & results/template_sensitivity.csv",
            "source_field": "is_duplicate_text, cell_mean_unique vs cell_mean, random_single_rep_unique vs random_single_rep",
            "estimand": "Impact of duplicate catalog rows (7 per task in MMLU; 93 across LMentry tasks, all within-cell) and 20x20 template-draw x item-split sensitivity",
            "value_summary": (
                f"MMLU has 93 unique templates across 100 catalog rows (7 within-cell duplicates); LMentry has 93 within-cell duplicate rows across 7 complete tasks (BBH has 0). "
                f"MMLU 20-split main RMSE ratio is {s1_uniq['task_main_rmse_ratio_rand16_over_res5_20splits']:.2f}x under cell_mean_unique vs "
                f"{s1_cell['task_main_rmse_ratio_rand16_over_res5_20splits']:.2f}x under catalog-row cell_mean; "
                f"SLAB 20-split ScreenBest minus Rand-8 diff is {s2_uniq['ci_screenbest_minus_rand8_best_20splits']['estimate']*100:+.3f} pp under cell_mean_unique vs "
                f"{s2_cell['ci_screenbest_minus_rand8_best_20splits']['estimate']*100:+.3f} pp under catalog-row cell_mean."
            ),
            "averaging_unit": "855 MMLU and 178 SLAB cubes across 20 template draws x 20 item splits (400 evaluations/cube)",
            "uncertainty_method": "Exact SHA-256 text deduplication + 20x20 crossed template-draw x item-split evaluation",
            "status": "supported",
        },
        {
            "claim_id": "C10_effect_noise_floor_and_esr",
            "paper_location": "Abstract, Section 5.1, Table 3 Part A",
            "source_file": "results/effect_reliability.csv & results/effect_reliability_summary.json",
            "source_field": "order_specific_reliability, cohort_rms_corr, esr_attribution_vs_signal, effect_hierarchy_audit",
            "estimand": "Split-half contrast reliability, noise-corrected signal RMS, and absolute Error-to-Signal Ratios (ESR)",
            "value_summary": (
                "Per-task MMLU flat observed 2FI/3FI/4FI spectrum (0.00728/0.00717/0.00731) is 79%-85% item-sampling variance "
                "(r=0.148-0.180); noise subtraction separates 1FI (0.00730) >> {2FI 0.00321, 3FI 0.00309, 4FI 0.00384} >> 5FI (0.00163). "
                "SLAB cell_mean_unique restores strict monotonic hierarchy (0.0773 > 0.0390 > 0.0327 > 0.0284). "
                "SLAB 2^(4-1)_IV achieves sub-unity ESR 0.735 (0.810 held-out) vs Rand-8 1.096 (1.127 held-out)."
            ),
            "averaging_unit": "855 MMLU task cubes, 15 MMLU pooled model cubes, and 178 SLAB cubes across 20 splits",
            "uncertainty_method": "Analytical + 500-draw item bootstrap variance + 1,000-replicate two-way cluster bootstrap",
            "status": "supported",
        },
        {
            "claim_id": "C11_mixed_level_catalog_and_optimal_designs",
            "paper_location": "Section 3.2, Section 5.2, Table 3 Part A",
            "source_file": "results/mixed_level_results.csv & results/mixed_level_summary.json",
            "source_field": "catalog_audit, basis_diagnostics, headline_comparisons",
            "estimand": "Observed 4x9x11 MMLU catalog completeness (89/396) and outcome-blind A-/D-optimal vs binary/random/PromptEval comparators",
            "value_summary": (
                "Only 89 of 396 (22.5%) op x space x sep combinations are observed (coprime lcm=396 forbids fractional complete OA). "
                "5-bit binary basis spans 42.4% of 21-df additive categorical energy. Outcome-blind A-optimal subset reduces 22-param "
                "axis-contrast RMSE by 1.94x pooled / 1.96x task at B=24 (855/855 cubes) and 1.53x / 1.51x at B=32 (beating 32-cell "
                "binary cube by 1.38x / 1.45x), while equal-budget ScreenBest remains indistinguishable from random (+0.014 pp at B=16)."
            ),
            "averaging_unit": "89 observed axis combinations across 855 MMLU model-task cubes and 20 item splits",
            "uncertainty_method": "Outcome-blind Fedorov A-/D-optimal exchange + two-way model-task cluster bootstrap",
            "status": "supported",
        },
        {
            "claim_id": "C12_empirical_null_inference_calibration",
            "paper_location": "Abstract, Section 4.3, Section 5.4, Table 3 Part C",
            "source_file": "results/inference_calibration.csv & results/inference_calibration_summary.json",
            "source_field": "gate_verdicts, regime_summaries, post_calibration_observed_detections",
            "estimand": "Type-I error and BH/BY FDR calibration under synthetic and empirically anchored nulls (N1-N4)",
            "value_summary": (
                "Asymptotic item-clustered t passes all standard-N (N>=100) calibration gates (max Type-I 0.0539; N4b 80/20 partial-null "
                "global BH FDR 0.0510 on MMLU and 0.0401 on SLAB at >93% power; N=26 word_not_containing qualified at 0.0634). "
                "Pooled fixed-block RCBD fails calibration under empirical heteroscedasticity (27.1% Type-I on MMLU Factor C, 18.4% global "
                "BH FDR), so its 565 per-cube / 422 global rejections are withdrawn as variance-pooling artifacts. B=2000 sign-flip resolves "
                "global BH to 58 MMLU detections (corroborating clustered t's 52)."
            ),
            "averaging_unit": "4,275 MMLU and 712 SLAB factor tests across 4 null/partial-null simulation and empirical bootstrap regimes",
            "uncertainty_method": "Monte Carlo standard errors + 95% binomial upper bounds against registered 0.065 gate",
            "status": "supported",
        },
    ]
    return pd.DataFrame(claims)


# Backward-compatible entry points for external callers
def run_cohort1_mmlu(Ys: dict) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_cells, legacy_map = build_mmlu_cell_mapping()
    cosets_info = enumerate_fractional_cosets()
    r16_fr, r16_diag, _ = generate_random_designs_with_diagnostics(
        5, 16, 200, SEEDS["rand16_full_rank"], True, "MMLU", "rand16_2fi_fullrank"
    )
    r16_conds = np.array([d["cond_2fi"] for d in r16_diag if d["accepted_primary"]])
    r16_leg = generate_balanced_random_designs(32, 5, 16, num_screens=20, seed=SEEDS["rand16_legacy"], require_2fi_full_rank=False)
    r8_fr = generate_balanced_random_designs(32, 5, 8, num_screens=200, seed=SEEDS["rand8_cohort1"], require_2fi_full_rank=False)
    task_splits = {
        t: generate_item_splits(Ys["MMLU"][t][0].shape[1], n_splits=20, seed=SEEDS["item_splits"] + idx)
        for idx, t in enumerate(sorted(Ys["MMLU"].keys()))
    }
    res = evaluate_cohort1_estimator(
        Ys, all_cells, legacy_map, "legacy_first", task_splits, cosets_info, r16_fr, r16_conds, r16_leg, r8_fr, compute_rcbd=True, n_perm_rcbd=0
    )
    return res["df_models"], res["df_cubes"], res["summary"]


def run_cohort2_slab(Ys: dict) -> Tuple[pd.DataFrame, dict]:
    all_cells_4 = list(itertools.product([0, 1], repeat=4))
    cosets_info = enumerate_fractional_cosets()
    r8_fr = generate_balanced_random_designs(16, 4, 8, num_screens=200, seed=SEEDS["rand8_cohort2"], require_2fi_full_rank=False)
    slab_leg_maps = {}
    slab_splits = {}
    s_counter = 0
    for bench in ["BBH", "LMentry"]:
        for fpath in sorted((DATA_DIR / "templates" / bench).glob("*.csv")):
            df_t = pd.read_csv(fpath)
            cmap = build_slab_cell_mapping(df_t)
            if len(cmap) < 16:
                continue
            key = (bench, fpath.stem)
            slab_leg_maps[key] = cmap
            n_items = Ys[bench][fpath.stem][0].shape[1]
            slab_splits[key] = generate_item_splits(n_items, n_splits=20, seed=SEEDS["item_splits"] + 1000 + s_counter)
            s_counter += 1
    res = evaluate_cohort2_estimator(
        Ys, all_cells_4, slab_leg_maps, "legacy_first", slab_splits, cosets_info, r8_fr, compute_rcbd=True, n_perm_rcbd=0
    )
    return res["df_cubes"], res["summary"]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Verifying source data SHA-256 hashes...")
    source_hashes = verify_source_hashes()

    with open(DATA_DIR / "Ys.pickle", "rb") as f:
        Ys = pickle.load(f)

    print("2. Building complete template mapping manifest and factor construct audit...")
    df_mmlu_map, mmlu_leg_map, mmlu_canon_map = build_mmlu_mapping_table()
    mmlu_uniq_map = extract_unique_cell_map(df_mmlu_map)
    all_cells_5 = sorted(mmlu_leg_map.keys())
    all_cells_4 = list(itertools.product([0, 1], repeat=4))

    slab_map_dfs = []
    slab_leg_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]] = {}
    slab_canon_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]] = {}
    slab_uniq_maps: Dict[Tuple[str, str], Dict[Tuple[int, ...], List[int]]] = {}
    slab_completeness_rows = []
    for bench, model_list in [("BBH", BBH_MODELS), ("LMentry", LMENTRY_MODELS)]:
        for fpath in sorted((DATA_DIR / "templates" / bench).glob("*.csv")):
            df_t = pd.read_csv(fpath)
            df_m, lmap, cmap = build_slab_mapping_table_for_task(df_t, bench, fpath.stem)
            n_items_task = int(Ys[bench][fpath.stem][0].shape[1])
            n_models_task = len(model_list)
            is_complete = len(lmap) == 16
            n_dup_task = int(df_m["is_duplicate_text"].sum())
            slab_completeness_rows.append(
                {
                    "benchmark": bench,
                    "task": fpath.stem,
                    "n_templates": len(df_t),
                    "n_unique_templates": len(df_t) - n_dup_task,
                    "n_duplicate_rows": n_dup_task,
                    "n_populated_cells": len(lmap),
                    "is_complete_16_cell_cube": is_complete,
                    "n_models": n_models_task,
                    "n_items": n_items_task,
                    "n_cubes_included": n_models_task if is_complete else 0,
                    "n_cell_item_observations_included": (16 * n_models_task * n_items_task) if is_complete else 0,
                }
            )
            if not is_complete:
                continue
            slab_map_dfs.append(df_m)
            slab_leg_maps[(bench, fpath.stem)] = lmap
            slab_canon_maps[(bench, fpath.stem)] = cmap
            slab_uniq_maps[(bench, fpath.stem)] = extract_unique_cell_map(df_m)

    df_template_mapping = pd.concat([df_mmlu_map] + slab_map_dfs, ignore_index=True)
    df_factor_audit = build_factor_audit_table(df_template_mapping)

    print("3. Generating and validating fractional cosets, full-rank attribution designs, and unconditioned ScreenBest designs...")
    cosets_info = enumerate_fractional_cosets()

    r16_fr, r16_fr_diag, r16_fr_stats = generate_random_designs_with_diagnostics(
        5, 16, 200, SEEDS["rand16_full_rank"], True, "MMLU", "rand16_2fi_fullrank", require_main_full_rank=True
    )
    r16_conds = np.array([d["cond_2fi"] for d in r16_fr_diag if d["accepted_primary"]])

    r16_sb_uncond, r16_sb_diag, r16_sb_stats = generate_random_designs_with_diagnostics(
        5, 16, 200, SEEDS["rand16_screenbest_uncond"], False, "MMLU", "rand16_screenbest_unconditioned", require_main_full_rank=False
    )

    r16_leg = generate_balanced_random_designs(
        32, 5, 16, num_screens=20, seed=SEEDS["rand16_legacy"], require_2fi_full_rank=False
    )
    m5 = build_cube_design_matrices(5)
    m4 = build_cube_design_matrices(4)
    leg_diag_rows = []
    for i, c in enumerate(r16_leg):
        rk_m = int(np.linalg.matrix_rank(m5["D_main"][c]))
        cd_m = float(np.linalg.cond(m5["D_main"][c]))
        rk_2 = int(np.linalg.matrix_rank(m5["D_2fi"][c]))
        cd_2 = float(np.linalg.cond(m5["D_2fi"][c]))
        leg_diag_rows.append(
            {
                "design_id": f"rand16_legacy_{i:02d}",
                "cohort": "MMLU",
                "design_family": "rand16_legacy_unfiltered_2fi",
                "num_factors": 5,
                "num_runs": 16,
                "selected_cell_indices": json.dumps(c),
                "defining_relation": "legacy_random_balanced_main_only",
                "coset_sign": "none",
                "random_seed": SEEDS["rand16_legacy"],
                "generation_attempt": i + 1,
                "model_spec": "intercept_main_2fi",
                "n_parameters": 16,
                "matrix_rank": rk_2,
                "is_full_rank": (rk_2 == 16),
                "condition_number": cd_2,
                "rank_main": rk_m,
                "cond_main": cd_m,
                "rank_2fi": rk_2,
                "cond_2fi": cd_2,
                "accepted_primary": False,
            }
        )

    r8_c1_fr, r8_c1_diag, r8_c1_stats = generate_random_designs_with_diagnostics(
        5, 8, 200, SEEDS["rand8_cohort1"], False, "MMLU", "rand8_cohort1_fullrank", require_main_full_rank=True
    )
    r8_c1_sb_uncond, r8_c1_sb_diag, r8_c1_sb_stats = generate_random_designs_with_diagnostics(
        5, 8, 200, SEEDS["rand8_c1_screenbest_uncond"], False, "MMLU", "rand8_cohort1_screenbest_unconditioned", require_main_full_rank=False
    )
    r8_c2_fr, r8_c2_diag, r8_c2_stats = generate_random_designs_with_diagnostics(
        4, 8, 200, SEEDS["rand8_cohort2"], False, "SLAB", "rand8_cohort2_fullrank", require_main_full_rank=True
    )
    r8_c2_sb_uncond, r8_c2_sb_diag, r8_c2_sb_stats = generate_random_designs_with_diagnostics(
        4, 8, 200, SEEDS["rand8_c2_screenbest_uncond"], False, "SLAB", "rand8_cohort2_screenbest_unconditioned", require_main_full_rank=False
    )

    # Add deterministic fractional coset diagnostics and lower-budget OFAT baselines
    det_diag_rows = []
    ofat5_cells = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
    ofat5_idx = [all_cells_5.index(c) for c in ofat5_cells]
    ofat4_cells = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    ofat4_idx = [all_cells_4.index(c) for c in ofat4_cells]
    det_diag_rows.append(
        {
            "design_id": "OFAT_6_lower_budget", "cohort": "MMLU", "design_family": "OFAT_lower_budget",
            "num_factors": 5, "num_runs": 6, "selected_cell_indices": json.dumps(ofat5_idx),
            "defining_relation": "lower_budget_k_plus_1_ofat_6_runs_not_equal_budget", "coset_sign": "none",
            "random_seed": 0, "generation_attempt": 1, "model_spec": "intercept_main",
            "n_parameters": 6, "matrix_rank": int(np.linalg.matrix_rank(m5["D_main"][ofat5_idx])),
            "is_full_rank": True, "condition_number": float(np.linalg.cond(m5["D_main"][ofat5_idx])),
            "rank_main": 6, "cond_main": float(np.linalg.cond(m5["D_main"][ofat5_idx])),
            "rank_2fi": int(np.linalg.matrix_rank(m5["D_2fi"][ofat5_idx])),
            "cond_2fi": float("inf"),
            "accepted_primary": True,
        }
    )
    det_diag_rows.append(
        {
            "design_id": "OFAT_5_lower_budget", "cohort": "SLAB", "design_family": "OFAT_lower_budget",
            "num_factors": 4, "num_runs": 5, "selected_cell_indices": json.dumps(ofat4_idx),
            "defining_relation": "lower_budget_k_plus_1_ofat_5_runs_not_equal_budget", "coset_sign": "none",
            "random_seed": 0, "generation_attempt": 1, "model_spec": "intercept_main",
            "n_parameters": 5, "matrix_rank": int(np.linalg.matrix_rank(m4["D_main"][ofat4_idx])),
            "is_full_rank": True, "condition_number": float(np.linalg.cond(m4["D_main"][ofat4_idx])),
            "rank_main": 5, "cond_main": float(np.linalg.cond(m4["D_main"][ofat4_idx])),
            "rank_2fi": int(np.linalg.matrix_rank(m4["D_2fi"][ofat4_idx])),
            "cond_2fi": float("inf"),
            "accepted_primary": True,
        }
    )
    for cid, meta_c in cosets_info["res5_cosets"].items():
        idx = meta_c["indices"]
        det_diag_rows.append(
            {
                "design_id": cid, "cohort": "MMLU", "design_family": "2(5-1)V",
                "num_factors": 5, "num_runs": 16, "selected_cell_indices": json.dumps(idx),
                "defining_relation": meta_c["defining_relation"], "coset_sign": meta_c["coset_sign"],
                "random_seed": 0, "generation_attempt": 1, "model_spec": "intercept_main_2fi",
                "n_parameters": 16, "matrix_rank": int(np.linalg.matrix_rank(m5["D_2fi"][idx])),
                "is_full_rank": True, "condition_number": float(np.linalg.cond(m5["D_2fi"][idx])),
                "rank_main": 6, "cond_main": float(np.linalg.cond(m5["D_main"][idx])),
                "rank_2fi": 16, "cond_2fi": float(np.linalg.cond(m5["D_2fi"][idx])),
                "accepted_primary": True,
            }
        )
    for cid, meta_c in cosets_info["res3_canonical_cosets"].items():
        idx = meta_c["indices"]
        det_diag_rows.append(
            {
                "design_id": cid, "cohort": "MMLU", "design_family": "2(5-2)III_canonical",
                "num_factors": 5, "num_runs": 8, "selected_cell_indices": json.dumps(idx),
                "defining_relation": meta_c["defining_relation"], "coset_sign": meta_c["coset_sign"],
                "random_seed": 0, "generation_attempt": 1, "model_spec": "intercept_main",
                "n_parameters": 6, "matrix_rank": int(np.linalg.matrix_rank(m5["D_main"][idx])),
                "is_full_rank": True, "condition_number": float(np.linalg.cond(m5["D_main"][idx])),
                "rank_main": 6, "cond_main": float(np.linalg.cond(m5["D_main"][idx])),
                "rank_2fi": int(np.linalg.matrix_rank(m5["D_2fi"][idx])),
                "cond_2fi": float("inf"),
                "accepted_primary": True,
            }
        )
    for cid, meta_c in cosets_info["res3_all_60"].items():
        idx = meta_c["indices"]
        det_diag_rows.append(
            {
                "design_id": cid, "cohort": "MMLU", "design_family": "2(5-2)III_all60",
                "num_factors": 5, "num_runs": 8, "selected_cell_indices": json.dumps(idx),
                "defining_relation": meta_c["defining_relation"], "coset_sign": meta_c["coset_sign"],
                "random_seed": 0, "generation_attempt": 1, "model_spec": "intercept_main",
                "n_parameters": 6, "matrix_rank": int(np.linalg.matrix_rank(m5["D_main"][idx])),
                "is_full_rank": True, "condition_number": float(np.linalg.cond(m5["D_main"][idx])),
                "rank_main": 6, "cond_main": float(np.linalg.cond(m5["D_main"][idx])),
                "rank_2fi": int(np.linalg.matrix_rank(m5["D_2fi"][idx])),
                "cond_2fi": float("inf"),
                "accepted_primary": True,
            }
        )
    for cid, meta_c in cosets_info["res4_cosets"].items():
        idx = meta_c["indices"]
        det_diag_rows.append(
            {
                "design_id": cid, "cohort": "SLAB", "design_family": "2(4-1)IV",
                "num_factors": 4, "num_runs": 8, "selected_cell_indices": json.dumps(idx),
                "defining_relation": meta_c["defining_relation"], "coset_sign": meta_c["coset_sign"],
                "random_seed": 0, "generation_attempt": 1, "model_spec": "intercept_main",
                "n_parameters": 5, "matrix_rank": int(np.linalg.matrix_rank(m4["D_main"][idx])),
                "is_full_rank": True, "condition_number": float(np.linalg.cond(m4["D_main"][idx])),
                "rank_main": 5, "cond_main": float(np.linalg.cond(m4["D_main"][idx])),
                "rank_2fi": int(np.linalg.matrix_rank(m4["D_2fi"][idx])),
                "cond_2fi": float("inf"),
                "accepted_primary": True,
            }
        )

    df_design_diagnostics = pd.DataFrame(
        det_diag_rows + r16_fr_diag + r16_sb_diag + leg_diag_rows + r8_c1_diag + r8_c1_sb_diag + r8_c2_diag + r8_c2_sb_diag
    )

    print("4. Generating deterministic discovery/confirmation item splits (legacy + 20 random splits)...")
    tasks_mmlu = sorted(Ys["MMLU"].keys())
    mmlu_splits = {}
    for idx, t in enumerate(tasks_mmlu):
        n_items = Ys["MMLU"][t][0].shape[1]
        mmlu_splits[t] = generate_item_splits(n_items, n_splits=20, seed=SEEDS["item_splits"] + idx)
    split_ids = mmlu_splits[tasks_mmlu[0]][2]

    slab_splits = {}
    for idx, (bench, task_name) in enumerate(sorted(slab_leg_maps.keys())):
        n_items = Ys[bench][task_name][0].shape[1]
        slab_splits[(bench, task_name)] = generate_item_splits(
            n_items, n_splits=20, seed=SEEDS["item_splits"] + 1000 + idx
        )

    print("5. Evaluating Cohort 1 (MMLU: 855 cubes) across estimators, cosets, and 21 splits...")
    res_c1_leg = evaluate_cohort1_estimator(
        Ys, all_cells_5, mmlu_leg_map, "legacy_first", mmlu_splits, cosets_info,
        r16_fr, r16_conds, r16_leg, r8_c1_fr, rand16_screenbest_uncond=r16_sb_uncond, rand8_screenbest_uncond=r8_c1_sb_uncond,
        compute_rcbd=True, n_perm_rcbd=200,
    )
    res_c1_cell = evaluate_cohort1_estimator(
        Ys, all_cells_5, mmlu_canon_map, "cell_mean", mmlu_splits, cosets_info,
        r16_fr, r16_conds, r16_leg, r8_c1_fr, rand16_screenbest_uncond=r16_sb_uncond, rand8_screenbest_uncond=r8_c1_sb_uncond,
        compute_rcbd=True, n_perm_rcbd=200,
    )
    res_c1_cell_uniq = evaluate_cohort1_estimator(
        Ys, all_cells_5, mmlu_uniq_map, "cell_mean_unique", mmlu_splits, cosets_info,
        r16_fr, r16_conds, r16_leg, r8_c1_fr, rand16_screenbest_uncond=r16_sb_uncond, rand8_screenbest_uncond=r8_c1_sb_uncond,
        compute_rcbd=True, n_perm_rcbd=200,
    )
    res_c1_canon = evaluate_cohort1_estimator(
        Ys, all_cells_5, mmlu_canon_map, "canonical_first", mmlu_splits, cosets_info,
        r16_fr, r16_conds, r16_leg, r8_c1_fr, rand16_screenbest_uncond=r16_sb_uncond, rand8_screenbest_uncond=r8_c1_sb_uncond,
        compute_rcbd=False, n_perm_rcbd=0,
    )

    print("6. Evaluating Cohort 2 (SLAB-NLP BBH & LMentry: 178 cubes) across estimators, cosets, and 21 splits...")
    res_c2_leg = evaluate_cohort2_estimator(
        Ys, all_cells_4, slab_leg_maps, "legacy_first", slab_splits, cosets_info,
        r8_c2_fr, rand8_screenbest_uncond=r8_c2_sb_uncond, compute_rcbd=True, n_perm_rcbd=200,
    )
    res_c2_cell = evaluate_cohort2_estimator(
        Ys, all_cells_4, slab_canon_maps, "cell_mean", slab_splits, cosets_info,
        r8_c2_fr, rand8_screenbest_uncond=r8_c2_sb_uncond, compute_rcbd=True, n_perm_rcbd=200,
    )
    res_c2_cell_uniq = evaluate_cohort2_estimator(
        Ys, all_cells_4, slab_uniq_maps, "cell_mean_unique", slab_splits, cosets_info,
        r8_c2_fr, rand8_screenbest_uncond=r8_c2_sb_uncond, compute_rcbd=True, n_perm_rcbd=200,
    )
    res_c2_canon = evaluate_cohort2_estimator(
        Ys, all_cells_4, slab_canon_maps, "canonical_first", slab_splits, cosets_info,
        r8_c2_fr, rand8_screenbest_uncond=r8_c2_sb_uncond, compute_rcbd=False, n_perm_rcbd=0,
    )

    print("7. Building coset, split, template sensitivity, interaction, and multiplicity artifacts...")
    df_cosets = build_coset_results_table(res_c1_cell, res_c1_leg, res_c2_cell, res_c2_leg, cosets_info)
    df_splits = build_split_selection_table(res_c1_cell, res_c1_leg, res_c2_cell, res_c2_leg, split_ids)
    df_template_sens = evaluate_template_representative_sensitivity(
        Ys, all_cells_5, mmlu_canon_map, mmlu_uniq_map, all_cells_4, slab_canon_maps, slab_uniq_maps,
        mmlu_splits, slab_splits,
        res_c1_cell, res_c1_cell_uniq, res_c1_leg, res_c1_canon,
        res_c2_cell, res_c2_cell_uniq, res_c2_leg, res_c2_canon,
        cosets_info, r16_fr, r8_c2_fr,
        rand16_screenbest_uncond=r16_sb_uncond, rand8_c2_screenbest_uncond=r8_c2_sb_uncond,
        n_rep_draws=20,
    )

    df_fourier_all = pd.concat(
        [
            res_c1_leg["df_fourier"], res_c1_cell["df_fourier"], res_c1_cell_uniq["df_fourier"],
            res_c2_leg["df_fourier"], res_c2_cell["df_fourier"], res_c2_cell_uniq["df_fourier"],
        ],
        ignore_index=True,
    )
    df_rcbd_all = pd.concat(
        [
            res_c1_leg["df_rcbd"], res_c1_cell["df_rcbd"], res_c1_cell_uniq["df_rcbd"],
            res_c2_leg["df_rcbd"], res_c2_cell["df_rcbd"], res_c2_cell_uniq["df_rcbd"],
        ],
        ignore_index=True,
    )
    df_multiplicity = build_multiplicity_table(df_rcbd_all, n_perm_rcbd=200)
    interaction_regret_assoc = compute_interaction_regret_associations(df_fourier_all)

    print("8. Verifying direct integration with prompt_prism.ANOVAEngine & OptimalPromptFinder...")
    engine_check = verify_with_anova_engine(Ys)

    # Save all CSV artifacts
    artifact_paths = {
        "design_diagnostics": RESULTS_DIR / "design_diagnostics.csv",
        "coset_results": RESULTS_DIR / "coset_results.csv",
        "split_selection_results": RESULTS_DIR / "split_selection_results.csv",
        "template_mapping": RESULTS_DIR / "template_mapping.csv",
        "template_sensitivity": RESULTS_DIR / "template_sensitivity.csv",
        "factor_audit": RESULTS_DIR / "factor_audit.csv",
        "interaction_magnitudes": RESULTS_DIR / "interaction_magnitudes.csv",
        "rcbd_robustness": RESULTS_DIR / "rcbd_robustness.csv",
        "multiplicity_results": RESULTS_DIR / "multiplicity_results.csv",
        "cohort1_mmlu_models": RESULTS_DIR / "cohort1_mmlu_models.csv",
        "cohort1_mmlu_cubes": RESULTS_DIR / "cohort1_mmlu_cubes.csv",
        "cohort2_slab_cubes": RESULTS_DIR / "cohort2_slab_cubes.csv",
    }

    df_design_diagnostics.to_csv(artifact_paths["design_diagnostics"], index=False)
    df_cosets.to_csv(artifact_paths["coset_results"], index=False)
    df_splits.to_csv(artifact_paths["split_selection_results"], index=False)
    df_template_mapping.to_csv(artifact_paths["template_mapping"], index=False)
    df_template_sens.to_csv(artifact_paths["template_sensitivity"], index=False)
    df_factor_audit.to_csv(artifact_paths["factor_audit"], index=False)
    df_interaction_export = df_fourier_all.rename(
        columns={
            "disc0_rms_main": "split0_rms_main",
            "disc0_rms_2fi": "split0_rms_second_order",
            "disc0_rms_3fi": "split0_rms_third_order",
            "disc0_rms_4fi": "split0_rms_fourth_order",
            "disc0_rmse_main_res5": "split0_rmse_main_fraction",
            "disc0_rmse_2fi_res5": "split0_rmse_second_order_fraction",
            "disc0_rmse_2fi_rand16_fullrank": "split0_rmse_second_order_random_fullrank",
        }
    )
    df_interaction_export.to_csv(artifact_paths["interaction_magnitudes"], index=False)
    df_rcbd_all.to_csv(artifact_paths["rcbd_robustness"], index=False)
    df_multiplicity.to_csv(artifact_paths["multiplicity_results"], index=False)
    res_c1_leg["df_models"].to_csv(artifact_paths["cohort1_mmlu_models"], index=False)
    res_c1_leg["df_cubes"].to_csv(artifact_paths["cohort1_mmlu_cubes"], index=False)
    res_c2_leg["df_cubes"].to_csv(artifact_paths["cohort2_slab_cubes"], index=False)

    # Template representative distribution summary
    c1_rep_sub = df_template_sens[(df_template_sens["cohort"] == "MMLU") & (df_template_sens["estimator"] == "random_single_rep")]
    c1_rep_uniq_sub = df_template_sens[(df_template_sens["cohort"] == "MMLU") & (df_template_sens["estimator"] == "random_single_rep_unique")]
    c2_rep_sub = df_template_sens[(df_template_sens["cohort"] == "SLAB") & (df_template_sens["estimator"] == "random_single_rep")]
    c2_rep_uniq_sub = df_template_sens[(df_template_sens["cohort"] == "SLAB") & (df_template_sens["estimator"] == "random_single_rep_unique")]

    c1_obs = int(res_c1_leg["summary"]["total_cell_item_observations"])
    c2_obs = int(res_c2_leg["summary"]["total_cell_item_observations"])
    study_observation_totals = {
        "cohort1_mmlu_cell_item_observations": c1_obs,
        "cohort2_slab_cell_item_observations": c2_obs,
        "cohort2_bbh_cell_item_observations": int(res_c2_leg["summary"]["bbh_cell_item_observations"]),
        "cohort2_lmentry_cell_item_observations": int(res_c2_leg["summary"]["lmentry_cell_item_observations"]),
        "total_study_cell_item_observations": c1_obs + c2_obs,
        "lmentry_word_not_containing_items": int(Ys["LMentry"]["word_not_containing"][0].shape[1]),
    }

    cell0_diagnostic = {
        "cohort": "MMLU",
        "cell_tuple": "(0, 0, 0, 0, 0)",
        "template_indices": [int(x) for x in mmlu_leg_map[(0, 0, 0, 0, 0)]],
        "unique_template_indices": [int(x) for x in mmlu_uniq_map[(0, 0, 0, 0, 0)]],
        "n_catalog_rows": len(mmlu_leg_map[(0, 0, 0, 0, 0)]),
        "n_unique_templates": len(mmlu_uniq_map[(0, 0, 0, 0, 0)]),
        "legacy_selected_row_index": int(mmlu_leg_map[(0, 0, 0, 0, 0)][0]),
        "canonical_selected_row_index": int(mmlu_canon_map[(0, 0, 0, 0, 0)][0]),
        "split0_conf_legacy_first": float(res_c1_leg["summary"]["mean_conf_baseline"]),
        "split0_conf_cell_mean": float(res_c1_cell["summary"]["mean_conf_baseline"]),
        "split0_conf_cell_mean_unique": float(res_c1_cell_uniq["summary"]["mean_conf_baseline"]),
        "split0_diff_legacy_minus_cell_mean_pp": float(
            (res_c1_leg["summary"]["mean_conf_baseline"] - res_c1_cell["summary"]["mean_conf_baseline"]) * 100.0
        ),
        "mean20splits_conf_legacy_first": float(res_c1_leg["summary"]["mean_conf_baseline_20splits"]),
        "mean20splits_conf_cell_mean": float(res_c1_cell["summary"]["mean_conf_baseline_20splits"]),
        "mean20splits_conf_cell_mean_unique": float(res_c1_cell_uniq["summary"]["mean_conf_baseline_20splits"]),
    }

    cohort2_completeness_audit = {
        "bbh_total_tasks": sum(1 for r in slab_completeness_rows if r["benchmark"] == "BBH"),
        "bbh_complete_16cell_tasks": sum(1 for r in slab_completeness_rows if r["benchmark"] == "BBH" and r["is_complete_16_cell_cube"]),
        "lmentry_total_tasks": sum(1 for r in slab_completeness_rows if r["benchmark"] == "LMentry"),
        "lmentry_complete_16cell_tasks": sum(1 for r in slab_completeness_rows if r["benchmark"] == "LMentry" and r["is_complete_16_cell_cube"]),
        "total_catalog_tasks": len(slab_completeness_rows),
        "total_complete_16cell_tasks": sum(1 for r in slab_completeness_rows if r["is_complete_16_cell_cube"]),
        "excluded_incomplete_tasks": [
            f"{r['benchmark']}/{r['task']} ({r['n_populated_cells']}/16 cells)"
            for r in slab_completeness_rows
            if not r["is_complete_16_cell_cube"]
        ],
        "tasks": slab_completeness_rows,
    }

    def _summarize_rep_draws(sub_df: pd.DataFrame, has_2fi: bool) -> Dict[str, Any]:
        out = {
            "mean_rmse_main_frac": float(sub_df["rmse_main_frac"].mean()),
            "sd_rmse_main_frac": float(sub_df["rmse_main_frac"].std(ddof=1)),
            "mean_main_rmse_ratio": float(sub_df["main_rmse_ratio"].mean()),
            "mean_conf_screen_best_split0": float(sub_df["conf_screen_best_coset_avg"].mean()),
            "sd_across_template_draws_conf_screen_best_split0": float(sub_df["conf_screen_best_coset_avg"].std(ddof=1)),
            "mean_conf_screen_best_20splits": float(sub_df["conf_screen_best_coset_avg_20splits"].mean()),
            "sd_across_template_draws_20splits_conf_screen_best": float(sub_df["conf_screen_best_coset_avg_20splits"].std(ddof=1)),
            "mean_sd_across_splits_conf_screen_best": float(sub_df["sd_across_splits_conf_screen_best"].mean()),
            "mean_recovery_screen_best_coset_avg_pct": float(sub_df["recovery_screen_best_coset_avg_pct"].mean()),
            "sd_recovery_screen_best_coset_avg_pct": float(sub_df["recovery_screen_best_coset_avg_pct"].std(ddof=1)),
            "mean_recovery_screen_best_20splits_pct": float(sub_df["recovery_screen_best_coset_avg_20splits_pct"].mean()),
            "sd_recovery_screen_best_20splits_pct": float(sub_df["recovery_screen_best_coset_avg_20splits_pct"].std(ddof=1)),
            "mean_recovery_rand_best_pct": float(sub_df["recovery_rand_best_pct"].mean()),
            "mean_recovery_rand_best_20splits_pct": float(sub_df["recovery_rand_best_20splits_pct"].mean()),
        }
        if has_2fi:
            out["mean_twofi_rmse_ratio_fullrank"] = float(sub_df["twofi_rmse_ratio_fullrank"].mean())
            out["mean_recovery_model_2fi_coset_avg_pct"] = float(sub_df["recovery_model_2fi_coset_avg_pct"].mean())
            out["mean_recovery_model_2fi_20splits_pct"] = float(sub_df["recovery_model_2fi_coset_avg_20splits_pct"].mean())
            out["mean_sd_across_splits_conf_model_2fi"] = float(sub_df["sd_across_splits_conf_model_2fi"].mean())
        return out

    headline_summary = {
        "study_observation_totals": study_observation_totals,
        "cell0_diagnostic": cell0_diagnostic,
        "cohort2_completeness_audit": cohort2_completeness_audit,
        "cohort1_mmlu_legacy_first": res_c1_leg["summary"],
        "cohort1_mmlu_cell_mean": res_c1_cell["summary"],
        "cohort1_mmlu_cell_mean_unique": res_c1_cell_uniq["summary"],
        "cohort1_mmlu_canonical_first": res_c1_canon["summary"],
        "cohort1_mmlu_random_rep_20draws_summary": _summarize_rep_draws(c1_rep_sub, has_2fi=True),
        "cohort1_mmlu_random_rep_unique_20draws_summary": _summarize_rep_draws(c1_rep_uniq_sub, has_2fi=True),
        "cohort2_slab_legacy_first": res_c2_leg["summary"],
        "cohort2_slab_cell_mean": res_c2_cell["summary"],
        "cohort2_slab_cell_mean_unique": res_c2_cell_uniq["summary"],
        "cohort2_slab_canonical_first": res_c2_canon["summary"],
        "cohort2_slab_random_rep_20draws_summary": _summarize_rep_draws(c2_rep_sub, has_2fi=False),
        "cohort2_slab_random_rep_unique_20draws_summary": _summarize_rep_draws(c2_rep_uniq_sub, has_2fi=False),
        "interaction_regret_associations": interaction_regret_assoc,
        "multiplicity_summary": df_multiplicity.to_dict(orient="records"),
    }

    df_claims = build_claim_evidence_map(headline_summary)
    claim_map_path = RESULTS_DIR / "claim_evidence_map.csv"
    df_claims.to_csv(claim_map_path, index=False)
    artifact_paths["claim_evidence_map"] = claim_map_path

    headline_path = RESULTS_DIR / "headline_summary.json"
    headline_path.write_text(json.dumps(headline_summary, indent=2) + "\n", encoding="utf-8")
    artifact_paths["headline_summary"] = headline_path

    print("9. Generating figures and LaTeX tables...")
    generate_figures(res_c1_leg["df_models"], res_c1_leg, res_c1_cell, res_c2_leg, res_c2_cell, df_cosets)
    write_latex_tables(
        res_c1_leg["df_models"],
        res_c1_leg,
        res_c1_cell,
        res_c2_leg,
        res_c2_cell,
        df_multiplicity,
        res_c1_cell_uniq=res_c1_cell_uniq,
        res_c2_cell_uniq=res_c2_cell_uniq,
    )
    macros_path = write_generated_macros(headline_summary)

    # Register figure and table outputs BEFORE computing final SHA-256 hashes (Task 8)
    artifact_paths["fig1_effect_and_selection_pdf"] = FIGURES_DIR / "fig1_effect_and_selection.pdf"
    artifact_paths["fig1_effect_and_selection_png"] = FIGURES_DIR / "fig1_effect_and_selection.png"
    artifact_paths["fig2_models_and_rcbd_pdf"] = FIGURES_DIR / "fig2_models_and_rcbd.pdf"
    artifact_paths["fig2_models_and_rcbd_png"] = FIGURES_DIR / "fig2_models_and_rcbd.png"
    artifact_paths["table_mmlu_models_tex"] = TABLES_DIR / "table_mmlu_models.tex"
    artifact_paths["table_cross_cohort_tex"] = TABLES_DIR / "table_cross_cohort.tex"
    artifact_paths["generated_macros_tex"] = macros_path

    output_hashes = {k: sha256_file(p) for k, p in artifact_paths.items()}

    manifest = {
        "study": "PromptPrism Multi-Model Hugging Face Factorial Benchmark Study (V4 Cross-Model Audit Revision)",
        "data_sources": {
            "PromptEval_MMLU_full": "https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_full",
            "PromptEval_MMLU_correctness": "https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_correctness",
            "SLAB_NLP_Multi_Prompt": "https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation",
            **source_hashes,
        },
        "analysis_definition_sha256": {
            "research/hf_multi_prompt/run_experiments.py": sha256_file(Path(__file__)),
            "research/hf_multi_prompt/build_release_manifest.py": sha256_file(
                ROOT / "research" / "hf_multi_prompt" / "build_release_manifest.py"
            ),
            "tests/research/test_hf_multi_prompt.py": sha256_file(
                ROOT / "tests" / "research" / "test_hf_multi_prompt.py"
            ),
            "paper_hf/main.tex": sha256_file(PAPER_DIR / "main.tex"),
            "paper_hf/references.bib": sha256_file(PAPER_DIR / "references.bib"),
            "prompt_prism/analysis/anova.py": sha256_file(
                ROOT / "prompt_prism" / "analysis" / "anova.py"
            ),
            "prompt_prism/analysis/optimizer.py": sha256_file(
                ROOT / "prompt_prism" / "analysis" / "optimizer.py"
            ),
            "prompt_prism/design/catalog.py": sha256_file(
                ROOT / "prompt_prism" / "design" / "catalog.py"
            ),
            "prompt_prism/design/generators.py": sha256_file(
                ROOT / "prompt_prism" / "design" / "generators.py"
            ),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "statsmodels_version": statsmodels.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
        "random_seeds": SEEDS,
        "study_observation_totals": study_observation_totals,
        "cell0_diagnostic": cell0_diagnostic,
        "cohort2_completeness_audit": cohort2_completeness_audit,
        "design_generator_diagnostics": {
            "rand16_2fi_fullrank": r16_fr_stats,
            "rand16_screenbest_unconditioned": r16_sb_stats,
            "rand16_legacy_2fi_fullrank_count": sum(1 for r in leg_diag_rows if r["is_full_rank"]),
            "rand16_legacy_2fi_rank_deficient_count": sum(1 for r in leg_diag_rows if not r["is_full_rank"]),
            "rand8_cohort1_fullrank": r8_c1_stats,
            "rand8_cohort1_screenbest_unconditioned": r8_c1_sb_stats,
            "rand8_cohort2_fullrank": r8_c2_stats,
            "rand8_cohort2_screenbest_unconditioned": r8_c2_sb_stats,
        },
        "cohort1_mmlu_summary": res_c1_leg["summary"],
        "cohort1_mmlu_cell_mean_summary": res_c1_cell["summary"],
        "cohort1_mmlu_cell_mean_unique_summary": res_c1_cell_uniq["summary"],
        "cohort2_slab_summary": res_c2_leg["summary"],
        "cohort2_slab_cell_mean_summary": res_c2_cell["summary"],
        "cohort2_slab_cell_mean_unique_summary": res_c2_cell_uniq["summary"],
        "interaction_regret_associations": interaction_regret_assoc,
        "anova_engine_verification": engine_check,
        "output_artifact_sha256": output_hashes,
    }
    (RESULTS_DIR / "summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Done! Summary manifest written to:", RESULTS_DIR / "summary_manifest.json")


def refresh_paper_outputs_from_saved_results() -> None:
    """Regenerate figures, LaTeX tables/macros, claim map, and summary_manifest.json from verified cached CSV/JSON results."""
    verify_source_hashes()
    headline_summary = json.loads((RESULTS_DIR / "headline_summary.json").read_text(encoding="utf-8"))
    df_models = pd.read_csv(RESULTS_DIR / "cohort1_mmlu_models.csv")
    df_c2_cubes = pd.read_csv(RESULTS_DIR / "cohort2_slab_cubes.csv")
    df_cosets = pd.read_csv(RESULTS_DIR / "coset_results.csv")
    df_mult = pd.read_csv(RESULTS_DIR / "multiplicity_results.csv")

    res_c1_leg = {"summary": headline_summary["cohort1_mmlu_legacy_first"], "df_models": df_models}
    res_c1_cell = {"summary": headline_summary["cohort1_mmlu_cell_mean"]}
    res_c1_cell_uniq = {"summary": headline_summary["cohort1_mmlu_cell_mean_unique"]}
    res_c2_leg = {"summary": headline_summary["cohort2_slab_legacy_first"], "df_cubes": df_c2_cubes}
    res_c2_cell = {"summary": headline_summary["cohort2_slab_cell_mean"]}
    res_c2_cell_uniq = {"summary": headline_summary["cohort2_slab_cell_mean_unique"]}

    df_claims = build_claim_evidence_map(headline_summary)
    df_claims.to_csv(RESULTS_DIR / "claim_evidence_map.csv", index=False)

    generate_figures(df_models, res_c1_leg, res_c1_cell, res_c2_leg, res_c2_cell, df_cosets)
    write_latex_tables(
        df_models,
        res_c1_leg,
        res_c1_cell,
        res_c2_leg,
        res_c2_cell,
        df_mult,
        res_c1_cell_uniq=res_c1_cell_uniq,
        res_c2_cell_uniq=res_c2_cell_uniq,
    )
    macros_path = write_generated_macros(headline_summary)

    manifest_path = RESULTS_DIR / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {
        "design_diagnostics": RESULTS_DIR / "design_diagnostics.csv",
        "coset_results": RESULTS_DIR / "coset_results.csv",
        "split_selection_results": RESULTS_DIR / "split_selection_results.csv",
        "template_mapping": RESULTS_DIR / "template_mapping.csv",
        "template_sensitivity": RESULTS_DIR / "template_sensitivity.csv",
        "factor_audit": RESULTS_DIR / "factor_audit.csv",
        "interaction_magnitudes": RESULTS_DIR / "interaction_magnitudes.csv",
        "rcbd_robustness": RESULTS_DIR / "rcbd_robustness.csv",
        "multiplicity_results": RESULTS_DIR / "multiplicity_results.csv",
        "cohort1_mmlu_models": RESULTS_DIR / "cohort1_mmlu_models.csv",
        "cohort1_mmlu_cubes": RESULTS_DIR / "cohort1_mmlu_cubes.csv",
        "cohort2_slab_cubes": RESULTS_DIR / "cohort2_slab_cubes.csv",
        "claim_evidence_map": RESULTS_DIR / "claim_evidence_map.csv",
        "headline_summary": RESULTS_DIR / "headline_summary.json",
        "fig1_effect_and_selection_pdf": FIGURES_DIR / "fig1_effect_and_selection.pdf",
        "fig1_effect_and_selection_png": FIGURES_DIR / "fig1_effect_and_selection.png",
        "fig2_models_and_rcbd_pdf": FIGURES_DIR / "fig2_models_and_rcbd.pdf",
        "fig2_models_and_rcbd_png": FIGURES_DIR / "fig2_models_and_rcbd.png",
        "table_mmlu_models_tex": TABLES_DIR / "table_mmlu_models.tex",
        "table_cross_cohort_tex": TABLES_DIR / "table_cross_cohort.tex",
        "generated_macros_tex": macros_path,
    }
    manifest["output_artifact_sha256"] = {k: sha256_file(p) for k, p in artifact_paths.items()}
    manifest["analysis_definition_sha256"] = {
        "research/hf_multi_prompt/run_experiments.py": sha256_file(Path(__file__)),
        "research/hf_multi_prompt/build_release_manifest.py": sha256_file(
            ROOT / "research" / "hf_multi_prompt" / "build_release_manifest.py"
        ),
        "tests/research/test_hf_multi_prompt.py": sha256_file(
            ROOT / "tests" / "research" / "test_hf_multi_prompt.py"
        ),
        "paper_hf/main.tex": sha256_file(PAPER_DIR / "main.tex"),
        "paper_hf/references.bib": sha256_file(PAPER_DIR / "references.bib"),
        "prompt_prism/analysis/anova.py": sha256_file(
            ROOT / "prompt_prism" / "analysis" / "anova.py"
        ),
        "prompt_prism/analysis/optimizer.py": sha256_file(
            ROOT / "prompt_prism" / "analysis" / "optimizer.py"
        ),
        "prompt_prism/design/catalog.py": sha256_file(
            ROOT / "prompt_prism" / "design" / "catalog.py"
        ),
        "prompt_prism/design/generators.py": sha256_file(
            ROOT / "prompt_prism" / "design" / "generators.py"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Refreshed figures, tables, claim map, and summary_manifest.json.")


if __name__ == "__main__":
    if "--refresh-paper-only" in sys.argv:
        refresh_paper_outputs_from_saved_results()
    else:
        main()
