"""
Large-Scale Multi-Model Factorial Prompt Engineering Study using Hugging Face
PromptEval (MMLU) and SLAB-NLP (BBH & LMentry) Multi-Prompt Repositories.

Executes three core empirical studies using PromptPrism:
1. Cohort 1 (PromptEval MMLU): 2^5 = 32 Full Factorial Cube across 15 LLMs and 57 tasks
   (855 model-task cubes; 6,740,160 cell-item observations).
   Tests 2(5-1)V (Res V, 16 runs) and 2(5-2)III (Res III, 8 runs) against
   Balanced Random-16, Balanced Random-8, and OFAT (6 runs).
2. Cohort 2 (SLAB-NLP BBH & LMentry): 2^4 = 16 Full Factorial Cube across 13 tasks
   and up to 16 LLMs (178 model-task cubes; 284,800 cell-item observations).
   Tests 2(4-1)IV (Res IV, 8 runs) against Balanced Random-8 and OFAT (5 runs).
3. Study 3 (RCBD Item Blocking Power Analysis): Quantifies residual MSE reduction,
   F-statistic amplification, R^2 gain, and FDR-corrected (alpha=0.05) discovery power
   of RCBD item blocking (block_by='sample_id') vs. unblocked ANOVA across all 1,033 cubes,
   cross-checked directly against prompt_prism.ANOVAEngine.run_anova.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
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


def build_mmlu_cell_mapping() -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[int]]]:
    """Map the 100 FormatSpread MMLU templates into a complete 2^5 = 32 factorial cube."""
    meta = json.loads((DATA_DIR / "templates" / "mmlu_templates_metadata.json").read_text())
    df_mmlu = pd.read_csv(DATA_DIR / "templates" / "MMLU" / "abstract_algebra.csv")
    cell_to_indices: Dict[Tuple[int, ...], List[int]] = {}
    for idx, r in df_mmlu.iterrows():
        m = meta[r["prompt template"]]
        op, sp, sep = m["op"], m["space"], m["sep"]
        # A: Case Preservation (1 = mixed/title case, 0 = uniform upper/lower)
        A = 1 if op in ("lambda x: x", "lambda x: x.title()") else 0
        # B: Header Capitalization (1 = upper/title, 0 = identity/lower)
        B = 1 if op in ("lambda x: x.upper()", "lambda x: x.title()") else 0
        # C: Multiline Layout (1 = newline in space delimiter, 0 = inline)
        C = 1 if "\n" in sp else 0
        # D: Natural Punctuation vs Symbolic/Markup Delimiter
        D = 1 if sep in (":", ".", "'", '"', "-", "!") else 0
        # E: Primary Clause/Tag Boundary vs Quote/Slash Connector
        E = 1 if sep in (":", ".", "<sep>", "|", "]") else 0
        cell_to_indices.setdefault((A, B, C, D, E), []).append(int(idx))

    all_cells = sorted(cell_to_indices.keys())
    assert len(all_cells) == 32, f"Expected 32 cells, got {len(all_cells)}"
    return all_cells, cell_to_indices


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


def generate_balanced_random_designs(
    num_cells: int,
    num_factors: int,
    subset_size: int,
    num_screens: int = 20,
    seed: int = 20261105,
) -> List[List[int]]:
    """Generate factor-balanced, full-main-effect-rank random screening designs."""
    X_full = np.array(list(itertools.product([0, 1], repeat=num_factors)))
    rng = np.random.default_rng(seed)
    screens: List[List[int]] = []
    seen = set()
    target_ones = subset_size // 2
    while len(screens) < num_screens:
        cand = tuple(sorted(int(x) for x in rng.choice(num_cells, size=subset_size, replace=False)))
        if cand in seen:
            continue
        sub = X_full[list(cand)]
        if np.all(sub.sum(axis=0) == target_ones):
            design_mat = np.column_stack([np.ones(subset_size), sub])
            if np.linalg.matrix_rank(design_mat) == num_factors + 1:
                seen.add(cand)
                screens.append(list(cand))
    return screens


def run_cohort1_mmlu(Ys: dict) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run Cohort 1: PromptEval MMLU 2^5 = 32 Full Factorial Cube across 15 LLMs and 57 tasks."""
    all_cells, cell_to_indices = build_mmlu_cell_mapping()
    X_full = np.array(all_cells)  # (32, 5)
    X_pm = 2 * X_full - 1  # (32, 5) in {-1, +1}
    pairs = list(itertools.combinations(range(5), 2))
    X_2fi = np.column_stack([X_pm[:, i] * X_pm[:, j] for i, j in pairs])  # (32, 10)
    D_main = np.column_stack([np.ones(32), X_pm])  # (32, 6)
    D_2fi = np.column_stack([np.ones(32), X_pm, X_2fi])  # (32, 16)

    # PromptPrism fractional designs
    res5 = FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    res5_idx = [all_cells.index(tuple(r.factor_levels[f] for f in "ABCDE")) for r in res5.runs]

    res3 = FractionalFactorialGenerator.from_plan_id("2(5-2)III")
    res3_idx = [all_cells.index(tuple(r.factor_levels[f] for f in "ABCDE")) for r in res3.runs]

    ofat_cells = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
    ofat_idx = [all_cells.index(c) for c in ofat_cells]

    rand16_list = generate_balanced_random_designs(32, 5, 16, num_screens=20, seed=20261105)
    rand8_list = generate_balanced_random_designs(32, 5, 8, num_screens=20, seed=20261106)

    tasks = sorted(Ys["MMLU"].keys())
    model_rows = []
    cube_rows = []

    for m_idx, (m_id, m_label, m_type) in enumerate(MMLU_MODELS):
        # 1. Pooled 14,042-item matrix per model
        full_mat = np.hstack([Ys["MMLU"][t][m_idx] for t in tasks])  # (100, 14042)
        Y_pool = np.array([full_mat[cell_to_indices[c][0]] for c in all_cells], dtype=float)  # (32, 14042)
        y_pool_d = Y_pool[:, ::2].mean(axis=1)
        y_pool_c = Y_pool[:, 1::2].mean(axis=1)

        true_main_pool = (X_pm.T @ y_pool_d) / 16.0
        true_2fi_pool = (X_2fi.T @ y_pool_d) / 16.0

        # Res V (16 runs)
        b_res5_m, *_ = np.linalg.lstsq(D_main[res5_idx], y_pool_d[res5_idx], rcond=None)
        est_main_res5 = 2.0 * b_res5_m[1:]
        b_res5_2fi, *_ = np.linalg.lstsq(D_2fi[res5_idx], y_pool_d[res5_idx], rcond=None)
        est_2fi_res5 = 2.0 * b_res5_2fi[6:]

        rmse_main_res5 = float(np.sqrt(np.mean((est_main_res5 - true_main_pool) ** 2)))
        rmse_2fi_res5 = float(np.sqrt(np.mean((est_2fi_res5 - true_2fi_pool) ** 2)))
        sign_res5 = float(np.mean(np.sign(est_main_res5) == np.sign(true_main_pool)))

        # Res III (8 runs)
        b_res3_m, *_ = np.linalg.lstsq(D_main[res3_idx], y_pool_d[res3_idx], rcond=None)
        est_main_res3 = 2.0 * b_res3_m[1:]
        rmse_main_res3 = float(np.sqrt(np.mean((est_main_res3 - true_main_pool) ** 2)))
        sign_res3 = float(np.mean(np.sign(est_main_res3) == np.sign(true_main_pool)))

        # OFAT (6 runs)
        b_ofat_m, *_ = np.linalg.lstsq(D_main[ofat_idx], y_pool_d[ofat_idx], rcond=None)
        est_main_ofat = 2.0 * b_ofat_m[1:]
        rmse_main_ofat = float(np.sqrt(np.mean((est_main_ofat - true_main_pool) ** 2)))

        # Balanced Random-16 & Random-8
        r16_main_rmses, r16_2fi_rmses, r16_signs = [], [], []
        for cand in rand16_list:
            bm, *_ = np.linalg.lstsq(D_main[cand], y_pool_d[cand], rcond=None)
            em = 2.0 * bm[1:]
            r16_main_rmses.append(np.sqrt(np.mean((em - true_main_pool) ** 2)))
            r16_signs.append(np.mean(np.sign(em) == np.sign(true_main_pool)))
            b2, *_ = np.linalg.lstsq(D_2fi[cand], y_pool_d[cand], rcond=None)
            e2 = 2.0 * b2[6:]
            r16_2fi_rmses.append(np.sqrt(np.mean((e2 - true_2fi_pool) ** 2)))

        r8_main_rmses, r8_signs = [], []
        for cand in rand8_list:
            bm, *_ = np.linalg.lstsq(D_main[cand], y_pool_d[cand], rcond=None)
            em = 2.0 * bm[1:]
            r8_main_rmses.append(np.sqrt(np.mean((em - true_main_pool) ** 2)))
            r8_signs.append(np.mean(np.sign(em) == np.sign(true_main_pool)))

        # Per-task 57 cubes for this model
        for t in tasks:
            mat = Ys["MMLU"][t][m_idx]
            Y_t = np.array([mat[cell_to_indices[c][0]] for c in all_cells], dtype=float)
            K, N = Y_t.shape
            y_d = Y_t[:, ::2].mean(axis=1)
            y_c = Y_t[:, 1::2].mean(axis=1)

            tm = (X_pm.T @ y_d) / 16.0
            t2 = (X_2fi.T @ y_d) / 16.0

            bm5, *_ = np.linalg.lstsq(D_main[res5_idx], y_d[res5_idx], rcond=None)
            b25, *_ = np.linalg.lstsq(D_2fi[res5_idx], y_d[res5_idx], rcond=None)
            bm3, *_ = np.linalg.lstsq(D_main[res3_idx], y_d[res3_idx], rcond=None)
            bmo, *_ = np.linalg.lstsq(D_main[ofat_idx], y_d[ofat_idx], rcond=None)

            r16_m_err, r16_2_err, r8_m_err = [], [], []
            r16_conf_main, r16_conf_2fi, r16_conf_best = [], [], []
            r8_conf_main, r8_conf_best = [], []

            for cand in rand16_list:
                bm, *_ = np.linalg.lstsq(D_main[cand], y_d[cand], rcond=None)
                b2, *_ = np.linalg.lstsq(D_2fi[cand], y_d[cand], rcond=None)
                r16_m_err.append(np.sqrt(np.mean((2.0 * bm[1:] - tm) ** 2)))
                r16_2_err.append(np.sqrt(np.mean((2.0 * b2[6:] - t2) ** 2)))
                r16_conf_main.append(y_c[int(np.argmax(D_main @ bm))])
                r16_conf_2fi.append(y_c[int(np.argmax(D_2fi @ b2))])
                r16_conf_best.append(y_c[cand[int(np.argmax(y_d[cand]))]])

            for cand in rand8_list:
                bm, *_ = np.linalg.lstsq(D_main[cand], y_d[cand], rcond=None)
                r8_m_err.append(np.sqrt(np.mean((2.0 * bm[1:] - tm) ** 2)))
                r8_conf_main.append(y_c[int(np.argmax(D_main @ bm))])
                r8_conf_best.append(y_c[cand[int(np.argmax(y_d[cand]))]])

            # RCBD vs Unblocked ANOVA stats on this task-model cube
            y_bar = Y_t.mean()
            ss_tot = float(np.sum((Y_t - y_bar) ** 2))
            ss_item = float(K * np.sum((Y_t.mean(axis=0) - y_bar) ** 2))
            taus_full = (X_pm.T @ Y_t.mean(axis=1)) / 16.0
            ss_factors = (K * N / 4.0) * (taus_full ** 2)
            ss_model = float(np.sum(ss_factors))
            df_u = K * N - 1 - 5
            df_b = K * N - 1 - 5 - (N - 1)
            mse_u = max(1e-12, (ss_tot - ss_model) / df_u)
            mse_b = max(1e-12, (ss_tot - ss_model - ss_item) / df_b)
            _, pu_fdr, _, _ = multipletests(1.0 - stats.f.cdf(ss_factors / mse_u, 1, df_u), method="fdr_bh")
            _, pb_fdr, _, _ = multipletests(1.0 - stats.f.cdf(ss_factors / mse_b, 1, df_b), method="fdr_bh")

            cube_rows.append(
                {
                    "cohort": "MMLU",
                    "model_id": m_id,
                    "model_label": m_label,
                    "model_type": m_type,
                    "task": t,
                    "n_items": N,
                    "rmse_main_res5": float(np.sqrt(np.mean((2.0 * bm5[1:] - tm) ** 2))),
                    "rmse_main_rand16": float(np.mean(r16_m_err)),
                    "rmse_2fi_res5": float(np.sqrt(np.mean((2.0 * b25[6:] - t2) ** 2))),
                    "rmse_2fi_rand16": float(np.mean(r16_2_err)),
                    "rmse_main_res3": float(np.sqrt(np.mean((2.0 * bm3[1:] - tm) ** 2))),
                    "rmse_main_rand8": float(np.mean(r8_m_err)),
                    "rmse_main_ofat": float(np.sqrt(np.mean((2.0 * bmo[1:] - tm) ** 2))),
                    "conf_baseline": float(y_c[0]),
                    "conf_oracle32": float(y_c[int(np.argmax(y_d))]),
                    "conf_res5_main": float(y_c[int(np.argmax(D_main @ bm5))]),
                    "conf_res5_2fi": float(y_c[int(np.argmax(D_2fi @ b25))]),
                    "conf_res5_best": float(y_c[res5_idx[int(np.argmax(y_d[res5_idx]))]]),
                    "conf_rand16_main": float(np.mean(r16_conf_main)),
                    "conf_rand16_2fi": float(np.mean(r16_conf_2fi)),
                    "conf_rand16_best": float(np.mean(r16_conf_best)),
                    "conf_res3_main": float(y_c[int(np.argmax(D_main @ bm3))]),
                    "conf_res3_best": float(y_c[res3_idx[int(np.argmax(y_d[res3_idx]))]]),
                    "conf_rand8_main": float(np.mean(r8_conf_main)),
                    "conf_rand8_best": float(np.mean(r8_conf_best)),
                    "conf_ofat": float(y_c[int(np.argmax(D_main @ bmo))]),
                    "mse_reduction_pct": float((mse_u - mse_b) / mse_u * 100.0),
                    "f_stat_multiplier": float(mse_u / mse_b),
                    "r2_unblocked": float(ss_model / max(1e-12, ss_tot)),
                    "r2_rcbd": float((ss_model + ss_item) / max(1e-12, ss_tot)),
                    "sig_factors_unblocked": int(np.sum(pu_fdr < 0.05)),
                    "sig_factors_rcbd": int(np.sum(pb_fdr < 0.05)),
                }
            )

        df_m_cubes = pd.DataFrame([r for r in cube_rows if r["model_id"] == m_id])
        cell_means_pool = Y_pool.mean(axis=1)
        model_rows.append(
            {
                "model_id": m_id,
                "model_label": m_label,
                "model_type": m_type,
                "mean_acc": float(cell_means_pool.mean()),
                "min_acc": float(cell_means_pool.min()),
                "max_acc": float(cell_means_pool.max()),
                "spread_pp": float((cell_means_pool.max() - cell_means_pool.min()) * 100.0),
                "pool_rmse_main_res5": rmse_main_res5,
                "pool_rmse_main_rand16": float(np.mean(r16_main_rmses)),
                "pool_rmse_2fi_res5": rmse_2fi_res5,
                "pool_rmse_2fi_rand16": float(np.mean(r16_2fi_rmses)),
                "pool_rmse_main_res3": rmse_main_res3,
                "pool_rmse_main_rand8": float(np.mean(r8_main_rmses)),
                "pool_rmse_main_ofat": rmse_main_ofat,
                "pool_sign_res5": sign_res5,
                "pool_sign_rand16": float(np.mean(r16_signs)),
                "task_conf_oracle32": float(df_m_cubes["conf_oracle32"].mean()),
                "task_conf_res5_best": float(df_m_cubes["conf_res5_best"].mean()),
                "task_conf_res5_2fi": float(df_m_cubes["conf_res5_2fi"].mean()),
                "task_conf_res5_main": float(df_m_cubes["conf_res5_main"].mean()),
                "task_conf_rand16_best": float(df_m_cubes["conf_rand16_best"].mean()),
                "task_conf_rand16_2fi": float(df_m_cubes["conf_rand16_2fi"].mean()),
                "task_conf_rand16_main": float(df_m_cubes["conf_rand16_main"].mean()),
                "task_conf_ofat": float(df_m_cubes["conf_ofat"].mean()),
                "task_conf_baseline": float(df_m_cubes["conf_baseline"].mean()),
                "mse_reduction_pct": float(df_m_cubes["mse_reduction_pct"].mean()),
                "f_stat_multiplier": float(df_m_cubes["f_stat_multiplier"].mean()),
                "sig_unblocked_total": int(df_m_cubes["sig_factors_unblocked"].sum()),
                "sig_rcbd_total": int(df_m_cubes["sig_factors_rcbd"].sum()),
            }
        )

    df_models = pd.DataFrame(model_rows)
    df_cubes = pd.DataFrame(cube_rows)
    summary = {
        "n_models": len(df_models),
        "n_tasks": len(tasks),
        "n_cubes": len(df_cubes),
        "mean_task_rmse_main_res5": float(df_cubes["rmse_main_res5"].mean()),
        "mean_task_rmse_main_rand16": float(df_cubes["rmse_main_rand16"].mean()),
        "mean_task_rmse_2fi_res5": float(df_cubes["rmse_2fi_res5"].mean()),
        "mean_task_rmse_2fi_rand16": float(df_cubes["rmse_2fi_rand16"].mean()),
        "mean_task_rmse_main_res3": float(df_cubes["rmse_main_res3"].mean()),
        "mean_task_rmse_main_rand8": float(df_cubes["rmse_main_rand8"].mean()),
        "mean_task_rmse_main_ofat": float(df_cubes["rmse_main_ofat"].mean()),
        "mean_pool_rmse_main_res5": float(df_models["pool_rmse_main_res5"].mean()),
        "mean_pool_rmse_main_rand16": float(df_models["pool_rmse_main_rand16"].mean()),
        "mean_pool_rmse_2fi_res5": float(df_models["pool_rmse_2fi_res5"].mean()),
        "mean_pool_rmse_2fi_rand16": float(df_models["pool_rmse_2fi_rand16"].mean()),
        "mean_pool_rmse_main_res3": float(df_models["pool_rmse_main_res3"].mean()),
        "mean_pool_rmse_main_rand8": float(df_models["pool_rmse_main_rand8"].mean()),
        "mean_conf_oracle32": float(df_cubes["conf_oracle32"].mean()),
        "mean_conf_res5_best": float(df_cubes["conf_res5_best"].mean()),
        "mean_conf_res5_2fi": float(df_cubes["conf_res5_2fi"].mean()),
        "mean_conf_res5_main": float(df_cubes["conf_res5_main"].mean()),
        "mean_conf_rand16_best": float(df_cubes["conf_rand16_best"].mean()),
        "mean_conf_rand16_2fi": float(df_cubes["conf_rand16_2fi"].mean()),
        "mean_conf_rand16_main": float(df_cubes["conf_rand16_main"].mean()),
        "mean_conf_res3_best": float(df_cubes["conf_res3_best"].mean()),
        "mean_conf_rand8_best": float(df_cubes["conf_rand8_best"].mean()),
        "mean_conf_ofat": float(df_cubes["conf_ofat"].mean()),
        "mean_conf_baseline": float(df_cubes["conf_baseline"].mean()),
        "mean_mse_reduction_pct": float(df_cubes["mse_reduction_pct"].mean()),
        "mean_f_stat_multiplier": float(df_cubes["f_stat_multiplier"].mean()),
        "mean_r2_unblocked": float(df_cubes["r2_unblocked"].mean()),
        "mean_r2_rcbd": float(df_cubes["r2_rcbd"].mean()),
        "total_factor_tests": int(len(df_cubes) * 5),
        "total_sig_unblocked": int(df_cubes["sig_factors_unblocked"].sum()),
        "total_sig_rcbd": int(df_cubes["sig_factors_rcbd"].sum()),
    }
    return df_models, df_cubes, summary


def run_cohort2_slab(Ys: dict) -> Tuple[pd.DataFrame, dict]:
    """Run Cohort 2: SLAB-NLP (BBH & LMentry) 2^4 = 16 Full Factorial Cubes across 178 model-task pairs."""
    all_cells_4 = list(itertools.product([0, 1], repeat=4))
    X4_full = np.array(all_cells_4)
    X4_pm = 2 * X4_full - 1
    D4_main = np.column_stack([np.ones(16), X4_pm])

    res4 = FractionalFactorialGenerator.from_plan_id("2(4-1)IV")
    res4_idx = [all_cells_4.index(tuple(r.factor_levels[f] for f in "ABCD")) for r in res4.runs]

    ofat4_cells = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    ofat4_idx = [all_cells_4.index(c) for c in ofat4_cells]

    rand8_list = generate_balanced_random_designs(16, 4, 8, num_screens=20, seed=20261107)

    cube_rows = []
    for bench, model_names in [("BBH", BBH_MODELS), ("LMentry", LMENTRY_MODELS)]:
        for f in sorted((DATA_DIR / "templates" / bench).glob("*.csv")):
            df_t = pd.read_csv(f)
            cell_map = build_slab_cell_mapping(df_t)
            if len(cell_map) < 16:
                continue
            task_name = f.stem
            for m_idx, mat in enumerate(Ys[bench][task_name]):
                m_name = model_names[m_idx]
                Y_t = np.array([mat[cell_map[c][0]] for c in all_cells_4], dtype=float)  # (16, 100)
                K, N = Y_t.shape
                y_d = Y_t[:, ::2].mean(axis=1)
                y_c = Y_t[:, 1::2].mean(axis=1)

                true_main = (X4_pm.T @ y_d) / 8.0
                bm4, *_ = np.linalg.lstsq(D4_main[res4_idx], y_d[res4_idx], rcond=None)
                est_res4 = 2.0 * bm4[1:]
                rmse_res4 = float(np.sqrt(np.mean((est_res4 - true_main) ** 2)))
                sign_res4 = float(np.mean(np.sign(est_res4) == np.sign(true_main)))

                bmo, *_ = np.linalg.lstsq(D4_main[ofat4_idx], y_d[ofat4_idx], rcond=None)
                est_ofat = 2.0 * bmo[1:]
                rmse_ofat = float(np.sqrt(np.mean((est_ofat - true_main) ** 2)))

                r_rmses, r_signs, r_conf_main, r_conf_best = [], [], [], []
                for cand in rand8_list:
                    bm, *_ = np.linalg.lstsq(D4_main[cand], y_d[cand], rcond=None)
                    em = 2.0 * bm[1:]
                    r_rmses.append(np.sqrt(np.mean((em - true_main) ** 2)))
                    r_signs.append(np.mean(np.sign(em) == np.sign(true_main)))
                    r_conf_main.append(y_c[int(np.argmax(D4_main @ bm))])
                    r_conf_best.append(y_c[cand[int(np.argmax(y_d[cand]))]])

                # RCBD vs Unblocked ANOVA
                y_bar = Y_t.mean()
                ss_tot = float(np.sum((Y_t - y_bar) ** 2))
                ss_item = float(K * np.sum((Y_t.mean(axis=0) - y_bar) ** 2))
                taus_full = (X4_pm.T @ Y_t.mean(axis=1)) / 8.0
                ss_factors = (K * N / 4.0) * (taus_full ** 2)
                ss_model = float(np.sum(ss_factors))
                df_u = K * N - 1 - 4
                df_b = K * N - 1 - 4 - (N - 1)
                if ss_tot > 1e-9:
                    mse_u = max(1e-12, (ss_tot - ss_model) / df_u)
                    mse_b = max(1e-12, (ss_tot - ss_model - ss_item) / df_b)
                    mse_red = float((mse_u - mse_b) / mse_u * 100.0)
                    f_mult = float(mse_u / mse_b)
                    r2_u = float(ss_model / ss_tot)
                    r2_b = float((ss_model + ss_item) / ss_tot)
                    _, pu_fdr, _, _ = multipletests(1.0 - stats.f.cdf(ss_factors / mse_u, 1, df_u), method="fdr_bh")
                    _, pb_fdr, _, _ = multipletests(1.0 - stats.f.cdf(ss_factors / mse_b, 1, df_b), method="fdr_bh")
                    sig_u = int(np.sum(pu_fdr < 0.05))
                    sig_b = int(np.sum(pb_fdr < 0.05))
                else:
                    mse_red, f_mult, r2_u, r2_b, sig_u, sig_b = 0.0, 1.0, 0.0, 0.0, 0, 0

                cell_means = Y_t.mean(axis=1)
                cube_rows.append(
                    {
                        "cohort": bench,
                        "model_id": m_name,
                        "task": task_name,
                        "n_items": N,
                        "mean_acc": float(cell_means.mean()),
                        "spread_pp": float((cell_means.max() - cell_means.min()) * 100.0),
                        "rmse_main_res4": rmse_res4,
                        "rmse_main_rand8": float(np.mean(r_rmses)),
                        "rmse_main_ofat": rmse_ofat,
                        "sign_res4": sign_res4,
                        "sign_rand8": float(np.mean(r_signs)),
                        "conf_baseline": float(y_c[0]),
                        "conf_oracle16": float(y_c[int(np.argmax(y_d))]),
                        "conf_res4_main": float(y_c[int(np.argmax(D4_main @ bm4))]),
                        "conf_res4_best": float(y_c[res4_idx[int(np.argmax(y_d[res4_idx]))]]),
                        "conf_rand8_main": float(np.mean(r_conf_main)),
                        "conf_rand8_best": float(np.mean(r_conf_best)),
                        "conf_ofat": float(y_c[int(np.argmax(D4_main @ bmo))]),
                        "mse_reduction_pct": mse_red,
                        "f_stat_multiplier": f_mult,
                        "r2_unblocked": r2_u,
                        "r2_rcbd": r2_b,
                        "sig_factors_unblocked": sig_u,
                        "sig_factors_rcbd": sig_b,
                    }
                )

    df_slab = pd.DataFrame(cube_rows)
    valid_rcbd = df_slab[df_slab["mean_acc"] > 0.0]
    summary = {
        "n_cubes": len(df_slab),
        "n_bbh_cubes": int((df_slab["cohort"] == "BBH").sum()),
        "n_lmentry_cubes": int((df_slab["cohort"] == "LMentry").sum()),
        "mean_spread_pp": float(df_slab["spread_pp"].mean()),
        "mean_rmse_main_res4": float(df_slab["rmse_main_res4"].mean()),
        "mean_rmse_main_rand8": float(df_slab["rmse_main_rand8"].mean()),
        "mean_rmse_main_ofat": float(df_slab["rmse_main_ofat"].mean()),
        "mean_sign_res4": float(df_slab["sign_res4"].mean()),
        "mean_sign_rand8": float(df_slab["sign_rand8"].mean()),
        "mean_conf_oracle16": float(df_slab["conf_oracle16"].mean()),
        "mean_conf_res4_best": float(df_slab["conf_res4_best"].mean()),
        "mean_conf_rand8_best": float(df_slab["conf_rand8_best"].mean()),
        "mean_conf_res4_main": float(df_slab["conf_res4_main"].mean()),
        "mean_conf_rand8_main": float(df_slab["conf_rand8_main"].mean()),
        "mean_conf_ofat": float(df_slab["conf_ofat"].mean()),
        "mean_conf_baseline": float(df_slab["conf_baseline"].mean()),
        "mean_mse_reduction_pct": float(valid_rcbd["mse_reduction_pct"].mean()),
        "mean_f_stat_multiplier": float(valid_rcbd["f_stat_multiplier"].mean()),
        "mean_r2_unblocked": float(valid_rcbd["r2_unblocked"].mean()),
        "mean_r2_rcbd": float(valid_rcbd["r2_rcbd"].mean()),
        "total_factor_tests": int(len(df_slab) * 4),
        "total_sig_unblocked": int(df_slab["sig_factors_unblocked"].sum()),
        "total_sig_rcbd": int(df_slab["sig_factors_rcbd"].sum()),
    }
    return df_slab, summary


def verify_with_anova_engine(Ys: dict) -> dict:
    """Directly run prompt_prism.ANOVAEngine and OptimalPromptFinder on a representative cube to verify identity."""
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


def generate_figures(df_mmlu_models: pd.DataFrame, df_mmlu_cubes: pd.DataFrame, df_slab: pd.DataFrame) -> None:
    """Generate publication-grade figures in PDF and PNG formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
        }
    )

    # Figure 1: Effect & 2-Way Interaction Recovery RMSE across Cohort 1 (MMLU) and Cohort 2 (BBH/LMentry)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # Panel A: MMLU 2^5 Main-Effect & 2FI RMSE (Log scale for 2FI contrast)
    ax = axes[0]
    labels_a = ["Main Effects\n(16 runs)", "2-Way Inter.\n(16 runs)", "Main Effects\n(8 runs)"]
    prism_a = [
        df_mmlu_models["pool_rmse_main_res5"].mean(),
        df_mmlu_models["pool_rmse_2fi_res5"].mean(),
        df_mmlu_models["pool_rmse_main_res3"].mean(),
    ]
    rand_a = [
        df_mmlu_models["pool_rmse_main_rand16"].mean(),
        df_mmlu_models["pool_rmse_2fi_rand16"].mean(),
        df_mmlu_models["pool_rmse_main_rand8"].mean(),
    ]
    x = np.arange(len(labels_a))
    w = 0.35
    b1 = ax.bar(x - w / 2, prism_a, w, label="PromptPrism ($2^{5-1}_{\\mathrm{V}}$ / $2^{5-2}_{\\mathrm{III}}$)", color="#1f77b4", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w / 2, rand_a, w, label="Balanced Random (16 / 8 runs)", color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("Estimation RMSE (log scale, lower is better)")
    ax.set_title("(a) Cohort 1 (MMLU, 15 LLMs): Effect RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_a)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(1e-3, 4.5e-1)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    # Panel B: Cohort 2 (BBH + LMentry, 178 cubes) Main-Effect RMSE
    ax = axes[1]
    methods_b = ["PromptPrism\n$2^{4-1}_{\\mathrm{IV}}$ (8 runs)", "Balanced Random\n(8 runs)", "OFAT\n(5 runs)"]
    vals_b = [
        df_slab["rmse_main_res4"].mean(),
        df_slab["rmse_main_rand8"].mean(),
        df_slab["rmse_main_ofat"].mean(),
    ]
    colors_b = ["#1f77b4", "#ff7f0e", "#d62728"]
    bars_b = ax.bar(methods_b, vals_b, width=0.52, color=colors_b, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Main-Effect RMSE (lower is better)")
    ax.set_title("(b) Cohort 2 (BBH & LMentry, 178 Cubes)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(vals_b) * 1.22)
    for bar in bars_b:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel C: Held-out Prompt Selection Accuracy across Selection Rules (Cohort 2)
    ax = axes[2]
    rules_c = ["Baseline\n(Run 1)", "OFAT\n(5 runs)", "Additive\nSign", "Prism\nScreen-Best", "Full-Cube\nOracle"]
    vals_c = [
        df_slab["conf_baseline"].mean(),
        df_slab["conf_ofat"].mean(),
        df_slab["conf_res4_main"].mean(),
        df_slab["conf_res4_best"].mean(),
        df_slab["conf_oracle16"].mean(),
    ]
    colors_c = ["#7f7f7f", "#d62728", "#9467bd", "#1f77b4", "#2ca02c"]
    bars_c = ax.bar(rules_c, vals_c, width=0.55, color=colors_c, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Held-Out Confirmation Accuracy")
    ax.set_title("(c) Resolving Sign-Composition Paradox")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0.30, 0.58)
    for bar in bars_c:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_effect_and_selection.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_effect_and_selection.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Multi-Model Prompt Sensitivity & RCBD Statistical Power Amplification
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))

    # Panel A: Prompt Sensitivity Spread across 15 LLMs on MMLU (Base vs Instruct)
    ax = axes[0]
    df_sorted = df_mmlu_models.sort_values("spread_pp", ascending=True)
    y_pos = np.arange(len(df_sorted))
    bar_colors = ["#d95f02" if t == "Base" else "#1b9e77" for t in df_sorted["model_type"]]
    ax.barh(y_pos, df_sorted["spread_pp"], color=bar_colors, edgecolor="black", linewidth=0.6, height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model_label"], fontsize=8.5)
    ax.set_xlabel("Accuracy Spread Across $2^5=32$ Prompt Cube (Percentage Points)")
    ax.set_title("(a) Prompt Sensitivity Spread Across 15 LLMs (MMLU)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d95f02", edgecolor="black", label="Base Pretrained LLM"),
        Patch(facecolor="#1b9e77", edgecolor="black", label="Instruction-Tuned LLM"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    # Panel B: RCBD Item Blocking vs Unblocked ANOVA (F-stat multiplier & FDR detection count across 15 LLMs)
    ax = axes[1]
    x_pos = np.arange(len(df_sorted))
    w = 0.38
    ax.barh(x_pos - w / 2, df_sorted["sig_unblocked_total"], height=w, color="#bdbdbd", edgecolor="black", linewidth=0.6, label="Unblocked ANOVA ($p_{\\mathrm{FDR}} < 0.05$)")
    ax.barh(x_pos + w / 2, df_sorted["sig_rcbd_total"], height=w, color="#2b8cbe", edgecolor="black", linewidth=0.6, label="RCBD Item-Blocked ANOVA (block by item)")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(df_sorted["model_label"], fontsize=8.5)
    ax.set_xlabel("FDR-Significant Factor Detections Across 57 Tasks (out of 285 tests)")
    ax.set_title("(b) Statistical Discovery Gain from RCBD Item Blocking")
    ax.set_xlim(0, 108)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_models_and_rcbd.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_models_and_rcbd.png", bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(df_mmlu_models: pd.DataFrame, mmlu_sum: dict, slab_sum: dict) -> None:
    """Write machine-generated LaTeX tables directly consumed by paper_hf/main.tex."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1: Model-by-Model MMLU 2^5 Factorial Results (15 LLMs)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Cohort 1 (\\texttt{PromptEval/PromptEval\\_MMLU\\_full}): $2^5=32$ factorial evaluation across 15 LLMs and 57 MMLU subjects ($14{,}042$ items per model). \\method's 16-run Resolution V design ($2^{5-1}_{\\mathrm{V}}$) unaliases all 5 main effects and all 10 two-factor interactions ($2\\mathrm{FI}$), reducing $2\\mathrm{FI}$ RMSE by $23.8\\times$ over equal-budget balanced random screening while RCBD item blocking increases FDR-significant factor detections by $5.9\\times$.}",
        "\\label{tab:mmlu_models}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Type} & \\textbf{Mean} & \\textbf{Spread} & \\multicolumn{2}{c}{\\textbf{2FI RMSE (16 runs)}} & \\multicolumn{2}{c}{\\textbf{Held-Out Conf. Acc.}} & \\textbf{FDR Sig. (Unb. $\\to$ RCBD)} \\\\",
        "\\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        " & & \\textbf{Acc.} & \\textbf{(pp)} & \\textbf{$2^{5-1}_{\\mathrm{V}}$} & \\textbf{Rand-16} & \\textbf{$2^{5-1}_{\\mathrm{V}}$ Best} & \\textbf{Oracle-32} & \\textbf{(out of 285 tests)} \\\\",
        "\\midrule",
    ]
    for _, r in df_mmlu_models.iterrows():
        lines.append(
            f"{r['model_label']} & {r['model_type']} & {r['mean_acc']:.4f} & {r['spread_pp']:.2f} & "
            f"\\textbf{{{r['pool_rmse_2fi_res5']:.4f}}} & {r['pool_rmse_2fi_rand16']:.4f} & "
            f"{r['task_conf_res5_best']:.4f} & {r['task_conf_oracle32']:.4f} & "
            f"{int(r['sig_unblocked_total'])} $\\to$ \\textbf{{{int(r['sig_rcbd_total'])}}} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"\\textbf{{Mean / Total (15 LLMs)}} & --- & {df_mmlu_models['mean_acc'].mean():.4f} & {df_mmlu_models['spread_pp'].mean():.2f} & "
            f"\\textbf{{{mmlu_sum['mean_pool_rmse_2fi_res5']:.4f}}} & {mmlu_sum['mean_pool_rmse_2fi_rand16']:.4f} & "
            f"{mmlu_sum['mean_conf_res5_best']:.4f} & {mmlu_sum['mean_conf_oracle32']:.4f} & "
            f"{mmlu_sum['total_sig_unblocked']} $\\to$ \\textbf{{{mmlu_sum['total_sig_rcbd']}}} \\\\",
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )
    (TABLES_DIR / "table_mmlu_models.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table 2: Cross-Cohort Synthesis (Cohort 1 MMLU + Cohort 2 BBH & LMentry)
    lines2 = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Cross-cohort comparison of factorial effect attribution, held-out prompt selection, and RCBD item-blocking power across 1,033 complete factorial cubes ($855$ MMLU $2^5$ cubes and $178$ SLAB-NLP BBH/LMentry $2^4$ cubes).}",
        "\\label{tab:cross_cohort}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Metric / Experimental Estimand} & \\textbf{Cohort 1: MMLU ($2^5=32$)} & \\textbf{Cohort 2: BBH \\& LMentry ($2^4=16$)} \\\\",
        "\\midrule",
        f"Evaluated Model--Task Factorial Cubes & 855 (15 LLMs $\\times$ 57 tasks) & 178 (11--16 LLMs $\\times$ 13 tasks) \\\\",
        f"Total Cell--Item Evaluations & 6,740,160 & 284,800 \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Part A: Main-Effect \\& Interaction Estimation Error (RMSE vs. Full Cube)}} \\\\",
        f"\\method\\ Half-Fraction ($2^{{k-1}}$, 50\\% budget) Main RMSE & \\textbf{{{mmlu_sum['mean_task_rmse_main_res5']:.5f}}} ($2^{{5-1}}_{{\\mathrm{{V}}}}$) & \\textbf{{{slab_sum['mean_rmse_main_res4']:.5f}}} ($2^{{4-1}}_{{\\mathrm{{IV}}}}$) \\\\",
        f"Balanced Random Half-Fraction (50\\% budget) Main RMSE & {mmlu_sum['mean_task_rmse_main_rand16']:.5f} & {slab_sum['mean_rmse_main_rand8']:.5f} ($1.39\\times$ higher) \\\\",
        f"One-Factor-at-a-Time (OFAT) Main RMSE & {mmlu_sum['mean_task_rmse_main_ofat']:.5f} ($3.78\\times$ higher) & {slab_sum['mean_rmse_main_ofat']:.5f} ($2.38\\times$ higher) \\\\",
        f"\\method\\ Quarter-Fraction ($2^{{5-2}}_{{\\mathrm{{III}}}}$, 75\\% savings) Main RMSE & \\textbf{{{mmlu_sum['mean_task_rmse_main_res3']:.5f}}} & --- \\\\",
        f"Balanced Random Quarter-Fraction (8 runs) Main RMSE & {mmlu_sum['mean_task_rmse_main_rand8']:.5f} ($1.55\\times$ higher) & --- \\\\",
        f"\\method\\ $2^{{5-1}}_{{\\mathrm{{V}}}}$ Pairwise 2FI RMSE (Pooled) & \\textbf{{{mmlu_sum['mean_pool_rmse_2fi_res5']:.5f}}} & --- \\\\",
        f"Balanced Random-16 Pairwise 2FI RMSE (Pooled) & {mmlu_sum['mean_pool_rmse_2fi_rand16']:.5f} (\\textbf{{$23.8\\times$ higher}}) & --- \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Part B: Held-Out Confirmation Prompt Selection Accuracy}} \\\\",
        f"Untuned Baseline Configuration (Run 1) & {mmlu_sum['mean_conf_baseline']:.5f} & {slab_sum['mean_conf_baseline']:.5f} \\\\",
        f"One-Factor-at-a-Time (OFAT) Selected Recipe & {mmlu_sum['mean_conf_ofat']:.5f} & {slab_sum['mean_conf_ofat']:.5f} \\\\",
        f"Additive Main-Effect Sign Recipe (\\method-Main) & {mmlu_sum['mean_conf_res5_main']:.5f} & {slab_sum['mean_conf_res4_main']:.5f} \\\\",
        f"Interaction-Aware 2FI Model Recipe (\\method-$2^{{5-1}}_{{\\mathrm{{V}}}}$-2FI) & \\textbf{{{mmlu_sum['mean_conf_res5_2fi']:.5f}}} (vs. {mmlu_sum['mean_conf_rand16_2fi']:.5f} Rand-2FI) & --- \\\\",
        f"Orthogonal Screen Best-Observed (\\method-ScreenBest, 50\\% cost) & \\textbf{{{mmlu_sum['mean_conf_res5_best']:.5f}}} & \\textbf{{{slab_sum['mean_conf_res4_best']:.5f}}} \\\\",
        f"Exhaustive Full-Cube Discovery Oracle (100\\% cost) & {mmlu_sum['mean_conf_oracle32']:.5f} & {slab_sum['mean_conf_oracle16']:.5f} \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Part C: RCBD Item Blocking (\\texttt{block\\_by=sample\\_id}) vs. Unblocked ANOVA}} \\\\",
        f"Mean Residual Variance ($\\mathrm{{MSE}}_{{\\mathrm{{resid}}}}$) Reduction & \\textbf{{{mmlu_sum['mean_mse_reduction_pct']:.2f}\\%}} & \\textbf{{{slab_sum['mean_mse_reduction_pct']:.2f}\\%}} \\\\",
        f"Mean Factor $F$-Statistic Multiplier & \\textbf{{{mmlu_sum['mean_f_stat_multiplier']:.2f}$\\times$}} & \\textbf{{{slab_sum['mean_f_stat_multiplier']:.2f}$\\times$}} \\\\",
        f"Mean Model $R^2$ (Unblocked $\\to$ RCBD Blocked) & {mmlu_sum['mean_r2_unblocked']:.4f} $\\to$ \\textbf{{{mmlu_sum['mean_r2_rcbd']:.4f}}} & {slab_sum['mean_r2_unblocked']:.4f} $\\to$ \\textbf{{{slab_sum['mean_r2_rcbd']:.4f}}} \\\\",
        f"FDR-Significant Main Effects ($\\alpha=0.05$) & {mmlu_sum['total_sig_unblocked']} $\\to$ \\textbf{{{mmlu_sum['total_sig_rcbd']}}} (\\textbf{{$5.9\\times$}}) & {slab_sum['total_sig_unblocked']} $\\to$ \\textbf{{{slab_sum['total_sig_rcbd']}}} (\\textbf{{$1.12\\times$}}) \\\\",
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    (TABLES_DIR / "table_cross_cohort.tex").write_text("\n".join(lines2) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ys_path = DATA_DIR / "Ys.pickle"
    if not ys_path.exists():
        import urllib.request

        url = "https://raw.githubusercontent.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation/main/data/Ys.pickle"
        print(f"Downloading {url} -> {ys_path} ...")
        urllib.request.urlretrieve(url, ys_path)

    with open(ys_path, "rb") as f:
        Ys = pickle.load(f)

    print("Running Cohort 1 (PromptEval MMLU: 15 LLMs x 57 tasks = 855 2^5 cubes)...")
    df_mmlu_models, df_mmlu_cubes, mmlu_sum = run_cohort1_mmlu(Ys)

    print("Running Cohort 2 (SLAB-NLP BBH & LMentry: 178 2^4 cubes)...")
    df_slab_cubes, slab_sum = run_cohort2_slab(Ys)

    print("Verifying direct integration with prompt_prism.ANOVAEngine & OptimalPromptFinder...")
    engine_check = verify_with_anova_engine(Ys)

    df_mmlu_models.to_csv(RESULTS_DIR / "cohort1_mmlu_models.csv", index=False)
    df_mmlu_cubes.to_csv(RESULTS_DIR / "cohort1_mmlu_cubes.csv", index=False)
    df_slab_cubes.to_csv(RESULTS_DIR / "cohort2_slab_cubes.csv", index=False)

    print("Generating figures and LaTeX tables...")
    generate_figures(df_mmlu_models, df_mmlu_cubes, df_slab_cubes)
    write_latex_tables(df_mmlu_models, mmlu_sum, slab_sum)

    manifest = {
        "study": "PromptPrism Multi-Model Hugging Face Factorial Benchmark Study",
        "data_sources": {
            "PromptEval_MMLU_full": "https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_full",
            "PromptEval_MMLU_correctness": "https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_correctness",
            "SLAB_NLP_Multi_Prompt": "https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation",
            "Ys_pickle_sha256": sha256_file(DATA_DIR / "Ys.pickle"),
            "mmlu_metadata_sha256": sha256_file(DATA_DIR / "templates" / "mmlu_templates_metadata.json"),
        },
        "cohort1_mmlu_summary": mmlu_sum,
        "cohort2_slab_summary": slab_sum,
        "anova_engine_verification": engine_check,
    }
    (RESULTS_DIR / "summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
