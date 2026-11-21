"""
PromptPrism 2.0 Calibrated Attribution, Item-Sampling Noise-Floor, and Token-Pruning Diagnostics.

Grounded in the 1,033-cube retrospective empirical-null calibration audit and 501,300-call
prospective Real-v4 study:
- Uses contrast-specific item-clustered t-tests (`df = N - 1`, exact-null zero-variance contrasts -> p = 1.0,
  unanimous non-zero zero-variance paired contrasts -> exact Rademacher p = 2^(1 - N))
  rather than pooled fixed-block RCBD ANOVA (which inflates global BH FDR to 18.4% under
  heterogeneous contrast variance).
- Adds optional Rademacher sign-flip randomization checks (recommended whenever N < 100).
- Decomposes observed main-effect RMS into expected item-sampling noise vs. noise-corrected signal.
- Uses paired Two-One-Sided-Tests (TOST) against an explicit practical equivalence margin so
  factors are marked `SAFE_TO_PRUNE` only when statistically negligible, separating true bloat
  from underpowered `INCONCLUSIVE_NEED_MORE_ITEMS` factors.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

PRUNING_VERDICTS: Tuple[str, ...] = (
    "KEEP",
    "KEEP_TARGETED",
    "REMOVE_HARMFUL",
    "CONFLICT_REVIEW",
    "UNTESTED",
    "SAFE_TO_PRUNE",
    "INCONCLUSIVE_NEED_MORE_ITEMS",
)

VALID_COVERAGE_SOURCES: Tuple[str, ...] = (
    "synthetic",
    "human",
    "log",
    "mixed",
    "none",
)

_ZERO_VAR_T_SENTINEL: float = 1e6


class CalibratedFactorDiagnostic(BaseModel):
    """Per-factor calibrated item-clustered attribution and TOST token-pruning diagnostic."""

    factor_id: str
    mean_level_0: float
    mean_level_1: float
    effect_delta: float
    oriented_effect: float
    se_item_clustered: float
    t_statistic: float
    p_value_clustered: float
    q_value_bh: float
    p_value_sign_flip: Optional[float] = None
    ci95_low: float
    ci95_high: float
    tost_p_value: float
    pruning_verdict: str  # KEEP | KEEP_TARGETED | REMOVE_HARMFUL | CONFLICT_REVIEW | UNTESTED | SAFE_TO_PRUNE | INCONCLUSIVE_NEED_MORE_ITEMS
    token_cost: Optional[int] = None
    n_targeted: Optional[int] = None
    delta_targeted: Optional[float] = None
    oriented_effect_targeted: Optional[float] = None
    se_targeted: Optional[float] = None
    ci95_low_targeted: Optional[float] = None
    ci95_high_targeted: Optional[float] = None
    p_value_targeted: Optional[float] = None
    q_value_bh_targeted: Optional[float] = None
    p_value_sign_flip_targeted: Optional[float] = None
    p_noninferiority_targeted: Optional[float] = None
    coverage_source: str = "none"


class NoiseFloorSummary(BaseModel):
    """Decomposition of observed main-effect RMS into item-sampling noise vs. repeatable signal."""

    n_items: int
    rms_observed: float
    rms_item_noise: float
    rms_signal_corrected: float
    noise_variance_share_pct: float
    small_sample_warning: bool


class CalibratedAttributionAudit(BaseModel):
    """Complete PromptPrism 2.0 calibrated attribution and token-pruning audit report."""

    schema_version: str = "2.1.0"
    target_metric: str
    maximize: bool = True
    alpha: float = 0.05
    equivalence_margin: float = 0.01
    targeted_margin: Optional[float] = None
    min_targeted_items: Optional[int] = None
    baseline_only_full_set: bool = False
    inference_method: str = "item_clustered_contrast_t_test_with_bh_and_tost"
    n_items: int
    n_runs: int
    noise_floor: NoiseFloorSummary
    factors: List[CalibratedFactorDiagnostic] = Field(default_factory=list)
    estimated_prunable_tokens: Optional[int] = None
    test_set_sha256: Optional[str] = None
    coverage_source: str = "none"
    evidence_tier: str = "predicted"
    judge_scored_fraction: Optional[float] = None
    generator_model_id: Optional[str] = None
    model_id: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        direction = "maximize" if self.maximize else "minimize"
        has_coverage_cols = self.coverage_source != "none" or any(
            f.n_targeted is not None for f in self.factors
        )
        items_note = (
            " (baseline items only; targeted items evaluated separately)"
            if self.baseline_only_full_set
            else ""
        )
        lines = [
            f"# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`{self.target_metric}`)",
            "",
            f"- **Optimization Direction:** `{direction}`",
            f"- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha={self.alpha}`)",
            f"- **Pruning Gate:** TOST practical equivalence (`|delta| < {self.equivalence_margin:.4f}`)",
        ]
        if has_coverage_cols and self.targeted_margin is not None:
            min_t_str = (
                f", `min_targeted_items={self.min_targeted_items}`"
                if self.min_targeted_items is not None
                else ""
            )
            lines.append(
                f"- **Targeted Coverage Gate:** One-sided non-inferiority (`oriented_delta_T < {self.targeted_margin:.4f}`{min_t_str}, source=`{self.coverage_source}`)"
            )
        lines.extend(
            [
                f"- **Paired Benchmark Items (`N`):** `{self.n_items}` across `{self.n_runs}` experimental runs{items_note}",
                f"- **Main-Effect Noise Floor:** `RMS_obs={self.noise_floor.rms_observed:.4f}`, "
                f"`RMS_noise={self.noise_floor.rms_item_noise:.4f}`, "
                f"`RMS_signal_corr={self.noise_floor.rms_signal_corrected:.4f}` "
                f"(item noise accounts for **{self.noise_floor.noise_variance_share_pct:.1f}%** of observed variance)",
            ]
        )
        if self.estimated_prunable_tokens is not None:
            lines.append(
                f"- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `{self.estimated_prunable_tokens}` tokens/prompt"
            )
        if self.test_set_sha256:
            lines.append(f"- **Frozen Test Set SHA-256:** `{self.test_set_sha256}`")
        if self.judge_scored_fraction is not None:
            lines.append(
                f"- **LLM-Judge Scored Share:** `{self.judge_scored_fraction * 100.0:.1f}%`"
            )
        for w in self.warnings:
            lines.append(f"- **Warning:** {w}")

        if has_coverage_cols:
            lines.extend(
                [
                    "",
                    "| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Covered N | Δ targeted | NI p (T) | Verdict | Tokens |",
                    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |",
                    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|",
                ]
            )

        for f in self.factors:
            flip_str = (
                f"{f.p_value_sign_flip:.4g}" if f.p_value_sign_flip is not None else "-"
            )
            tok_str = str(f.token_cost) if f.token_cost is not None else "-"
            if has_coverage_cols:
                cov_n_str = str(f.n_targeted) if f.n_targeted is not None else "0"
                delta_t_str = (
                    f"{f.delta_targeted:+.4f}" if f.delta_targeted is not None else "-"
                )
                ni_t_str = (
                    f"{f.p_noninferiority_targeted:.4g}"
                    if f.p_noninferiority_targeted is not None
                    else "-"
                )
                lines.append(
                    f"| **{f.factor_id}** | {f.mean_level_0:.4f} | {f.mean_level_1:.4f} | "
                    f"{f.effect_delta:+.4f} | {f.se_item_clustered:.4f} | "
                    f"[{f.ci95_low:+.4f}, {f.ci95_high:+.4f}] | "
                    f"{f.p_value_clustered:.4g} | {f.q_value_bh:.4g} | {flip_str} | "
                    f"{f.tost_p_value:.4g} | {cov_n_str} | {delta_t_str} | {ni_t_str} | "
                    f"`{f.pruning_verdict}` | {tok_str} |"
                )
            else:
                lines.append(
                    f"| **{f.factor_id}** | {f.mean_level_0:.4f} | {f.mean_level_1:.4f} | "
                    f"{f.effect_delta:+.4f} | {f.se_item_clustered:.4f} | "
                    f"[{f.ci95_low:+.4f}, {f.ci95_high:+.4f}] | "
                    f"{f.p_value_clustered:.4g} | {f.q_value_bh:.4g} | {flip_str} | "
                    f"{f.tost_p_value:.4g} | `{f.pruning_verdict}` | {tok_str} |"
                )

        lines.extend(
            [
                "",
                f"> **Evidence Tier:** `{self.evidence_tier}` (attribution diagnostic on evaluated items only; "
                "validate any pruned or selected prompt on untouched confirmation items and issue a "
                "hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).",
            ]
        )
        return "\n".join(lines)


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray([p_values[i] for i in order], dtype=float)
    q_sorted = np.empty(m, dtype=float)
    running_min = 1.0
    for idx in range(m - 1, -1, -1):
        val = ranked[idx] * m / (idx + 1)
        running_min = min(running_min, val)
        q_sorted[idx] = min(1.0, max(0.0, running_min))
    q_out = np.empty(m, dtype=float)
    for pos, orig_idx in enumerate(order):
        q_out[orig_idx] = q_sorted[pos]
    return [float(x) for x in q_out]


def _exact_unanimous_sign_flip_p(n_eff: int) -> float:
    """Exact two-sided Rademacher sign-flip p-value when all n_eff paired contrasts are identical and non-zero."""
    if n_eff <= 1:
        return 1.0
    if n_eff > 1000:
        return 0.0
    return float(min(1.0, math.ldexp(1.0, 1 - n_eff)))


def _normalize_id_collection(
    values: Optional[Collection[Any]], *, param_name: str
) -> Set[str]:
    """Normalize a collection of sample IDs into a set of strings, rejecting bare str/bytes."""
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        raise TypeError(
            f"{param_name} must be a collection of sample IDs, not a bare {type(values).__name__}: {values!r}"
        )
    return {str(v) for v in values}


def _parse_targets_factors_cell(val: Any) -> List[str]:
    """Parse one cell of a `targets_factors` column into a list of factor IDs/names."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set, np.ndarray)):
        out: List[str] = []
        for item in val:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    if s.startswith("["):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [
                        str(x).strip()
                        for x in parsed
                        if x is not None and str(x).strip()
                    ]
            except Exception:
                continue
    sep = ";" if ";" in s else ","
    return [part.strip() for part in s.split(sep) if part.strip()]


def _extract_coverage_from_column(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    col_name: str = "targets_factors",
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Build factor_id -> set(sample_ids) map from a `targets_factors` dataframe column."""
    factor_set = {str(f) for f in factor_cols}
    cov_map: Dict[str, Set[str]] = {str(f): set() for f in factor_cols}
    unknown_tags: Set[str] = set()

    for item_id, raw_val in zip(df["_prism_item_id"], df[col_name]):
        tags = _parse_targets_factors_cell(raw_val)
        for tag in tags:
            if tag in factor_set:
                cov_map[tag].add(str(item_id))
            else:
                unknown_tags.add(tag)

    return cov_map, unknown_tags


def _compute_targeted_stats(
    df: pd.DataFrame,
    fid: str,
    target_col: str,
    covered_ids: Set[str],
    *,
    sign: float,
    alpha: float,
    targeted_margin: float,
    n_sign_flip: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Compute item-clustered targeted contrast stats and one-sided non-inferiority test on covered items."""
    empty_res: Dict[str, Any] = {
        "n_targeted": 0,
        "delta_targeted": None,
        "oriented_effect_targeted": None,
        "se_targeted": None,
        "ci95_low_targeted": None,
        "ci95_high_targeted": None,
        "p_value_targeted": None,
        "p_value_sign_flip_targeted": None,
        "p_noninferiority_targeted": None,
    }
    if not covered_ids:
        return empty_res

    sub = df[df["_prism_item_id"].isin(covered_ids)]
    if sub.empty:
        return empty_res

    piv = sub.pivot_table(
        index="_prism_item_id", columns=fid, values=target_col, aggfunc="mean"
    )
    if 0 not in piv.columns or 1 not in piv.columns:
        return empty_res

    paired = piv[[0, 1]].dropna()
    n_t = int(len(paired))
    if n_t == 0:
        return empty_res

    c_vec = (paired[1] - paired[0]).to_numpy(dtype=float)
    delta_t = float(np.mean(c_vec))
    oriented_t = float(sign * delta_t)

    if n_t < 2:
        return {
            **empty_res,
            "n_targeted": n_t,
            "delta_targeted": delta_t,
            "oriented_effect_targeted": oriented_t,
        }

    sd_t = float(np.std(c_vec, ddof=1))
    se_t = float(sd_t / np.sqrt(n_t))
    df_t = n_t - 1

    if se_t <= 1e-12:
        ci_low_t = delta_t
        ci_high_t = delta_t
        if abs(delta_t) <= 1e-12:
            p_t = 1.0
            p_flip_t = 1.0 if n_sign_flip > 0 else None
        else:
            p_exact = _exact_unanimous_sign_flip_p(n_t)
            p_t = p_exact
            p_flip_t = p_exact if n_sign_flip > 0 else None
        # Non-inferiority of removing clause: H0: oriented_t >= targeted_margin vs H1: oriented_t < targeted_margin
        p_ni_t = 0.0 if oriented_t < targeted_margin else 1.0
    else:
        t_stat_t = float(delta_t / se_t)
        p_t = float(2.0 * stats.t.sf(abs(t_stat_t), df=df_t))
        t_crit_t = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_t))
        ci_low_t = float(delta_t - t_crit_t * se_t)
        ci_high_t = float(delta_t + t_crit_t * se_t)

        # One-sided non-inferiority of removing clause at level alpha:
        # H0: oriented_t >= targeted_margin (removing loses >= margin)
        # H1: oriented_t < targeted_margin (removing loses < margin)
        t_ni_t = (targeted_margin - oriented_t) / se_t
        p_ni_t = float(stats.t.sf(t_ni_t, df=df_t))

        if n_sign_flip > 0:
            signs = rng.choice((-1.0, 1.0), size=(n_sign_flip, n_t))
            perm_means = (signs @ c_vec) / float(n_t)
            p_flip_t = float(
                (1.0 + np.sum(np.abs(perm_means) >= abs(delta_t) - 1e-15))
                / (n_sign_flip + 1.0)
            )
        else:
            p_flip_t = None

    return {
        "n_targeted": n_t,
        "delta_targeted": delta_t,
        "oriented_effect_targeted": oriented_t,
        "se_targeted": se_t,
        "ci95_low_targeted": ci_low_t,
        "ci95_high_targeted": ci_high_t,
        "p_value_targeted": p_t,
        "p_value_sign_flip_targeted": p_flip_t,
        "p_noninferiority_targeted": p_ni_t,
    }


def run_calibrated_attribution_audit(
    data: pd.DataFrame,
    factor_cols: Sequence[str],
    target_col: str,
    *,
    sample_col: str = "sample_id",
    alpha: float = 0.05,
    maximize: bool = True,
    equivalence_margin: float = 0.01,
    token_costs: Optional[Mapping[str, int]] = None,
    n_sign_flip: int = 2000,
    seed: int = 42,
    coverage: Optional[Mapping[str, Collection[str]]] = None,
    baseline_sample_ids: Optional[Collection[str]] = None,
    min_targeted_items: int = 20,
    targeted_margin: Optional[float] = None,
    coverage_source: Optional[str] = None,
    test_set_sha256: Optional[str] = None,
    judge_scored_fraction: Optional[float] = None,
    generator_model_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> CalibratedAttributionAudit:
    """Run PromptPrism 2.0 calibrated item-clustered attribution, noise-floor, TOST pruning, and coverage-gate audit."""
    if not factor_cols:
        raise ValueError("factor_cols cannot be empty")
    if target_col not in data.columns:
        raise ValueError(f"target_col {target_col!r} not found in dataframe")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must lie in (0, 0.5), got {alpha}")
    if equivalence_margin <= 0.0:
        raise ValueError(f"equivalence_margin must be > 0, got {equivalence_margin}")
    if min_targeted_items < 1:
        raise ValueError(f"min_targeted_items must be >= 1, got {min_targeted_items}")
    if targeted_margin is not None and targeted_margin <= 0.0:
        raise ValueError(f"targeted_margin must be > 0, got {targeted_margin}")
    if coverage_source is not None and coverage_source not in VALID_COVERAGE_SOURCES:
        raise ValueError(
            f"coverage_source must be one of {VALID_COVERAGE_SOURCES}, got {coverage_source!r}"
        )
    if judge_scored_fraction is not None and (
        math.isnan(judge_scored_fraction) or not (0.0 <= judge_scored_fraction <= 1.0)
    ):
        raise ValueError(
            f"judge_scored_fraction must lie in [0.0, 1.0], got {judge_scored_fraction}"
        )

    df = data.copy()
    warnings_list: List[str] = []

    if sample_col in df.columns:
        df["_prism_item_id"] = df[sample_col].astype(str)
    else:
        warnings_list.append(
            f"Column {sample_col!r} not found; treating each row as an independent unpaired observation "
            "(item-clustered pairing unavailable)."
        )
        df["_prism_item_id"] = [f"row_{i}" for i in range(len(df))]

    factor_cols_str = [str(c) for c in factor_cols]
    factor_col_set = set(factor_cols_str)

    for col in factor_cols_str:
        if col not in df.columns:
            raise ValueError(f"Factor column {col!r} not found in dataframe")
        vals = set(df[col].dropna().unique())
        if not vals.issubset({0, 1, 0.0, 1.0, -1, -1.0}):
            raise ValueError(
                f"Factor column {col!r} must be binary ({{0, 1}} or {{-1, +1}}), got {sorted(vals)}"
            )
        if -1 in vals or -1.0 in vals:
            df[col] = (df[col].astype(float) > 0).astype(int)
        else:
            df[col] = df[col].astype(int)

    # Resolve coverage map from explicit argument or dataframe `targets_factors` column
    coverage_map: Optional[Dict[str, Set[str]]] = None
    if coverage is not None:
        normalized_cov = {str(k): v for k, v in coverage.items()}
        raw_cov_keys = set(normalized_cov.keys())
        matched_keys = raw_cov_keys & factor_col_set
        unmatched_keys = sorted(raw_cov_keys - factor_col_set)
        if raw_cov_keys and not matched_keys:
            raise ValueError(
                f"Coverage map keys {sorted(raw_cov_keys)} do not match any factor_cols "
                f"{factor_cols_str}. Check factor ID vs. factor name consistency."
            )
        if unmatched_keys:
            warnings_list.append(
                f"Coverage map contains keys not in factor_cols (ignored): {unmatched_keys}."
            )
        coverage_map = {
            fid: _normalize_id_collection(
                normalized_cov.get(fid), param_name=f"coverage[{fid!r}]"
            )
            for fid in factor_cols_str
        }
    elif "targets_factors" in df.columns:
        coverage_map, unknown_tags = _extract_coverage_from_column(
            df, factor_cols_str, col_name="targets_factors"
        )
        if unknown_tags:
            warnings_list.append(
                f"Column 'targets_factors' contains tags not in factor_cols (ignored): {sorted(unknown_tags)}."
            )

    has_coverage = coverage_map is not None
    eff_coverage_source = (
        coverage_source
        if coverage_source is not None
        else ("human" if has_coverage else "none")
    )
    eff_targeted_margin: Optional[float] = (
        float(targeted_margin)
        if targeted_margin is not None
        else (max(2.0 * equivalence_margin, 0.05) if has_coverage else None)
    )
    eff_min_targeted: Optional[int] = min_targeted_items if has_coverage else None
    evidence_tier = (
        "predicted (synthetic tests)"
        if eff_coverage_source in {"synthetic", "mixed"}
        else "predicted"
    )

    if not has_coverage:
        warnings_list.append(
            "No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised."
        )

    if (
        generator_model_id
        and model_id
        and generator_model_id.strip().lower() == model_id.strip().lower()
    ):
        warnings_list.append(
            f"Generator model ({generator_model_id!r}) equals model under test ({model_id!r}): "
            "synthetic checks may share model-specific blind spots."
        )

    # Separate baseline-only primary estimand F when baseline_sample_ids is provided
    if baseline_sample_ids is not None:
        baseline_id_set = _normalize_id_collection(
            baseline_sample_ids, param_name="baseline_sample_ids"
        )
        df_full = df[df["_prism_item_id"].isin(baseline_id_set)]
        if df_full.empty:
            raise ValueError("baseline_sample_ids matched no rows in dataframe")
        baseline_only_full_set = True
    else:
        df_full = df
        baseline_only_full_set = False

    n_runs = (
        int(df_full["run_id"].nunique())
        if "run_id" in df_full.columns
        else int(df_full[factor_cols_str].drop_duplicates().shape[0])
    )

    sign = 1.0 if maximize else -1.0
    rng = np.random.default_rng(seed)

    raw_stats: List[Dict[str, Any]] = []
    n_items_per_factor: List[int] = []
    obs_sq_list: List[float] = []
    noise_var_list: List[float] = []

    for fid in factor_cols_str:
        # Compute paired per-item mean at level 1 minus level 0 on primary estimand F
        piv = df_full.pivot_table(
            index="_prism_item_id", columns=fid, values=target_col, aggfunc="mean"
        )
        if 0 in piv.columns and 1 in piv.columns:
            paired = piv[[0, 1]].dropna()
            if len(paired) >= 2:
                c_vec = (paired[1] - paired[0]).to_numpy(dtype=float)
                mean_0 = float(paired[0].mean())
                mean_1 = float(paired[1].mean())
                n_eff = int(len(c_vec))
                delta = float(np.mean(c_vec))
                sd = float(np.std(c_vec, ddof=1))
                se = float(sd / np.sqrt(n_eff))
                df_t = n_eff - 1
            else:
                g0 = (
                    df_full.loc[df_full[fid] == 0, target_col]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                g1 = (
                    df_full.loc[df_full[fid] == 1, target_col]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                c_vec = None
                mean_0 = float(np.mean(g0)) if g0.size else 0.0
                mean_1 = float(np.mean(g1)) if g1.size else 0.0
                delta = mean_1 - mean_0
                n_eff = int(min(g0.size, g1.size))
                v0 = float(np.var(g0, ddof=1)) / max(1, g0.size) if g0.size > 1 else 0.0
                v1 = float(np.var(g1, ddof=1)) / max(1, g1.size) if g1.size > 1 else 0.0
                se = float(np.sqrt(v0 + v1))
                df_t = max(1, g0.size + g1.size - 2)
        else:
            raise ValueError(
                f"Factor {fid!r} must contain both level 0 and level 1 observations"
            )

        n_items_per_factor.append(n_eff)
        obs_sq_list.append(delta**2)
        noise_var_list.append(se**2)

        # Zero-variance guard:
        # - Exact null (delta == 0, se == 0) -> conservative p = 1.0 (matches empirical-null suite).
        # - Unanimous non-zero paired contrast (c_vec len >= 2, |delta| > 0, se == 0) -> exact Rademacher p = 2^(1 - n).
        if se <= 1e-12:
            ci_low = delta
            ci_high = delta
            p_tost = 0.0 if abs(delta) < equivalence_margin else 1.0
            if c_vec is not None and len(c_vec) >= 2 and abs(delta) > 1e-12:
                t_stat = float(math.copysign(_ZERO_VAR_T_SENTINEL, delta))
                p_exact = _exact_unanimous_sign_flip_p(n_eff)
                p_clust = p_exact
                p_flip = p_exact if n_sign_flip > 0 else None
            else:
                t_stat = 0.0
                p_clust = 1.0
                p_flip = 1.0 if (c_vec is not None and n_sign_flip > 0) else None
        else:
            t_stat = float(delta / se)
            p_clust = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_t))
            ci_low = float(delta - t_crit * se)
            ci_high = float(delta + t_crit * se)

            # TOST for H0: |delta| >= equivalence_margin vs H1: |delta| < equivalence_margin
            t_lower = (delta + equivalence_margin) / se
            t_upper = (equivalence_margin - delta) / se
            p_lower = float(stats.t.sf(t_lower, df=df_t))
            p_upper = float(stats.t.sf(t_upper, df=df_t))
            p_tost = max(p_lower, p_upper)

            if c_vec is not None and n_sign_flip > 0:
                signs = rng.choice((-1.0, 1.0), size=(n_sign_flip, len(c_vec)))
                perm_means = (signs @ c_vec) / float(len(c_vec))
                p_flip = float(
                    (1.0 + np.sum(np.abs(perm_means) >= abs(delta) - 1e-15))
                    / (n_sign_flip + 1.0)
                )
            else:
                p_flip = None

        raw_stats.append(
            {
                "factor_id": fid,
                "mean_level_0": mean_0,
                "mean_level_1": mean_1,
                "effect_delta": delta,
                "oriented_effect": sign * delta,
                "se_item_clustered": se,
                "t_statistic": t_stat,
                "p_value_clustered": p_clust,
                "p_value_sign_flip": p_flip,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "tost_p_value": p_tost,
            }
        )

    # Compute targeted stats after F loop so RNG sequence for F sign-flips is bit-for-bit preserved
    targeted_stats_list: List[Optional[Dict[str, Any]]] = []
    if has_coverage and coverage_map is not None and eff_targeted_margin is not None:
        for fid in factor_cols_str:
            targeted_stats_list.append(
                _compute_targeted_stats(
                    df=df,
                    fid=fid,
                    target_col=target_col,
                    covered_ids=coverage_map.get(fid, set()),
                    sign=sign,
                    alpha=alpha,
                    targeted_margin=eff_targeted_margin,
                    n_sign_flip=n_sign_flip,
                    rng=rng,
                )
            )
    else:
        targeted_stats_list = [None for _ in factor_cols_str]

    q_vals = _benjamini_hochberg([r["p_value_clustered"] for r in raw_stats])

    # Separate BH family across only factors with a valid paired T-test (n_targeted >= 2)
    q_vals_t: List[Optional[float]] = [None for _ in raw_stats]
    valid_t_indices = [
        idx
        for idx, ts in enumerate(targeted_stats_list)
        if ts is not None and ts["p_value_targeted"] is not None
    ]
    if valid_t_indices:
        bh_t = _benjamini_hochberg(
            [float(targeted_stats_list[idx]["p_value_targeted"]) for idx in valid_t_indices]  # type: ignore[index]
        )
        for idx, q_t in zip(valid_t_indices, bh_t):
            q_vals_t[idx] = q_t

    n_items_min = int(min(n_items_per_factor)) if n_items_per_factor else 0
    small_n_warn = n_items_min < 100
    if small_n_warn:
        warnings_list.append(
            f"Small paired item count (N={n_items_min} < 100): asymptotic item-clustered t-tests can exhibit "
            "mild discrete-skewness inflation; verify p (Flip) randomization bounds or collect >=100 items."
        )

    rms_obs_sq = float(np.mean(obs_sq_list))
    rms_noise_sq = float(np.mean(noise_var_list))
    rms_sig_sq = max(0.0, rms_obs_sq - rms_noise_sq)
    noise_share = (
        float(min(100.0, (rms_noise_sq / rms_obs_sq) * 100.0))
        if rms_obs_sq > 1e-15
        else 0.0
    )

    noise_summary = NoiseFloorSummary(
        n_items=n_items_min,
        rms_observed=float(np.sqrt(rms_obs_sq)),
        rms_item_noise=float(np.sqrt(rms_noise_sq)),
        rms_signal_corrected=float(np.sqrt(rms_sig_sq)),
        noise_variance_share_pct=noise_share,
        small_sample_warning=small_n_warn,
    )

    factor_diags: List[CalibratedFactorDiagnostic] = []
    prunable_tokens = 0
    has_any_token_cost = token_costs is not None and len(token_costs) > 0

    for row, q_bh, t_stats, q_bh_t in zip(
        raw_stats, q_vals, targeted_stats_list, q_vals_t
    ):
        fid = row["factor_id"]
        oriented = row["oriented_effect"]
        t_cost = int(token_costs[fid]) if (token_costs and fid in token_costs) else None

        n_t: Optional[int] = t_stats["n_targeted"] if t_stats is not None else None
        oriented_t: Optional[float] = (
            t_stats["oriented_effect_targeted"] if t_stats is not None else None
        )
        p_ni_t: Optional[float] = (
            t_stats["p_noninferiority_targeted"] if t_stats is not None else None
        )

        f_sig = q_bh < alpha
        f_helps = f_sig and oriented > 0.0
        f_hurts = f_sig and oriented < 0.0

        t_sig = q_bh_t is not None and q_bh_t < alpha
        t_helps = t_sig and oriented_t is not None and oriented_t > 0.0
        t_hurts = t_sig and oriented_t is not None and oriented_t < 0.0

        # 11-row coverage-aware verdict truth table (reduces identically to 4 verdicts when coverage=None)
        if f_helps and t_hurts:
            verdict = "CONFLICT_REVIEW"
        elif f_helps:
            verdict = "KEEP"
        elif f_hurts and has_coverage and (n_t is None or n_t < min_targeted_items):
            verdict = "CONFLICT_REVIEW"
            warnings_list.append(
                f"Factor {fid!r} hurts primary metric but has insufficient targeted coverage "
                f"(n_targeted={n_t or 0} < {min_targeted_items}); routed to CONFLICT_REVIEW instead of REMOVE_HARMFUL."
            )
        elif f_hurts and t_helps:
            verdict = "CONFLICT_REVIEW"
        elif f_hurts:
            verdict = "REMOVE_HARMFUL"
        elif has_coverage and (n_t is None or n_t == 0):
            verdict = "UNTESTED"
        elif has_coverage and n_t < min_targeted_items:
            verdict = "INCONCLUSIVE_NEED_MORE_ITEMS"
        elif t_helps:
            verdict = "KEEP_TARGETED"
        elif t_hurts:
            verdict = "REMOVE_HARMFUL"
        elif row["tost_p_value"] < alpha and (
            not has_coverage or (p_ni_t is not None and p_ni_t < alpha)
        ):
            verdict = "SAFE_TO_PRUNE"
            if t_cost is not None:
                prunable_tokens += t_cost
        else:
            verdict = "INCONCLUSIVE_NEED_MORE_ITEMS"

        t_fields = t_stats if t_stats is not None else {}
        factor_diags.append(
            CalibratedFactorDiagnostic(
                **row,
                q_value_bh=q_bh,
                pruning_verdict=verdict,
                token_cost=t_cost,
                q_value_bh_targeted=q_bh_t,
                coverage_source=eff_coverage_source,
                **t_fields,
            )
        )

    # Rank by |t_statistic| descending so strongest drivers appear first
    factor_diags.sort(key=lambda d: (-abs(d.t_statistic), d.factor_id))

    return CalibratedAttributionAudit(
        target_metric=target_col,
        maximize=maximize,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        targeted_margin=eff_targeted_margin,
        min_targeted_items=eff_min_targeted,
        baseline_only_full_set=baseline_only_full_set,
        n_items=n_items_min,
        n_runs=n_runs,
        noise_floor=noise_summary,
        factors=factor_diags,
        estimated_prunable_tokens=prunable_tokens if has_any_token_cost else None,
        test_set_sha256=test_set_sha256,
        coverage_source=eff_coverage_source,
        evidence_tier=evidence_tier,
        judge_scored_fraction=judge_scored_fraction,
        generator_model_id=generator_model_id,
        model_id=model_id,
        warnings=warnings_list,
    )
