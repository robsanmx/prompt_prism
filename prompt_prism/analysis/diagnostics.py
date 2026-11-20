"""
PromptPrism 2.0 Calibrated Attribution, Item-Sampling Noise-Floor, and Token-Pruning Diagnostics.

Grounded in the 1,033-cube retrospective empirical-null calibration audit and 501,300-call
prospective Real-v4 study:
- Uses contrast-specific item-clustered t-tests (`df = N - 1`, zero-variance contrasts -> p = 1.0)
  rather than pooled fixed-block RCBD ANOVA (which inflates global BH FDR to 18.4% under
  heterogeneous contrast variance).
- Adds optional Rademacher sign-flip randomization checks (recommended whenever N < 100).
- Decomposes observed main-effect RMS into expected item-sampling noise vs. noise-corrected signal.
- Uses paired Two-One-Sided-Tests (TOST) against an explicit practical equivalence margin so
  factors are marked `SAFE_TO_PRUNE` only when statistically negligible, separating true bloat
  from underpowered `INCONCLUSIVE_NEED_MORE_ITEMS` factors.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


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
    pruning_verdict: (
        str  # KEEP | REMOVE_HARMFUL | SAFE_TO_PRUNE | INCONCLUSIVE_NEED_MORE_ITEMS
    )
    token_cost: Optional[int] = None


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

    schema_version: str = "2.0.0"
    target_metric: str
    maximize: bool = True
    alpha: float = 0.05
    equivalence_margin: float = 0.01
    inference_method: str = "item_clustered_contrast_t_test_with_bh_and_tost"
    n_items: int
    n_runs: int
    noise_floor: NoiseFloorSummary
    factors: List[CalibratedFactorDiagnostic] = Field(default_factory=list)
    estimated_prunable_tokens: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        direction = "maximize" if self.maximize else "minimize"
        lines = [
            f"# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`{self.target_metric}`)",
            "",
            f"- **Optimization Direction:** `{direction}`",
            f"- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha={self.alpha}`)",
            f"- **Pruning Gate:** TOST practical equivalence (`|delta| < {self.equivalence_margin:.4f}`)",
            f"- **Paired Benchmark Items (`N`):** `{self.n_items}` across `{self.n_runs}` experimental runs",
            f"- **Main-Effect Noise Floor:** `RMS_obs={self.noise_floor.rms_observed:.4f}`, "
            f"`RMS_noise={self.noise_floor.rms_item_noise:.4f}`, "
            f"`RMS_signal_corr={self.noise_floor.rms_signal_corrected:.4f}` "
            f"(item noise accounts for **{self.noise_floor.noise_variance_share_pct:.1f}%** of observed variance)",
        ]
        if self.estimated_prunable_tokens is not None:
            lines.append(
                f"- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `{self.estimated_prunable_tokens}` tokens/prompt"
            )
        for w in self.warnings:
            lines.append(f"- **Warning:** {w}")

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
                "> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; "
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
) -> CalibratedAttributionAudit:
    """Run PromptPrism 2.0 calibrated item-clustered attribution, noise-floor, and TOST pruning audit."""
    if not factor_cols:
        raise ValueError("factor_cols cannot be empty")
    if target_col not in data.columns:
        raise ValueError(f"target_col {target_col!r} not found in dataframe")
    if not (0.0 < alpha < 0.5):
        raise ValueError(f"alpha must lie in (0, 0.5), got {alpha}")
    if equivalence_margin <= 0.0:
        raise ValueError(f"equivalence_margin must be > 0, got {equivalence_margin}")

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

    for col in factor_cols:
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

    n_runs = (
        int(df["run_id"].nunique())
        if "run_id" in df.columns
        else int(df[list(factor_cols)].drop_duplicates().shape[0])
    )

    sign = 1.0 if maximize else -1.0
    rng = np.random.default_rng(seed)

    raw_stats = []
    n_items_per_factor: List[int] = []
    obs_sq_list: List[float] = []
    noise_var_list: List[float] = []

    for fid in factor_cols:
        # Compute paired per-item mean at level 1 minus level 0
        piv = df.pivot_table(
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
                g0 = df.loc[df[fid] == 0, target_col].dropna().to_numpy(dtype=float)
                g1 = df.loc[df[fid] == 1, target_col].dropna().to_numpy(dtype=float)
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

        # Conservative zero-variance guard (matches our empirical calibration suite: zero-var -> p = 1.0)
        if se <= 1e-12:
            t_stat = 0.0
            p_clust = 1.0
            ci_low = delta
            ci_high = delta
            p_tost = 0.0 if abs(delta) < equivalence_margin else 1.0
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
                "factor_id": str(fid),
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

    q_vals = _benjamini_hochberg([r["p_value_clustered"] for r in raw_stats])
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

    for row, q_bh in zip(raw_stats, q_vals):
        fid = row["factor_id"]
        oriented = row["oriented_effect"]
        t_cost = int(token_costs[fid]) if (token_costs and fid in token_costs) else None

        if q_bh < alpha and oriented > 0.0:
            verdict = "KEEP"
        elif q_bh < alpha and oriented < 0.0:
            verdict = "REMOVE_HARMFUL"
        elif row["tost_p_value"] < alpha:
            verdict = "SAFE_TO_PRUNE"
            if t_cost is not None:
                prunable_tokens += t_cost
        else:
            verdict = "INCONCLUSIVE_NEED_MORE_ITEMS"

        factor_diags.append(
            CalibratedFactorDiagnostic(
                **row,
                q_value_bh=q_bh,
                pruning_verdict=verdict,
                token_cost=t_cost,
            )
        )

    # Rank by |t_statistic| descending so strongest drivers appear first
    factor_diags.sort(key=lambda d: (-abs(d.t_statistic), d.factor_id))

    return CalibratedAttributionAudit(
        target_metric=target_col,
        maximize=maximize,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        n_items=n_items_min,
        n_runs=n_runs,
        noise_floor=noise_summary,
        factors=factor_diags,
        estimated_prunable_tokens=prunable_tokens if has_any_token_cost else None,
        warnings=warnings_list,
    )
