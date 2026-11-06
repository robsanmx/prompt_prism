"""
Judge Validation and Human-Agreement Analysis (Phase 4 Task 15).
Quantifies inter-rater reliability (human-human) and judge-human agreement,
calibration, and systematic bias.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
RATING_MANIFEST_PATH = REPO_ROOT / "research" / "data" / "human_rating_manifest.csv"
RESULTS_DIR = REPO_ROOT / "research" / "results"


def analyze_judge_validation(
    manifest_path: Path = RATING_MANIFEST_PATH,
    output_csv: Path = RESULTS_DIR / "judge_validation.csv",
) -> pd.DataFrame:
    """
    Compute inter-rater reliability and judge-human agreement statistics.
    """
    df = pd.read_csv(manifest_path)

    results = []
    # Overall and per-task analysis
    task_groups = [("overall", df)] + list(df.groupby("task_id"))

    for group_name, group_df in task_groups:
        r1 = group_df["human_rater_1"].values
        r2 = group_df["human_rater_2"].values
        h_adj = group_df["human_adjudicated"].values
        j_score = group_df["llm_judge_score"].values

        # Normalize human score to [0, 1]
        h_norm = (h_adj - 1.0) / 4.0

        # Human-human agreement (Spearman rho and Pearson r)
        r_human_spearman, _ = stats.spearmanr(r1, r2)
        r_human_pearson, _ = stats.pearsonr(r1, r2)

        # Judge-human agreement
        r_judge_spearman, p_spearman = stats.spearmanr(h_norm, j_score)
        r_judge_pearson, p_pearson = stats.pearsonr(h_norm, j_score)

        # Bland-Altman systematic bias (judge - human)
        diffs = j_score - h_norm
        mean_bias = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        loa_lower = mean_bias - 1.96 * sd_diff
        loa_upper = mean_bias + 1.96 * sd_diff

        results.append({
            "stratum": group_name,
            "sample_size": len(group_df),
            "human_human_spearman": float(r_human_spearman),
            "human_human_pearson": float(r_human_pearson),
            "judge_human_spearman": float(r_judge_spearman),
            "judge_human_pearson": float(r_judge_pearson),
            "systematic_bias": mean_bias,
            "limits_of_agreement_lower": loa_lower,
            "limits_of_agreement_upper": loa_upper,
        })

    out_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Judge validation results saved to {output_csv}")
    print(out_df.to_string())

    return out_df


if __name__ == "__main__":
    analyze_judge_validation()
