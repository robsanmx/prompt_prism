"""Generate manuscript figures directly from the locked real-v3 summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "research" / "results" / "real_gemini_v3" / "summary.json"
OUT = Path(__file__).resolve().parent


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "figure.dpi": 180,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    style()
    tasks = ["schema_extraction", "grounded_qa", "intent_classification"]
    labels = ["Entity\nextraction", "Grounded\nQA", "Intent\nclassification"]

    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    x = np.arange(len(tasks))
    width = 0.19
    series = [
        ("Baseline", "baseline_score", "#9ca3af"),
        ("PromptPrism", "prism_score", "#2563eb"),
        ("Balanced random", "random_score", "#f59e0b"),
        ("Discovery oracle", "oracle_score", "#16a34a"),
    ]
    for index, (label, key, color) in enumerate(series):
        values = [data["tasks"][task][key] for task in tasks]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Confirmation score")
    ax.set_ylim(0.40, 0.94)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "fig_real_v3_scores")

    fig, ax = plt.subplots(figsize=(3.8, 3.25))
    prism = [data["h2_effect_recovery"][task]["prism_rmse"] for task in tasks]
    random = [data["h2_effect_recovery"][task]["random_mean_rmse"] for task in tasks]
    ax.bar(x - width / 2, prism, width, label="PromptPrism", color="#2563eb")
    ax.bar(x + width / 2, random, width, label="Balanced random", color="#f59e0b")
    ax.set_xticks(x, ["Entity", "QA", "Intent"])
    ax.set_ylabel("Main-effect RMSE")
    ax.set_ylim(0, 0.0135)
    ax.legend(ncol=1, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, "fig_real_v3_effect_rmse")


if __name__ == "__main__":
    main()
