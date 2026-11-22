# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `120` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0103`, `RMS_noise=0.0053`, `RMS_signal_corr=0.0088` (item noise accounts for **26.8%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `463` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **dspy_demo_3** | 0.9646 | 0.9802 | +0.0156 | 0.0058 | [+0.0041, +0.0272] | 0.008467 | 0.04233 | 0.004498 | 0.2275 | `KEEP` | 180 |
| **dspy_demo_1** | 0.9646 | 0.9802 | +0.0156 | 0.0078 | [+0.0002, +0.0310] | 0.04649 | 0.1162 | 0.06597 | 0.2871 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 177 |
| **dspy_instruction** | 0.9740 | 0.9708 | -0.0031 | 0.0023 | [-0.0077, +0.0015] | 0.1808 | 0.3013 | 0.3838 | 2.066e-11 | `SAFE_TO_PRUNE` | 98 |
| **dspy_demo_2** | 0.9698 | 0.9750 | +0.0052 | 0.0052 | [-0.0051, +0.0155] | 0.3193 | 0.3992 | 0.4433 | 0.002654 | `SAFE_TO_PRUNE` | 184 |
| **dspy_demo_4** | 0.9729 | 0.9719 | -0.0010 | 0.0038 | [-0.0085, +0.0064] | 0.7828 | 0.7828 | 1 | 8.84e-07 | `SAFE_TO_PRUNE` | 181 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
