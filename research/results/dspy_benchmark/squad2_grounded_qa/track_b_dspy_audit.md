# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `120` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0053`, `RMS_noise=0.0053`, `RMS_signal_corr=0.0001` (item noise accounts for **99.9%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `679` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **dspy_demo_2** | 0.8873 | 0.8965 | +0.0092 | 0.0068 | [-0.0043, +0.0226] | 0.1797 | 0.558 | 0.1939 | 0.05628 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 180 |
| **dspy_demo_4** | 0.8892 | 0.8945 | +0.0053 | 0.0053 | [-0.0052, +0.0158] | 0.3186 | 0.558 | 0.3473 | 0.003183 | `SAFE_TO_PRUNE` | 222 |
| **dspy_instruction** | 0.8937 | 0.8901 | -0.0035 | 0.0036 | [-0.0107, +0.0037] | 0.3348 | 0.558 | 0.3303 | 7.118e-06 | `SAFE_TO_PRUNE` | 77 |
| **dspy_demo_1** | 0.8901 | 0.8937 | +0.0036 | 0.0054 | [-0.0071, +0.0143] | 0.5064 | 0.633 | 0.5277 | 0.001481 | `SAFE_TO_PRUNE` | 197 |
| **dspy_demo_3** | 0.8926 | 0.8912 | -0.0015 | 0.0048 | [-0.0109, +0.0080] | 0.7595 | 0.7595 | 0.7596 | 8.785e-05 | `SAFE_TO_PRUNE` | 183 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
