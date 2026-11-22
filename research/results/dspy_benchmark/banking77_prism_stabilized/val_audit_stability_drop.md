# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`drop`)

- **Optimization Direction:** `minimize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `60` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0032`, `RMS_noise=0.0047`, `RMS_signal_corr=0.0000` (item noise accounts for **100.0%** of observed variance)
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.
- **Warning:** Small paired item count (N=60 < 100): asymptotic item-clustered t-tests can exhibit mild discrete-skewness inflation; verify p (Flip) randomization bounds or collect >=100 items.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **demo_4** | -0.0083 | -0.0021 | +0.0063 | 0.0055 | [-0.0048, +0.0173] | 0.2603 | 0.7088 | 0.5017 | 0.007597 | `SAFE_TO_PRUNE` | - |
| **demo_2** | -0.0042 | -0.0063 | -0.0021 | 0.0036 | [-0.0093, +0.0052] | 0.5681 | 0.7088 | 1 | 3.42e-06 | `SAFE_TO_PRUNE` | - |
| **strict_contract** | -0.0042 | -0.0063 | -0.0021 | 0.0036 | [-0.0093, +0.0052] | 0.5681 | 0.7088 | 1 | 3.42e-06 | `SAFE_TO_PRUNE` | - |
| **xml_multiturn** | -0.0063 | -0.0042 | +0.0021 | 0.0036 | [-0.0052, +0.0093] | 0.5681 | 0.7088 | 1 | 3.42e-06 | `SAFE_TO_PRUNE` | - |
| **demo_1** | -0.0063 | -0.0042 | +0.0021 | 0.0056 | [-0.0090, +0.0132] | 0.7088 | 0.7088 | 1 | 0.001021 | `SAFE_TO_PRUNE` | - |
| **demo_3** | -0.0063 | -0.0042 | +0.0021 | 0.0056 | [-0.0090, +0.0132] | 0.7088 | 0.7088 | 1 | 0.001021 | `SAFE_TO_PRUNE` | - |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
