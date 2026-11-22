# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `60` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0000`, `RMS_noise=0.0000`, `RMS_signal_corr=0.0000` (item noise accounts for **0.0%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `792` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.
- **Warning:** Small paired item count (N=60 < 100): asymptotic item-clustered t-tests can exhibit mild discrete-skewness inflation; verify p (Flip) randomization bounds or collect >=100 items.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Δ Drop | Stab. Verdict | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---:|
| **demo_1** | 1.0000 | 1.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 1 | 1 | 1 | 0 | +0.0021 | `NEUTRAL` | `SAFE_TO_PRUNE` | 251 |
| **demo_2** | 1.0000 | 1.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 1 | 1 | 1 | 0 | -0.0021 | `NEUTRAL` | `SAFE_TO_PRUNE` | 247 |
| **demo_3** | 1.0000 | 1.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 1 | 1 | 1 | 0 | +0.0021 | `NEUTRAL` | `SAFE_TO_PRUNE` | 113 |
| **demo_4** | 1.0000 | 1.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 1 | 1 | 1 | 0 | -0.0021 | `NEUTRAL` | `SAFE_TO_PRUNE` | 165 |
| **strict_contract** | 1.0000 | 1.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 1 | 1 | 1 | 0 | -0.0021 | `NEUTRAL` | `SAFE_TO_PRUNE` | 16 |
| **xml_multiturn** | 1.0000 | 1.0000 | +0.0000 | 0.0000 | [+0.0000, +0.0000] | 1 | 1 | 1 | 0 | +0.0021 | `NEUTRAL` | `SAFE_TO_PRUNE` | 0 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
