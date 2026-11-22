# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `60` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0078`, `RMS_noise=0.0071`, `RMS_signal_corr=0.0033` (item noise accounts for **82.6%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `395` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.
- **Warning:** Small paired item count (N=60 < 100): asymptotic item-clustered t-tests can exhibit mild discrete-skewness inflation; verify p (Flip) randomization bounds or collect >=100 items.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Δ Drop | Stab. Verdict | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---:|
| **demo_1** | 0.9196 | 0.9347 | +0.0151 | 0.0114 | [-0.0076, +0.0378] | 0.1892 | 0.575 | 0.1994 | 0.3332 | +0.0074 | `NEUTRAL` | `INCONCLUSIVE_NEED_MORE_ITEMS` | 100 |
| **demo_4** | 0.9301 | 0.9242 | -0.0060 | 0.0048 | [-0.0156, +0.0037] | 0.2206 | 0.575 | 0.2389 | 0.002567 | -0.0132 | `NEUTRAL` | `SAFE_TO_PRUNE` | 146 |
| **strict_contract** | 0.9312 | 0.9231 | -0.0081 | 0.0075 | [-0.0231, +0.0070] | 0.2875 | 0.575 | 0.3473 | 0.05862 | -0.0092 | `NEUTRAL` | `INCONCLUSIVE_NEED_MORE_ITEMS` | 18 |
| **demo_2** | 0.9245 | 0.9299 | +0.0054 | 0.0063 | [-0.0073, +0.0181] | 0.397 | 0.5955 | 0.4173 | 0.01249 | -0.0031 | `NEUTRAL` | `SAFE_TO_PRUNE` | 120 |
| **demo_3** | 0.9283 | 0.9260 | -0.0024 | 0.0045 | [-0.0114, +0.0066] | 0.6006 | 0.7207 | 0.5972 | 0.000119 | +0.0025 | `NEUTRAL` | `SAFE_TO_PRUNE` | 128 |
| **xml_multiturn** | 0.9278 | 0.9265 | -0.0013 | 0.0057 | [-0.0126, +0.0101] | 0.8252 | 0.8252 | 0.8306 | 0.0007943 | -0.0076 | `NEUTRAL` | `SAFE_TO_PRUNE` | 1 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
