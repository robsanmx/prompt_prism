# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `120` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0052`, `RMS_noise=0.0047`, `RMS_signal_corr=0.0024` (item noise accounts for **78.8%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `792` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **demo_2** | 0.9542 | 0.9458 | -0.0083 | 0.0053 | [-0.0188, +0.0021] | 0.1171 | 0.4752 | 0.1739 | 0.01451 | `SAFE_TO_PRUNE` | 247 |
| **demo_1** | 0.9531 | 0.9469 | -0.0063 | 0.0049 | [-0.0159, +0.0034] | 0.2021 | 0.4752 | 0.3383 | 0.002799 | `SAFE_TO_PRUNE` | 251 |
| **demo_4** | 0.9479 | 0.9521 | +0.0042 | 0.0039 | [-0.0035, +0.0119] | 0.2869 | 0.4752 | 0.5032 | 4.332e-05 | `SAFE_TO_PRUNE` | 165 |
| **xml_multiturn** | 0.9479 | 0.9521 | +0.0042 | 0.0044 | [-0.0046, +0.0129] | 0.3479 | 0.4752 | 0.5092 | 0.0002484 | `SAFE_TO_PRUNE` | 0 |
| **demo_3** | 0.9479 | 0.9521 | +0.0042 | 0.0049 | [-0.0055, +0.0139] | 0.396 | 0.4752 | 0.5652 | 0.0007826 | `SAFE_TO_PRUNE` | 113 |
| **strict_contract** | 0.9510 | 0.9490 | -0.0021 | 0.0044 | [-0.0109, +0.0067] | 0.6393 | 0.6393 | 0.8231 | 4.739e-05 | `SAFE_TO_PRUNE` | 16 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
