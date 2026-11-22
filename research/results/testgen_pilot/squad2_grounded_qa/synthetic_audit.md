# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`check_suite`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Targeted Coverage Gate:** One-sided non-inferiority (`oriented_delta_T < 0.0800`, `min_targeted_items=20`, source=`synthetic`)
- **Paired Benchmark Items (`N`):** `19` across `16` experimental runs (baseline items only; targeted items evaluated separately)
- **Main-Effect Noise Floor:** `RMS_obs=0.0757`, `RMS_noise=0.0301`, `RMS_signal_corr=0.0695` (item noise accounts for **15.8%** of observed variance)
- **Frozen Test Set SHA-256:** `00b6280736b1bbff2fc72b08ef8864d21e44416ef481782b6887c39c45c1533f`
- **LLM-Judge Scored Share:** `0.0%`
- **Warning:** Small paired item count (N=19 < 100): asymptotic item-clustered t-tests can exhibit mild discrete-skewness inflation; verify p (Flip) randomization bounds or collect >=100 items.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Covered N | Δ targeted | NI p (T) | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **brief_reasoning** | 0.7500 | 0.6316 | -0.1184 | 0.0388 | [-0.1999, -0.0369] | 0.006859 | 0.03797 | 0.01198 | 0.9897 | 41 | -0.0793 | 9.749e-06 | `REMOVE_HARMFUL` | - |
| **self_verification** | 0.6414 | 0.7401 | +0.0987 | 0.0365 | [+0.0220, +0.1754] | 0.01453 | 0.03797 | 0.02595 | 0.9776 | 0 | - | - | `KEEP` | - |
| **few_shot_demos** | 0.6480 | 0.7336 | +0.0855 | 0.0332 | [+0.0158, +0.1552] | 0.01899 | 0.03797 | 0.02994 | 0.9681 | 153 | +0.1348 | 0.9976 | `KEEP` | - |
| **strict_output_contract** | 0.6645 | 0.7171 | +0.0526 | 0.0259 | [-0.0017, +0.1070] | 0.05678 | 0.08517 | 0.1118 | 0.8885 | 153 | +0.1152 | 0.9787 | `KEEP_TARGETED` | - |
| **grounding_guardrail** | 0.6809 | 0.7007 | +0.0197 | 0.0197 | [-0.0217, +0.0612] | 0.3306 | 0.3967 | 0.5868 | 0.4948 | 129 | +0.0223 | 6.679e-13 | `KEEP_TARGETED` | - |
| **specialist_persona** | 0.6842 | 0.6974 | +0.0132 | 0.0211 | [-0.0313, +0.0576] | 0.5416 | 0.5416 | 0.7725 | 0.375 | 10 | +0.0500 | 0.153 | `INCONCLUSIVE_NEED_MORE_ITEMS` | - |

> **Evidence Tier:** `predicted (synthetic tests)` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
