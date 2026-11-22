# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `120` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0107`, `RMS_noise=0.0069`, `RMS_signal_corr=0.0082` (item noise accounts for **42.0%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `395` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **demo_1** | 0.8676 | 0.8907 | +0.0231 | 0.0109 | [+0.0016, +0.0446] | 0.03533 | 0.1143 | 0.01449 | 0.6128 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 100 |
| **strict_contract** | 0.8848 | 0.8735 | -0.0114 | 0.0054 | [-0.0222, -0.0006] | 0.0381 | 0.1143 | 0.04348 | 0.05795 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 18 |
| **demo_4** | 0.8804 | 0.8779 | -0.0025 | 0.0045 | [-0.0115, +0.0064] | 0.5759 | 0.8189 | 0.5697 | 9.597e-05 | `SAFE_TO_PRUNE` | 146 |
| **demo_2** | 0.8774 | 0.8808 | +0.0034 | 0.0076 | [-0.0117, +0.0185] | 0.6558 | 0.8189 | 0.7451 | 0.0155 | `SAFE_TO_PRUNE` | 120 |
| **xml_multiturn** | 0.8780 | 0.8803 | +0.0022 | 0.0054 | [-0.0085, +0.0130] | 0.6824 | 0.8189 | 0.6982 | 0.0007175 | `SAFE_TO_PRUNE` | 1 |
| **demo_3** | 0.8795 | 0.8788 | -0.0008 | 0.0058 | [-0.0122, +0.0106] | 0.8934 | 0.8934 | 0.8926 | 0.0005659 | `SAFE_TO_PRUNE` | 128 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
