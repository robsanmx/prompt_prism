# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `120` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0140`, `RMS_noise=0.0080`, `RMS_signal_corr=0.0115` (item noise accounts for **33.0%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `126` tokens/prompt
- **Warning:** No coverage map provided: SAFE_TO_PRUNE does not imply the clause was exercised.

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **demo_4** | 0.9573 | 0.9802 | +0.0229 | 0.0106 | [+0.0020, +0.0438] | 0.03203 | 0.1075 | 0.02399 | 0.6085 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 100 |
| **demo_1** | 0.9594 | 0.9781 | +0.0187 | 0.0088 | [+0.0013, +0.0362] | 0.03584 | 0.1075 | 0.04548 | 0.4438 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 100 |
| **xml_multiturn** | 0.9625 | 0.9750 | +0.0125 | 0.0073 | [-0.0020, +0.0270] | 0.08976 | 0.1795 | 0.1164 | 0.1534 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 297 |
| **strict_contract** | 0.9729 | 0.9646 | -0.0083 | 0.0061 | [-0.0203, +0.0036] | 0.1711 | 0.2566 | 0.2314 | 0.02812 | `SAFE_TO_PRUNE` | 18 |
| **demo_2** | 0.9646 | 0.9729 | +0.0083 | 0.0079 | [-0.0074, +0.0240] | 0.2954 | 0.3545 | 0.3863 | 0.07193 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 110 |
| **demo_3** | 0.9698 | 0.9677 | -0.0021 | 0.0068 | [-0.0155, +0.0113] | 0.759 | 0.759 | 0.8696 | 0.004649 | `SAFE_TO_PRUNE` | 108 |

> **Evidence Tier:** `predicted` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
