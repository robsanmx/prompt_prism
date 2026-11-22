# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`check_suite`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Targeted Coverage Gate:** One-sided non-inferiority (`oriented_delta_T < 0.0800`, `min_targeted_items=20`, source=`synthetic`)
- **Paired Benchmark Items (`N`):** `100` across `16` experimental runs (baseline items only; targeted items evaluated separately)
- **Main-Effect Noise Floor:** `RMS_obs=0.0079`, `RMS_noise=0.0063`, `RMS_signal_corr=0.0048` (item noise accounts for **63.0%** of observed variance)
- **Frozen Test Set SHA-256:** `58b52e1fc5dc8ce7667f2bad12fdedd7294c1705fbabca34c9ad582952450ae9`
- **LLM-Judge Scored Share:** `0.0%`

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Covered N | Δ targeted | NI p (T) | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **few_shot_demos** | 0.9725 | 0.9900 | +0.0175 | 0.0064 | [+0.0048, +0.0302] | 0.007503 | 0.04502 | 0.01796 | 0.3487 | 23 | +0.0598 | 0.2316 | `KEEP` | - |
| **specialist_persona** | 0.9838 | 0.9788 | -0.0050 | 0.0043 | [-0.0136, +0.0036] | 0.2502 | 0.6796 | 0.4052 | 0.0003863 | 4 | +0.0625 | 0.3988 | `INCONCLUSIVE_NEED_MORE_ITEMS` | - |
| **grounding_guardrail** | 0.9838 | 0.9788 | -0.0050 | 0.0083 | [-0.0215, +0.0115] | 0.5492 | 0.6796 | 0.6906 | 0.03719 | 35 | +0.7571 | 1 | `KEEP_TARGETED` | - |
| **strict_output_contract** | 0.9825 | 0.9800 | -0.0025 | 0.0043 | [-0.0111, +0.0061] | 0.5663 | 0.6796 | 0.7725 | 5.522e-05 | 39 | +0.1218 | 0.9518 | `KEEP_TARGETED` | - |
| **self_verification** | 0.9825 | 0.9800 | -0.0025 | 0.0043 | [-0.0111, +0.0061] | 0.5663 | 0.6796 | 0.7804 | 5.522e-05 | 0 | - | - | `UNTESTED` | - |
| **brief_reasoning** | 0.9825 | 0.9800 | -0.0025 | 0.0083 | [-0.0190, +0.0140] | 0.7647 | 0.7647 | 0.9621 | 0.01909 | 3 | +0.0000 | 0 | `INCONCLUSIVE_NEED_MORE_ITEMS` | - |

> **Evidence Tier:** `predicted (synthetic tests)` (attribution diagnostic on evaluated items only; validate any pruned or selected prompt on untouched confirmation items and issue a hashed `OperatingEnvelopeSpec` certificate before claiming perturbation stability).
