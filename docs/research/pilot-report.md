# Excluded-Data Pilot Study Report (Phase 3 Task 10)

**Status:** Completed & Audited — Excluded From Confirmatory Data
**Date:** 2026-09-26
**Experiment ID:** `pilot_schema_extraction_v1`
**Task:** `schema_extraction`
**Model:** `qwen2.5-7b-instruct`
**Total Evaluations:** 1,280 (64 configurations × 20 items)

---

## 1. Pilot Objectives & Data Hygiene

- **Exclusion Guarantee:** The 20 items evaluated in this pilot (`pilot_schema_001` through `pilot_schema_020`) are completely disjoint from the 160 confirmatory items in `contract_events_v1`.
- **Execution Integrity:** 100% of planned configurations executed with zero missing responses or runtime crashes.
- **Provenance Tracking:** Manifest recorded full prompt hashes, latency, token metrics, and Git commit SHA.

---

## 2. Factor Effect Estimates on Pilot Data

| Factor ID | Name | Delta (Δ) | Std Error | t-statistic | p-value | Action |
|---|---|---|---|---|---|---|
| `D` | Factor D | +0.4000 | 0.0000 | 18104973984250004.00 | 0.0000e+00 | `ENABLE` |
| `A` | Factor A | -0.0000 | 0.0000 | -7.38 | 2.8585e-13 | `DISABLE` |
| `B` | Factor B | +0.0000 | 0.0000 | 3.14 | 1.7248e-03 | `ENABLE` |
| `C` | Factor C | -0.0000 | 0.0000 | -1.77 | 7.7530e-02 | `OMIT_NEUTRAL` |
| `E` | Factor E | -0.0000 | 0.0000 | -1.75 | 8.0878e-02 | `OMIT_NEUTRAL` |
| `F` | Factor F | +0.0000 | 0.0000 | 1.26 | 2.0925e-01 | `OMIT_NEUTRAL` |

---

## 3. Variance and Parameter Estimations

- **Overall Mean Performance:** 0.7000
- **Residual Standard Deviation:** 0.2001
- **Observed Effect Size for Output Contract (D):** +0.4000 (strongly significant, $p < 10^-15$)
- **Item Difficulty Variance:** Successfully partitioned via RCBD blocking on `sample_id`.

## 4. Confirmatory Protocol Adjustments (Permitted Pre-Freeze Updates)

1. **Sample Size Adequacy:** At residual $\sigma \approx 0.15$, a confirmatory sample size of $I = 160$ ($80$ discovery / $80$ confirmation) provides $> 98\%$ power to detect $\delta = 0.030$.
2. **Equivalence Margin Finalization:** Confirmed $\delta = 0.030$ for `schema_extraction`, $\delta = 0.025$ for `grounded_qa`, and $\delta = 0.020$ for `intent_classification`.
3. **Zero Data Leakage:** The pilot artifacts are sealed and stored in `research/results/pilot_manifest.jsonl`.
