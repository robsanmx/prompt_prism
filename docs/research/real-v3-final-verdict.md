# Real-v3 final verdict

**Experiment:** `prompt-prism-real-v3-gemini-2026-09-26`  
**Completed:** 2026-09-26  
**Requested and returned model:** Gemini 2.5 Flash  
**Protocol hash:** `46933473419012fe20b3a0229b46851bfcbaf292ab2e4f73d3c1f824646be416`

## Verdict

PromptPrism did **not** outperform equal-budget, factor-balanced random screening
for held-out prompt selection in this experiment.

The registered primary statistic was the mean across three tasks of random-screen
confirmation score minus PromptPrism confirmation score. Negative values favor
PromptPrism. The result was:

- observed difference: **+0.00110**;
- 95% selection-aware item-bootstrap interval: **[-0.02734, +0.02869]**;
- one-sided p-value: **0.4568**;
- decision at alpha 0.05: **H1 unsupported**.

This is a failure to show superiority, not evidence that the two methods are
equivalent. The interval includes meaningful advantages in either direction.

## What the experiment does support

The 16-run Resolution IV fraction estimated full-cube discovery main effects more
accurately than the twenty balanced random designs on every task. Mean effect RMSE
was **0.00398** for PromptPrism and **0.00931** for random screening; mean sign
agreement was **88.9%** and **74.2%**, respectively.

Relative to exhaustive discovery screening, PromptPrism used:

- 16 rather than 64 configurations;
- 3,840 rather than 15,360 calls;
- 1,138,511 rather than 4,550,470 total tokens;
- $0.5144 rather than $2.0487 in provider-listed cost.

This establishes the expected roughly 75% resource reduction versus exhaustive
screening. It does not provide a resource advantage over balanced random screening,
which used the same 16-configuration budget.

## Per-task held-out results

| Task | PromptPrism | Balanced random | Discovery-selected oracle | Random minus PromptPrism |
|---|---:|---:|---:|---:|
| Entity extraction | 0.51110 | 0.51546 | 0.53268 | +0.00436 |
| Grounded QA | 0.87717 | 0.87548 | 0.88009 | -0.00168 |
| Intent classification | 0.86250 | 0.86312 | 0.85000 | +0.00062 |

## Integrity result

- 30,720 total real API calls: 15,360 discovery and 15,360 confirmation.
- 30,720 unique cells and unique provider response IDs.
- Zero terminal provider errors.
- 30,718 `STOP` and two `MAX_TOKENS` completions; the latter remained in the
  denominator with score zero under the frozen rule.
- One returned model version throughout: `gemini-2.5-flash`.
- 8,551,232 input tokens, 634,703 output tokens, zero thought tokens.
- Total provider-listed cost: **$4.1521**.

## Scientific scope

The conclusion is limited to Gemini 2.5 Flash, deterministic decoding, these six
prompt factors, and the three frozen public-data tasks. The prospective freeze is
a local cryptographic content lock, not an externally timestamped registration.
The earlier `research/results/real_gemini/` run remains an exploratory artifact and
is excluded from the real-v3 paper.

## Final artifacts

- Protocol: `research/protocol/real_v3/preregistration.md`
- Statistical plan: `research/protocol/real_v3/statistical_analysis_plan.md`
- Frozen designs: `research/protocol/real_v3/design_registry.json`
- Dataset manifest: `research/data/real_v3/dataset_manifest.json`
- Discovery provenance: `research/results/real_gemini_v3/discovery_manifest.jsonl`
- Confirmation provenance: `research/results/real_gemini_v3/confirmation_manifest.jsonl`
- Frozen recommendations: `research/results/real_gemini_v3/recommendations.json`
- Machine-readable summary: `research/results/real_gemini_v3/summary.json`
- Per-task results: `research/results/real_gemini_v3/results.csv`
- Paper: `paper/main.pdf`

Final provenance hashes:

```text
discovery       2510ed02d94a9e4254fe6911c8f939b4a82248903e84cb57b687ca0617190178
confirmation    88ab4d6d4f1f1f7efb1a3f9d198322df65579085b19090ec3d4ede80b34e7adc
recommendations cbd5c83c5d95dbfa2b4d1edbff3067450f285e127eb70f1870c8c7aca9be52c6
```
