# Prompt Prism real-v3 prospective protocol

Frozen study identifier: `prompt-prism-real-v3-gemini-2026-09-26`

## Scope and claim

This experiment tests whether a registered 16-run Resolution IV prompt design selects a better prompt than equally sized, factor-balanced random screening designs on one real hosted LLM. The external-validity claim is limited to Gemini 2.5 Flash, the three registered tasks, these public datasets, and deterministic decoding. It is not a cross-model test.

## Model and inference controls

- Requested model: `gemini-2.5-flash`; every returned `modelVersion` and `responseId` is recorded.
- Temperature: 0; top-p: 1; seed: 42 for every configuration and item.
- Maximum output tokens: 512; thinking budget: 0.
- Factors A--F are the only prompt elements that vary. Dataset item, output limit, model settings, and label inventory are fixed within each task.
- Current paid-tier cost assumptions frozen for reporting: $0.30 per million input tokens and $2.50 per million output tokens, including thought tokens if present.

## Tasks and public data

1. `schema_extraction`: sentence passages and human entity spans from re3d. Primary metric: micro-style per-item set F1 over Person, Organisation, Location, Money, and Temporal mentions.
2. `grounded_qa`: SQuAD 2.0 development passages, including 15% unanswerable questions. Primary metric: SQuAD-style token F1.
3. `intent_classification`: a registered ten-label subset of the official Banking77 test split, balanced by label. Primary metric: exact intent accuracy.

Each task has 20 pilot items used only for prompt examples and validation, 80 discovery items, and 80 confirmation items. Documents/passages/utterances are disjoint across partitions. Pilot items never enter effect estimation or final scoring.

## Design and execution order

The six binary factors are A system role, B silent analysis instruction, C two task-specific demonstrations, D strict output contract, E input-only grounding rule, and F silent verification instruction. The full cube has 64 configurations. The registered 16-run fraction uses `E=ABC` and `F=BCD`, giving defining words `ABCE`, `BCDF`, and `ADEF` and Resolution IV for main effects.

All 64 configurations run on all 80 discovery items first (15,360 requests). Only the discovery manifest may be read to select recipes. A content lock then freezes the discovery manifest, recommendations, comparator designs, dataset files, analysis code, and protocol. Confirmation execution is prohibited until that lock verifies. All 64 configurations then run on all 80 confirmation items (15,360 requests), enabling the registered selection-aware bootstrap without reusing discovery observations.

## Comparators and primary hypothesis

Twenty 16-run random designs are generated before inference with seed 20260926. Each has eight observations per level of every factor, contains no duplicate configuration, and has a full-rank main-effects matrix. Each design selects its best observed configuration on discovery; their confirmation scores are averaged.

Prompt Prism estimates six main-effect contrasts on the registered fraction and selects the full-cube recipe matching each contrast's sign (zero maps to -1). The exhaustive discovery oracle selects the best mean score among all 64 configurations and is evaluated unchanged on confirmation.

For task `t`, define `delta_t = random_score_t - prism_score_t`, which is also Prism regret minus random regret relative to the same discovery-selected oracle. The primary statistic is the unweighted mean of `delta_t` across the three tasks. Negative values favor Prompt Prism. The one-sided alternative is `mean(delta_t) < 0` at alpha 0.05.

The primary p-value and 95% interval use 2,000 stratified bootstrap replicates. Within every task and replicate, discovery items and confirmation items are independently resampled with replacement. Prism, every random comparator, and the oracle are reselected from the resampled discovery data and evaluated on the resampled confirmation data. Support requires both a negative observed mean and one-sided bootstrap p < 0.05. There is no early stopping or task exclusion.

## Secondary analyses

- H2 compares main-effect recovery against full-cube discovery marginal effects using RMSE and sign agreement; random-design results are averaged across all 20 frozen designs.
- H3 compares each selected prompt with run 1 on held-out confirmation items using paired item bootstraps and Benjamini-Hochberg correction across tasks.
- H4 reports confirmation improvement over run 1 as a share of the discovery-oracle improvement; a zero or negative denominator is reported as not applicable.
- H5 reports actual input/output/thought tokens, latency, and provider-listed cost for the 16-run screens and exhaustive 64-run cube.
- Confirmation-best configuration is descriptive only and never substitutes for the discovery-selected oracle.

## Failure handling

Requests are resumable by a content-addressed key. HTTP 429 and 5xx responses receive bounded exponential backoff. Terminal failures and empty/MAX_TOKENS outputs score zero and remain in the denominator. No response is manually corrected. The manifest is append-only and records hashes, exact prompt/system text, raw response, metrics, request identifiers, returned model version, usage, latency, retries, finish reason, and errors.

This is a local prospective freeze with cryptographic hashes created before inference; it is not an externally timestamped registry entry.
