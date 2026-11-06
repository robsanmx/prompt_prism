# Audit of the Real-Gemini Result Artifacts

**Audit date:** 2026-09-26  
**Artifacts reviewed:**

- `research/results/real_gemini/gemini_manifest.jsonl`
- `research/results/real_gemini/recommendations.json`
- `research/results/real_gemini/gemini_summary.json`
- `research/results/real_gemini/gemini_results.csv`

## Verdict

These artifacts are usable as an **exploratory API and analysis smoke test**. They
are not valid confirmatory evidence for the registered real-LLM paper.

The raw manifest is internally coherent and consistent with a live Gemini run, but
the study uses generated template datasets, a nonregistered model, one quarter of
the registered item count, one of three registered models, an unregistered uniform
random comparator, and an invalid recommendation-lock sequence. The summary also
uses the wrong oracle for reported regret and does not execute the registered
aggregate hypothesis test.

## Checks that passed

- All 7,680 JSONL records parse.
- Trial IDs, cache keys, and recorded request IDs are unique.
- All prompt and response SHA-256 hashes match their stored text.
- Every task/split cell contains 64 configurations and 20 items per configuration.
- No row has a recorded provider error or retry.
- Re-running the frozen task evaluators reproduces every stored metric exactly.
- Recommendations' fractional-design effect estimates reproduce from the discovery
  rows.
- CSV task scores and JSON totals reproduce from the manifest.
- Token totals are 1,016,128 input and 208,839 output tokens.
- The reported $0.1388613 cost equals the manifest sum under the adapter's hard-coded
  rates. It is not evidence of the provider's billed amount.

The four artifact hashes at audit time are:

```text
manifest         4875923005b6457c410c9556c603bc535d03de0ab0722a7eb336319f97e8369b
recommendations  50d8b4b6624594abf13a3dab8804ed4b675e74f30ad0528547b37717fe2ce022
summary          d66ea9ef9566e5d5df10a282f01938d926dc756c3a809605771ee636fb0fb38f
CSV              961cd54f1da4c7816560ca68e30061648652a7abfe80365b164d14bc0c9ad8af
```

## Critical scientific findings

### 1. The datasets are generated templates, not the attributed corpora

`research/data/real_v2/acquire.py` programmatically fabricates company names,
acquisition sentences, `Alpha-XX` research dossiers, and customer inquiries. It
then assigns source strings naming SEC EDGAR, SQuAD/Natural Questions, and
Banking77. It does not download or extract observations from those sources.

The dataset manifest therefore overstates provenance. These are synthetic task
items submitted to a real model. They cannot support claims about performance on
SEC filings, SQuAD, Natural Questions, Banking77, or naturally occurring inputs.

The artifact registry does not hash the three JSONL datasets. The dataset audit
hashes only sorted item IDs, so text and targets can change without invalidating a
split hash.

### 2. The run is outside the registered confirmatory design

The preregistration specifies 3 models, 80 discovery and 80 confirmation items per
task, and 122,880 calls. The artifacts contain:

- one unregistered model: `gemini-2.5-flash`;
- 20 discovery and 20 confirmation items per task;
- one generation per configuration/item;
- 7,680 calls total;
- three task–model cells rather than nine.

Gemini may be a useful exploratory extension, but it cannot be substituted for the
registered Llama, OpenAI, and Anthropic matrix after registration.

### 3. The recommendation was not locked before confirmation collection

The runner queues and generates discovery and confirmation outputs together. Only
after the entire response manifest is complete does it calculate and write
`recommendations.json`. Filesystem times show the last manifest write at
22:40:56Z and the recommendations file at 22:41:25Z.

The confirmation responses therefore existed before the claimed pre-lock. The
recommendation hash is printed but not stored with a timestamp, manifest hash,
analysis hash, or external registration. The actual runner is absent from the
artifact hash registry.

### 4. The frozen code and source revision do not authenticate this run

Every manifest row records Git commit `d3020cd...`, but that commit contains none
of the `research/` code or data. The research tree is untracked, and the claimed
`v2.0-confirmatory-freeze` tag is absent. One registered analysis-file hash already
mismatches the current file.

The local preregistration timestamp is not an external timestamp. It does not prove
that protocol and analysis choices preceded the data.

### 5. Reported regret uses the wrong oracle

The registered estimand evaluates the configuration selected by exhaustive search
on discovery items against confirmation items. The Gemini summary instead uses
the maximum confirmation score directly. This looks at the held-out outcomes and
inflates every individual regret.

Recomputed values are:

| Task | Prism score | Random score | Baseline | Discovery-selected oracle on confirmation | Reported oracle | Correct Prism regret | Prism minus random regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Schema extraction | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Grounded QA | 0.0722 | 0.1231 | 0.1038 | 0.2800 | 0.4100 | 0.2078 | +0.0509 |
| Intent classification | 0.0500 | 0.1000 | 0.0000 | 0.0000 | 0.1000 | -0.0500 | +0.0500 |

Negative realized regret is possible here because the registered comparator is the
discovery-selected oracle, which may generalize worse than another selected prompt.

The oracle cancels from the Prism-minus-random contrast, so the mean contrast
remains `+0.033618`. Positive values favor random search. Directionally, these
three exploratory cells do not support H1.

### 6. The primary registered inference was not run

The summary declares H1 from `mean_delta_regret < 0` alone. It contains no
three-cell aggregate bootstrap p-value or confidence interval, and the protocol
requires nine task–model cells plus both the registered direction and `p < 0.05`.

The per-task bootstrap also has two defects:

1. The code uses Python's process-randomized `hash(task_id)` to form comparator and
   bootstrap seeds. The candidate configurations change with `PYTHONHASHSEED`.
2. The random candidate list is not stored per task. During the later bootstrap
   loop, the variable contains only the list from the last task, so schema and QA
   confidence intervals and p-values are computed with the intent task's random
   candidate set rather than their own selection sets.

The comparator is an unconstrained uniform sample, while the registered primary
comparator is factor-balanced random design. The full 16-run candidate sets are not
written to `recommendations.json`, so the reported random-search selections cannot
be independently reconstructed.

### 7. Task and treatment validity is poor

- Factor C is registered as two vetted task-specific demonstrations. The runner
  inserts one generic sentence that contains neither an input nor an output.
- Factor D applies the same JSON instruction to extraction, QA, and classification
  without defining task-specific keys.
- The intent prompt never lists the seven permitted intent labels. Its mean valid
  label rate is only 1.48%, so the experiment mostly measures whether the model
  guesses the project's private label names.
- The QA evaluator scores the entire often-verbose response against a short answer,
  mixing factual correctness with response length and format.
- The schema items all share one simple generated sentence template. The selected,
  random, baseline, and oracle prompts all score 1.0 on confirmation, producing a
  ceiling cell with no optimization information.
- The config calls extraction's per-item average over four fields “micro F1,” and
  calls the classification endpoint “macro F1,” but the implemented endpoints are
  mean field F1 and item accuracy, respectively.

### 8. Model provenance is incomplete

All records are structurally consistent with calls through the Gemini adapter, but
the adapter records the requested model alias rather than a returned immutable
model revision. `system_fingerprint` is absent. Gemini request IDs in the manifest
are locally generated fallbacks, and raw HTTP response metadata is not archived.
The manifest therefore cannot independently authenticate the backend snapshot.

The seed changes with `run_id`, so each prompt configuration receives a different
seed. If the endpoint honors seeds even at temperature zero, prompt effects are
confounded with generation seed. Seeds should be held constant across configurations
within a replicate.

### 9. Truncation is material and omitted from the summary

There are 607 `MAX_TOKENS` completions (7.9% of all records): 165 extraction, 246
QA, and 196 intent. They are not recorded as errors or summarized as terminal
generation failures. For extraction, their mean score is 0.036 versus 0.990 for
`STOP`, so this omission materially hides an outcome mechanism.

## Hypothesis disposition for this exploratory run

- **H1:** Unsupported directionally. PromptPrism's mean regret contrast is 3.36
  percentage points worse than the recorded random-search winners. Confirmatory
  inference is invalid for the reasons above.
- **H2:** Not tested as registered. Fractional-vs-oracle RMSE is reproducible, but
  the balanced-random comparator is absent. QA has no oracle main effect above its
  SESOI.
- **H3:** Unsupported as a general claim. PromptPrism ties baseline on extraction,
  underperforms it on QA, and improves on intent.
- **H4:** Unsupported. Extraction has a zero uplift denominator; QA has negative
  uplift relative to baseline; intent's discovery-oracle denominator is zero.
- **H5:** Only the arithmetic 16-vs-64 configuration reduction is established.
  The run evaluated all 64 configurations and does not report measured cost,
  latency, or failure savings for the 16-run subset.

## Required disposition

Do not cite these artifacts as confirmatory paper results. Preserve them under an
explicit exploratory/pilot label because they are useful for diagnosing provider,
prompt, evaluator, and analysis failures.

Before the next empirical run:

1. acquire real, traceable dataset records and hash their complete content;
2. register Gemini explicitly or execute the already registered model matrix;
3. use the registered 80/80 items and power-supported replication;
4. generate and lock discovery recommendations before confirmation is generated or
   made accessible;
5. store stable integer seeds and all random comparator candidate sets;
6. use the discovery-selected oracle in reported regret;
7. run the aggregate item-cluster bootstrap and require both direction and p-value;
8. render task-specific demonstrations and output contracts;
9. hold decoding seed constant across prompt configurations within each replicate;
10. report truncations and other finish reasons by task and configuration.
