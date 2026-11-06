# Real-LLM Experiment Plan for the PromptPrism Paper

**Version:** 1.0  
**Date:** 2026-09-26  
**Status:** Replacement protocol draft; no real confirmatory LLM data have been collected  
**Paper type:** Empirical LLM methods paper with simulation used only for calibration

## 1. Scientific reset

The current paper cannot supply evidence for an empirical LLM claim. Its Study B
and Study C outputs were produced by deterministic fixtures and explicit formulas.
Those artifacts may remain as software fixtures, but they cannot be described as
model generations, pilot observations, confirmatory observations, human-rated
outputs, or evidence of external validity.

The replacement study will use a new experiment identifier, new result paths, new
hashes, and an externally timestamped preregistration. It will never overwrite or
silently mix with the existing `confirmatory_study_v1` artifacts.

The paper's central question will be:

> Across preregistered task–model cells, does a 16-configuration Resolution IV
> fractional factorial design estimate full-factorial marginal prompt effects and
> select a prompt with lower held-out regret than a budget-matched random design?

The paper will report four claims separately:

1. **Effect estimation:** agreement with marginal effects from the exhaustive cube.
2. **Prompt selection:** held-out regret of the prompt selected on discovery items.
3. **Generalization:** performance relative to the baseline on untouched items.
4. **Efficiency:** configurations, calls, tokens, elapsed time, and monetary cost.

No result in one category will be used as proof of another.

## 2. Design overview

### 2.1 Registered cells

The preferred confirmatory matrix has three objective tasks and three frozen model
snapshots from distinct deployment settings:

| Dimension | Registered levels |
|---|---|
| Tasks | schema-bound extraction; grounded question answering; intent classification |
| Models | one reproducible open-weight model; one economical API model; one model from a different provider family |
| Prompt factors | six binary semantic components, A–F |
| Configurations | all 64 combinations per task–model cell |
| Items | 160 eligible items per task, split 80 discovery / 80 confirmation |
| Primary design | regular `2^(6-2)` Resolution IV fraction, 16 configurations |
| Primary comparator | 16-configuration factor-balanced random design |
| Secondary comparators | uniform random search; random-design linear model; baseline; exhaustive discovery oracle |

The exact model identifiers remain provisional until each provider passes a real
endpoint smoke test. They must then be frozen by immutable identifier or archived
weights. A provider alias such as `latest` is not acceptable.

### 2.2 Generation volume

With the current proposed replication rule—two generations for the open-weight
model and one for each API model—the confirmatory matrix contains:

```text
3 tasks × 64 configurations × 160 items × (2 + 1 + 1 replicates)
= 122,880 generation calls
```

Primary metrics are deterministic and require no judge calls. Before
preregistration, the excluded-data pilot will estimate input/output tokens, runtime,
failure rates, API charges, and local compute consumption. The final protocol may
reduce or increase replication only from this pilot and power analysis, never after
confirmatory outcomes are inspected.

### 2.3 Experimental and inferential units

- Treatment: one complete prompt configuration.
- Observation: one model response for a configuration, item, model, and replicate.
- Repeated-measures cluster: benchmark item.
- Generalization cell: a preregistered task–model pair.
- Target population: items meeting the registered eligibility rules within the
  studied task classes and behavior of the exact frozen model snapshots.

Responses sharing an item will never be analyzed as independent observations.

## 3. Real datasets and outcomes

Every confirmatory item must be drawn from an identifiable public or publishable
dataset. Generated sentences such as numbered template variations are prohibited.
For each dataset, record the source release, retrieval method, license, original ID,
normalization steps, exclusions, duplicate detection, and split hash.

### 3.1 Schema-bound extraction

- Use naturally occurring text with expert or source-derived structured targets.
- Primary outcome: micro-averaged field-level F1 computed per item and aggregated
  under the registered repeated-measures procedure.
- Secondary outcomes: exact record match and schema-validity rate.
- Invalid or unparsable output is a behavioral failure, not missing data.

### 3.2 Grounded question answering

- Use questions paired with fixed evidence passages and reviewed reference answers.
- Include answerable and unanswerable cases under a frozen proportion.
- Primary outcome: normalized answer-token F1 or a deterministic fact-unit measure.
- Secondary outcomes: exact match, abstention accuracy, and unsupported-claim rate.

### 3.3 Intent classification

- Use a real, licensed multiclass intent corpus with stable labels.
- Primary outcome: item-level correctness, summarized as macro-F1 at the cell level.
- Secondary outcomes: accuracy, per-class recall, and invalid-label rate.

An LLM judge will not be a primary endpoint. If an exploratory judge metric is
retained, its model, prompt, score direction, and cache must be frozen, and a
stratified human sample must validate it before any judge-based claim is made.

### 3.4 Split and leakage controls

The discovery and confirmation partitions will be stratified by label, source, and
registered difficulty variables. Near duplicates, common source documents, or
templated siblings may not cross partitions. Split identifiers are frozen before
any confirmatory model call. The selection pipeline can read discovery targets and
responses only; confirmation access is enabled after recommendations and analysis
code are hashed.

## 4. Prompt factors and design integrity

The six factors remain role framing, task decomposition, demonstrations, output
contract, grounding constraint, and verification pass. Exact task-specific text
will be published verbatim.

Before the pilot, automated rendering tests must prove that toggling one factor
changes only its declared prompt section. Demonstrations may not contain any
confirmatory item or close paraphrase. Token-length differences will be measured
and reported because a factor may act partly through added context length.

The registered fraction uses generators `E = ABC` and `F = BCD`, with defining
relation `I = ABCE = BCDF = ADEF`; it is Resolution IV. All design calculations use
`-1/+1` contrast coding. A marginal effect is reported as the level-1 minus level-0
difference, which equals twice the fitted coefficient under this coding.

Main-effect composition will not be described as interaction-safe. Before a prompt
is finalized, the workflow will inspect the exhaustive two-factor interaction
surface. Any post-screening interaction rule must be registered before data
collection or labeled exploratory.

## 5. Model execution protocol

### 5.1 Provider adapters

Every production adapter must make an actual inference request and return:

- raw response text and finish reason;
- exact requested and returned model identifiers;
- request ID or local generation ID;
- decoding parameters and seed when supported;
- provider system fingerprint or backend revision when available;
- input/output token counts;
- start/end timestamps and latency;
- retry count, terminal error, and safety refusal status;
- cost derived from a dated pricing manifest, or measured local compute time.

No confirmatory code path may accept the deterministic fixture client used by v1.
Fixtures remain available only behind an explicit test or simulation mode, and
their manifests must carry `data_origin: simulated`.

### 5.2 Decoding and replication

The default confirmatory setting is temperature zero with fixed maximum output
length and all supported sampling parameters recorded. The pilot repeats a
preregistered subset to quantify residual nondeterminism. The replication rule is:

- if duplicate disagreement is below the frozen tolerance, one generation per API
  cell is sufficient;
- otherwise use the preregistered replicate count for every affected cell;
- the open-weight anchor receives at least two recorded seeds unless deterministic
  decoding is demonstrated bit-for-bit on the frozen environment.

### 5.3 Randomization, caching, and failures

Configuration order is randomized within item blocks and interleaved across time.
The cache key covers model identifier, model revision, all decoding parameters,
system prompt, rendered prompt, item ID, seed, and adapter version. Paid requests
are resumable without duplication.

Operational retries use a frozen bounded policy. Provider errors that remain after
retries are missing and reported by condition. Refusals, invalid JSON, and invalid
class labels are observed model behavior and receive the registered failure score.
Stopping is permitted only for predefined operational or budget conditions, never
because interim accuracy looks favorable or unfavorable.

## 6. Hypotheses and estimands

### 6.1 Primary hypothesis: held-out selection regret

Within each task–model cell, both methods receive 16 discovery configurations and
the same discovery items. PromptPrism selects a configuration from its fitted main
effects. The balanced-random method fits the same model to each of a frozen set of
random 16-configuration draws. Each selected configuration is scored on untouched
confirmation items.

For a metric to maximize:

```text
regret(method) = confirmation score(discovery exhaustive winner)
               - confirmation score(method's discovery-selected prompt)
```

The primary contrast is `regret(PromptPrism) - regret(random)`, weighted equally
across the nine task–model cells. Negative values favor PromptPrism. Random-design
seeds and their aggregation are frozen before data collection.

Inference uses a cell-stratified, item-cluster bootstrap. Discovery and confirmation
items are resampled within cell, the complete selection procedure is rerun, and the
equal-cell aggregate contrast is recomputed. The one-sided decision requires both
the registered direction and `p < 0.05`; a positive mean alone can never establish
support.

### 6.2 Secondary hypotheses

- **H2 — Effect recovery:** Resolution IV has lower marginal-effect RMSE and higher
  sign agreement above the SESOI than the balanced-random design.
- **H3 — Baseline generalization:** the discovery-selected prompt improves over
  the all-zero baseline on confirmation items under a multiplicity-controlled test.
- **H4 — Uplift efficiency:** the selected prompt captures the registered fraction
  of exhaustive discovery-oracle uplift when the oracle uplift exceeds the SESOI.
- **H5 — Resource efficiency:** the method uses 75% fewer configurations than the
  64-configuration oracle; call, token, time, and dollar savings are empirical
  measurements and are not inferred from the configuration ratio.

If the oracle uplift denominator is zero or below the SESOI, uplift captured is
reported as `NA`, never as 100%.

### 6.3 Multiplicity and uncertainty

- H1 is the single primary test.
- Secondary cell-level and factor-level families use Benjamini–Hochberg control at
  the preregistered false-discovery rate.
- Bootstrap resampling occurs at the item level and retains all configurations and
  replicates belonging to each sampled item.
- Mixed-effects or generalized models are sensitivity analyses; disagreements with
  the bootstrap are reported.
- Practical neutrality requires a preregistered equivalence test. Nonsignificance
  alone does not justify removing a factor.

## 7. Calibration and power gates

Simulation validates the estimator and analysis code; it does not count as LLM
evidence. Each core scenario receives at least 2,000 Monte Carlo replications. The
random-design and OFAT comparators must use real fitted estimators, confidence
intervals, and p-values—never fixed placeholders.

Before confirmatory execution, the following must pass:

1. noise-free programmed effects are recovered to numerical tolerance;
2. nominal 95% intervals achieve the frozen coverage tolerance;
3. Type I error and false-discovery behavior meet the frozen bounds;
4. the chosen item count achieves the registered power for the SESOI;
5. reported regret matches its mathematical definition and cannot be negative when
   defined against the same realization's true oracle;
6. every analysis decision is tested on synthetic and excluded pilot data.

## 8. Preregistration and audit trail

The existing local “frozen” files do not constitute a new preregistration. After
the pilot, create version 2 materials and deposit them in an external timestamped
registry before confirmatory calls begin. The deposit includes:

- protocol and statistical analysis plan;
- dataset IDs, split hashes, and eligibility rules;
- exact prompt factors and rendered design matrix;
- model identifiers, parameters, and replication rules;
- provider failure and stopping rules;
- random comparator seeds;
- frozen analysis source and environment lock;
- hashes for every registered artifact and the Git tag.

Changes after the timestamp require a dated deviation entry. Outcome-dependent
changes are exploratory and cannot replace the registered analysis.

## 9. Data products and reproducibility

Use separate v2 namespaces:

```text
research/configs/real_v2/
research/data/real_v2/
research/results/real_v2/
research/protocol/real_v2/
```

Raw response records are append-only and content-addressed. Public releases contain
all redistributable prompts, responses, metrics, manifests, code, hashes, and
environment metadata. Restricted source text is represented by stable IDs and a
reproducible acquisition script. A clean environment must regenerate all derived
CSV, JSON, tables, figures, and manuscript numbers from the raw manifest.

The current v1 fixture results will be visibly labeled `simulated` and excluded
from every empirical build target.

## 10. Paper reconstruction

The paper is written in two stages.

Before results exist, retain only the introduction, verified related work,
registered methods, and empty table/figure shells. The abstract and conclusion use
no numerical performance claims. The novelty statement will be bounded to the
specific method and comparison supported by the literature review; “first” claims
require systematic evidence and are otherwise removed.

After the analysis is unlocked, tables and figures are generated directly from v2
artifacts. The results section reports supported, null, and adverse outcomes with
the same prominence. The discussion separates statistical performance from tool
engineering, reports exact task/model boundaries, and discloses that the authors
developed PromptPrism.

Required main-paper displays are:

1. design and data-flow diagram;
2. cell-level effect-recovery plot against the exhaustive oracle;
3. cell-level held-out regret with bootstrap intervals;
4. baseline, selected, and exhaustive-discovery-winner confirmation scores;
5. configurations, calls, tokens, time, failures, and dollars by method;
6. interaction/aliasing diagnostic and all negative-result cells.

Tables must fit the page width and contain no embedded figure numbering.

## 11. Decision gates

| Gate | Requirement to proceed |
|---|---|
| G0 — Reset | v1 fixtures labeled simulated; v2 paths cannot read v1 results |
| G1 — Engineering | actual-provider smoke tests, provenance, cache, and leakage tests pass |
| G2 — Calibration | estimator validity, coverage, Type I error, and power gates pass |
| G3 — Pilot | real excluded-data pilot reconciles calls, tokens, failures, time, and cost |
| G4 — Registration | external timestamp, hashes, tag, and independent protocol audit complete |
| G5 — Collection | all expected real calls reconcile or have terminal failure records |
| G6 — Analysis | registered script runs once; deviations and sensitivity results are published |
| G7 — Submission | clean reproduction and independent scientific review pass |

If a gate fails, the project fixes the cause, narrows the claim before registration,
or reports the failure. It does not substitute simulations for missing empirical
evidence.

## 12. Decisions needed before paid execution

The protocol can be implemented and tested without credentials. Before the pilot,
the study owner must approve:

1. the maximum API and local-compute budget;
2. which provider accounts and immutable model snapshots are available;
3. the public dataset choices after license and redistribution review;
4. the external preregistration repository and raw-response archive;
5. whether the preferred 122,880-call matrix is affordable or whether the claim
   must be narrowed before registration.
