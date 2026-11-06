# Research Plan: Scientific Validation of PromptPrism

> **Historical plan — superseded for confirmatory execution on 2026-09-26.** The
> audit of the first paper draft found that its alleged confirmatory outputs came
> from deterministic fixtures rather than LLM inference. Use
> [`real-llm-experiment-plan.md`](real-llm-experiment-plan.md) for the replacement
> empirical study. This file remains as the original design record.

**Status:** Protocol draft; no confirmatory data have been collected or analyzed.

**Working paper title:** *Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization*

**Primary objective:** Determine when fractional factorial designs recover prompt-component effects and select high-performing prompts with fewer evaluated configurations than exhaustive search or budget-matched alternatives.

## 1. Decision this research must support

The project currently makes three related claims that require different evidence:

1. **Estimation claim:** a fractional design can estimate useful prompt-component effects.
2. **Selection claim:** combining the estimated effects produces a prompt that performs well on unseen items.
3. **Efficiency claim:** the method reaches that result with fewer prompt configurations, model calls, tokens, time, or dollars than reasonable alternatives.

The paper will report these claims separately. A method may estimate main effects well while selecting a poor prompt because of interactions, or reduce the configuration count without reducing judge cost. No single ANOVA table establishes all three claims.

The primary scientific question is:

> For a fixed task, model, factor set, and evaluation budget, how accurately do fractional factorial designs recover exhaustive-factorial effects, and how much held-out performance is lost when their recommended prompt replaces the exhaustive optimum?

## 2. Scope and claim boundary

The confirmatory claim will be limited to the task classes, model snapshots, factor definitions, and metrics actually studied. The paper will not claim that a prompt component has a universal effect across tasks or models.

The experimental treatment is a **prompt configuration**. A model response for one configuration, benchmark item, and generation replicate is an observation. The benchmark item is the repeated-measures unit because the same item is evaluated under every configuration assigned to a design.

“Run” will always mean one prompt configuration. Model-call counts will be reported separately:

```text
generation calls = configurations × benchmark items × generation replicates
judge calls      = generation outputs × judge metrics × judge replicates
```

This prevents a 16-configuration design from being described as only 16 API calls.

## 3. Preregistered estimands

For each fixed task-model pair, let `Y(c, i, r)` be the score for configuration `c`, item `i`, and replicate `r`.

### 3.1 Marginal main effect

For factor `j`, the exhaustive-oracle estimand is the average score difference between level 1 and level 0, averaging uniformly over all configurations of the other factors and over the target item population:

```text
tau_j = mean[Y | factor j = 1] - mean[Y | factor j = 0]
```

The paper will call this a *marginal main effect*. Conditional regression coefficients will be labeled separately when interactions are present.

### 3.2 Prompt-selection regret

Let `c_full` be the configuration selected by exhaustive search using discovery items only,
and let `c_method` be the configuration selected by the method under evaluation using the
same discovery items. For a metric to maximize, the primary regret is:

```text
regret = confirmation score of c_full
       - confirmation score of c_method
```

For metrics to minimize, the sign is reversed. The full-factorial and fractional methods
therefore face the same discovery-to-confirmation shift. The best configuration selected
after looking at confirmation scores will be reported only as a descriptive upper bound,
because using it as the primary oracle would bias regret upward. The fraction of exhaustive
uplift captured will be a secondary, easier-to-interpret measure.

### 3.3 Practical neutrality

A factor will be called practically neutral only when an equivalence test supports that its effect lies inside a preregistered smallest effect size of interest, `[-delta, +delta]`. Failure to reject a conventional null hypothesis is not evidence of neutrality.

`delta` will be chosen per primary metric before confirmatory analysis, using domain consequences and simulation-based power calculations. Both the raw metric scale and the standardized effect will be reported.

### 3.4 Cost

Cost will be reported along four axes:

- evaluated configurations;
- generation and judge calls;
- input/output/judge tokens and monetary cost;
- wall-clock latency under a declared concurrency limit.

The paper will distinguish reductions relative to a full factorial from improvements relative to realistic budget-matched methods.

## 4. Research questions and hypotheses

### Primary hypothesis

At a budget of 16 configurations for six binary factors, a preregistered Resolution IV fractional design will have lower mean held-out selection regret than 16 configurations selected uniformly at random, aggregated across the confirmatory task-model pairs.

The aggregation rule, minimum detectable difference, and statistical test will be frozen after the pilot and before confirmatory results are exposed.

### Secondary hypotheses

- **H2 — Effect recovery:** For oracle effects larger than `delta`, the fractional design will recover the correct direction more often and with lower root mean squared error than a budget-matched random design.
- **H3 — Calibration:** Confidence intervals will attain their nominal coverage in simulation and acceptable empirical bootstrap stability on real tasks.
- **H4 — Resolution sensitivity:** Resolution III estimates will degrade as two-factor interactions increase; foldover or higher-resolution confirmation will reduce sign reversals.
- **H5 — Generalization:** A prompt selected on discovery items will improve on the baseline prompt on untouched confirmation items.
- **H6 — Efficiency:** The selected prompt will capture a preregistered fraction of exhaustive-oracle uplift at a lower measured evaluation cost.

### Explicit null and failure interpretation

The primary null is that the fractional design does not reduce expected held-out regret relative to the budget-matched random baseline. A null or adverse result will not be reframed as success. It will narrow the product claim to effect screening, particular interaction regimes, or particular task classes as supported by the evidence.

## 5. Three-study program

The paper will combine controlled simulation, exhaustive real-model experiments, and prospective confirmation. Each study answers a different validity question.

### Study A — Monte Carlo statistical validation

**Purpose:** Verify the implementation against known truth and map the conditions under which each design succeeds or fails.

Simulate at least 2,000 datasets per preregistered scenario with:

- factor counts `k ∈ {5, 6, 10}`;
- full factorial, Resolution III, Resolution IV, Resolution V, Plackett-Burman, and random balanced designs where defined;
- zero, sparse, and dense two-factor interactions;
- selected higher-order interactions that collide with known aliases;
- weak, moderate, and strong item-difficulty variance;
- continuous, bounded, and binary outcomes;
- homoscedastic and heteroscedastic response noise;
- stochastic generation and correlated judge error;
- missing responses and invalid structured outputs.

The simulation will measure:

- effect bias and root mean squared error;
- sign recovery above `delta`;
- Type I error, false discovery rate, and power;
- 90% and 95% interval coverage;
- selection regret and probability of selecting the oracle configuration;
- sensitivity to design resolution and interaction sparsity;
- behavior of the current “omit nonsignificant factors” rule versus equivalence-gated alternatives.

Simulation is a hard gate. If additive scenarios do not recover their programmed effects or interval coverage is materially wrong, real-model confirmatory work pauses until the implementation or analysis is corrected.

### Study B — Exhaustive empirical oracle

**Purpose:** Compare fractional and alternative methods against an observed full-factorial response surface.

The core design uses six binary prompt factors:

| ID | Functional factor | Level 0 | Level 1 |
|---|---|---|---|
| A | Role framing | no role | task-specific expert role |
| B | Task decomposition | direct instruction | explicit subtask decomposition |
| C | Demonstrations | zero-shot | three fixed examples |
| D | Output contract | informal output request | explicit schema/format contract |
| E | Grounding constraint | no added constraint | task-specific evidence constraint |
| F | Verification pass | answer directly | check the answer against requirements |

Exact wording will be task-specific, frozen before the pilot, and published verbatim. Factors will remain semantically separable. Temperature and model choice will not be mixed into the primary prompt-component factorial because they change the data-generating process rather than the prompt text.

All `2^6 = 64` configurations will be evaluated. The primary fractional condition will use the repository’s 16-configuration Resolution IV design. Because the exhaustive response cube contains every configuration, competing designs can be replayed against identical cached outputs without additional generation cost.

#### Task strata

The primary tasks will emphasize objective evaluation:

1. **Schema-bound information extraction** — primary metric: field-level F1; secondary: exact match and schema validity.
2. **Grounded question answering** — primary metric: deterministic fact-unit agreement against reviewed references; secondary: citation/grounding measures.
3. **Intent or policy classification** — primary metric: macro-F1; secondary: accuracy and abstention quality.

Open-ended summarization or writing may be included as an exploratory fourth stratum, but an LLM judge will not be the sole primary outcome of the confirmatory paper.

#### Models

Use at least three frozen model snapshots spanning:

- one open-weight model runnable with fixed decoding seeds;
- one smaller or economical API model;
- one higher-capability API model from a different family.

Exact provider, model identifier, endpoint version, decoding parameters, access date, and system-level settings will be recorded. At least one open-weight model will receive the complete exhaustive study so the response artifact can be reproduced. API-model scope may be reduced only through the preregistered power and budget gate, not after inspecting results.

#### Items and split

The provisional target is at least 160 eligible items per task, stratified before a locked 50/50 discovery-confirmation split. Final item counts will be set by Study A power simulations. Deduplication, contamination checks, exclusion rules, class balance, provenance, and licenses will be documented before execution.

The full factorial may be generated for both halves to establish an oracle, but optimization code will receive discovery labels only. Confirmation results will remain encrypted, access-controlled, or otherwise unavailable to the analysis author until the preregistered analysis script is frozen.

#### Replication and randomization

- Use at least two independent decoding seeds for open-weight stochastic models.
- For nominally deterministic APIs, run a pilot to measure residual nondeterminism and justify the replicate count.
- Randomize configuration order within item blocks and interleave conditions over time.
- Record timestamps, backend identifiers, retries, safety refusals, token counts, and latency.
- If a provider silently changes a model during a block, treat the period as a separate block or rerun the affected block according to the preregistered rule.

### Study C — Prospective held-out validation

**Purpose:** Test the workflow as it would actually be used, without the benefit of an already observed exhaustive cube.

Select at least two new task-model pairs that were not used to tune the protocol. For each pair:

1. freeze the baseline prompt, factors, metrics, `delta`, design, and budget;
2. run PromptPrism on discovery items;
3. generate its recommended prompt without inspecting confirmation outcomes;
4. run the baseline, PromptPrism recommendation, random-search winner, and any other preregistered finalists on untouched confirmation items;
5. perform a Resolution V, full-factorial, or targeted foldover confirmation when the screening design is Resolution III;
6. report every task-model pair, including negative or ambiguous results.

Study C is the main defense against retrospective replay producing an unrealistically clean result.

## 6. Budget-matched comparators

Comparator budgets will be defined in configurations and model calls. When a method uses fewer configurations, its remaining budget will be assigned according to a preregistered rule rather than silently discarded.

### Effect-estimation comparators

- **Random balanced design:** 16 configurations selected to maintain factor-level balance where possible, followed by the same regression family.
- **One-factor-at-a-time ablation:** baseline plus one factor toggle at a time; extra budget goes to preregistered replication.
- **Plackett-Burman screening:** included as a second screening design, with its alias limitations reported.
- **Full factorial oracle:** establishes the target estimands and is not treated as a budget-matched method.

### Prompt-selection comparators

- **Uniform random search:** select the best observed whole configuration under the same call budget.
- **Random design plus fitted linear model:** estimate factor effects from random configurations and select the fitted optimum.
- **Sequential model-based search:** include only if its implementation and hyperparameters can be frozen before the pilot ends.
- **Baseline prompt:** the current production or task-author prompt.

Effect estimation and prompt selection will have separate leaderboards. A black-box optimizer should not be penalized for failing to explain factors, and an explanatory design should not be declared the best optimizer solely because its coefficients are interpretable.

## 7. Analysis plan

### 7.1 Oracle analysis

From the full factorial, calculate marginal main effects and all estimable two-factor interactions directly by paired contrasts over the same items. The exhaustive best configuration will be selected on discovery items and evaluated on confirmation items. The paper will show both the discovery optimum and the confirmation oracle to expose optimizer overfitting.

### 7.2 Fractional-design analysis

The primary analysis will use factor coding and models appropriate to the outcome:

- continuous approximately symmetric scores: linear model with item blocking plus item-clustered uncertainty;
- binary outcomes: logistic mixed-effects model or paired cluster bootstrap;
- bounded or heavily non-normal scores: cluster bootstrap and a preregistered generalized model as sensitivity analysis.

The current fixed-block OLS/ANOVA result will be reported as the tool’s native analysis. A mixed-effects or item-clustered analysis will test whether its uncertainty generalizes beyond the sampled items. Disagreements will be reported rather than selecting whichever analysis is favorable.

### 7.3 Repeated-measures uncertainty

All bootstrap procedures will resample benchmark items as clusters, retaining every configuration and generation replicate for a sampled item. Individual model responses will not be treated as independent observations when they share an item.

### 7.4 Multiplicity

- One primary hypothesis and endpoint will be designated.
- Secondary factor-effect families will use Benjamini-Hochberg false discovery rate control.
- Confirmatory two-factor interactions will be limited to a preregistered subset or treated as exploratory.
- All measured outcomes will appear in the appendix, including adverse metrics.

### 7.5 Equivalence and removal decisions

Factors will be classified as:

- **beneficial:** the corrected interval excludes zero in the beneficial direction and exceeds any required practical threshold;
- **harmful:** the corrected interval excludes zero in the harmful direction;
- **equivalently neutral:** the equivalence test supports `[-delta, +delta]`;
- **inconclusive:** none of the above.

Only equivalently neutral factors support a scientific claim that removal preserves quality. Token savings may still motivate an engineering choice for an inconclusive factor, but that choice will be labeled as a cost-quality tradeoff.

### 7.6 Missingness and failures

Provider errors will receive bounded retries and then remain missing, with rates reported by configuration. Invalid task outputs are observed behavioral failures and will receive the metric’s preregistered failure score. Analyses will include an intention-to-evaluate view and a sensitivity analysis for differential missingness.

### 7.7 LLM judges and human validation

If judge metrics are used:

- judge prompts, model versions, score directions, thresholds, and caches will be frozen;
- presentation order and factor labels will be blinded;
- at least 20% of judged outputs, stratified by score and condition, will receive independent human ratings;
- inter-rater reliability and judge-human agreement will be reported;
- deterministic or human-reviewed outcomes remain primary whenever available.

## 8. Preregistered success and falsification criteria

Exact numeric thresholds will be frozen after Study A and the pilot, before confirmatory data are revealed. The provisional targets are:

- at least 90% correct effect direction for oracle effects larger than `delta`;
- 90%–98% empirical coverage for nominal 95% intervals in core simulations;
- false discovery rate no more than 0.08 when targeting 0.05 in core simulations;
- median held-out regret no more than 10% of the oracle uplift or one task-specific `delta`, whichever is stricter;
- lower paired mean regret than the primary random baseline;
- no unconfirmed production recommendation from a Resolution III result.

The central claim will be considered unsupported if any of the following occurs:

- the implementation fails to recover known additive effects in simulation;
- interval coverage or false discovery control is materially wrong;
- the primary fractional design does not improve held-out regret over its budget-matched baseline;
- significant screening effects commonly reverse under de-aliasing;
- the composed recommendation fails on confirmation items despite apparently correct individual effects;
- conclusions depend on one task, one model family, one metric, or an unvalidated judge.

These outcomes can still support a useful failure-mode paper or a narrower claim.

## 9. Pilot and locking procedure

The pilot uses one task, one model, and data excluded from the confirmatory set. It may be used to repair code, estimate variance, finalize `delta`, and adjust sample size. It may not be used to select favorable primary outcomes, factor definitions, or model families.

After the pilot:

1. freeze protocol and analysis versions with a Git tag;
2. preregister hypotheses, exclusions, sample sizes, metrics, and analysis code;
3. publish or timestamp hashes of prompts, datasets, and configuration files;
4. lock confirmation labels and outputs;
5. execute the confirmatory study once;
6. label every deviation and every post hoc analysis.

## 10. Reproducibility package

The paper should be accompanied by:

```text
research/
  protocol/          preregistration, deviations, ethics and data statements
  configs/           factors, designs, models, decoding, budgets
  data/              manifests, checksums, licenses, split IDs; no restricted raw data
  simulation/        data-generating processes and Monte Carlo runners
  experiments/       immutable execution manifests and resumable runners
  analysis/          frozen confirmatory analysis and exploratory notebooks
  results/           machine-readable aggregate tables and figure inputs
paper/
  main.tex
  references.bib
  figures/
  tables/
```

Large raw responses should live in a versioned artifact store with immutable checksums and a DOI when licensing allows. The repository should contain a small public fixture that reproduces the complete pipeline.

Required provenance fields include Git commit, environment lock hash, model identifier, provider, request parameters, prompt hash, item ID, seed, timestamp, retry history, response hash, evaluator version, judge configuration, token usage, latency, and cost.

## 11. Paper structure

### Abstract

State the problem, the fractional-design approach, the three-study evaluation, the primary quantitative result, and the boundary conditions. Do not write a positive conclusion until the confirmatory analysis is unlocked.

### 1. Introduction

- Prompt optimization is combinatorial and often evaluated without controlled experiments.
- Factorial designs offer interpretable, budget-efficient component estimates under explicit alias and sparsity assumptions.
- The paper asks whether those theoretical advantages survive stochastic models, heterogeneous benchmark items, bounded metrics, and prompt interactions.
- Contributions: a validation protocol, an exhaustive response-surface benchmark, a budget-matched comparison, and an open reproducibility package.

### 2. Related work

Cover prompt component ablation, automatic prompt optimization, best-arm identification, DoE and multicomponent interventions, statistical LLM evaluation, mixed-effects models, and LLM-judge reliability. Verify every citation and publication status through a reproducible search before claiming novelty. Replace unsupported “first” claims with precise comparisons unless the search establishes priority.

### 3. Method

Define factors, design matrices, resolution, aliasing, blocking, estimands, optimizer rules, equivalence tests, and cost accounting. Clearly distinguish marginal effects from conditional coefficients.

### 4. Experimental design

Describe tasks, datasets, models, splits, baselines, randomization, replication, power analysis, preregistration, and the locked analysis workflow.

### 5. Results

Report simulation calibration first, exhaustive effect recovery second, prompt-selection regret third, and prospective validation fourth. Lead with the primary outcome and include uncertainty for every comparison.

### 6. Failure modes and sensitivity analyses

Show interaction density, design resolution, item count, judge noise, model stochasticity, metric family, and model-family heterogeneity. Include cases where PromptPrism loses.

### 7. Discussion and limitations

Separate what is supported for fixed task-model pairs from claims about portability. Discuss researcher degrees of freedom in choosing factors and levels, semantic coupling between components, temporal model drift, benchmark representativeness, and the cost of exhaustive validation.

### 8. Conclusion

State the narrowest conclusion justified by the preregistered evidence.

## 12. Planned figures and tables

- **Figure 1:** Experimental workflow from factor definition through held-out confirmation.
- **Figure 2:** Effect-recovery error and interval coverage by resolution and interaction density.
- **Figure 3:** Held-out regret versus configuration and call budget.
- **Figure 4:** PromptPrism versus comparators across task-model pairs with paired confidence intervals.
- **Figure 5:** Alias-driven sign reversals and their correction through foldover or higher resolution.
- **Table 1:** Designs, run counts, resolution, and estimable effects.
- **Table 2:** Tasks, datasets, models, metrics, item counts, and replication.
- **Table 3:** Primary and secondary outcomes with uncertainty and multiplicity corrections.
- **Table 4:** Generation calls, judge calls, tokens, latency, and monetary cost.
- **Appendix:** Full factor wording, alias maps, exclusions, deviations, complete results, and human-evaluation rubric.

## 13. Literature-review protocol

The existing repository review is a useful starting map, not evidence for a priority claim. Before submission:

1. search at least ACL Anthology, arXiv, Semantic Scholar, Crossref, IEEE Xplore, ACM Digital Library, and Google Scholar;
2. publish exact search strings, dates, inclusion criteria, and a screening log;
3. forward- and backward-chain the closest papers;
4. verify titles, authors, dates, venues, URLs/DOIs, and whether each work is peer reviewed;
5. distinguish prompt selection, prompt generation, component ablation, order-of-addition designs, and factorial component screening;
6. rerun the search immediately before submission.

Suggested query families include combinations of:

- `prompt engineering` with `factorial design`, `fractional factorial`, `design of experiments`, or `ANOVA`;
- `prompt optimization` with `component`, `ablation`, `interaction`, or `screening design`;
- `LLM evaluation` with `blocking`, `mixed effects`, `paired evaluation`, `power analysis`, or `judge reliability`.

## 14. Ethics, licensing, and disclosure

- Use datasets whose licenses permit automated model evaluation and redistribution of identifiers or derived scores.
- Remove personal or sensitive information and document any residual risk.
- Obtain appropriate review or consent before recruiting human annotators; report compensation and working conditions.
- Report model-provider terms that restrict response redistribution.
- Disclose that PromptPrism’s authors are evaluating their own software and mitigate this through preregistration, public artifacts, and an independent statistical review.
- Report all provider funding, credits, affiliations, and conflicts of interest.

## 15. Milestones and decision gates

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| 0. Protocol | Verified literature map, estimands, hypotheses, factor catalog | Independent protocol review complete |
| 1. Simulation | Monte Carlo suite and calibration report | Core additive scenarios pass; failures understood |
| 2. Pilot | One complete task-model response cube | Variance, cost, and power assumptions updated without outcome shopping |
| 3. Preregistration | Timestamped protocol, frozen code, hashed materials | Confirmation data remain inaccessible |
| 4. Exhaustive study | Full response cubes and method replays | Data-integrity checks pass |
| 5. Prospective study | Locked recommendations and confirmation evaluations | All registered cells reported |
| 6. Analysis | Primary results plus robustness report | Reproducibility audit passes |
| 7. Manuscript | Complete paper and artifact appendix | Statistical and adversarial reviews resolved |
| 8. Release | Preprint, code, data manifests, artifact DOI | Licenses and disclosures verified |

## 16. Provisional resource model

For `k = 6`, `I` items, and `R` replicates:

```text
full factorial calls       = 64 × I × R
16-run fractional calls    = 16 × I × R
calls saved per task-model = 48 × I × R
```

At the provisional `I = 160`, `R = 2`, one exhaustive task-model cell requires 20,480 generation calls. The corresponding fractional experiment requires 5,120. With three tasks and three models, a complete exhaustive study would require 184,320 generation calls before judging. This is a planning upper bound; the preregistered power analysis will determine whether fewer items, fewer exhaustive model cells, or local inference can preserve the scientific objective.

The budget will reserve capacity for retries, prospective confirmation, judge validation, storage, and at least one rerun caused by infrastructure failure. Budget reductions will be made before data collection and documented; they will not be selected after seeing which experimental cells favor the method.

## 17. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Strong prompt interactions | Main-effect recipe selects a poor combination | Exhaustive oracle, interaction stress tests, foldover/Resolution V confirmation |
| Model drift during execution | Treatment comparisons mix model versions | Frozen snapshots, interleaved execution, timestamps, block-level drift rules |
| Item pseudoreplication | Confidence intervals are too narrow | Item-clustered bootstrap or mixed effects; paired contrasts |
| `p > 0.05` treated as neutrality | Useful factors are removed without evidence | Equivalence testing with preregistered `delta` |
| Judge bias or instability | Outcome favors phrasing rather than quality | Objective primary metrics, blinding, cache/version control, human validation |
| Optimizer overfits discovery items | Apparent gains disappear | Locked confirmation split and prospective study |
| Selective task/model reporting | Inflated generality | Register and report every task-model cell |
| Unsupported novelty claim | Credibility and review risk | Reproducible literature search and cautious wording |
| High exhaustive cost | Study cannot finish | Simulation first, local-model oracle, staged budget gate |
| Self-evaluation conflict | Confirmation bias | Preregistration, public artifacts, independent statistician/adversarial review |

## 18. Deliberate exclusions

- No universal claim that one prompt recipe transfers across models.
- No production recommendation based solely on a Resolution III screen.
- No claim that nonsignificance proves irrelevance.
- No primary endpoint based only on an unvalidated LLM judge.
- No post hoc removal of unfavorable task-model cells.
- No “first” or “mathematically optimal” language without evidence matching that scope.
- No model fine-tuning, retrieval-system optimization, or hyperparameter search in the primary factorial study.

## 19. Completion criteria

The research program is complete when:

- the protocol and primary analysis are preregistered;
- simulation calibration and failure regimes are reported;
- at least the preregistered exhaustive and prospective task-model cells are complete;
- all primary and secondary outcomes are reproduced from immutable manifests by one command or documented pipeline;
- the manuscript reports positive, null, and adverse findings;
- an independent statistical review and a fresh-context adversarial review have been resolved;
- citations, licenses, disclosures, and artifact hashes have been verified;
- the paper and reproducibility package can be evaluated without relying on private conversational context.
