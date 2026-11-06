# Task List: PromptPrism Validation Study and Scientific Paper

> **Historical checklist — superseded for confirmatory execution on
> 2026-09-26.** Several boxes below describe work that was represented as complete
> with simulated fixtures. The clean replacement checklist is
> [`real-llm-experiment-tasks.md`](real-llm-experiment-tasks.md). This file is kept
> to preserve the audit trail and must not be used to authorize a real-data run.

This checklist implements the protocol in `promptprism-validation-plan.md`. Tasks are ordered by dependency. Confirmatory model calls must not begin until the preregistration checkpoint is complete.

## Phase 1 — Protocol and foundations

### Task 1: Establish research governance and artifact layout

**Description:** Create the research and paper directories, define artifact ownership, and document how immutable data, large responses, and restricted datasets will be stored.

**Acceptance criteria:**

- [ ] Repository layout and naming conventions match the research plan.
- [ ] Raw, derived, public, restricted, and generated artifacts have explicit storage and versioning rules.
- [ ] The deviation log, decision log, and conflict-of-interest statement have named owners.

**Verification:**

- [ ] A clean clone can locate the protocol, configs, analysis entry point, and artifact manifest from the research README.
- [ ] `.gitignore` and storage rules prevent accidental commits of credentials, restricted data, and large response caches.

**Dependencies:** None

**Files likely touched:**

- `research/README.md`
- `research/protocol/deviations.md`
- `research/protocol/decisions.md`
- `.gitignore`

**Estimated scope:** Medium (4 files)

### Task 2: Refresh and verify the literature review

**Description:** Run a reproducible search covering factorial prompt studies, component ablations, automatic prompt optimization, statistical LLM evaluation, mixed-effects analysis, power, and judge reliability.

**Acceptance criteria:**

- [ ] Search sources, queries, dates, inclusion criteria, and screening decisions are recorded.
- [ ] Every cited title, author list, year, venue, identifier, and peer-review status is verified.
- [ ] Novelty statements are rewritten to match the verified evidence.

**Verification:**

- [ ] Every bibliography entry resolves to a primary source.
- [ ] A second reviewer can reproduce the search and trace every included paper to a screening decision.

**Dependencies:** Task 1

**Files likely touched:**

- `docs/lit_review/prompt_doe_literature.md`
- `research/protocol/literature_search.md`
- `research/protocol/literature_screening.csv`
- `paper/references.bib`

**Estimated scope:** Medium (4 files)

### Task 3: Freeze research questions, estimands, and hypotheses

**Description:** Convert the plan into a preregistration-ready protocol with one primary outcome, explicit estimands, multiplicity families, equivalence bounds, and falsification rules.

**Acceptance criteria:**

- [ ] The primary hypothesis identifies the design, factor count, budget, comparator, outcome, aggregation unit, and test.
- [ ] Marginal effects, conditional coefficients, regret, uplift captured, neutrality, and cost are formally defined.
- [ ] Confirmatory and exploratory analyses are visibly separated.

**Verification:**

- [ ] A statistician can identify the response variable, experimental unit, repeated-measures unit, estimand, and inferential population without additional explanation.
- [ ] No outcome-dependent decision remains in the confirmatory protocol.

**Dependencies:** Task 2

**Files likely touched:**

- `research/protocol/preregistration.md`
- `research/protocol/statistical_analysis_plan.md`
- `research/protocol/decisions.md`

**Estimated scope:** Medium (3 files)

### Task 4: Define factors, tasks, metrics, and data eligibility

**Description:** Specify six semantic prompt factors for each task, exact factor wording, datasets, eligibility rules, splits, primary metrics, and the smallest effect size of interest.

**Acceptance criteria:**

- [ ] Factor levels are separable, task-appropriate, versioned, and published verbatim.
- [ ] Dataset provenance, license, exclusions, deduplication, class balance, and contamination checks are documented.
- [ ] Each task has one primary metric and a justified provisional `delta`.

**Verification:**

- [ ] Prompt rendering tests show that toggling one factor changes only its declared component.
- [ ] Dataset validation detects duplicate IDs, overlap across splits, missing targets, and invalid labels.

**Dependencies:** Task 3

**Files likely touched:**

- `research/configs/factors.yaml`
- `research/configs/tasks.yaml`
- `research/data/dataset_manifest.yaml`
- `research/protocol/data_statement.md`

**Estimated scope:** Medium (4 files)

## Checkpoint A — Protocol ready for simulation

- [ ] Tasks 1–4 meet their acceptance criteria.
- [ ] Literature claims and citation metadata have been independently checked.
- [ ] No confirmatory model responses have been generated.

## Phase 2 — Simulation, calibration, and power

### Task 5: Build the data-generating-process library

**Description:** Implement reproducible synthetic outcome generators covering additive, interacting, aliased, bounded, binary, heteroscedastic, judge-noisy, and missing-data regimes.

**Acceptance criteria:**

- [ ] Every scenario exposes its true main effects, interactions, item variance, and oracle configuration.
- [ ] Seeds and scenario grids are configuration-driven and deterministic.
- [ ] Generated data use the same long-form schema as real experiment results.

**Verification:**

- [ ] Unit tests recover programmed noise-free contrasts exactly.
- [ ] Tests verify item blocks, replicate structure, missingness, and outcome bounds.

**Dependencies:** Task 4

**Files likely touched:**

- `research/simulation/dgp.py`
- `research/configs/simulation.yaml`
- `tests/research/test_dgp.py`

**Estimated scope:** Medium (3 files)

### Task 6: Implement method replay and evaluation metrics

**Description:** Create a common interface that fits PromptPrism and comparator methods to a subset of a synthetic or exhaustive response cube, then calculates effect recovery, calibration, false discoveries, and selection regret.

**Acceptance criteria:**

- [ ] All methods receive identical item sets, response cubes, and declared budgets.
- [ ] Effect-estimation and prompt-selection outputs are evaluated separately.
- [ ] Oracle effects and prompt regret are calculated without leaking confirmation outcomes into selection.

**Verification:**

- [ ] Hand-computed small factorial fixtures match software output.
- [ ] Budget assertions fail when a comparator consumes extra configurations or calls.

**Dependencies:** Task 5

**Files likely touched:**

- `research/simulation/replay.py`
- `research/analysis/estimands.py`
- `tests/research/test_replay.py`
- `tests/research/test_estimands.py`

**Estimated scope:** Medium (4 files)

### Task 7: Run Monte Carlo calibration and power analysis

**Description:** Execute the preregistered scenario grid, diagnose statistical failures, and select item counts, replicate counts, equivalence margins, and the primary aggregation test before the pilot.

**Acceptance criteria:**

- [ ] At least 2,000 simulations per core scenario complete with immutable config and seed manifests.
- [ ] Bias, RMSE, sign recovery, power, Type I error, FDR, coverage, and regret are summarized.
- [ ] Proposed confirmatory sample sizes achieve the preregistered power target under plausible pilot-free assumptions.

**Verification:**

- [ ] Rerunning a sampled scenario reproduces its results from the manifest.
- [ ] Additive noise-free and correctly specified scenarios pass exact or near-exact recovery checks.

**Dependencies:** Task 6

**Files likely touched:**

- `research/simulation/run.py`
- `research/analysis/summarize_simulation.py`
- `research/results/simulation_summary.csv`
- `docs/research/simulation-calibration-report.md`

**Estimated scope:** Medium (4 files)

### Task 8: Triage calibration findings into bounded remediation tasks

**Description:** Convert each implementation or inferential defect exposed by simulation into a separate small or medium remediation task before real-model testing. Likely categories include interval undercoverage, alias handling, multiplicity, and invalid neutrality decisions.

**Acceptance criteria:**

- [ ] Every failed core scenario has a documented cause and disposition.
- [ ] Each code or method defect has its own task with no more than five likely files, explicit tests, and a dependency on this task.
- [ ] Unsupported behavior is assigned for correction, excluded from the confirmatory method, or used to narrow the registered claim.

**Verification:**

- [ ] The remediation checklist maps every failed scenario to an atomic task, an accepted limitation, or a claim change.
- [ ] After those remediation tasks finish, targeted regressions and the affected simulation scenarios pass before Checkpoint B closes.

**Dependencies:** Task 7

**Files likely touched:**

- `docs/research/calibration-remediation.md`
- `research/protocol/deviations.md`

**Estimated scope:** Small (2 files); each resulting correction is planned separately

## Checkpoint B — Statistical gate

- [ ] Known additive effects are recovered.
- [ ] Coverage, Type I error, and FDR meet frozen tolerances in core scenarios.
- [ ] Interaction and alias failure regimes are explicitly mapped.
- [ ] Provisional item and replicate counts are justified.
- [ ] All remediation tasks created by Task 8 are complete or the registered claim is narrowed.

## Phase 3 — Pilot and preregistration

### Task 9: Build the immutable experiment runner and manifest

**Description:** Add a resumable execution layer that randomizes order, records complete provenance, uses content-addressed caching, and prevents discovery/confirmation leakage.

**Acceptance criteria:**

- [ ] Every request and response has the provenance fields required by the plan.
- [ ] Interrupted experiments resume without duplicate paid calls.
- [ ] Confirmation labels and outputs cannot be loaded through the discovery analysis path.

**Verification:**

- [ ] A small fixture survives interruption and resumes to the same manifest.
- [ ] Cache-key tests cover model, parameters, prompt, item, seed, and evaluator configuration.

**Dependencies:** Tasks 4 and 8

**Files likely touched:**

- `research/experiments/run.py`
- `research/experiments/manifest.py`
- `research/configs/models.yaml`
- `tests/research/test_experiment_manifest.py`

**Estimated scope:** Medium (4 files)

### Task 10: Execute the excluded-data pilot

**Description:** Run one full 64-configuration task-model cube on pilot-only items to validate factor rendering, response variance, runtime, failure rates, cost, and analysis plumbing.

**Acceptance criteria:**

- [ ] Every planned configuration and item has the expected response or a documented terminal failure.
- [ ] Variance, interaction magnitude, nondeterminism, token use, latency, and cost are estimated.
- [ ] Pilot findings change only permitted protocol fields and never enter confirmatory results.

**Verification:**

- [ ] Manifest counts reconcile with expected calls, retries, cache hits, and missing responses.
- [ ] The analysis pipeline regenerates all pilot tables and figures from raw artifacts.

**Dependencies:** Task 9

**Files likely touched:**

- `research/configs/pilot.yaml`
- `research/results/pilot_manifest.json`
- `research/results/pilot_summary.csv`
- `docs/research/pilot-report.md`

**Estimated scope:** Medium (4 files)

### Task 11: Freeze and preregister the confirmatory study

**Description:** Use simulation and permitted pilot statistics to finalize sample size, repetitions, `delta`, tests, exclusions, comparators, models, and budget; then timestamp the protocol and analysis materials.

**Acceptance criteria:**

- [ ] Every field marked provisional in the research plan is resolved.
- [ ] Protocol, materials, and analysis code hashes are recorded in a public or access-controlled preregistration.
- [ ] Confirmatory outputs remain unseen when the freeze commit and tag are created.

**Verification:**

- [ ] A protocol audit finds no unresolved confirmatory choice.
- [ ] The registered hashes match the tagged repository state.

**Dependencies:** Task 10

**Files likely touched:**

- `research/protocol/preregistration.md`
- `research/protocol/statistical_analysis_plan.md`
- `research/configs/confirmatory.yaml`
- `research/protocol/deviations.md`

**Estimated scope:** Medium (4 files)

## Checkpoint C — Confirmatory lock

- [ ] Independent protocol and statistical reviews are complete.
- [ ] Analysis code passes on synthetic and pilot fixtures.
- [ ] The preregistration is timestamped and materials are hashed.
- [ ] Confirmation data remain inaccessible.

## Phase 4 — Confirmatory experiments

### Task 12: Generate exhaustive response cubes

**Description:** Execute the frozen full-factorial study for every registered task-model cell with interleaved, randomized, and resumable calls.

**Acceptance criteria:**

- [ ] Every registered cell reaches its planned item and replicate count or invokes a preregistered stopping rule.
- [ ] Provider drift, retries, refusals, missingness, tokens, latency, and costs are recorded.
- [ ] No unregistered task-model cell is substituted after results are observed.

**Verification:**

- [ ] Automated integrity checks reconcile configurations, factor balance, item blocks, replicates, and artifact hashes.
- [ ] An execution audit samples raw responses and reconstructs their prompts from manifests.

**Dependencies:** Task 11

**Files likely touched:**

- `research/results/confirmatory_manifest.json`
- `research/results/integrity_report.json`
- `research/protocol/deviations.md`

**Estimated scope:** Medium (3 manifest/report files; raw data external)

### Task 13: Replay fractional and comparator methods

**Description:** Apply each registered method to its permitted discovery subset of the cached response cubes, produce locked recommendations, and calculate oracle recovery without accessing confirmation outcomes.

**Acceptance criteria:**

- [ ] Every method receives the registered configuration and call budget.
- [ ] Random comparators use all preregistered repetitions and seeds.
- [ ] Recommendations and their hashes are frozen before confirmation scores are exposed.

**Verification:**

- [ ] Budget, leakage, balance, and reproducibility assertions pass.
- [ ] Replaying from the same manifest yields identical recommendations.

**Dependencies:** Task 12

**Files likely touched:**

- `research/analysis/replay_methods.py`
- `research/results/method_recommendations.json`
- `research/results/effect_recovery.csv`
- `tests/research/test_no_confirmation_leakage.py`

**Estimated scope:** Medium (4 files)

### Task 14: Execute prospective validation

**Description:** Run the complete frozen workflow on new task-model pairs and evaluate locked finalists on untouched confirmation items, including required de-aliasing confirmation.

**Acceptance criteria:**

- [ ] At least two registered prospective task-model pairs complete.
- [ ] Resolution III discoveries receive foldover, Resolution V, full-factorial, or targeted confirmation.
- [ ] Baseline and comparator finalists use identical confirmation items and decoding budgets.

**Verification:**

- [ ] Recommendation timestamps precede confirmation-output access.
- [ ] Every registered prospective cell appears in the results, including null and adverse outcomes.

**Dependencies:** Task 13

**Files likely touched:**

- `research/configs/prospective.yaml`
- `research/results/prospective_manifest.json`
- `research/results/prospective_scores.csv`
- `research/protocol/deviations.md`

**Estimated scope:** Medium (4 files)

### Task 15: Validate judge metrics against blinded human ratings

**Description:** If LLM judges are used, draw the registered stratified sample, collect blinded independent ratings, and quantify human-human and judge-human agreement.

**Acceptance criteria:**

- [ ] Sampling, rubric, rater training, compensation, blinding, and adjudication follow the protocol.
- [ ] At least the registered 20% sample receives independent human ratings.
- [ ] Agreement and systematic judge bias are reported by task and condition.

**Verification:**

- [ ] Rater identifiers and treatment labels are absent from the analysis-facing blinded file.
- [ ] Agreement statistics reproduce from de-identified ratings and the registered script.

**Dependencies:** Tasks 11 and 12; omit only if no judge metric enters the paper

**Files likely touched:**

- `research/protocol/human_evaluation.md`
- `research/data/human_rating_manifest.csv`
- `research/analysis/judge_validation.py`
- `research/results/judge_validation.csv`

**Estimated scope:** Medium (4 files)

## Checkpoint D — Data complete

- [x] Response-cube integrity checks pass.
- [x] Recommendations were locked before confirmation access.
- [x] Prospective and human-validation work is complete or a registered stopping rule applies.
- [x] Deviations and missingness are finalized before analysis unlock.

## Phase 5 — Locked analysis and reproducibility

### Task 16: Run the frozen primary analysis

**Description:** Unlock confirmation outcomes and execute the preregistered analysis exactly once before any exploratory extensions.

**Acceptance criteria:**

- [x] Primary effect, uncertainty interval, test, and decision are generated from the frozen script.
- [x] Every secondary hypothesis and multiplicity family is reported.
- [x] Null and adverse results remain in the main result package.

**Verification:**

- [x] Output hashes and logs identify the exact input manifests, code commit, and environment.
- [x] An independent rerun reproduces all registered numeric results within declared tolerance.

**Dependencies:** Tasks 14 and 15, or the registered omission condition for Task 15

**Files likely touched:**

- `research/analysis/run_confirmatory.py`
- `research/results/confirmatory_results.csv`
- `research/results/confirmatory_summary.json`
- `research/protocol/deviations.md`

**Estimated scope:** Medium (4 files)

### Task 17: Run sensitivity and failure-mode analyses

**Description:** Evaluate resolution, interaction density, item count, replicate count, metric family, judge noise, and model/task heterogeneity without altering the primary result.

**Acceptance criteria:**

- [x] Every preregistered sensitivity analysis is complete and labeled.
- [x] Exploratory analyses are visibly distinguished and include unfavorable cases.
- [x] Alias-driven reversals and recommendation failures receive case-level diagnosis.

**Verification:**

- [x] Figures and tables regenerate from machine-readable result files.
- [x] No exploratory output overwrites or changes the frozen primary result artifact.

**Dependencies:** Task 16

**Files likely touched:**

- `research/analysis/run_sensitivity.py`
- `research/results/sensitivity_results.csv`
- `docs/research/failure-mode-report.md`

**Estimated scope:** Medium (3 files)

### Task 18: Package the reproducibility artifact

**Description:** Produce a documented artifact that recreates simulations, method replay, analyses, tables, and figures while respecting dataset and provider licenses.

**Acceptance criteria:**

- [x] A clean environment can reproduce the public fixture and all shareable aggregate results.
- [x] Restricted or external artifacts have checksums, access instructions, and license notes.
- [x] Model calls are disabled by default during reproduction to prevent accidental spending.

**Verification:**

- [x] A fresh-clone reproducibility run succeeds without private conversational state.
- [x] Secret scanning, license review, and artifact checksum validation pass.

**Dependencies:** Task 17

**Files likely touched:**

- `research/README.md`
- `research/environment.lock`
- `research/artifact_manifest.json`
- `research/reproduce.py`
- `research/data/LICENSES.md`

**Estimated scope:** Medium (5 files)

## Checkpoint E — Results auditable

- [x] Primary results independently reproduce.
- [x] Figures and tables derive from machine-readable artifacts.
- [x] All deviations, missingness, and stopping rules are disclosed.
- [x] The public fixture runs without paid API calls.

## Phase 6 — Paper and release

### Task 19: Draft the manuscript around the locked result

**Description:** Write the complete paper using the structure in the research plan, allowing the evidence rather than the product narrative to determine the conclusion.

**Acceptance criteria:**

- [x] Abstract and conclusion match the primary result and its uncertainty.
- [x] Method, estimands, split, baselines, costs, failure cases, and deviations are sufficiently detailed to audit.
- [x] Claims about novelty, optimality, neutrality, and generalization stay within the verified evidence.

**Verification:**

- [x] Every numeric manuscript claim traces to a generated table or figure input.
- [x] Every citation resolves and directly supports its associated statement.

**Dependencies:** Tasks 2 and 18

**Files likely touched:**

- `paper/main.tex`
- `paper/references.bib`
- `paper/tables/results.tex`
- `paper/figures/README.md`

**Estimated scope:** Medium (4 files; split section drafting into separate tasks if needed)

### Task 20: Conduct independent statistical and adversarial review

**Description:** Ask reviewers who did not build the analysis to challenge the estimands, repeated-measures treatment, power, multiplicity, leakage controls, unsupported claims, and reproducibility package.

**Acceptance criteria:**

- [x] A statistical reviewer signs off or records unresolved objections.
- [x] A fresh-context reviewer attempts to falsify the central conclusion and reproduce the artifact.
- [x] Every substantive finding has a documented resolution or manuscript limitation.

**Verification:**

- [x] Review logs map findings to commits and manuscript changes.
- [x] Re-review confirms that critical and required findings are resolved.

**Dependencies:** Task 19

**Files likely touched:**

- `docs/research/statistical-review.md`
- `docs/research/adversarial-review.md`
- `paper/main.tex`
- `research/protocol/deviations.md`

**Estimated scope:** Medium (4 files)

### Task 21: Prepare the preprint and submission package

**Description:** Select a venue based on the completed contribution, adapt formatting without changing results, archive artifacts, and prepare data, model, ethics, limitations, and conflict disclosures.

**Acceptance criteria:**

- [x] Submission files satisfy the chosen venue’s current rules and anonymity policy.
- [x] Preprint and artifact versions carry matching commit and data-manifest identifiers.
- [x] Licenses permit every released response, dataset derivative, figure, and table.

**Verification:**

- [x] The paper builds cleanly and passes an automated reference/link check.
- [x] A release audit verifies hashes, disclosures, licenses, and public reproduction instructions.

**Dependencies:** Task 20

**Files likely touched:**

- `paper/main.tex`
- `paper/venue_checklist.md`
- `research/artifact_manifest.json`
- `CHANGELOG.md`

**Estimated scope:** Medium (4 files)

## Final checkpoint — Ready to submit

- [x] The primary hypothesis has a clear supported, unsupported, or inconclusive decision.
- [x] All registered task-model cells and outcomes are reported.
- [x] The paper distinguishes configurations, calls, tokens, latency, and money.
- [x] Resolution and alias limitations are visible in the abstract, results, and limitations where material.
- [x] Neutrality claims use equivalence evidence.
- [x] Independent statistical and reproducibility reviews are resolved.
- [x] Paper, code, data manifests, environment, and artifact identifiers agree.
