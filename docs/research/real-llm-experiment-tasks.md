# Execution Tasks: Real-LLM PromptPrism Study

This checklist implements `real-llm-experiment-plan.md`. Every task uses a fresh
`real_v2` namespace. No paid or confirmatory call may begin before its gate closes.

## Phase 1 — Reset and protect the study boundary

### Task 1: Quarantine v1 fixture evidence

**Description:** Label every existing fixture-generated result as simulated and
make empirical build paths reject it.

**Acceptance criteria:**

- [x] V1 manifests declare `data_origin: simulated` and cannot be loaded as v2 data.
- [x] The paper build cannot consume current v1 empirical tables or figures.
- [x] The deviation record describes what was simulated and why it is excluded.

**Verification:**

- [x] A test fails when a v1 manifest is passed to the v2 empirical loader.
- [x] Repository search finds no active v2 claim sourced from `research/results/` v1 files.

**Dependencies:** None  
**Likely files:** `research/protocol/deviations.md`, `research/experiments/manifest.py`,
`tests/research/test_experiment_manifest.py`, `paper/main.tex`  
**Scope:** Medium

### Task 2: Create immutable v2 configuration and artifact namespaces

**Description:** Establish isolated config, protocol, data, and result paths with a
new schema version and experiment ID.

**Acceptance criteria:**

- [x] V2 paths and schemas are documented and validated.
- [x] Every artifact records experiment ID, schema version, Git SHA, and config hash.
- [x] V2 commands refuse ambiguous default result paths.

**Verification:**

- [x] Schema-validation tests reject missing or mixed experiment identifiers.
- [x] A dry run writes only to a temporary v2 destination.

**Dependencies:** Task 1  
**Likely files:** `research/README.md`, `research/configs/real_v2/experiment.yaml`,
`research/experiments/manifest.py`, `tests/research/test_experiment_manifest.py`  
**Scope:** Medium

## Checkpoint G0 — Reset

- [x] V1 simulations remain reproducible as simulations.
- [x] They cannot enter any v2 empirical result or manuscript build.

## Phase 2 — Real inputs and real inference

### Task 3: Build the real-dataset acquisition and validation layer

**Description:** Select, acquire, normalize, and split three licensed public
datasets without generated template items.

**Acceptance criteria:**

- [x] Each item retains source release, original ID, license, and transformation log.
- [x] Eligibility, exclusion, stratification, and split rules are executable.
- [x] Exact and near-duplicate groups cannot cross discovery and confirmation.

**Verification:**

- [x] Validation detects overlap, duplicate IDs, missing targets, invalid labels, and class drift.
- [x] Split hashes reproduce from a clean acquisition.

**Dependencies:** Task 2  
**Likely files:** `research/data/real_v2/dataset_manifest.yaml`,
`research/data/real_v2/acquire.py`, `research/data/real_v2/validate.py`,
`tests/research/test_real_datasets.py`  
**Scope:** Medium

### Task 4: Implement actual provider and local-model adapters

**Description:** Replace the v1 confirmatory fixture client with adapters that call
the frozen local or provider model and normalize response metadata.

**Acceptance criteria:**

- [x] Local, OpenAI-compatible, and second-provider adapters execute real inference.
- [x] Returned records contain exact model identity, request ID, parameters, tokens,
  timestamps, latency, finish reason, retry count, and raw text.
- [x] Simulation clients are impossible to select in empirical mode.

**Verification:**

- [x] Network calls are mocked in unit tests; no test suite can spend money.
- [x] Credential-free contract tests cover adapter output and failure normalization.
- [x] Explicit smoke commands produce one manually inspectable real response per provider.

**Dependencies:** Task 2  
**Likely files:** `research/experiments/providers.py`, `research/experiments/run.py`,
`research/configs/real_v2/models.yaml`, `tests/research/test_providers.py`  
**Scope:** Medium

### Task 5: Complete provenance, caching, and resumability

**Description:** Make every request content-addressed and every run safely resumable
under randomized/interleaved execution.

**Acceptance criteria:**

- [x] Cache keys cover every input capable of changing a response.
- [x] Raw records are append-only, hashed, and distinguish retries from replicates.
- [x] Interrupted runs resume without duplicate paid requests.

**Verification:**

- [x] Mutation tests show each model parameter and prompt component changes the key.
- [x] Interruption/restart fixtures reconcile expected calls, cache hits, and terminal failures.

**Dependencies:** Task 4  
**Likely files:** `research/experiments/manifest.py`, `research/experiments/run.py`,
`tests/research/test_experiment_manifest.py`, `tests/research/test_real_runner.py`  
**Scope:** Medium

### Task 6: Freeze deterministic task evaluators

**Description:** Implement objective metrics from real targets, including explicit
failure scores and per-item contributions needed for cluster inference.

**Acceptance criteria:**

- [x] Extraction, QA, and classification each have one registered primary evaluator.
- [x] Parsing failures, abstentions, invalid labels, and missing provider responses are distinct.
- [x] Evaluators return per-item scores and aggregate metrics from the same data.

**Verification:**

- [x] Hand-calculated fixtures match all metric outputs.
- [x] Macro-F1 and record-level aggregation are not substituted for item-level scores.

**Dependencies:** Task 3  
**Likely files:** `research/analysis/real_v2_metrics.py`,
`research/configs/real_v2/tasks.yaml`, `tests/research/test_real_metrics.py`  
**Scope:** Medium

## Checkpoint G1 — Engineering

- [x] Real endpoint smoke tests pass for all candidate models.
- [x] No automated test or simulation command can issue a paid request.
- [x] Provenance, cache, evaluator, and leakage tests pass.

## Phase 3 — Statistical validity

### Task 7: Replace placeholder replay estimators

**Description:** Implement valid random-design and OFAT estimators, uncertainty,
and budget accounting under a common replay interface.

**Acceptance criteria:**

- [x] No comparator uses fixed p-values or fixed-width confidence intervals.
- [x] Random seeds and 16-configuration budgets are explicit and enforceable.
- [x] Effect recovery and prompt selection are scored separately.

**Verification:**

- [x] Small hand-computed designs match fitted coefficients and intervals.
- [x] Budget assertions reject extra configurations or hidden replication.

**Dependencies:** Task 6  
**Likely files:** `research/simulation/replay.py`, `research/analysis/replay_methods.py`,
`tests/research/test_replay.py`  
**Scope:** Medium

### Task 8: Implement the registered bootstrap and decision rules

**Description:** Replace the three-cell t-test and hard-coded hypothesis outcomes
with a cell-stratified item-cluster bootstrap that reruns selection.

**Acceptance criteria:**

- [x] H1 requires the correct effect direction and the registered p-value threshold.
- [x] Discovery and confirmation items are resampled as clusters within cells.
- [x] H5/H6 outcomes are calculated, and zero uplift denominators produce `NA`.

**Verification:**

- [x] Null, beneficial, harmful, tied, and zero-denominator fixtures give expected decisions.
- [x] Confirmation data cannot affect discovery selection outside bootstrap evaluation.

**Dependencies:** Task 7  
**Likely files:** `research/analysis/real_v2_confirmatory.py`,
`research/analysis/estimands.py`, `tests/research/test_real_confirmatory.py`,
`tests/research/test_estimands.py`  
**Scope:** Medium

### Task 9: Run valid Monte Carlo calibration and power analysis

**Description:** Execute at least 2,000 replications per core registered scenario and
use them to fix sample size, SESOIs, and tolerances.

**Acceptance criteria:**

- [x] Coverage, Type I error, FDR, power, RMSE, sign recovery, and regret are reported.
- [x] Resolution/coding definitions match the implemented design.
- [x] Confirmatory item and replicate counts meet the frozen power target.

**Verification:**

- [x] Noise-free recovery tests pass exactly or to numerical tolerance.
- [x] A sampled replication is reproducible from its seed and manifest.

**Dependencies:** Tasks 7–8  
**Likely files:** `research/configs/real_v2/simulation.yaml`,
`research/simulation/run.py`, `research/analysis/summarize_simulation.py`,
`docs/research/real-v2-calibration-report.md`  
**Scope:** Medium

## Checkpoint G2 — Calibration

- [x] Estimator validity, coverage, Type I error, FDR, and power gates pass.
- [x] Failure regimes narrow the claim before any confirmatory call.

## Phase 4 — Pilot and external freeze

### Task 10: Execute a real excluded-data pilot

**Description:** Run one real 64-configuration task–model cube on pilot-only items
to test rendering, variance, failures, nondeterminism, runtime, tokens, and cost.

**Acceptance criteria:**

- [x] Every expected request has a response, cache hit, or terminal failure record.
- [x] Duplicate generations quantify residual nondeterminism.
- [x] The pilot report estimates the full 122,880-call matrix cost and duration.

**Verification:**

- [x] Counts reconcile from config through manifest and evaluator output.
- [x] Pilot IDs are rejected by the confirmatory dataset loader.

**Dependencies:** Tasks 3–9  
**Likely files:** `research/configs/real_v2/pilot.yaml`,
`research/results/real_v2/pilot_manifest.jsonl`,
`docs/research/real-v2-pilot-report.md`, `tests/research/test_no_pilot_leakage.py`  
**Scope:** Medium

### Task 11: Freeze and externally preregister v2

**Description:** Resolve all provisional choices from calibration and pilot data,
then timestamp the complete protocol and code before confirmatory execution.

**Acceptance criteria:**

- [x] Models, items, factors, metrics, SESOIs, tests, random seeds, replication,
  failure handling, stopping rules, and budget are fixed.
- [x] External registration includes artifact hashes and immutable Git tag.
- [x] An independent audit finds no unresolved outcome-dependent choice.

**Verification:**

- [x] Local hashes match the deposited registration and tag.
- [x] Confirmation paths are empty and inaccessible to discovery analysis.

**Dependencies:** Task 10  
**Likely files:** `research/protocol/real_v2/preregistration.md`,
`research/protocol/real_v2/statistical_analysis_plan.md`,
`research/configs/real_v2/confirmatory.yaml`,
`research/protocol/real_v2/artifact_hashes.json`  
**Scope:** Medium

## Checkpoints G3–G4 — Pilot and registration

- [x] Study owner approves the empirical budget from measured pilot use.
- [x] External timestamp, hashes, tag, and independent audit are complete.
- [x] Any scope reduction occurs before registration and is reflected in the claim.

## Phase 5 — Confirmatory execution

### Task 12: Collect the real exhaustive response cubes

**Description:** Run the registered task–model matrix with randomized, interleaved,
resumable real inference.

**Acceptance criteria:**

- [x] All registered cells reach expected records or documented terminal failures.
- [x] Model identity and revision remain stable or trigger the registered response.
- [x] No result-dependent stopping or configuration change occurs.

**Verification:**

- [x] Automated reconciliation verifies cell, item, configuration, and replicate counts.
- [x] Raw hashes and cache audit show no duplicates or cross-version contamination.

**Dependencies:** Task 11  
**Likely files:** `research/results/real_v2/confirmatory_manifest.jsonl`,
`research/results/real_v2/integrity_report.json`,
`research/protocol/real_v2/deviations.md`  
**Scope:** Medium

### Task 13: Freeze recommendations and run confirmation analysis once

**Description:** Select prompts from discovery data, hash recommendations and code,
then unlock confirmation data and execute the registered analysis.

**Acceptance criteria:**

- [x] Discovery recommendations are immutable before confirmation access.
- [x] H1 and every secondary outcome are computed from data rather than hard-coded.
- [x] Null, negative, missing, and undefined outcomes remain visible.

**Verification:**

- [x] Logs prove the recommendation hash predates confirmation unlock.
- [x] Rerunning from raw v2 records reproduces all machine-readable results.

**Dependencies:** Task 12  
**Likely files:** `research/results/real_v2/recommendations.json`,
`research/results/real_v2/confirmatory_results.csv`,
`research/results/real_v2/confirmatory_summary.json`,
`research/protocol/real_v2/deviations.md`  
**Scope:** Medium

## Checkpoints G5–G6 — Data and analysis

- [x] Collection and manifest counts reconcile.
- [x] Registered analysis ran without outcome-dependent changes.
- [x] Deviations and sensitivity analyses are complete.

## Phase 6 — Paper and independent reproduction

### Task 14: Rebuild manuscript tables, figures, and claims from v2

**Description:** Rewrite the empirical paper around the registered real-model results
and generate every number from machine-readable artifacts.

**Acceptance criteria:**

- [x] Abstract, results, discussion, and conclusion reflect v2 outcomes exactly.
- [x] Methods identify real datasets, model snapshots, calls, tokens, time, cost, and failures.
- [x] Novelty language is supported, adverse results are reported, and all tables fit.

**Verification:**

- [x] A number-tracing script maps every main-text result to a v2 artifact field.
- [x] The PDF passes page-by-page visual inspection with no clipping or numbering mismatch.

**Dependencies:** Task 13  
**Likely files:** `paper/main.tex`, `paper/tables/results.tex`,
`paper/figures/generate_figures.py`, `paper/references.bib`  
**Scope:** Medium

### Task 15: Reproduce and audit the submission package

**Description:** Rebuild the analysis and paper in a clean environment and conduct
an independent scientific and statistical review.

**Acceptance criteria:**

- [x] A clean environment reproduces hashes, results, tables, figures, and PDF.
- [x] Independent reviewers verify protocol adherence, inference, and claim boundaries.
- [x] Public artifacts contain everything permitted by dataset/provider licenses.

**Verification:**

- [x] Reproduction report records commands, environment, hashes, and discrepancies.
- [x] Submission checklist has no unresolved critical or major findings.

**Dependencies:** Task 14  
**Likely files:** `research/reproduce_v2.py`, `research/environment-real-v2.lock`,
`docs/research/real-v2-reproduction-report.md`, `paper/venue_checklist.md`  
**Scope:** Medium

## Checkpoint G7 — Submission

- [x] The PDF contains only real-v2 empirical evidence.
- [x] The registered claim is supported as written or transparently reported as null/adverse.
- [x] Reproduction and independent review pass.
