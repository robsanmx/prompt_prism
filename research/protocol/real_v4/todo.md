# Real-v4 execution checklist

## Task 1: Freeze the two co-primary claims

**Description:** Convert the proposed effect-recovery and selection estimands into a
versioned statistical specification with practical margins and a four-outcome
interpretation table.

**Acceptance criteria:**
- [ ] H1 and H2 equations, superiority thresholds, practical margins, direction,
  alpha, and tie rules are explicit.
- [ ] Success requires both claims and cannot be inferred from a secondary metric.
- [ ] Equivalence language is excluded unless a separate TOST is registered.

**Verification:** Independent calculation tests reproduce decisions on synthetic pass,
fail, and boundary cases.

**Dependencies:** None.  
**Likely files:** `research/protocol/real_v4/preregistration.md`, analysis tests.  
**Scope:** Small.

## Task 2: Develop and freeze the four-candidate selector

**Description:** Implement the sign recipe, best observed run, conservative thresholded
recipe, and weakest-factor alternative, followed by validation-set selection.

**Acceptance criteria:**
- [ ] Candidate generation is deterministic and returns at most four unique run IDs.
- [ ] The rule is tuned only on v3/simulations and never reads v4 confirmation data.
- [ ] Balanced random receives the identical 16-screen/4-validation/1-final budget.

**Verification:** Unit tests cover zero effects, tied effects, duplicate candidates,
interactions, and deterministic tie handling.

**Dependencies:** Task 1.  
**Likely files:** new `research/real_v4/selection.py` and tests.  
**Scope:** Medium.

## Task 3: Run blinded power and Type I calibration

**Description:** Simulate the complete screen-to-validation-to-confirmation procedure,
including task/model heterogeneity and comparator selection.

**Acceptance criteria:**
- [ ] Both co-primary tests have at least 90% power at the frozen design alternatives;
  stronger margin-exceedance claims are powered at alternatives above those margins.
- [ ] Each test has at most 5% Type I error under registered null regimes.
- [ ] The selected item and cell counts are justified in a machine-readable report.

**Verification:** Fixed-seed replay reproduces every calibration table and figure.

**Dependencies:** Tasks 1-2.  
**Likely files:** simulation module, calibration config, calibration results.  
**Scope:** Medium.

## Checkpoint A: Scientific feasibility

- [ ] Candidate selector clears the algorithm gate.
- [ ] Both co-primary endpoints clear the power gate.
- [ ] If either fails, revise before acquiring v4 outcomes.

## Task 4: Acquire six objective task datasets

**Description:** Freeze licensed, traceable sources and group-disjoint pilot, screening,
validation, and confirmation partitions.

**Acceptance criteria:**
- [ ] At least six tasks span four capabilities and use deterministic primary metrics.
- [ ] Complete files and derived splits have SHA-256 hashes and source revisions.
- [ ] No document/passage/problem/utterance group crosses partitions or overlaps v3.

**Verification:** Automated count, uniqueness, target, license, hash, and leakage tests.

**Dependencies:** Checkpoint A.  
**Likely files:** `research/data/real_v4/`, dataset builder, integrity tests.  
**Scope:** Medium per task.

## Task 5: Add and validate hosted and local model provenance

**Description:** Configure Gemini, Anthropic, and the locally cached
`mlx-community/Qwen3-8B-4bit` snapshot
`545dc4251c05440727734bcd94334791f6ab0192`, with exact request settings and complete
hosted or local provenance.

**Acceptance criteria:**
- [ ] Gemini and Anthropic credentials are non-empty; `ANTHROPIC_API_KEY` is currently
  the missing value.
- [ ] Install and lock an MLX-compatible serving runtime, load the cached Qwen snapshot,
  and expose it through a dedicated adapter or the existing OpenAI-compatible local
  adapter. No model download or silent conversion occurs during the confirmatory run.
- [ ] Hash every resolved Qwen file and record quantization, tokenizer/chat template,
  runtime/dependency versions, OS, accelerator, memory, context/output limits, decoding
  values, seed handling, and `enable_thinking=false`.
- [ ] A 100-call pilot for each model captures IDs/versions where available, tokens,
  finish reasons, safety/error metadata, retries, latency, and hosted cost or local
  throughput/peak-memory measurements.
- [ ] Repeat the Qwen pilot from the same frozen requests and record exact-output and
  score agreement. If either is below the preregistered threshold, freeze a repeated-run
  aggregation rule before screening.
- [ ] Canonical message-rendering fixtures prove that prompt factors, system/user roles,
  output limits, and metrics have equivalent semantics across all three adapters.

**Verification:** Network-mocked hosted tests, local adapter and serialization tests,
model-file hash replay, and a quarantined three-model real pilot report.

**Dependencies:** Checkpoint A.  
**Likely files:** provider/local adapters, `local_qwen_inventory.json`, model lockfile,
provenance schema, provider tests.  
**Scope:** Medium.

## Task 6: Implement phase locks and resumable execution

**Description:** Enforce screen, candidate lock, validation, final-selection lock, and
confirmation as separate append-only phases.

**Acceptance criteria:**
- [ ] A later phase refuses to run if any prior manifest, recommendation, protocol,
  dataset, code, or design hash changes.
- [ ] Confirmation cannot be queued until final selections are locked.
- [ ] Resume produces no duplicate cells or response IDs.

**Verification:** Tampering and interrupted-run integration tests.

**Dependencies:** Tasks 2, 4, and 5.  
**Likely files:** real-v4 runner, lock module, integration tests.  
**Scope:** Medium.

## Task 7: Implement hierarchical co-primary analysis

**Description:** Add paired hierarchical bootstrap inference that preserves the crossed
task/model structure, cell robustness gates, random-design distributions, effect RMSE,
and operational/audit cost separation.

**Acceptance criteria:**
- [ ] H1 and H2 match hand-calculated fixtures and synthetic known-truth experiments.
- [ ] The registered tasks and three models are fixed primary strata; whole item groups
  are resampled within tasks while all model, configuration, and method outcomes remain
  paired.
- [ ] Task-model cells have equal registered weights in the aggregate, and no analysis
  treats the 18 crossed cells as independent observations.
- [ ] Task-resampling, leave-one-task-out, and multiway bootstraps are clearly labeled
  robustness analyses and do not silently broaden the primary claim.
- [ ] Primary, sensitivity, cell-level, finish-reason, and resource outputs are distinct.

**Verification:** Statistical regression tests and deterministic 2,000-replicate replay.

**Dependencies:** Tasks 1-2.  
**Likely files:** real-v4 analysis, schemas, tests.  
**Scope:** Medium.

## Checkpoint B: Pre-registration readiness

- [ ] Full suite and real-v4 focused tests pass.
- [ ] Dataset, design, provider, failure, and power audits pass.
- [ ] Worst-case hosted spend and local runtime are calculated and approved.
- [ ] Protocol and hashes are externally timestamped before screening.

## Task 8: Execute screening and lock candidates

**Description:** Run the frozen screening cubes, compute both methods' candidates, and
lock all screen-derived artifacts.

**Acceptance criteria:**
- [ ] Expected cells are complete with no duplicates or unreported failures.
- [ ] Candidates reproduce from screening-only data.
- [ ] Candidate lock verifies before validation begins.

**Verification:** Independent manifest replay and hash audit.

**Dependencies:** Checkpoint B.  
**Likely files:** real-v4 results manifests and locks.  
**Scope:** Execution.

## Task 9: Validate candidates and lock final prompts

**Description:** Evaluate the frozen candidate unions on untouched validation items and
lock one final prompt per method, task, model, and random design.

**Acceptance criteria:**
- [ ] Validation contains no screen or confirmation groups.
- [ ] Both methods use the same registered validation budget.
- [ ] Final selections reproduce exactly and are locked before confirmation.

**Verification:** Selection replay and phase-boundary audit.

**Dependencies:** Task 8.  
**Likely files:** validation manifest, final selections, lock.  
**Scope:** Execution.

## Task 10: Execute confirmation and issue the two-part verdict

**Description:** Run only frozen final prompts, execute the registered analysis once,
and update the scientific paper without changing the decision rules.

**Acceptance criteria:**
- [ ] Confirmation contains every expected paired item and no unlocked prompt.
- [ ] H1 and H2 are decided exactly by the registered intersection-union rule.
- [ ] Paper, summary, CSV, manifests, and provenance hashes agree exactly.

**Verification:** Independent audit, full tests, PDF render inspection, and artifact-hash
replay.

**Dependencies:** Task 9.  
**Likely files:** confirmation manifest, summary, results, manuscript.  
**Scope:** Execution and reporting.

## Checkpoint C: Complete

- [ ] Both claim decisions are reported, including negative or mixed outcomes.
- [ ] No universal claim exceeds the preregistered task-model scope.
- [ ] All raw provenance, code, data hashes, locks, and final paper are reviewable.
