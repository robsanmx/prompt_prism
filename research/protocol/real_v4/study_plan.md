# Real-v4 study plan: demonstrate effect recovery and prompt selection

## Objective

Real-v4 will test two co-primary claims within a clearly bounded set of tasks and
models:

1. **Effect recovery:** a structured fractional design estimates the full-cube
   marginal effects more accurately than an equal-size balanced random design.
2. **Prompt selection:** a two-stage PromptPrism procedure selects a prompt with a
   higher held-out score than an equal-budget two-stage balanced random procedure.

Both claims must pass. A win on effect estimation alone will not be described as a
prompt-selection improvement. A win on selection alone will not establish the
attribution benefit of factorial screening. Science cannot prove universal
superiority; the intended conclusion is superiority for the equal-weighted mixture of
the registered task distributions and the three exact model conditions, subject to the
frozen practical margins.

## What v3 established and what is lacking

Real-v3 established the engineering and effect-estimation premise:

- PromptPrism mean effect RMSE was 0.00398 versus 0.00931 for balanced random
  screening.
- The 16-run fraction used 75% fewer configurations than the 64-run cube.
- The selection result was effectively zero: random minus PromptPrism was +0.00110,
  95% CI [-0.02734, +0.02869], one-sided p=0.4568.

The evidence needed for a stronger conclusion is missing in five places:

1. The sign-only recipe treats every nonzero estimated effect as certain, including
   effects very close to zero.
2. It composes a potentially unobserved six-factor recipe without a candidate
   validation stage, so interactions can invalidate the additive prediction.
3. There were only three task cells and one model. The selection interval therefore
   remained too wide and the external scope was narrow.
4. The study had a local cryptographic lock but no external timestamp.
5. The effect-recovery result was secondary and had no preregistered practical
   margin or formal decision rule.

## Architecture decision: screen, validate candidates, then confirm

PromptPrism-v4 will retain the registered 16-run Resolution IV screen with
`E=ABC` and `F=BCD`. It will replace the direct sign recipe with a frozen four-candidate
policy:

1. the sign-composed main-effects recipe;
2. the best configuration actually observed in the 16-run fraction;
3. a conservative recipe that activates only effects whose item-bootstrap interval
   clears a frozen practical threshold;
4. an uncertainty alternative obtained by flipping the weakest selected factor in
   the sign recipe.

The four candidates will be evaluated on a candidate-validation partition that is
separate from both screening and final confirmation. The best validation candidate
becomes the final PromptPrism prompt. Ties use the lowest full-cube run ID.

The balanced-random comparator receives exactly the same method budget:

- 16 factor-balanced, full-rank screening configurations;
- its four best screening configurations on the same validation-item count;
- one final selected configuration on confirmation.

Twenty random design sets will be frozen before inference. The primary random score
will be their mean, with the design-to-design distribution reported. Cached responses
may be shared across overlapping configurations, but resource accounting will report
both unique research calls and the calls a single deployment of each method requires.

The candidate policy will be developed only on real-v3 and simulation data. All v3
items and outputs are quarantined from v4 confirmation. If cross-validation on the
development data shows no plausible selection gain, the confirmatory API run will not
start; the method will be revised first.

## Co-primary estimands and decision rules

### H1: practical effect-recovery improvement

For every task-model cell, the full 64-run screening cube defines six reference
marginal effects. Let `RMSE_P` be PromptPrism's six-effect RMSE and `RMSE_R` the
mean RMSE across the twenty balanced random screens. Define:

`G_effect = RMSE_R - RMSE_P`.

Positive values favor PromptPrism. H1 passes if the one-sided 95% lower confidence
bound is above zero. Before data collection, also freeze a smallest effect-recovery
improvement `delta_effect` using blinded simulation. The recommended starting candidate
is 0.002 absolute RMSE and at least 25% relative RMSE reduction. The result is called
practically strong only if the lower bound also clears the absolute margin and the
relative reduction criterion is met.

### H2: practical held-out selection improvement

For confirmation item `i` in task-model cell `c`, let `P_ci` be the score of the
PromptPrism-v4 selected prompt and `R_ci` the score averaged across the final prompts
selected by the twenty random procedures. Define:

`G_select = mean(P_ci - R_ci)`,

with equal weight per task-model cell. H2 passes if the one-sided 95% lower confidence
bound is above zero. The recommended starting smallest effect of interest is +0.01 on
the zero-to-one score scale; practical superiority additionally requires the lower
bound to exceed that frozen margin.

H1 and H2 are co-primary and success requires both, an intersection-union decision.
Each can be tested at one-sided alpha 0.05 without inflating the probability that both
false claims pass. Raw effects, intervals, and p-values will still be reported. A
failure to reject will not be described as equivalence. Equivalence would require a
separate TOST margin and analysis.

## Experimental population

### Models

Use three preregistered model conditions from distinct families and deployment modes:

1. the frozen Gemini hosted model used by the real runner;
2. a frozen Anthropic hosted model; and
3. local `mlx-community/Qwen3-8B-4bit`, snapshot
   `545dc4251c05440727734bcd94334791f6ab0192`.

The Qwen snapshot is already cached locally. Its configuration records Qwen3-8B,
4-bit quantization with group size 64, and a 40,960-token maximum position setting.
It is not yet executable in the current environment: no MLX/MLX-LM runtime is installed,
and the local Ollama manifests do not include the 8B model. Runtime installation and a
successful load pilot are therefore preconditions, not assumed capabilities.
The observed files, sizes, SHA-256 hashes, and runtime gaps are recorded in
`local_qwen_inventory.json`.

Freeze exact requested aliases, returned hosted-model metadata, decoding parameters,
provider adapters, and prices before screening. For Qwen also freeze the model snapshot,
SHA-256 hashes and sizes of every resolved model file, quantization, tokenizer and chat
template hashes, inference-runtime and dependency versions, OS and accelerator details,
and the actual context/output limits. Use `enable_thinking=false` for Qwen unless a
different reasoning-mode policy is explicitly preregistered for every model. The claim
about the local condition is bounded to that exact weight/runtime/hardware stack.

The current `.env` still needs a non-empty `ANTHROPIC_API_KEY`. The Qwen condition needs
no provider credential, but it must pass the same output, parser, and provenance gates
as the hosted conditions. Local compute is not recorded as zero resource use: report
wall time, input/output tokens, peak memory, and throughput, and optionally energy.

### Tasks

Use at least six objectively scored tasks spanning at least four capabilities:

- structured entity or field extraction;
- answerable and unanswerable grounded QA;
- fine-grained intent classification;
- exact-answer quantitative or symbolic reasoning;
- executable code generation or transformation;
- one additional structured classification or retrieval task.

Every task must have deterministic scoring, a documented license, complete content
hashes, and enough unused groups for new pilot, screen, validation, and confirmation
partitions. Avoid LLM-as-judge metrics in the co-primary endpoints. If they are used as
secondary outcomes, cache and blind them separately.

### Initial sizing envelope

The starting design for power simulation is:

- 20 pilot items per task;
- 120 screening items per task;
- 60 candidate-validation items per task;
- 240 confirmation items per task;
- six tasks and the three-model Gemini/Anthropic/Qwen matrix, giving 18 task-model cells;
- two repeated confirmation executions when the provider does not expose an immutable
  deterministic snapshot or the local repeatability pilot detects nondeterminism.

This is not the final sample size. Before preregistration, simulate the complete
selection process using v3 empirical response surfaces, cross-model correlation within
tasks, and plausible task/model heterogeneity. Increase tasks, model conditions, or
items until both co-primary superiority tests reach at
least 90% power when the true gains equal the frozen design alternatives (initially
0.002 RMSE and 0.01 score), and maintain no more than 5% Type I error under null
surfaces. Power for the stronger claim that a confidence bound exceeds the practical
margin must be calculated at a larger alternative, not at the margin itself. More
independent tasks should be preferred over many additional items in only three tasks.

As an optimistic rough planning check, the v3 interval implies an aggregate standard
error near 0.0143. Under ideal scaling from the 240 v3 confirmation observations, six tasks x
three models x 240 items would reduce it to about 0.0034. A true one-point selection
gain would then have roughly 91% one-sided power. The calculation treats observations
as if they scaled independently and therefore cannot establish power for the crossed
design. The required end-to-end simulation must preserve shared-task and cross-model
correlations, model heterogeneity, selection error, and unequal score variance; it can
increase the number of tasks or items and can reject this envelope before any v4
outcomes are acquired.

With the starting envelope, one deployment of either selection method uses 1,920
screen calls (`16 x 120`), 240 validation calls (`4 x 60`), and 240 final-confirmation
calls per task-model cell, or 2,400 calls per cell before optional confirmation
repetition. Across 18 cells this is 43,200 method calls. The research audit also
evaluates the 64-run screening cube, but that oracle overhead is not counted as the
operational cost of either method.

## Split and lock sequence

1. Acquire and hash source datasets; allocate whole documents, passages, problems, or
   user utterances to exactly one partition.
2. Use pilot data to verify prompts, parsers, output limits, and endpoint compatibility.
3. Freeze the design registry, candidate rule, random designs, metrics, power report,
   code hashes, and failure rules.
4. Register the complete protocol and hashes in an externally timestamped repository
   such as OSF before the first screening call.
5. Execute screening. Lock the screening manifest before computing candidate lists.
6. Compute and lock four candidates per method and cell.
7. Execute candidate validation. Lock all final selected configurations.
8. Execute only the locked final configurations on untouched confirmation items.
9. Run the frozen hierarchical analysis once. Any additional analysis is labeled
   exploratory.

No confirmation response may exist before the final-selection lock. The runner must
enforce phase-specific manifests and refuse later phases when any protocol, data,
candidate, or prior-manifest hash changes.

## Statistical analysis

- Preserve paired item outcomes across configurations and methods.
- Treat the registered tasks and three models as fixed strata for the primary finite-
  benchmark estimand. Resample whole item groups within each task while carrying all
  model, configuration, and method outcomes together. This preserves the crossed
  task-by-model dependence rather than treating 18 cells as independent.
- Repeat screening and candidate selection inside bootstrap replicates for H1. For H2,
  the observed final selections remain locked and confirmation items are resampled in
  paired form; selection variability is assessed in a nested-bootstrap sensitivity
  analysis. Task-resampling, leave-one-task-out, and multiway task/model bootstraps are
  robustness analyses only. Six curated tasks and three models do not support a broad
  task- or model-population claim.
- Report every cell, not only the aggregate. Require directionally positive estimates
  in a preregistered majority of cells as a robustness gate; the recommended gate is at
  least 12 of 18 cells for selection and no model family with a negative aggregate.
- Report the distribution across the twenty random designs, not only its mean.
- Report baseline and discovery-selected-oracle scores as secondary diagnostics. Never
  replace the discovery-selected oracle with the confirmation-best run.
- Treat terminal errors and truncations according to a frozen rule and keep them in the
  denominator. Report finish reasons by task, model, method, and factor configuration.
- Estimate aggregate effects with equal task-model-cell weights and report hosted and
  local deployment strata separately. A high or low Qwen score must not change the
  weight assigned to its cell.

## Fail-fast gates

The expensive confirmatory run starts only when all gates pass:

1. **Algorithm gate:** nested v3/simulation evaluation shows the four-candidate procedure
   is at least plausible at the selection margin without weakening effect recovery.
2. **Power gate:** >=90% simulated power and <=5% Type I error for both co-primary tests.
3. **Data gate:** all sources, licenses, groups, targets, and split hashes pass audit.
4. **Model gate:** Gemini and Anthropic credentials work, Qwen3-8B loads from the frozen
   snapshot, and every condition completes a 100-call pilot with no material truncation.
   Hosted version/response IDs and local file/runtime/hardware provenance are captured.
   Repeating the local pilot quantifies exact-output and score repeatability and freezes
   the replicate rule used if execution is not deterministic.
5. **Registration gate:** an external timestamp and immutable protocol URL exist.
6. **Budget gate:** a worst-case estimate covering hosted spend and local runtime is
   approved.

## Interpretation matrix

| H1 effect recovery | H2 selection | Allowed conclusion |
|---|---|---|
| Pass | Pass | PromptPrism-v4 improves both attribution and final selection within the registered scope. |
| Pass | Fail | Factorial screening improves attribution, but selection superiority remains unsupported. |
| Fail | Pass | The selector performs better, but the factorial effect-estimation claim is unsupported. |
| Fail | Fail | Neither improvement is established. |

## Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Candidate rule is tuned to v3 | Inflated apparent gain | Quarantine v3 from v4; freeze one rule before acquiring v4 outcomes. |
| Near-zero effects flip recipes | Selection instability | Practical threshold, conservative candidate, and validation stage. |
| Interactions dominate | Additive recipe fails | Include best observed run and uncertainty alternative; consider a 32-run Resolution V sensitivity arm. |
| Too few independent tasks | Wide aggregate interval | Use at least six tasks; preserve the crossed task/model structure in power analysis. |
| Mutable model aliases | Nonreproducible result | Capture returned version/IDs; repeat confirmation; bound claim to observed versions. |
| Local runtime or quantization drifts | Qwen condition cannot be reproduced | Freeze snapshot, file hashes, runtime lock, chat template, hardware, and decoding mode. |
| Qwen execution is nondeterministic | Understated confirmation uncertainty | Run a repeated pilot and use the preregistered replicate rule when outputs or scores differ. |
| Provider message semantics differ | Prompt factors are not comparable | Test canonical rendered messages and freeze each adapter's system/user serialization. |
| Random comparator is cherry-picked | Unfair comparison | Freeze twenty balanced designs and report their full distribution. |
| Research oracle hides operational cost | Misleading efficiency claim | Separate operational method cost from full-cube audit overhead. |
| Local-only lock | Weak temporal evidence | External preregistration with uploaded hashes before calls. |

## Open decisions before implementation

1. Final practical margins for H1 and H2 after blinded power simulation.
2. Exact Gemini and Anthropic versions and the Qwen MLX serving/runtime version.
3. The six objective tasks and availability of enough nonoverlapping groups.
4. Whether the Resolution V 32-run arm is a secondary benchmark or excluded for budget.
5. Maximum approved provider spend.
