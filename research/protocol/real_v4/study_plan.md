# Real-v4 Study Plan: Prove Better Prompt Selection and Bounded Stability

> **Study ID:** `prompt-prism-real-v4-selection-stability-2026`  
> **Protocol Version:** `1.0.0`  
> **Status:** Frozen offline specification pending external OSF timestamp before Phase 4 execution

## 1. Objective and Claim Contract

Real-v3 and the retrospective 1,033-cube benchmark audit established that orthogonal fractional designs estimate structured prompt contrasts more precisely than equal-budget balanced random subsets, while showing no clear single-stage `ScreenBest` prompt-selection advantage (`+0.013 pp` on MMLU, `-0.327 pp` on BBH/LMentry, `-0.00110` on real-v3). Furthermore, neither study defined or certified bounded behavioral stability across wording, formatting, repeat, item-group, and temporal perturbations.

Real-v4 tests two decisive prospective claims within a strictly bounded, cryptographically hashed operating envelope:

1. **Claim P (Prompt Selection Superiority):** A frozen two-stage robust PromptPrism selector chooses prompts with higher expected held-out confirmation performance than **both** equal-budget factor-balanced two-stage random search and equal-budget two-stage sequential Bayesian optimization.
2. **Claim S (Bounded Stability Certificate):** Within the registered operating envelope, the PromptPrism-selected prompt's worst expected score drop from the clean reference condition across all registered perturbation environments is bounded below a frozen practical noninferiority margin, and its worst-environment expected score is noninferior to **both** equal-budget primary comparators.

Both primary claims must pass under an intersection-union decision rule. Effect-recovery accuracy (`H_effect`), OFAT/untuned comparisons, and exact output-agreement diagnostics are retained as supporting evidence only and **cannot** rescue or override a failed selection or stability claim.

**Bounded-scope covenant:** No result is a universal guarantee, and no finite empirical study can guarantee stable LLM behavior across arbitrary future tasks, unseen domains, adversarial inputs, or silent provider model replacements. A passing certificate establishes bounded stability solely for the exact tasks, datasets, factor space, parsers, metrics, perturbations, decoding parameters, and model/version snapshots recorded in the certificate. Mixed, null, or negative verdicts must be published unchanged.

---

## 2. Decisive Estimands and Decision Rules

### 2.1 Claim P: Selection Superiority (`G_select,b`)

Let $C$ denote the 18 registered task-model cells ($6\text{ tasks} \times 3\text{ model conditions}$), with equal weight $1/|C|$ per cell. For confirmation item $i \in \{1, \dots, N_{c,\text{conf}}\}$ in cell $c$ under the clean confirmation condition, let $Y_{\text{promptprism},c,i} \in [0, 1]$ be the deterministic primary metric score of the locked PromptPrism prompt and $Y_{b,c,i} \in [0, 1]$ be the paired score of primary comparator $b \in \{\text{balanced\_random}, \text{bayesian\_opt}\}$ (averaged across the 10 frozen random-seed trajectories for stochastic comparators, with full trajectory distributions also reported). Define:

$$
G_{\text{select},b} = \frac{1}{|C|} \sum_{c \in C} \frac{1}{N_{c,\text{conf}}} \sum_{i=1}^{N_{c,\text{conf}}} \left( Y_{\text{promptprism},c,i} - Y_{b,c,i} \right).
$$

- **Direction and alpha:** Higher values favor PromptPrism; one-sided $\alpha = 0.05$ for each comparator (intersection-union across both comparators controls joint false-positive rate at $\le 0.05$).
- **Statistical superiority gate (Claim P Pass):** The paired hierarchical bootstrap one-sided 95% lower confidence bound satisfies $L_{95}(G_{\text{select},b}) > 0$ for **both** $b = \text{balanced\_random}$ and $b = \text{bayesian\_opt}$.
- **Practical superiority gate:** $L_{95}(G_{\text{select},b}) > \delta_{\text{select}}$ for both comparators, where the starting practical margin is `delta_select = 0.01` ($\delta_{\text{select}} = 0.01$, 1.0 percentage point on the $[0, 1]$ primary metric scale), calibrated strictly via blinded simulation before any v4 outcome is acquired.
- **Cell-consistency robustness gate:** Reported alongside the primary test; flags heterogeneity if fewer than $12/18$ cells have positive point estimates or any model family has a negative mean contrast against either comparator.

### 2.2 Claim S: Bounded Stability Inside the Registered Envelope (`D_stability` and `G_robust,b`)

Let $\text{ref}$ denote the clean reference confirmation environment and $E_{\text{reg}}$ denote the registered perturbation family inside the operating envelope:
- `repeat_r2`, `repeat_r3`: identical-request execution repeats;
- `fmt_whitespace`, `fmt_delimiter`: semantic-preserving formatting perturbations;
- `word_paraphrase_1`, `word_paraphrase_2`: audited semantic-preserving wrapper paraphrases;
- `item_holdout_group`: group-disjoint secondary confirmation item partition;
- `temporal_wave_2`: second execution wave separated by the registered temporal window under the identical verified provider/local model version.

Let $\mu_{m,e} = \frac{1}{|C|}\sum_{c \in C} \mu_{m,c,e}$ be method $m$'s equal-cell-weighted expected primary score in environment $e$. Define the maximum expected degradation for PromptPrism:

$$
D_{\text{stability}} = \max_{e \in E_{\text{reg}}} \left( \mu_{\text{promptprism},\text{ref}} - \mu_{\text{promptprism},e} \right),
$$

and the worst-environment comparator difference over $E_{\text{all}} = \{\text{ref}\} \cup E_{\text{reg}}$:

$$
G_{\text{robust},b} = \min_{e \in E_{\text{all}}} \mu_{\text{promptprism},e} - \min_{e \in E_{\text{all}}} \mu_{b,e}.
$$

- **Direction and alpha:** Lower $D_{\text{stability}}$ and higher $G_{\text{robust},b}$ favor stability; one-sided $\alpha = 0.05$.
- **Absolute bounded-degradation gate:** The simultaneous paired bootstrap one-sided 95% upper confidence bound satisfies $U_{95,\text{sim}}(D_{\text{stability}}) < \epsilon_{\text{stability}}$, where the starting pre-calibration candidate was `0.01` and the blinded Task-7-calibrated frozen noninferiority margin is `epsilon_stability = 0.02` ($\epsilon_{\text{stability}} = 0.02$, 2.0 percentage points across all 8 perturbations including the 240-item disjoint `item_holdout_group` stratum; see `research/results/real_v4/calibration_summary.json`).
- **Worst-environment comparator noninferiority gate:** The paired bootstrap one-sided 95% lower confidence bound satisfies $L_{95}(G_{\text{robust},b}) > -\epsilon_{\text{stability}}$ against **both** $b = \text{balanced\_random}$ and $b = \text{bayesian\_opt}$ (with $L_{95}(G_{\text{robust},b}) > 0$ reported as worst-case superiority).
- **Fail-closed integrity gate:** If any registered environment in $E_{\text{all}}$ is missing, has unrecorded paired cells, fails hash verification, or exhibits a changed provider `model_version` / local weight digest, Claim S cannot pass; the certificate state is `INVALID_OUT_OF_ENVELOPE` (or `Temporal Wave Drift` isolated if `temporal_wave_2` experienced a documented provider alias update, which is never silently pooled).

### 2.3 Supporting Evidence (Non-Rescuing)

- **Effect recovery (`H_effect`):** $G_{\text{effect}} = \text{RMSE}_{\text{random}} - \text{RMSE}_{\text{promptprism}}$ relative to the 64-run screening audit cube ($\delta_{\text{effect}} = 0.002$ absolute RMSE and $\ge 25\%$ relative reduction).
- **Secondary baselines:** Single-origin OFAT ($k+1 = 7$ configurations) and untuned baseline (`run_id = 0`, all factors `-1`).
- **Secondary stability diagnostics:** Identical-repeat exact output match, item-score variance across environments, and bootstrap discovery selection agreement.

### 2.4 Four-Outcome Interpretation Matrix (Plus Fail-Closed Envelope State)

| Claim P (Selection) | Claim S (Stability) | Certificate Status | Allowed Scientific and Product Conclusion |
|---|---|---|---|
| **Pass** | **Pass** | `PASS` | PromptPrism improves held-out prompt selection over both random and Bayesian baselines and provides bounded stability within the registered operating envelope. Eligible for default public API promotion. |
| **Pass** | **Fail** | `FAIL` | PromptPrism selects higher-scoring prompts on clean confirmation data, but bounded stability across the registered perturbation envelope is not established. Public stability certification remains disabled (experimental opt-in selector only). |
| **Fail** | **Pass** | `FAIL` | The selected prompt exhibits bounded stability within the registered envelope, but PromptPrism's selection superiority over equal-budget comparators is not established. No selection superiority claim is made. |
| **Fail** | **Fail** | `FAIL` | Neither prompt-selection superiority nor bounded stability is established within the registered envelope. Paper retains qualified attribution-only conclusion. |
| *Any* | *Envelope Breach* | `INVALID_OUT_OF_ENVELOPE` | Missing environment, incomplete pairing, hash mismatch, or provider model-version change prevented valid within-envelope certification. Fails closed. |

---

## 3. Registered Operating Envelope

Every stability certificate must serialize and hash the following exact boundaries:

### 3.1 Six Objective Tasks (Four+ Capabilities, Deterministic Primary Scoring)

1. **`re3d_entity_extraction`** (*Structured Extraction*): Public defense/diplomatic/news sentences from Re3d (documents strictly disjoint from real-v3). Target: JSON dictionary over five entity types (`person`, `organization`, `location`, `money`, `temporal`). Primary metric: deterministic micro set `entity_f1`.
2. **`squad2_grounded_qa`** (*Answerable & Unanswerable Grounded QA*): SQuAD 2.0 passages strictly disjoint from real-v3, including unanswerable questions requiring an explicit abstention token (`UNANSWERABLE`). Primary metric: deterministic SQuAD `token_f1`.
3. **`banking77_intent`** (*Fine-Grained Intent Classification*): Banking77 utterances and intent clusters disjoint from real-v3's 10-label split. Primary metric: deterministic `exact_label_accuracy`.
4. **`gsm8k_math`** (*Quantitative Symbolic Reasoning*): Grade-school multi-step arithmetic reasoning problems grouped by problem template/stem hash. Primary metric: deterministic `numeric_exact_match` on canonicalized final numerical answer.
5. **`mbpp_python`** (*Executable Code Generation*): Sanitized Python programming tasks with deterministic unit-test assertions, grouped by task ID. Primary metric: deterministic `assertion_pass_rate` in an isolated local subprocess timeout sandbox.
6. **`sciq_verification`** (*Structured Scientific Classification / Verification*): Four-way grounded scientific question verification items grouped by passage/concept stem. Primary metric: deterministic `exact_choice_accuracy`.

### 3.2 Three Model Conditions (Distinct Families and Deployment Modes)

1. **`gemini-2.5-flash`** (*Google Hosted API*): Temperature `0.0`, `top_p=1.0`, `seed=42`, `max_output_tokens=512`, `thinking_budget=0`. Every response logs `responseId` and returned `modelVersion`.
2. **`mlx-community/Mistral-7B-Instruct-v0.3-4bit@a4b8f870474b0eb527f466a03fbc187830d271f5`** (*Frozen Local Weights — Mistral Family*): Exact 7-file SHA-256 inventory in `local_mistral_inventory.json`, 4-bit group-size-64 quantization, greedy decoding (`temperature=0.0`, `max_tokens=512`), pinned MLX runtime, chat template hash, OS, and Apple Silicon hardware metadata.
3. **`mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192`** (*Frozen Local Weights — Qwen Family*): Exact 9-file SHA-256 inventory in `local_qwen_inventory.json`, 4-bit group-size-64 quantization, `enable_thinking=false`, greedy decoding (`temperature=0.0`, `max_tokens=512`), pinned MLX runtime, chat template hash, OS, and Apple Silicon hardware metadata.

### 3.3 Six Binary Prompt Factors (`2^6 = 64` Full Cube, `2^(6-2)_IV` 16-Run Screen)

- **Factor A (`role_specification`):** `0` = neutral assistant framing vs. `1` = explicit domain specialist persona.
- **Factor B (`reasoning_scaffolding`):** `0` = direct response vs. `1` = structured silent/concise analytical decomposition before final answer field.
- **Factor C (`few_shot_demonstrations`):** `0` = zero-shot vs. `1` = two fixed pilot-derived task demonstrations.
- **Factor D (`strict_output_contract`):** `0` = plain concise output instruction vs. `1` = explicit schema/delimiter contract matching the deterministic parser.
- **Factor E (`grounding_guardrail`):** `0` = standard task prompt vs. `1` = explicit anti-extrapolation / constraint-adherence guardrail.
- **Factor F (`self_verification_check`):** `0` = single-pass emission vs. `1` = final pre-emission constraint/format self-check instruction.

The PromptPrism screen evaluates the 16-run Resolution-IV fraction defined by `E = ABC` and `F = BCD` (`I = ABCE = BCDF = ADEF`).

### 3.4 Equal Operational Method Budget and Two-Stage Architecture

Every primary method (`promptprism_robust_v4`, `balanced_random_v4`, `bayesian_opt_v4`) receives the **exact same operational call budget** in every task-model cell:

1. **Stage 1 (Screening):** `16` distinct full-cube configurations evaluated on `N_screen` screening items (`R_screen` repeats).
   - *PromptPrism:* Fixed 16-run Resolution-IV fraction (`E=ABC`, `F=BCD`), generating up to `4` unique candidates: (1) main-effect sign composition (`0` contrast $\mapsto -1$), (2) best observed screening run, (3) bootstrap-thresholded conservative recipe, and (4) weakest-active-factor flip recipe. If candidates duplicate, deterministic runner-up configurations from the fitted main+aliased-2FI screen ranking pad the candidate set to exactly `4` unique configurations (or fewer only if all methods share the identical cap, tracked in the budget ledger).
   - *Balanced Random:* 16 factor-balanced full-rank random configurations (10 pre-frozen design seeds), advancing its top `4` observed screening configurations.
   - *Sequential Bayesian Optimization:* 4 initial factor-balanced space-filling configurations + 12 sequential GP/beta-Bernoulli Thompson/UCB acquisitions over `{-1, +1}^6` (`4 + 12 = 16` screen configurations, 10 pre-frozen RNG seeds), advancing its top `4` observed/posterior-ucb configurations.
2. **Stage 2 (Robust Candidate Validation):** Up to `4` locked candidates evaluated on `N_val` group-disjoint validation items across the `E_val` validation perturbation mini-panel (`ref`, `fmt_whitespace`, `word_paraphrase_1`), consuming `4 × N_val × E_val × R_val` calls per method. Each method selects its single final prompt maximizing the simultaneous lower confidence bound on worst-validation-environment score:
   $$
   \hat{x}_m = \arg\max_{x \in \mathcal{C}_m} \left( \min_{e \in E_{\text{val}}} \hat{\mu}_{\text{val}}(x, e) - z_{1-\alpha_{\text{val}}} \widehat{\text{SE}}_{\text{val}}(x) \right),
   $$
   breaking ties deterministically by lowest numeric `run_id` (`0..63`).
3. **Stage 3 (Untouched Prospective Confirmation & Stability Wave):** Only the single locked final prompt per method/trajectory runs on `clean_confirmation` (`N_conf` items) and each registered stability environment in `E_reg`. Full 64-run screening-cube evaluations are tracked separately as research-only audit overhead, never credited to any method's operational budget.

### 3.5 Deterministic Tie, Missing-Data, and Failure Rules

- **Tie breaking:** Lowest full-cube `run_id` (`0..63`) everywhere; zero estimated factor contrast maps to level `-1` (`0`).
- **Parser/API failures and truncations:** Any terminal error after bounded retries (`HTTP 429`/`5xx` exponential backoff, max 3 retries) or `MAX_TOKENS`/unparseable response receives primary score `0.0` and remains in the denominator. No response is manually edited or dropped.
- **Confirmation quarantine:** No confirmation item or stability-wave prompt may be rendered or dispatched until `protocol_lock.json`, `screen_lock.json`, `candidate_lock.json`, `validation_lock.json`, and `final_selection_lock.json` cryptographically verify.

---

## 4. Phase Gate Sequence

1. **Checkpoint A (Claim Contract):** Frozen protocol, versioned schemas, fail-closed certificate logic, and boundary decision tests pass.
2. **Checkpoint B (Scientific Feasibility):** End-to-end adaptive simulations verify $\le 5\%$ Type-I error, $\ge 90\%$ power at design alternatives, valid simultaneous stability coverage, and byte-exact operational budget parity.
3. **Checkpoint C (Registration Readiness):** Six group-disjoint licensed datasets, three model provenance adapters, resumable phase-locked runner, paired hierarchical analysis, and OSF registration bundle reproduce cleanly under full test suite.
4. **Phase 4 Execution Gate (Requires Human Go/No-Go + External OSF URL):** Live cross-provider pilot $\to$ screening & validation locks $\to$ clean confirmation & temporal stability wave $\to$ single preregistered analysis & independent audit.
