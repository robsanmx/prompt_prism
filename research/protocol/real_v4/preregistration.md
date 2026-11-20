# Preregistration Protocol: PromptPrism Real-v4 Prospective Selection and Bounded Stability Study

- **Study Identifier:** `prompt-prism-real-v4-selection-stability-2026`
- **Protocol Schema Version:** `1.0.0`
- **Registry Target:** Open Science Framework (OSF) immutable registration prior to Phase 4 screening
- **Date Frozen (Offline Specification):** `2026-11-18`
- **Data Acquisition Status at Registration:** Zero real-v4 screening, validation, confirmation, or stability-wave LLM calls executed (`prospective_outcomes_acquired: false`).

---

## 1. Hypotheses and Co-Primary Decision Contract

### 1.1 Claim P: Prospective Prompt Selection Superiority

- **Null Hypothesis ($H_{0,\text{P}}$):** There exists at least one primary equal-budget comparator $b \in \{\text{balanced\_random}, \text{bayesian\_opt}\}$ such that $G_{\text{select},b} \le 0$.
- **Alternative Hypothesis ($H_{1,\text{P}}$):** $G_{\text{select},\text{balanced\_random}} > 0$ **and** $G_{\text{select},\text{bayesian\_opt}} > 0$.
- **Primary Estimand:**
  $$
  G_{\text{select},b} = \frac{1}{18} \sum_{c=1}^{18} \frac{1}{N_{c,\text{conf}}} \sum_{i=1}^{N_{c,\text{conf}}} \left( Y_{\text{promptprism},c,i} - Y_{b,c,i} \right)
  $$
  evaluated on untouched `clean_confirmation` items under canonical reference rendering (`env_ref`).
- **Alpha & Pass Criterion:** One-sided $\alpha = 0.05$ per comparator under an intersection-union test (IUT). Claim P passes if and only if the paired hierarchical bootstrap one-sided 95% lower confidence bound satisfies $L_{95}(G_{\text{select},b}) > 0$ against **both** comparators.
- **Practical Superiority Criterion:** Reported alongside statistical superiority; requires $L_{95}(G_{\text{select},b}) > \delta_{\text{select}}$ with frozen margin $\delta_{\text{select}} = 0.01$ (`1.0 pp` on $[0, 1]$ primary metrics).

### 1.2 Claim S: Bounded Behavioral Stability Inside the Registered Envelope

- **Null Hypothesis ($H_{0,\text{S}}$):** Either the worst expected perturbation degradation exceeds or equals $\epsilon_{\text{stability}}$ ($D_{\text{stability}} \ge \epsilon_{\text{stability}}$), **or** there exists a primary comparator $b \in \{\text{balanced\_random}, \text{bayesian\_opt}\}$ against which PromptPrism is inferior by at least $\epsilon_{\text{stability}}$ on worst-environment performance ($G_{\text{robust},b} \le -\epsilon_{\text{stability}}$).
- **Alternative Hypothesis ($H_{1,\text{S}}$):** $D_{\text{stability}} < \epsilon_{\text{stability}}$ **and** $G_{\text{robust},\text{balanced\_random}} > -\epsilon_{\text{stability}}$ **and** $G_{\text{robust},\text{bayesian\_opt}} > -\epsilon_{\text{stability}}$.
- **Primary Estimands:**
  $$
  D_{\text{stability}} = \max_{e \in E_{\text{reg}}} \left( \mu_{\text{promptprism},\text{ref}} - \mu_{\text{promptprism},e} \right), \qquad
  G_{\text{robust},b} = \min_{e \in E_{\text{all}}} \mu_{\text{promptprism},e} - \min_{e \in E_{\text{all}}} \mu_{b,e},
  $$
  where $E_{\text{reg}} = \{\text{repeat\_r2}, \text{repeat\_r3}, \text{fmt\_whitespace}, \text{fmt\_delimiter}, \text{word\_paraphrase\_1}, \text{word\_paraphrase\_2}, \text{item\_holdout\_group}, \text{temporal\_wave\_2}\}$ and $E_{\text{all}} = \{\text{ref}\} \cup E_{\text{reg}}$.
- **Alpha, Margin & Pass Criterion:** Simultaneous one-sided $\alpha = 0.05$ across $E_{\text{reg}}$ via paired maximum-drop bootstrap; blinded-simulation-calibrated frozen noninferiority margin $\epsilon_{\text{stability}} = 0.02$ (`epsilon_stability = 0.02`, updated from initial `0.01` candidate prior to any prospective v4 data acquisition because two disjoint $6 \times 240 = 1{,}440$-item confirmation strata have irreducible group-sampling SE $\approx 0.0055\text{--}0.0085$). Claim S passes if and only if:
  1. Envelope integrity is `VALID` (zero missing environments, zero unpaired cells, zero hash mismatches, zero unquarantined provider version changes);
  2. $U_{95,\text{sim}}(D_{\text{stability}}) < \epsilon_{\text{stability}}$; and
  3. $L_{95}(G_{\text{robust},\text{balanced\_random}}) > -\epsilon_{\text{stability}}$ and $L_{95}(G_{\text{robust},\text{bayesian\_opt}}) > -\epsilon_{\text{stability}}$.

### 1.3 Overall Study Verdict & Publication Covenant

- **Overall Study Pass:** Requires **both** Claim P = `PASS` and Claim S = `PASS`.
- **Supporting Endpoints (Secondary Only):**
  - `H_effect`: 6-factor main-effect RMSE reduction relative to 10 frozen balanced-random 16-run screens against the 64-run screening cube ($\delta_{\text{effect}} = 0.002$, relative reduction $\ge 25\%$).
  - Secondary comparators: 7-run OFAT (`run_id` 0 plus six single-factor flips) and untuned baseline (`run_id = 0`).
  - Secondary repeatability metrics: identical-repeat exact-string match rate, score SD across repeats, and discovery resample selection agreement.
  - **Non-rescuing rule:** Secondary endpoints cannot convert a failed Claim P or Claim S into a pass.
- **Anti-Universal-Guarantee Rule:** Any stability claim or certificate applies strictly to the named/hashed operating envelope below. Provider model-version changes in `temporal_wave_2` trigger an explicit `OUT_OF_ENVELOPE` boundary split rather than silent pooling.
- **Publication Commitment:** All four `(Claim P, Claim S)` outcomes (`Pass/Pass`, `Pass/Fail`, `Fail/Pass`, `Fail/Fail`) and any `INVALID_OUT_OF_ENVELOPE` events will be reported verbatim in the manuscript and release manifest.

---

## 2. Registered Operating Envelope

### 2.1 Tasks, Capabilities, Partitions, and Deterministic Metrics

| Task ID | Capability Family | Public Source & License | Group Unit (Zero Cross-Split Overlap) | Primary Metric (`[0, 1]`) | Parser Failure Rule |
|---|---|---|---|---|---|
| `re3d_entity_extraction` | Structured Extraction | Re3d (Open Gov / CC-BY per-corpus, non-v3 docs) | Source document ID | `entity_f1` (micro set F1) | Invalid JSON / missing schema $\mapsto 0.0$ |
| `squad2_grounded_qa` | Grounded QA (w/ Abstention) | SQuAD 2.0 Dev (CC BY-SA 4.0, non-v3 contexts) | Passage context hash | `token_f1` | Empty / truncated / unparseable $\mapsto 0.0$ |
| `banking77_intent` | Fine-Grained Intent Classification | Banking77 (CC BY 4.0, non-v3 label/utterance groups) | Intent-cluster utterance stem | `exact_label_accuracy` | Out-of-inventory label / empty $\mapsto 0.0$ |
| `gsm8k_math` | Quantitative Reasoning | GSM8K (MIT License) | Problem stem template hash | `numeric_exact_match` | Missing final numeric value $\mapsto 0.0$ |
| `mbpp_python` | Executable Code Generation | MBPP Sanitized (CC BY 4.0) | Programming task ID | `assertion_pass_rate` | Syntax / timeout / runtime error $\mapsto 0.0$ |
| `sciq_verification` | Structured Scientific Verification | SciQ (CC BY-NC 3.0 / AllenAI research license) | Support-passage concept stem | `exact_choice_accuracy` | Ambiguous / missing option letter $\mapsto 0.0$ |

**Partition Sizes per Task (Calibrated Baseline):**
- `pilot`: 20 items (quarantined after adapter/parser/repeatability check)
- `screen`: 120 items
- `validation`: 60 items
- `clean_confirmation`: 240 items
- `stability_confirmation` (`item_holdout_group`): 240 difficulty-stratified group-disjoint items

### 2.2 Model Conditions and Decoding Controls

1. `gemini-2.5-flash` (Google Gemini API): `temperature=0.0`, `top_p=1.0`, `seed=42`, `max_output_tokens=512`, `thinking_budget=0`. Pinned to pilot-verified `modelVersion` (`gemini-2.5-flash`).
2. `mlx-community/Mistral-7B-Instruct-v0.3-4bit@a4b8f870474b0eb527f466a03fbc187830d271f5` (Local MLX): 4-bit quantization (`group_size=64`), `temperature=0.0`, `max_tokens=512`, verified against all 7 file SHA-256 digests in `local_mistral_inventory.json`.
3. `mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192` (Local MLX): 4-bit quantization (`group_size=64`), `temperature=0.0`, `max_tokens=512`, `enable_thinking=false`, verified against all 9 file SHA-256 digests in `local_qwen_inventory.json`.

**Repeatability Policy:** During Task 12 pilot, each model executes 3 identical passes over the 20 pilot items per task. If exact score agreement across repeats is `< 1.0`, `R_conf = 3` identical repeats (`ref`, `repeat_r2`, `repeat_r3`) are mandatory; even under observed pilot determinism, `repeat_r2` and `repeat_r3` remain active environments in $E_{\text{reg}}$ to audit confirmation determinism continuously.

### 2.3 Factorial Space, Candidate Rules, and Equal Operational Budget

- **6 Binary Factors (`64` full-cube runs `0..63`):** `A` (`role_specification`), `B` (`reasoning_scaffolding`), `C` (`few_shot_demonstrations`), `D` (`strict_output_contract`), `E` (`grounding_guardrail`), `F` (`self_verification_check`).
- **Tie Rule:** Deterministic lowest `run_id` (`0..63`); zero estimated contrast maps to `-1` (binary level `0`).
- **Operational Call Budget per Method per Cell:**
  - Screening: `16 configs × 120 items × R_screen = 1,920 × R_screen` calls.
  - Validation: `4 candidates × 60 items × 3 val_envs × R_val = 720 × R_val` calls (`val_envs = [ref, fmt_whitespace, word_paraphrase_1]`).
  - Every primary method (`promptprism_robust_v4`, `balanced_random_v4`, `bayesian_opt_v4`) is enforced by `BudgetLedger` to consume the exact same screen and validation configuration-item-environment-repeat counts.

---

## 3. Cryptographic Phase-Lock Sequence

1. `protocol_lock.json`: Hashes protocol Markdown files, schemas, perturbation registry, model registry, dataset manifest, and calibrated config. Uploaded to OSF prior to Phase 4 screening.
2. `pilot_lock.json`: Freezes pilot repeatability measurements and verified provider model version strings; quarantines pilot items.
3. `screen_lock.json`: Freezes append-only `screen_manifest.jsonl` before candidate generation.
4. `candidate_lock.json`: Freezes the $\le 4$ candidate `run_id`s per method, trajectory, and cell.
5. `validation_lock.json` & `final_selection_lock.json`: Freezes validation outcomes and the single final `run_id` per method, trajectory, and cell **before** any confirmation or stability-wave prompt can be rendered or queued.
6. `raw_results_lock.json`: Freezes `confirmation_manifest.jsonl` and `stability_manifest.jsonl` before running the single confirmatory statistical analysis.
