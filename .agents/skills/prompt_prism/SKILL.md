---
name: prompt_prism
description: >-
  PromptPrism 2.0 framework for LLM prompt diagnostics, clause attribution, safe prompt pruning
  (item-clustered contrast t-tests + TOST equivalence), noise-floor auditing, and CI/CD release
  gating. Use whenever the user wants to find which prompt clauses help or hurt, prune dead prompt
  bloat safely, separate real prompt deltas from eval noise, or gate prompt releases and canary
  drift checks in CI/CD.
---

# PromptPrism 2.0: Prompt Diagnostics, Safe Pruning & Release Gating

Use this skill whenever you or the user need to:
- **Attribute modular prompt clauses** across 3 to 23 binary or multi-level factors (persona, few-shot examples, reasoning cues, output contracts, guardrails, delimiters) using balanced `2^(k-p)` fractional factorial or Plackett-Burman designs (35.38% lower 6-factor main-effect RMSE prospectively; 1.40x–4.11x lower contrast RMSE retrospectively).
- **Remove harmful instructions & prune dead bloat (`SAFE_TO_PRUNE`)** — reducing instruction interference / attention dilution, latency and uncached output-token overhead, prompt maintenance debt, and uncached/dynamic input tokens (even when static system prefixes are discounted by prompt caching). Separates proven negligible clauses (`SAFE_TO_PRUNE` via paired TOST equivalence) from underpowered `INCONCLUSIVE_NEED_MORE_ITEMS` factors.
- **Stop chasing eval noise** by measuring the item-sampling noise floor (`RMS_obs`, `RMS_noise`, `RMS_signal_corr`, noise variance share `%`) and running calibrated contrast-specific item-clustered `t`-tests (4.0%–5.1% FDR at `N >= 100`) plus 2,000-draw Rademacher sign-flip randomization for small `N`, instead of miscalibrated pooled fixed-block RCBD ANOVA.
- **Gate production releases & detect drift in CI/CD:**
  - **`prism verify-cert` (offline CI gate, zero LLM calls):** Fails closed whenever any **declared** `OperatingEnvelopeSpec` boundary changes (`task_id`, `model_id`, `model_version`, `dataset_partition_hash`, `parser_hash`, `decoding_config`, `perturbation_env_ids`).
  - **Scheduled `prism certify` canary runs (behavioral drift check):** Because providers can rotate weights silently without changing `model_version`, re-run `prism certify` periodically on a fresh temporal/canary perturbation batch to catch silent behavioral regressions (`U95_sim < epsilon`).

> **Scope & Empirical Honesty:**
> - **Not a black-box prompt optimizer:** Do **not** pitch PromptPrism as an automated prompt search algorithm that beats equal-budget random or Bayesian search on raw peak accuracy. Across both our 1,033-cube retrospective audit (`+0.013 pp` on MMLU; `-0.327 pp` on BBH/LMentry) and our preregistered 501,300-call Real-v4 prospective study (`FAIL_BOTH`: `+0.22 pp` vs. balanced random, `p=0.161`; `+0.14 pp` vs. Bayesian optimization, `p=0.244`), equal-budget random/Bayesian screens tie on winner score. PromptPrism's value is **clause attribution, safe pruning, calibrated eval diagnostics, and release gating**.
> - **Modular clauses vs. coupled system prompts:** Independent binary on/off factors assume clauses can be toggled without making the prompt incoherent. For interdependent rules, ordering-sensitive instructions, or coupled tool descriptions, **bundle coupled clauses into intact multi-level variants** using `Factor(levels=[...])` instead of toggling incoherent sub-clauses.
> - **Single-turn scope vs. multi-turn agent trajectories:** Empirical benchmarks and calibration cover single-turn tasks (classification, extraction, QA, math, code). Full multi-turn agent trajectories are noisy and expensive at `N >= 100` items x 16 runs x perturbations; for agentic systems, target **isolated single-step nodes** (intent router, tool selector, RAG groundedness, output verifier) or replayed state snapshots.

---

## When to Use Which Design

| Scenario | Factors (`k`) | Recommended Design | Runs (`n`) | Resolution | Aliasing & Attribution Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Quick Bloat Screening** | 5 – 11 | `2(5-2)III`, `2(7-4)III`, `2(11-7)III`, or `PB-12` | 8 – 16 | **Res III** | Cuts runs by 75%+; reduces main-effect RMSE by 1.53x across all 60 regular 8-run MMLU fractions vs. full-rank Random-8 (1.59x on canonical cosets). Main effects alias 2FIs — always confirm survivors before deployment. |
| **Clean Main-Effect Attribution** | 4 – 8 | `2(4-1)IV`, `2(6-2)IV`, `2(7-3)IV`, `2(8-4)IV` | 8 – 32 | **Res IV** | Main effects unaliased with 2FIs (aliased with 3FIs); 2FIs alias in pairs. Cuts 6-factor main-effect RMSE by **35.38% prospectively** (winning 17/18 cells) and 1.40x retrospectively (177/178 Cohort-2 cubes). |
| **Interaction Modeling & Audits** | 3 – 5 | `2(5-1)V`, `2(6-1)VI`, or Full Factorial | 16 – 32 | **Res V+** | 2FI matrix is strictly orthogonal (`kappa = 1.0`, avoiding the 55.2% rank-deficiency rate of 16-run balanced random screens). Cuts MMLU 2FI-product RMSE by **4.09x pooled / 4.11x per task**. |
| **Small Factor Sets** | 2 – 3 | `FullFactorial` (`2^k`) | 4 – 8 | **Full** | Tests all `2^k` corners with exact Walsh–Fourier decomposition across all orders. |

---

## Step-by-Step v2.0 Agent Workflow

### Step 1: Decompose Modular Clauses into Factors
Isolate modular prompt clauses so each factor can be toggled cleanly without breaking prompt coherence. If two instructions are tightly coupled or order-dependent, bundle them into coherent variants inside a multi-level `Factor(levels=[...])`:

```python
from prompt_prism import Factor

factors = [
    Factor.binary(
        name="specialist_persona",
        level_0_content="",
        level_1_content="You are a senior domain expert and principal auditor.",
        description="Role / persona instruction",
    ),
    Factor.binary(
        name="few_shot_demos",
        level_0_content="",
        level_1_content="Example 1:\nInput: ...\nOutput: ...",
        description="Few-shot demonstration",
    ),
    Factor.binary(
        name="brief_reasoning",
        level_0_content="",
        level_1_content="Decompose the problem briefly before answering.",
        description="Reasoning scaffolding",
    ),
    Factor.binary(
        name="strict_output_contract",
        level_0_content="Return plain text.",
        level_1_content="OUTPUT FORMAT: Return a valid JSON object matching the schema.",
        description="Strict output contract",
    ),
    Factor.binary(
        name="grounding_guardrail",
        level_0_content="",
        level_1_content="Rule: Never extrapolate or fabricate facts not in the context.",
        description="Grounding constraint",
    ),
]
```

### Step 2: Configure Deterministic Metrics or Cached LLM Judges (`deepeval >= 4.2.0`)
Prefer deterministic metrics when ground truth exists; attach `JudgeCache` whenever using LLM judges. In `deepeval >= 4.2.0`, **every metric is higher-is-better** (`1.0` = pass / no hallucination / no toxicity).

```python
from prompt_prism import ExactMatch, F1Score, JSONValidation, deepeval_metric, JudgeCache

deterministic_metrics = [ExactMatch(), F1Score(), JSONValidation()]

judge_cache = JudgeCache(db_path="judge_cache.db")
factual_geval = deepeval_metric(
    "g_eval",
    name="factual_accuracy",
    criteria="Determine whether actual output factually aligns with expected golden target.",
    evaluation_steps=[
        "Check that key policy rules, numbers, and deadlines match expected target.",
        "Penalize contradictory or fabricated statements.",
    ],
    cache=judge_cache,
)
faithfulness = deepeval_metric("faithfulness", cache=judge_cache)
```

### Step 3: Execute Orthogonal Screening with Response Caching
Keep benchmark splits cleanly partitioned: **Screen** (attribution/pruning), **Untouched Confirmation** (`confirmation-validated`), and **Perturbation/Temporal Canary Wave** (`stability-certified`).

```python
from prompt_prism import Experiment

exp = Experiment.from_factors(
    factors=factors,
    design="2(5-1)V",  # 16 runs instead of 32; kappa=1.0 for main + 2FI
    system_prompt="Execute the task adhering strictly to guidelines.",
    data_template="Context:\n{{ policy_context }}\n\nQuestion:\n{{ input }}",
    metrics=deterministic_metrics,
    target_metric="f1_score",
    title="Production Prompt Attribution & Pruning Audit",
)

results = exp.run(
    dataset=screen_dataset,       # Aim for N >= 100 paired items for standard-N calibration
    client=call_llm,              # Any callable fn(prompt) -> str or LLMClient
    max_workers=4,
    cache_db="prompt_cache.db",
)
```

### Step 4: Run Calibrated Attribution, Noise Floor & Pruning Audit (`exp.diagnose`)
**Do NOT gate significance on pooled fixed-block RCBD ANOVA** (empirical-null audits show pooled RCBD inflates global BH FDR to **18.4%** on MMLU and commits **27.1%** false positives on high-variance formatting factors). Use `exp.diagnose()` (or `prism diagnose`), which runs contrast-specific item-clustered `t`-tests (4.0%–5.1% FDR at `N >= 100`), reports the item-sampling noise floor, runs 2,000-draw sign-flip checks for small `N`, and classifies each clause with paired TOST equivalence:

```python
audit = exp.diagnose(
    alpha=0.05,
    equivalence_margin=0.02,  # |delta| < 2.0 pp required before SAFE_TO_PRUNE
    n_sign_flip=2000,         # Corroborating randomization check (critical when N < 100)
)
print(audit.to_markdown())

for f in audit.factors:
    print(f.factor_id, f.pruning_verdict, f"delta={f.effect_delta:+.4f}", f"q={f.q_value_bh:.4g}")
```

**How to act on each verdict:**
1. **`KEEP`**: BH `q < alpha` and positive effect — retain this clause.
2. **`REMOVE_HARMFUL`**: BH `q < alpha` and negative effect — remove/invert to eliminate instruction interference and unnecessary latency/output tokens.
3. **`SAFE_TO_PRUNE`**: TOST confirms `|delta| < margin` at `alpha` — safe to delete to simplify prompt maintenance, cut latency, and save uncached tokens (`audit.estimated_prunable_tokens`).
4. **`INCONCLUSIVE_NEED_MORE_ITEMS`**: Neither significant nor proven equivalent — do **not** claim "no effect"; collect more paired items before pruning.

### Step 5: Validate on Untouched Confirmation Data & Gate Releases (`prism certify` / `prism verify-cert`)
To promote a pruned prompt from `predicted` to `confirmation-validated` or `stability-certified`, evaluate it on **untouched held-out items** plus registered semantic-preserving perturbations (`fmt_whitespace`, `fmt_delimiter`, `word_paraphrase_1`, `word_paraphrase_2`, `repeat_r2`, `temporal_wave_2`) pinned to an explicit SHA-256-hashed `OperatingEnvelopeSpec`:

```python
from prompt_prism import (
    OperatingEnvelopeSpec,
    evaluate_simultaneous_perturbation_certificate,
)

envelope = OperatingEnvelopeSpec(
    task_id="banking77_intent",
    model_id="gemini-2.5-flash",
    model_version="gemini-2.5-flash",
    dataset_partition_hash="sha256_confirmation_split",
    parser_hash="sha256_exact_label_parser",
    perturbation_env_ids=["fmt_whitespace", "word_paraphrase_1", "temporal_wave_2"],
)

cert = evaluate_simultaneous_perturbation_certificate(
    envelope=envelope,
    reference_scores=conf_ref_scores,
    perturbation_scores_by_env={
        "fmt_whitespace": ws_scores,
        "word_paraphrase_1": para_scores,
        "temporal_wave_2": wave2_scores,
    },
    group_ids=conf_group_ids,
    epsilon_stability=0.02,
    n_bootstrap=2000,
)

# Fail-closed check: any declared mismatch in model_version, dataset, parser, or envs fails verification.
# To catch silent provider weight drift (same model_version), re-run certify on scheduled temporal waves.
final_report = exp.selection_report(
    confirmation_scores=conf_ref_scores,
    confirmation_envelope=envelope,
    stability_certificate=cert,
    request_envelope=envelope,
)
print(final_report.to_markdown())
```

*(Legacy / Experimental note: Single-stage `exp.analyze(strategy="screen_best")` remains supported for 1.x compatibility and always reports `evidence_state = "predicted"`. Two-stage simultaneous-LCB selection requires explicit opt-in via `exp.experimental_select_two_stage(..., experimental_opt_in=True)` or `ExperimentalTwoStageSelector(experimental_opt_in=True)`.)*

---

## CLI Quick Reference (`prompt-prism` / `prism`)

```bash
prism --version

# 1. Design discovery & generation
prism list-designs --factors 6 --min-resolution 4
prism alias --plan "2(6-2)IV"
prism design --factors 6 --plan "2(6-2)IV" --output design_matrix.csv
prism list-metrics

# 2. Calibrated attribution, item-sampling noise floor & TOST pruning audit
prism diagnose \
  --data screen_results.csv \
  --target exact_match \
  --sample-col sample_id \
  --prune-margin 0.02 \
  --token-costs specialist_persona=8,brief_reasoning=12,few_shot_demos=63,strict_output_contract=19 \
  --output-report diagnostic_audit.md \
  --json-out diagnostic_audit.json

# 3. Issue a stability certificate on confirmation + perturbation/canary data (exit 2 on FAIL/INVALID)
prism certify \
  --data stability_wave.csv \
  --target score \
  --task-id banking77_intent \
  --model-id gemini-2.5-flash \
  --model-version gemini-2.5-flash \
  --dataset-hash sha256_conf_split \
  --parser-hash sha256_exact_label \
  --epsilon 0.02 \
  --output stability_cert.json \
  --fail-on-error

# 4. Offline CI/CD config gate — exits 0 ONLY if certificate is PASS and declared envelope matches 100%
prism verify-cert \
  --cert stability_cert.json \
  --task-id banking77_intent \
  --model-id gemini-2.5-flash \
  --model-version gemini-2.5-flash \
  --dataset-hash sha256_conf_split \
  --parser-hash sha256_exact_label \
  --envs fmt_whitespace word_paraphrase_1 temporal_wave_2
```

---

## Critical Methodological Guardrails for Agents

1. **Never Gate Factor Significance on Pooled Fixed-Block RCBD ANOVA:** Always use `exp.diagnose()` / `prism diagnose` (contrast-specific item-clustered `t`-tests + BH FDR; plus `B >= 2,000` sign-flip randomization when `N < 100`). Pooled RCBD inflates global BH FDR to 18.4%.
2. **Check the Noise Floor Before Interpreting Effects:** On small per-task benchmarks (~100 items), 79%–85% of apparent higher-order variance is item-sampling noise. Inspect `audit.noise_floor.noise_variance_share_pct` before chasing small swings.
3. **Never Equate `p >= 0.05` with "Safe to Delete":** Only prune a prompt clause when `pruning_verdict == "SAFE_TO_PRUNE"` (TOST proves `|delta| < margin` at `alpha`). If `INCONCLUSIVE_NEED_MORE_ITEMS`, either keep the clause or evaluate more items.
4. **Bundle Coupled Clauses & Scope Agent Systems to Single-Step Nodes:** Never toggle interdependent instructions or tool schemas independently if doing so breaks prompt coherence (use multi-level `Factor(levels=[...])`). For multi-turn agents, ablate isolated single-step nodes or replayed state snapshots rather than noisy end-to-end trajectories.
5. **Distinguish Declared Config Gates from Silent Drift Checks:** `prism verify-cert` catches *declared* config/version changes offline at zero LLM cost; detecting *silent* provider weight changes requires periodic `prism certify` re-runs on fresh temporal/canary waves.
6. **Never Deploy Unvalidated Additive Sign Extrapolation Under Interactions:** Synthesizing an unseen corner purely from main-effect signs suffers severe regret (8.74 pp on Cohort 2) when interactions are large; always validate any candidate prompt on untouched confirmation items.
7. **Cache Everything:** Always pass `cache_db` on `exp.run(...)` and `JudgeCache` on `deepeval_metric(...)` (`temperature=0.0` on judges).
