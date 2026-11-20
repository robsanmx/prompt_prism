---
name: prompt_prism
description: >-
  PromptPrism 2.0 framework for LLM Prompt Diagnostics, Token-Bloat Pruning, Calibrated
  Factorial Attribution (item-clustered contrast t-tests + TOST equivalence), and Fail-Closed
  Operating-Envelope Stability Certification. Use whenever the user wants to ablate prompt
  components, prune dead prompt tokens without accuracy loss, separate real formatting effects
  from item-sampling noise, or gate prompt releases and model migrations in CI/CD.
---

# 💎 PromptPrism 2.0: Prompt Diagnostics, Token Pruning & Stability Certification

Use this skill whenever you or the user need to:
- **Diagnose & attribute prompt components** across 3 to 23 binary/multi-level factors (persona, few-shot examples, chain-of-thought, output contracts, guardrails, delimiters) using orthogonal **Fractional Factorial ($2^{k-p}$)** or **Plackett-Burman** designs ($35.38\%$ lower 6-factor main-effect RMSE prospectively; $1.40\times$–$4.11\times$ lower contrast RMSE retrospectively).
- **Kill cargo-cult prompt bloat safely (`SAFE_TO_PRUNE`)** by separating statistically negligible clauses (verified via paired **TOST practical equivalence** within a predeclared margin) from underpowered `INCONCLUSIVE_NEED_MORE_ITEMS` factors—cutting prompt tokens and latency without guessing.
- **Stop chasing eval noise** by measuring the **item-sampling noise floor** (`RMS_obs` vs. `RMS_noise` vs. noise-corrected `RMS_corr`) and running **calibrated contrast-specific item-clustered $t$-tests** (plus $B \ge 2{,}000$ Rademacher sign-flip randomization when $N < 100$) instead of miscalibrated pooled fixed-block RCBD ANOVA.
- **Gate production releases & model migrations in CI/CD** using explicit evidence tiers (`predicted` $\rightarrow$ `confirmation-validated` $\rightarrow$ `stability-certified`) and fail-closed SHA-256-hashed **`OperatingEnvelopeSpec` stability certificates** (`prism certify` & `prism verify-cert`).

> **Scope & Empirical Honesty (7.0M Retrospective + 501,300-Call Prospective Real-v4 Audit):**
> Do **not** pitch PromptPrism as a black-box search optimizer that beats equal-budget random or Bayesian search on raw single-winner accuracy. Across both our 1,033-cube retrospective audit ($+0.013\text{ pp}$ on MMLU; $-0.327\text{ pp}$ on BBH/LMentry) and our preregistered 501,300-call Real-v4 prospective study (`FAIL_BOTH`: $+0.22\text{ pp}$ vs. balanced random, $p=0.161$; $+0.14\text{ pp}$ vs. Bayesian optimization, $p=0.244$), equal-budget random/Bayesian screens with Stage-2 validation tie on winner score. PromptPrism 2.0's proven advantage is **interpretable factor attribution, safe token pruning, calibrated paired inference, and envelope-bounded stability certification** ($U_{95,\text{sim}} = 0.0048 < 0.02$ across 7 paired prompt/repeat/temporal perturbation environments).

---

## 📐 When to Use Which Design

| Scenario | Factors ($k$) | Recommended Design | Runs ($N$) | Resolution | Aliasing & Empirical Attribution Behavior |
|:---|:---:|:---:|:---:|:---:|:---|
| **Quick Bloat Screening** | 5 – 11 | `2(5-2)III`, `2(7-4)III`, `2(11-7)III`, or `PB-12` | 8 – 16 | **Res III** | Cuts runs by $75\%+$; reduces main-effect RMSE by $1.53\times$ across all 60 regular 8-run MMLU fractions vs. full-rank Random-8 ($1.59\times$ on canonical cosets). Main effects alias 2FIs—always confirm survivors before deployment. |
| **Clean Main-Effect Attribution** | 4 – 8 | `2(4-1)IV`, `2(6-2)IV`, `2(7-3)IV`, `2(8-4)IV` | 8 – 32 | **Res IV** | Main effects unaliased with 2FIs (aliased with 3FIs); 2FIs alias in pairs. Reduces 6-factor main-effect RMSE by **$35.38\%$ prospectively** (winning $17/18$ cells) and $1.40\times$ retrospectively ($177/178$ Cohort-2 cubes). |
| **Interaction Modeling & Audits** | 3 – 5 | `2(5-1)V`, `2(6-1)VI`, or Full Factorial | 16 – 32 | **Res V+** | $D_{\text{2FI}}$ is strictly orthogonal ($\kappa = 1.0$, avoiding the $55.2\%$ rank-deficiency rate of 16-run balanced random screens). Reduces MMLU 2FI-product RMSE by **$4.09\times$ pooled / $4.11\times$ per task**. |
| **Small Factor Sets** | 2 – 3 | `FullFactorial` ($2^k$) | 4 – 8 | **Full** | Tests all $2^k$ corners with exact Walsh–Fourier decomposition across all orders. |

---

## 🛠️ Step-by-Step v2.0 Agent Workflow

### Step 1: Decompose the Prompt into Clean Binary Factors
Isolate candidate prompt clauses so each factor can be cleanly ablated and its token cost measured:

```python
from prompt_prism import Factor

factors = [
    Factor.binary(
        name="expert_persona",
        level_0_content="",
        level_1_content="You are a senior domain expert and principal auditor.",
        description="Role / persona instruction",
    ),
    Factor.binary(
        name="few_shot_examples",
        level_0_content="",
        level_1_content="Example 1:\nInput: ...\nOutput: ...",
        description="Few-shot demonstration",
    ),
    Factor.binary(
        name="chain_of_thought",
        level_0_content="",
        level_1_content="Think step by step before answering.",
        description="Reasoning scaffolding",
    ),
    Factor.binary(
        name="strict_json_schema",
        level_0_content="Return plain text.",
        level_1_content="OUTPUT FORMAT: Return a valid JSON object matching the schema.",
        description="Strict output contract",
    ),
    Factor.binary(
        name="negative_guardrail",
        level_0_content="",
        level_1_content="Rule: Never extrapolate or fabricate facts not in the context.",
        description="Grounding constraint",
    ),
]
```

---

### Step 2: Configure Deterministic Metrics or Cached LLM Judges (`deepeval >= 4.2.0`)
Prefer deterministic metrics when ground truth exists; attach `JudgeCache` whenever using LLM judges. Note that in `deepeval >= 4.2.0`, **every metric is higher-is-better** (`1.0` = pass / no hallucination / no toxicity).

```python
from prompt_prism import ExactMatch, F1Score, JSONValidation, deepeval_metric, JudgeCache

deterministic_metrics = [ExactMatch(), F1Score(), JSONValidation()]

# Optional semantic / reference-free evaluation with SQLite caching
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

---

### Step 3: Execute Orthogonal Screening with Response Caching
Keep your benchmark split cleanly partitioned: **Screen** (attribution/pruning), **Untouched Confirmation** (`confirmation-validated`), and **Perturbation/Time Wave** (`stability-certified`).

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

---

### Step 4: Run v2.0 Calibrated Attribution, Noise Floor & Token-Pruning Audit (`exp.diagnose`)
**Do NOT gate significance on pooled fixed-block RCBD ANOVA** (empirical-null audits show pooled RCBD inflates global BH FDR to **$18.4\%$** on MMLU and commits **$27.1\%$** false positives on high-variance formatting factors). Use `exp.diagnose()` (or `prism diagnose`), which runs contrast-specific item-clustered $t$-tests ($4.0\%$–$5.1\%$ FDR at $N \ge 100$), reports the item-sampling noise floor, runs $2{,}000$-draw sign-flip checks for small $N$, and classifies each factor with a **TOST equivalence pruning gate**:

```python
audit = exp.diagnose(
    alpha=0.05,
    equivalence_margin=0.01,  # |delta| < 1.0 pp required before SAFE_TO_PRUNE
    n_sign_flip=2000,         # Corroborating randomization check (critical when N < 100)
)
print(audit.to_markdown())

for f in audit.factors:
    print(f.factor_id, f.pruning_verdict, f"delta={f.effect_delta:+.4f}", f"q={f.q_value_bh:.4g}")
```

**How to act on each verdict:**
1. **`KEEP`**: BH $q < \alpha$ and positive effect — retain this clause.
2. **`REMOVE_HARMFUL`**: BH $q < \alpha$ and negative effect — remove/invert immediately.
3. **`SAFE_TO_PRUNE`**: TOST confirms $|\Delta| < \text{margin}$ at $\alpha$ — safe to delete to cut prompt tokens and latency (`audit.estimated_prunable_tokens`).
4. **`INCONCLUSIVE_NEED_MORE_ITEMS`**: Neither significant nor proven equivalent — do **not** claim "no effect"; collect more paired items before pruning.

---

### Step 5: Inspect Legacy Screening Recommendations (`predicted` Tier Only)
When extracting a candidate recipe from a screen, avoid unvalidated additive sign extrapolation when second-order interactions are large ($8.74\text{ pp}$ confirmation regret on Cohort 2; $r=0.478$ with 2FI magnitude). Always wrap single-stage outputs with `exp.selection_report(...)`, which explicitly marks them **`predicted`** (`can_display_passing_stability_badge=False`):

```python
report = exp.analyze(block_by="sample_id", include_interactions=True, strategy="screen_best")
sel_report = exp.selection_report(report=report, strategy="screen_best")
assert sel_report.evidence_state == "predicted"
print(sel_report.to_markdown())
```

---

### Step 6: Validate on Untouched Confirmation Data & Issue Fail-Closed Stability Certificate
To promote a pruned or selected prompt from `predicted` to `confirmation-validated` or `stability-certified`, evaluate it on **untouched held-out items** plus registered semantic-preserving perturbations (`fmt_whitespace`, `fmt_delimiter`, `word_paraphrase_1`, `word_paraphrase_2`, `repeat_r2`, `temporal_wave_2`) pinned to an explicit SHA-256-hashed `OperatingEnvelopeSpec`:

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

# Fail-closed runtime/CI check: any mismatch in model_version, dataset, parser, or envs refuses validation
final_report = exp.selection_report(
    report=report,
    confirmation_scores=conf_ref_scores,
    confirmation_envelope=envelope,
    stability_certificate=cert,
    request_envelope=envelope,
)
print(final_report.to_markdown())
```

*(Note: Two-stage simultaneous-LCB candidate selection is available strictly via explicit opt-in `exp.experimental_select_two_stage(..., experimental_opt_in=True)` or `ExperimentalTwoStageSelector(experimental_opt_in=True)` because the Real-v4 promotion gate required `PASS_BOTH` and observed `FAIL_BOTH`.)*

---

## 💻 CLI v2.0 Quick Reference (`prompt-prism` / `prism`)

```bash
# Check version
prism --version

# 1. Design discovery & generation
prism list-designs --factors 6 --min-resolution 4
prism alias --plan "2(6-2)IV"
prism design --factors 6 --plan "2(6-2)IV" --output design_matrix.csv
prism list-metrics

# 2. [v2.0] Calibrated attribution, item-sampling noise floor & TOST token-bloat pruning
prism diagnose \
  --data screen_results.csv \
  --target f1_score \
  --sample-col sample_id \
  --prune-margin 0.01 \
  --token-costs A=120,B=450,C=90,D=160 \
  --output-report diagnostic_audit.md \
  --json-out diagnostic_audit.json

# 3. [v2.0] Issue a fail-closed stability envelope certificate (exit 2 on FAIL/INVALID with --fail-on-error)
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

# 4. [v2.0] CI/CD release gate — exits 0 ONLY if certificate is PASS and live envelope matches 100%
prism verify-cert \
  --cert stability_cert.json \
  --task-id banking77_intent \
  --model-id gemini-2.5-flash \
  --model-version gemini-2.5-flash \
  --dataset-hash sha256_conf_split \
  --parser-hash sha256_exact_label \
  --envs fmt_whitespace word_paraphrase_1 temporal_wave_2

# 5. Legacy single-stage ANOVA & Pareto chart (explicitly labeled evidence tier: predicted)
prism analyze --data screen_results.csv --target f1_score --strategy screen_best --output-report report.md
```

---

## 🚨 Critical Methodological Guardrails for Agents

1. **Never Gate Factor Significance on Pooled Fixed-Block RCBD ANOVA:** Always use `exp.diagnose()` / `prism diagnose` (contrast-specific item-clustered $t$-tests + BH FDR; plus $B \ge 2{,}000$ sign-flip randomization when $N < 100$). Pooled RCBD divides high-variance contrasts by an averaged denominator ($3.51\times$ too small on MMLU Factor $C$), inflating global BH FDR to $18.4\%$.
2. **Check the Noise Floor Before Interpreting Effects:** On small per-task benchmarks ($\sim 100$ items), $79\%$–$85\%$ of apparent higher-order variance is item-sampling noise. Inspect `audit.noise_floor.noise_variance_share_pct` before chasing small swings.
3. **Never Equate $p \ge 0.05$ with "Safe to Delete":** Only prune a prompt clause when `pruning_verdict == "SAFE_TO_PRUNE"` (TOST proves $|\Delta| < \text{margin}$ at $\alpha$). If `INCONCLUSIVE_NEED_MORE_ITEMS`, either keep the clause or evaluate more items.
4. **Never Deploy Unvalidated Additive Sign Extrapolation Under Interactions:** Synthesizing an unseen corner purely from main-effect signs suffers severe regret ($8.74\text{ pp}$ on Cohort 2) when interactions or template offsets are large. Prefer `screen_best` or two-stage validation on untouched items.
5. **Respect Evidence Tiers (`predicted` $\neq$ `confirmation-validated` $\neq$ `stability-certified`):** Never report in-sample screen accuracy as final production performance, and never display or claim stability outside a verified, SHA-256-hashed `OperatingEnvelopeSpec`.
6. **Cache Everything:** Always pass `cache_db` on `exp.run(...)` and `JudgeCache` on `deepeval_metric(...)` (`temperature=0.0` on judges).
