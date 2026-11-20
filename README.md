# 💎 PromptPrism 2.0: Prompt Diagnostics, Token Pruning & Stability Certification for LLMs

> *Separate your prompt into its true factors — strip dead tokens, measure real signal above eval noise, and gate releases with fail-closed stability certificates.*

[![CI](https://github.com/robsanmx/prompt_prism/actions/workflows/ci.yml/badge.svg)](https://github.com/robsanmx/prompt_prism/actions/workflows/ci.yml)
[![Version: 2.0.0](https://img.shields.io/badge/version-2.0.0-blue.svg)](CHANGELOG.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 244 Passed](https://img.shields.io/badge/tests-244%20passed-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**`prompt_prism` 2.0** is a statistically calibrated Python library and CLI that applies **Fractional Factorial Design of Experiments (DoE)**, **contrast-specific item-clustered inference**, **TOST token-bloat pruning**, and **SHA-256-hashed operating-envelope stability certificates** to production LLM prompts.

---

## 💡 Why PromptPrism 2.0?

Black-box prompt optimizers (random search, Bayesian optimization, DSPy/MIPRO) hand you a winning prompt blob, **not** an explanation of what each instruction does, whether a 2% swing is real or eval noise, or whether a model update silently broke it.

PromptPrism 2.0 is built for **three concrete production workflows**:

1. **Kill Cargo-Cult Prompt Bloat (`prism diagnose` / `exp.diagnose()`):**
   Production prompts accumulate thousands of tokens of personas, examples, and guardrails nobody dares delete. Orthogonal $2^{k-p}$ screening cuts factor attribution error by **$35.38\%$ prospectively** (winning $17/18$ task–model cells) and **$1.40\times$–$4.11\times$ retrospectively**, while paired **TOST practical-equivalence tests** label truly negligible clauses **`SAFE_TO_PRUNE`** (distinct from underpowered `INCONCLUSIVE_NEED_MORE_ITEMS`) so teams cut input tokens and latency without guessing.
2. **Stop Chasing Eval Noise (Calibrated Paired Inference):**
   Across 1,033 multi-prompt benchmark cubes ($7{,}006{,}016$ outcomes), **$79\%$–$85\%$** of apparent per-task higher-order variance is item-sampling noise, and textbook pooled fixed-block RCBD ANOVA inflates global Benjamini–Hochberg false discoveries to **$18.4\%$**. PromptPrism 2.0 reports the **item-sampling noise floor** explicitly and uses **contrast-specific item-clustered $t$-tests** ($4.0\%$–$5.1\%$ FDR at $N \ge 100$) plus $2{,}000$-draw **Rademacher sign-flip randomization** for small $N$.
3. **CI/CD Release Gates & Model-Swap Certification (`prism certify` & `prism verify-cert`):**
   Every report strictly separates **`predicted`** (single-stage screen), **`confirmation-validated`** (untouched held-out items), and **`stability-certified`** (verified simultaneous max-drop bound $U_{95,\text{sim}} < \epsilon$ across registered formatting, paraphrase, repeat, and temporal perturbations). Certificates hash the exact **`OperatingEnvelopeSpec`** (`task_id`, `model_id`, `model_version`, `dataset_partition_hash`, `parser_hash`, `perturbation_env_ids`, `decoding_config`) and **fail closed** in CI whenever a provider silently rotates model weights or any boundary drifts.

---

## ✨ What's New in v2.0

| Capability | PromptPrism 1.x | PromptPrism 2.0 |
|:---|:---|:---|
| **Core Positioning** | Single-stage prompt optimizer | **Prompt Diagnostics, Token Pruning & Stability Certification** |
| **Primary Significance Test** | Pooled fixed-block RCBD ANOVA *(now exploratory only; 18.4% empirical-null FDR)* | **Contrast-specific item-clustered $t$-tests** + BH FDR ($4.0\%$–$5.1\%$ FDR) + **2,000-draw sign-flip check** |
| **Noise-Floor Audit** | Not reported | **`NoiseFloorSummary`** (`RMS_obs`, `RMS_noise`, `RMS_corr`, noise variance share `%`, small-$N$ guard) |
| **Token-Bloat Pruning** | `p >= 0.05` labeled "neutral" | **Paired TOST equivalence gate**: `KEEP`, `REMOVE_HARMFUL`, **`SAFE_TO_PRUNE`** (+ tokens saved), or `INCONCLUSIVE_NEED_MORE_ITEMS` |
| **Evidence Tiers** | Unlabeled recommendations | Explicit **`predicted`**, **`confirmation-validated`**, and **`stability-certified`** tiers (`PromptSelectionReport`) |
| **CI/CD Stability Gates** | None | **`prism diagnose`**, **`prism certify`**, and fail-closed **`prism verify-cert`** (`OperatingEnvelopeSpec` + `PromptStabilityCertificate`) |
| **Two-Stage Selection** | Single-stage only | **`ExperimentalTwoStageSelector`** (explicit `experimental_opt_in=True` only; reflects honest Real-v4 `FAIL_BOTH` gate) |

---

## 📦 Installation

```bash
# Clone and install PromptPrism 2.0
git clone https://github.com/robsanmx/prompt_prism.git
cd prompt_prism
pip install -e .

# Optional extras
pip install -e ".[viz]"       # Matplotlib Pareto & interaction charts
pip install -e ".[deepeval]"  # DeepEval >= 4.2.0 LLM-as-a-Judge metrics
```

---

## ⚡ Quickstart: Diagnose, Prune & Certify in 5 Minutes

```python
from prompt_prism import (
    Experiment,
    Factor,
    ExactMatch,
    F1Score,
    OperatingEnvelopeSpec,
    evaluate_simultaneous_perturbation_certificate,
)

# 1. Decompose candidate prompt clauses into clean binary factors
factors = [
    Factor.binary("persona", level_0_content="", level_1_content="You are an expert catalog analyst."),
    Factor.binary("few_shot", level_0_content="", level_1_content="Example: Nike Air Max -> Brand: Nike"),
    Factor.binary("cot", level_0_content="", level_1_content="Think step by step before answering."),
    Factor.binary("format_json", level_0_content="Plain text.", level_1_content="Return strict JSON object."),
    Factor.binary("guardrail", level_0_content="", level_1_content="Only use facts stated in the input."),
]

# 2. Screen orthogonally (16 runs instead of 32; kappa=1.0 for main + 2FI)
exp = Experiment.from_factors(
    factors=factors,
    design="2(5-1)V",
    system_prompt="Extract product attributes from listings.",
    data_template="Title: {{ title }}\nDescription: {{ description }}",
    metrics=[ExactMatch(), F1Score()],
    target_metric="f1_score",
    title="Product Extraction Attribution & Pruning Audit",
)

results = exp.run(dataset=screen_dataset, client=my_llm_client, max_workers=4, cache_db="prompt_cache.db")

# 3. [v2.0] Calibrated attribution, noise floor & TOST token-bloat pruning
audit = exp.diagnose(alpha=0.05, equivalence_margin=0.01, n_sign_flip=2000)
print(audit.to_markdown())

# 4. Single-stage report explicitly marks evidence_state="predicted" (never proven stable)
pred_report = exp.selection_report(strategy="screen_best")
assert pred_report.evidence_state == "predicted"

# 5. [v2.0] Validate on untouched confirmation + perturbations and issue fail-closed certificate
envelope = OperatingEnvelopeSpec(
    task_id="product_extraction",
    model_id="gemini-2.5-flash",
    model_version="gemini-2.5-flash",
    dataset_partition_hash="sha256_confirmation_split",
    parser_hash="sha256_f1_parser",
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
    epsilon_stability=0.02,
    n_bootstrap=2000,
)
certified_report = exp.selection_report(
    strategy="screen_best",
    confirmation_scores=conf_ref_scores,
    confirmation_envelope=envelope,
    stability_certificate=cert,
    request_envelope=envelope,
)
print(certified_report.to_markdown())
```

### Sample v2.0 `prism diagnose` Output

```markdown
# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`f1_score`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0100`)
- **Paired Benchmark Items (`N`):** `120` across `8` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.1683`, `RMS_noise=0.0005`, `RMS_signal_corr=0.1683` (item noise accounts for **0.0%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `320` tokens/prompt

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **A** | 0.4250 | 0.6750 | +0.2500 | 0.0005 | [+0.2490, +0.2510] | 1.2e-180 | 3.6e-180 | 0.001996 | 1 | `KEEP` | 80 |
| **B** | 0.6250 | 0.4750 | -0.1500 | 0.0005 | [-0.1510, -0.1490] | 4.8e-154 | 7.2e-154 | 0.001996 | 1 | `REMOVE_HARMFUL` | 100 |
| **C** | 0.5500 | 0.5501 | +0.0001 | 0.0005 | [-0.0009, +0.0011] | 0.8412 | 0.8412 | 0.8383 | 4.1e-34 | `SAFE_TO_PRUNE` | 320 |
```

---

## 💻 CLI v2.0 Usage (`prompt-prism` / `prism`)

```bash
# Check installed version
prism --version
# prompt-prism 2.0.0

# 1. Discover & export orthogonal fractional factorial plans
prism list-designs --factors 6 --min-resolution 4
prism alias --plan "2(6-2)IV"
prism design --factors 6 --plan "2(6-2)IV" --output design_matrix.csv
prism list-metrics

# 2. [v2.0] Calibrated attribution, noise floor & TOST token-bloat pruning audit (CSV or JSONL)
prism diagnose \
  --data screen_results.csv \
  --target f1_score \
  --sample-col sample_id \
  --prune-margin 0.01 \
  --token-costs A=80,B=100,C=320 \
  --output-report diag.md \
  --json-out diag.json

# 3. [v2.0] Generate a fail-closed stability envelope certificate (exit code 2 on FAIL/INVALID with --fail-on-error)
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

# 4. [v2.0] CI/CD release gate — exits 0 ONLY when PASS and live envelope matches 100% (exits 1 on drift)
prism verify-cert \
  --cert stability_cert.json \
  --task-id banking77_intent \
  --model-id gemini-2.5-flash \
  --model-version gemini-2.5-flash \
  --dataset-hash sha256_conf_split \
  --parser-hash sha256_exact_label \
  --envs fmt_whitespace word_paraphrase_1 temporal_wave_2

# 5. Legacy single-stage ANOVA & Pareto chart (explicitly marked evidence tier: predicted)
prism analyze --data screen_results.csv --target f1_score --strategy screen_best --output-report report.md
```

---

## 🤖 LLM-as-a-Judge & DeepEval Integration

`prompt_prism` integrates with **DeepEval** (`pip install prompt-prism[deepeval]`) for reference-based (`g_eval`, `contextual_recall`, `contextual_precision`, `json_correctness`) and reference-free (`answer_relevancy`, `faithfulness`, `hallucination`, `toxicity`, `bias`, `contextual_relevancy`, `summarization`) scoring with SQLite `JudgeCache`:

```python
from prompt_prism import Experiment, deepeval_metric, JudgeCache

judge_cache = JudgeCache(db_path="judge_cache.db")
factual_geval = deepeval_metric(
    "g_eval",
    name="factual_accuracy",
    criteria="Determine whether actual output factually agrees with expected golden target.",
    evaluation_steps=["Check key facts/numbers match target.", "Penalize fabricated claims."],
    cache=judge_cache,
)
faithfulness = deepeval_metric("faithfulness", cache=judge_cache)
```

> **🧭 Score direction and `threshold`:** PromptPrism targets **deepeval >= 4.2.0**, where every
> metric scores in the same direction — `1.0` is a pass, `0.0` is a failure, and `threshold` is
> the **minimum** passing score. `hallucination`, `toxicity` and `bias` score the *absence* of
> the problem, so higher is better for them too. deepeval <= 4.1.10 scored those three as the
> proportion of violations (lower was better, and `threshold` was a maximum); the extra is
> pinned `deepeval>=4.2.0,<5` because reading the new scores with the old direction makes the
> optimizer recommend the **more** toxic prompt. `Experiment.analyze()` and `Experiment.diagnose()`
> pick the direction from `Metric.higher_is_better` automatically; `--minimize` sets it explicitly
> for lower-is-better metrics (e.g. latency or error rate).

---

## 🛡️ Experimental Two-Stage Selection & Honest Promotion Gate

Per our preregistered Real-v4 promotion policy (requiring both Claim P and Claim S to pass), `PROMOTED_TO_DEFAULT_PUBLIC_API` is **`False`**:
- Existing `additive`, `screen_best`, and `interaction_2fi` callers remain 100% backward compatible and report `evidence_state = "predicted"`.
- Two-stage screen → multi-environment simultaneous-LCB selection is available explicitly as an **experimental opt-in** via `ExperimentalTwoStageSelector(experimental_opt_in=True)` or `exp.experimental_select_two_stage(..., experimental_opt_in=True)` without marketing a default stability assurance.

---

## 🤖 Agent Skill v2.0

PromptPrism ships with an updated **v2.0 Agent Skill** at [`.agents/skills/prompt_prism/SKILL.md`](.agents/skills/prompt_prism/SKILL.md) so AI coding assistants run calibrated attribution, noise-floor audits, TOST token pruning (`SAFE_TO_PRUNE`), and fail-closed CI certificate verification (`prism diagnose`, `prism certify`, `prism verify-cert`) instead of uncalibrated ANOVA or unvalidated additive recipes.

---

## 🏛️ Architecture

```
prompt_prism/
├── core/                  # Factor, Level, DesignMatrix, Trial, OperatingEnvelopeSpec, PromptStabilityCertificate, PromptSelectionReport
├── design/                # 2^(k-p) FractionalFactorial, Plackett-Burman, FullFactorial, AliasStructure, Recommender
├── template/              # Modular PromptComposer, Jinja2, multi-turn chat templates
├── runner/                # Parallel ExperimentRunner, SQLite ResponseCache, LLMClient / CallableLLM / MockLLM
├── evaluation/            # Deterministic metrics + DeepEval (>=4.2.0) integration & SQLite JudgeCache
├── analysis/              # v2.0 Calibrated diagnostics & TOST pruning (diagnostics.py), Envelope & two-stage selection (selection.py), Effects, ANOVA, Optimizer
├── visualization/         # Main Effects, Pareto, Interaction plots & ASCII charts
├── reporting/             # Automated Markdown & HTML reports
└── cli/                   # PromptPrism 2.0 CLI: diagnose, certify, verify-cert, analyze, design, alias, list-designs, list-metrics
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 🤝 Contributing & Community

- 📖 [Contributing Guide](CONTRIBUTING.md)
- 🛡️ [Code of Conduct](CODE_OF_CONDUCT.md)
- 🔒 [Security Policy](SECURITY.md)
- 📜 [Changelog](CHANGELOG.md)

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
