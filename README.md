# PromptPrism 2.0

> *Find out which clauses in your prompt actually help, which ones hurt, and which ones you can safely delete — without getting fooled by eval noise.*

[![CI](https://github.com/robsanmx/prompt_prism/actions/workflows/ci.yml/badge.svg)](https://github.com/robsanmx/prompt_prism/actions/workflows/ci.yml)
[![Version: 2.0.0](https://img.shields.io/badge/version-2.0.0-blue.svg)](CHANGELOG.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 244 Passed](https://img.shields.io/badge/tests-244%20passed-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PromptPrism** is a Python library and CLI for **prompt diagnostics, safe clause pruning, noise-floor auditing, and CI/CD release gating**.

Instead of tweaking one line at a time or guessing whether a 2% eval bump is real, PromptPrism toggles modular prompt clauses in small, balanced combinations, scores them on the same paired eval set, and tells you **which clauses help, which actively hurt, which are safe to delete, and when you simply need more eval items**.

---

## What PromptPrism Is — and Isn't

### Built for three production problems

1. **Find harmful instructions & prune dead prompt bloat (`prism diagnose` / `exp.diagnose()`):**
   Long production prompts accumulate personas, guardrails, formatting rules, and "think step by step" cues nobody dares touch. Even when static system prefixes are discounted by **prompt caching**, dead or conflicting clauses still cause **instruction interference / attention dilution**, **higher latency and uncached output-token drag**, **harder maintenance**, and extra cost on dynamic or uncached prefixes. PromptPrism tests every clause cleanly and labels each one `KEEP`, `REMOVE_HARMFUL`, `SAFE_TO_PRUNE`, or `INCONCLUSIVE_NEED_MORE_ITEMS`.
2. **Stop chasing eval noise:**
   Small eval sets (50–150 items) are noisy. PromptPrism explicitly measures the **item-sampling noise floor** alongside every factor estimate so your team stops shipping prompt edits based on random benchmark jitter.
3. **Gate prompt releases & monitor drift (`prism certify` & `prism verify-cert`):**
   - **`prism verify-cert` (offline CI gate, zero LLM calls):** Verifies a saved SHA-256-hashed `OperatingEnvelopeSpec` certificate against your live deployment settings (`task_id`, `model_id`, `model_version`, `dataset_partition_hash`, `parser_hash`, `decoding_config`, `perturbation_env_ids`). It fails closed whenever any **declared** boundary changes.
   - **Scheduled `prism certify` canary runs (behavioral drift check):** Providers can rotate weights silently without changing `model_version`, which a static hash check cannot detect. Re-running `prism certify` on a scheduled temporal/canary batch measures real accuracy drop across registered formatting, paraphrase, repeat, and temporal perturbations and fails when the worst-case drop exceeds your tolerance (`epsilon`).

### What PromptPrism is NOT

> **PromptPrism is not a black-box prompt optimizer.**
> Across our own 7.0M-outcome retrospective benchmarks and 501,300-call prospective study, equal-budget prompt selection (`ScreenBest`) tied random search and Bayesian optimization on raw peak accuracy (`+0.01` to `+0.3 pp`; our prospective study failed both its selection-lift and cross-group stability promotion gates).
>
> If you want automated prompt search, use DSPy/MIPRO or random/Bayesian search. Use PromptPrism when you want **interpretable clause attribution, safe pruning, calibrated eval diagnostics, and fail-closed CI release gates**.

---

## Real Output Example (`banking77` on `gemini-2.5-flash`)

Below is an **unmodified `prism diagnose` audit** from a real 1,920-call live run on `banking77_intent` (`N = 120` paired eval items × 16 balanced prompt runs with `gemini-2.5-flash`, equivalence margin `|delta| < 0.02`, measured input-token deltas):

```markdown
# PromptPrism 2.0 Calibrated Attribution & Token-Pruning Audit (`exact_match`)

- **Optimization Direction:** `maximize`
- **Primary Inference:** Contrast-specific item-clustered `t`-test + Benjamini-Hochberg FDR (`alpha=0.05`)
- **Pruning Gate:** TOST practical equivalence (`|delta| < 0.0200`)
- **Paired Benchmark Items (`N`):** `120` across `16` experimental runs
- **Main-Effect Noise Floor:** `RMS_obs=0.0238`, `RMS_noise=0.0088`, `RMS_signal_corr=0.0221` (item noise accounts for **13.7%** of observed variance)
- **Estimated Safe Token Savings (`SAFE_TO_PRUNE`):** `8` tokens/prompt

| Factor | Lvl 0 | Lvl 1 | Delta (L1-L0) | SE (Clust.) | 95% Interval | p (Clust.) | q (BH) | p (Flip) | TOST p | Verdict | Tokens |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **self_verification** | 0.9208 | 0.8979 | -0.0229 | 0.0063 | [-0.0353, -0.0105] | 0.0003867 | 0.001904 | 0.0009995 | 0.6786 | `REMOVE_HARMFUL` | 15 |
| **brief_reasoning** | 0.9271 | 0.8917 | -0.0354 | 0.0102 | [-0.0555, -0.0153] | 0.0006889 | 0.001904 | 0.0004998 | 0.9341 | `REMOVE_HARMFUL` | 12 |
| **grounding_guardrail** | 0.9229 | 0.8958 | -0.0271 | 0.0080 | [-0.0429, -0.0113] | 0.0009519 | 0.001904 | 0.001499 | 0.8114 | `REMOVE_HARMFUL` | 19 |
| **strict_output_contract** | 0.8979 | 0.9208 | +0.0229 | 0.0072 | [+0.0086, +0.0373] | 0.00198 | 0.00297 | 0.001999 | 0.656 | `KEEP` | 19 |
| **few_shot_demos** | 0.9000 | 0.9187 | +0.0187 | 0.0123 | [-0.0055, +0.0430] | 0.1287 | 0.1545 | 0.1459 | 0.4595 | `INCONCLUSIVE_NEED_MORE_ITEMS` | 63 |
| **specialist_persona** | 0.9083 | 0.9104 | +0.0021 | 0.0075 | [-0.0128, +0.0170] | 0.7828 | 0.7828 | 0.8841 | 0.009549 | `SAFE_TO_PRUNE` | 8 |
```

**What this immediately tells an engineer:**
- **3 "best-practice" clauses were actively hurting accuracy** by `-2.3` to `-3.5 pp` (`brief_reasoning`, `grounding_guardrail`, `self_verification`) — removing them improves accuracy, cuts uncached output tokens, and simplifies the prompt.
- **`specialist_persona`** is proven dead weight within `+/-2 pp` (`TOST p = 0.0095`, `SAFE_TO_PRUNE`).
- **`strict_output_contract`** genuinely helps (`+2.29 pp`, `KEEP`).
- **`few_shot_demos`** has `p = 0.13`, but its 95% interval `[-0.55 pp, +4.30 pp]` is too wide to call it dead weight — PromptPrism honestly flags it **`INCONCLUSIVE_NEED_MORE_ITEMS`** instead of letting you blindly delete a potentially helpful `+1.9 pp` block.

---

## Installation

Install directly from GitHub without cloning:

```bash
pip install git+https://github.com/robsanmx/prompt_prism.git

# Optional extras
pip install "prompt-prism[viz] @ git+https://github.com/robsanmx/prompt_prism.git"       # Matplotlib charts
pip install "prompt-prism[deepeval] @ git+https://github.com/robsanmx/prompt_prism.git"  # DeepEval >= 4.2.0 LLM judges
```

Or clone for local development:

```bash
git clone https://github.com/robsanmx/prompt_prism.git
cd prompt_prism
pip install -e ".[dev,viz]"
```

*(Automated PyPI publishing for `pip install prompt-prism` is configured via Trusted Publishing in [`.github/workflows/publish.yml`](.github/workflows/publish.yml).)*

---

## Quickstart: Diagnose, Prune & Gate in 5 Minutes

```python
from prompt_prism import (
    Experiment,
    Factor,
    ExactMatch,
    F1Score,
    OperatingEnvelopeSpec,
    evaluate_simultaneous_perturbation_certificate,
)

# 1. Define modular prompt clauses as factors
factors = [
    Factor.binary("specialist_persona", level_0_content="", level_1_content="You are an expert banking intent classifier."),
    Factor.binary("brief_reasoning", level_0_content="", level_1_content="Explain briefly before answering."),
    Factor.binary("few_shot_demos", level_0_content="", level_1_content="Example: 'lost my card' -> lost_or_stolen_card"),
    Factor.binary("strict_output_contract", level_0_content="Answer concisely.", level_1_content="Return only the exact snake_case label."),
    Factor.binary("grounding_guardrail", level_0_content="", level_1_content="Never invent unsupported labels."),
]

# 2. Run a balanced 16-run screen instead of all 32 combinations
exp = Experiment.from_factors(
    factors=factors,
    design="2(5-1)V",
    system_prompt="Classify the customer banking request.",
    data_template="Query: {{ query }}",
    metrics=[ExactMatch(), F1Score()],
    target_metric="exact_match",
    title="Banking77 Clause Attribution & Pruning Audit",
)

results = exp.run(
    dataset=screen_dataset,     # Recommend N >= 100 paired items
    client=my_llm_client,       # Any fn(prompt) -> str or LLMClient
    max_workers=4,
    cache_db="prompt_cache.db",
)

# 3. Attribute each clause, measure eval noise, and flag safe prunes
audit = exp.diagnose(alpha=0.05, equivalence_margin=0.02, n_sign_flip=2000)
print(audit.to_markdown())

# 4. Validate your pruned prompt on untouched confirmation + perturbation data and issue a certificate
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
    epsilon_stability=0.02,
    n_bootstrap=2000,
)
print(cert.status, cert.d_stability_upper_95_sim)
```

---

## CLI Usage (`prompt-prism` / `prism`)

```bash
prism --version
# prompt-prism 2.0.0

# 1. Inspect available balanced screening plans & export a design matrix
prism list-designs --factors 6 --min-resolution 4
prism alias --plan "2(6-2)IV"
prism design --factors 6 --plan "2(6-2)IV" --output design_matrix.csv
prism list-metrics

# 2. Run calibrated clause attribution, noise-floor audit & pruning verdicts (CSV or JSONL)
prism diagnose \
  --data screen_results.csv \
  --target exact_match \
  --sample-col sample_id \
  --prune-margin 0.02 \
  --token-costs specialist_persona=8,brief_reasoning=12,few_shot_demos=63,strict_output_contract=19 \
  --output-report diag.md \
  --json-out diag.json

# 3. Certify a prompt on untouched confirmation + perturbation/temporal canary data (exit 2 on FAIL/INVALID)
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

# 4. Offline CI/CD config gate — exits 0 ONLY if cert passed and declared envelope matches 100% (exits 1 on drift)
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

## Scope & Limitations

PromptPrism works best when used inside the boundaries where its assumptions actually hold:

1. **Modular clauses vs. tightly coupled prompts:**
   Binary on/off ablation assumes each clause can be toggled without making the prompt incoherent. That fits modular blocks (persona, few-shot examples, output contract, reasoning cue, standalone guardrails). For complex agent system prompts with **interdependent rules, ordering sensitivity, or coupled tool descriptions**, removing one clause can break the prompt rather than merely weaken it. In those cases, bundle coupled rules into intact alternative blocks using a multi-level `Factor(levels=[...])` instead of toggling sub-sentences independently.
2. **Single-turn scope vs. multi-turn agent trajectories:**
   All empirical validation covers single-turn tasks (classification, entity extraction, grounded QA, math, code generation). Running `N >= 100` paired items × 16 prompt designs × perturbations across end-to-end **multi-turn agent trajectories** is slow, costly, and high-variance. For agentic systems today, target **isolated single-step nodes** (intent router, tool selector, RAG groundedness/reranker, final response verifier) or replayed state snapshots where each turn is fast and paired.
3. **Declared envelope drift vs. silent provider drift:**
   `prism verify-cert` runs offline at zero LLM cost and catches **declared** changes (`model_version`, dataset/parser hash, decoding config, perturbation suite). If a model provider silently updates weights without changing `model_version`, the envelope hash still matches — catching silent behavioral drift requires periodically re-running `prism certify` on a fresh canary/temporal batch.
4. **Prompt caching vs. token cost:**
   Because modern LLM APIs heavily discount cached static system prefixes, cutting 50 tokens from a static prompt rarely moves the bill by itself. The stronger practical reasons to prune are **removing clauses that actively degrade accuracy (instruction interference / attention dilution)**, **cutting latency and uncached output tokens**, **simplifying prompt maintenance**, and saving input tokens on dynamic or uncached prompt sections.

---

## How the Statistics Work

For teams auditing the statistical methodology behind `exp.diagnose()` and `prism certify`:

- **Why orthogonal `2^(k-p)` screening?** Compared to equal-budget random subsets, orthogonal fractional factorial designs cut 6-factor main-effect attribution error by **35.38% prospectively** (winning 17/18 task–model cells) and **1.40x–4.11x retrospectively** (avoiding the 55.2% rank-deficiency rate of 16-run balanced random screens on 5-factor 2FI models).
- **Why item-clustered `t`-tests instead of pooled RCBD ANOVA?** Across 1,033 multi-prompt cubes (7,006,016 outcomes), textbook pooled fixed-block RCBD ANOVA underestimates contrast variance on high-variance formatting factors and inflates global Benjamini–Hochberg false discoveries to **18.4%**. PromptPrism computes each factor's per-item paired contrast separately (`df = N - 1`), achieving **4.0%–5.1% FDR at `N >= 100`**, corroborated by a 2,000-draw **Rademacher sign-flip randomization test** for smaller `N`.
- **Noise-floor decomposition:** Across our benchmark suite, **79%–85%** of apparent per-task higher-order variance is item-sampling noise. `audit.noise_floor` reports `RMS_obs`, `RMS_noise`, `RMS_signal_corrected`, and the percentage of observed variance attributable to item noise.
- **TOST practical-equivalence pruning:** Failing to reject `p >= 0.05` does not prove a clause is useless — it often just means `N` is too small. PromptPrism runs paired **Two One-Sided Tests (TOST)** against `equivalence_margin` so only clauses whose 90% CI lies strictly inside `[-margin, +margin]` are marked `SAFE_TO_PRUNE`; underpowered clauses are marked `INCONCLUSIVE_NEED_MORE_ITEMS`.
- **Simultaneous perturbation certificates:** `evaluate_simultaneous_perturbation_certificate()` computes a paired bootstrap upper 95% simultaneous bound (`U95_sim`) on the worst-case accuracy drop across all registered perturbation environments (`U95_sim < epsilon` required for `PASS`) and binds the result to a SHA-256 hash of `OperatingEnvelopeSpec`.

---

## LLM-as-a-Judge & DeepEval Integration

PromptPrism integrates with **DeepEval >= 4.2.0** for reference-based (`g_eval`, `contextual_recall`, `contextual_precision`, `json_correctness`) and reference-free (`answer_relevancy`, `faithfulness`, `hallucination`, `toxicity`, `bias`, `contextual_relevancy`, `summarization`) scoring with SQLite `JudgeCache`:

```python
from prompt_prism import deepeval_metric, JudgeCache

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

> **Score direction (`deepeval >= 4.2.0`):** In `deepeval >= 4.2.0`, every metric scores in the same direction (`1.0` = pass / no hallucination / no toxicity; `0.0` = fail). `deepeval <= 4.1.10` scored `hallucination`, `toxicity`, and `bias` in reverse, so the extra is pinned to `deepeval>=4.2.0,<5`. `Experiment.diagnose()` and `Experiment.analyze()` infer direction automatically from `Metric.higher_is_better`; pass `--minimize` in the CLI for lower-is-better metrics (such as latency or error rate).

---

## Agent Skill v2.0

PromptPrism ships with an Agent Skill at [`.agents/skills/prompt_prism/SKILL.md`](.agents/skills/prompt_prism/SKILL.md) so coding agents run calibrated clause attribution, noise-floor audits, TOST pruning, and fail-closed CI certificate verification instead of uncalibrated ANOVA or unvalidated prompt recipes.

---

## Legacy & Experimental APIs

- **Single-stage `analyze` / `selection_report` (1.x compatibility):** `exp.analyze(strategy="screen_best")` and `prism analyze --data screen_results.csv --target score --strategy screen_best` remain supported for backward compatibility, and `exp.selection_report(...)` explicitly labels single-stage outputs as `evidence_state = "predicted"` (never proven stable until validated on untouched confirmation and perturbation data).
- **Experimental two-stage selection:** Because our preregistered prospective promotion gate required both selection lift and cross-group stability to pass (`PROMOTED_TO_DEFAULT_PUBLIC_API = False`), two-stage simultaneous-LCB candidate selection is exposed strictly via explicit opt-in: `ExperimentalTwoStageSelector(experimental_opt_in=True)` or `exp.experimental_select_two_stage(..., experimental_opt_in=True)`.

---

## Architecture

```
prompt_prism/
├── core/                  # Factor, Level, DesignMatrix, Trial, OperatingEnvelopeSpec, PromptStabilityCertificate, PromptSelectionReport
├── design/                # 2^(k-p) FractionalFactorial, Plackett-Burman, FullFactorial, AliasStructure, Recommender
├── template/              # Modular PromptComposer, Jinja2, chat templates
├── runner/                # Parallel ExperimentRunner, SQLite ResponseCache, LLMClient / CallableLLM / MockLLM
├── evaluation/            # Deterministic metrics + DeepEval (>=4.2.0) integration & SQLite JudgeCache
├── analysis/              # Calibrated diagnostics & TOST pruning (diagnostics.py), Envelope & two-stage selection (selection.py), Effects, ANOVA, Optimizer
├── visualization/         # Main Effects, Pareto, Interaction plots & ASCII charts
├── reporting/             # Automated Markdown & HTML reports
└── cli/                   # CLI: diagnose, certify, verify-cert, analyze, design, alias, list-designs, list-metrics
```

---

## Running Tests

```bash
pytest
```

---

## Contributing & License

- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
