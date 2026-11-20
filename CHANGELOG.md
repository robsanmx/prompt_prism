# Changelog

All notable changes to **PromptPrism** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-11-20

### Added
- **Calibrated Attribution, Noise-Floor & TOST Prompt Pruning (`prompt_prism/analysis/diagnostics.py`, `Experiment.diagnose`, `prism diagnose`)**:
  - Contrast-specific item-clustered `t`-tests with Benjamini–Hochberg FDR (`df = N - 1`, conservative zero-variance guard) and optional 2,000-draw Rademacher sign-flip randomization for small item counts (`N < 100`).
  - Main-effect item-sampling noise-floor decomposition (`RMS_obs`, `RMS_noise`, `RMS_signal_corrected`, noise variance share `%`).
  - Paired Two One-Sided Tests (TOST) pruning verdicts (`KEEP`, `REMOVE_HARMFUL`, `SAFE_TO_PRUNE`, `INCONCLUSIVE_NEED_MORE_ITEMS`) with automatic/custom token-savings estimation.
- **Explicit Evidence Tiers & Fail-Closed Envelope Stability Certificates (`prompt_prism/core/models.py`, `prompt_prism/analysis/selection.py`, `Experiment.selection_report`, `prism certify`, `prism verify-cert`)**:
  - Public reports explicitly separate `predicted`, `confirmation-validated`, and `stability-certified` states.
  - `OperatingEnvelopeSpec` and `PromptStabilityCertificate` hash pinned task, model, model-version, dataset, parser, decoding, algorithm, and perturbation boundaries and fail closed (exit code `1` in `prism verify-cert`) whenever any declared boundary differs.
  - `ExperimentalTwoStageSelector` and `Experiment.experimental_select_two_stage` require explicit `experimental_opt_in=True` reflecting the preregistered Real-v4 promotion gate verdict (`FAIL_BOTH`).
- **Prospective Real-v4 Study (`research/real_v4/`)**:
  - Completed 501,300-call cryptographically phase-locked prospective study across 6 single-turn tasks and 3 model families (`gemini-2.5-flash`, `Mistral-7B-Instruct-v0.3-4bit`, `Qwen3-8B-4bit`).
- **CLI v2.0 (`prompt-prism` / `prism`) & Agent Skill v2.0 (`.agents/skills/prompt_prism/SKILL.md`)**:
  - Added `--version`, `diagnose`, `certify`, `verify-cert`, and `--strategy` / `--include-interactions` flags on `analyze` with explicit `predicted` evidence-tier notice.
  - Realigned package positioning, README, and Agent Skill around Prompt Diagnostics, Safe Clause Pruning, Calibrated Attribution, and CI/CD Release Gating.

### Changed
- Bumped package version in `pyproject.toml` and `prompt_prism/__init__.py` to `2.0.0`.
- Re-positioned pooled fixed-block RCBD ANOVA as exploratory only (citing the 18.4% global BH empirical-null FDR finding); contrast-specific item-clustered `t`-tests are now the primary v2.0 diagnostic inference method.

---

## [1.2.0] - 2026-11-06

### Added
- **Multi-Model Hugging Face Factorial Benchmark Study (`paper_hf/` & `research/hf_multi_prompt/`)**:
  - Large-scale evaluation across **1,033 complete factorial cubes** ($7{,}024{,}960$ cell–item evaluations) combining `PromptEval/PromptEval_MMLU_full` ($855$ $2^5=32$ cubes across 15 LLMs and 57 MMLU subjects) and the `SLAB-NLP` multi-prompt benchmark suite ($178$ $2^4=16$ cubes across `BBH` and `LMentry`).
  - Standalone manuscript (`paper_hf/main.tex`, `paper_hf/main.pdf`), publication figures (`paper_hf/figures/`), LaTeX tables (`paper_hf/tables/`), and reproducible runner (`research/hf_multi_prompt/run_experiments.py`).
- **Epistasis-Aware & Screen-Anchored Prompt Selection (`OptimalPromptFinder`)**:
  - Added `strategy` support (`"additive"`, `"screen_best"`, `"interaction_2fi"`, and `"auto"`) to `OptimalPromptFinder.find_optimal_prompt`, `AnalysisReport.generate`, and `Experiment.analyze` to resolve the Sign-Composition Paradox when prompt factors interact epistatically.
  - `ANOVAEngine.run_anova` now records observed cell means (`cell_means`) and grand mean in `ANOVAResult.metadata`.
  - Fixed marginal main-effect extraction in `EffectAnalyzer.compute_main_effects` when interaction terms (`include_interactions=True`) are included in Patsy treatment-coded OLS models.

---

## [1.1.0] - 2026-09-26

### Added
- **Scientific Validation Package & Empirical Suite**:
  - Full implementation of preregistered empirical protocol (`research/protocol/`) covering Study A (Monte Carlo simulations across 8 regimes), Study B (7,680 confirmatory evaluations across 3 tasks), Study C (prospective holdout validation), and Study D (judge validation against blinded human ratings).
  - Synthetic Data Generating Process (`research/simulation/dgp.py`) with continuous, bounded, and binary outcome generators, repeated-measures item variance, interaction density, and judge noise.
  - Monte Carlo simulation engine and calibration reporting (`research/simulation/run.py`, `summarize_simulation.py`).
  - Content-addressed execution runner and manifest tracking (`research/experiments/manifest.py`, `run.py`).
  - Two-One-Sided-Tests (TOST) equivalence testing against Smallest Effect Size of Interest (SESOI $\delta \in \{0.020, 0.025, 0.030\}$).
  - Human vs. LLM judge agreement validation (`research/analysis/judge_validation.py`) reporting Spearman $\rho$, Pearson $r$, Bland-Altman limits of agreement, and systematic bias.
  - Reproducibility pipeline script (`research/reproduce.py`) executing complete offline verification in $\sim 6$ seconds with zero paid API calls.
  - Research artifact manifest (`research/artifact_manifest.json`) cataloging cryptographic SHA-256 digests and file sizes for all research artifacts.
- **Preprint Manuscript & Documentation**:
  - Scientific paper (*"Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization"*) in `paper/main.tex` with verified `paper/references.bib`, `paper/tables/results.tex`, and publication figures in `paper/figures/`.
  - Independent statistical review (`docs/research/statistical-review.md`) and fresh-context adversarial review (`docs/research/adversarial-review.md`).
  - Venue submission checklist (`paper/venue_checklist.md`).
  - Diagnostic failure mode reports on aliased collisions and antagonistic cliff regimes (`docs/research/failure-mode-report.md`, `calibration-remediation.md`).

---

## [1.0.0] - 2026-09-01

### Added
- **Fractional Factorial Catalog**:
  - 28 standard Box-Hunter / Montgomery orthogonal $2^{k-p}$ experimental designs from $2^{3-1}_{\text{III}}$ (4 runs) up to $2^{15-11}_{\text{III}}$ (16 runs).
  - Plackett-Burman screening designs for $k \le 23$ factors in $N \in \{8, 12, 16, 20, 24\}$ runs.
  - Automatic defining relation resolution ($\text{Res III}, \text{IV}, \text{V}, \text{VI}, \text{VII}, \text{VIII}$) and alias matrix calculation.
- **Statistical ANOVA & RCBD Engine**:
  - Randomized Complete Block Design (RCBD) blocking on `sample_id` to eliminate benchmark variance.
  - Main effects and 2-factor interaction calculation ($\Delta = \bar{Y}_+ - \bar{Y}_-$).
  - OLS regression with Type I & Type II Sums of Squares, $F$-statistic, and $p$-values.
  - Effect size measures: Partial $\eta^2$, Generalized $\eta^2$, $\omega^2$, and Cohen's $d$.
  - Multiple hypothesis testing adjustments (Bonferroni and Benjamini-Hochberg FDR).
- **Prompt Composition & Modular Template Engine**:
  - Modular section composer with conditional toggling and ordering.
  - Jinja2 and Python string template support.
  - Multi-turn chat message generation.
- **Universal LLM Adapter & Execution Engine**:
  - Direct integration with any Python callable (`fn(prompt) -> str`).
  - Thread-safe multi-threaded trial runner with rate limiting and exponential backoff.
  - In-memory and persistent SQLite response caching (`ResponseCache`).
- **Evaluation & DeepEval LLM-as-a-Judge**:
  - Deterministic metrics: `ExactMatch`, `F1Score`, `JSONValidation`, `KeyValuesExtractionOverlap`, `LevenshteinSimilarity`, `RegexMatch`, and custom metric functions.
  - DeepEval LLM judge metrics (`AnswerRelevancy`, `Faithfulness`, `Hallucination`, `Toxicity`, `Bias`, `ContextualPrecision`, `ContextualRecall`, `Summarization`, `GEval`, `JSONCorrectness`) with `JudgeCache`.
- **Optimal Prompt Finder & Recommendations**:
  - Automated classification of factors into Positive Boosters, Harmful Drops, and Neutral Token-Bloat.
  - Statistically optimal prompt generation.
- **Visualization & Reporting**:
  - Main effects plots, Pareto charts of standardized effects, and interaction matrix plots.
  - ASCII terminal charts for CLI and headless environments.
  - Markdown, HTML, and JSON report exporters.
- **CLI & Agent Skill**:
  - `prompt-prism` (and `prism`) command-line interface with `list-designs`, `design`, `analyze`, `list-metrics`, and `inspect-aliases`.
  - Built-in Antigravity Agent Skill (`.agents/skills/prompt_prism/SKILL.md`) for automated prompt self-healing loops.
