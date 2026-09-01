# Changelog

All notable changes to **PromptPrism** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
