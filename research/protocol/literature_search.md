# Literature Review Protocol & Search Log

**Owner:** Roberto Sanchez (`@robsanmx`), Methodology Lead  
**Execution Date:** 2026-09-26  
**Status:** Completed & Verified  

---

## 1. Search Objectives & Scope

To support the validation of PromptPrism and establish an accurate, non-hyperbolic scientific context, this systematic search maps all related literature across:
1. Fractional factorial designs and Design of Experiments (DoE) applied to prompt engineering.
2. Component ablation and prompt decomposition studies.
3. Automatic prompt optimization and best-arm identification.
4. Statistically rigorous LLM evaluation (blocking, mixed-effects models, repeated measures, and power analysis).
5. LLM-as-a-judge calibration, agreement, and reliability.

---

## 2. Databases & Search Engines

The search was conducted across the following indexing platforms:
* **arXiv** (cs.CL, cs.AI, stat.ME, stat.AP)
* **ACL Anthology** (ACL, EMNLP, NAACL, EACL)
* **IEEE Xplore**
* **ACM Digital Library**
* **Semantic Scholar**
* **Crossref**
* **Google Scholar**

---

## 3. Query Families & Syntax

| Family | Target Concept | Query String |
|---|---|---|
| **QF-1** | Prompt DoE & Factorial Screening | `("prompt engineering" OR "prompt optimization") AND ("factorial design" OR "fractional factorial" OR "design of experiments" OR "ANOVA")` |
| **QF-2** | Component Ablation & Interactions | `("prompt optimization" OR "prompt engineering") AND ("component ablation" OR "prompt interaction" OR "screening design")` |
| **QF-3** | Statistical LLM Evaluation | `("LLM evaluation" OR "benchmark evaluation") AND ("mixed effects" OR "GLMM" OR "blocking" OR "paired evaluation" OR "power analysis")` |
| **QF-4** | Prompt Selection & Bandits | `"prompt optimization" AND ("best arm identification" OR "multi-armed bandit" OR "Bayesian optimization")` |
| **QF-5** | Order of Addition & Permutation | `("prompt engineering" OR "in-context learning") AND ("order of addition" OR "permutation sensitivity" OR "prompt order")` |

---

## 4. Eligibility Criteria

### Inclusion Criteria (IC):
* **IC-1:** Proposes or evaluates a method for decomposing, ablating, or optimizing prompt components (e.g. instructions, few-shot examples, personas, constraints).
* **IC-2:** Investigates experimental design, orthogonal arrays, ANOVA, mixed-effects models, or formal hypothesis testing in prompt optimization or LLM benchmarking.
* **IC-3:** Formalizes uncertainty estimation (confidence intervals, bootstrap clustering, power analysis) for LLM evaluation.
* **IC-4:** Peer-reviewed conference/journal papers or established preprint releases with reproducible code/benchmarks.

### Exclusion Criteria (EC):
* **EC-1:** Papers proposing prompt templates without empirical evaluation or comparison.
* **EC-2:** Pure continuous prompt tuning (soft prompt embeddings) unrelated to discrete, interpretable prompt components.
* **EC-3:** Purely theoretical DoE papers with no computational or natural language processing application (except classical foundations: Box et al., Montgomery).
* **EC-4:** Unreviewed opinion pieces or marketing whitepapers lacking formal mathematical or experimental specifications.

---

## 5. Methodological Taxonomy

To prevent overclaiming novelty, related works are classified into six distinct methodological paradigms:

1. **Full-Prompt Black-Box Selection**: Treats candidate prompts as atomic, indivisible arms (e.g. TRIPLE multi-armed bandits, Bayesian prompt optimization). Does not decompose prompts into constituent components.
2. **Evolutionary & Gradient Prompt Generation**: Iteratively generates or edits prompt text via reflection or local search (e.g. DSPy, AutoPrompt, EvoPrompt, PE2). Focuses on text generation rather than factor effect estimation.
3. **One-Factor-At-A-Time (OFAT) Ablation**: Removes or toggles a single component relative to a fixed baseline (e.g. Venkatappa 2026). Conflates interaction terms and does not evaluate orthogonal configurations.
4. **Order-of-Addition (OofA) Permutation Testing**: Studies the sequential ordering of fixed prompt elements (e.g. Ying et al. 2026). Optimizes component order rather than binary presence/absence.
5. **Statistical Benchmark Modeling**: Models item-level difficulty and model capability using GLMMs or paired resampling (e.g. NIST AI 800-3, Keller et al. 2026; Spark-LLM-Eval). Evaluates fixed models/prompts rather than optimizing factorial recipes.
6. **Factorial Component Screening (PromptPrism)**: Applies two-level fractional factorial designs ($2^{k-p}$), explicit alias analysis, RCBD item-level blocking, and ANOVA/equivalence testing to identify significant components and construct winning prompts.

---

## 6. Forward & Backward Citation Tracing

For key focal papers:
* Forward-chained citations from Vazquez et al. (2025, 2026) and Ying et al. (2026).
* Backward-chained foundational references in multicomponent interventions (Collins et al., MOST framework).
* Traced benchmark variance and sample-size studies (Anthropic 2024; Databricks p1 2026; NIST 2026).

All screened candidates and decisions are recorded in [`research/protocol/literature_screening.csv`](literature_screening.csv).
