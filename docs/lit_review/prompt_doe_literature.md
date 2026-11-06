# Literature Review: Experimental Design & Statistical Testing for LLM Prompt Optimization

**Date:** 2026-09-26  
**Scope:** Papers that apply formal statistics (factorial designs, ANOVA, regression, bandits, power analysis, confidence intervals) to prompt engineering, prompt optimization, or LLM evaluation.

---

## 1. PromptPrism's Core Thesis

PromptPrism applies **classical Design of Experiments (DoE)** to a concrete problem:

> *A prompt is a multi-factor process. Each component (persona, few-shot examples, chain-of-thought, JSON format, guardrails) can be toggled on or off. A fractional factorial design (`2^(k-p)`) lets you screen many components in a small number of runs. ANOVA + RCBD (blocking on benchmark items) lets you separate true factor effects from noise and item difficulty.*

Key mathematical ingredients:
- **Fractional factorial designs** (`2^(k-p)`) to reduce `2^k` to 16 runs while keeping main effects clean (Resolution III/IV/V)
- **XOR generators** (`E = A⊕B⊕C⊕D`) to construct the design matrix
- **Alias structure** (`I = ABCDE`) to know exactly which higher-order effects are aliased with which main effects
- **RCBD (Randomized Complete Block Design)** on `sample_id` to remove item-to-item variance from the residual
- **ANOVA** (Type I/II sums of squares, F-statistics, p-values) to test significance
- **Effect sizes** (partial η², ω², Cohen's d) to quantify magnitude
- **Multiple comparisons corrections** (Bonferroni, Benjamini-Hochberg FDR)
- **Automatic optimal prompt generation** (combine all significant boosters, drop all neutral/harmful)

---

## 2. Related Work — Categorized by Method

### 2.1 Order-of-Addition (OofA) Experiments — Vazquez et al. (2026)

**Title:** *"Prompt engineering using order-of-addition experiments: An application to generating two-level fractional factorial designs"*
**Authors:** Duoduo Danny Ying, Alan R. Vazquez, Hongquan Xu (UCLA Stat / Tec de Monterrey)
**Date:** July 2026 (arXiv: 2607.05537)

**What they do:**
- Focus on **order dependency** — the observation that the *position* of prompt elements (shots, instructions, answers) affects LLM performance
- Use **Order-of-Addition (OofA) designs**, a specialized variant of factorial designs for ordering problems
- Fit a **logistic regression model** to success rates across designs; use step-wise regression (BIC) to get a reduced model with interaction terms; report McFadden's pseudo R²
- Demonstrate on 3 experiments with gpt-4.1 and gemini-2.5-flash, optimizing the success rate of constructing fractional factorial designs

**How it differs from PromptPrism:**
- They study **order/permutation** (123 vs 321 vs 213) of prompt elements, not **binary presence/absence** (is the persona on or off?)
- They use a logistic regression on *success rates* (a Bernoulli outcome), not a continuous score
- They don't do RCBD on benchmark items or blocking — the items are the design tests themselves
- They use OofA designs (a niche design class from chemical engineering) rather than standard fractional factorial designs
- The *response* (success rate of design construction) is a meta-task — they are using LLMs to *build* factorial designs, not using factorial designs to *optimize* the LLM's output

**Bottom line:** Same statistical pedigree (factorial designs), different application (ordering vs. presence/absence), different design class (OofA vs. standard fractional factorial), different response (success rate vs. arbitrary metric). **Complementary, not competitive.**

---

### 2.2 LLMs for Constructing Fractional Factorial Designs — Vazquez et al. (2025)

**Title:** *"A systematic assessment of Large Language Models for constructing two-level fractional factorial designs"*
**Authors:** Alan R. Vazquez et al. (Tec de Monterrey / UCLA)
**Date:** Dec 2025 (arXiv: 2512.17113)

**What they do:**
- Test whether GPT-5.1 and Gemini 2.5-Flash can *construct* fractional factorial designs (8-, 16-, 32-run) with maximum resolution and minimum aberration
- Use various prompting techniques (role prompting, context, zero-shot CoT, shots)
- Compare LLM-constructed designs against the best-known designs in the statistical literature

**Bottom line:** This paper is about **using LLMs as tools to create designs**, not about using DoE to optimize prompts. **Not directly related**, but the authors' statistical expertise shows that applying DoE concepts to LLMs is a natural extension of their research program.

---

### 2.3 Systematic Prompt Ablation — Venkatappa (2026)

**Title:** *"What Does a Prompt Actually Do? A Systematic Ablation Study of Prompt Components in LLM Code Generation"*
**Author:** Rukmangadha Peddapalli Venkatappa (Digit88 Technologies)
**Venue:** IEEE SSE 2026 (July 2026)

**What they do:**
- **6 semantic components** independently removed across **7 prompt variants** → 42 experiment runs (3 tasks × 7 variants)
- Scored on 5 features each using LLM-as-judge + manual code inspection
- Report: 95% confidence intervals, Cohen's κ = 0.52 (moderate agreement), output stability (Wilson CI)
- 5-type taxonomy of prompt-to-code relationships

**How it differs from PromptPrism:**
- They remove **one component at a time** (classical ablation). This is **not a factorial design**: the 7 variants are {all 6, all 5, all 4, all 3, all 2, all 1, none}. These are *not* orthogonal. Each variant's score conflates multiple factor effects.
- **No factorial structure** (no ±1 coding, no orthogonality, no alias analysis, no design matrix)
- **No RCBD** on items (no blocking on sample_id)
- **No ANOVA** (they report point estimates and CIs, not F-tests or p-values for individual factors)
- They *do* use Cohen's κ and Wilson CIs — useful for agreement/stability, but orthogonal to the main-effect estimation question

**Bottom line:** A well-executed **observational** study, but it doesn't isolate individual factor effects the way a factorial design does. PromptPrism gives you the same data (component importance) in far fewer runs, with formal significance tests, and an automatically generated optimal prompt.

---

### 2.4 Multi-Armed Bandits for Prompt Selection — TRIPLE (Shen et al., 2024)

**Title:** *"Efficient Prompt Optimization Through the Lens of Best Arm Identification"*
**Authors:** Chengshuai Shi, Kun Yang, Zihan Chen, Jundong Li, Jing Yang, Cong Shen
**Venue:** NeurIPS 2024 (arXiv: 2402.09723)

**What they do:**
- Treat a pool of candidate prompts as **arms in a multi-armed bandit (MAB)**
- Use fixed-budget best arm identification (BAI-FB) algorithms (TRIPLE-SH, TRIPLE-CR) to efficiently allocate a limited budget to test prompts
- Propose a novel connection: **prompt optimization = bandit selection problem**
- Also explore embedding-based acceleration (NeuralUCB, BO-EI)

**How it differs from PromptPrism:**
- **Does not test components within a prompt.** It tests *entire* candidate prompts (e.g., "Prompt A" vs "Prompt B") as indivisible units
- **No factor decomposition** — you never learn that "persona alone adds +0.12" because they never decompose prompts into factors
- **Bandit algorithms** are designed to find a single "best" prompt quickly; they don't produce effect-size estimates, alias maps, or ANOVA tables
- No RCBD, no blocking, no multiple-comparison correction

**Bottom line:** A fundamentally **different problem** (select from existing prompts vs. discover which components matter). TRIPLE is great when you have 100 candidate prompts and want to find the best one quickly. PromptPrism is what you use when you want to *build* the prompts from components and understand *why*.

---

### 2.5 Prompt Variance & Learnability — p1 (Databricks, 2026)

**Title:** *"p1: Better Prompt Optimization with Fewer Prompts"*
**Authors:** Databricks AI Research (D. Agrawal et al.)
**Date:** April 2026 (arXiv: 2604.08801)

**What they do:**
- Systematically study the **variance of prompt rewards** on IFBench and AIME
- Show that when system prompts look similar in expected reward, they become indistinguishable — diluting the signal for automatic optimization
- Ablate K (number of top prompts) and M (number of evaluation items per prompt) to find optimal trade-offs
- Show that using a subset of training examples (S) can yield cleaner signals

**Bottom line:** **Highly relevant** as a validation of PromptPrism's key insight about variance, but they don't use DoE — they study how much data you need. Their conclusion that "more evaluation items reduce variance" aligns with (and motivates) PromptPrism's RCBD approach.

---

### 2.6 Statistical Framework for LLM Evaluations — Anthropic (2024)

**Title:** *"A Statistical Approach to LLM Evaluations"* (referenced throughout the literature)
**Institution:** Anthropic (cited in Cameron Wolfe's Substack series, Spark-LLM-Eval, NIST.AI.800-3)

**Key contributions:**
- Treats LLM evaluation as an **experiment** (not just a point comparison)
- Recommends **paired evaluations** (compute the difference score per item, test the mean difference)
- Derives a **sample size formula** (power analysis) for evaluating LLMs
- Reports **confidence intervals**, not just point estimates

**Bottom line:** This is the **methodological foundation** for the entire field. PromptPrism inherits these principles: paired evaluation (per sample_id), confidence in effects (via p-values), sample size awareness (RCBD reduces required sample size).

---

### 2.7 Spark-LLM-Eval (2026)

**Title:** *"Spark-LLM-Eval: A Distributed Framework for Statistically Rigorous Large Language Model Evaluation"*
**Authors:** (Various, likely at an AI research org)
**Date:** March 2026 (arXiv: 2603.28769)

**What they do:**
- Distributed evaluation framework (Apache Spark) with **statistical rigor built in**
- Every reported metric includes **bootstrap confidence intervals**
- Model comparisons use **paired t-tests, McNemar's test, Wilcoxon signed-rank** (depending on metric type)
- **Response caching** (content-addressable hashing via Delta Lake) for cost-efficient iteration

**How it differs from PromptPrism:**
- It's an **evaluation framework**, not an optimization/DoE framework
- It compares **complete prompts** against each other (A vs B), not **components within a prompt** (A_on vs A_off)
- No factorial design, no component decomposition, no RCBD
- Uses bootstrap CIs (good for complex metrics) instead of parametric ANOVA (good for factorial structures)

**Bottom balance:** Complementary infrastructure for evaluation, but doesn't address *how to test components* rather than *how to compare complete prompts*.

---

### 2.8 NIST — GLMMs for LLM Evaluation (2026)

**Title:** *"Expanding the AI Evaluation Toolbox with Statistical Models"* (NIST.AI.800-3)
**Date:** Feb 2026

**What they do:**
- Apply **Generalized Linear Mixed Models (GLMMs)** to LLM benchmark evaluations
- Decompose total variance into **inter- and intra-item components**
- Report accuracy estimates with 95% CIs on GPQA-Diamond
- GLMMs use data across *all* tested LMs for more efficient uncertainty quantification

**How it differs from PromptPrism:**
- It's a **modeling framework** for estimating LLM accuracy with proper uncertainty — not a design-of-experiments method
- The goal is to report **more precise point estimates and CIs**, not to test *whether* a component matters
- Uses random effects (item difficulty as a random effect, not a fixed block) — different philosophy from RCBD

**Bottom balance:** GLMMs and RCBD both handle item difficulty, but GLMMs focus on *uncertainty quantification* while RCBD focuses on *factor isolation*. They could be combined, but GLMMs don't replace factorial screening.

---

### 2.9 Biostatistics — Factorial Designs for Multicomponent Interventions (2020)

**Title:** *"Developing Multicomponent Interventions using Fractional Factorial Designs"* (PMC2746448)
**Authors:** (Biostatistics literature, e.g., Collins et al.)
**Date:** 2020

**What they do:**
- Use full and fractional factorial designs to **screen out inactive components** of a behavioral intervention
- Objective: *evaluate* combined medication vs. evaluate *which components matter*
- Use regression approach (can handle unbalanced data) and logistic regression for binary outcomes
- Follow-up studies fine-tune levels of significant components

**How it differs from PromptPrism:**
- Same statistical methodology (factorial designs), but in a **different domain** (clinical trials, not LLMs)
- Same conceptual framework: *screen components → keep significant ones → fine-tune levels*
- PromptPrism's approach is the **exact translation** of this methodology into the LLM prompt domain

---

### 2.10 "A Survey of Automatic Prompt Engineering: An Optimization Perspective" (2026)

**Title:** (Li et al., 2026, arXiv: 2502.11560)
**Authors:** Wenwu Li, Xiangfeng Wang, Wenhao Li, Bo Jin

**What they do:**
- Comprehensive survey covering: FM-based optimization (PE2, AutoPrompt, GrIPS, GEPA, EvoPrompt), evolutionary computing, gradient-based (HPME, AutoPrompt, ZOPO), RL (TEMPERA, PACE), and multi-objective strategies
- Organizes methods by optimization space (discrete, continuous, hybrid) and optimization strategy

**What's missing from the survey (relative to PromptPrism):**
- **No section on factorial designs / DoE / ANOVA for prompt component screening**
- No mention of RCBD or blocking
- No mention of alias maps, resolution, or design matrices
- The survey covers "how to *find* a good prompt" (via search, evolution, gradient, RL) but not "how to *analyze* what components matter" (via DoE + ANOVA)

---

## 3. Comparative Matrix

| Paper | Method | What They Test | Factor Decomposition? | RCBD/Blocking? | Multiple Testing? | Effect Sizes? |
|---|---|---|---|---|---|---|
| **PromptPrism** | Fractional factorial + ANOVA + RCBD | **Component presence** (on/off) | ✅ Yes — main effects + interactions | ✅ Sample_id | ✅ Bonferroni, BH-FDR | ✅ η², ω², d |
| Vazquez et al. 2026 (OofA) | Order-of-addition + logistic regression | **Component ordering** (permutation) | ✅ Yes — but ordering, not presence | ❌ | ❌ (stepwise regression) | R² (pseudo) |
| Vazquez et al. 2025 | Prompt assessment (LLMs construct designs) | LLM ability to *build* designs | N/A | N/A | N/A | N/A |
| Venkatappa 2026 | Ablation (remove 1 at a time) | **Component contribution** | ❌ No — conflated (ablation) | ❌ | ❌ | Cohen's κ (agreement) |
| TRIPLE (NeurIPS 2024) | Multi-armed bandits (BAI-FB) | **Full prompt selection** | ❌ No — indivisible prompts | ❌ | ❌ | Reward (point estimates) |
| p1 (Databricks 2026) | Variance analysis (ablation of K, M) | Prompt variance & learnability | ❌ (no) | ❌ | ❌ | Reward variance |
| Spark-LLM-Eval 2026 | Bootstrap CIs + significance tests | **Full prompt comparison** | ❌ No | ❌ | ❌ (paired tests) | Accuracy (with CIs) |
| NIST.AI.800-3 2026 | GLMMs for benchmarking | LLM accuracy with uncertainty | ❌ No | ❌ (random effects) | ❌ | Accuracy + 95% CIs |
| Biostatistics 2020 | Fractional factorial (clinical trials) | Intervention components | ✅ Yes | Partial (baseline covariates) | ❌ | Regression coefficients |

---

## 4. Methodological Distinction and Specific Contribution

### 4.1 Comparison Across Prompt Optimization Paradigms

Existing literature in prompt optimization predominantly falls into:
- **Black-box selection** (e.g. TRIPLE, Shen et al., 2024), which identifies top-performing whole prompts but treats prompt text as an atomic unit without factor decomposition.
- **Iterative prompt generation** (e.g. DSPy, Khattab et al., 2024; APE, Zhou et al., 2023), which optimizes text prompts via heuristics, LLM reflection, or reinforcement learning without estimating component-level effects.
- **Order-of-addition experiments** (e.g. Ying, Vazquez, & Xu, 2026), which optimize the sequence/permutation of fixed components rather than screening binary component inclusion.
- **One-factor-at-a-time (OFAT) ablations** (e.g. Venkatappa, 2026), which evaluate non-orthogonal sequential removals and cannot isolate aliased interaction structures.

PromptPrism focuses specifically on **orthogonal two-level fractional factorial screening ($2^{k-p}$)** of discrete semantic prompt components, combined with **RCBD blocking on benchmark items** to isolate item difficulty.

### 4.2 Key Features and Scientific Scope

| Capability | Methodological Implementation | Comparison to Existing Work |
|---|---|---|
| **Orthogonal Component Screening** | Regular two-level fractional designs ($2^{k-p}$) and Plackett-Burman designs | Replaces non-orthogonal OFAT ablations with balanced, orthogonal factor contrast matrices |
| **Repeated-Measures Item Blocking** | RCBD blocking on `sample_id` | Partitions item-difficulty variance from residual error, extending principles from Anthropic (2024) and NIST (2026) |
| **Explicit Alias & Resolution Tracking** | Closed-form defining relations and word-length resolution (Res III, IV, V) | Explicitly reports confounding aliases rather than assuming factor independence |
| **Equivalence-Gated Selection** | Two One-Sided Tests (TOST) / $[-\delta, +\delta]$ interval testing for factor neutrality | Avoids the common fallacy of treating $p > 0.05$ as evidence of factor neutrality |
| **Multiplicity Correction** | Benjamini-Hochberg FDR and Bonferroni adjustments | Controls false discovery rates across simultaneous factor screening tests |
| **Effect Size Quantification** | Partial $\eta^2$, $\omega^2$, and standardized Cohen's $d$ | Quantifies practical magnitude alongside statistical significance |

---

## 5. Methodological Positioning for the Paper

### 5.1 Grounding in Classical Experimental Design

Fractional factorial designs, ANOVA, and RCBD are well-established statistical foundations (Box et al., 2005; Montgomery, 2017). Adapting these tools to prompt optimization requires formalizing prompt components as experimental factors and benchmark items as repeated-measures clustering units.

### 5.2 Comparative Positioning

| Paradigm | Primary Objective | Key Distinction from Factorial Screening |
|---|---|---|
| **TRIPLE (NeurIPS 2024)** | Bandit selection of whole prompts | Identifies best arm among existing candidate prompts; does not decompose prompts into constituent components |
| **Ying et al. (arXiv 2026)** | Order-of-addition permutation | Analyzes sequential ordering of components; complementary to binary presence/absence screening |
| **Venkatappa (IEEE SSE 2026)** | OFAT component ablation | Sequential one-at-a-time removal conflates interaction terms across non-orthogonal configurations |
| **NIST AI 800-3 (2026)** | GLMM benchmark modeling | Models benchmark variance and model ability; does not screen factorial prompt recipes |

### 5.3 Core Thesis

> While heuristic and search-based prompt engineering focuses on finding *an* effective prompt, factorial experimental design provides a structured framework to estimate *which* components reliably contribute to performance, quantify their individual and interaction effects, and measure how much held-out performance is preserved when using a fractional design.

---

## 6. Papers to Cite in the Article

| Citation | Why Cite |
|---|---|
| **Vazquez et al. 2026** (OofA) | Nearest conceptual neighbor; acknowledges the DoE lineage; contrasts OofA (ordering) vs. factorial (presence) |
| **Vazquez et al. 2025** | Shows the DoE ↔ LLM connection predates this work; establishes statistical credibility |
| **TRIPLE (NeurIPS 2024)** | The leading bandit-based approach; explicitly contrast "select a prompt" vs. "analyze prompt components" |
| **Anthropic Statistical Approach (2024)** | The foundational paper that treats LLM evaluation as an experiment; motivates RCBD |
| **Venkatappa 2026** (Ablation) | A well-executed alternative approach; contrast ablation (non-factorial) vs. factorial design |
| **Collins et al. (biostatistics 2020)** | The canonical source for factorial screening of multicomponent interventions — directly analogous to prompt component screening |
| **Spark-LLM-Eval 2026** | Shows the community is converging on statistical rigor for LLM evaluation; PromptPrism provides the *design* layer on top |
