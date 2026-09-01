# 💎 PromptPrism: Statistical Prompt Optimization & Factorial Analysis for LLMs

> *Separate your prompt into its true factors — like light through a prism.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Coverage: 82%+](https://img.shields.io/badge/coverage-82%25-green.svg)]()

**`prompt_prism`** is a universal, statistically rigorous Python library that applies **Fractional Factorial Design of Experiments (DoE)** and **Analysis of Variance (ANOVA)** to isolate, quantify, and optimize LLM prompts with 90%+ cost reduction and zero guesswork.

---

## 💡 Why PromptPrism?

In Prompt Engineering, a prompt is composed of multiple candidate factors:
- 🎭 **Persona / Role** (e.g. Expert vs None)
- 📝 **Task Clarity & Formatting** (e.g. Markdown vs JSON Schema)
- 💡 **Few-Shot Examples** (e.g. 0 vs 3 examples)
- 🧠 **Reasoning Style** (e.g. Standard vs Chain-of-Thought)
- 🔒 **Negative Constraints & Guardrails**
- 🌡️ **Sampling Hyperparameters** (e.g. Temperature $0.1$ vs $0.7$)

### The Exponential Cost Problem
Testing $k = 10$ binary prompt factors via Full Factorial requires **$2^{10} = 1,024$ prompt variants**. Across a 50-item benchmark dataset, that means **$51,200$ LLM calls** costing hundreds of dollars and hours of latency.

### The Fractional Factorial Solution
Using **$2^{10-6}_{\text{III}}$ Fractional Factorial Design**, `prompt_prism` tests all 10 factors in **just 16 runs** ($98.4\%$ cost reduction!), while ANOVA mathematically isolates the true individual effect ($\Delta$), standard error, $t$-statistic, and $p$-value for every single factor.

```
Full Factorial (10 factors):             1,024 runs  💸 ($$$)
Fractional Factorial 2^(10-6)_III:          16 runs  ⚡ (-98.4% cost!)
Plackett-Burman Screening (11 factors):     12 runs  🚀 (-99.4% cost!)
```

---

## 🚀 Key Features

- 📐 **Full & Fractional Factorial Catalog ($2^{k-p}$)**: 28 standard Box-Hunter / Montgomery orthogonal designs from $2^{3-1}_{\text{III}}$ ($4$ runs) up to $2^{15-11}_{\text{III}}$ ($16$ runs) with automated alias structure and resolution calculation ($\text{Res III}$, $\text{IV}$, $\text{V}$, $\text{VI}$, $\text{VII}$, $\text{VIII}$).
- 🎲 **Plackett-Burman Screening**: Hadamard-matrix based orthogonal designs for screening $k \le 23$ factors in $N \in \{8, 12, 16, 20, 24\}$ runs.
- 🧩 **Modular Prompt Composer**: Dynamic section assembly, Jinja2 template interpolation, Python string templates, and multi-turn chat message generation.
- 🔌 **Universal LLM Adapter**: Works out of the box with **ANY** Python function (`fn(prompt) -> str`), OpenAI, Anthropic, Gemini, Vertex AI, LiteLLM, or local models.
- 📊 **Comprehensive Metrics & DeepEval Integration**:
  - Deterministic metrics: Exact Match, F1 Score, JSON Schema Validity, Key-Value Extraction Overlap, Levenshtein Similarity, Regex Matching, and Custom callables.
  - LLM-as-a-Judge metrics (`pip install prompt-prism[deepeval]`): Answer Relevancy, Faithfulness, Hallucination, Toxicity, Bias, Contextual Precision/Recall, Summarization, and G-Eval with automatic judge response caching (`JudgeCache`).
- 📈 **Complete ANOVA & RCBD Engine**:
  - Randomized Complete Block Design (RCBD) blocking on `sample_id` to eliminate item difficulty variance.
  - Main Effects & 2-Factor Interactions ($\Delta = \bar{Y}_+ - \bar{Y}_-$)
  - Type I & Type II Sums of Squares, $F$-statistic, $p$-values
  - Effect Sizes: Partial $\eta^2$, Generalized $\eta^2$, $\omega^2$, Cohen's $d$
  - Multiple hypothesis corrections (Bonferroni, Benjamini-Hochberg FDR)
- 🎯 **Optimal Prompt Finder**: Automatically classifies factors into **Positive Boosters**, **Harmful Drops**, and **Neutral Token-Bloat**, generating the statistically optimal prompt configuration.
- 🎨 **Visualizations & Reports**: Main Effects plots, Pareto charts of effects, 2-Factor Interaction matrices, Daniel Half-Normal plots, Markdown reports, and HTML exports.
- 💾 **SQLite Response Caching**: Prevents redundant API spend across reruns.
- 🤖 **Agent Skill & Self-Healing Loops**: Includes an Antigravity Agent Skill (`.agents/skills/prompt_prism/SKILL.md`) enabling autonomous agents to run closed-loop prompt self-healing and continuous prompt improvement.
- 💻 **CLI Interface**: `prompt-prism list-designs`, `prompt-prism design`, and `prompt-prism analyze`.

---

## 📦 Installation

```bash
# Clone and install
git clone https://github.com/robsanmx/prompt_prism.git
cd prompt_prism
pip install -e .

# With visualization dependencies
pip install -e ".[viz]"
```

---

## ⚡ Quickstart: 5-Minute Example

```python
from prompt_prism import Experiment, Factor, ExactMatch, F1Score

# 1. Define Prompt Factors
factors = [
    Factor.binary("persona", level_0_content="", level_1_content="You are an expert catalog analyst."),
    Factor.binary("few_shot", level_0_content="", level_1_content="Example: Product: Nike Air Max -> Brand: Nike"),
    Factor.binary("cot", level_0_content="", level_1_content="Think step by step before answering."),
    Factor.binary("format_json", level_0_content="PlainText", level_1_content="Strict JSON object"),
    Factor.binary("constraints", level_0_content="", level_1_content="Only use facts mentioned in the input."),
]

# 2. Setup Experiment (auto-recommends 2^(5-1)V design with 16 runs)
exp = Experiment.from_factors(
    factors=factors,
    design="2(5-1)V", # 16 runs, Resolution V (unaliased main effects & 2-factor interactions)
    system_prompt="Extract product attributes from listings.",
    data_template="Offer Title: {{ title }}\nDescription: {{ description }}",
    metrics=[ExactMatch(), F1Score()],
    target_metric="f1_score",
    title="Product Attribute Extraction DoE",
)

# 3. Benchmark Dataset
dataset = [
    {"id": 1, "title": "Nike Air Max 90 Red", "description": "Men sneakers size 42", "target": "Brand: Nike, Color: Red"},
    {"id": 2, "title": "Adidas Ultraboost 21", "description": "Running shoes black", "target": "Brand: Adidas, Color: Black"},
    {"id": 3, "title": "Puma Suede Classic", "description": "Retro sneakers blue", "target": "Brand: Puma, Color: Blue"},
]

# 4. Run with any LLM callable or API
def my_llm_client(prompt: str) -> str:
    # Call your preferred LLM provider here (OpenAI, Anthropic, Gemini, local model)
    return "Brand: Nike, Color: Red"

results = exp.run(dataset=dataset, client=my_llm_client, max_workers=4)

# 5. Analyze ANOVA with RCBD blocking & Get Optimal Prompt
report = exp.analyze(block_by="sample_id")
print(report.to_markdown())

# 6. Retrieve Configured Optimal Prompt Template
optimal_template = exp.get_optimal_prompt_template()
```

---

## 📊 Sample ANOVA Output & Pareto Chart

```
=== Pareto Chart of Standardized Effects (f1_score) ===
Factor ID | Name                     | Effect Δ | t-value | Chart
----------+--------------------------+----------+---------+-----------------------------------------
   A      | persona                  |  +0.3520 |   14.21 | ██████████████████████████████ [*** SIG ***]
   B      | few_shot                 |  +0.2480 |   10.02 | █████████████████████          [*** SIG ***]
   D      | format_json              |  -0.1210 |    4.88 | ██████████                     [*** SIG ***]
   C      | cot                      |  +0.0150 |    0.60 | █
   E      | constraints              |  -0.0080 |    0.32 | ░
```

### Generated Executive Recommendation:
> 🎯 **Optimal Prompt Recipe:**
> - 🟢 **ENABLE** `persona` (+35.2% accuracy, $p < 0.001$)
> - 🟢 **ENABLE** `few_shot` (+24.8% accuracy, $p < 0.001$)
> - 🔴 **DISABLE** `format_json` (-12.1% accuracy, $p = 0.002$) $\rightarrow$ *Negative constraint was overly restricting extraction!*
> - ⚪ **OMIT** `cot` and `constraints` ($p > 0.5$) $\rightarrow$ *Omit to reduce token usage and latency without losing accuracy.*

---

## 🤖 LLM-as-a-Judge & DeepEval Integration

`prompt_prism` integrates with **DeepEval** to evaluate complex, semantic, and subjective qualities using LLM judges. It supports two evaluation paradigms:

```bash
pip install prompt-prism[deepeval]
```

> **Note:** `prompt_prism` requires **Python 3.11+**, which also satisfies DeepEval's minimum (its dependency chain, e.g. `langchain-core`, needs Python 3.10+).

### 1. 🎯 Reference-Based Evaluation (WITH Golden Datasets)
When you have ground-truth expected answers, DeepEval evaluates semantic equivalence, factual agreement, and retrieval recall against the golden targets:

```python
from prompt_prism import Experiment, Factor, deepeval_metric, JudgeCache

# Setup SQLite Judge Cache to prevent repeated API calls
judge_cache = JudgeCache(db_path="judge_cache.db")

# A. Golden Dataset with reference targets
golden_dataset = [
    {
        "id": "case_01",
        "input": "Explain why the customer was charged a $50 late fee.",
        "target": "The payment was received on Nov 18, 3 days after the Nov 15 deadline.",  # <-- GOLDEN TARGET
    }
]

# B. G-Eval: Evaluates factual agreement against the Golden Target
factual_geval = deepeval_metric(
    "g_eval",
    name="factual_accuracy",
    criteria="Determine whether the actual output factually agrees with the expected golden target.",
    evaluation_steps=[
        "Check if all key facts from the expected output are preserved in the actual output.",
        "Penalize if the actual output contradicts the expected target or fabricates facts.",
    ],
    cache=judge_cache,
)

# C. Contextual Recall: Measures how much of the golden target was retrieved in context
context_recall = deepeval_metric("contextual_recall", cache=judge_cache)
```

### 2. 🔍 Reference-Free Evaluation (WITHOUT Golden Datasets)
When evaluating open-ended queries or streaming logs without ground-truth labels:

```python
# Answer Relevancy: Checks if the output directly answers the user prompt
relevancy = deepeval_metric("answer_relevancy", threshold=0.7, cache=judge_cache)

# Faithfulness & Hallucination: Verifies grounding against retrieved documents
faithfulness = deepeval_metric("faithfulness", cache=judge_cache)
hallucination = deepeval_metric("hallucination", cache=judge_cache)
```

### 3. Run Factorial Experiment with Judge Metrics

```python
exp = Experiment.from_factors(
    factors=factors,
    design="2(5-1)V",
    metrics=[factual_geval, relevancy],
    target_metric="factual_accuracy",
)
results = exp.run(dataset=golden_dataset, client=my_llm)
report = exp.analyze(block_by="sample_id")
```

> **💡 Best Practice:** Always set `temperature=0` on your LLM judge model and pass `cache=JudgeCache(...)` so that repeated ANOVA runs do not incur unnecessary judge API costs.

---

## 🤖 Agent Skill & Self-Healing Loops

`prompt_prism` includes a built-in agent skill at `.agents/skills/prompt_prism/SKILL.md`. AI coding assistants and autonomous agents can use this skill to run closed-loop **Self-Healing & Prompt Improvement Loops**:

1. **Failure Diagnosis**: Cluster production errors or edge-case failures.
2. **Hypothesis Synthesis**: Propose candidate fix factors ($A, B, C, \dots$).
3. **Orthogonal Tournament**: Run $2^{k-p}$ design on failure cases.
4. **ANOVA Gatekeeping**: Lock in fixes with $p < 0.05$ and $\Delta > 0$; purge harmful mutations ($\Delta < 0$) and placebo bloat ($p > 0.05$).
5. **Continuous Promotion**: Deploy the statistically verified prompt.

---

## 🛠️ CLI Usage

`prompt_prism` includes a command-line interface `prompt-prism`:

### 1. List Available Standard DoE Plans
```bash
prompt-prism list-designs --factors 7
```
```
Plan ID         Factors  Runs   Resolution   Generators
-----------------------------------------------------------------
2(7-1)VII       7        64     Res 7        G=ABCDEF
2(7-2)IV        7        32     Res 4        F=ABCD, G=ABDE
2(7-3)IV        7        16     Res 4        E=ABC, F=BCD, G=ACD
2(7-4)III       7        8      Res 3        D=AB, E=AC, F=BC, G=ABC
```

### 2. Generate a Design Matrix CSV
```bash
prompt-prism design --factors 5 --runs 16 --output design_matrix.csv
```

### 3. Run ANOVA on Results CSV
```bash
prompt-prism analyze --data experiment_results.csv --target f1_score --output-report report.md
```

---

## 🏛️ Architecture

```
prompt_prism/
├── core/                  # Core abstractions: Factor, Level, FactorSet, DesignMatrix, RunConfig, Trial
├── design/                # DoE Engine: FractionalFactorial (2^(k-p)), Plackett-Burman, FullFactorial, Aliasing
├── template/              # Prompt Composition: Modular Sections, Jinja2, Chat Messages
├── runner/                # Execution: Multi-threaded Runner, SQLite Cache, CallableLLM, MockLLM
├── evaluation/            # Metrics: ExactMatch, F1, JSON, ExtractionOverlap, Levenshtein, Custom
├── analysis/              # Statistics: Main Effects, Interactions, OLS, ANOVA (Type I/II/III), Optimizer
├── visualization/         # Plotting: Main Effects, Pareto Chart, Interactions, ASCII Charts
├── reporting/             # Reports: Automated Markdown, HTML, and JSON generators
├── cli/                   # Command Line Tool: prompt-prism
└── ...
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Output:
```
============================== 48 passed in 5.06s ==============================
```

---

## 📜 License

MIT License. Designed and built with ❤️ for universal prompt optimization.
