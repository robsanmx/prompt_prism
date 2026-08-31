---
name: prompt_prism
description: >-
  Universal framework for optimizing and improving LLM prompts using Fractional Factorial
  Design of Experiments (DoE) and ANOVA. Use whenever the user wants to test multiple prompt
  variations, identify statistically significant prompt factors, reduce evaluation costs by 90%+,
  eliminate prompt guesswork, or find the mathematically optimal prompt recipe.
---

# 🔬 PromptPrism: Design of Experiments & ANOVA for Prompt Engineering

Use this skill whenever you or the user need to:
- **Systematically optimize an LLM prompt** across multiple design factors (persona, few-shot examples, chain-of-thought, output schemas, guardrails, constraints).
- **Quantify the exact performance impact ($\Delta$)** of each prompt component with statistical confidence ($p$-values, $t$-statistics, effect sizes).
- **Screen 5 to 23 candidate prompt factors** in a fraction of the cost ($90\%$ to $98\%$ cheaper than full factorial testing) using **Fractional Factorial ($2^{k-p}$)** or **Plackett-Burman** orthogonal designs.
- **Isolate true factor effects from dataset noise** using **Randomized Complete Block Design (RCBD)**.
- **Generate the mathematically winning prompt** and produce publication-grade ANOVA tables, Pareto charts, and diagnostic reports.

---

## 📐 When to Use Which Design

| Scenario | Factors ($k$) | Recommended Design | Runs ($N$) | Resolution | Note |
|:---|:---:|:---:|:---:|:---:|:---|
| **Quick Screening** | 5 – 11 | `2(7-4)III`, `2(11-7)III`, or `PB-12` | 8 – 16 | **Res III** | Ranks candidate factors fast; main effects aliased with 2-way interactions. Follow up with confirmation run. |
| **Clean Main Effects** | 4 – 8 | `2(4-1)IV`, `2(6-2)IV`, `2(7-3)IV`, `2(8-4)IV` | 8 – 32 | **Res IV** | Main effects clean of 2-factor interactions. |
| **Production Grade / Confirmation** | 3 – 5 | `2(5-1)V`, `2(6-1)VI`, or Full Factorial | 16 – 32 | **Res V+** | Main effects & 2-factor interactions unaliased. Safe for direct production shipping. |
| **Small Number of Factors** | 2 – 3 | `FullFactorial` ($2^k$) | 4 – 8 | **Full** | Tests all $2^k$ combinations with full interaction resolution. |

---

## 🛠️ Step-by-Step Agent Workflow

### Step 1: Define the Prompt Factors
Formulate binary ($0 = \text{Off/Baseline}, 1 = \text{On/Variant}$) or multi-level candidate factors:

```python
from prompt_prism import Factor, FactorSet

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
        description="Reasoning style",
    ),
    Factor.binary(
        name="strict_json_schema",
        level_0_content="Return plain text.",
        level_1_content="OUTPUT FORMAT: Return a valid JSON object matching the schema.",
        description="Output formatting",
    ),
    Factor.binary(
        name="negative_guardrail",
        level_0_content="",
        level_1_content="Rule: Never extrapolate or hallucinate facts not in the context.",
        description="Hallucination constraint",
    ),
]
```

---

### Step 2: Prepare Benchmark Dataset & Evaluation Metrics
Prepare a representative test dataset (typically 10 to 50 items) with ground-truth targets:

```python
from prompt_prism import ExactMatch, F1Score, JSONValidation, KeyValuesExtractionOverlap

# Benchmark sample cases
dataset = [
    {"id": 1, "text": "...", "target": "..."},
    {"id": 2, "text": "...", "target": "..."},
]

# Metrics suite
metrics = [ExactMatch(), F1Score(), JSONValidation()]
```

---

### Step 3: Initialize the Experiment Orchestrator

```python
from prompt_prism import Experiment

exp = Experiment.from_factors(
    factors=factors,
    design="2(5-1)V",  # Or let it auto-select with max_runs=16
    system_prompt="Execute the task adhering strictly to guidelines.",
    data_template="Input data:\n{{ text }}",
    metrics=metrics,
    target_metric="f1_score",
    title="Production Prompt Optimization DoE",
)
```

---

### Step 4: Connect LLM Provider & Execute Trials
Connect **ANY** LLM provider (OpenAI, Anthropic, Gemini, Vertex, local models, or custom Python callable):

```python
def call_llm(prompt: str) -> str:
    # Call your preferred LLM provider here
    # response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    # return response.choices[0].message.content
    return "Predicted output"

# Run experiment (automatically parallelized and cached)
results = exp.run(
    dataset=dataset,
    client=call_llm,
    max_workers=4,
    cache_db="prompt_cache.db",  # Optional SQLite cache to avoid redundant API cost
)
```

---

### Step 5: Analyze with ANOVA & Randomized Complete Block Design (RCBD)

Always run analysis with **RCBD blocking on `sample_id`** to factor out dataset item difficulty from the residual error:

```python
report = exp.analyze(
    block_by="sample_id",  # RCBD blocking
    include_interactions=True,
    alpha=0.05,
)

# Print comprehensive markdown report
print(report.to_markdown())
```

---

### Step 6: Interpret Results & Extract the Winning Prompt

The optimizer automatically classifies factors into:
1. 🟢 **Positive Boosters (MUST ENABLE)**: Factors with statistically significant positive $\Delta$ ($p < 0.05$).
2. 🔴 **Harmful Penalties (MUST DISABLE)**: Factors with statistically significant negative $\Delta$ ($p < 0.05$).
3. ⚪ **Neutral Bloat (OMIT)**: Factors with $p \ge 0.05$. Omit these to reduce token usage and API latency without accuracy loss.

```python
# 1. Inspect recommendation
opt = report.optimal_recommendation
print("Optimal Factor Levels:", opt.optimal_factor_levels)
print("Expected Gain:", f"{opt.expected_gain_absolute:+.4f} ({opt.expected_gain_pct:+.1f}%)")

# 2. Retrieve configured optimal prompt template
winning_template = exp.get_optimal_prompt_template()

# 3. Render sample prompt for production
production_prompt = exp.composer.compose_text(
    run_config=exp.design.runs[0].model_copy(update={"factor_levels": opt.optimal_factor_levels}),
    data={"text": "New sample input"},
)
print("Winning Prompt:\n", production_prompt)
```

---

### Step 7: Screen $\to$ Confirm Loop (If using Resolution III)

If the initial experiment was a **Resolution III screening design** ($2^{7-4}_{\text{III}}$, $2^{11-7}_{\text{III}}$, or Plackett-Burman), suggest and run a confirmation design over the surviving factors before shipping to production:

```python
# Automatically generates an unaliased Resolution V confirmation design
confirmation_design = exp.suggest_confirmation_design(max_runs=16)
print("Confirmation Plan:", confirmation_design.plan_id)
```

---

## 📊 Visualizations & Plotting

Export publication-grade diagnostic plots:

```python
from prompt_prism import plot_main_effects, plot_pareto_effects, plot_interaction_effects

# Main Effects Plot (shows mean performance per factor level)
plot_main_effects(report.anova_result, save_path="main_effects.png")

# Pareto Chart of Standardized Effects (highlights significant factors vs t-critical)
plot_pareto_effects(report.anova_result, save_path="pareto_effects.png")

# 2-Factor Interaction Plots
plot_interaction_effects(report.anova_result, save_path="interactions.png")
```

---

## 💻 CLI Commands Quick Reference

```bash
# 1. List available orthogonal designs for k factors
prompt-prism list-designs --factors 7

# 2. Generate and export a Design Matrix to CSV
prompt-prism design --factors 5 --runs 16 --output design_matrix.csv

# 3. Analyze pre-computed experimental results CSV
prompt-prism analyze --data experiment_results.csv --target f1_score --output-report report.md
```

---

## 🚨 Critical Methodological Rules for Agents

1. **Always Block on `sample_id`**: Never treat $(N \times M)$ trials as independent unblocked observations. Always pass `block_by="sample_id"` to avoid pseudoreplication.
2. **Respect Design Resolution**:
   - **Res III**: Treat significant factors as *candidates*. Always advise running a confirmation design.
   - **Res IV**: Main effects are clean, but 2-way interactions are aliased.
   - **Res V+**: Both main effects and 2-way interactions are unaliased and safe for direct production decisions.
3. **Drop Neutral Factors**: If a factor has $p > 0.05$, recommend level 0 (omit) to optimize token efficiency and latency.
