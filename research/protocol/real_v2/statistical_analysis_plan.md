# Statistical Analysis Plan (SAP) for PromptPrism Real-LLM Confirmatory Study v2
**Document Identifier:** SAP-REAL-V2-2026-09-26  
**Status:** FROZEN PRE-CONFIRMATORY REGISTRATION  
**Date:** 2026-09-26  
**Author:** trev21mx@gmail.com  

---

## 1. Study Overview & Confirmatory Design

### 1.1 Scope
This Statistical Analysis Plan governs the confirmatory evaluation of **PromptPrism Resolution IV ($2^{6-2}_{\text{IV}}$)** prompt factor screening versus budget-matched comparators across real LLM inference calls on curated benchmark datasets.

### 1.2 Confirmatory Factorial Space
- **Factors:** 6 binary factors $A, B, C, D, E, F \in \{-1, +1\}$ (contrast coded).
- **Design:** $2^{6-2}_{\text{IV}}$ regular fractional factorial design (16 distinct runs).
- **Generators:** $E = ABC$, $F = BCD$.
- **Defining Relation:** $I = ABCE = BCDF = ADEF$.
- **Resolution:** IV (no main effect is aliased with any 2-factor interaction; 2-factor interactions are aliased with each other in three alias sets: $AB=CE, AC=BE, BC=AE$, plus $AD=EF, AF=DE$, etc.).

### 1.3 Confirmatory Response Matrix (122,880 calls)
$$\text{Total Calls} = 3\text{ tasks} \times 64\text{ configurations} \times 160\text{ items} \times 4\text{ replicate-weights} = 122{,}880$$
- **3 Tasks:**
  1. `schema_extraction`: Micro-averaged field F1 ($SESOI = 0.030$).
  2. `grounded_qa`: Exact match / token unigram F1 with abstention accuracy ($SESOI = 0.025$).
  3. `intent_classification`: Classification accuracy / macro F1 ($SESOI = 0.020$).
- **3 Models:**
  1. `meta-llama/Llama-3.1-8B-Instruct` (Open-weight anchor, 2 replicates).
  2. `gpt-4o-mini-2024-07-18` (Economical commercial API, 1 replicate).
  3. `claude-3-5-haiku-20241022` (Alternative commercial family, 1 replicate).
- **Items:** 160 total items per task = 80 discovery items + 80 confirmation items. Pilot partition items ($N=20$) are strictly quarantined and cannot enter confirmatory analysis.

---

## 2. Hypotheses & Decision Rules

### 2.1 Primary Hypothesis H1: Held-Out Selection Regret
- **Estimand:**
  $$\text{Regret}_c(\text{Method}) = \bar{Y}_c(\text{Oracle}_{\text{disc}}) - \bar{Y}_c(\text{Selected}_{\text{disc}})$$
  evaluated on held-out **confirmation** items within cell $c \in \{1, \dots, 9\}$.
  $$\Delta_{\text{agg}} = \frac{1}{9} \sum_{c=1}^9 \left[ \text{Regret}_c(\text{PromptPrism}) - \text{Regret}_c(\text{BalancedRandom}) \right]$$
- **Inferential Method:** Cell-stratified item-cluster bootstrap ($B = 2{,}000$ iterations).
  In each bootstrap replicate $b$:
  1. Resample discovery items with replacement within cell.
  2. Rerun selection procedure for PromptPrism (RCBD main effects) and Balanced Random (16-run draw).
  3. Resample confirmation items with replacement within cell.
  4. Evaluate held-out regret of both selected configurations.
  5. Compute cell difference and unweighted aggregate $\Delta_{\text{agg}}^{(b)}$.
- **Decision Criterion:**
  $$\text{H1 Supported} \iff \bar{\Delta}_{\text{agg}} < 0 \quad \text{AND} \quad p = \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_{\text{agg}}^{(b)} \ge 0) < 0.05$$
  *Rule:* A positive mean or $p \ge 0.05$ strictly results in UNSUPPORTED.

### 2.2 Secondary Hypotheses

#### H2: Marginal Effect Recovery
- **Estimand:** RMSE and sign agreement between PromptPrism estimated main effects and exhaustive discovery oracle effects for all factors where $|\tau_{\text{oracle}}| \ge \text{SESOI}$.
- **Decision Criterion:** PromptPrism achieves $\text{RMSE} < \text{RMSE}_{\text{random}}$ and $\text{Sign Agreement} \ge 90\%$.

#### H3: Baseline Generalization
- **Estimand:** Improvement of PromptPrism selected configuration over the all-minus-1 baseline $c_0$ on confirmation items:
  $$\Delta_{\text{base}, c} = \bar{Y}_c(\text{Selected}) - \bar{Y}_c(c_0)$$
- **Inference:** Paired t-test across confirmation items with Benjamini-Hochberg False Discovery Rate control ($\alpha = 0.05$) across the 9 cells.

#### H4: Uplift Efficiency
- **Estimand:**
  $$\text{Efficiency}_c = \frac{\bar{Y}_c(\text{Selected}) - \bar{Y}_c(c_0)}{\bar{Y}_c(\text{Oracle}) - \bar{Y}_c(c_0)}$$
- **SESOI Exemption Rule:** If $\bar{Y}_c(\text{Oracle}) - \bar{Y}_c(c_0) \le \text{SESOI}_c$, efficiency is undefined and reported as `NaN` / `SESOI_DENOMINATOR_EXCLUSION`. It is NEVER reported as 100%.
- **Decision Criterion:** In non-excluded cells, $\text{Efficiency}_c \ge 0.95$.

#### H5: Resource Efficiency
- **Estimand:** PromptPrism utilizes exactly 16 configurations ($75\%$ reduction compared to the 64-configuration exhaustive oracle). Empirical call, token, latency, and cost savings are directly measured and reported without assuming linear proportionality.

---

## 3. Boundary Protections & Protocol Integrity

1. **Quarantine of Simulated Fixtures (DEV-002):** No simulated fixtures or v1 mock artifacts may enter the empirical dataset.
2. **Quarantine of Pilot Items:** 20 pilot items per task are strictly segregated; any attempt to load pilot records in confirmatory routines raises an immediate `ValueError`.
3. **Resumable Content-Addressed Execution:** All API requests are content-addressed by SHA-256 hashes of `(model_id, prompt_text, temperature, seed)`.
4. **Pre-Unlock Recommendation Lock:** All discovery recommendations and analysis scripts must be hashed and committed before confirmation data is evaluated.
