# Research Decision Log

**Owner:** Roberto Sanchez (`@robsanmx`), Principal Investigator & Methodology Lead  
**Scope:** Covers all architectural, statistical, and experimental protocol decisions for the scientific validation of PromptPrism.  
**Rule:** Every substantive choice that alters methodology, estimands, designs, or analysis must be recorded here with date, rationale, and alternatives considered before implementation.

---

## Log Entries

### DEC-001: Separation of Estimation, Selection, and Efficiency Claims
* **Date:** 2026-09-26
* **Owner:** Roberto Sanchez
* **Status:** Adopted
* **Context:** Preliminary documentation sometimes described a 16-run design as achieving "98% cost savings", conflating the size of the parameter search space ($2^k$) with monetary API cost, judge cost, and sample efficiency. Furthermore, recovering main effects does not guarantee that composing them yields an optimal prompt when interactions exist.
* **Decision:** Formally decouple the research into three separate claims:
  1. *Estimation claim*: Accuracy of recovering marginal main effects ($\tau_j$).
  2. *Selection claim*: Low held-out regret of the composed prompt ($c_{\text{method}}$ vs $c_{\text{full}}$).
  3. *Efficiency claim*: Quantified reduction across 4 distinct axes: configurations, model calls, token/monetary cost, and wall-clock latency.
* **Rationale:** Preserves scientific integrity and protects against invalid generalizations.
* **Alternatives Considered:** A single composite utility score; rejected because it obscures interaction failures and judge overhead.

---

### DEC-002: Benchmark Item as the Repeated-Measures Unit
* **Date:** 2026-09-26
* **Owner:** Roberto Sanchez
* **Status:** Adopted
* **Context:** In factorial prompt evaluation, multiple prompt configurations are evaluated across the same benchmark items. Treating every response as an independent observation induces severe pseudoreplication and artificially deflates standard errors.
* **Decision:** Designate the benchmark item as the repeated-measures clustering unit. All inferential procedures (ANOVA, cluster bootstrap, linear mixed-effects) must account for item-level clustering.
* **Rationale:** Prevents false positive findings caused by item difficulty variance.
* **Alternatives Considered:** Treating items as fixed effects only in OLS; retained as tool's internal view, but mixed-effects / clustered bootstrap required for confirmatory claims.

---

### DEC-003: Equivalence Testing for Practical Neutrality
* **Date:** 2026-09-26
* **Owner:** Roberto Sanchez
* **Status:** Adopted
* **Context:** In traditional hypothesis testing, failing to reject $H_0: \beta = 0$ ($p \ge 0.05$) is often mistakenly interpreted as "this prompt component has no effect and can be safely removed". In reality, $p \ge 0.05$ often reflects low statistical power.
* **Decision:** Practical neutrality requires an equivalence test (e.g. Two One-Sided Tests, TOST) demonstrating that the $95\%$ confidence interval falls entirely within $[-\delta, +\delta]$, where $\delta$ is the smallest effect size of interest.
* **Rationale:** Prevents dropping useful prompt components under underpowered conditions.
* **Alternatives Considered:** Strict $p$-value threshold ($p > 0.05$); rejected as scientifically invalid for establishing equivalence.

---

### DEC-004: Strict 50/50 Discovery-Confirmation Split with Locked Analysis
* **Date:** 2026-09-26
* **Owner:** Roberto Sanchez
* **Status:** Adopted
* **Context:** Iterative tuning of prompt factors or selection heuristics on test data leads to optimizer overfitting.
* **Decision:** Benchmark datasets will be split 50/50 into discovery and confirmation subsets before any experiments run. The full factorial and fractional methods will select candidate prompts using discovery data only. Confirmation outcomes will remain encrypted/inaccessible until the analysis code and recommendations are frozen.
* **Rationale:** Guarantees unbiased evaluation of generalization and selection regret.
* **Alternatives Considered:** $k$-fold cross-validation; rejected for primary confirmation due to risk of leakage across prompt compositions, though permissible in exploratory sensitivity analysis.

---

### DEC-005: Freezing Primary Estimand as Held-Out Selection Regret (H1)
* **Date:** 2026-09-26
* **Owner:** Roberto Sanchez
* **Status:** Adopted
* **Context:** The literature lacks a unified benchmark criterion for comparing prompt optimizers; black-box optimizers find combinations without explaining factors, while DoE explains factors and recommends a recipe.
* **Decision:** Designate **Held-Out Selection Regret** ($\text{Regret} = \bar{Y}_{\text{conf}}(c_{\text{full}}) - \bar{Y}_{\text{conf}}(c_{\text{method}})$) on unseen confirmation items as the single primary confirmatory endpoint (Hypothesis H1). Effect recovery accuracy ($\tau_j$), calibration, and uplift ratio are formal secondary endpoints evaluated under FDR control.
* **Rationale:** Regret directly quantifies the practical engineering cost of using a 16-run fractional design instead of an exhaustive 64-run full factorial.
* **Alternatives Considered:** In-sample ANOVA $R^2$; rejected because in-sample explanatory power does not guarantee held-out generalization.

---

### DEC-006: Confirmatory Parameter Freeze and Pre-Data Locking (Checkpoint C)
* **Date:** 2026-09-26
* **Owner:** Roberto Sanchez
* **Status:** Adopted & Locked
* **Context:** Completion of Study A Monte Carlo calibration and the excluded-data pilot provides empirical justification for sample size, equivalence bounds, and models. All parameters must be sealed before confirmatory execution.
* **Decision:** Freeze the confirmatory experimental matrix: 3 tasks (`schema_extraction`, `grounded_qa`, `intent_classification`), 3 models (`qwen2.5-7b-instruct`, `gpt-4o-mini-2024-07-18`, `claude-3-5-sonnet-20241022`), $I = 160$ items per task (80 discovery / 80 confirmation), Resolution IV $2^{6-2}_{\text{IV}}$ (16 runs), and equivalence bounds $\delta \in \{0.030, 0.025, 0.020\}$. Cryptographic SHA-256 hashes of all configuration materials are recorded in `research/configs/confirmatory.yaml` and `research/protocol/preregistration.md`.
* **Rationale:** Eliminates researcher degrees of freedom and establishes a watertight audit trail.
* **Alternatives Considered:** Allowing post-hoc sample size adjustments; rejected as a violation of preregistered confirmatory protocols.
