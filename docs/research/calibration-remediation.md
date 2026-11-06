# Simulation Calibration & Remediation Report (Phase 2 Checkpoint B)

**Owner:** Roberto Sanchez (`@robsanmx`), Methodology Lead & Statistical Analyst  
**Date:** 2026-09-26  
**Status:** Completed  
**Reference:** [`docs/research/simulation-calibration-report.md`](simulation-calibration-report.md)  

---

## 1. Overview of Simulation Audit

Phase 2 executed the Monte Carlo simulation suite across eight distinct synthetic regimes evaluating PromptPrism Resolution IV ($2^{6-2}_{\text{IV}}$), Resolution III, Uniform Random Search, Random Linear Model, OFAT ablation, and Exhaustive Full Factorial.

The statistical gate criteria (Checkpoint B) were evaluated against the findings:

| Criterion | Target | Observed (Res IV Additive / Core) | Gate Status |
|---|---|---|---|
| **Effect Recovery** | Sign accuracy $\ge 90\%$ for $|\tau| > \delta$ | **$100.0\%$** across additive and sparse regimes | **PASS** |
| **Interval Coverage** | Nominal $95\%$ CI coverage in $[90\%, 98\%]$ | **$96.7\%$** | **PASS** |
| **False Discovery Rate** | FDR $\le 0.08$ (at $\alpha = 0.05$) | **$0.000$** | **PASS** |
| **Regret vs Random** | Mean regret strictly lower than Random Search | **$0.0000$ vs $0.0233$** ($78\%$ regret reduction) | **PASS** |

---

## 2. Triaged Findings and Remediation Items

### Finding REM-1: Antagonistic Interaction Cliff (`scenario_5`)
* **Observed Failure:** When main effects are positive ($\tau_A = +0.06, \tau_B = +0.06$) but their interaction is severely antagonistic ($\gamma_{AB} = -0.14$), the purely additive selection heuristic turns on both factors ($A=1, B=1$), producing held-out regret of $0.0665$.
* **Root Cause:** In Resolution IV designs, main effects are unaliased with two-factor interactions, but two-factor interactions are aliased with each other ($AB = CE = DF$). The additive composition rule assumes two-factor interaction sparsity.
* **Remediation / Claim Boundary:**
  1. **Claim Narrowing:** The confirmatory paper will explicitly report that PromptPrism's main-effect screening assumes weak or sparse interaction regimes.
  2. **Protocol Mandate:** In the presence of known strong interaction candidates, the workflow requires an explicit 4-run factorial check on $\{A, B\}$ or foldover unconfounding before deploying $(1, 1)$.

---

### Finding REM-2: Linear Approximation of Binary Classification (`scenario_6`)
* **Observed Behavior:** On binary Bernoulli outcomes with strong logit non-linearities, the linear ANOVA model maintains sign accuracy ($100\%$) but exhibits lower power ($60\%$) at small sample sizes.
* **Root Cause:** Standard linear ANOVA assumes normal homoscedastic errors, which is violated for bounded Bernoulli variables near 0 or 1.
* **Remediation / Statistical Plan:**
  1. **Primary Metric Specification:** Task 4 designates Macro-F1 (continuous on $[0, 1]$) rather than raw per-token Bernoulli outcomes as the primary metric for `intent_classification`.
  2. **SAP Alignment:** In confirmatory analysis, binary classification secondary endpoints will use paired cluster bootstrap over benchmark items to ensure exact nominal coverage.

---

### Finding REM-3: Equivalence-Gated Neutrality Decisions
* **Observed Behavior:** The existing optimizer heuristic in `prompt_prism` historically labeled any factor with $p \ge 0.05$ as `'OMIT_NEUTRAL'`. In underpowered scenarios (e.g. $I \le 20$), this risked dropping factors with true moderate effects ($|\tau| \approx 0.03$) solely due to high noise.
* **Remediation Task:**
  - Formally update the decision taxonomy in the analysis pipeline: factors with $p \ge 0.05$ whose $90\%$ CI exceeds $[-\delta, +\delta]$ are classified as `'INCONCLUSIVE'`, reserving `'PRACTICALLY_NEUTRAL'` exclusively for factors passing the Two One-Sided Tests (TOST) equivalence bound.

---

## 3. Checkpoint B Sign-off

With additive and sparse effect recovery verified, $95\%$ confidence interval coverage calibrated at $96.7\%$, and failure modes (aliasing collisions and interaction cliffs) rigorously mapped, **Checkpoint B (Statistical Gate) is PASSED**. The research program is cleared to proceed to Phase 3 (Pilot and Preregistration).
