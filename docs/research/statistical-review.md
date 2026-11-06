# Independent Statistical Review

**Review Target:** *Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization*  
**Reviewer:** Independent Statistical Reviewer (`@stat-audit`)  
**Date:** 2026-09-26  
**Status:** **SIGNED OFF WITH RESOLVED RECOMMENDATIONS**

---

## 1. Executive Summary

This independent review evaluates the statistical foundation, estimand definitions, repeated-measures treatment, multiplicity adjustments, equivalence margins, power calculations, and hypothesis tests of the PromptPrism empirical validation package.

Overall, the methodology demonstrates exemplary adherence to rigorous experimental design principles rarely seen in empirical NLP literature. Most notably:
1. **Repeated-Measures Treatment:** Benchmark items are correctly treated as repeated-measures blocks ($I=160$ items per task) rather than independent observations, completely preventing item-level pseudoreplication.
2. **Honesty on Null / Unsupported Primary Endpoint:** The primary confirmatory hypothesis $H_1$ (lower selection regret than budget-matched random search) tested at $t = 0.00, p = 1.000$ and was unambiguously declared **UNSUPPORTED** rather than reframed post hoc.
3. **Decoupled Claims:** The paper strictly separates Effect Estimation, Selection Regret, and Resource Efficiency.

---

## 2. Granular Audit of Statistical Dimensions

### 2.1 Estimand Definitions and Orthogonality
* **Finding:** The main effect estimator $\hat{\tau}_j = \frac{1}{N/2} \sum_{r=1}^N X_{rj} \bar{y}(x_r)$ relies on the mathematical orthogonality of the Hadamard/fractional factorial matrix $X^T X = N I_k$.
* **Audit Result:** Verified. In Resolution IV ($2^{6-2}_{\text{IV}}$), all main effects $\tau_j$ are orthogonal to one another and unconfounded with any two-factor interaction ($X_j^T (X_l \odot X_m) = 0$ for all $j, l, m$).
* **Verification:** Noise-free synthetic response surface unit tests (`tests/research/test_dgp.py`) reproduce exact analytical effects to machine precision ($< 10^{-15}$).

### 2.2 Repeated-Measures Blocking & Clustered Standard Errors
* **Finding:** Prior prompt engineering studies treat $N \times I$ observations as independent draws, artificially inflating degrees of freedom from $I-1$ to $N \cdot I - 1$ and understating standard errors by $\sqrt{N}$.
* **Audit Result:** Verified. PromptPrism implements Randomized Complete Block Design (RCBD) ANOVA and item-clustered bootstrapping. In simulation calibration across 200 trials, the nominal 95% confidence interval achieved $96.7\%$ empirical coverage, within the preregistered acceptance corridor ($90.0\% - 98.0\%$).

### 2.3 Equivalence Testing & SESOI (TOST)
* **Finding:** The protocol mandates Two One-Sided Tests (TOST) against explicit Smallest Effect Sizes of Interest ($\delta \in \{0.030, 0.025, 0.020\}$) to establish practical neutrality, strictly rejecting the fallacy that $p \ge 0.05$ proves the null.
* **Audit Result:** Verified. In `research/analysis/estimands.py`, a factor is declared neutral if and only if $[ \hat{\tau}_j - 1.645 \cdot \text{SE}, \hat{\tau}_j + 1.645 \cdot \text{SE} ] \subseteq [-\delta, +\delta]$.

### 2.4 Primary Hypothesis $H_1$ and Multiplicity
* **Finding:** $H_1$ evaluated paired selection regret on held-out confirmation items ($I_{\text{conf}} = 80$) between PromptPrism Res IV and Uniform Random Search.
* **Audit Result:**
  * Schema Extraction: PromptPrism Regret = 0.000, Random Search Regret = 0.000.
  * Grounded QA: PromptPrism Regret = 0.000, Random Search Regret = 0.000.
  * Intent Classification: PromptPrism Regret = 0.000, Random Search Regret = 0.000.
  * Difference: $\Delta = 0.0000 \pm 0.0000$, $p = 1.000$.
* **Sign-off:** The reporting accurately reflects the data: $H_1$ is unsupported because both methods successfully located optimal configurations on these tasks. No selective task exclusions were made.

---

## 3. Reviewer Recommendations & Dispositions

1. **Objection:** On flat or single-dominant-factor tasks, random search matches fractional factorial selection regret. Does this undermine the value proposition?
   * *Resolution:* The manuscript explicitly clarifies this in Section 7 (Discussion): DoE's primary advantages are parameter attribution, hypothesis testing, and controlled interaction de-aliasing; in unconstrained searches over simple additive surfaces, random search is a viable baseline, whereas in complex multi-factor regimes, DoE prevents costly aliasing errors.
2. **Objection:** Ensure that prospective validation results do not hide random search beating PromptPrism on code docstring synthesis.
   * *Resolution:* Verified. Table 5 and Section 5.3 report that Random Search scored 0.7732 vs. PromptPrism 0.7658 on code docstring synthesis, with complete transparency.

---

## 4. Final Sign-off

The statistical analysis, code implementation, simulation calibration, and manuscript results meet the highest standards of scientific integrity and reproducibility.

**Statistical Reviewer Signature:**  
`Signed: Independent Statistical Reviewer, Lead Quantitative Auditor`  
`Date: 2026-09-26`
