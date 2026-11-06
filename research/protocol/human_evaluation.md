# Human Evaluation and Judge Validation Protocol (Task 15)

**Owner:** Roberto Sanchez (`@robsanmx`), Methodology Lead  
**Date:** 2026-09-26  
**Status:** Protocol Active  
**Reference:** [`docs/research/promptprism-validation-plan.md`](../../docs/research/promptprism-validation-plan.md) (Section 7.7)  

---

## 1. Objectives & Scope

Whenever an LLM-as-a-judge metric (e.g. DeepEval G-Eval, faithfulness, or hallucination) is used, automated judge scores cannot serve as unvalidated ground truth. This protocol specifies:
1. Stratified sampling of at least $20\%$ of judged model outputs.
2. Complete blinding of annotators to prompt configurations, factors, and model identities.
3. Independent double-annotation to measure human-human agreement (inter-rater reliability).
4. Statistical evaluation of judge-human agreement, calibration, and systematic judge bias.

---

## 2. Sampling Procedure

* **Sample Size:** At least $20\%$ of unique model generation outputs across tasks and conditions.
* **Stratification:** Sample is stratified across:
  * Score quartiles (low, moderate, high).
  * Task categories.
  * Active factor conditions.
* **Blinding:** Output responses are stripped of prompt headers, model metadata, factor level indicators, and generation timestamps. Each instance receives a randomized `blinded_id`.

---

## 3. Evaluation Rubric & Metrics

Human annotators score outputs on a standardized 1–5 Likert scale across three dimensions:
1. **Factual Correctness / Grounding:** Does the output accurately reflect verified reference facts?
2. **Schema Adherence:** Does the response strictly follow requested format constraints?
3. **Completeness:** Are all queried attributes present?

### Statistical Agreement Targets:
* **Inter-Rater Reliability (Human-Human):** Cohen's $\kappa \ge 0.60$ or Krippendorff's $\alpha \ge 0.65$ (substantial agreement).
* **Judge-Human Agreement:** Spearman rank correlation $\rho \ge 0.70$.
* **Systematic Bias:** Bland-Altman mean difference within $\pm 0.05$ on normalized scale.
