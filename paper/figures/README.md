# Paper Figures Directory

This directory contains the figures for the manuscript:
*Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization*

## Figure Manifest

1. **Figure 1: `fig1_workflow.png` / `fig1_workflow.pdf`**
   - Experimental workflow from factor definition ($k=6$), orthogonal matrix construction, repeated-measures blocking across items, and held-out confirmation testing.

2. **Figure 2: `fig2_effect_recovery.pdf` / `fig2_effect_recovery.png`**
   - (a) Main effect estimation RMSE across four synthetic regimes (Additive, Sparse 2FI, Dense 2FI, Aliased Collision), comparing PromptPrism Resolution IV ($N=16$), OFAT Ablation ($N=7$), and Uniform Random Search ($N=16$).
   - (b) Empirical 95% Confidence Interval coverage across 200 Monte Carlo trials per regime.

3. **Figure 3: `fig3_regret_vs_budget.pdf` / `fig3_regret_vs_budget.png`**
   - (a) Effect estimation RMSE scaling as a function of benchmark item sample size ($I \in \{10, 20, 40, 80, 160\}$).
   - (b) Uplift efficiency frontier: Percentage of exhaustive oracle uplift captured as configuration runs increase from 7 to 64.

4. **Figure 4: `fig4_alias_collision.pdf` / `fig4_alias_collision.png`**
   - Illustration of aliased collision failure mode: In Resolution III ($2^{6-3}$), a strong two-factor interaction ($BD = +0.08$) bleeds into the null main effect $A$, producing a false positive. In Resolution IV ($2^{6-2}$), main effects remain clear of two-factor interactions.

5. **Figure 5: `fig5_prospective_and_judge.pdf` / `fig5_prospective_and_judge.png`**
   - (a) Prospective validation on unseen task-model cells (`code_docstring_synthesis` and `financial_sentiment_analysis`).
   - (b) Judge-human agreement calibration scatter plot demonstrating strong correlation ($r = 0.986$, $\rho = 0.975$) and bounded systematic bias ($+0.045$).

## Reproduction Command

All figures can be deterministically regenerated from machine-readable result files via:
```bash
python paper/figures/generate_figures.py
```
