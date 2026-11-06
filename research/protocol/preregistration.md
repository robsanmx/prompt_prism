# Formal Preregistration Protocol

**Title:** Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization  
**Principal Investigator:** Roberto Sanchez (`@robsanmx`)  
**Protocol Version:** 2.0 (Checkpoint C Confirmatory Lock)  
**Date:** 2026-09-26  
**Status:** FROZEN & LOCKED (Pre-Confirmatory Data Collection)  

---

## 1. Study Information and Primary Question

### 1.1 Research Question
For a fixed task, language model snapshot, factor set ($k=6$), and evaluation budget (16 configurations), how accurately do two-level Resolution IV fractional factorial designs recover exhaustive full-factorial marginal main effects, and how much held-out performance is lost (selection regret) when their recommended prompt replaces the exhaustive optimum?

### 1.2 Claims Under Investigation
This research investigates three independent claims:
1. **Estimation Claim:** A $2^{6-2}_{\text{IV}}$ fractional factorial design recovers marginal main effects with low root-mean-squared error (RMSE) and high sign agreement ($>90\%$ for true $|\tau| > \delta$).
2. **Selection Claim:** Composing a prompt from the estimated main effects produces a configuration with low selection regret on unseen confirmation benchmark items.
3. **Efficiency Claim:** The method attains competitive selection performance while evaluating $75\%$ fewer prompt configurations (16 vs. 64) and reducing overall API/judge resource expenditure.

---

## 2. Experimental Design and Units

* **Target Population:** Natural language understanding and structured extraction tasks evaluated under frozen model snapshots.
* **Experimental Factor Set:** 6 binary semantic factors ($A, B, C, D, E, F$), yielding $2^6 = 64$ exhaustive configurations.
* **Primary Screening Design:** Regular $2^{6-2}_{\text{IV}}$ fractional factorial design (16 runs, Generators: $E = ABC, F = BCD$, Defining Relation: $I = ABCE = BCDF = ADEF$).
* **Experimental Unit:** A prompt configuration $c \in \{0, 1\}^6$.
* **Repeated-Measures Unit:** Benchmark item $i \in \{1, \dots, I\}$ (each item is evaluated across all assigned configurations).
* **Observation:** Response score $Y(c, i, r)$ for configuration $c$, item $i$, and generation replicate $r \in \{1, \dots, R\}$.
* **Split Strategy:** Strict 50/50 item split stratified by difficulty/subclass into **Discovery** items ($I_{\text{disc}}$) and **Confirmation** items ($I_{\text{conf}}$). All prompt selection and effect estimation use $I_{\text{disc}}$ only.

---

## 3. Preregistered Hypotheses

### Primary Hypothesis (H1: Selection Regret)
* **Statement:** At a budget of 16 configurations for 6 binary factors, the prompt configuration $c_{\text{ResIV}}$ selected by the PromptPrism Resolution IV design using discovery items will achieve significantly lower mean held-out selection regret on untouched confirmation items than a prompt configuration $c_{\text{Random}}$ selected by a budget-matched uniform random design of 16 configurations, aggregated across confirmatory task-model pairs.
* **Null Hypothesis ($H_{0,1}$):** Mean held-out regret of $c_{\text{ResIV}}$ is greater than or equal to that of $c_{\text{Random}}$ ($\mu_{\text{regret, ResIV}} \ge \mu_{\text{regret, Random}}$).
* **Test Statistic & Decision Rule:** Paired cluster bootstrap test over benchmark items with $\alpha = 0.05$. If $p < 0.05$ and $\Delta \mu < 0$, $H_{0,1}$ is rejected in favor of PromptPrism.

### Secondary Hypotheses (Confirmatory Family)
* **H2 (Effect Recovery):** For factors with oracle marginal effect $|\tau_j| > \delta$, the Resolution IV design recovers the correct sign $\ge 90\%$ of the time and achieves strictly lower RMSE than a budget-matched 16-run random design.
* **H3 (Confidence Interval Coverage):** In core simulation regimes, nominal $95\%$ confidence intervals for main effects attain empirical coverage between $90\%$ and $98\%$.
* **H4 (Resolution Sensitivity):** Under dense two-factor interaction regimes, Resolution III ($2^{6-3}_{\text{III}}$, 8 runs) estimates exhibit elevated sign reversals due to aliasing; foldover de-aliasing or upgrading to Resolution IV restores sign accuracy.
* **H5 (Generalization):** On confirmation items, the PromptPrism recommended prompt $c_{\text{ResIV}}$ achieves performance strictly superior to the unoptimized baseline prompt $c_{\text{baseline}}$ ($\Delta > 0, p < 0.05$).
* **H6 (Uplift Efficiency):** The PromptPrism recommended prompt captures $\ge 80\%$ of the exhaustive-oracle uplift ($c_{\text{full}}$) while evaluating only $25\%$ of the configurations (16 vs. 64).

---

## 4. Multiplicity and Error Control

* **Primary Endpoint:** Single primary hypothesis (H1). No multiplicity correction required for H1.
* **Secondary Endpoint Family:** All secondary factor tests within a task-model cell apply the **Benjamini-Hochberg (BH) False Discovery Rate (FDR)** procedure at $q = 0.05$.
* **Interactions:** Two-factor interactions are evaluated as exploratory unless explicitly specified in the confirmatory SAP.

---

## 5. Explicit Falsification Criteria

The primary scientific claim of PromptPrism will be declared **unsupported** if any of the following occurs:
1. Monte Carlo simulations fail to recover known additive effects or yield nominal interval coverage below $90\%$ in correctly specified scenarios.
2. Resolution IV selection regret is statistically indistinguishable from or worse than budget-matched random search on confirmation items.
3. The composed optimal prompt fails to generalize beyond baseline on confirmation items ($c_{\text{ResIV}} \le c_{\text{baseline}}$).
4. De-aliasing foldover reveals that significant screening decisions frequently reversed direction upon unconfounding.

---

## 6. Frozen Materials and Cryptographic Checksums

At Checkpoint C, the experimental protocol, factor catalogs, task specifications, model configurations, and dataset splits are locked. All subsequent confirmatory executions must match these SHA-256 digests:

| File | SHA-256 Digest |
|---|---|
| `research/configs/factors.yaml` | `b08300d77163b98bcd32ccadaf5baf90fb1e534ec9db286677536e22cc315f6d` |
| `research/configs/tasks.yaml` | `d03d15af5c5412fdb7258108053c40d38159529237a23371d1b1f136e7c426b9` |
| `research/configs/models.yaml` | `0978493398389bf0bb297342920d5830ecf6346c103414f57e215c025ff460cf` |
| `research/data/dataset_manifest.yaml` | `52475eccb61d5c1c596ccd246e7a6149cffc33f31c8c3fa2e8f9ba8273e7785d` |
| `research/configs/confirmatory.yaml` | Frozen Checkpoint C Manifest |
