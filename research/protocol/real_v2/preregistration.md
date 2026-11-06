# Formal Preregistration Document: PromptPrism Empirical Validation (Study v2)
**Registration Hash / Identifier:** PREREG-REAL-V2-2026-09-26  
**Timestamp:** 2026-09-26T21:00:00Z  
**Primary Contact:** Roberto (trev21mx@gmail.com)  
**Repository:** `robsanmx/prompt_prism`  

---

## 1. Study Information

- **Title:** Empirical Validation of Fractional Factorial Design of Experiments for LLM Prompt Optimization
- **Research Question:** Does a Resolution IV fractional factorial design ($2^{6-2}_{\text{IV}}$, 16 runs) achieve lower held-out prompt selection regret than budget-matched uniform random search under real model inference?
- **Primary Design:** $2^{6-2}_{\text{IV}}$ regular fraction with generators $E = ABC, F = BCD$ and contrast coding $\{-1, +1\}$.
- **Primary Hypotheses:**
  - **H1 (Primary):** Aggregate selection regret across 9 task–model cells is strictly lower for PromptPrism than budget-matched random search ($\bar{\Delta}_{\text{regret}} < 0$, $p < 0.05$ via cell-stratified item-cluster bootstrap).
  - **H2 (Effect Recovery):** PromptPrism main effects achieve lower RMSE and $\ge 90\%$ sign agreement above SESOI compared to random designs.
  - **H3 (Baseline Generalization):** Selected prompt improves over all-negative baseline $c_0$ on untouched confirmation items under Benjamini-Hochberg FDR control ($\alpha = 0.05$).
  - **H4 (Uplift Efficiency):** Selected prompt captures $\ge 95\%$ of exhaustive discovery-oracle uplift when oracle uplift exceeds SESOI (reported as `NaN` if uplift $\le$ SESOI).
  - **H5 (Resource Efficiency):** Consumes 75% fewer configurations than full factorial (16 vs 64), with empirical call, token, latency, and monetary cost quantified.

---

## 2. Experimental Sampling Plan

- **Tasks (3):**
  1. `schema_extraction`: Micro-averaged field F1 (SESOI = 0.030)
  2. `grounded_qa`: Token F1 with abstention accuracy (SESOI = 0.025)
  3. `intent_classification`: Classification accuracy (SESOI = 0.020)
- **Models (3):**
  1. Open-weight: `meta-llama/Llama-3.1-8B-Instruct` (2 replicates)
  2. Proprietary economical: `gpt-4o-mini-2024-07-18` (1 replicate)
  3. Multi-family alternative: `claude-3-5-haiku-20241022` (1 replicate)
- **Partitions:**
  - 80 discovery items per task (fixed partition, sealed from confirmation)
  - 80 confirmation items per task (locked, held out until recommendations frozen)
  - 20 pilot items per task (quarantined from confirmatory evaluations)
- **Total Call Budget:**
  $$3\text{ tasks} \times 64\text{ configurations} \times 160\text{ items} \times 4\text{ replicate-weights} = 122{,}880\text{ calls}$$

---

## 3. Stopping Rule & Interim Decisions

- **Stopping Rule:** Complete fixed-sample collection of the 122,880 calls.
- **No Sequential Peek:** No data-dependent stopping, early termination, or configuration alteration is permitted.
- **Missingness Handling:** Network timeouts or provider rate limits retry up to 5 times with exponential backoff; unresolvable failures are recorded with `error` provenance and imputed as 0 score.

---

## 4. Preregistration Artifact Verification

Prior to unlocking confirmation data, SHA-256 digests of all configuration files, dataset manifests, and analysis scripts are frozen into `research/protocol/real_v2/artifact_hashes.json`.
