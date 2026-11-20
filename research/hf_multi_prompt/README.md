# Multi-Model Factorial Prompt Engineering Study (`PromptEval` + `SLAB-NLP`)

This directory contains the reproducible dataset cache, constructed factorial contrast pipeline, statistical sensitivity analyses, and figure/table generators for the standalone paper:

> **Separating Attribution, Noise Floors, and Selection in Multi-Prompt LLM Benchmarks: An Empirical Design-of-Experiments and Calibration Audit Across 1,033 Benchmark Cubes** (`paper_hf/main.pdf`)

## Data Provenance & Integrity

- **Primary Matrices (`data/Ys.pickle`)**:
  - SHA-256: `4300e539e82f9c9f3f3ba9b4ad024ec0335f1c7b8914a7a43554055e4506416a`
  - Contains complete per-item binary correctness matrices across $7{,}006{,}016$ analyzed cell–item observations:
    - **Cohort 1 (`MMLU`)**: 57 subjects $\times$ 15 open-weights LLMs $\times$ 100 FormatSpread templates $\times$ 14,042 items (`PromptEval/PromptEval_MMLU_full` & `PromptEval/PromptEval_MMLU_correctness`), mapped into 855 complete $2^5=32$ factorial contrast cubes ($6{,}740{,}160$ cell–item observations).
    - **Cohort 2 (`SLAB-NLP` `BBH` & `LMentry`)**: Completeness-selected subset of $6/15$ `BBH` tasks ($\times$ 11 LLMs $= 66$ cubes; $105{,}600$ cell–item observations) and $7/10$ `LMentry` tasks ($\times$ 16 LLMs $= 112$ cubes; $160{,}256$ cell–item observations across six 100-item tasks and one 26-item task, `word_not_containing`), yielding 178 complete $2^4=16$ factorial contrast cubes ($265{,}856$ cell–item observations; 12 incomplete tasks with 7–15 populated cells are documented in `headline_summary.json`).
- **Template Catalogs (`data/templates/`)**:
  - `mmlu_templates_metadata.json` (SHA-256: `fc3b3834c91a3dfa714283c477acce9a52d224ca4246bb11c756b74f340401f8`)
  - Task-level prompt template CSVs in `data/templates/MMLU/`, `data/templates/BBH/`, and `data/templates/LMentry/`; all 83 included catalog-file hashes and their aggregate digest are recorded and verified in `results/summary_manifest.json`.

## V4 Estimand & Reporting Contract

1. **Attribution vs. Selection Comparators**:
   - **Attribution (`design-validity`)**: Fitted OLS main-effect and second-order-product models use factor-balanced, full-rank random comparators (`rand16_attr_2fi_fullrank` with $\mathrm{rank}(D_{\mathrm{2FI}})=16$, `rand8_attr_main_fullrank` with $\mathrm{rank}(D_{\mathrm{main}})=k+1$). In MMLU, five bit contrasts encode three catalog axes: A/B derive from `op`, C from `space`, and D/E from `sep`. AB and DE are within-axis categorical products, and bit-product order need not equal interaction order among underlying axes. Lower-budget one-factor-at-a-time (`OFAT`, $k+1$ runs) additively extrapolates from reference Cell 0 and is reported separately.
   - **Selection (`ScreenBest`)**: Non-parametric `ScreenBest` uses **unconditioned** factor-balanced random comparators (`rand16_screenbest_unconditioned`, `rand8_screenbest_unconditioned`) without filtering on 2FI regression rank (while retaining 2FI-conditioned random `ScreenBest` as a diagnostic).
2. **Tie Policy & Reference-Corner Sensitivity**:
   - **Primary Tie Estimand (`mean_tied`)**: Discovery ties within $\epsilon = 10^{-12}$ average confirmation accuracy across all tied discovery winners $\mathcal{W}$. This is the expected result of uniform random selection among tied winners, not a deterministic deployable prompt. `first_index` and `seeded_random` are retained as sensitivity policies; deployment requires a predeclared reproducible tie rule.
   - **Reference Corner (`Cell 0`)**: `Cell 0` is treated as a reference corner rather than an externally privileged untuned prompt. Baseline-free confirmation accuracy and confirmation regret vs. the full-cube discovery oracle are primary, supplemented by Ratio-of-Means ($\mathrm{RoM}$) recovery across all $K \in \{32, 16\}$ reference cells and positive-denominator cube fractions.
3. **Template Deduplication & Crossed Split Sensitivity**:
   - Full SHA-256 template hashes (`template_sha256_full`) identify duplicate catalog templates (7 per task in MMLU; 93 across LMentry tasks). Both unique-template-weighted (`cell_mean_unique`) and catalog-row-weighted (`cell_mean`) stratum averages are computed, and 20 random single-template draws are crossed with all 20 repeated 50/50 item splits ($20 \times 20 = 400$ evaluations per cube).
4. **Conditional Item Inference, Shared-Model Panel Resampling, and Non-Equivalence Reporting**:
   - Models, tasks, and templates are fixed; benchmark items are sampling units for the item-level contrast $t$-tests (`scipy.stats.t.sf`, with zero-variance contrasts assigned $p=1$). Their asymptotic interpretation requires independent or sufficiently weakly dependent item contrasts and a studentized CLT. If items are treated as a fixed finite set, the $p$-values are model-based diagnostics. The 200-draw Rademacher sign-flip tests ($\min p = 1/201 \approx 0.00498$) additionally require joint sign-vector invariance or a randomized sharp null.
   - Two-way model–task resampling preserves shared `model_id` clusters jointly across `BBH` and `LMentry` in Cohort 2 and is interpreted only as conditional reweighting sensitivity for the observed finite panel, not population inference.
   - Paired `ScreenBest` differences are reported with conditional panel-reweighting intervals. Normal-Wald calculations across explicit margins $\Delta \in \{0.05, 0.10, 0.25, 0.50, 1.00\}\text{ pp}$ are nonconfirmatory sensitivity summaries and do not establish equivalence.

## Generated Artifacts (`results/`, `paper_hf/figures/`, `paper_hf/tables/`)

Running `run_experiments.py` verifies source SHA-256 hashes and deterministically generates:

### Summary & Claim Traceability Manifests
- `results/headline_summary.json`: Complete machine-readable summary of all headline metrics across estimators (`legacy_first`, `cell_mean`, `canonical_first`, and 20 `random_single_rep` draws), cosets, 20 repeated random 50/50 item splits, conditional model–task panel-reweighting intervals, nonconfirmatory normal-Wald margin sensitivity, `study_observation_totals` ($7{,}006{,}016$ total cell–item obs.), `cell0_diagnostic` (row 65 versus a five-row/four-unique-template Cell-0 stratum), and `cohort2_completeness_audit` ($6/15$ BBH and $7/10$ LMentry complete 16-cell tasks).
- `results/claim_evidence_map.csv`: Structured mapping from every headline claim (`C1`–`C12`) in `paper_hf/main.tex` to its exact source file, field, estimand, averaging unit, and uncertainty method.
- `results/effect_reliability.csv` & `results/effect_reliability_summary.json` (`effect_reliability.py`): Split-half contrast reliability, item-sampling variance vs. noise-corrected RMS, and attribution/held-out Error-to-Signal Ratios (ESR).
- `results/mixed_level_results.csv` & `results/mixed_level_summary.json` (`mixed_level_analysis.py`): Audit of the `89/396` observed MMLU `(op, space, sep)` combinations, outcome-blind A-/D-optimal subsets (`B=16, 24, 32`), and equal-budget PromptEval 1PL Rasch comparisons.
- `results/inference_calibration.csv` & `results/inference_calibration_summary.json` (`inference_calibration.py`): Synthetic and empirically anchored null Type-I error and BH/BY FDR calibration across Unblocked ANOVA, Item-Clustered $t$-tests, Rademacher Sign-Flip (`B=200` and `B=2000`), and Pooled Fixed-Block RCBD.
- `results/summary_manifest.json`: Full provenance manifest including Python/library versions, fixed random seeds, design generator rejection counts, `ANOVAEngine` verification, all 83 template-catalog-file hashes, analysis-defining source hashes, and SHA-256 digests of all output CSV/JSON files.
- `paper_hf/release_manifest.json`: Post-build SHA-256 release manifest for the final PDF, summary manifest, and PromptPrism Python sources.

### Primary & Sensitivity CSV Artifacts
- `results/design_diagnostics.csv`: Matrix rank (`rank_main`, `rank_2fi`), condition number (`cond_main`, `cond_2fi`), and acceptance status for all deterministic fractional cosets, all 60 regular 8-run $2^{5-2}_{\mathrm{III}}$ fractions (`2(5-2)III_all60`), 200 full-rank balanced `Rand-16` designs (`rank(D_2fi) == 16`), 20 legacy `Rand-16` designs (showing 12/20 rank-deficient in the 16-parameter 2FI space), and 200 full-rank `Rand-8` designs.
- `results/coset_results.csv`: Effect attribution RMSE and confirmation selection accuracy across both cosets of $2^{5-1}_{\mathrm{V}}$ ($I=\pm ABCDE$), all 4 canonical cosets and all 60 regular 8-run orientations of $2^{5-2}_{\mathrm{III}}$ ($1.53\times$ all-60 vs. $1.59\times$ 4-coset main-effect RMSE gain), and both cosets of $2^{4-1}_{\mathrm{IV}}$ ($I=\pm ABCD$).
- `results/split_selection_results.csv`: Held-out confirmation accuracy and Oracle Uplift Recovery across the legacy even/odd item split (`split 0`) and 20 seeded random 50/50 discovery/confirmation splits (`split_00`..`split_19`).
- `results/template_mapping.csv`: Complete template-to-cell mapping manifest for all 100 MMLU templates and all SLAB-NLP task templates, including full SHA-256 text hashes (`template_sha256_full`, with a convenience 16-character prefix), factor levels, and canonical vs. legacy within-cell ranks.
- `results/template_sensitivity.csv`: Comparison of single-template estimators (`legacy_first`, `canonical_first`, 20 `random_single_rep` draws) and multi-template stratum sensitivity (`cell_mean`, which averages 1–6 MMLU or 1–56 SLAB templates per cell).
- `results/factor_audit.csv`: Construct-validity audit of all binary factors in Cohort 1 and Cohort 2, documenting exact executable rules, template counts per level, and length/lexical co-variation.
- `results/interaction_magnitudes.csv`: Exact full-cube orthogonal Fourier decomposition (`rms_main`, `rms_2fi`, `rms_3fi`, `rms_4fi`, variance shares, reconstruction error) and additive-sign regret per cube.
- `results/rcbd_robustness.csv`: Per-factor test statistics and $p$-values comparing Unblocked ANOVA, primary asymptotic Item-Clustered contrast $t$-tests, 200-draw Rademacher sign-flip permutation tests, and exploratory additive Fixed-Block RCBD (`block_by="sample_id"`).
- `results/multiplicity_results.csv`: Benjamini–Hochberg (BH) and Benjamini–Yekutieli (BY) adjusted rejection counts ($\alpha=0.05$) across `per_cube_bh`, `per_model_bh`, and `global_cohort_bh` families. Counts are conditional diagnostics, not statistical power estimates or an aggregate FDR guarantee across analyses.
- `results/cohort1_mmlu_models.csv`, `results/cohort1_mmlu_cubes.csv`, `results/cohort2_slab_cubes.csv`: Model-level and cube-level summary tables.

## Reproducing All Experiments, Tests, Figures, and Paper PDF

1. **Run the complete experiment and sensitivity suite**:
   ```bash
   /Users/roberto/proyectods/prompt_prism/.venv/bin/python research/hf_multi_prompt/run_experiments.py
   ```

2. **Run the unit, synthetic-statistical, and artifact regression test suite**:
   ```bash
   /Users/roberto/proyectods/prompt_prism/.venv/bin/python -m pytest -p no:rerunfailures tests/research/test_hf_multi_prompt.py -q
   ```

3. **Compile the standalone paper PDF**:
   ```bash
   cd paper_hf && /opt/homebrew/bin/tectonic main.tex
   ```

4. **Write the post-build release manifest**:
   ```bash
   /Users/roberto/proyectods/prompt_prism/.venv/bin/python research/hf_multi_prompt/build_release_manifest.py
   ```
