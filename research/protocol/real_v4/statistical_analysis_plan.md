# Statistical Analysis Plan (SAP): PromptPrism Real-v4

- **Study Identifier:** `prompt-prism-real-v4-selection-stability-2026`
- **SAP Version:** `1.0.0`
- **Bootstrap Replicates:** `B = 2,000` (fixed primary seed `20261118`)
- **Primary Significance Level:** One-sided $\alpha = 0.05$ per primary hypothesis component under an intersection-union framework

---

## 1. Units of Analysis, Stratum Weights, and Pairing Structure

1. **Primary Sampling Unit Within Tasks:** Whole source item groups (`group_id`: document, passage, intent cluster, problem stem, programming task, or support concept). Resampling always draws whole `group_id` clusters with replacement within each task stratum $t \in \{1, \dots, 6\}$ so items sharing a source group never split across bootstrap draws.
2. **Crossed Task-by-Model Dependence Preservation:** The 6 tasks and 3 model conditions (`gemini-2.5-flash`, `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, `mlx-community/Qwen3-8B-4bit`) are treated as fixed primary strata defining $|C| = 18$ cells with **equal weight $1/18$ per cell**. Within task $t$, a bootstrap draw of item groups $\mathcal{G}_t^{*(b)}$ carries **all** 3 models, **all** primary and secondary methods, **all** 10 random/Bayesian baseline trajectories, **all** registered perturbation environments $e \in E_{\text{all}}$, and **all** execution repeats together. No primary analysis treats the 18 crossed cells or the $|E_{\text{all}}|$ environments as independent observations.
3. **Stochastic Baseline Aggregation:** Both `balanced_random_v4` and `bayesian_opt_v4` execute $M_{\text{traj}} = 10$ pre-frozen seeds (matching `calibration.yaml`). For item $i$ in cell $c$ and environment $e$, comparator $b$'s primary paired score is the expectation over the 10 pre-frozen trajectories:
   $$
   Y_{b,c,e,i} = \frac{1}{10} \sum_{r=1}^{10} Y_{b^{(r)},c,e,i}.
   $$
   The full distribution across the 10 individual trajectories (min, 5th, median, 95th, max, and per-seed win rate) is reported alongside the mean.

---

## 2. Primary Inference Procedures

### 2.1 Claim P: Selection Superiority (`G_select,b`)

For locked final prompts evaluated on `clean_confirmation` (`env_ref`):
1. Compute observed cell means $\hat{\Delta}_{b,c} = \frac{1}{N_{c,\text{conf}}} \sum_{i=1}^{N_{c,\text{conf}}} (Y_{\text{promptprism},c,\text{ref},i} - Y_{b,c,\text{ref},i})$ and equal-weighted aggregate $\hat{G}_{\text{select},b} = \frac{1}{18}\sum_{c=1}^{18} \hat{\Delta}_{b,c}$ for each $b \in \{\text{balanced\_random}, \text{bayesian\_opt}\}$.
2. Across $B = 2,000$ group-paired within-task bootstrap replicates, compute $\hat{G}_{\text{select},b}^{*(r)}$.
3. Compute the one-sided 95% percentile lower confidence bound $L_{95}(G_{\text{select},b}) = Q_{0.05}(\{\hat{G}_{\text{select},b}^{*(r)}\}_{r=1}^B)$ and finite-sample Monte Carlo one-sided $p$-value:
   $$
   p_{\text{select},b} = \frac{1 + \sum_{r=1}^B \mathbb{I}(\hat{G}_{\text{select},b}^{*(r)} - \hat{G}_{\text{select},b} \le -\hat{G}_{\text{select},b})}{B + 1} \quad \text{(paired centered test)}
   $$
   alongside the direct percentile check $L_{95}(G_{\text{select},b}) > 0$.
4. **Decision Rule:**
   - Statistical superiority passes iff $L_{95}(G_{\text{select},\text{balanced\_random}}) > 0$ **and** $L_{95}(G_{\text{select},\text{bayesian\_opt}}) > 0$.
   - Practical superiority passes iff $L_{95}(G_{\text{select},\text{balanced\_random}}) > \delta_{\text{select}}$ **and** $L_{95}(G_{\text{select},\text{bayesian\_opt}}) > \delta_{\text{select}}$ ($\delta_{\text{select}} = 0.01$).
5. **Nested Adaptive Resampling Sensitivity:** Because final prompts are adaptively chosen via screen + validation, a secondary nested-bootstrap analysis re-runs candidate generation and robust validation inside resampled screen/validation groups simultaneously with confirmation resampling, verifying coverage of the unconditional adaptive procedure.

### 2.2 Claim S: Simultaneous Maximum-Drop Noninferiority & Worst-Environment Check

1. **Envelope Pre-Check (Fail-Closed):** Verify that every registered environment $e \in E_{\text{all}}$ is present across all 18 cells, has complete paired records, matches registered prompt/perturbation hashes, and exhibits zero unquarantined provider `model_version` changes. Any violation terminates with `certificate_status = INVALID_OUT_OF_ENVELOPE` (`claim_s_pass = False`). Note: `item_holdout_group` uses its own group-disjoint item set (`stability_confirmation`); within task $t$, bootstrap replicate $r$ resamples `clean_confirmation` groups and `stability_confirmation` groups independently while keeping models and methods strictly paired within each item group!
2. **Simultaneous Max-Drop Upper Confidence Bound ($U_{95,\text{sim}}(D_{\text{stability}})$):**
   - For each perturbation environment $e \in E_{\text{reg}}$, let $\hat{d}_e = \hat{\mu}_{\text{promptprism},\text{ref}} - \hat{\mu}_{\text{promptprism},e}$ and $\hat{D}_{\text{stability}} = \max_{e \in E_{\text{reg}}} \hat{d}_e$.
   - To avoid upward max-bias distortion and guarantee simultaneous familywise noninferiority coverage across all $|E_{\text{reg}}|$ environments at level $1 - \alpha = 0.95$:
     Compute bootstrap environment differences $\hat{d}_e^{*(r)}$ ($r = 1, \dots, B$), standard errors $\hat{s}_e = \text{SD}_r(\hat{d}_e^{*(r)})$ (floored at $10^{-12}$), and the simultaneous studentized maximum pivot:
     $$
     T_{\max}^{*(r)} = \max_{e \in E_{\text{reg}}} \frac{\hat{d}_e^{*(r)} - \hat{d}_e}{\hat{s}_e}.
     $$
     Let $q_{0.95} = Q_{0.95}(\{T_{\max}^{*(r)}\}_{r=1}^B)$ be the 95th percentile of the simultaneous studentized maximum distribution. Then the simultaneous one-sided 95% upper confidence bound on the worst-case drop is:
     $$
     U_{95,\text{sim}}(D_{\text{stability}}) = \max_{e \in E_{\text{reg}}} \left( \hat{d}_e + q_{0.95} \hat{s}_e \right).
     $$
     (When all $\hat{s}_e = 0$ identically on noiseless deterministic fixtures, $U_{95,\text{sim}}(D_{\text{stability}}) = \hat{D}_{\text{stability}}$.)
   - **Pass Criterion 1:** $U_{95,\text{sim}}(D_{\text{stability}}) < \epsilon_{\text{stability}}$ (strictly below calibrated frozen $\epsilon_{\text{stability}} = 0.02$).
3. **Worst-Environment Comparator Noninferiority ($G_{\text{robust},b}$):**
   - Compute $\hat{G}_{\text{robust},b} = \min_{e \in E_{\text{all}}} \hat{\mu}_{\text{promptprism},e} - \min_{e \in E_{\text{all}}} \hat{\mu}_{b,e}$ and its bootstrap replicates $\hat{G}_{\text{robust},b}^{*(r)}$.
   - Let $L_{95}(G_{\text{robust},b}) = Q_{0.05}(\{\hat{G}_{\text{robust},b}^{*(r)}\}_{r=1}^B)$.
   - **Pass Criterion 2:** $L_{95}(G_{\text{robust},\text{balanced\_random}}) > -\epsilon_{\text{stability}}$ **and** $L_{95}(G_{\text{robust},\text{bayesian\_opt}}) > -\epsilon_{\text{stability}}$.

---

## 3. Tie, Missing-Data, Terminal-Failure, and Drift Rules

1. **Ties:** Deterministic lowest `run_id` (`0..63`); zero factor contrast maps to `-1` (`0`).
2. **Missing Data vs. Terminal Execution Failure:**
   - A documented provider terminal failure (e.g. `MAX_TOKENS`, content filter, unparseable output after bounded retry) is recorded in the manifest with `score = 0.0` and kept in the denominator.
   - An *unexecuted* or *unpaired* expected `(cell, method, environment, item, repeat)` record is a **missing-cell integrity error** and fails closed (`INVALID_OUT_OF_ENVELOPE`).
3. **Temporal Provider Version Drift:** If `temporal_wave_2` returns a hosted `model_version` string differing from the locked `env_ref` `model_version`, that model's temporal wave is classified as an **envelope change** (`PROVIDER_VERSION_DRIFT`), reported in a dedicated drift table, and triggers `INVALID_OUT_OF_ENVELOPE` for cross-wave stability certification rather than silently pooling across distinct model weights.

---

## 4. Supporting, Sensitivity, and Exploratory Outputs

All outputs are emitted into distinct machine-readable keys in `summary.json`:
1. `primary_verdict`: Claim P, Claim S, overall intersection-union verdict, and `stability_certificate.json` state.
2. `cell_level_breakdown`: All 18 task-model cell estimates, intervals, and majority-sign diagnostics ($12/18$ cell consistency check and per-model-family sign check).
3. `sensitivity_analyses`: Leave-one-task-out (`LOTO`), leave-one-model-out (`LOMO`), multiway task-model cluster bootstrap, and nested adaptive selection bootstrap.
4. `supporting_effect_recovery`: 6-factor main-effect RMSE vs. 64-run screening cube (`H_effect`), OFAT-7, and untuned `run_id=0`.
5. `repeatability_and_drift`: Exact string match rate across `ref`/`repeat_r2`/`repeat_r3`, within-item score variance, and `temporal_wave_2` version-audit report.
6. `resource_reconciliation`: Exact operational `BudgetLedger` totals per method vs. research-only 64-run audit overhead, hosted USD spend, and local Qwen wall-time/token/memory telemetry.
