# Statistical Analysis Plan (SAP)

**Title:** Statistical Analysis Plan for PromptPrism Scientific Validation  
**Lead Statistician / Author:** Roberto Sanchez (`@robsanmx`)  
**Protocol Reference:** [`research/protocol/preregistration.md`](preregistration.md)  
**Status:** Frozen Pre-Data Collection  

---

## 1. Mathematical Estimands

Let $c \in \{0, 1\}^k$ denote a prompt configuration of $k$ binary factors, $i \in \{1, \dots, I\}$ denote a benchmark item, and $r \in \{1, \dots, R\}$ denote a generation replicate. Let $Y(c, i, r)$ denote the observed performance score.

### 1.1 Marginal Main Effect ($\tau_j$)
For factor $j \in \{1, \dots, k\}$, the population marginal main effect is defined as the expectation over uniform distribution across all $2^{k-1}$ combinations of other factors and over benchmark items:
$$\tau_j = \mathbb{E}_{X_{-j}, i, r}\left[ Y(X_j = 1, X_{-j}, i, r) \right] - \mathbb{E}_{X_{-j}, i, r}\left[ Y(X_j = 0, X_{-j}, i, r) \right]$$

In an exhaustive full-factorial sample, the oracle contrast estimator is:
$$\hat{\tau}_{j, \text{oracle}} = \frac{1}{2^{k-1} I R} \sum_{c: c_j=1} \sum_{i=1}^I \sum_{r=1}^R Y(c, i, r) - \frac{1}{2^{k-1} I R} \sum_{c: c_j=0} \sum_{i=1}^I \sum_{r=1}^R Y(c, i, r)$$

In an orthogonal fractional factorial design with run count $N$, using $\pm 1$ contrast coding $W_{cj} = 2c_j - 1$:
$$\hat{\tau}_{j, \text{frac}} = \frac{2}{N I R} \sum_{c \in \mathcal{D}} \sum_{i=1}^I \sum_{r=1}^R W_{cj} Y(c, i, r)$$

### 1.2 Prompt-Selection Regret
Let $I_{\text{disc}}$ and $I_{\text{conf}}$ be disjoint discovery and confirmation sets of items.
Let $c_{\text{full}}$ be the configuration chosen by exhaustive search on $I_{\text{disc}}$:
$$c_{\text{full}} = \arg\max_{c \in \{0, 1\}^k} \frac{1}{|I_{\text{disc}}| R} \sum_{i \in I_{\text{disc}}} \sum_{r=1}^R Y(c, i, r)$$
Let $c_{\text{method}}$ be the configuration recommended by an evaluation method (e.g. PromptPrism $2^{6-2}_{\text{IV}}$ or budget-matched Random Search) evaluated on $I_{\text{disc}}$.

The primary held-out selection regret is evaluated on untouched confirmation items $I_{\text{conf}}$:
$$\text{Regret}(c_{\text{method}}) = \bar{Y}_{\text{conf}}(c_{\text{full}}) - \bar{Y}_{\text{conf}}(c_{\text{method}})$$
where $\bar{Y}_{\text{conf}}(c) = \frac{1}{|I_{\text{conf}}| R} \sum_{i \in I_{\text{conf}}} \sum_{r=1}^R Y(c, i, r)$.
*(For metrics to minimize, e.g. latency or error rate, the subtraction order is inverted).*

### 1.3 Proportion of Oracle Uplift Captured
When $c_{\text{full}}$ achieves positive uplift over baseline $c_{\text{base}} = (0, \dots, 0)$:
$$\text{UpliftRatio}(c_{\text{method}}) = \frac{\bar{Y}_{\text{conf}}(c_{\text{method}}) - \bar{Y}_{\text{conf}}(c_{\text{base}})}{\bar{Y}_{\text{conf}}(c_{\text{full}}) - \bar{Y}_{\text{conf}}(c_{\text{base}})}$$

### 1.4 Practical Neutrality via Equivalence Testing
Let $\delta > 0$ denote the preregistered Smallest Effect Size of Interest (SESOI). A factor $j$ is deemed **Practically Neutral** if and only if the Two One-Sided Tests (TOST) $90\%$ confidence interval $[\hat{\tau}_{j, \text{lower}}, \hat{\tau}_{j, \text{upper}}]$ is entirely contained within $[-\delta, +\delta]$:
$$-\delta < \hat{\tau}_{j, \text{lower}} \quad \text{and} \quad \hat{\tau}_{j, \text{upper}} < \delta$$
A failure to reject the standard null ($p \ge 0.05$) without equivalence is classified as **Inconclusive**.

### 1.5 Cost Accounting Metrics
Cost is quantified along four independent dimensions:
1. **Evaluated Configurations ($N$):** Count of distinct prompt templates tested.
2. **Total Model Calls:**
   $$\text{Calls}_{\text{gen}} = N \times I \times R, \quad \text{Calls}_{\text{judge}} = \text{Calls}_{\text{gen}} \times M_{\text{judge}} \times R_{\text{judge}}$$
3. **Token & Dollar Expenditure:** Total input tokens, output tokens, and cash cost computed from frozen provider pricing manifests.
4. **Wall-clock Latency:** Total execution time under a declared concurrency limit ($K_{\text{workers}} = 4$).

---

## 2. Statistical Models and Inference

### 2.1 Tool-Native Model (Randomized Complete Block Design OLS)
For fast internal reporting, PromptPrism fits a fixed-effects ANOVA model blocking on item ID:
$$Y_{cir} = \mu + \sum_{j=1}^k \beta_j W_{cj} + \sum_{j < l} \gamma_{jl} (W_{cj} W_{cl}) + \alpha_i + \epsilon_{cir}$$
where $\alpha_i$ represents fixed item blocks and $\epsilon_{cir} \sim \mathcal{N}(0, \sigma^2)$.

### 2.2 Confirmatory Inferential Model (Item-Clustered Mixed-Effects)
To ensure valid generalization across benchmark items, confirmatory inference uses a linear mixed-effects model with random item intercepts:
$$Y_{cir} = \mu + \sum_{j=1}^k \beta_j W_{cj} + u_i + \epsilon_{cir}, \quad u_i \sim \mathcal{N}(0, \sigma_u^2), \quad \epsilon_{cir} \sim \mathcal{N}(0, \sigma_\epsilon^2)$$
Standard errors and Wald confidence intervals are compared against the cluster bootstrap.

### 2.3 Repeated-Measures Item Cluster Bootstrap
For non-normal, bounded, or F1 metrics, uncertainty is estimated via cluster bootstrap:
1. Resample $I$ benchmark items with replacement.
2. For each bootstrap resample $b \in \{1, \dots, B\}$ ($B = 2{,}000$), extract all configurations and replicates for the sampled items.
3. Compute $\hat{\tau}_j^{(b)}$ and $\text{Regret}^{(b)}$.
4. Compute empirical percentiles ($2.5\%$, $97.5\%$) for 95% confidence intervals.

---

## 3. Factor Classification Taxonomy

Each screened factor $j$ is classified into exactly one mutually exclusive category:
1. 🟢 **Beneficial:** Adjusted $p_{\text{BH}} < 0.05$ and $\hat{\tau}_j > \delta$. (Recommendation: Set Level 1).
2. 🔴 **Harmful:** Adjusted $p_{\text{BH}} < 0.05$ and $\hat{\tau}_j < -\delta$. (Recommendation: Set Level 0).
3. ⚪ **Practically Neutral:** $90\%$ CI $\subset [-\delta, +\delta]$. (Recommendation: Set Level 0 to minimize tokens/latency without loss of quality).
4. ❓ **Inconclusive:** Neither beneficial, harmful, nor practically neutral. (Recommendation: Default to Level 0, labeled as an empirical trade-off under limited power).

---

## 4. Separation of Confirmatory and Exploratory Analyses

* **Confirmatory:**
  - Primary Hypothesis H1 (paired regret test across task-model cells).
  - Main effect estimation accuracy and interval coverage (H2, H3).
  - Prespecified primary metrics per task stratum.
* **Exploratory:**
  - Unregistered two-factor interaction terms.
  - Subgroup analyses by item length or difficulty quintiles.
  - LLM judge evaluations on open-ended creative tasks.
  - Post-hoc comparisons of optimizer variants.
