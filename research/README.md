# PromptPrism Research Package: Scientific Validation

This directory contains the protocol, configurations, data manifests, simulation infrastructure, experiment runners, and analysis pipelines for the empirical validation study of PromptPrism:

**Paper Working Title:** *Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization*  
**Core Reference Plan:** [`docs/research/promptprism-validation-plan.md`](../docs/research/promptprism-validation-plan.md)  
**Execution Tasks:** [`docs/research/promptprism-validation-tasks.md`](../docs/research/promptprism-validation-tasks.md)

---

## 1. Directory Structure

```text
research/
├── README.md               # This navigation & governance guide
├── configs/                # Experiment configurations
│   ├── real_v2/            # Frozen Real-LLM v2 study configurations (122,880 calls)
│   │   ├── experiment.yaml # Matrix, protocol, call budgets, paths
│   │   ├── tasks.yaml      # Extraction, QA, and Intent task definitions
│   │   ├── models.yaml     # Frozen snapshots (Llama 3.1 8B, GPT-4o-mini, Claude 3.5 Haiku)
│   │   └── factors.yaml    # Six binary prompt factors (A-F, 2^(6-2)_IV)
│   ├── factors.yaml        # V1 Factor catalog
│   ├── tasks.yaml          # V1 Task schemas
│   └── simulation.yaml     # DGP parameters & scenario grids
├── protocol/               # Preregistration, SAP, decisions, deviations, COI
│   ├── preregistration.md  # Preregistered protocol & frozen hypotheses
│   ├── statistical_analysis_plan.md # Formal estimands & statistical models
│   ├── decisions.md        # Architecture & methodology decision log
│   ├── deviations.md       # Pre- and post-data collection deviation log (DEV-001, DEV-002)
│   └── coi_statement.md    # Conflict of interest & author disclosures
├── data/                   # Data manifests, split IDs, and licenses
│   ├── real_v2/            # Licensed real public datasets for v2 study
│   ├── dataset_manifest.yaml # Datasets, provenance, splits, & licenses
│   └── LICENSES.md         # License terms and redistribution permissions
├── results/                # Versioned machine-readable outputs
│   ├── real_v2/            # Real-LLM study empirical outputs (data_origin: real)
│   └── confirmatory_*      # Quarantined Study v1 simulated fixtures (data_origin: simulated)

```

---

## 2. Artifact Storage and Versioning Rules

To ensure computational reproducibility without bloating the repository or leaking proprietary/restricted data:

| Tier | Category | Storage Location | Versioning / Retention Rule | Git Status |
|---|---|---|---|---|
| **Tier 1** | **Code & Configs** | `research/configs/`, `research/protocol/`, `research/analysis/` | Frozen by Git commit SHA and tag at Checkpoint C. | **Tracked** |
| **Tier 2** | **Manifests & Metadata** | `research/data/*.yaml`, `research/results/*.json` | Immutable JSON/YAML containing checksums (SHA-256), seeds, token counts, and hyperparameters. | **Tracked** |
| **Tier 3** | **Aggregate Results** | `research/results/*.csv` | Machine-readable tabular summaries consumed directly by paper tables/figures. | **Tracked** |
| **Tier 4** | **Raw Model Responses** | `research/data/raw/` (or external Zenodo/OSF store) | Content-addressed JSONL by input prompt hash. Never committed to GitHub. Archived to long-term storage with a DOI upon publication. | **Git-Ignored** |
| **Tier 5** | **Restricted / Licensed Data** | External / Secured Cache | Downloaded on demand via reproducible download script using official APIs/URLs. | **Git-Ignored** |

---

## 3. Governance and Named Owners

All modifications to protocols, decisions, and deviations require sign-off from named owners:

| Role | Primary Responsibility | Designated Owner |
|---|---|---|
| **Principal Investigator / Methodology Lead** | Experimental design, hypothesis formulation, literature review integrity | Roberto Sanchez (`@robsanmx`) |
| **Statistical Lead** | Statistical analysis plan, equivalence margins ($\delta$), Monte Carlo verification, power analysis | Statistical Reviewer / Lead Analyst |
| **Systems & Infrastructure Lead** | Execution pipeline, content-addressed caching, provenance manifests, reproducibility | Infrastructure Lead |
| **Independent Reviewer (Adversarial)** | Pre-registration audit, leakage verification, code reproduction | External / Independent Auditor |

Logs and Statements:
- **Decision Log**: [`research/protocol/decisions.md`](protocol/decisions.md) (Maintained by Methodology Lead)
- **Deviation Log**: [`research/protocol/deviations.md`](protocol/deviations.md) (Maintained by Methodology Lead & Systems Lead)
- **Conflict of Interest Statement**: [`research/protocol/coi_statement.md`](protocol/coi_statement.md) (Maintained by All Authors)

---

## 4. Reproducibility Entry Points

1. **Single-Command Verification (Checkpoint E)**:
   ```bash
   python research/reproduce.py
   ```
2. **Protocol & SAP**: Read [`research/protocol/preregistration.md`](protocol/preregistration.md) and [`research/protocol/statistical_analysis_plan.md`](protocol/statistical_analysis_plan.md).
3. **Simulation Calibration (Study A)**:
   ```bash
   python -m research.simulation.run --config research/configs/simulation.yaml
   ```
4. **Confirmatory Replay & Analysis (Studies B & C)**:
   ```bash
   python -m research.analysis.run_confirmatory --input research/results/confirmatory_manifest.jsonl
   ```
5. **End-to-End Test & Verification**:
   ```bash
   pytest tests/research/
   ```
