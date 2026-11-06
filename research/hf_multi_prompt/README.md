# Multi-Model Factorial Prompt Engineering Study (`PromptEval` + `SLAB-NLP`)

This directory contains the reproducible dataset cache, factorial cube extraction pipeline, statistical analysis, and figure/table generators for the standalone paper:

> **Factorial Prompt Engineering at Scale: Evaluating Fractional Designs, Epistatic Interactions, and RCBD Blocking Across 1,033 Multi-Prompt LLM Benchmark Cubes** (`paper_hf/main.pdf`)

## Data Provenance & Integrity

- **Primary Matrices (`data/Ys.pickle`)**:
  - SHA-256: `4300e539e82f9c9f3f3ba9b4ad024ec0335f1c7b8914a7a43554055e4506416a`
  - Contains complete per-item binary correctness matrices across:
    - **Cohort 1 (`MMLU`)**: 57 subjects $\times$ 15 open-weights LLMs $\times$ 100 FormatSpread templates $\times$ 14,042 items (`PromptEval/PromptEval_MMLU_full` & `PromptEval/PromptEval_MMLU_correctness`).
    - **Cohort 2 (`SLAB-NLP` `BBH` & `LMentry`)**: 15 `BBH` tasks ($\times$ 11 LLMs $\times$ ~187 templates) and 10 `LMentry` tasks ($\times$ 16 LLMs $\times$ ~258 templates), yielding 178 complete $2^4=16$ factorial cubes across 6 `BBH` tasks and 7 `LMentry` tasks.
- **Template Catalogs (`data/templates/`)**:
  - `mmlu_templates_metadata.json` (SHA-256: `fc3b3834c91a3dfa714283c477acce9a52d224ca4246bb11c756b74f340401f8`)
  - Task-level prompt template CSVs in `data/templates/MMLU/`, `data/templates/BBH/`, and `data/templates/LMentry/`.

## Reproducing All Experiments, Figures, and Paper PDF

1. **Run the experiment and analysis suite** (executes Cohort 1 $2^5=32$ and Cohort 2 $2^4=16$ evaluations, verifies `prompt_prism.ANOVAEngine` & `OptimalPromptFinder`, writes CSV outputs to `results/`, and regenerates `paper_hf/figures/` and `paper_hf/tables/`):
   ```bash
   /Users/roberto/proyectods/prompt_prism/.venv/bin/python research/hf_multi_prompt/run_experiments.py
   ```

2. **Compile the standalone paper PDF**:
   ```bash
   cd paper_hf && /opt/homebrew/bin/tectonic main.tex
   ```
