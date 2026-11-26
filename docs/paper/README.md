# Auditing, Pruning, and Stabilizing Compiled Prompts

**Full Title:** *Auditing, Pruning, and Stabilizing Compiled Prompts: Coverage-Aware Equivalence Testing and Perturbation Stability Certification for LLMs*  
**Author:** Roberto Sanchez (PromptPrism Research Group)  
**Date:** November 24, 2026  

[![Read Paper PDF](https://img.shields.io/badge/PDF-Read%20Full%20Paper%20(12%20pages)-b31b1b.svg)](Auditing_Pruning_and_Stabilizing_Compiled_Prompts.pdf)
[![arXiv Source Bundle](https://img.shields.io/badge/arXiv-LaTeX%20Source%20(.tar.gz)-blue.svg)](arxiv_paper2.tar.gz)

---

## Abstract

Prompts for Large Language Models (LLMs) often grow long, costly, and fragile as instructions, few-shot examples, and safety rules accumulate by trial and error. Automatic prompt optimizers such as DSPy `MIPROv2` automate prompt writing, but they typically produce opaque 1,100+ token prompts without proving which pieces actually improve accuracy, whether rare safety rules survive, or whether small formatting changes break performance. We present **PromptPrism 2.1**, an open-source statistical auditing framework that tests prompt components in balanced experiments and certifies prompt stability. Compared directly against DSPy `MIPROv2` on `gemini-2.5-flash` across three benchmarks (Banking77, SQuAD 2.0, and GSM8K; `N = 240` held-out items), PromptPrism delivers three practical benefits:

1. **Safe bloat removal:** Auditing compiled DSPy prompts reveals that DSPy's generated instructions are statistically dead weight on Banking77 and SQuAD 2.0, cutting prompt length by **39%–57%** with matched accuracy and passing stability bounds.
2. **Rare guardrail protection:** A coverage-aware pruning gate cuts accidental deletion of rare safety rules from **18%–95% down to 0%** in calibration and catches **85%** of out-of-scope Banking77 queries where DSPy scores **0%**.
3. **Compact, stable standalone prompts:** Using structured multi-turn XML framing without DSPy or an expensive proposer model, PromptPrism matches or edges DSPy on Banking77 (**96.3% vs. 95.8%**, −39% tokens) and GSM8K (**95.8% vs. 94.6%**, −67% tokens), passes strict perturbation stability across all three tasks, and shows why structural scaffolds must stay fixed when pruning redundant pieces.

---

## Files in This Directory

| File | Description |
|:---|:---|
| [`Auditing_Pruning_and_Stabilizing_Compiled_Prompts.pdf`](Auditing_Pruning_and_Stabilizing_Compiled_Prompts.pdf) | Compiled 12-page paper PDF (2 vector figures, 6 tables, 19 references) |
| [`main.tex`](main.tex) | arXiv-ready `pdfLaTeX` source (`\pdfoutput=1`, pure 7-bit ASCII) |
| [`main.bbl`](main.bbl) | Pre-compiled BibTeX bibliography for arXiv AutoTeX |
| [`references.bib`](references.bib) | Raw BibTeX references |
| [`fig_p2_dspy_comparison.pdf`](fig_p2_dspy_comparison.pdf) | Figure 1: Token cost & accuracy vs. DSPy `MIPROv2` + out-of-scope safety catch rate |
| [`fig_p2_stability_and_pruning.pdf`](fig_p2_stability_and_pruning.pdf) | Figure 2: Perturbation stability certificates (`U95_sim`) + fixing the simultaneous pruning hazard |
| [`arxiv_paper2.tar.gz`](arxiv_paper2.tar.gz) | Flattened arXiv submission source bundle (`main.tex` + `main.bbl` + both figures) |
| [`arxiv_paper2_metadata.txt`](arxiv_paper2_metadata.txt) | Copy-paste arXiv submission metadata (1,638-char abstract, categories, checklist) |

---

## Reproducing the Benchmarks

All experiment scripts and frozen JSON summaries are tracked in this repository:
- **Monte Carlo coverage-gate calibration:** [`research/simulation/coverage_calibration.py`](../../research/simulation/coverage_calibration.py) → [`research/results/coverage_calibration.json`](../../research/results/coverage_calibration.json)
- **Live synthetic test-set pilot:** [`research/experiments/testgen_real_pilot.py`](../../research/experiments/testgen_real_pilot.py) → [`research/results/testgen_pilot/`](../../research/results/testgen_pilot/)
- **DSPy `MIPROv2` benchmark & post-compilation bloat audit:** [`research/experiments/dspy_benchmark.py`](../../research/experiments/dspy_benchmark.py) → [`research/results/dspy_benchmark/combined_summary.json`](../../research/results/dspy_benchmark/combined_summary.json)
- **PromptPrism-native stability upgrade:** [`research/experiments/prism_stability_upgrade_pilot.py`](../../research/experiments/prism_stability_upgrade_pilot.py) → [`research/results/dspy_benchmark/`](../../research/results/dspy_benchmark/)

---

## Citation

```bibtex
@misc{sanchez2026auditing,
  title={Auditing, Pruning, and Stabilizing Compiled Prompts: Coverage-Aware Equivalence Testing and Perturbation Stability Certification for {LLMs}},
  author={Sanchez, Roberto},
  year={2026},
  month={November},
  howpublished={PromptPrism Research Group Technical Report},
  url={https://github.com/robsanmx/prompt_prism/blob/main/docs/paper/Auditing_Pruning_and_Stabilizing_Compiled_Prompts.pdf}
}
```
