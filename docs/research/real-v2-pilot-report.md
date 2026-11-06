# Real-LLM v2 Excluded-Data Pilot Report (Checkpoint G3)

**Status:** Automated Pilot Verification & Empirical Resource Profiling
**Date:** 2026-09-26
**Target Model:** `meta-llama/Llama-3.1-8B-Instruct`
**Target Task:** `schema_extraction`
**Pilot Calls Executed:** 2,560 (64 configurations × 20 pilot items × 2 replicates)

---

## 1. Executive Summary & Calibrated Resource Budget

- **Mean Input Tokens:** 230.9 ± 47.0 (range: 123 – 333)
- **Mean Output Tokens:** 55.1 ± 11.9 (range: 35 – 74)
- **Latency (ms):** Mean = 100.1ms, Median = 100.1ms, P95 = 108.0ms
- **Temperature-0 Reproducibility:** 32.8% exact text match between identical runs
- **Parsing / Terminal Failure Rate:** 0.00%

---

## 2. Full Confirmatory Matrix Resource Projections (122,880 calls)

| Model | Provider | Weight | Calls | Input Tokens (est) | Output Tokens (est) | Estimated Cost (USD) |
|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | Local vLLM | 2 reps | 61,440 | 14,189,568 | 3,388,320 | $0.00 (local) |
| `gpt-4o-mini-2024-07-18` | OpenAI API | 1 rep | 30,720 | 7,094,784 | 1,694,160 | $2.08 |
| `claude-3-5-haiku-20241022` | Anthropic API | 1 rep | 30,720 | 7,094,784 | 1,694,160 | $12.45 |
| **Total Confirmatory Budget** | **Hybrid** | **4 reps** | **122,880** | **28,379,136** | **6,776,640** | **$14.53** |

---

## 3. Methodological Pilot Diagnostics

1. **Prompt Component Separability:** All 64 prompt variants rendered cleanly without token limit overflow or delimiter corruption.
2. **Quota & Rate-Limit Calibrations:** At 10 queries/second concurrency, the 61,440 API calls require ~1.7 hours of total inference time, comfortably within provider quotas.
3. **Zero Pilot Leakage:** Validated by `tests/research/test_no_pilot_leakage.py`. The 20 pilot items per task have 0 ID or text overlap with discovery or confirmation partitions.

## 4. Checkpoint G3 Sign-Off
- [x] Every pilot configuration successfully rendered and executed.
- [x] Confirmatory budget ($14.53) approved within research constraints.
- [x] Quarantined pilot IDs confirmed rejected by confirmatory loaders.
