# Statistical analysis plan: real-v3

The primary unit is an evaluation item. Scores are first averaged across items within task and configuration. The three tasks receive equal weight in the primary aggregate, regardless of metric scale, because every primary metric is bounded from zero to one.

Tie breaking is deterministic: the lowest numeric run ID wins a tie. This applies to random-screen winners, the exhaustive discovery oracle, and the descriptive confirmation best. A zero estimated factor contrast selects level -1.

The primary bootstrap resamples item rows, never individual configuration responses, so all 64 paired configuration outcomes for an item move together. Discovery and confirmation resampling are independent. Selection is repeated inside every replicate. The Monte Carlo p-value is `(1 + count(delta >= 0)) / (B + 1)` for the one-sided alternative that Prompt Prism has lower regret. The percentile interval is the 2.5th and 97.5th percentiles of bootstrap aggregate deltas. `B=2000`; the random seed is 42.

For H2, a factor effect is the mean score at +1 minus the mean score at -1. The full 64-run discovery cube supplies the reference effect. Prompt Prism uses its registered 16 runs. Each random design uses its own 16 runs. RMSE and exact sign agreement (zero is its own sign) are computed across six factors.

For H3, the observed discovery selections are fixed, and confirmation items are resampled in paired form for the selected run and run 1. Two-sided bootstrap p-values are `2 * min(P(diff <= 0), P(diff >= 0))`, capped at one. Benjamini-Hochberg correction is applied across the three task p-values.

All numeric summaries are generated directly from the two frozen JSONL manifests. Any post hoc diagnostic is labeled as such.
