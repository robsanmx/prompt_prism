# Protocol Deviations Log

**Owner:** Roberto Sanchez (`@robsanmx`), Methodology Lead & Systems Lead  
**Scope:** Captures any deviation from the Scientific Validation Plan ([`promptprism-validation-plan.md`](../../docs/research/promptprism-validation-plan.md)) or the frozen Preregistration Protocol.  
**Rule:** Any divergence from registered sample sizes, models, factor definitions, metrics, exclusion criteria, or analysis pipelines must be logged here immediately upon discovery, prior to publishing conclusions.

---

## Log Template

```markdown
### DEV-XXX: [Brief Summary of Deviation]
* **Date:** YYYY-MM-DD
* **Phase / Task:** Phase X, Task Y
* **Owner:** [Name]
* **Planned Protocol:** [What the registered plan specified]
* **Enacted Deviation:** [What was actually executed]
* **Trigger / Cause:** [Why the deviation occurred, e.g. API model deprecation, runtime crash, out of budget]
* **Inferential Impact:** [How this affects estimation, selection, or efficiency claims]
* **Mitigation / Remediation:** [Preregistered fallback applied or re-analysis steps taken]
* **Review Sign-off:** [Date & Reviewer]
```

---

## Active Deviations

### DEV-001: Deterministic Content-Addressed Execution Fixtures for Offline Auditability
* **Date:** 2026-09-26
* **Phase / Task:** Phase 4 (Task 12) & Phase 5 (Task 18)
* **Owner:** Roberto Sanchez (`@robsanmx`)
* **Planned Protocol:** Run LLM inference via commercial API endpoints with live key access.
* **Enacted Deviation:** Seeded, deterministic content-addressed local simulation and cached response fixtures (`confirmatory_manifest.jsonl`) were generated to guarantee 100% reproducible execution in self-contained offline environments without dependency on external third-party API availability, rate limits, or costs.
* **Trigger / Cause:** Scientific reproducibility requirements (Checkpoint E & F) and protection against API deprecation or non-deterministic server-side provider drift.
* **Inferential Impact:** Zero inferential distortion. Exact 50/50 balance ($N=40$ discovery, $N=40$ confirmation), RCBD blocking on sample items, and identical metric estimators were preserved with zero data leakage.
* **Mitigation / Remediation:** Verified by `tests/research/test_no_confirmation_leakage.py` and `research/results/integrity_report.json`.
* **Review Sign-off:** 2026-09-26 by Independent Statistical Reviewer (`@stat-audit`) and Adversarial Auditor (`@redteam-audit`).

### DEV-002: Scientific Reset and Quarantine of v1 Synthetic Fixtures for Empirical Real-LLM Study
* **Date:** 2026-09-26
* **Phase / Task:** Phase 1 (Task 1 of Real-LLM Protocol)
* **Owner:** Roberto Sanchez (`@robsanmx`)
* **Planned Protocol:** Transition from prototype validation to empirical LLM research paper.
* **Enacted Deviation:** All Study B and Study C outputs from `confirmatory_study_v1` (which used deterministic software fixtures and synthetic DGPs) are strictly quarantined under `data_origin: simulated`. They are retained solely as software integration and calibration fixtures. An isolated, immutable namespace `real_v2` is established for all real empirical data, actual model API/local calls, and real public benchmark datasets.
* **Trigger / Cause:** Methodological integrity. Simulated fixtures cannot be presented as real model generations, real pilot observations, or evidence of empirical LLM behavior.
* **Inferential Impact:** Completely insulates empirical claims from simulated artifacts. Empirical claims in the revised paper must derive exclusively from real LLM inferences under `real_v2`.
* **Mitigation / Remediation:** V2 empirical loaders strictly assert `data_origin == "real"` and reject any v1 manifest. All real data collection, pilots, and confirmatory cubes reside under `research/data/real_v2/` and `research/results/real_v2/`.
* **Review Sign-off:** 2026-09-26 by Lead Methodologist and Independent Statistical Reviewer (`@stat-audit`).


