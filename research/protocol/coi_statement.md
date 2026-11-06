# Conflict of Interest and Author Disclosures

**Owner:** Roberto Sanchez (`@robsanmx`), Lead Author & Primary Developer  
**Date:** 2026-09-26  
**Status:** Active  

---

## 1. Primary Conflict of Interest Disclosure

The authors of this study are the creators, maintainers, and primary developers of the `prompt_prism` open-source library. This dual role creates an inherent risk of confirmation bias, selective reporting, or favorable benchmark framing.

To mitigate this conflict of interest and ensure objective scientific rigor, the research adheres to the following preregistered guardrails:

1. **Independent Preregistration & Locking**: Hypotheses, estimands, factor sets, benchmark datasets, analysis scripts, and equivalence thresholds ($\delta$) are timestamped and frozen *before* confirmatory data are collected.
2. **Access-Controlled Confirmation**: Confirmation benchmark items and their target labels remain completely isolated and inaccessible to optimization code.
3. **Transparent Failure Reporting**: Adverse and null results (e.g. instances where PromptPrism yields higher regret than random search, or where aliasing in Resolution III causes sign reversals) will be reported with equal prominence to positive results.
4. **Third-Party Statistical & Adversarial Review**: Independent reviewers who did not author the PromptPrism library will conduct adversarial audits of the analysis code, data integrity, and manuscript claims (as specified in Task 20).
5. **Full Computational Reproducibility**: All simulation code, experimental manifests, cached responses, and analysis scripts will be publicly released under permissive open-source licenses with permanent DOIs.

---

## 2. Funding and Resource Disclosures

* **Compute & API Credits**: All commercial model API calls (e.g. OpenAI, Anthropic, Google Gemini) used during the pilot and confirmatory phases will be reported with exact provider names, account tiers, and expenditure amounts.
* **Institutional Affiliations**: Any institutional or organizational affiliations and sponsorships will be disclosed in full in the manuscript.
