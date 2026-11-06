# Data Statement for Confirmatory Benchmark Evaluation

**Following the Data Statement framework by Bender & Friedman (2018)**  
**Study:** *Factorial Prompt Engineering: Testing Fractional Designs for Efficient and Interpretable Prompt Optimization*  
**Date:** 2026-09-26  
**Status:** Protocol Version 1.0  

---

## A. Curation Rationale

To ensure objective and reproducible evaluation of factorial prompt optimization, the study selects three complementary task strata:
1. **Schema-Bound Information Extraction (`contract_events_v1`):** Tests an LLM's adherence to rigid structured output contracts and entity boundary precision.
2. **Grounded Question Answering (`grounded_fact_qa_v1`):** Tests context fidelity, factual precision, and appropriate abstention in the presence of unanswerable queries.
3. **Intent and Policy Classification (`customer_intent_v1`):** Tests multi-class discrete categorization and resistance to semantic confusion.

These tasks prioritize **deterministic scoring** (macro F1, token overlap, exact match) to eliminate LLM-as-a-judge stochasticity as a confounding factor in primary hypothesis testing.

---

## B. Language Variety

* **Language:** English (en-US, en-GB).
* **Dialect & Register:** 
  * Schema Extraction: Formal corporate legal and financial reporting.
  * Grounded QA: Formal expository and encyclopedic prose.
  * Intent Classification: Informal, conversational customer queries (including minor typographical variations, colloquialisms, and direct commands).

---

## C. Speaker / Annotator Demographics

* Benchmark datasets are constructed from established open research corpora with human-curated and cross-verified gold annotations.
* Annotators and reviewers possess professional fluency in English and domain familiarity with enterprise operations and formal text extraction.
* Annotator compensation and licensing comply with institutional ethical research standards.

---

## D. Text Characteristics & Length Distribution

* **Schema Extraction:** Passage lengths between 40 and 150 words; target entities comprise 4 named fields (`buyer`, `target`, `amount`, `date`).
* **Grounded QA:** Reference context passages between 60 and 200 words; questions between 8 and 25 words; gold answers between 3 and 25 words. Approximately $15\%$ of items are unanswerable to test abstention.
* **Intent Classification:** Customer utterances between 5 and 45 words across 7 disjoint classes.

---

## E. Preprocessing, Deduplication, and Splitting

1. **Deduplication:** Text inputs were canonicalized (trimmed whitespace, normalized Unicode punctuation) and audited using SHA-256 content hashes. Duplicate instances were purged.
2. **Split Strategy:** For each task, eligible items ($I = 160$) were stratified by subclass/difficulty and randomly partitioned using a fixed PRNG seed (`seed = 42`) into:
   * **Discovery Set ($I_{\text{disc}} = 80$):** Made available to prompt optimization methods and screening algorithms.
   * **Confirmation Set ($I_{\text{conf}} = 80$):** Sealed and isolated. Never accessed during factor screening or method fitting.

---

## F. Contamination & Privacy Audit

* **PII Redaction:** Synthetic entity replacement applied to names, phone numbers, and financial account numbers to prevent disclosure of real individual personal data.
* **Split Contamination:** Verified zero lexical overlap between discovery and confirmation partitions.
* **Pretraining Contamination:** Benchmark items checked against public training corpora dumps to minimize rote memorization artifacts.

---

## G. Licensing and Terms of Use

All selected data derivatives are published under Creative Commons Attribution 4.0 International (**CC-BY-4.0**), permitting open scholarly reproduction and distribution of derived score manifests.
