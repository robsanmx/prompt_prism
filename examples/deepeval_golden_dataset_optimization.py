"""
Tutorial: Optimizing LLM Prompts with DeepEval LLM-as-a-Judge and Golden Datasets.

This tutorial demonstrates how to use PromptPrism with DeepEval in two distinct modes:
1. Reference-Based Evaluation: Using a Golden Dataset (expected targets) with G-Eval and Contextual Recall.
2. Reference-Free Evaluation: Measuring Answer Relevancy and Faithfulness without ground truth labels.
"""

from typing import Any, Dict, List
from prompt_prism import (
    DeepEvalMetric,
    Experiment,
    Factor,
    JudgeCache,
    MockLLM,
    deepeval_metric,
    plot_main_effects,
    plot_pareto_effects,
)


def main():
    print("=" * 80)
    print(" 💎 PromptPrism Tutorial: DeepEval LLM Judge with Golden Datasets")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Define Candidate Prompt Factors
    # -----------------------------------------------------------------------
    factors = [
        Factor.binary(
            name="expert_persona",
            level_0_content="",
            level_1_content="You are a Senior Compliance Auditor and Legal Specialist.",
            description="Role / persona specification",
        ),
        Factor.binary(
            name="few_shot_exemplars",
            level_0_content="",
            level_1_content=(
                "Example 1:\nQuery: Can I get a refund after 40 days?\n"
                "Answer: No, the maximum refund window is 30 days.\n"
            ),
            description="Few-shot demonstration",
        ),
        Factor.binary(
            name="reasoning_cot",
            level_0_content="State the final answer directly.",
            level_1_content="Think step-by-step: quote relevant policy sections before stating the conclusion.",
            description="Reasoning style",
        ),
        Factor.binary(
            name="hallucination_guardrail",
            level_0_content="",
            level_1_content="Strict Rule: Never state terms or fees not explicitly in the policy context.",
            description="Anti-hallucination guardrail",
        ),
        Factor.binary(
            name="conciseness_constraint",
            level_0_content="",
            level_1_content="Keep your explanation under 3 sentences.",
            description="Length constraint",
        ),
    ]

    # -----------------------------------------------------------------------
    # Step 2: Prepare a Golden Dataset (Reference Inputs + Expected Targets)
    # -----------------------------------------------------------------------
    golden_dataset = [
        {
            "id": "case_01",
            "query": "What is the fee for late contract cancellation?",
            "target": "$50 cancellation fee if cancelled within 14 days; non-refundable thereafter.",
            "policy_text": "Clause 4.1: Cancellation within 14 days incurs a $50 administrative fee. After 14 days, fees are non-refundable.",
        },
        {
            "id": "case_02",
            "query": "Does the warranty cover accidental liquid damage?",
            "target": "No, liquid damage is strictly excluded from standard warranty coverage.",
            "policy_text": "Clause 8.3: Standard warranty covers manufacturing defects. Accidental liquid or physical damage is excluded.",
        },
        {
            "id": "case_03",
            "query": "Can enterprise accounts request custom SLA terms?",
            "target": "Yes, enterprise tier customers can negotiate custom 99.99% uptime SLAs.",
            "policy_text": "Clause 12.0: Enterprise customers with annual spend over $100k may negotiate bespoke 99.99% SLA terms.",
        },
        {
            "id": "case_04",
            "query": "What is the return window for hardware peripherals?",
            "target": "Hardware peripherals have a 14-day return window from delivery date.",
            "policy_text": "Clause 6.2: Return window for hardware accessories and peripherals is 14 calendar days.",
        },
    ]

    # -----------------------------------------------------------------------
    # Step 3: Configure DeepEval Judge Metrics & JudgeCache
    # -----------------------------------------------------------------------
    # The JudgeCache prevents paying for duplicate judge evaluations across ANOVA reruns
    judge_cache = JudgeCache(db_path="judge_cache.db")

    # Check if deepeval is installed, otherwise use simulated judge for tutorial demonstration
    try:
        import deepeval
        # Mode A: Reference-Based Metric (Uses the Golden "target" for factual truth alignment)
        factual_geval = deepeval_metric(
            "g_eval",
            name="factual_accuracy_geval",
            criteria="Determine whether the actual output factually agrees with the expected golden target.",
            evaluation_steps=[
                "Check if all key policy numbers, terms, and constraints from the expected output are preserved.",
                "Penalize if the actual output contradicts the expected output or fabricates extra clauses.",
                "Ignore stylistic or phrasing differences if the core facts are identical.",
            ],
            cache=judge_cache,
        )

        # Mode B: Reference-Free Metric (Checks directness of the answer without needing golden truth)
        relevancy_metric = deepeval_metric(
            "answer_relevancy",
            threshold=0.7,
            cache=judge_cache,
        )
    except ImportError:
        # Simulated judge fallback for tutorial run without deepeval installed
        class SimulatedGEvalJudge:
            def __init__(self):
                self.score = 0.85
                self.reason = "Simulated factual correctness score."

            def measure(self, test_case):
                # Simulate higher score when persona and few-shot are active
                has_persona = "Compliance Auditor" in test_case.input
                has_few_shot = "Example 1" in test_case.input
                base = 0.60
                if has_persona:
                    base += 0.15
                if has_few_shot:
                    base += 0.20
                self.score = min(1.0, base)
                return self.score

        factual_geval = DeepEvalMetric(
            metric_factory=SimulatedGEvalJudge,
            name="factual_accuracy_geval",
            cache=judge_cache,
        )
        relevancy_metric = DeepEvalMetric(
            metric_factory=SimulatedGEvalJudge,
            name="deepeval_answer_relevancy",
            cache=judge_cache,
        )

    metrics = [factual_geval, relevancy_metric]

    # -----------------------------------------------------------------------
    # Step 4: Setup the PromptPrism Experiment Orchestrator
    # -----------------------------------------------------------------------
    # Using Resolution V design: 16 runs instead of 32 (50% savings, unaliased main effects)
    exp = Experiment.from_factors(
        factors=factors,
        design="2(5-1)V",
        system_prompt="You are an enterprise policy Q&A assistant.",
        data_template="Policy Context:\n{{ policy_text }}\n\nCustomer Question:\n{{ query }}",
        metrics=metrics,
        target_metric="factual_accuracy_geval",
        title="Enterprise Policy Q&A Prompt Optimization",
    )

    # -----------------------------------------------------------------------
    # Step 5: Connect LLM Provider & Execute Trials
    # -----------------------------------------------------------------------
    client = MockLLM(default_response="$50 cancellation fee applies within 14 days per Clause 4.1.")

    print(f"\n🚀 Running {len(exp.design.runs)} experimental runs across {len(golden_dataset)} golden cases...")
    print(f"📊 Estimated judge calls: {exp.estimate_judge_calls(golden_dataset)} (cached in SQLite)")

    results = exp.run(dataset=golden_dataset, client=client, max_workers=4)
    print(f"✅ Completed {len(results.trials)} trials successfully.")

    # -----------------------------------------------------------------------
    # Step 6: Perform ANOVA with Randomized Complete Block Design (RCBD)
    # -----------------------------------------------------------------------
    # Blocking on sample_id isolates query difficulty from true factor effects
    report = exp.analyze(block_by="sample_id")

    print("\n" + "=" * 80)
    print(" 📊 Executive ANOVA Analysis & Optimal Recipe")
    print("=" * 80)
    print(report.to_markdown())

    # -----------------------------------------------------------------------
    # Step 7: Export Configured Winning Prompt
    # -----------------------------------------------------------------------
    optimal_template = exp.get_optimal_prompt_template()
    sample_prompt = exp.composer.compose_text(
        run_config=exp.design.runs[0].model_copy(
            update={"factor_levels": report.optimal_recommendation.optimal_factor_levels}
        ),
        data={
            "policy_text": "Clause 5: Returns accepted within 30 days.",
            "query": "Can I return an item?",
        },
    )

    print("\n🏆 Production-Ready Optimal Prompt:")
    print("-" * 50)
    print(sample_prompt)
    print("-" * 50)


if __name__ == "__main__":
    main()
