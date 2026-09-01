"""
Tutorial: Optimizing a Retrieval-Augmented Generation (RAG) Prompt with Fractional Factorial DoE.
"""

import importlib.util

from prompt_prism import (
    ExactMatch,
    Experiment,
    F1Score,
    Factor,
    MockLLM,
)


def main():
    print("=" * 70)
    print("🧠 RAG PROMPT OPTIMIZATION VIA FRACTIONAL FACTORIAL DoE")
    print("=" * 70)

    # 1. Define 6 RAG Prompt Factors
    factors = [
        Factor.binary(
            name="system_role",
            level_0_content="You are a helpful assistant.",
            level_1_content="You are a rigorous enterprise research assistant adhering strictly to source truth.",
            description="System persona guidance",
        ),
        Factor.binary(
            name="citation_rule",
            level_0_content="",
            level_1_content="Mandatory: Every factual claim must cite its source bracketed like [Doc 1].",
            description="Citation requirement",
        ),
        Factor.binary(
            name="refusal_policy",
            level_0_content="",
            level_1_content="If the retrieved context does not contain the answer, reply: 'I cannot find that in the documents.'",
            description="Out-of-context refusal constraint",
        ),
        Factor.binary(
            name="reasoning_cot",
            level_0_content="",
            level_1_content="Analyze the documents step by step before stating your final answer.",
            description="Chain of thought reasoning",
        ),
        Factor.binary(
            name="xml_framing",
            level_0_content="Documents: {context}",
            level_1_content="<documents>\n{context}\n</documents>",
            description="XML tags enclosing context",
        ),
        Factor.binary(
            name="conciseness",
            level_0_content="",
            level_1_content="Keep your response under 3 concise sentences.",
            description="Length constraint",
        ),
    ]

    # 2. Setup Experiment using 2^(6-2)IV (16 runs, Resolution IV)
    exp = Experiment.from_factors(
        factors=factors,
        design="2(6-2)IV",  # 16 runs instead of 64 runs!
        system_prompt="{{ system_role }}",
        data_template="Context:\n{{ xml_framing }}\n\nUser Question: {{ question }}",
        metrics=[F1Score(), ExactMatch()],
        target_metric="f1_score",
        title="Enterprise RAG Prompt Optimization",
    )

    print(
        f"\n✅ Created Design: {exp.design.plan_id} with {exp.design.num_runs} runs across {exp.design.num_factors} factors."
    )

    # 3. Test Dataset with retrieval_context for judge field mapping
    dataset = [
        {
            "id": 1,
            "context": "Doc 1: Project Apollo was founded in 2021. Doc 2: Apollo budget was $5M.",
            "retrieval_context": [
                "Project Apollo was founded in 2021.",
                "Apollo budget was $5M.",
            ],
            "question": "When was Apollo founded and what was its budget?",
            "target": "Project Apollo was founded in 2021 with a budget of $5M [Doc 1] [Doc 2].",
        },
        {
            "id": 2,
            "context": "Doc 1: Python was created by Guido van Rossum in 1991.",
            "retrieval_context": ["Python was created by Guido van Rossum in 1991."],
            "question": "Who created Python and when?",
            "target": "Python was created by Guido van Rossum in 1991 [Doc 1].",
        },
    ]

    # 4. Simulated LLM
    def rag_llm(prompt: str, **kwargs) -> str:
        p = str(prompt)
        has_xml = "<documents>" in p
        has_cite = "Every factual claim must cite" in p

        if has_cite and has_xml:
            return "Project Apollo was founded in 2021 with a budget of $5M [Doc 1] [Doc 2]."
        elif has_cite:
            return "Apollo was founded in 2021 with $5M budget [Doc 1]."
        else:
            return "Apollo was founded in 2021."

    client = MockLLM(response_generator=lambda p, kw: rag_llm(p))

    # 5. Run & Analyze deterministic metrics
    results = exp.run(dataset=dataset, client=client, max_workers=2)
    print(f"✅ Finished running {len(results.trials)} trials.")
    report = exp.analyze()
    print(report.to_markdown())

    # 6. Optional DeepEval Judge metrics demonstration
    if importlib.util.find_spec("deepeval") is not None:
        from prompt_prism.evaluation.deepeval_metrics import deepeval_metric
        from prompt_prism.evaluation.judge_cache import JudgeCache

        print("\n" + "=" * 70)
        print("🔍 DEEPEVAL LLM JUDGE EVALUATION (Faithfulness & Answer Relevancy)")
        print("=" * 70)

        judge_cache = JudgeCache()
        faith_metric = deepeval_metric("faithfulness", cache=judge_cache)
        rel_metric = deepeval_metric("answer_relevancy", cache=judge_cache)

        deepeval_exp = Experiment.from_factors(
            factors=factors,
            design="2(6-2)IV",
            system_prompt="{{ system_role }}",
            data_template="Context:\n{{ xml_framing }}\n\nUser Question: {{ question }}",
            metrics=[faith_metric, rel_metric, F1Score()],
            target_metric="deepeval_faithfulness",
            title="Enterprise RAG - DeepEval Judge Optimization",
        )
        print(f"Estimated Judge calls: {deepeval_exp.estimate_judge_calls(dataset)}")
    else:
        print(
            "\nℹ️ [Note] DeepEval not installed. Install with `pip install prompt-prism[deepeval]` to enable LLM-as-judge metrics."
        )


if __name__ == "__main__":
    main()
