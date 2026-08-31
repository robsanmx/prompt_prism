"""
End-to-End Tutorial: Optimizing an E-commerce Extraction Prompt using Fractional Factorial DoE & ANOVA.
"""

import pandas as pd
from prompt_prism import (
    ExactMatch,
    Experiment,
    F1Score,
    Factor,
    KeyValuesExtractionOverlap,
    MockLLM,
    plot_main_effects,
    plot_pareto_effects,
)


def main():
    print("=" * 70)
    print("🔬 PROMPT OPTIMIZATION VIA FRACTIONAL FACTORIAL DESIGN (DoE) & ANOVA")
    print("=" * 70)

    # Step 1: Define the 5 Candidate Factors
    factors = [
        Factor.binary(
            name="persona",
            level_0_content="",
            level_1_content="You are a senior product taxonomy specialist at a leading e-commerce marketplace.",
            description="Expert persona instruction",
        ),
        Factor.binary(
            name="few_shot_examples",
            level_0_content="",
            level_1_content="Example 1:\nInput: Notebook Lenovo Ideapad 1 4GB 128GB\nOutput: {'brand': 'Lenovo', 'model': 'Ideapad 1', 'ram': '4GB', 'storage': '128GB'}",
            description="Few-shot exemplar demonstration",
        ),
        Factor.binary(
            name="chain_of_thought",
            level_0_content="",
            level_1_content="Think step-by-step: first identify vertical category, then extract each attribute with text evidence.",
            description="Chain-of-thought step-by-step reasoning",
        ),
        Factor.binary(
            name="strict_json_format",
            level_0_content="Output: key: value pairs.",
            level_1_content="OUTPUT FORMAT: Return a valid JSON object matching {attribute_id: value}.",
            description="JSON formatting instruction",
        ),
        Factor.binary(
            name="negative_constraints",
            level_0_content="",
            level_1_content="Strict Rule: Never extrapolate or hallucinate attributes not explicitly present in the text.",
            description="Hallucination guardrail constraint",
        ),
    ]

    # Step 2: Initialize Experiment with 2^(5-1)V design (16 runs, Resolution V)
    exp = Experiment.from_factors(
        factors=factors,
        design="2(5-1)V",
        system_prompt="Your task is to extract structured catalog attributes from product offer text.",
        data_template="Product Title: {{ title }}\nDescription: {{ description }}",
        metrics=[KeyValuesExtractionOverlap(), ExactMatch(), F1Score()],
        target_metric="extraction_f1",
        title="Catalog Attribute Extraction DoE",
    )

    print(f"\n✅ Created Design: {exp.design.plan_id} with {exp.design.num_runs} prompt runs across {exp.design.num_factors} factors.")
    print("Design Matrix Preview:")
    print(exp.design.to_dataframe().head(8).to_string(index=False))

    # Step 3: Test Benchmark Dataset
    dataset = [
        {
            "id": 1,
            "title": "Notebook Lenovo Ideapad 1 4GB 128GB SSD 14 Intel Celeron Silver",
            "description": "Lenovo laptop for daily work. 14 inch HD display, 4GB RAM, 128GB SSD storage.",
            "target": '{"brand": "Lenovo", "model": "Ideapad 1", "ram": "4GB", "storage": "128GB", "color": "Silver"}',
        },
        {
            "id": 2,
            "title": "Zapatillas Nike Air Max 90 White Black Men Size 42",
            "description": "Original Nike sneakers with air cushion sole and leather upper.",
            "target": '{"brand": "Nike", "model": "Air Max 90", "color": "White Black", "size": "42"}',
        },
        {
            "id": 3,
            "title": "Smart TV Samsung 55 UHD 4K Crystal HDR10+",
            "description": "55 inch Samsung 4K television with Crystal processor and smart apps.",
            "target": '{"brand": "Samsung", "screen_size": "55 inch", "resolution": "4K UHD"}',
        },
    ]

    # Step 4: Simulated LLM with ground truth responses
    def simulated_llm(prompt: str, **kwargs) -> str:
        p = str(prompt)
        # Check factors
        has_persona = "senior product taxonomy specialist" in p
        has_few_shot = "Example 1:" in p
        has_json = "valid JSON object" in p

        if has_few_shot and has_json:
            return '{"brand": "Lenovo", "model": "Ideapad 1", "ram": "4GB", "storage": "128GB", "color": "Silver"}'
        elif has_few_shot:
            return "brand: Lenovo, model: Ideapad 1, ram: 4GB, storage: 128GB"
        elif has_persona:
            return '{"brand": "Lenovo", "model": "Ideapad 1"}'
        else:
            return "brand: unknown"

    client = MockLLM(response_generator=lambda p, kw: simulated_llm(p))

    # Step 5: Execute Trials
    print(f"\n🚀 Running {len(exp.design.runs)} experimental prompt variants across {len(dataset)} items...")
    results = exp.run(dataset=dataset, client=client, max_workers=4)
    print(f"✅ Completed {len(results.trials)} trials.")

    # Step 6: ANOVA Analysis & Optimal Prompt Recommendation
    print("\n" + "=" * 70)
    print("📈 STATISTICAL ANALYSIS & ANOVA RESULTS")
    print("=" * 70)
    report = exp.analyze()
    print(report.to_markdown())

    # Step 7: Export Visualization Plots
    plot_main_effects(report.anova_result, save_path="main_effects_plot.png")
    plot_pareto_effects(report.anova_result, save_path="pareto_effects_plot.png")
    print("\n📊 Saved diagnostic plots: 'main_effects_plot.png' and 'pareto_effects_plot.png'")

    # Step 8: Get Configured Winning Prompt
    opt_template = exp.get_optimal_prompt_template()
    sample_composed = exp.composer.compose_text(
        run_config=exp.design.runs[0].model_copy(update={"factor_levels": report.optimal_recommendation.optimal_factor_levels}),
        data=dataset[0],
    )
    print("\n" + "=" * 70)
    print("🏆 FINAL OPTIMIZED WINNING PROMPT RECIPE:")
    print("=" * 70)
    print(sample_composed)
    print("=" * 70)


if __name__ == "__main__":
    main()
