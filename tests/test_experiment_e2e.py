"""
End-to-End Integration test for complete PromptPrism Experiment workflow.
"""

import pandas as pd
from prompt_prism.core.factors import Factor
from prompt_prism.evaluation.metrics import ExactMatch, F1Score
from prompt_prism.experiment import Experiment
from prompt_prism.runner.client import MockLLM


def test_full_experiment_e2e(tmp_path):
    # 1. Define Factors
    factors = [
        Factor.binary("persona", level_0_content="", level_1_content="You are an expert catalog specialist."),
        Factor.binary("few_shot", level_0_content="", level_1_content="Example: Brand=Nike"),
        Factor.binary("cot", level_0_content="", level_1_content="Think step by step."),
        Factor.binary("format_json", level_0_content="PlainText", level_1_content="StrictJSON"),
        Factor.binary("constraints", level_0_content="", level_1_content="Only extract existing values."),
    ]

    # 2. Test Dataset
    dataset = [
        {"id": 1, "title": "Nike Air Max 90 Red", "description": "Men sneakers size 42", "target": "Brand: Nike, Color: Red"},
        {"id": 2, "title": "Adidas Ultraboost 21", "description": "Running shoes black", "target": "Brand: Adidas, Color: Black"},
        {"id": 3, "title": "Puma Suede Classic", "description": "Retro sneakers blue", "target": "Brand: Puma, Color: Blue"},
    ]

    # 3. Create simulated response generator that responds better when persona and few_shot are present
    def mock_llm_fn(prompt: str, **kwargs):
        p_str = str(prompt)
        score = 0.3
        if "expert catalog specialist" in p_str:
            score += 0.35
        if "Example: Brand=Nike" in p_str:
            score += 0.30
        if "StrictJSON" in p_str:
            score += 0.10
        return "Brand: Nike, Color: Red" if score >= 0.80 else "Brand: Unknown"

    client = MockLLM(response_generator=lambda prompt, kwargs: mock_llm_fn(prompt))

    # 4. Instantiate Experiment
    exp = Experiment.from_factors(
        factors=factors,
        design="2(5-1)V",
        system_prompt="Extract attributes from product offers.",
        data_template="Offer: {{ title }}\nDescription: {{ description }}",
        metrics=[ExactMatch(), F1Score()],
        target_metric="exact_match",
        title="Sneakers Attribute Extraction PromptPrism",
    )

    # 5. Run Experiment
    results = exp.run(
        dataset=dataset,
        client=client,
        max_workers=2,
    )

    assert len(results.trials) == 16 * 3  # 16 runs * 3 items = 48 trials

    # 6. Analyze ANOVA
    report = exp.analyze()
    assert report.anova_result is not None
    assert report.anova_result.target_metric == "exact_match"

    # 7. Check Optimal Prompt Recommendation
    opt = report.optimal_recommendation
    assert opt is not None
    assert opt.optimal_factor_levels["A"] == 1  # Persona
    assert opt.optimal_factor_levels["B"] == 1  # Few shot

    # 8. Export Reports
    md_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"
    report.to_markdown(save_path=md_path)
    report.to_html(save_path=html_path)

    assert md_path.exists()
    assert html_path.exists()
    assert "# Sneakers Attribute Extraction PromptPrism" in md_path.read_text()

    # 9. Get Configured Optimal Template
    opt_template = exp.get_optimal_prompt_template()
    composed = exp.composer.compose_text(
        run_config=exp.design.runs[0].model_copy(update={"factor_levels": opt.optimal_factor_levels}),
        data={"title": "Test Shoe", "description": "Test Desc"},
    )
    assert "expert catalog specialist" in composed
    assert "Example: Brand=Nike" in composed
