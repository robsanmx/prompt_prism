"""
Unit tests for PromptSection, PromptTemplate, and PromptComposer.
"""

from prompt_doe.core.factors import Factor, FactorSet
from prompt_doe.core.models import RunConfig
from prompt_doe.template.composer import PromptComposer, PromptSection, PromptTemplate


def test_prompt_section_render():
    sec = PromptSection(
        id="task",
        content="Extract attributes from: {{ text }}",
        prefix="### TASK:\n",
    )
    rendered = sec.render(data={"text": "Nike Shoes"})
    assert rendered == "### TASK:\nExtract attributes from: Nike Shoes"


def test_prompt_composer_modular_sections():
    f1 = Factor.binary("persona", level_0_content="", level_1_content="You are an expert AI.")
    f2 = Factor.binary("format", level_0_content="Return plaintext.", level_1_content="Return JSON object.")
    f_set = FactorSet([f1, f2])

    template = PromptTemplate.from_factors(
        factors=f_set,
        system_prompt="Base system instruction.",
        data_template="Product: {{ title }}",
    )
    composer = PromptComposer(template=template, factors=f_set)

    # Run config with persona=1, format=0
    run1 = RunConfig(run_id=1, factor_levels={"A": 1, "B": 0})
    text1 = composer.compose_text(run1, data={"title": "Laptop"})

    assert "Base system instruction." in text1
    assert "You are an expert AI." in text1
    assert "Return plaintext." in text1
    assert "Product: Laptop" in text1

    # Run config with persona=0, format=1
    run2 = RunConfig(run_id=2, factor_levels={"A": 0, "B": 1})
    text2 = composer.compose_text(run2, data={"title": "Laptop"})
    assert "You are an expert AI." not in text2
    assert "Return JSON object." in text2


def test_prompt_composer_chat_messages():
    f1 = Factor.binary("persona", level_1_content="Expert role")
    f_set = FactorSet([f1])

    tmpl = PromptTemplate()
    tmpl.add_section(id="sys", content="System instruction", role="system", position=0)
    tmpl.add_section(id="persona", factor_id="A", role="system", position=1)
    tmpl.add_section(id="user_data", content="Input: {{ val }}", role="user", position=2)

    composer = PromptComposer(template=tmpl, factors=f_set)
    messages = composer.compose_messages(RunConfig(run_id=1, factor_levels={"A": 1}), data={"val": "123"})

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "System instruction" in messages[0]["content"]
    assert "Expert role" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Input: 123" in messages[1]["content"]


def test_prompt_composer_jinja_master():
    tmpl = PromptTemplate(
        master_template="""
{% if persona %}Persona: {{ persona }}{% endif %}
Task: Extract data from {{ item_name }}.
Format: {{ format }}
        """.strip()
    )
    f1 = Factor.binary("persona", level_1_content="Senior Auditor")
    f2 = Factor.binary("format", level_0_content="YAML", level_1_content="JSON")
    f_set = FactorSet([f1, f2])

    composer = PromptComposer(template=tmpl, factors=f_set)
    text = composer.compose_text(RunConfig(run_id=1, factor_levels={"A": 1, "B": 1}), data={"item_name": "Invoice #42"})

    assert "Persona: Senior Auditor" in text
    assert "Task: Extract data from Invoice #42." in text
    assert "Format: JSON" in text
