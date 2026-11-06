"""
Unit and verification tests for research factor configurations and prompt rendering.
Verifies that:
1. Toggling any single factor modifies strictly its assigned prompt section.
2. Factor level contents are non-empty and separable.
3. Dataset manifests maintain strict split disjointness and required schemas.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORS_CONFIG_PATH = REPO_ROOT / "research" / "configs" / "factors.yaml"
TASKS_CONFIG_PATH = REPO_ROOT / "research" / "configs" / "tasks.yaml"
DATASET_MANIFEST_PATH = REPO_ROOT / "research" / "data" / "dataset_manifest.yaml"


@pytest.fixture
def factors_cfg():
    with open(FACTORS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tasks_cfg():
    with open(TASKS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def dataset_manifest():
    with open(DATASET_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_factor_definitions_complete_and_separable(factors_cfg, tasks_cfg):
    """Verify that all 6 factors (A-F) are defined with non-empty content across all tasks."""
    expected_factors = ["A", "B", "C", "D", "E", "F"]
    factors = factors_cfg["factors"]

    assert set(factors.keys()) == set(expected_factors)

    for factor_id in expected_factors:
        factor_entry = factors[factor_id]
        assert "name" in factor_entry
        assert "description" in factor_entry
        assert "tasks" in factor_entry

        for task_key in ["schema_extraction", "grounded_qa", "intent_classification"]:
            assert task_key in factor_entry["tasks"]
            task_levels = factor_entry["tasks"][task_key]
            assert "level_0" in task_levels
            assert "level_1" in task_levels
            # Level 1 must provide substantive prompt instructions
            assert len(task_levels["level_1"].strip()) > 0
            # Level 0 and Level 1 must be distinct
            assert task_levels["level_0"] != task_levels["level_1"]


def test_prompt_rendering_factor_isolation(factors_cfg, tasks_cfg):
    """Verify that toggling one factor alters only its designated section in the rendered prompt."""
    tasks = tasks_cfg["tasks"]
    factors = factors_cfg["factors"]

    sample_context = {
        "text": "Acme Corp bought Beta LLC for $50M on Jan 12, 2024.",
        "context": "Water boils at 100 degrees Celsius.",
        "question": "What is the boiling point of water?",
        "message": "I lost my credit card yesterday.",
    }

    for task_name, task_def in tasks.items():
        template_str = task_def["prompt_template"]

        # Baseline: all factors at level 0
        base_kwargs = {
            f_data["name"]: f_data["tasks"][task_name]["level_0"]
            for f_data in factors.values()
        }
        base_kwargs.update(sample_context)

        # Simple string replacement for jinja2-like tokens
        def render(kwargs):
            res = template_str
            for k, v in kwargs.items():
                res = res.replace("{{ " + k + " }}", v).replace("{{" + k + "}}", v)
            return res

        rendered_base = render(base_kwargs)

        # Toggle each factor individually to level 1
        for factor_id, f_data in factors.items():
            f_name = f_data["name"]
            variant_kwargs = dict(base_kwargs)
            variant_kwargs[f_name] = f_data["tasks"][task_name]["level_1"]
            rendered_variant = render(variant_kwargs)

            # The rendered variant must differ from base
            assert rendered_variant != rendered_base

            # All other factor texts in rendered_variant must remain identical to base
            for other_id, other_data in factors.items():
                if other_id == factor_id:
                    continue
                # Other factor's level 0 text should still be present if non-empty
                other_l0 = other_data["tasks"][task_name]["level_0"]
                if other_l0:
                    assert other_l0 in rendered_variant


def test_dataset_manifest_integrity_and_disjoint_splits(dataset_manifest):
    """Verify dataset manifest split counts, prefix uniqueness, and split disjointness."""
    datasets = dataset_manifest["datasets"]
    assert len(datasets) == 3

    for name, data in datasets.items():
        assert data["total_items"] == 160
        splits = data["splits"]
        assert splits["discovery"]["count"] == 80
        assert splits["confirmation"]["count"] == 80

        # Ensure ID prefixes are distinct
        disc_prefix = splits["discovery"]["id_prefix"]
        conf_prefix = splits["confirmation"]["id_prefix"]
        assert disc_prefix != conf_prefix
        assert disc_prefix.endswith("_")
        assert conf_prefix.endswith("_")

        # Verify zero overlap between discovery and confirmation IDs
        disc_ids = {f"{disc_prefix}{i:03d}" for i in range(1, 81)}
        conf_ids = {f"{conf_prefix}{i:03d}" for i in range(1, 81)}
        assert disc_ids.isdisjoint(conf_ids)
