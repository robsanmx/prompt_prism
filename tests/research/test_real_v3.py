"""Regression tests for the prospective Gemini real-v3 study."""

import json

import numpy as np
import pandas as pd
import pytest

from research.real_v3.analysis import analyze_task, fit_main_effects
from research.real_v3.datasets import DATA_ROOT, INTENT_LABELS
from research.real_v3.design import (
    FACTOR_IDS,
    all_factorial_configs,
    balanced_random_designs,
    resolution_iv_run_ids,
    stable_seed,
)
from research.real_v3.locking import (
    assert_lock_matches,
    create_recommendation_lock,
)
from research.real_v3.metrics import (
    evaluate_entity_extraction,
    evaluate_grounded_qa,
    evaluate_intent,
)
from research.real_v3.prompts import render_prompt


def test_stable_seed_does_not_depend_on_python_hash_randomization():
    assert stable_seed("grounded_qa", base=42) == stable_seed("grounded_qa", base=42)
    assert stable_seed("grounded_qa", base=42) != stable_seed(
        "schema_extraction", base=42
    )
    assert stable_seed("grounded_qa", base=42) == 1718039911


def test_balanced_random_designs_are_reproducible_balanced_and_full_rank():
    configs = all_factorial_configs()
    first = balanced_random_designs(configs, count=8, seed=20260926)
    second = balanced_random_designs(configs, count=8, seed=20260926)
    assert first == second
    assert len({tuple(x) for x in first}) == 8

    for run_ids in first:
        assert len(run_ids) == 16
        assert len(set(run_ids)) == 16
        matrix = np.array(
            [[configs[run_id][f] for f in FACTOR_IDS] for run_id in run_ids]
        )
        assert (matrix.sum(axis=0) == 0).all()
        design = np.column_stack([np.ones(len(matrix)), matrix])
        assert np.linalg.matrix_rank(design) == 7


def test_registered_fraction_is_balanced_full_rank_and_resolution_iv():
    configs = all_factorial_configs()
    run_ids = resolution_iv_run_ids(configs)
    assert len(run_ids) == 16
    matrix = np.array([[configs[run_id][f] for f in FACTOR_IDS] for run_id in run_ids])
    assert (matrix.sum(axis=0) == 0).all()
    assert np.linalg.matrix_rank(np.column_stack([np.ones(16), matrix])) == 7
    assert all(
        configs[r]["E"] == configs[r]["A"] * configs[r]["B"] * configs[r]["C"]
        for r in run_ids
    )
    assert all(
        configs[r]["F"] == configs[r]["B"] * configs[r]["C"] * configs[r]["D"]
        for r in run_ids
    )


def test_prompts_use_task_specific_examples_contracts_and_label_inventory():
    factors = {f: 1 for f in FACTOR_IDS}

    schema_item = {
        "text": "Alice met Example Org in London on Monday.",
        "target": {
            "person": ["Alice"],
            "organization": ["Example Org"],
            "location": ["London"],
            "money": [],
            "temporal": ["Monday"],
        },
    }
    qa_item = {
        "context": "The mission launched in 2020.",
        "question": "When did it launch?",
    }
    intent_item = {"utterance": "My card has not arrived."}
    examples = {
        "schema_extraction": [(schema_item, schema_item["target"])],
        "grounded_qa": [({**qa_item}, {"answer": "2020"})],
        "intent_classification": [(intent_item, {"intent": "card_arrival"})],
    }

    schema = render_prompt("schema_extraction", schema_item, factors, examples, [])
    qa = render_prompt("grounded_qa", qa_item, factors, examples, [])
    intent = render_prompt(
        "intent_classification",
        intent_item,
        factors,
        examples,
        ["card_arrival", "card_not_working"],
    )

    assert schema.system_prompt
    assert '"person"' in schema.user_prompt
    assert '"answer"' in qa.user_prompt
    assert '"intent"' in intent.user_prompt
    assert "card_arrival" in intent.user_prompt
    assert "card_not_working" in intent.user_prompt
    assert "Example input" in schema.user_prompt
    assert "Example input" in qa.user_prompt
    assert "Example input" in intent.user_prompt
    assert "exact matching structured fields" not in qa.user_prompt


def test_task_evaluators_parse_direct_and_json_outputs_without_substring_guessing():
    target = {
        "person": ["Alice"],
        "organization": ["Example Org"],
        "location": [],
        "money": [],
        "temporal": [],
    }
    entity = evaluate_entity_extraction(
        '{"person":["Alice"],"organization":[],"location":[],"money":[],"temporal":[]}',
        target,
    )
    assert entity["entity_f1"] == pytest.approx(2 / 3)
    assert entity["schema_validity"] == 1.0

    assert (
        evaluate_grounded_qa('{"answer":"Ariane 5"}', ["Ariane 5"], True)["token_f1"]
        == 1.0
    )
    assert evaluate_grounded_qa("Ariane 5", ["Ariane 5"], True)["token_f1"] == 1.0
    assert evaluate_grounded_qa('{"answer":null}', [], False)["token_f1"] == 1.0

    labels = ["card_arrival", "card_not_working"]
    assert (
        evaluate_intent('{"intent":"card_arrival"}', "card_arrival", labels)["accuracy"]
        == 1.0
    )
    assert evaluate_intent("card_arrival", "card_arrival", labels)["accuracy"] == 1.0
    assert (
        evaluate_intent(
            "Maybe card_arrival or card_not_working", "card_arrival", labels
        )["accuracy"]
        == 0.0
    )


def test_fit_main_effects_uses_only_supplied_fraction():
    rows = []
    configs = all_factorial_configs()
    for run_id in range(1, 65):
        cfg = configs[run_id]
        rows.append({"run_id": run_id, "score": 0.5 + 0.1 * cfg["A"], **cfg})
    effects = fit_main_effects(pd.DataFrame(rows), metric_col="score")
    assert effects["A"] == pytest.approx(0.2)


def test_analysis_uses_discovery_selected_oracle_not_confirmation_best():
    # Run 1 wins discovery but run 2 wins confirmation. Registered regret must use
    # run 1 evaluated on confirmation, allowing negative realized regret.
    rows = []
    configs = all_factorial_configs()
    for run_id in range(1, 65):
        for item in range(4):
            rows.append(
                {
                    "split": "discovery",
                    "sample_id": f"d{item}",
                    "run_id": run_id,
                    "score": 1.0 if run_id == 1 else 0.0,
                    **configs[run_id],
                }
            )
            conf_score = 1.0 if run_id == 2 else (0.4 if run_id == 1 else 0.0)
            rows.append(
                {
                    "split": "confirmation",
                    "sample_id": f"c{item}",
                    "run_id": run_id,
                    "score": conf_score,
                    **configs[run_id],
                }
            )

    result = analyze_task(
        pd.DataFrame(rows),
        prism_run_id=2,
        random_designs=[[1] + list(range(3, 18))],
        bootstrap_iterations=100,
        seed=7,
    )
    assert result["discovery_oracle_run_id"] == 1
    assert result["confirmation_best_run_id"] == 2
    assert result["oracle_score"] == pytest.approx(0.4)
    assert result["prism_regret"] == pytest.approx(-0.6)


def test_recommendation_lock_detects_manifest_or_recommendation_tampering(tmp_path):
    manifest = tmp_path / "discovery.jsonl"
    recommendations = tmp_path / "recommendations.json"
    lock_path = tmp_path / "recommendation-lock.json"
    manifest.write_text('{"split":"discovery"}\n', encoding="utf-8")
    recommendations.write_text('{"schema_extraction":{"run_id":1}}\n', encoding="utf-8")

    create_recommendation_lock(
        manifest, recommendations, lock_path, protocol_hash="abc"
    )
    assert_lock_matches(manifest, recommendations, lock_path, protocol_hash="abc")

    recommendations.write_text('{"schema_extraction":{"run_id":2}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="recommendation hash"):
        assert_lock_matches(manifest, recommendations, lock_path, protocol_hash="abc")


def test_frozen_public_data_splits_have_expected_counts_and_no_group_leakage():
    for task in ["schema_extraction", "grounded_qa", "intent_classification"]:
        groups = {}
        for split, expected in {
            "pilot": 20,
            "discovery": 80,
            "confirmation": 80,
        }.items():
            path = DATA_ROOT / "splits" / f"{task}_{split}.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            assert len(rows) == expected
            assert len({row["sample_id"] for row in rows}) == expected
            groups[split] = {row["group_id"] for row in rows}
        assert groups["pilot"].isdisjoint(groups["discovery"])
        assert groups["pilot"].isdisjoint(groups["confirmation"])
        assert groups["discovery"].isdisjoint(groups["confirmation"])

    for split, per_label in {"pilot": 2, "discovery": 8, "confirmation": 8}.items():
        path = DATA_ROOT / "splits" / f"intent_classification_{split}.jsonl"
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert {
            label: sum(row["target"] == label for row in rows)
            for label in INTENT_LABELS
        } == {label: per_label for label in INTENT_LABELS}
