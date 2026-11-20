"""Golden render, metric-invariance, and negative out-of-envelope tests for Task 6 perturbations."""

from __future__ import annotations

import json

import pytest

from research.real_v4.perturbations import (
    PERTURBATION_REGISTRY_PATH,
    REGISTERED_TASKS,
    OutOfEnvelopePerturbationError,
    audit_perturbation_invariance,
    build_perturbation_environments,
    render_prompt_instance,
    score_task_response,
    write_perturbation_registry,
)
from research.real_v4.selection import FACTOR_IDS


def _fixture_items_and_outputs():
    return {
        "re3d_entity_extraction": (
            {
                "sample_id": "s1",
                "input_text": "Chancellor Friedrich Merz arrived in Berlin on Friday with $25 million.",
                "target": {
                    "person": ["Friedrich Merz"],
                    "organization": [],
                    "location": ["Berlin"],
                    "money": ["$25 million"],
                    "temporal": ["Friday"],
                },
            },
            [
                '{"person": ["Friedrich Merz"], "organization": [], "location": ["Berlin"], "money": ["$25 million"], "temporal": ["Friday"]}',
                '{"person": ["Friedrich Merz"], "organization": ["EU"], "location": [], "money": [], "temporal": []}',
                "malformed non-json",
                "",
            ],
        ),
        "squad2_grounded_qa": (
            {
                "sample_id": "s2",
                "context": "Penicillin was discovered by Alexander Fleming in 1928.",
                "question": "Who discovered penicillin?",
                "target": ["Alexander Fleming", "Fleming"],
            },
            [
                "FINAL_ANSWER: Alexander Fleming",
                "FINAL_ANSWER: UNANSWERABLE",
                "Joseph Lister",
                "",
            ],
        ),
        "banking77_intent": (
            {
                "sample_id": "s3",
                "input_text": "My physical debit card still hasn't come in the post.",
                "options": [
                    "card_arrival",
                    "card_not_working",
                    "declined_card_payment",
                ],
                "target": "card_arrival",
            },
            [
                "FINAL_ANSWER: card_arrival",
                "FINAL_ANSWER: declined_card_payment",
                "FINAL_ANSWER: invented_label",
                "",
            ],
        ),
        "gsm8k_math": (
            {
                "sample_id": "s4",
                "question": "Leo has 5 bags with 9 marbles each and gives away 7 marbles. How many remain?",
                "target": "38",
            },
            [
                "5 * 9 = 45. 45 - 7 = 38.\nFINAL_ANSWER: 38",
                "FINAL_ANSWER: 45",
                "No numbers here",
                "",
            ],
        ),
        "mbpp_python": (
            {
                "sample_id": "s5",
                "prompt": "Write `cube_int(n)` returning `n ** 3`.",
                "signature_hint": "def cube_int(n):",
                "target": ["assert cube_int(2) == 8", "assert cube_int(-3) == -27"],
            },
            [
                "```python\ndef cube_int(n):\n    return n ** 3\n```",
                "```python\ndef cube_int(n):\n    return abs(n ** 3)\n```",
                "def broken_syntax(:",
                "",
            ],
        ),
        "sciq_verification": (
            {
                "sample_id": "s6",
                "support": "Mitosis produces two genetically identical diploid daughter cells.",
                "question": "How many daughter cells does mitosis produce?",
                "options": {"A": "One", "B": "Two", "C": "Four", "D": "Eight"},
                "target": "B",
            },
            [
                "FINAL_ANSWER: B",
                "FINAL_ANSWER: C",
                "Ambiguous text",
                "",
            ],
        ),
    }


def test_perturbation_registry_reproducibility_and_all_task_invariance(tmp_path):
    reg1 = write_perturbation_registry(tmp_path / "reg1.json")
    reg2 = write_perturbation_registry(tmp_path / "reg2.json")
    assert reg1 == reg2
    assert (tmp_path / "reg1.json").read_text(encoding="utf-8") == (
        tmp_path / "reg2.json"
    ).read_text(encoding="utf-8")

    # Verify checked-in registry matches freshly generated deterministic payload
    checked_in = json.loads(PERTURBATION_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert checked_in["registry_sha256"] == reg1["registry_sha256"]

    specs = build_perturbation_environments()
    assert len(specs) == 9
    fixtures = _fixture_items_and_outputs()
    levels = {f: (idx % 2) for idx, f in enumerate(FACTOR_IDS)}

    for task_id in REGISTERED_TASKS:
        item, sample_outputs = fixtures[task_id]
        # First sample output is full score 1.0, malformed/empty is 0.0
        assert score_task_response(task_id, sample_outputs[0], item) == pytest.approx(
            1.0
        )
        assert score_task_response(task_id, sample_outputs[-1], item) == pytest.approx(
            0.0
        )

        ref_inst = render_prompt_instance(task_id, item, levels, env_id="ref")
        for spec in specs:
            pert_inst = render_prompt_instance(
                task_id, item, levels, env_id=spec.env_id
            )
            audit = audit_perturbation_invariance(
                ref_inst,
                pert_inst,
                ref_item=item,
                perturbed_item=item,
                pinned_model_version="v1.0-pinned",
                observed_model_version="v1.0-pinned",
                sample_outputs=sample_outputs,
                allow_disjoint_item=spec.uses_disjoint_item_group,
            )
            assert audit["semantic_invariant"] is True
            assert pert_inst.parent_prompt_sha256 == ref_inst.rendered_sha256


def test_negative_fixtures_reject_meaning_factor_target_options_context_and_version_drift():
    fixtures = _fixture_items_and_outputs()
    item, outs = fixtures["sciq_verification"]
    levels = {f: 1 for f in FACTOR_IDS}
    ref_inst = render_prompt_instance("sciq_verification", item, levels, env_id="ref")
    pert_inst = render_prompt_instance(
        "sciq_verification", item, levels, env_id="word_paraphrase_1"
    )

    # 1. Unregistered environment rejected
    with pytest.raises(
        OutOfEnvelopePerturbationError, match="Unregistered environment"
    ):
        render_prompt_instance(
            "sciq_verification", item, levels, env_id="adversarial_jailbreak"
        )

    # 2. Mutated target rejected
    bad_target_item = {**item, "target": "A"}
    bad_target_inst = render_prompt_instance(
        "sciq_verification", bad_target_item, levels, env_id="word_paraphrase_1"
    )
    with pytest.raises(OutOfEnvelopePerturbationError, match="Target answer mutated"):
        audit_perturbation_invariance(
            ref_inst,
            bad_target_inst,
            ref_item=item,
            perturbed_item=bad_target_item,
            pinned_model_version="v1",
            observed_model_version="v1",
            sample_outputs=outs,
        )

    # 3. Mutated answer options rejected
    bad_opts_item = {
        **item,
        "options": {"A": "Two", "B": "One", "C": "Four", "D": "Eight"},
    }
    bad_opts_inst = render_prompt_instance(
        "sciq_verification", bad_opts_item, levels, env_id="word_paraphrase_1"
    )
    with pytest.raises(
        OutOfEnvelopePerturbationError, match="Answer options inventory mutated"
    ):
        audit_perturbation_invariance(
            ref_inst,
            bad_opts_inst,
            ref_item=item,
            perturbed_item=bad_opts_item,
            pinned_model_version="v1",
            observed_model_version="v1",
            sample_outputs=outs,
        )

    # 4. Mutated factual context rejected
    bad_ctx_item = {**item, "support": "Meiosis produces four haploid cells."}
    bad_ctx_inst = render_prompt_instance(
        "sciq_verification", bad_ctx_item, levels, env_id="word_paraphrase_1"
    )
    with pytest.raises(OutOfEnvelopePerturbationError, match="Factual context mutated"):
        audit_perturbation_invariance(
            ref_inst,
            bad_ctx_inst,
            ref_item=item,
            perturbed_item=bad_ctx_item,
            pinned_model_version="v1",
            observed_model_version="v1",
            sample_outputs=outs,
        )

    # 5. Flipped factor level rejected
    flipped_levels = {**levels, "D": 0}
    flipped_inst = render_prompt_instance(
        "sciq_verification", item, flipped_levels, env_id="word_paraphrase_1"
    )
    with pytest.raises(OutOfEnvelopePerturbationError, match="Factor levels mutated"):
        audit_perturbation_invariance(
            ref_inst,
            flipped_inst,
            ref_item=item,
            perturbed_item=item,
            pinned_model_version="v1",
            observed_model_version="v1",
            sample_outputs=outs,
        )

    # 6. Provider model version drift rejected
    with pytest.raises(
        OutOfEnvelopePerturbationError, match="Provider model version drift"
    ):
        audit_perturbation_invariance(
            ref_inst,
            pert_inst,
            ref_item=item,
            perturbed_item=item,
            pinned_model_version="claude-3-5-haiku-20241022",
            observed_model_version="claude-3-5-haiku-20250401",
            sample_outputs=outs,
        )
