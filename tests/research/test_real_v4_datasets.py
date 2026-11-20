"""Automated license, hash, uniqueness, target, prompt-rendering, and zero-v3-leakage tests for Task 8."""

from __future__ import annotations

import json

import pytest

from research.data.real_v4.build import (
    AUDIT_REPORT_PATH,
    MANIFEST_PATH,
    PARTITION_ORDER,
    PARTITION_SIZES,
    V4_SPLITS_DIR,
    audit_real_v4_datasets,
)
from research.real_v3.datasets import _read_jsonl
from research.real_v4.perturbations import (
    REGISTERED_TASKS,
    render_prompt_instance,
    score_task_response,
)
from research.real_v4.selection import FACTOR_IDS


def test_real_v4_dataset_manifest_and_audit_pass_all_gates():
    assert MANIFEST_PATH.exists()
    assert AUDIT_REPORT_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    saved_audit = json.loads(AUDIT_REPORT_PATH.read_text(encoding="utf-8"))
    live_audit = audit_real_v4_datasets(manifest, V4_SPLITS_DIR)

    assert live_audit["audit_passed"] is True
    assert live_audit["error_count"] == 0
    assert live_audit["total_items_across_tasks"] == 6 * sum(PARTITION_SIZES.values())
    assert live_audit["audit_sha256"] == saved_audit["audit_sha256"]

    for task_id in REGISTERED_TASKS:
        t_info = live_audit["tasks"][task_id]
        assert t_info["cross_partition_group_overlaps"] == 0
        for part in PARTITION_ORDER:
            assert t_info["partition_counts"][part] == PARTITION_SIZES[part]


def test_sample_items_render_cleanly_and_gold_targets_score_1_0():
    levels = {f: 1 for f in FACTOR_IDS}
    for task_id in REGISTERED_TASKS:
        pilot_file = V4_SPLITS_DIR / f"{task_id}_pilot.jsonl"
        rows = list(_read_jsonl(pilot_file))
        assert len(rows) == 20
        item = rows[0]

        # Render prompt on real dataset item
        rendered = render_prompt_instance(task_id, item, levels, env_id="ref")
        assert rendered.task_id == task_id
        assert len(rendered.user_text) > 40

        # Construct synthetic oracle response from item["target"] and verify deterministic score == 1.0
        if task_id == "re3d_entity_extraction":
            oracle_out = json.dumps(item["target"])
            assert score_task_response(task_id, oracle_out, item) == pytest.approx(1.0)
        elif task_id == "squad2_grounded_qa":
            oracle_out = (
                "FINAL_ANSWER: UNANSWERABLE"
                if item.get("is_unanswerable")
                else f"FINAL_ANSWER: {item['target'][0]}"
            )
            assert score_task_response(task_id, oracle_out, item) == pytest.approx(1.0)
        elif task_id in ("banking77_intent", "gsm8k_math", "sciq_verification"):
            oracle_out = f"FINAL_ANSWER: {item['target']}"
            assert score_task_response(task_id, oracle_out, item) == pytest.approx(1.0)
        elif task_id == "mbpp_python":
            assert item["signature_hint"].startswith("def ")
            assert all(t.startswith("assert ") for t in item["target"])


def test_audit_detects_tampered_group_leakage_or_hash_mismatch(tmp_path):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for p in V4_SPLITS_DIR.glob("*.jsonl"):
        (tmp_path / p.name).write_bytes(p.read_bytes())

    # Tamper one split row to duplicate a group_id from pilot into clean_confirmation
    pilot_rows = list(_read_jsonl(tmp_path / "gsm8k_math_pilot.jsonl"))
    conf_rows = list(_read_jsonl(tmp_path / "gsm8k_math_clean_confirmation.jsonl"))
    conf_rows[0]["group_id"] = pilot_rows[0]["group_id"]
    (tmp_path / "gsm8k_math_clean_confirmation.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in conf_rows) + "\n",
        encoding="utf-8",
    )

    bad_audit = audit_real_v4_datasets(manifest, tmp_path)
    assert bad_audit["audit_passed"] is False
    assert any("split_hash_mismatch" in e for e in bad_audit["errors"])
    assert any(
        "group_overlap:gsm8k_math:pilot:clean_confirmation" in e
        for e in bad_audit["errors"]
    )
