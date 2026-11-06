"""
Tests for RealExperimentRunner with resumability and provenance.
"""

from research.experiments.manifest import ExperimentManifest
from research.experiments.providers import MockProviderAdapter
from research.experiments.real_runner import RealExperimentRunner


def test_real_runner_resumability_and_provenance(tmp_path):
    manifest_path = tmp_path / "real_manifest.jsonl"
    manifest = ExperimentManifest("exp_real_test", manifest_path=manifest_path)

    call_count = {"count": 0}

    class CountingMockAdapter(MockProviderAdapter):
        def generate(self, request):
            call_count["count"] += 1
            res = super().generate(request)
            res.data_origin = "real"  # Mark as real for test loader
            return res

    adapter = CountingMockAdapter(model_id="mock-qwen-7b")

    def mock_evaluator(resp: str, item: dict):
        return {"field_f1": 0.95}

    def render_prompt(f_levels: dict, item: dict):
        return f"Prompt-{f_levels.get('A', 0)}-{item['id']}"

    runner = RealExperimentRunner(
        manifest=manifest,
        adapter=adapter,
        task_id="schema_extraction",
        evaluator_fn=mock_evaluator,
        seed=42,
    )

    configs = [(1, {"A": 0}), (2, {"A": 1})]
    items = [{"id": f"item_{i}"} for i in range(4)]  # 2 configs * 4 items = 8 trials

    # Run 1: first config only (4 trials)
    stats1 = runner.run_cell(
        configurations=configs[:1],
        items=items,
        render_prompt_fn=render_prompt,
        split="discovery",
        randomize_order=False,
    )
    assert stats1["executed_calls"] == 4
    assert stats1["cache_hits"] == 0
    assert call_count["count"] == 4

    # Run 2: both configs (total 8 trials, 4 should be cache hits)
    stats2 = runner.run_cell(
        configurations=configs,
        items=items,
        render_prompt_fn=render_prompt,
        split="discovery",
        randomize_order=False,
    )
    assert stats2["executed_calls"] == 4
    assert stats2["cache_hits"] == 4
    assert call_count["count"] == 8

    # Verify records in manifest
    assert len(manifest.records) == 8
    for r in manifest.records:
        assert r.task_id == "schema_extraction"
        assert r.metrics["field_f1"] == 0.95
        assert r.split == "discovery"
        assert r.data_origin == "real"
