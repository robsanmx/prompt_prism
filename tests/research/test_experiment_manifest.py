"""
Tests for ExperimentManifest and ResumableExperimentRunner in research/experiments/.
Verifies provenance capture, interruption resumption without duplicate calls,
cache key sensitivity, and discovery/confirmation split isolation.
"""

from pathlib import Path
import pytest

from research.experiments.manifest import ExperimentManifest, ProvenanceRecord
from research.experiments.run import ResumableExperimentRunner


def test_cache_key_sensitivity():
    """Verify that changing model, prompt, item, or seed yields distinct cache keys."""
    base_key = ExperimentManifest.compute_cache_key(
        model_id="gpt-4o-mini",
        prompt_text="Extract entities: Acme Corp",
        sample_id="item_001",
        seed=42,
    )

    # Prompt change
    key_prompt = ExperimentManifest.compute_cache_key(
        model_id="gpt-4o-mini",
        prompt_text="Extract entities: Beta LLC",
        sample_id="item_001",
        seed=42,
    )
    assert key_prompt != base_key

    # Model change
    key_model = ExperimentManifest.compute_cache_key(
        model_id="claude-3-5-sonnet",
        prompt_text="Extract entities: Acme Corp",
        sample_id="item_001",
        seed=42,
    )
    assert key_model != base_key

    # Item change
    key_item = ExperimentManifest.compute_cache_key(
        model_id="gpt-4o-mini",
        prompt_text="Extract entities: Acme Corp",
        sample_id="item_002",
        seed=42,
    )
    assert key_item != base_key

    # Seed change
    key_seed = ExperimentManifest.compute_cache_key(
        model_id="gpt-4o-mini",
        prompt_text="Extract entities: Acme Corp",
        sample_id="item_001",
        seed=43,
    )
    assert key_seed != base_key


def test_interruption_resumption_and_no_duplicate_calls(tmp_path):
    """Verify that an interrupted experiment resumes using cache without duplicate paid calls."""
    manifest_file = tmp_path / "test_manifest.jsonl"
    manifest = ExperimentManifest(experiment_id="exp_resume_test", manifest_path=manifest_file)

    call_counts = {"count": 0}

    def mock_llm(prompt: str) -> str:
        call_counts["count"] += 1
        return f"Response to: {prompt}"

    def mock_evaluator(resp: str, item: dict) -> dict:
        return {"f1": 0.85}

    def render_prompt(f_levels: dict, item: dict) -> str:
        return f"Prompt-{f_levels.get('A', 0)}-{item['id']}"

    runner = ResumableExperimentRunner(
        manifest=manifest,
        task_id="test_task",
        model_id="mock_model",
        provider="mock_provider",
        evaluator_fn=mock_evaluator,
        client_fn=mock_llm,
        seed=123,
    )

    configs = [(1, {"A": 0}), (2, {"A": 1})]
    items = [{"id": f"item_{i}"} for i in range(5)]  # 2 configs * 5 items = 10 tasks

    # First run: run only the first 4 tasks by providing a subset
    initial_stats = runner.run_experiment(
        configurations=configs[:1],  # 1 config * 5 items = 5 tasks
        items=items,
        render_prompt_fn=render_prompt,
        split="discovery",
        randomize_order=False,
    )
    assert initial_stats["executed_calls"] == 5
    assert initial_stats["cache_hits"] == 0
    assert call_counts["count"] == 5

    # Simulate process restart by reloading manifest from disk
    manifest_reloaded = ExperimentManifest(experiment_id="exp_resume_test", manifest_path=manifest_file)
    assert len(manifest_reloaded.records) == 5

    runner_resumed = ResumableExperimentRunner(
        manifest=manifest_reloaded,
        task_id="test_task",
        model_id="mock_model",
        provider="mock_provider",
        evaluator_fn=mock_evaluator,
        client_fn=mock_llm,
        seed=123,
    )

    # Second run: run all 10 tasks
    resumed_stats = runner_resumed.run_experiment(
        configurations=configs,  # 2 configs * 5 items = 10 tasks
        items=items,
        render_prompt_fn=render_prompt,
        split="discovery",
        randomize_order=False,
    )

    # Exactly 5 should be cache hits, and only 5 newly executed
    assert resumed_stats["cache_hits"] == 5
    assert resumed_stats["executed_calls"] == 5
    assert call_counts["count"] == 10  # Total 10 calls, zero duplicate calls!

    integrity = manifest_reloaded.verify_integrity()
    assert integrity["total_records"] == 10
    assert integrity["has_duplicate_keys"] is False


def test_discovery_and_confirmation_isolation(tmp_path):
    """Verify that confirmation data cannot be loaded via the discovery path."""
    manifest = ExperimentManifest(experiment_id="exp_isolation_test")

    # Add 2 discovery records and 2 confirmation records
    for i in range(2):
        rec_d = ProvenanceRecord(
            trial_id=f"t_disc_{i}",
            experiment_id="exp_isolation_test",
            git_commit="abc",
            task_id="task",
            model_id="model",
            provider="prov",
            run_id=1,
            factor_levels={"A": 0},
            sample_id=f"d_item_{i}",
            split="discovery",
            prompt_text="p",
            prompt_hash="h1",
            response_text="r",
            response_hash="h2",
            metrics={"m": 1.0},
            tokens_input=10,
            tokens_output=10,
            cost_usd=0.001,
            latency_ms=100.0,
            timestamp="2026-09-26T00:00:00Z",
            seed=42,
        )
        manifest.add_record(rec_d, persist=False)

        rec_c = ProvenanceRecord(
            trial_id=f"t_conf_{i}",
            experiment_id="exp_isolation_test",
            git_commit="abc",
            task_id="task",
            model_id="model",
            provider="prov",
            run_id=1,
            factor_levels={"A": 1},
            sample_id=f"c_item_{i}",
            split="confirmation",
            prompt_text="p",
            prompt_hash="h1",
            response_text="r",
            response_hash="h2",
            metrics={"m": 1.0},
            tokens_input=10,
            tokens_output=10,
            cost_usd=0.001,
            latency_ms=100.0,
            timestamp="2026-09-26T00:00:00Z",
            seed=42,
        )
        manifest.add_record(rec_c, persist=False)

    # 1. Discovery dataframe contains strictly discovery trials
    disc_df = manifest.get_discovery_dataframe()
    assert len(disc_df) == 2
    assert (disc_df["split"] == "discovery").all()
    assert set(disc_df["sample_id"]) == {"d_item_0", "d_item_1"}

    # 2. Loading confirmation dataframe without unlock token raises PermissionError
    with pytest.raises(PermissionError, match="Confirmation split is locked"):
        manifest.get_confirmation_dataframe(unlock_token="")

    with pytest.raises(PermissionError, match="Confirmation split is locked"):
        manifest.get_confirmation_dataframe(unlock_token="WRONG_TOKEN")

    # 3. Unlocking with correct token returns confirmation rows
    conf_df = manifest.get_confirmation_dataframe(unlock_token="CONFIRMATORY_ANALYSIS_UNLOCKED")
    assert len(conf_df) == 2
    assert (conf_df["split"] == "confirmation").all()
    assert set(conf_df["sample_id"]) == {"c_item_0", "c_item_1"}


def test_real_v2_manifest_quarantine(tmp_path):
    """Verify that v2 empirical loader strictly rejects simulated and v1 manifests."""
    from research.experiments.manifest import load_real_v2_manifest

    # 1. Simulated manifest must be rejected by load_real_v2_manifest
    sim_path = tmp_path / "simulated_manifest.jsonl"
    sim_manifest = ExperimentManifest("exp_sim", manifest_path=sim_path)
    sim_rec = ProvenanceRecord(
        trial_id="t_sim_1",
        experiment_id="exp_sim",
        git_commit="git123",
        task_id="task1",
        model_id="model1",
        provider="prov1",
        run_id=1,
        factor_levels={"A": 0},
        sample_id="s1",
        split="discovery",
        prompt_text="p",
        prompt_hash="h1",
        response_text="r",
        response_hash="h2",
        metrics={"score": 0.5},
        tokens_input=10,
        tokens_output=10,
        cost_usd=0.001,
        latency_ms=50.0,
        timestamp="2026-09-26T00:00:00Z",
        seed=1,
        data_origin="simulated",
    )
    sim_manifest.add_record(sim_rec)

    with pytest.raises(ValueError, match="Origin mismatch.*Quarantined simulated fixtures cannot enter empirical analysis"):
        load_real_v2_manifest(sim_path)

    # 2. Existing confirmatory_study_v1 manifest must be detected and rejected as simulated
    v1_path = Path("research/results/confirmatory_manifest.jsonl")
    if v1_path.exists():
        with pytest.raises(ValueError, match="Origin mismatch.*Quarantined simulated fixtures cannot enter empirical analysis"):
            load_real_v2_manifest(v1_path)

    # 3. Real manifest with data_origin="real" loads successfully
    real_path = tmp_path / "real_manifest.jsonl"
    real_manifest = ExperimentManifest("real_v2_test", manifest_path=real_path)
    real_rec = ProvenanceRecord(
        trial_id="t_real_1",
        experiment_id="real_v2_test",
        git_commit="git123",
        task_id="task1",
        model_id="model1",
        provider="prov1",
        run_id=1,
        factor_levels={"A": 0},
        sample_id="s1",
        split="discovery",
        prompt_text="p",
        prompt_hash="h1",
        response_text="r",
        response_hash="h2",
        metrics={"score": 0.8},
        tokens_input=10,
        tokens_output=10,
        cost_usd=0.001,
        latency_ms=50.0,
        timestamp="2026-09-26T00:00:00Z",
        seed=1,
        data_origin="real",
    )
    real_manifest.add_record(real_rec)

    loaded = load_real_v2_manifest(real_path)
    assert len(loaded.records) == 1
    assert loaded.records[0].data_origin == "real"

