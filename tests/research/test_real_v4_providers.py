"""Network-mocked hosted adapter, local Qwen provenance, canonical semantics, and repeatability tests (Task 9)."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from research.real_v4.model_provenance import (
    MODEL_REGISTRY_PATH,
    ProvenanceEnvelopeError,
    evaluate_pilot_repeatability,
    load_qwen_inventory,
    verify_local_qwen_files_on_disk,
    write_model_registry,
)
from research.real_v4.providers import (
    CanonicalAdapterRequest,
    RealV4AnthropicAdapter,
    RealV4GeminiAdapter,
    RealV4LocalMistralMLXAdapter,
    RealV4LocalQwenMLXAdapter,
)


def _sample_request() -> CanonicalAdapterRequest:
    return CanonicalAdapterRequest(
        task_id="gsm8k_math",
        sample_id="gsm8k:0001",
        env_id="ref",
        system_text="You are a precision math specialist.",
        user_text="[INSTRUCTIONS]\nSolve accurately.\n\n[TASK INPUT]\nProblem: What is 7 * 8?",
        temperature=0.0,
        max_tokens=512,
        seed=42,
        enable_thinking=False,
    )


def test_canonical_render_semantics_equivalent_across_all_three_adapters(tmp_path):
    reg1 = write_model_registry(tmp_path / "m1.json")
    reg2 = write_model_registry(tmp_path / "m2.json")
    assert reg1["registry_sha256"] == reg2["registry_sha256"]
    checked = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert checked["registry_sha256"] == reg1["registry_sha256"]

    req = _sample_request()
    gem = RealV4GeminiAdapter(api_key="dummy")
    ant = RealV4AnthropicAdapter(api_key="dummy")
    mistral = RealV4LocalMistralMLXAdapter()
    qwen = RealV4LocalQwenMLXAdapter()

    sem_gem = gem.extract_canonical_semantics(gem.serialize_request_payload(req))
    sem_ant = ant.extract_canonical_semantics(ant.serialize_request_payload(req))
    sem_mis = mistral.extract_canonical_semantics(
        mistral.serialize_request_payload(req)
    )
    sem_qwen = qwen.extract_canonical_semantics(qwen.serialize_request_payload(req))

    assert (
        sem_gem
        == sem_ant
        == sem_mis
        == sem_qwen
        == {
            "system_role_text": req.system_text,
            "user_role_text": req.user_text,
            "temperature": 0.0,
            "max_tokens": 512,
            "thinking_enabled": False,
        }
    )


def test_mocked_gemini_and_anthropic_provenance_and_version_drift_detection():
    req = _sample_request()

    # 1. Gemini mocked transport: pinned match vs version drift vs missing modelVersion
    def gemini_handler(http_req: httpx.Request) -> httpx.Response:
        body = json.loads(http_req.content.decode("utf-8"))
        assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
        ver = (
            "gemini-2.5-flash-001"
            if "0001" in body["contents"][0]["parts"][0]["text"]
            or http_req.headers.get("X-Drift") != "1"
            else "drifted"
        )
        return httpx.Response(
            200,
            json={
                "responseId": "gem_resp_123",
                "modelVersion": ver,
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "FINAL_ANSWER: 56"}]},
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 0,
                },
            },
        )

    client_gem = httpx.Client(transport=httpx.MockTransport(gemini_handler))
    gem_ok = RealV4GeminiAdapter(
        pinned_model_version="gemini-2.5-flash-001",
        api_key="test_key",
        http_client=client_gem,
    )
    res_gem = gem_ok.execute(req)
    assert res_gem.raw_text == "FINAL_ANSWER: 56"
    assert res_gem.provenance.response_id == "gem_resp_123"
    assert res_gem.provenance.returned_model_version == "gemini-2.5-flash-001"
    assert res_gem.provenance.within_pinned_envelope is True

    # Pinned to a different version marks within_pinned_envelope=False with PROVIDER_VERSION_DRIFT
    gem_drift = RealV4GeminiAdapter(
        pinned_model_version="gemini-2.5-flash-000-old",
        api_key="test_key",
        http_client=client_gem,
    )
    res_gem_drift = gem_drift.execute(req)
    assert res_gem_drift.provenance.within_pinned_envelope is False
    assert res_gem_drift.provenance.envelope_breach_reason == "PROVIDER_VERSION_DRIFT"

    # Missing modelVersion raises ProvenanceEnvelopeError
    bad_gem_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"responseId": "id_only"})
        )
    )
    with pytest.raises(ProvenanceEnvelopeError, match="modelVersion"):
        RealV4GeminiAdapter(api_key="test_key", http_client=bad_gem_client).execute(req)

    # 2. Anthropic mocked transport including bounded retry accounting (zero silent retries)
    attempts = {"n": 0}

    def anthropic_handler(_r: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(
            200,
            json={
                "id": "msg_ant_999",
                "model": "claude-3-5-haiku-20241022",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "FINAL_ANSWER: 56"}],
                "usage": {"input_tokens": 28, "output_tokens": 6},
            },
        )

    client_ant = httpx.Client(transport=httpx.MockTransport(anthropic_handler))
    ant = RealV4AnthropicAdapter(
        pinned_model_version="claude-3-5-haiku-20241022",
        api_key="test_ant",
        http_client=client_ant,
    )
    res_ant = ant.execute(req)
    assert res_ant.raw_text == "FINAL_ANSWER: 56"
    assert res_ant.provenance.retry_count == 1
    assert res_ant.provenance.within_pinned_envelope is True


def test_local_qwen_provenance_and_repeatability_policy():
    inv = load_qwen_inventory()
    assert len(inv["files"]) == 9

    # Verify local Qwen cache files on disk if present on this developer machine
    from pathlib import Path

    snap_path = Path(inv["cache_snapshot"]).expanduser()
    if snap_path.exists():
        verified = verify_local_qwen_files_on_disk(
            snap_path, verify_full_safetensors_bytes=False
        )
        assert verified["weights_inventory_sha256"] == inv["weights_inventory_sha256"]

    req = _sample_request()
    qwen_stub = RealV4LocalQwenMLXAdapter(
        mlx_runner_fn=lambda payload: (
            "FINAL_ANSWER: 56",
            {
                "snapshot_revision": inv["revision"],
                "tokens_input": 32,
                "tokens_output": 4,
                "peak_memory_mb": 5312.5,
                "runtime_versions": {"mlx": "0.24.0", "mlx_lm": "0.24.0"},
            },
        )
    )
    res_qwen = qwen_stub.execute(req)
    assert res_qwen.provenance.within_pinned_envelope is True
    assert (
        res_qwen.provenance.local_weights_inventory_sha256
        == inv["weights_inventory_sha256"]
    )
    assert res_qwen.provenance.tokenizer_sha256 == inv["tokenizer_sha256"]
    assert res_qwen.provenance.peak_memory_mb == pytest.approx(5312.5)

    # Repeatability policy: deterministic -> repeat_count=1, nondeterministic -> repeat_count=3
    det_pol = evaluate_pilot_repeatability(
        [["A", "B", "C"], ["A", "B", "C"], ["A", "B", "C"]],
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    )
    assert det_pol.nondeterminism_detected is False
    assert det_pol.frozen_repeat_count == 1

    nondet_pol = evaluate_pilot_repeatability(
        [["A", "B", "C"], ["A", "B_alt", "C"], ["A", "B", "C"]],
        [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    )
    assert nondet_pol.nondeterminism_detected is True
    assert nondet_pol.frozen_repeat_count == 3
    assert nondet_pol.aggregation_rule == "mean_across_3_identical_repeats"


@pytest.mark.skipif(
    os.getenv("RUN_REAL_V4_LIVE_PILOT") != "1",
    reason="Quarantined live cross-provider pilot requires explicit human spend & OSF gate unlock",
)
def test_quarantined_live_provider_smoke_pilot():
    from dotenv import load_dotenv

    load_dotenv()
    req = _sample_request()
    gem = RealV4GeminiAdapter()
    res = gem.execute(req)
    assert res.provenance.response_id
    assert res.provenance.returned_model_version
    assert res.provenance.within_pinned_envelope is True

    qwen = RealV4LocalQwenMLXAdapter()
    res_qwen = qwen.execute(req)
    assert res_qwen.provenance.response_id
    assert res_qwen.provenance.within_pinned_envelope is True
    assert "56" in res_qwen.raw_text
