"""
Tests for Real-LLM v2 Provider Adapters.
Mocks all network calls to ensure zero external spending during test runs.
Verifies provenance capture, origin tagging, failure handling, and cost calculations.
"""

from unittest.mock import MagicMock, patch
import pytest

from research.experiments.providers import (
    AnthropicProviderAdapter,
    GeminiProviderAdapter,
    LocalVLLMAdapter,
    MockProviderAdapter,
    ModelRequest,
    ModelResponse,
    OpenAIProviderAdapter,
)


def test_mock_provider_origin_tagging():
    """Verify that MockProviderAdapter always marks data_origin='simulated'."""
    adapter = MockProviderAdapter(model_id="test-mock")
    req = ModelRequest(prompt="Hello", model_id="test-mock")
    res = adapter.generate(req)
    assert res.data_origin == "simulated"
    assert res.finish_reason == "stop"


@patch("requests.post")
def test_openai_adapter_provenance_and_mock(mock_post):
    """Verify OpenAI adapter returns full provenance fields without making live API calls."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-test-123",
        "model": "gpt-4o-mini-2024-07-18",
        "system_fingerprint": "fp_test_456",
        "choices": [{"message": {"content": '{"buyer": "Acme"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 35},
    }
    mock_post.return_value = mock_response

    adapter = OpenAIProviderAdapter(api_key="test-key-mock")
    req = ModelRequest(prompt="Extract buyer", model_id="gpt-4o-mini-2024-07-18", seed=42)
    res = adapter.generate(req)

    assert res.data_origin == "real"
    assert res.model_id == "gpt-4o-mini-2024-07-18"
    assert res.request_id == "chatcmpl-test-123"
    assert res.system_fingerprint == "fp_test_456"
    assert res.tokens_input == 120
    assert res.tokens_output == 35
    assert res.raw_text == '{"buyer": "Acme"}'
    assert res.cost_usd > 0.0


@patch("requests.post")
def test_anthropic_adapter_provenance_and_mock(mock_post):
    """Verify Anthropic adapter returns full provenance fields without making live API calls."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_test_789",
        "model": "claude-3-5-haiku-20241022",
        "content": [{"type": "text", "text": "Answer: 42"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 15},
    }
    mock_post.return_value = mock_response

    adapter = AnthropicProviderAdapter(api_key="test-key-mock")
    req = ModelRequest(prompt="What is 42?", model_id="claude-3-5-haiku-20241022")
    res = adapter.generate(req)

    assert res.data_origin == "real"
    assert res.model_id == "claude-3-5-haiku-20241022"
    assert res.request_id == "msg_test_789"
    assert res.tokens_input == 200
    assert res.tokens_output == 15
    assert res.raw_text == "Answer: 42"


@patch("requests.post")
def test_local_vllm_adapter_connection_error(mock_post):
    """Verify local vLLM adapter returns structured connection error when endpoint is down."""
    mock_post.side_effect = ConnectionError("Could not connect to localhost:8000")

    adapter = LocalVLLMAdapter(model_id="meta-llama/Llama-3.1-8B-Instruct")
    req = ModelRequest(prompt="Hello local", model_id="meta-llama/Llama-3.1-8B-Instruct")
    res = adapter.generate(req)

    assert res.data_origin == "real"
    assert res.finish_reason == "connection_error"
    assert "Could not connect" in res.error


@patch("requests.post")
def test_gemini_adapter_provenance_and_mock(mock_post):
    """Verify Gemini adapter returns full provenance fields without making live API calls."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": '{"buyer": "Acme Corp", "target": "Beta LLC"}'}]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 150,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 10,
            "totalTokenCount": 200,
        },
        "modelVersion": "gemini-2.5-flash-20260901",
        "responseId": "response-real-123",
    }
    mock_post.return_value = mock_response

    adapter = GeminiProviderAdapter(model_id="gemini-1.5-flash", api_key="test-gemini-key")
    req = ModelRequest(prompt="Extract buyer", model_id="gemini-1.5-flash", seed=42)
    res = adapter.generate(req)

    assert res.data_origin == "real"
    assert res.model_id == "gemini-1.5-flash"
    assert res.model_version == "gemini-2.5-flash-20260901"
    assert res.request_id == "response-real-123"
    assert res.finish_reason == "STOP"
    assert res.tokens_input == 150
    assert res.tokens_output == 40
    assert res.tokens_thoughts == 10
    assert res.tokens_total == 200
    assert res.raw_text == '{"buyer": "Acme Corp", "target": "Beta LLC"}'
    assert res.cost_usd > 0.0
    payload = mock_post.call_args.kwargs["json"]
    assert payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
