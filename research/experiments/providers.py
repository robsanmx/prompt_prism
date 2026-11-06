"""
Provider and Local-Model Adapters for Real-LLM v2 Inferences.
Implements standardized real model execution interfaces for:
1. OpenAI API (gpt-4o-mini, etc.)
2. Anthropic API (claude-3-5-haiku, etc.)
3. Local vLLM / HuggingFace endpoints (Llama-3.1-8B-Instruct)
4. Offline Test / Mock adapter (explicitly marked data_origin='simulated')

Captures complete provenance:
- Exact requested and returned model ID
- System fingerprint / backend revision
- Token usage (input / output)
- Execution latency & timestamps
- Retry counts and terminal failure status
"""

import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass


@dataclass
class ModelRequest:
    prompt: str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    seed: Optional[int] = 42
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    raw_text: str
    model_id: str
    request_id: str
    finish_reason: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    timestamp: str
    cost_usd: float
    data_origin: str  # "real" or "simulated"
    system_fingerprint: Optional[str] = None
    retry_count: int = 0
    error: Optional[str] = None
    model_version: Optional[str] = None
    tokens_thoughts: int = 0
    tokens_total: int = 0
    safety_ratings: Optional[Any] = None
    prompt_feedback: Optional[Dict[str, Any]] = None


class BaseProviderAdapter(ABC):
    """Abstract base class for inference providers."""

    def __init__(self, model_id: str, pricing_in_per_m: float = 0.0, pricing_out_per_m: float = 0.0):
        self.model_id = model_id
        self.pricing_in = pricing_in_per_m
        self.pricing_out = pricing_out_per_m

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.pricing_in + tokens_out * self.pricing_out) / 1_000_000.0

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute model generation and return full provenance response."""
        pass


class OpenAIProviderAdapter(BaseProviderAdapter):
    """Adapter for OpenAI Chat Completions API with frozen snapshots."""

    def __init__(self, model_id: str = "gpt-4o-mini-2024-07-18", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_id, pricing_in_per_m=0.150, pricing_out_per_m=0.600)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for real inference")

        import requests
        start_time = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            payload["seed"] = request.seed

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        latency = (time.perf_counter() - start_time) * 1000.0
        data = response.json()

        if response.status_code != 200:
            return ModelResponse(
                raw_text="",
                model_id=self.model_id,
                request_id=f"err_{uuid.uuid4().hex[:8]}",
                finish_reason="error",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                cost_usd=0.0,
                data_origin="real",
                error=data.get("error", {}).get("message", "Unknown API error"),
            )

        choice = data["choices"][0]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        return ModelResponse(
            raw_text=choice["message"]["content"],
            model_id=data.get("model", self.model_id),
            request_id=data.get("id", f"req_{uuid.uuid4().hex[:8]}"),
            finish_reason=choice.get("finish_reason", "stop"),
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cost_usd=self.calculate_cost(tokens_in, tokens_out),
            data_origin="real",
            system_fingerprint=data.get("system_fingerprint"),
        )


class AnthropicProviderAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Messages API."""

    def __init__(self, model_id: str = "claude-3-5-haiku-20241022", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_id, pricing_in_per_m=0.800, pricing_out_per_m=4.000)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required for real inference")

        import requests
        start_time = time.perf_counter()
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
        latency = (time.perf_counter() - start_time) * 1000.0
        data = response.json()

        if response.status_code != 200:
            return ModelResponse(
                raw_text="",
                model_id=self.model_id,
                request_id=f"err_{uuid.uuid4().hex[:8]}",
                finish_reason="error",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                cost_usd=0.0,
                data_origin="real",
                error=data.get("error", {}).get("message", "Unknown Anthropic API error"),
            )

        text_content = ""
        for blk in data.get("content", []):
            if blk.get("type") == "text":
                text_content += blk.get("text", "")

        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        return ModelResponse(
            raw_text=text_content,
            model_id=data.get("model", self.model_id),
            request_id=data.get("id", f"req_{uuid.uuid4().hex[:8]}"),
            finish_reason=data.get("stop_reason", "end_turn"),
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cost_usd=self.calculate_cost(tokens_in, tokens_out),
            data_origin="real",
        )


class LocalVLLMAdapter(BaseProviderAdapter):
    """Adapter for local vLLM / OpenAI-compatible local model server."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", base_url: str = "http://localhost:8000/v1", **kwargs):
        super().__init__(model_id, pricing_in_per_m=0.0, pricing_out_per_m=0.0)
        self.base_url = base_url

    def generate(self, request: ModelRequest) -> ModelResponse:
        import requests
        start_time = time.perf_counter()
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
        }
        try:
            res = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=60)
            latency = (time.perf_counter() - start_time) * 1000.0
            data = res.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return ModelResponse(
                raw_text=choice["message"]["content"],
                model_id=self.model_id,
                request_id=data.get("id", f"vllm_{uuid.uuid4().hex[:8]}"),
                finish_reason=choice.get("finish_reason", "stop"),
                tokens_input=usage.get("prompt_tokens", 0),
                tokens_output=usage.get("completion_tokens", 0),
                latency_ms=latency,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                cost_usd=0.0,
                data_origin="real",
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return ModelResponse(
                raw_text="",
                model_id=self.model_id,
                request_id=f"vllm_err_{uuid.uuid4().hex[:8]}",
                finish_reason="connection_error",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                cost_usd=0.0,
                data_origin="real",
                error=str(e),
            )


class GeminiProviderAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini API (generateContent endpoint)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        pricing_in_per_m: float = 0.300,
        pricing_out_per_m: float = 2.500,
        max_retries: int = 4,
        **kwargs,
    ):
        model = model_id or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        super().__init__(model, pricing_in_per_m=pricing_in_per_m, pricing_out_per_m=pricing_out_per_m)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.max_retries = max_retries

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable required for Gemini inference")

        import requests
        start_time = time.perf_counter()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "topP": request.top_p,
                "thinkingConfig": {
                    "thinkingBudget": request.extra_params.get("thinking_budget", 0),
                },
            },
        }
        if request.seed is not None:
            payload["generationConfig"]["seed"] = request.seed
        if request.system_prompt:
            payload["system_instruction"] = {"parts": [{"text": request.system_prompt}]}

        retry_count = 0
        response = None
        data = {}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 or response.status_code >= 500:
                    retry_count += 1
                    sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                    continue
                data = response.json()
                break
            except Exception as exc:
                retry_count += 1
                if attempt == self.max_retries - 1:
                    latency = (time.perf_counter() - start_time) * 1000.0
                    return ModelResponse(
                        raw_text="",
                        model_id=self.model_id,
                        request_id=f"gemini_err_{uuid.uuid4().hex[:8]}",
                        finish_reason="network_error",
                        tokens_input=0,
                        tokens_output=0,
                        latency_ms=latency,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        cost_usd=0.0,
                        data_origin="real",
                        error=str(exc),
                        retry_count=retry_count,
                    )
                time.sleep((2 ** attempt) + 0.2)

        latency = (time.perf_counter() - start_time) * 1000.0

        if response is None or response.status_code != 200:
            return ModelResponse(
                raw_text="",
                model_id=self.model_id,
                request_id=f"gemini_err_{uuid.uuid4().hex[:8]}",
                finish_reason="error",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                cost_usd=0.0,
                data_origin="real",
                error=data.get("error", {}).get("message", f"HTTP {response.status_code if response else 'ERR'}: Unknown Gemini API error"),
                retry_count=retry_count,
            )

        candidates = data.get("candidates", [])
        raw_text = ""
        finish_reason = "stop"
        if candidates:
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "stop")
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            if parts:
                raw_text = "".join(part.get("text", "") for part in parts)

        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0)
        tokens_out = usage.get("candidatesTokenCount", 0)
        tokens_thoughts = usage.get("thoughtsTokenCount", 0)
        tokens_total = usage.get(
            "totalTokenCount", tokens_in + tokens_out + tokens_thoughts
        )

        return ModelResponse(
            raw_text=raw_text,
            model_id=self.model_id,
            request_id=data.get("responseId", f"gemini_{uuid.uuid4().hex[:8]}"),
            finish_reason=finish_reason,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cost_usd=self.calculate_cost(tokens_in, tokens_out + tokens_thoughts),
            data_origin="real",
            model_version=data.get("modelVersion"),
            tokens_thoughts=tokens_thoughts,
            tokens_total=tokens_total,
            safety_ratings=candidates[0].get("safetyRatings") if candidates else None,
            prompt_feedback=data.get("promptFeedback"),
        )


class MockProviderAdapter(BaseProviderAdapter):
    """
    Offline testing adapter for unit and integration tests.
    Explicitly tags data_origin='simulated' to ensure empirical loaders reject it.
    """

    def __init__(self, model_id: str = "mock-model", response_text: str = '{"buyer": "Acme", "target": "Beta"}', **kwargs):
        super().__init__(model_id, pricing_in_per_m=0.0, pricing_out_per_m=0.0)
        self.response_text = response_text

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            raw_text=self.response_text,
            model_id=self.model_id,
            request_id=f"mock_{uuid.uuid4().hex[:8]}",
            finish_reason="stop",
            tokens_input=len(request.prompt.split()),
            tokens_output=len(self.response_text.split()),
            latency_ms=12.5,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cost_usd=0.0,
            data_origin="simulated", # Guaranteed to be rejected by empirical v2 loader
        )
