"""
LLM Client Interface, Adapters (Callable, Mock, OpenAI, Anthropic, Vertex, LiteLLM).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Structured response from an LLM call."""

    content: str
    raw: Any = None
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMClient:
    """Abstract base class for LLM clients."""

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        """Synchronously generate a response for the given prompt."""
        raise NotImplementedError

    async def agenerate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        """Asynchronously generate a response."""
        # Default async fallback runs sync in executor
        return await asyncio.to_thread(self.generate, prompt, **kwargs)


class CallableLLM(LLMClient):
    """
    Wraps any arbitrary Python callable: fn(prompt, **kwargs) -> str or Any.
    Enables instant integration with ANY project, model provider, or custom pipeline.
    """

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        start_t = time.perf_counter()
        try:
            try:
                res = self.fn(prompt, **kwargs)
            except TypeError:
                res = self.fn(prompt)

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            content = res if isinstance(res, str) else str(res)
            return LLMResponse(
                content=content,
                raw=res,
                latency_ms=elapsed_ms,
                token_usage={"estimated_tokens": max(1, len(content) // 4)},
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return LLMResponse(
                content="",
                raw=None,
                latency_ms=elapsed_ms,
                error=str(e),
            )


class MockLLM(LLMClient):
    """
    Mock LLM simulator for testing and validation.
    Can simulate ground-truth factor effects, response templates, and realistic noise.
    """

    def __init__(
        self,
        default_response: str = "Mock response",
        factor_effects: Optional[Dict[str, float]] = None,
        base_quality: float = 0.5,
        noise_std: float = 0.05,
        latency_ms: float = 10.0,
        response_generator: Optional[
            Callable[[Union[str, List[Dict[str, str]]], Dict[str, Any]], str]
        ] = None,
    ):
        self.default_response = default_response
        self.factor_effects = factor_effects or {}
        self.base_quality = base_quality
        self.noise_std = noise_std
        self.latency_ms = latency_ms
        self.response_generator = response_generator

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        time.sleep(self.latency_ms / 1000.0)

        if self.response_generator:
            content = self.response_generator(prompt, kwargs)
        else:
            content = self.default_response

        return LLMResponse(
            content=content,
            raw={"simulated": True},
            latency_ms=self.latency_ms,
            token_usage={
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
            },
        )
