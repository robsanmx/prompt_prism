"""Canonical Real-v4 model adapters with equivalent role semantics, batched/concurrent execution, and complete provenance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from research.real_v4.model_provenance import (
    ProvenanceEnvelopeError,
    ResponseProvenance,
    host_hardware_fingerprint,
    load_mistral_inventory,
    load_qwen_inventory,
    verify_local_mistral_files_on_disk,
    verify_local_qwen_files_on_disk,
)


@dataclass(frozen=True)
class CanonicalAdapterRequest:
    task_id: str
    sample_id: str
    env_id: str
    system_text: str
    user_text: str
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42
    enable_thinking: bool = False


@dataclass(frozen=True)
class CanonicalAdapterResponse:
    raw_text: str
    provenance: ResponseProvenance
    error: Optional[str] = None


class BaseRealV4Adapter(ABC):
    adapter_id: str
    model_id: str
    pinned_model_version: str

    @abstractmethod
    def serialize_request_payload(self, request: CanonicalAdapterRequest) -> Dict[str, Any]:
        """Return canonical serialized payload demonstrating role, text, limit, and decoding equivalence."""

    @abstractmethod
    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Project provider-specific payload back to canonical role/text/limit/decoding fields for audit."""

    @abstractmethod
    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        """Execute inference and return text plus complete fail-closed provenance."""

    def execute_batch(self, requests: Sequence[CanonicalAdapterRequest]) -> List[CanonicalAdapterResponse]:
        """Execute a sequence of requests preserving input order."""
        return [self.execute(req) for req in requests]

    def release_memory(self) -> None:
        """Optional hook to release cached GPU/model resources between models."""
        return None


class RealV4GeminiAdapter(BaseRealV4Adapter):
    adapter_id = "gemini_v1beta"

    def __init__(
        self,
        *,
        model_id: str = "gemini-2.5-flash",
        pinned_model_version: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        max_retries: int = 4,
        max_workers: int = 12,
        pricing_in_per_m: float = 0.30,
        pricing_out_per_m: float = 2.50,
    ) -> None:
        self.model_id = model_id
        self.pinned_model_version = pinned_model_version
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._injected_client = http_client is not None
        self._client = http_client
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.pricing_in = pricing_in_per_m
        self.pricing_out = pricing_out_per_m

    def serialize_request_payload(self, request: CanonicalAdapterRequest) -> Dict[str, Any]:
        return {
            "system_instruction": {"parts": [{"text": request.system_text}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_text}]}],
            "generationConfig": {
                "temperature": float(request.temperature),
                "topP": 1.0,
                "maxOutputTokens": int(request.max_tokens),
                "seed": int(request.seed),
                "thinkingConfig": {"thinkingBudget": 0 if not request.enable_thinking else 1024},
            },
        }

    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        gen = payload["generationConfig"]
        return {
            "system_role_text": payload["system_instruction"]["parts"][0]["text"],
            "user_role_text": payload["contents"][0]["parts"][0]["text"],
            "temperature": float(gen["temperature"]),
            "max_tokens": int(gen["maxOutputTokens"]),
            "thinking_enabled": bool(gen["thinkingConfig"]["thinkingBudget"] > 0),
        }

    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        if not self.api_key:
            raise ProvenanceEnvelopeError("GEMINI_API_KEY required for RealV4GeminiAdapter")
        payload = self.serialize_request_payload(request)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"

        is_live_net = not self._injected_client
        if self._client is None:
            self._client = httpx.Client(
                timeout=60.0,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=20),
            )
        client = self._client
        start = time.perf_counter()
        retries = 0
        resp: Optional[httpx.Response] = None
        transport_err: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                resp = client.post(url, json=payload)
            except httpx.HTTPError as exc:
                transport_err = str(exc)
                if attempt < self.max_retries - 1:
                    retries += 1
                    if is_live_net:
                        time.sleep(min(8.0, 1.0 * (2 ** attempt)))
                    continue
                resp = None
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries - 1:
                    retries += 1
                    if is_live_net:
                        time.sleep(min(8.0, 1.0 * (2 ** attempt)))
                    continue
            break

        latency_ms = (time.perf_counter() - start) * 1000.0
        if resp is None:
            prov = ResponseProvenance(
                adapter_id=self.adapter_id,
                requested_model_id=self.model_id,
                pinned_model_version=self.pinned_model_version,
                returned_model_version="HTTP_ERROR",
                response_id="http_transport_error",
                finish_reason="HTTP_ERROR",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                retry_count=retries,
                within_pinned_envelope=False,
                envelope_breach_reason="HTTP_TRANSPORT_ERROR",
            )
            return CanonicalAdapterResponse(raw_text="", provenance=prov, error=transport_err or "Transport error")
        data = resp.json() if resp.content else {}

        if resp.status_code != 200:
            prov = ResponseProvenance(
                adapter_id=self.adapter_id,
                requested_model_id=self.model_id,
                pinned_model_version=self.pinned_model_version,
                returned_model_version="HTTP_ERROR",
                response_id=f"http_{resp.status_code}",
                finish_reason="HTTP_ERROR",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                retry_count=retries,
                within_pinned_envelope=False,
                envelope_breach_reason=f"HTTP_{resp.status_code}",
            )
            return CanonicalAdapterResponse(
                raw_text="",
                provenance=prov,
                error=data.get("error", {}).get("message", f"HTTP {resp.status_code}"),
            )

        resp_id = str(data.get("responseId") or "").strip()
        ret_ver = str(data.get("modelVersion") or "").strip()
        if not resp_id or not ret_ver:
            raise ProvenanceEnvelopeError(
                f"Gemini response missing mandatory responseId={resp_id!r} or modelVersion={ret_ver!r}"
            )

        cands = data.get("candidates", [])
        raw_text = ""
        finish_reason = "STOP"
        if cands:
            finish_reason = str(cands[0].get("finishReason", "STOP"))
            parts = cands[0].get("content", {}).get("parts", [])
            raw_text = "".join(str(p.get("text", "")) for p in parts)

        usage = data.get("usageMetadata", {})
        tok_in = int(usage.get("promptTokenCount", 0))
        tok_out = int(usage.get("candidatesTokenCount", 0))
        tok_th = int(usage.get("thoughtsTokenCount", 0))
        cost = (tok_in * self.pricing_in + (tok_out + tok_th) * self.pricing_out) / 1_000_000.0

        ver_match = ret_ver == self.pinned_model_version
        prov = ResponseProvenance(
            adapter_id=self.adapter_id,
            requested_model_id=self.model_id,
            pinned_model_version=self.pinned_model_version,
            returned_model_version=ret_ver,
            response_id=resp_id,
            finish_reason=finish_reason,
            tokens_input=tok_in,
            tokens_output=tok_out,
            tokens_thoughts=tok_th,
            latency_ms=latency_ms,
            retry_count=retries,
            cost_usd=cost,
            within_pinned_envelope=ver_match,
            envelope_breach_reason=None if ver_match else "PROVIDER_VERSION_DRIFT",
        )
        return CanonicalAdapterResponse(raw_text=raw_text, provenance=prov)

    def execute_batch(self, requests: Sequence[CanonicalAdapterRequest]) -> List[CanonicalAdapterResponse]:
        if len(requests) <= 1 or self._injected_client:
            return [self.execute(r) for r in requests]
        if self._client is None:
            self._client = httpx.Client(
                timeout=60.0,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=20),
            )
        workers = min(self.max_workers, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.execute, requests))


class RealV4AnthropicAdapter(BaseRealV4Adapter):
    adapter_id = "anthropic_messages_v1"

    def __init__(
        self,
        *,
        model_id: str = "claude-3-5-haiku-20241022",
        pinned_model_version: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        max_retries: int = 3,
        pricing_in_per_m: float = 0.80,
        pricing_out_per_m: float = 4.00,
    ) -> None:
        self.model_id = model_id
        self.pinned_model_version = pinned_model_version
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._injected_client = http_client is not None
        self._client = http_client
        self.max_retries = max_retries
        self.pricing_in = pricing_in_per_m
        self.pricing_out = pricing_out_per_m

    def serialize_request_payload(self, request: CanonicalAdapterRequest) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "system": request.system_text,
            "messages": [{"role": "user", "content": request.user_text}],
            "temperature": float(request.temperature),
            "max_tokens": int(request.max_tokens),
        }

    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "system_role_text": payload["system"],
            "user_role_text": payload["messages"][0]["content"],
            "temperature": float(payload["temperature"]),
            "max_tokens": int(payload["max_tokens"]),
            "thinking_enabled": bool("thinking" in payload),
        }

    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        if not self.api_key:
            raise ProvenanceEnvelopeError("ANTHROPIC_API_KEY required for RealV4AnthropicAdapter")
        payload = self.serialize_request_payload(request)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        is_live_net = not self._injected_client
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        client = self._client
        start = time.perf_counter()
        retries = 0
        resp: Optional[httpx.Response] = None
        transport_err: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            except httpx.HTTPError as exc:
                transport_err = str(exc)
                if attempt < self.max_retries - 1:
                    retries += 1
                    if is_live_net:
                        time.sleep(min(8.0, 1.0 * (2 ** attempt)))
                    continue
                resp = None
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries - 1:
                    retries += 1
                    if is_live_net:
                        time.sleep(min(8.0, 1.0 * (2 ** attempt)))
                    continue
            break

        latency_ms = (time.perf_counter() - start) * 1000.0
        if resp is None:
            prov = ResponseProvenance(
                adapter_id=self.adapter_id,
                requested_model_id=self.model_id,
                pinned_model_version=self.pinned_model_version,
                returned_model_version="HTTP_ERROR",
                response_id="http_transport_error",
                finish_reason="HTTP_ERROR",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                retry_count=retries,
                within_pinned_envelope=False,
                envelope_breach_reason="HTTP_TRANSPORT_ERROR",
            )
            return CanonicalAdapterResponse(raw_text="", provenance=prov, error=transport_err or "Transport error")
        data = resp.json() if resp.content else {}

        if resp.status_code != 200:
            prov = ResponseProvenance(
                adapter_id=self.adapter_id,
                requested_model_id=self.model_id,
                pinned_model_version=self.pinned_model_version,
                returned_model_version="HTTP_ERROR",
                response_id=f"http_{resp.status_code}",
                finish_reason="HTTP_ERROR",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                retry_count=retries,
                within_pinned_envelope=False,
                envelope_breach_reason=f"HTTP_{resp.status_code}",
            )
            return CanonicalAdapterResponse(
                raw_text="",
                provenance=prov,
                error=data.get("error", {}).get("message", f"HTTP {resp.status_code}"),
            )

        resp_id = str(data.get("id") or "").strip()
        ret_ver = str(data.get("model") or "").strip()
        if not resp_id or not ret_ver:
            raise ProvenanceEnvelopeError(
                f"Anthropic response missing mandatory id={resp_id!r} or model={ret_ver!r}"
            )

        raw_text = "".join(
            str(blk.get("text", ""))
            for blk in data.get("content", [])
            if blk.get("type") == "text"
        )
        finish_reason = str(data.get("stop_reason", "end_turn"))
        usage = data.get("usage", {})
        tok_in = int(usage.get("input_tokens", 0))
        tok_out = int(usage.get("output_tokens", 0))
        cost = (tok_in * self.pricing_in + tok_out * self.pricing_out) / 1_000_000.0

        ver_match = ret_ver == self.pinned_model_version
        prov = ResponseProvenance(
            adapter_id=self.adapter_id,
            requested_model_id=self.model_id,
            pinned_model_version=self.pinned_model_version,
            returned_model_version=ret_ver,
            response_id=resp_id,
            finish_reason=finish_reason,
            tokens_input=tok_in,
            tokens_output=tok_out,
            latency_ms=latency_ms,
            retry_count=retries,
            cost_usd=cost,
            within_pinned_envelope=ver_match,
            envelope_breach_reason=None if ver_match else "PROVIDER_VERSION_DRIFT",
        )
        return CanonicalAdapterResponse(raw_text=raw_text, provenance=prov)


class _BaseLocalMLXAdapter(BaseRealV4Adapter):
    """Shared local Apple Silicon MLX adapter supporting verified snapshots and deterministic `batch_generate`."""

    _gpu_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_id: str,
        pinned_model_version: str,
        inventory: Dict[str, Any],
        verify_fn: Callable[..., Dict[str, Any]],
        chat_template_tag: str,
        merge_system_into_user_for_template: bool = False,
        batch_size: int = 32,
        mlx_runner_fn: Optional[Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        self.model_id = model_id
        self.pinned_model_version = pinned_model_version
        self._inventory = inventory
        self._verify_fn = verify_fn
        self._merge_system = merge_system_into_user_for_template
        self._batch_size = max(1, int(batch_size))
        self._mlx_runner_fn = mlx_runner_fn
        self._loaded_mlx_bundle: Optional[Tuple[Any, Any]] = None
        self._call_seq = 0
        self._chat_template_tag = chat_template_tag
        self._chat_template_sha = hashlib.sha256(chat_template_tag.encode("utf-8")).hexdigest()

    def serialize_request_payload(self, request: CanonicalAdapterRequest) -> Dict[str, Any]:
        return {
            "repo_id": self.model_id,
            "snapshot_revision": self._inventory["revision"],
            "messages": [
                {"role": "system", "content": request.system_text},
                {"role": "user", "content": request.user_text},
            ],
            "chat_template_kwargs": {"enable_thinking": bool(request.enable_thinking)},
            "sampling_params": {
                "temp": float(request.temperature),
                "max_tokens": int(request.max_tokens),
                "seed": int(request.seed),
            },
        }

    def extract_canonical_semantics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "system_role_text": payload["messages"][0]["content"],
            "user_role_text": payload["messages"][1]["content"],
            "temperature": float(payload["sampling_params"]["temp"]),
            "max_tokens": int(payload["sampling_params"]["max_tokens"]),
            "thinking_enabled": bool(payload["chat_template_kwargs"]["enable_thinking"]),
        }

    def _ensure_live_mlx_model(self) -> Tuple[Any, Any]:
        if self._loaded_mlx_bundle is None:
            import mlx_lm  # type: ignore

            snap_path = Path(self._inventory["cache_snapshot"]).expanduser()
            self._verify_fn(snap_path, verify_full_safetensors_bytes=False)
            model, tokenizer = mlx_lm.load(str(snap_path))
            self._loaded_mlx_bundle = (model, tokenizer)
        return self._loaded_mlx_bundle

    def release_memory(self) -> None:
        if self._loaded_mlx_bundle is not None:
            self._loaded_mlx_bundle = None
            gc.collect()
            try:
                import mlx.core as mx  # type: ignore

                mx.clear_cache()
            except Exception:
                pass

    def _render_prompt_string(self, tokenizer: Any, payload: Dict[str, Any]) -> str:
        msgs = payload["messages"]
        if self._merge_system:
            merged = [{"role": "user", "content": f"{msgs[0]['content']}\n\n{msgs[1]['content']}"}]
            return str(
                tokenizer.apply_chat_template(
                    merged,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return str(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                **payload["chat_template_kwargs"],
            )
        )

    def _build_response(
        self,
        request: CanonicalAdapterRequest,
        *,
        raw_text: str,
        ret_rev: str,
        finish_reason: str,
        tok_in: int,
        tok_out: int,
        latency_ms: float,
        peak_mb: float,
        runtime_versions: Dict[str, str],
    ) -> CanonicalAdapterResponse:
        self._call_seq += 1
        tps = float(tok_out / max(1e-6, latency_ms / 1000.0))
        req_digest = hashlib.sha256(
            f"{self.adapter_id}:{request.task_id}:{request.sample_id}:{request.env_id}:{ret_rev}:{self._call_seq}".encode("utf-8")
        ).hexdigest()[:16]
        ver_match = ret_rev == self.pinned_model_version
        prov = ResponseProvenance(
            adapter_id=self.adapter_id,
            requested_model_id=self.model_id,
            pinned_model_version=self.pinned_model_version,
            returned_model_version=ret_rev,
            response_id=f"mlx_{req_digest}",
            finish_reason=finish_reason,
            tokens_input=tok_in,
            tokens_output=tok_out,
            latency_ms=max(0.01, latency_ms),
            retry_count=0,
            cost_usd=0.0,
            peak_memory_mb=peak_mb,
            throughput_tok_per_s=tps,
            local_weights_inventory_sha256=self._inventory["weights_inventory_sha256"],
            tokenizer_sha256=self._inventory["tokenizer_sha256"],
            chat_template_sha256=self._chat_template_sha,
            runtime_versions=runtime_versions,
            hardware_platform=host_hardware_fingerprint(),
            within_pinned_envelope=ver_match,
            envelope_breach_reason=None if ver_match else "LOCAL_SNAPSHOT_DRIFT",
        )
        return CanonicalAdapterResponse(raw_text=raw_text, provenance=prov)

    def execute(self, request: CanonicalAdapterRequest) -> CanonicalAdapterResponse:
        payload = self.serialize_request_payload(request)
        start = time.perf_counter()

        if self._mlx_runner_fn is not None:
            raw_text, meta = self._mlx_runner_fn(payload)
            runtime_versions = dict(meta.get("runtime_versions", {"mlx": "injected_stub", "mlx_lm": "injected_stub"}))
            ret_rev = str(meta.get("snapshot_revision", self._inventory["revision"]))
            tok_in = int(meta.get("tokens_input", len(request.user_text.split())))
            tok_out = int(meta.get("tokens_output", max(1, len(raw_text.split()))))
            peak_mb = float(meta.get("peak_memory_mb", 5420.0))
            finish_reason = str(meta.get("finish_reason", "stop"))
            latency_ms = max(0.01, (time.perf_counter() - start) * 1000.0)
            return self._build_response(
                request,
                raw_text=raw_text,
                ret_rev=ret_rev,
                finish_reason=finish_reason,
                tok_in=tok_in,
                tok_out=tok_out,
                latency_ms=latency_ms,
                peak_mb=peak_mb,
                runtime_versions=runtime_versions,
            )

        return self.execute_batch([request])[0]

    def execute_batch(self, requests: Sequence[CanonicalAdapterRequest]) -> List[CanonicalAdapterResponse]:
        if not requests:
            return []
        if self._mlx_runner_fn is not None:
            return [self.execute(r) for r in requests]

        try:
            import mlx.core as mx  # type: ignore
            import mlx_lm  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except ImportError as exc:
            raise ProvenanceEnvelopeError(
                "Local MLX runtime (`mlx` and `mlx_lm`) is not installed; install and lock MLX before Phase 4 execution"
            ) from exc

        runtime_versions = {
            "mlx": str(getattr(mx, "__version__", "unknown")),
            "mlx_lm": str(getattr(mlx_lm, "__version__", "unknown")),
        }
        ret_rev = str(self._inventory["revision"])
        out_responses: List[CanonicalAdapterResponse] = []

        with self._gpu_lock:
            model, tokenizer = self._ensure_live_mlx_model()
            for offset in range(0, len(requests), self._batch_size):
                chunk_reqs = requests[offset : offset + self._batch_size]
                payloads = [self.serialize_request_payload(r) for r in chunk_reqs]
                tok_prompts = [
                    tokenizer.encode(self._render_prompt_string(tokenizer, p), add_special_tokens=False)
                    for p in payloads
                ]
                max_toks = int(payloads[0]["sampling_params"]["max_tokens"])
                temp = float(payloads[0]["sampling_params"]["temp"])
                seed = int(payloads[0]["sampling_params"]["seed"])

                mx.random.seed(seed)
                mx.reset_peak_memory()
                t0 = time.perf_counter()

                if len(chunk_reqs) == 1:
                    chunks = list(
                        mlx_lm.stream_generate(
                            model,
                            tokenizer,
                            prompt=tok_prompts[0],
                            max_tokens=max_toks,
                            sampler=make_sampler(temp=temp),
                        )
                    )
                    dt_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)
                    txt = "".join(c.text for c in chunks)
                    if chunks:
                        last = chunks[-1]
                        f_reason = str(getattr(last, "finish_reason", None) or "stop")
                        t_in = int(getattr(last, "prompt_tokens", len(tok_prompts[0])))
                        t_out = int(getattr(last, "generation_tokens", len(chunks)))
                        p_mb = float(getattr(last, "peak_memory", 0.0)) * 1024.0
                    else:
                        f_reason, t_in, t_out, p_mb = "stop", len(tok_prompts[0]), 0, 0.0
                    out_responses.append(
                        self._build_response(
                            chunk_reqs[0],
                            raw_text=txt,
                            ret_rev=ret_rev,
                            finish_reason=f_reason,
                            tok_in=t_in,
                            tok_out=t_out,
                            latency_ms=dt_ms,
                            peak_mb=p_mb,
                            runtime_versions=runtime_versions,
                        )
                    )
                else:
                    batch_res = mlx_lm.batch_generate(
                        model,
                        tokenizer,
                        prompts=tok_prompts,
                        max_tokens=max_toks,
                        sampler=make_sampler(temp=temp),
                        return_token_ids=True,
                    )
                    dt_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)
                    per_item_ms = dt_ms / len(chunk_reqs)
                    p_mb = float(getattr(batch_res.stats, "peak_memory", 0.0)) * 1024.0

                    for idx_r, req_item in enumerate(chunk_reqs):
                        txt = str(batch_res.texts[idx_r])
                        gen_ids = batch_res.token_ids[idx_r] if batch_res.token_ids is not None else []
                        t_out = int(len(gen_ids))
                        t_in = int(len(tok_prompts[idx_r]))
                        f_reason = "length" if t_out >= max_toks else "stop"
                        out_responses.append(
                            self._build_response(
                                req_item,
                                raw_text=txt,
                                ret_rev=ret_rev,
                                finish_reason=f_reason,
                                tok_in=t_in,
                                tok_out=t_out,
                                latency_ms=per_item_ms,
                                peak_mb=p_mb,
                                runtime_versions=runtime_versions,
                            )
                        )
        return out_responses


class RealV4LocalQwenMLXAdapter(_BaseLocalMLXAdapter):
    adapter_id = "local_mlx_qwen3_4bit"

    def __init__(
        self,
        *,
        model_id: str = "mlx-community/Qwen3-8B-4bit",
        pinned_model_version: str = "545dc4251c05440727734bcd94334791f6ab0192",
        batch_size: int = 32,
        mlx_runner_fn: Optional[Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            pinned_model_version=pinned_model_version,
            inventory=load_qwen_inventory(),
            verify_fn=verify_local_qwen_files_on_disk,
            chat_template_tag="qwen3_chatml_enable_thinking_false_v1",
            merge_system_into_user_for_template=False,
            batch_size=batch_size,
            mlx_runner_fn=mlx_runner_fn,
        )


class RealV4LocalMistralMLXAdapter(_BaseLocalMLXAdapter):
    adapter_id = "local_mlx_mistral7b_v03_4bit"

    def __init__(
        self,
        *,
        model_id: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        pinned_model_version: str = "a4b8f870474b0eb527f466a03fbc187830d271f5",
        batch_size: int = 32,
        mlx_runner_fn: Optional[Callable[[Dict[str, Any]], Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            pinned_model_version=pinned_model_version,
            inventory=load_mistral_inventory(),
            verify_fn=verify_local_mistral_files_on_disk,
            chat_template_tag="mistral_v03_inst_system_prefixed_v1",
            merge_system_into_user_for_template=True,
            batch_size=batch_size,
            mlx_runner_fn=mlx_runner_fn,
        )
