"""
Section 8.4: Real Pilot Harness (Two Small Tasks, Bounded Budget).

Validates the PromptPrism 2.1 test set generator + coverage-aware pruning gate on:
1. `banking77_intent` (6 Real-v4 factors on `gemini-2.5-flash`):
   Validates clause-role separation (`global_scaffold` vs. `conditional_rule`),
   runs the 16-run 2^(6-2)IV discovery screen on the synthetic discovery split
   plus 2-run confirmation, and compares against the full 64-run Real-v4 labelled
   ground truth (`research/results/real_v4/screen_manifest.jsonl`).
2. `squad2_grounded_qa` (6 Real-v4 factors with conditional abstention rule):
   Validates the rare-rule coverage gate (`KEEP_TARGETED` / `CONFLICT_REVIEW`)
   when unanswerable edge cases are generated and coverage-tagged.

By default runs in `--mode dry-run` (offline MockLLM, zero API cost).
Pass `--mode live` to execute against real APIs (supports built-in `httpx`
Gemini / Anthropic adapters using `.env` keys as well as `--client` / `--model`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import pandas as pd

from prompt_prism import (
    CheckSuiteMetric,
    DesignMatrix,
    Evaluator,
    Experiment,
    Factor,
    FactorSet,
    MockLLM,
    PromptTemplate,
    TestSet,
    TestSetGenerator,
)
from prompt_prism.analysis.diagnostics import (
    CalibratedAttributionAudit,
    run_calibrated_attribution_audit,
)
from prompt_prism.runner.client import LLMClient, LLMResponse
from prompt_prism.testgen.loaders import resolve_cli_client

ROOT = Path(__file__).resolve().parents[2]
REAL_V4_SPLITS_DIR = ROOT / "research" / "data" / "real_v4" / "splits"
REAL_V4_SCREEN_MANIFEST = (
    ROOT / "research" / "results" / "real_v4" / "screen_manifest.jsonl"
)

FACTOR_ID_TO_NAME: Dict[str, str] = {
    "A": "specialist_persona",
    "B": "brief_reasoning",
    "C": "few_shot_demos",
    "D": "strict_output_contract",
    "E": "grounding_guardrail",
    "F": "self_verification",
}

DEFAULT_BANKING77_OPTIONS: List[str] = [
    "Refund_not_showing_up",
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_linking",
    "card_payment_fee_charged",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
]


class GeminiHTTPClient(LLMClient):
    """Lightweight httpx client for Gemini v1beta (`GEMINI_API_KEY`) matching Real-v4 settings."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        json_mode: bool = False,
        thinking_budget: Optional[int] = None,
        max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.json_mode = bool(json_mode)
        if thinking_budget is None:
            self.thinking_budget = 128 if "pro" in model_id.lower() else 0
        else:
            self.thinking_budget = int(thinking_budget)
        self.max_retries = int(max_retries)
        self._client = httpx.Client(
            timeout=120.0,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=20),
        )
        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        system_parts: List[str] = []
        user_parts: List[str] = []
        if isinstance(prompt, list):
            for msg in prompt:
                role = str(msg.get("role", "user")).lower()
                text = str(msg.get("content", ""))
                if not text:
                    continue
                if role == "system":
                    system_parts.append(text)
                else:
                    user_parts.append(text)
        else:
            user_parts.append(str(prompt))

        gen_cfg: Dict[str, Any] = {
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "topP": 1.0,
            "maxOutputTokens": int(kwargs.get("max_tokens", self.max_output_tokens)),
            "thinkingConfig": {"thinkingBudget": self.thinking_budget},
        }
        if self.json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}
            ],
            "generationConfig": gen_cfg,
        }
        if system_parts:
            payload["system_instruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent?key={self.api_key}"
        )
        start = time.perf_counter()
        resp: Optional[httpx.Response] = None
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                last_err = str(exc)
                if attempt < self.max_retries - 1:
                    time.sleep(min(10.0, 1.0 * (2**attempt)))
                    continue
                resp = None
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                if attempt < self.max_retries - 1:
                    time.sleep(min(12.0, 1.5 * (2**attempt)))
                    continue
            break

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if resp is None:
            return LLMResponse(
                content="", latency_ms=elapsed_ms, error=last_err or "HTTP error"
            )
        data = resp.json() if resp.content else {}
        if resp.status_code != 200:
            msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            return LLMResponse(content="", latency_ms=elapsed_ms, error=str(msg))

        cands = data.get("candidates") or []
        parts = (cands[0].get("content") or {}).get("parts") or [] if cands else []
        text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))
        usage_meta = data.get("usageMetadata") or {}
        tok_in = int(usage_meta.get("promptTokenCount") or 0)
        tok_out = int(usage_meta.get("candidatesTokenCount") or 0)
        with self._lock:
            self.total_input_tokens += tok_in
            self.total_output_tokens += tok_out
            self.total_calls += 1
        return LLMResponse(
            content=text,
            raw=data,
            latency_ms=elapsed_ms,
            token_usage={
                "prompt_tokens": tok_in,
                "completion_tokens": tok_out,
                "total_tokens": tok_in + tok_out,
            },
        )


class AnthropicHTTPClient(LLMClient):
    """Lightweight httpx client for Anthropic Messages API (`ANTHROPIC_API_KEY`)."""

    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required.")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self._client = httpx.Client(
            timeout=120.0,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=10),
        )
        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def generate(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any
    ) -> LLMResponse:
        system_parts: List[str] = []
        messages: List[Dict[str, str]] = []
        if isinstance(prompt, list):
            for msg in prompt:
                role = str(msg.get("role", "user")).lower()
                text = str(msg.get("content", ""))
                if not text:
                    continue
                if role == "system":
                    system_parts.append(text)
                else:
                    messages.append({"role": role, "content": text})
        else:
            messages.append({"role": "user", "content": str(prompt)})

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        start = time.perf_counter()
        resp: Optional[httpx.Response] = None
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
            except httpx.HTTPError as exc:
                last_err = str(exc)
                if attempt < self.max_retries - 1:
                    time.sleep(min(10.0, 1.0 * (2**attempt)))
                    continue
                resp = None
                break
            if resp.status_code in (429, 529) or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
                if attempt < self.max_retries - 1:
                    time.sleep(min(15.0, 2.0 * (2**attempt)))
                    continue
            break

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if resp is None:
            return LLMResponse(
                content="", latency_ms=elapsed_ms, error=last_err or "HTTP error"
            )
        data = resp.json() if resp.content else {}
        if resp.status_code != 200:
            err = data.get("error", {})
            msg = (
                err.get("message", f"HTTP {resp.status_code}")
                if isinstance(err, dict)
                else str(err)
            )
            return LLMResponse(content="", latency_ms=elapsed_ms, error=str(msg))

        blocks = data.get("content") or []
        text = "".join(
            str(b.get("text", ""))
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        usage = data.get("usage") or {}
        tok_in = int(usage.get("input_tokens") or 0)
        tok_out = int(usage.get("output_tokens") or 0)
        with self._lock:
            self.total_input_tokens += tok_in
            self.total_output_tokens += tok_out
            self.total_calls += 1
        return LLMResponse(
            content=text,
            raw=data,
            latency_ms=elapsed_ms,
            token_usage={
                "prompt_tokens": tok_in,
                "completion_tokens": tok_out,
                "total_tokens": tok_in + tok_out,
            },
        )


def _load_banking77_options() -> List[str]:
    split_path = REAL_V4_SPLITS_DIR / "banking77_intent_pilot.jsonl"
    if split_path.exists():
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                opts = row.get("options")
                if isinstance(opts, list) and opts:
                    return [str(o) for o in opts]
    return list(DEFAULT_BANKING77_OPTIONS)


def build_v4_task_spec(task_id: str) -> Dict[str, Any]:
    """Return canonical 6-factor FactorSet, Real-v4-aligned PromptTemplate, and task description."""
    if task_id == "banking77_intent":
        options = _load_banking77_options()
        options_str = ", ".join(options)
        factors = FactorSet(
            [
                Factor.binary(
                    id="A",
                    name="specialist_persona",
                    level_0_content="You are a helpful and accurate assistant.",
                    level_1_content="You are a precision banking intent classification specialist. Follow all task rules accurately.",
                    description="Expert domain persona framing (system role)",
                    metadata={"clause_role": "global_scaffold"},
                ),
                Factor.binary(
                    id="B",
                    name="brief_reasoning",
                    level_0_content="Respond directly without unnecessary commentary.",
                    level_1_content="Decompose the problem concisely in 1-2 brief lines before emitting the final answer.",
                    description="Short step-by-step reasoning before answering",
                    metadata={"clause_role": "global_scaffold"},
                ),
                Factor.binary(
                    id="C",
                    name="few_shot_demos",
                    level_0_content="",
                    level_1_content=(
                        "[EXAMPLES]\n"
                        "Example 1:\nQuery: Why won't my card show up on the app?\n"
                        "Output: FINAL_ANSWER: card_linking\n"
                        "Example 2:\nQuery: The ATM kept my card and won't give it back.\n"
                        "Output: FINAL_ANSWER: card_swallowed"
                    ),
                    description="2-shot banking intent demonstrations",
                    metadata={"clause_role": "global_scaffold"},
                ),
                Factor.binary(
                    id="D",
                    name="strict_output_contract",
                    level_0_content="Return the matching intent label from the allowed list.",
                    level_1_content="STRICT CONTRACT: End your response with `FINAL_ANSWER: <exact_intent_label>` using only one label from the allowed inventory.",
                    description="Strict FINAL_ANSWER formatting contract",
                    metadata={"clause_role": "conditional_rule"},
                ),
                Factor.binary(
                    id="E",
                    name="grounding_guardrail",
                    level_0_content="",
                    level_1_content=(
                        "Rely strictly on the provided input and never invent unsupported facts, "
                        "labels, or assumptions. If the inquiry is completely outside banking or "
                        "does not match any allowed intent, reply FINAL_ANSWER: out_of_scope."
                    ),
                    description="Grounding and out-of-scope abstention guardrail",
                    metadata={"clause_role": "conditional_rule"},
                ),
                Factor.binary(
                    id="F",
                    name="self_verification",
                    level_0_content="",
                    level_1_content="Before finishing, silently verify that every constraint and formatting requirement is satisfied.",
                    description="Silent self-verification step",
                    metadata={"clause_role": "global_scaffold"},
                ),
            ]
        )
        template = PromptTemplate(default_role="user", delimiter="\n\n")
        template.add_section(
            id="system_persona", factor_id="A", role="system", position=0
        )
        template.add_section(
            id="preamble",
            content="[INSTRUCTIONS]\nRead the task input below and produce your response.",
            role="user",
            position=1,
        )
        template.add_section(
            id="brief_reasoning", factor_id="B", role="user", position=2
        )
        template.add_section(
            id="grounding_guardrail", factor_id="E", role="user", position=3
        )
        template.add_section(
            id="self_verification", factor_id="F", role="user", position=4
        )
        template.add_section(
            id="strict_output_contract", factor_id="D", role="user", position=5
        )
        template.add_section(
            id="few_shot_demos", factor_id="C", role="user", position=6
        )
        template.add_section(
            id="task_input",
            content=f"[TASK INPUT]\nAllowed Intents: {options_str}\nQuery: {{{{ query }}}}",
            role="user",
            position=7,
        )
        task_desc = (
            "Classify customer banking support queries into EXACTLY ONE of these 20 allowed "
            f"snake_case intent labels: {options_str}. "
            "For baseline and global_scaffold items, generate realistic customer queries (do NOT "
            "write the snake_case label string inside the query!) and attach a deterministic "
            "`contains` or `regex` check matching the exact correct snake_case label (so both "
            "bare `<label>` and `FINAL_ANSWER: <label>` pass on baseline). "
            "For `strict_output_contract` (D), check `regex` `FINAL_ANSWER:\\s*<exact_label>`. "
            "For `grounding_guardrail` (E), generate non-banking / out-of-scope requests (without "
            "writing `out_of_scope` inside the query) whose check requires `contains` `out_of_scope`."
        )
        return {"factors": factors, "template": template, "task_description": task_desc}

    if task_id == "squad2_grounded_qa":
        factors = FactorSet(
            [
                Factor.binary(
                    id="A",
                    name="specialist_persona",
                    level_0_content="You are a helpful and accurate assistant.",
                    level_1_content="You are a precision grounded QA specialist. Follow all task rules accurately.",
                    description="Expert grounded QA persona (system role)",
                    metadata={"clause_role": "global_scaffold"},
                ),
                Factor.binary(
                    id="B",
                    name="brief_reasoning",
                    level_0_content="Respond directly without unnecessary commentary.",
                    level_1_content="Decompose the problem concisely in 1-2 brief lines before emitting the final answer.",
                    description="Brief reasoning before answering",
                    metadata={"clause_role": "global_scaffold"},
                ),
                Factor.binary(
                    id="C",
                    name="few_shot_demos",
                    level_0_content="",
                    level_1_content=(
                        "[EXAMPLES]\n"
                        "Example 1:\nContext: Oxygen was isolated by Joseph Priestley in 1774.\n"
                        "Question: In what year did Priestley isolate oxygen?\nOutput: FINAL_ANSWER: 1774\n"
                        "Example 2:\nContext: Oxygen was isolated by Joseph Priestley in 1774.\n"
                        "Question: Where was Priestley born?\nOutput: FINAL_ANSWER: UNANSWERABLE"
                    ),
                    description="2-shot answerable + unanswerable demonstrations",
                    metadata={"clause_role": "global_scaffold"},
                ),
                Factor.binary(
                    id="D",
                    name="strict_output_contract",
                    level_0_content="Answer the question concisely from the context, or reply UNANSWERABLE if unsupported.",
                    level_1_content="STRICT CONTRACT: End your response with `FINAL_ANSWER: <exact span>` or `FINAL_ANSWER: UNANSWERABLE` if the context does not answer the question.",
                    description="Strict FINAL_ANSWER formatting contract",
                    metadata={"clause_role": "conditional_rule"},
                ),
                Factor.binary(
                    id="E",
                    name="grounding_guardrail",
                    level_0_content="",
                    level_1_content=(
                        "Rely strictly on the provided context and never invent unsupported facts, "
                        "labels, or assumptions. If the context does not explicitly answer the "
                        "question, reply UNANSWERABLE."
                    ),
                    description="Conditional abstention guardrail on unanswerable questions",
                    metadata={"clause_role": "conditional_rule"},
                ),
                Factor.binary(
                    id="F",
                    name="self_verification",
                    level_0_content="",
                    level_1_content="Before finishing, silently verify that every constraint and formatting requirement is satisfied.",
                    description="Silent self-verification step",
                    metadata={"clause_role": "global_scaffold"},
                ),
            ]
        )
        template = PromptTemplate(default_role="user", delimiter="\n\n")
        template.add_section(
            id="system_persona", factor_id="A", role="system", position=0
        )
        template.add_section(
            id="preamble",
            content="[INSTRUCTIONS]\nRead the task input below and produce your response.",
            role="user",
            position=1,
        )
        template.add_section(
            id="brief_reasoning", factor_id="B", role="user", position=2
        )
        template.add_section(
            id="grounding_guardrail", factor_id="E", role="user", position=3
        )
        template.add_section(
            id="self_verification", factor_id="F", role="user", position=4
        )
        template.add_section(
            id="strict_output_contract", factor_id="D", role="user", position=5
        )
        template.add_section(
            id="few_shot_demos", factor_id="C", role="user", position=6
        )
        template.add_section(
            id="task_input",
            content="[TASK INPUT]\nContext: {{ context }}\nQuestion: {{ question }}",
            role="user",
            position=7,
        )
        task_desc = (
            "SQuAD 2.0 grounded extractive QA: return the shortest exact answer span from Context "
            "answering Question, or reply UNANSWERABLE when Context does not state the answer. "
            "CRITICAL CHECK DESIGN: Never use a bare `contains` check for a span already inside "
            "Context (raw input echo would trivially pass and be filtered out!). Instead, for "
            "answerable items use an anchored `regex` matching the concise final answer on the "
            "last line, e.g. `(?i)(?:^|\\n|FINAL_ANSWER:\\s*)1774\\.?\\s*$`. "
            "For `strict_output_contract` (D), require `FINAL_ANSWER:` prefix via `regex`. "
            "For `grounding_guardrail` (E), generate tricky UNANSWERABLE questions whose context "
            "mentions related entities/dates (tempting hallucination, without the word "
            "UNANSWERABLE in the context) and check `regex` `(?i)\\bUNANSWERABLE\\b`."
        )
        return {"factors": factors, "template": template, "task_description": task_desc}

    raise ValueError(f"Unsupported pilot task_id={task_id!r}")


def load_real_pilot_seeds(task_id: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Load real seed examples from the Real-v4 pilot split (disjoint from screen/confirmation)."""
    split_path = REAL_V4_SPLITS_DIR / f"{task_id}_pilot.jsonl"
    if not split_path.exists():
        return []
    seeds: List[Dict[str, Any]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if task_id == "banking77_intent":
            query_text = (
                row.get("input_text")
                or row.get("query")
                or row.get("utterance")
                or row.get("text", "")
            )
            seeds.append({"query": query_text, "target": row.get("target")})
        elif task_id == "squad2_grounded_qa":
            seeds.append(
                {
                    "context": row.get("context", ""),
                    "question": row.get("question", ""),
                    "target": row.get("target"),
                }
            )
        if len(seeds) >= limit:
            break
    return seeds


def load_real_v4_reference_audit(
    task_id: str,
    model_id: str = "gemini-2.5-flash",
    equivalence_margin: float = 0.02,
) -> Optional[CalibratedAttributionAudit]:
    """Compute the unaliased full 64-run Real-v4 reference audit from screen_manifest.jsonl."""
    if not REAL_V4_SCREEN_MANIFEST.exists():
        return None
    from research.real_v4.selection import run_id_to_levels

    fids = ["A", "B", "C", "D", "E", "F"]
    rows: List[Dict[str, Any]] = []
    for line in REAL_V4_SCREEN_MANIFEST.read_text(encoding="utf-8").splitlines():
        if task_id not in line or model_id not in line:
            continue
        rec = json.loads(line)
        if rec.get("task_id") != task_id or rec.get("model_id") != model_id:
            continue
        lvls = run_id_to_levels(int(rec["run_id"]))
        rows.append(
            {
                "run_id": int(rec["run_id"]),
                "sample_id": str(rec["sample_id"]),
                **{FACTOR_ID_TO_NAME[f]: int(lvls[f]) for f in fids},
                "score": float(rec["score"]),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return run_calibrated_attribution_audit(
        data=df,
        factor_cols=[FACTOR_ID_TO_NAME[f] for f in fids],
        target_col="score",
        equivalence_margin=equivalence_margin,
        n_sign_flip=500,
        seed=42,
    )


def compare_synthetic_vs_real_audit(
    real_audit: CalibratedAttributionAudit,
    synth_audit: CalibratedAttributionAudit,
) -> Dict[str, Any]:
    """Check Section 8.4 success criteria comparing synthetic vs. real-data verdicts."""
    real_by_f = {d.factor_id: d for d in real_audit.factors}
    synth_by_f = {d.factor_id: d for d in synth_audit.factors}

    dangerous_flips: List[str] = []
    sign_agreements: Dict[str, bool] = {}
    rows: List[Dict[str, Any]] = []

    for fid, r_diag in real_by_f.items():
        s_diag = synth_by_f.get(fid)
        if s_diag is None:
            continue
        if (
            r_diag.pruning_verdict in {"KEEP", "REMOVE_HARMFUL"}
            and s_diag.pruning_verdict == "SAFE_TO_PRUNE"
        ):
            dangerous_flips.append(fid)
        if r_diag.pruning_verdict in {"KEEP", "REMOVE_HARMFUL"}:
            sign_agreements[fid] = (r_diag.effect_delta > 0) == (
                s_diag.effect_delta > 0
            )
        rows.append(
            {
                "factor": fid,
                "real_delta": r_diag.effect_delta,
                "real_verdict": r_diag.pruning_verdict,
                "synth_delta_F": s_diag.effect_delta,
                "synth_delta_T": s_diag.delta_targeted,
                "synth_n_T": s_diag.n_targeted,
                "synth_verdict": s_diag.pruning_verdict,
            }
        )

    brief_flipped_to_keep = (
        real_by_f.get("brief_reasoning") is not None
        and real_by_f["brief_reasoning"].pruning_verdict == "REMOVE_HARMFUL"
        and synth_by_f.get("brief_reasoning") is not None
        and synth_by_f["brief_reasoning"].pruning_verdict == "KEEP"
    )
    return {
        "zero_dangerous_prune_flips": len(dangerous_flips) == 0,
        "dangerous_flips": dangerous_flips,
        "brief_reasoning_not_tautological_keep": not brief_flipped_to_keep,
        "real_significant_sign_agreement": sign_agreements,
        "comparison": rows,
    }


def _make_dry_run_clients(task_id: str) -> Dict[str, LLMClient]:
    """Deterministic offline generator, tagger, and SUT clients for --mode dry-run."""
    words = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
        "amber",
        "bronze",
        "cobalt",
        "emerald",
        "garnet",
        "ivory",
    ]
    counters: Dict[str, int] = {
        "base": 0,
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0,
        "E": 0,
        "F": 0,
    }

    def _tag_str(prefix: str, i: int) -> str:
        return f"{prefix}_{words[i % len(words)]}_{words[(i * 7) % len(words)]}_{i}"

    def _gen(prompt: Any, _ctx: Dict[str, Any]) -> str:
        text = str(prompt)
        m_n = re.search(r"Generate (\d+)", text)
        n = int(m_n.group(1)) if m_n else 10
        m_f = re.search(r"(?:clause|scaffold) `([A-F])`", text)
        bucket = m_f.group(1) if m_f else "base"

        cases: List[Dict[str, Any]] = []
        for _ in range(n):
            counters[bucket] += 1
            idx = counters[bucket]
            tok = _tag_str(bucket, idx)
            if task_id == "banking77_intent":
                if bucket == "E":
                    cases.append(
                        {
                            "inputs": {
                                "query": f"UNRELATED_RECIPE_E {tok} chocolate cake baking steps"
                            },
                            "target": "FINAL_ANSWER: out_of_scope",
                            "checks": [{"kind": "contains", "value": "out_of_scope"}],
                            "reference_pass": "FINAL_ANSWER: out_of_scope",
                            "reference_fail": "FINAL_ANSWER: card_linking",
                        }
                    )
                elif bucket == "D":
                    cases.append(
                        {
                            "inputs": {
                                "query": f"STRICT_CONTRACT_D {tok} plastic never shows in app"
                            },
                            "target": "FINAL_ANSWER: card_linking",
                            "checks": [
                                {
                                    "kind": "contains",
                                    "value": "FINAL_ANSWER: card_linking",
                                }
                            ],
                            "reference_pass": "FINAL_ANSWER: card_linking",
                            "reference_fail": "card_linking",
                        }
                    )
                else:
                    cases.append(
                        {
                            "inputs": {
                                "query": f"BANKING_{bucket} {tok} plastic not showing in app"
                            },
                            "target": "card_linking",
                            "checks": [{"kind": "contains", "value": "card_linking"}],
                            "reference_pass": "FINAL_ANSWER: card_linking",
                            "reference_fail": "FINAL_ANSWER: age_limit",
                        }
                    )
            else:
                if bucket == "E":
                    cases.append(
                        {
                            "inputs": {
                                "context": f"NO_SPAN_E {tok} Mercury is the innermost planet.",
                                "question": f"Who discovered Neptune in {tok}?",
                            },
                            "target": "FINAL_ANSWER: UNANSWERABLE",
                            "checks": [
                                {
                                    "kind": "contains",
                                    "value": "FINAL_ANSWER: UNANSWERABLE",
                                }
                            ],
                            "reference_pass": "FINAL_ANSWER: UNANSWERABLE",
                            "reference_fail": "FINAL_ANSWER: Galileo",
                        }
                    )
                else:
                    cases.append(
                        {
                            "inputs": {
                                "context": f"GROUNDED_{bucket} {tok} The treaty was signed in Geneva.",
                                "question": f"Where was the treaty signed for {tok}?",
                            },
                            "target": "FINAL_ANSWER: Geneva",
                            "checks": [
                                {"kind": "contains", "value": "FINAL_ANSWER: Geneva"}
                            ],
                            "reference_pass": "FINAL_ANSWER: Geneva",
                            "reference_fail": "FINAL_ANSWER: Paris",
                        }
                    )
        return json.dumps({"cases": cases})

    def _tag(prompt: Any, _ctx: Dict[str, Any]) -> str:
        text = str(prompt)
        m = re.search(r"Cases:\n(\[.*\])\s*$", text, flags=re.DOTALL)
        items = json.loads(m.group(1)) if m else []
        out: Dict[str, List[str]] = {}
        for item in items:
            blob = json.dumps(item.get("inputs", {}))
            matched = [
                fid
                for fid in ("A", "B", "C", "D", "E", "F")
                if f"_{fid} " in blob or f"_{fid}_" in blob
            ]
            out[item["id"]] = matched
        return json.dumps({"tags": out})

    def _sut(prompt: Any, _ctx: Dict[str, Any]) -> str:
        text = (
            "\n".join(m["content"] for m in prompt)
            if isinstance(prompt, list)
            else str(prompt)
        )
        has_b = "Decompose the problem concisely" in text
        has_d = "STRICT CONTRACT:" in text
        has_e = "never invent unsupported facts" in text
        m_idx = re.search(r"_(\d+)\b", text)
        idx = int(m_idx.group(1)) if m_idx else 1

        if task_id == "banking77_intent":
            if "UNRELATED_RECIPE_E" in text:
                return (
                    "FINAL_ANSWER: out_of_scope"
                    if has_e
                    else "FINAL_ANSWER: card_linking"
                )
            if "STRICT_CONTRACT_D" in text:
                return "FINAL_ANSWER: card_linking" if has_d else "card_linking"
            if has_b and idx % 5 == 0:
                return "FINAL_ANSWER: wrong_label"
            return (
                "FINAL_ANSWER: card_linking" if has_d or idx % 4 != 0 else "wrong_label"
            )

        if "NO_SPAN_E" in text:
            return "FINAL_ANSWER: UNANSWERABLE" if has_e else "FINAL_ANSWER: Galileo"
        if has_b and idx % 5 == 0:
            return "FINAL_ANSWER: Paris"
        return "FINAL_ANSWER: Geneva"

    return {
        "generator": MockLLM(
            response_generator=_gen, latency_ms=0.0, model_id="dryrun-gen"
        ),
        "tagger": MockLLM(
            response_generator=_tag, latency_ms=0.0, model_id="dryrun-tagger"
        ),
        "sut": MockLLM(response_generator=_sut, latency_ms=0.0, model_id="dryrun-sut"),
    }


def _resolve_live_client(
    spec: Optional[str],
    *,
    default_provider: str,
    default_model: str,
    max_tokens: int,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> LLMClient:
    """Resolve `gemini:<model>`, `anthropic:<model>`, or `module:callable` into an LLMClient."""
    if not spec:
        if default_provider == "anthropic":
            return AnthropicHTTPClient(
                model_id=default_model, max_tokens=max_tokens, temperature=temperature
            )
        return GeminiHTTPClient(
            model_id=default_model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
    if spec.startswith("gemini:"):
        return GeminiHTTPClient(
            model_id=spec.split(":", 1)[1],
            max_output_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
    if spec.startswith("anthropic:"):
        return AnthropicHTTPClient(
            model_id=spec.split(":", 1)[1],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return resolve_cli_client(client_spec=spec, role="live")


def run_pilot_task(
    task_id: str,
    *,
    mode: str = "dry-run",
    out_dir: Path,
    n_baseline_discovery: int = 100,
    n_baseline_confirmation: int = 40,
    n_per_factor_discovery: int = 35,
    n_per_factor_confirmation: int = 20,
    generator_client: Optional[LLMClient] = None,
    tagger_client: Optional[LLMClient] = None,
    sut_client: Optional[LLMClient] = None,
    max_workers: int = 8,
    reuse_testset: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    spec = build_v4_task_spec(task_id)
    factors: FactorSet = spec["factors"]
    template: PromptTemplate = spec["template"]
    seeds = load_real_pilot_seeds(task_id, limit=8)

    if mode == "dry-run":
        clients = _make_dry_run_clients(task_id)
        gen_c = generator_client or clients["generator"]
        tag_c = tagger_client or clients["tagger"]
        sut_c = sut_client or clients["sut"]
        eff_workers = 1
    else:
        if generator_client is None or sut_client is None:
            raise ValueError(
                "Live pilot mode requires generator_client and sut_client."
            )
        gen_c = generator_client
        tag_c = tagger_client
        sut_c = sut_client
        eff_workers = max_workers

    task_out = out_dir / task_id
    testset_dir = task_out / "testset"

    if reuse_testset and (testset_dir / "manifest.json").exists():
        print(f"[{task_id}] Reusing frozen TestSet from {testset_dir}", flush=True)
        ts = TestSet.load(testset_dir)
    else:
        print(
            f"[{task_id}] Generating TestSet with {gen_c.model_id} "
            f"(baseline={n_baseline_discovery}+{n_baseline_confirmation}, "
            f"per_factor={n_per_factor_discovery}+{n_per_factor_confirmation})...",
            flush=True,
        )
        gen = TestSetGenerator(
            client=gen_c,
            tagger_client=tag_c,
            n_baseline_discovery=n_baseline_discovery,
            n_baseline_confirmation=n_baseline_confirmation,
            n_per_factor_discovery=n_per_factor_discovery,
            n_per_factor_confirmation=n_per_factor_confirmation,
            max_oversample_factor=3.0,
            batch_size=10,
            tagger_batch_size=30,
            seed=seed,
            judge_fallback=False,
        )
        ts = gen.generate(
            factors,
            template,
            task_description=spec["task_description"],
            seed_examples=seeds or None,
            freeze=True,
        )
        ts.save(testset_dir)
        print(
            f"[{task_id}] Frozen TestSet saved: discovery={len(ts.cases_for_split('discovery'))}, "
            f"confirmation={len(ts.cases_for_split('confirmation'))}, "
            f"sha={ts.manifest.cases_sha256[:12]}",
            flush=True,
        )

    print(
        f"[{task_id}] Running 16-run 2(6-2)IV discovery screen on {sut_c.model_id} "
        f"(workers={eff_workers})...",
        flush=True,
    )
    exp = Experiment(
        factors=factors,
        template=template,
        design="2(6-2)IV",
        evaluator=Evaluator([CheckSuiteMetric()]),
        target_metric="check_suite",
        title=f"Section 8.4 Pilot ({task_id})",
    )
    results = exp.run(
        ts,
        split="discovery",
        client=sut_c,
        max_workers=eff_workers,
        cache_db=str(task_out / "response_cache.db"),
    )
    df = results.to_dataframe().rename(columns=FACTOR_ID_TO_NAME)
    df.to_csv(task_out / "discovery_results.csv", index=False)

    cov_by_name = ts.coverage(split="discovery", key_by="name")
    synth_audit = run_calibrated_attribution_audit(
        data=df,
        factor_cols=[FACTOR_ID_TO_NAME[f] for f in factors.ids],
        target_col="check_suite",
        equivalence_margin=0.02,
        coverage=cov_by_name,
        baseline_sample_ids=ts.baseline_sample_ids("discovery"),
        coverage_source=ts.provenance,
        test_set_sha256=ts.manifest.cases_sha256,
        judge_scored_fraction=ts.judge_scored_fraction("discovery"),
        generator_model_id=str(ts.manifest.generator.get("model_id", "")),
        model_id=sut_c.model_id,
        n_sign_flip=500,
        seed=seed,
    )
    (task_out / "synthetic_audit.md").write_text(
        synth_audit.to_markdown() + "\n", encoding="utf-8"
    )
    (task_out / "synthetic_audit.json").write_text(
        synth_audit.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    # 2-run confirmation comparison on untouched confirmation split (all-ON baseline vs pruned candidate)
    verdict_by_name = {d.factor_id: d.pruning_verdict for d in synth_audit.factors}
    pruned_levels = {
        fid: (
            0
            if verdict_by_name.get(FACTOR_ID_TO_NAME[fid])
            in {"SAFE_TO_PRUNE", "REMOVE_HARMFUL"}
            else 1
        )
        for fid in factors.ids
    }
    conf_design = DesignMatrix.from_dataframe(
        pd.DataFrame(
            [
                {"run_id": 0, **{fid: 1 for fid in factors.ids}},
                {"run_id": 1, **pruned_levels},
            ]
        ),
        factor_ids=factors.ids,
        plan_id="2-run-confirmation",
    )
    conf_exp = Experiment(
        factors=factors,
        template=template,
        design=conf_design,
        evaluator=Evaluator([CheckSuiteMetric()]),
        target_metric="check_suite",
        title=f"Section 8.4 Confirmation ({task_id})",
    )
    conf_results = conf_exp.run(
        ts,
        split="confirmation",
        client=sut_c,
        max_workers=eff_workers,
        cache_db=str(task_out / "response_cache.db"),
    )
    conf_df = conf_results.to_dataframe()
    conf_df.to_csv(task_out / "confirmation_results.csv", index=False)
    conf_base_mean = float(conf_df[conf_df["run_id"] == 0]["check_suite"].mean())
    conf_pruned_mean = float(conf_df[conf_df["run_id"] == 1]["check_suite"].mean())

    real_audit = load_real_v4_reference_audit(task_id)
    summary: Dict[str, Any] = {
        "task_id": task_id,
        "mode": mode,
        "testset_sha256": ts.manifest.cases_sha256,
        "generator_model_id": ts.manifest.generator.get("model_id"),
        "tagger_model_id": ts.manifest.generator.get("tagger_model_id"),
        "sut_model_id": sut_c.model_id,
        "discovery_items": len(ts.cases_for_split("discovery")),
        "confirmation_items": len(ts.cases_for_split("confirmation")),
        "evidence_tier": synth_audit.evidence_tier,
        "warnings": synth_audit.warnings,
        "verdicts": verdict_by_name,
        "confirmation": {
            "pruned_levels": pruned_levels,
            "baseline_mean": conf_base_mean,
            "pruned_mean": conf_pruned_mean,
            "delta_pruned_minus_baseline": conf_pruned_mean - conf_base_mean,
        },
    }
    if real_audit is not None:
        summary["vs_real_v4"] = compare_synthetic_vs_real_audit(real_audit, synth_audit)
    (task_out / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Run Section 8.4 pilot (dry-run offline by default; --mode live for real API)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["banking77_intent", "squad2_grounded_qa"],
        choices=("banking77_intent", "squad2_grounded_qa"),
    )
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="Generator backend: 'anthropic:<model>', 'gemini:<model>', or 'module:callable'",
    )
    parser.add_argument(
        "--tagger",
        type=str,
        default=None,
        help="Tagger backend: 'anthropic:<model>', 'gemini:<model>', or 'module:callable'",
    )
    parser.add_argument(
        "--sut",
        type=str,
        default="gemini:gemini-2.5-flash",
        help="Model under test backend (default: gemini:gemini-2.5-flash)",
    )
    parser.add_argument("--n-baseline-discovery", type=int, default=100)
    parser.add_argument("--n-baseline-confirmation", type=int, default=40)
    parser.add_argument("--n-per-factor-discovery", type=int, default=35)
    parser.add_argument("--n-per-factor-confirmation", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-reuse-testset", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "research" / "results" / "testgen_pilot"),
    )
    args = parser.parse_args()

    gen_c = tag_c = sut_c = None
    if args.mode == "live":
        gen_c = _resolve_live_client(
            args.generator,
            default_provider="gemini",
            default_model="gemini-2.5-pro",
            max_tokens=8192,
            temperature=0.7,
            json_mode=True,
        )
        tag_c = _resolve_live_client(
            args.tagger or args.generator,
            default_provider="gemini",
            default_model="gemini-2.5-pro",
            max_tokens=4096,
            temperature=0.0,
            json_mode=True,
        )
        sut_c = _resolve_live_client(
            args.sut,
            default_provider="gemini",
            default_model="gemini-2.5-flash",
            max_tokens=512,
            temperature=0.0,
            json_mode=False,
        )

    out_dir = Path(args.out)
    for task_id in args.tasks:
        res = run_pilot_task(
            task_id,
            mode=args.mode,
            out_dir=out_dir,
            n_baseline_discovery=args.n_baseline_discovery,
            n_baseline_confirmation=args.n_baseline_confirmation,
            n_per_factor_discovery=args.n_per_factor_discovery,
            n_per_factor_confirmation=args.n_per_factor_confirmation,
            generator_client=gen_c,
            tagger_client=tag_c,
            sut_client=sut_c,
            max_workers=args.max_workers,
            reuse_testset=not args.no_reuse_testset,
        )
        print(json.dumps(res, indent=2), flush=True)

    if args.mode == "live":
        for label, client in (("generator", gen_c), ("tagger", tag_c), ("sut", sut_c)):
            if isinstance(client, (GeminiHTTPClient, AnthropicHTTPClient)):
                print(
                    f"[usage:{label}:{client.model_id}] calls={client.total_calls}, "
                    f"in_tok={client.total_input_tokens}, out_tok={client.total_output_tokens}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
