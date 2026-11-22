"""
PromptPrism vs. DSPy MIPROv2 Benchmark Harness (Track A + Track B).

Track A (Head-to-Head at Equal Call Budget):
    Compares PromptPrism's locked Real-v4 2^(6-2)IV + Stage-2 prompt against
    DSPy MIPROv2 (compiled on the exact same `screen` [N=120] + `validation` [N=60]
    splits under an equal-or-lower task-LM call budget <= 2,640 calls/task)
    on untouched `clean_confirmation` (N=240) plus 5 paired perturbation
    environments (`repeat_r2`, `fmt_whitespace`, `fmt_delimiter`,
    `word_paraphrase_1`, `word_paraphrase_2`). Both systems run live on the
    exact same `gemini-2.5-flash` endpoint (`temperature=0.0`, thinking off,
    `max_tokens=512`) and are scored by identical Real-v4 deterministic scorers.

Track B (Complementary Workflow: DSPy -> PromptPrism Audit, Prune & Certify):
    Takes the winning compiled DSPy MIPROv2 program, decomposes its rewritten
    instruction and individual compiled demonstrations into binary PromptPrism
    factors, runs a balanced orthogonal screen (`2^(k-p)`) on the `screen`
    split using DSPy's exact ChatAdapter formatting, runs calibrated
    `run_calibrated_attribution_audit` (`equivalence_margin=0.02`,
    `n_sign_flip=2000`) with measured prompt-token costs, prunes any
    `REMOVE_HARMFUL` or `SAFE_TO_PRUNE` components, and validates
    DSPy-Pruned vs. DSPy-Full on untouched `clean_confirmation` (N=240)
    plus paired perturbations and frozen synthetic edge cases (OOS / unanswerable).

Run with the isolated `.venv_dspy` environment:
    .venv_dspy/bin/python -m research.experiments.dspy_benchmark \
        --tasks banking77_intent squad2_grounded_qa gsm8k_math \
        --out research/results/dspy_benchmark
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import dspy
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

from prompt_prism.analysis.diagnostics import (
    CalibratedAttributionAudit,
    run_calibrated_attribution_audit,
)
from prompt_prism.analysis.selection import (
    evaluate_simultaneous_perturbation_certificate,
)
from prompt_prism.core.models import OperatingEnvelopeSpec
from prompt_prism.design.generators import (
    FractionalFactorialGenerator,
    FullFactorialGenerator,
)
from prompt_prism.evaluation.check_suite import CheckSuiteMetric
from prompt_prism.runner.client import LLMClient, LLMResponse
from prompt_prism.testgen.models import TestSet
from research.real_v4.perturbations import (
    parser_hashes_by_task,
    render_prompt_instance,
    score_task_response,
)
from research.real_v4.selection import run_id_to_levels


class GeminiMultiTurnClient(LLMClient):
    """Gemini v1beta client preserving genuine multi-turn user/model roles (exact Real-v4 decoding)."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        thinking_budget: int = 0,
        max_retries: int = 5,
    ) -> None:
        self.model_id = model_id
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
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

    def generate(self, prompt: Any, **kwargs: Any) -> LLMResponse:
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        if isinstance(prompt, list):
            for msg in prompt:
                role = str(msg.get("role", "user")).lower()
                text = str(msg.get("content", ""))
                if not text:
                    continue
                if role == "system":
                    system_parts.append(text)
                else:
                    g_role = "model" if role in ("assistant", "model") else "user"
                    # Merge consecutive same-role turns cleanly for Gemini API validity
                    if contents and contents[-1]["role"] == g_role:
                        contents[-1]["parts"].append({"text": text})
                    else:
                        contents.append({"role": g_role, "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": str(prompt)}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(kwargs.get("temperature", self.temperature)),
                "topP": 1.0,
                "maxOutputTokens": int(
                    kwargs.get("max_tokens", self.max_output_tokens)
                ),
                "thinkingConfig": {"thinkingBudget": self.thinking_budget},
            },
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
        parts = cands[0].get("content", {}).get("parts", []) if cands else []
        text = "".join(
            str(p.get("text", ""))
            for p in parts
            if isinstance(p, dict) and not p.get("thought")
        )
        usage = data.get("usageMetadata") or {}
        tok_in = int(usage.get("promptTokenCount") or 0)
        tok_out = int(usage.get("candidatesTokenCount") or 0)
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


ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "research" / "data" / "real_v4" / "splits"
REAL_V4_RESULTS_DIR = ROOT / "research" / "results" / "real_v4"
TESTGEN_PILOT_DIR = ROOT / "research" / "results" / "testgen_pilot"

PAIRED_ENV_IDS: Tuple[str, ...] = (
    "ref",
    "repeat_r2",
    "fmt_whitespace",
    "fmt_delimiter",
    "word_paraphrase_1",
    "word_paraphrase_2",
)

PERTURBATION_ONLY_ENV_IDS: Tuple[str, ...] = (
    "repeat_r2",
    "fmt_whitespace",
    "fmt_delimiter",
    "word_paraphrase_1",
    "word_paraphrase_2",
)


# ---------------------------------------------------------------------------
# Data Loading & Real-v4 Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_real_v4_splits(task_id: str) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "screen": load_jsonl(SPLITS_DIR / f"{task_id}_screen.jsonl"),
        "validation": load_jsonl(SPLITS_DIR / f"{task_id}_validation.jsonl"),
        "clean_confirmation": load_jsonl(
            SPLITS_DIR / f"{task_id}_clean_confirmation.jsonl"
        ),
    }


def load_promptprism_v4_winner(
    task_id: str, model_id: str = "gemini-2.5-flash"
) -> Tuple[int, Dict[str, int], Dict[str, Any]]:
    lock = json.loads(
        (REAL_V4_RESULTS_DIR / "final_selection_lock.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (REAL_V4_RESULTS_DIR / "summary.json").read_text(encoding="utf-8")
    )
    cell_key = f"{task_id}::{model_id}"
    run_id = int(lock["cells"][cell_key]["promptprism"]["primary"])
    levels = run_id_to_levels(run_id)
    v4_ref = summary["cell_level_breakdown"]["cells"].get(cell_key, {})
    return run_id, levels, v4_ref


# ---------------------------------------------------------------------------
# DSPy Task Signatures, Modules, Perturbation Wrappers & Metrics
# ---------------------------------------------------------------------------


class Banking77Signature(dspy.Signature):
    """Classify the customer banking support query into exactly one intent label from the allowed_intents list. Return only the exact intent label string."""

    allowed_intents: str = dspy.InputField(
        desc="Comma-separated list of valid banking intent labels"
    )
    query: str = dspy.InputField(desc="Customer banking support query")
    answer: str = dspy.OutputField(desc="Exactly one intent label from allowed_intents")


class Squad2Signature(dspy.Signature):
    """Answer the question using only the shortest exact answer span from context. If the context does not state the answer, reply UNANSWERABLE."""

    context: str = dspy.InputField(desc="Reference passage")
    question: str = dspy.InputField(desc="Question about the passage")
    answer: str = dspy.OutputField(
        desc="Shortest exact answer span from context, or UNANSWERABLE"
    )


class Gsm8kSignature(dspy.Signature):
    """Solve the grade-school arithmetic word problem step by step and output only the final numeric answer."""

    question: str = dspy.InputField(desc="Math word problem")
    answer: str = dspy.OutputField(
        desc="Final numeric answer (digits only, optional minus sign or decimal)"
    )


class Re3dSignature(dspy.Signature):
    """Extract named entities from the sentence as a single-line JSON object with exact keys: person, organization, location, money, temporal (each mapping to a list of strings)."""

    sentence: str = dspy.InputField(desc="Input sentence")
    answer: str = dspy.OutputField(desc="Valid single-line JSON object")


class SciqSignature(dspy.Signature):
    """Select the single correct option letter (A, B, C, or D) using the support passage."""

    support: str = dspy.InputField(desc="Scientific support passage")
    question: str = dspy.InputField(desc="Multiple-choice question")
    options: str = dspy.InputField(desc="Four labeled options A, B, C, D")
    answer: str = dspy.OutputField(desc="Single option letter: A, B, C, or D")


def build_dspy_student(task_id: str) -> Tuple[dspy.Module, str]:
    """Return DSPy student module (natural architecture per task) and base instruction."""
    if task_id == "banking77_intent":
        mod = dspy.Predict(Banking77Signature)
        return mod, Banking77Signature.instructions
    if task_id == "squad2_grounded_qa":
        mod = dspy.Predict(Squad2Signature)
        return mod, Squad2Signature.instructions
    if task_id == "gsm8k_math":
        mod = dspy.ChainOfThought(Gsm8kSignature)
        return mod, Gsm8kSignature.instructions
    if task_id == "re3d_entity_extraction":
        mod = dspy.Predict(Re3dSignature)
        return mod, Re3dSignature.instructions
    if task_id == "sciq_verification":
        mod = dspy.Predict(SciqSignature)
        return mod, SciqSignature.instructions
    raise ValueError(f"Unsupported task_id={task_id!r}")


def _apply_dspy_input_perturbation(
    inputs: Dict[str, str], primary_key: str, env_id: str
) -> Dict[str, str]:
    """Apply Real-v4-equivalent semantic-preserving perturbation to DSPy input fields."""
    out = dict(inputs)
    val = out[primary_key]
    if env_id in ("ref", "repeat_r2"):
        return out
    if env_id == "fmt_whitespace":
        out[primary_key] = f"\n\n{val}\n\n"
    elif env_id == "fmt_delimiter":
        hdr = primary_key.upper()
        out[primary_key] = f"=== {hdr} ===\n{val}\n=== END {hdr} ==="
    elif env_id == "word_paraphrase_1":
        out[primary_key] = (
            f"Carefully review the provided input below and supply your response.\n{val}"
        )
    elif env_id == "word_paraphrase_2":
        out[primary_key] = (
            f"Examine the input material below and provide the requested response.\n{val}"
        )
    else:
        raise ValueError(f"Unsupported env_id={env_id!r}")
    return out


def item_to_dspy_inputs(
    task_id: str, item: Mapping[str, Any], env_id: str = "ref"
) -> Tuple[Dict[str, str], str]:
    """Convert a Real-v4 item dict into DSPy signature input kwargs (+ primary field name)."""
    if task_id == "banking77_intent":
        raw = {
            "allowed_intents": ", ".join(str(o) for o in item["options"]),
            "query": str(item["input_text"]),
        }
        return _apply_dspy_input_perturbation(raw, "query", env_id), "query"
    if task_id == "squad2_grounded_qa":
        raw = {
            "context": str(item["context"]),
            "question": str(item["question"]),
        }
        return _apply_dspy_input_perturbation(raw, "question", env_id), "question"
    if task_id == "gsm8k_math":
        raw = {"question": str(item["question"])}
        return _apply_dspy_input_perturbation(raw, "question", env_id), "question"
    if task_id == "re3d_entity_extraction":
        raw = {"sentence": str(item["input_text"])}
        return _apply_dspy_input_perturbation(raw, "sentence", env_id), "sentence"
    if task_id == "sciq_verification":
        opts = item["options"]
        opt_str = "\n".join(f"({k}) {opts[k]}" for k in ("A", "B", "C", "D"))
        raw = {
            "support": str(item["support"]),
            "question": str(item["question"]),
            "options": opt_str,
        }
        return _apply_dspy_input_perturbation(raw, "question", env_id), "question"
    raise ValueError(f"Unsupported task_id={task_id!r}")


def item_to_dspy_example(task_id: str, item: Mapping[str, Any]) -> dspy.Example:
    inputs, _ = item_to_dspy_inputs(task_id, item, env_id="ref")
    if task_id == "squad2_grounded_qa":
        gold_list = item.get("target") or []
        ans = str(gold_list[0]) if gold_list else "UNANSWERABLE"
    elif task_id == "re3d_entity_extraction":
        ans = json.dumps(item["target"], ensure_ascii=False)
    else:
        ans = str(item["target"])
    ex = dspy.Example(**inputs, answer=ans, _raw_item=dict(item)).with_inputs(
        *inputs.keys()
    )
    return ex


def make_dspy_metric(task_id: str) -> Callable[[dspy.Example, Any, Any], float]:
    def _metric(example: dspy.Example, pred: Any, trace: Any = None) -> float:
        raw_out = str(getattr(pred, "answer", "") or "")
        return float(score_task_response(task_id, raw_out, example._raw_item))

    return _metric


# ---------------------------------------------------------------------------
# Usage Accounting & Paired Statistics Helpers
# ---------------------------------------------------------------------------


def snapshot_lm_usage(lm: dspy.LM) -> Dict[str, int]:
    """Summarize total calls, prompt tokens, completion tokens from dspy.LM.history."""
    calls = len(lm.history)
    in_tok = 0
    out_tok = 0
    reason_tok = 0
    for entry in lm.history:
        usage = entry.get("usage") or {}
        in_tok += int(usage.get("prompt_tokens") or 0)
        out_tok += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        reason_tok += int(details.get("reasoning_tokens") or 0)
    return {
        "calls": calls,
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "reasoning_tokens": reason_tok,
    }


def diff_usage(after: Mapping[str, int], before: Mapping[str, int]) -> Dict[str, int]:
    return {k: int(after[k] - before[k]) for k in after}


def paired_item_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    group_ids: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute paired item-clustered (and group-clustered if provided) diff = mean(A - B)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    if group_ids is not None and len(set(group_ids)) < n:
        df_g = pd.DataFrame({"g": list(group_ids), "d": diff})
        g_means = df_g.groupby("g")["d"].mean().to_numpy(dtype=float)
        n_eff = len(g_means)
        sd = float(np.std(g_means, ddof=1)) if n_eff > 1 else 0.0
        se = sd / np.sqrt(max(1, n_eff))
        df_t = max(1, n_eff - 1)
    else:
        n_eff = n
        sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(max(1, n))
        df_t = max(1, n - 1)

    if se <= 1e-12:
        t_stat = 0.0
        p_val = 1.0 if abs(mean_diff) <= 1e-12 else 0.0
        ci_lo = mean_diff
        ci_hi = mean_diff
    else:
        t_stat = float(mean_diff / se)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_t))
        ci_lo = float(mean_diff - t_crit * se)
        ci_hi = float(mean_diff + t_crit * se)
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "delta_a_minus_b": mean_diff,
        "se": float(se),
        "n_clusters": int(n_eff),
        "t_stat": t_stat,
        "p_value": p_val,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
    }


def _dspy_prompt_to_messages(
    module: dspy.Module, inputs: Mapping[str, str]
) -> List[Dict[str, str]]:
    """Render exact DSPy ChatAdapter messages as standard OpenAI/Gemini chat dicts."""
    _, pred = list(module.named_predictors())[0]
    adapter = dspy.ChatAdapter()
    prompt_obj = adapter.format(pred.signature, pred.demos, dict(inputs))
    msgs: List[Dict[str, str]] = []
    if getattr(prompt_obj, "system", None):
        msgs.append({"role": "system", "content": str(prompt_obj.system)})
    for m in getattr(prompt_obj, "messages", ()):
        role = str(getattr(m, "role", "user"))
        parts = getattr(m, "parts", ())
        text = "".join(str(getattr(p, "text", "")) for p in parts)
        msgs.append({"role": role, "content": text})
    return msgs


def _parse_dspy_answer(module: dspy.Module, raw_output: str) -> str:
    """Parse DSPy ChatAdapter output fields back into the `answer` string (with raw fallback)."""
    _, pred = list(module.named_predictors())[0]
    adapter = dspy.ChatAdapter()
    try:
        parsed = adapter.parse(pred.signature, raw_output)
        ans = parsed.get("answer")
        if ans is not None and str(ans).strip():
            return str(ans).strip()
    except Exception:
        pass
    return str(raw_output or "").strip()


# ---------------------------------------------------------------------------
# Parallel Evaluators on Pinned `gemini-2.5-flash` (`thinkingBudget=0`)
# ---------------------------------------------------------------------------


def evaluate_promptprism_on_envs(
    *,
    task_id: str,
    factor_levels: Mapping[str, int],
    items: Sequence[Mapping[str, Any]],
    env_ids: Sequence[str],
    client: GeminiMultiTurnClient,
    max_workers: int = 10,
) -> Dict[str, Any]:
    """Evaluate a PromptPrism factor configuration across paired environments live."""
    scores_by_env: Dict[str, List[float]] = {e: [0.0] * len(items) for e in env_ids}
    prompt_tokens_ref: List[int] = []
    completion_tokens_ref: List[int] = []
    latencies_ref: List[float] = []
    lock = threading.Lock()

    jobs: List[Tuple[str, int, Mapping[str, Any]]] = [
        (env_id, idx, item) for env_id in env_ids for idx, item in enumerate(items)
    ]

    def _run_one(
        job: Tuple[str, int, Mapping[str, Any]],
    ) -> Tuple[str, int, float, int, int, float]:
        env_id, idx, item = job
        inst = render_prompt_instance(task_id, item, factor_levels, env_id=env_id)
        messages = [
            {"role": "system", "content": inst.system_text},
            {"role": "user", "content": inst.user_text},
        ]
        resp = client.generate(messages)
        sc = (
            0.0
            if resp.error
            else float(score_task_response(task_id, resp.content, item))
        )
        usage = resp.token_usage or {}
        return (
            env_id,
            idx,
            sc,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            float(resp.latency_ms or 0.0),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            env_id, idx, sc, p_tok, c_tok, lat = fut.result()
            with lock:
                scores_by_env[env_id][idx] = sc
                if env_id == "ref":
                    prompt_tokens_ref.append(p_tok)
                    completion_tokens_ref.append(c_tok)
                    latencies_ref.append(lat)

    return {
        "scores_by_env": scores_by_env,
        "env_means": {e: float(np.mean(v)) for e, v in scores_by_env.items()},
        "mean_prompt_tokens_ref": (
            float(np.mean(prompt_tokens_ref)) if prompt_tokens_ref else 0.0
        ),
        "mean_completion_tokens_ref": (
            float(np.mean(completion_tokens_ref)) if completion_tokens_ref else 0.0
        ),
        "mean_latency_ms_ref": float(np.mean(latencies_ref)) if latencies_ref else 0.0,
    }


def evaluate_dspy_module_on_envs(
    *,
    task_id: str,
    module: dspy.Module,
    items: Sequence[Mapping[str, Any]],
    env_ids: Sequence[str],
    client: GeminiMultiTurnClient,
    max_workers: int = 10,
) -> Dict[str, Any]:
    """Evaluate a DSPy module across paired environments using the exact pinned GeminiHTTPClient."""
    scores_by_env: Dict[str, List[float]] = {e: [0.0] * len(items) for e in env_ids}
    prompt_tokens_ref: List[int] = []
    completion_tokens_ref: List[int] = []
    latencies_ref: List[float] = []
    lock = threading.Lock()

    jobs: List[Tuple[str, int, Mapping[str, Any]]] = [
        (env_id, idx, item) for env_id in env_ids for idx, item in enumerate(items)
    ]

    def _run_one(
        job: Tuple[str, int, Mapping[str, Any]],
    ) -> Tuple[str, int, float, int, int, float]:
        env_id, idx, item = job
        inp, _ = item_to_dspy_inputs(task_id, item, env_id=env_id)
        messages = _dspy_prompt_to_messages(module, inp)
        resp = client.generate(messages)
        if resp.error:
            sc = 0.0
        else:
            parsed_ans = _parse_dspy_answer(module, resp.content)
            sc = float(score_task_response(task_id, parsed_ans, item))
        usage = resp.token_usage or {}
        return (
            env_id,
            idx,
            sc,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            float(resp.latency_ms or 0.0),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_run_one, j) for j in jobs]
        for fut in as_completed(futs):
            env_id, idx, sc, p_tok, c_tok, lat = fut.result()
            with lock:
                scores_by_env[env_id][idx] = sc
                if env_id == "ref":
                    prompt_tokens_ref.append(p_tok)
                    completion_tokens_ref.append(c_tok)
                    latencies_ref.append(lat)

    return {
        "scores_by_env": scores_by_env,
        "env_means": {e: float(np.mean(v)) for e, v in scores_by_env.items()},
        "mean_prompt_tokens_ref": (
            float(np.mean(prompt_tokens_ref)) if prompt_tokens_ref else 0.0
        ),
        "mean_completion_tokens_ref": (
            float(np.mean(completion_tokens_ref)) if completion_tokens_ref else 0.0
        ),
        "mean_latency_ms_ref": float(np.mean(latencies_ref)) if latencies_ref else 0.0,
    }


def compute_stability_cert(
    *,
    task_id: str,
    system_label: str,
    eval_out: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    epsilon_stability: float = 0.02,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    p_hashes = parser_hashes_by_task()
    envelope = OperatingEnvelopeSpec(
        task_id=f"{task_id}:{system_label}",
        model_id="gemini-2.5-flash",
        model_version="gemini-2.5-flash",
        dataset_partition_hash=f"{task_id}_clean_confirmation_240",
        parser_hash=p_hashes[task_id],
        decoding_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 512,
            "thinking_budget": 0,
        },
        perturbation_env_ids=list(PERTURBATION_ONLY_ENV_IDS),
    )
    scores_by_env = eval_out["scores_by_env"]
    group_ids = [str(it["group_id"]) for it in items]
    cert = evaluate_simultaneous_perturbation_certificate(
        envelope=envelope,
        reference_scores=scores_by_env["ref"],
        perturbation_scores_by_env={
            e: scores_by_env[e] for e in PERTURBATION_ONLY_ENV_IDS
        },
        group_ids=group_ids,
        epsilon_stability=epsilon_stability,
        n_bootstrap=n_bootstrap,
        seed=42,
    )
    worst_env_id = min(
        PERTURBATION_ONLY_ENV_IDS, key=lambda e: eval_out["env_means"][e]
    )
    return {
        "status": cert.status,
        "reference_mean": float(cert.reference_mean),
        "max_drop_est": float(cert.d_stability_estimate),
        "u95_sim": float(cert.d_stability_upper_95_sim),
        "worst_env_id": worst_env_id,
        "env_means": dict(eval_out["env_means"]),
    }


# ---------------------------------------------------------------------------
# Track B: Decompose Compiled DSPy Module -> PromptPrism Orthogonal Audit
# ---------------------------------------------------------------------------


def clone_dspy_module_with_levels(
    *,
    compiled_module: dspy.Module,
    base_instruction: str,
    compiled_instruction: str,
    compiled_demos: Sequence[Any],
    factor_levels: Mapping[str, int],
    demo_factor_names: Sequence[str],
    has_instruction_factor: bool,
) -> dspy.Module:
    mod = copy.deepcopy(compiled_module)
    _, pred = list(mod.named_predictors())[0]
    if has_instruction_factor:
        instr = (
            compiled_instruction
            if int(factor_levels["dspy_instruction"]) == 1
            else base_instruction
        )
        pred.signature = pred.signature.with_instructions(instr)
    kept_demos = [
        demo
        for demo, fname in zip(compiled_demos, demo_factor_names)
        if int(factor_levels[fname]) == 1
    ]
    pred.demos = kept_demos
    return mod


def build_track_b_design(n_factors: int):
    if n_factors <= 3:
        return FullFactorialGenerator.generate(n_factors)
    if n_factors == 4:
        return FractionalFactorialGenerator.from_plan_id("2(4-1)IV")
    if n_factors == 5:
        return FractionalFactorialGenerator.from_plan_id("2(5-1)V")
    if n_factors == 6:
        return FractionalFactorialGenerator.from_plan_id("2(6-2)IV")
    if n_factors == 7:
        return FractionalFactorialGenerator.from_plan_id("2(7-3)IV")
    return FractionalFactorialGenerator.from_plan_id("2(8-4)IV")


def run_track_b_audit_and_prune(
    *,
    task_id: str,
    compiled_module: dspy.Module,
    base_instruction: str,
    screen_items: Sequence[Mapping[str, Any]],
    confirmation_items: Sequence[Mapping[str, Any]],
    client: GeminiMultiTurnClient,
    out_dir: Path,
    max_workers: int = 10,
    equivalence_margin: float = 0.02,
) -> Dict[str, Any]:
    """Decompose DSPy's compiled prompt into binary PromptPrism factors, diagnose, prune, and confirm."""
    _, pred = list(compiled_module.named_predictors())[0]
    compiled_instruction = str(pred.signature.instructions or "")
    compiled_demos = list(getattr(pred, "demos", []) or [])

    has_instruction_factor = compiled_instruction.strip() != base_instruction.strip()
    factor_names: List[str] = []
    if has_instruction_factor:
        factor_names.append("dspy_instruction")
    # Cap at up to 7 individual demo factors (keeps k <= 8, 16-run Res IV/V)
    demo_groups: List[List[Any]] = []
    if len(compiled_demos) <= 7:
        for d in compiled_demos:
            demo_groups.append([d])
    else:
        # Pair excess demos cleanly so k <= 8
        for i in range(0, len(compiled_demos), 2):
            demo_groups.append(compiled_demos[i : i + 2])

    demo_factor_names = [f"dspy_demo_{i + 1}" for i in range(len(demo_groups))]
    factor_names.extend(demo_factor_names)

    if len(factor_names) < 2:
        return {
            "skipped": True,
            "reason": f"Compiled DSPy program has only {len(factor_names)} active toggleable component(s).",
            "compiled_instruction": compiled_instruction,
            "num_compiled_demos": len(compiled_demos),
        }

    design = build_track_b_design(len(factor_names))
    design_label = design.plan_id or f"FullFactorial(2^{len(factor_names)})"
    design_fids = list(design.factor_ids)
    fid_to_name = dict(zip(design_fids, factor_names))

    print(
        f"[{task_id}][Track B] Auditing {len(factor_names)} DSPy components "
        f"({factor_names}) via {design_label} ({len(design.runs)} runs x {len(screen_items)} items)...",
        flush=True,
    )

    # Execute orthogonal screen across all (run_id, item) pairs in parallel
    rows: List[Dict[str, Any]] = []
    lock = threading.Lock()
    jobs: List[Tuple[int, Dict[str, int], int, Mapping[str, Any]]] = []
    for run in design.runs:
        named_levels = {
            fid_to_name[fid]: int(run.factor_levels[fid]) for fid in design_fids
        }
        for idx, item in enumerate(screen_items):
            jobs.append((int(run.run_id), named_levels, idx, item))

    def _build_mod_from_groups(named_levels: Mapping[str, int]) -> dspy.Module:
        mod = copy.deepcopy(compiled_module)
        _, p_sub = list(mod.named_predictors())[0]
        if has_instruction_factor:
            instr = (
                compiled_instruction
                if int(named_levels["dspy_instruction"]) == 1
                else base_instruction
            )
            p_sub.signature = p_sub.signature.with_instructions(instr)
        kept: List[Any] = []
        for grp, dname in zip(demo_groups, demo_factor_names):
            if int(named_levels[dname]) == 1:
                kept.extend(grp)
        p_sub.demos = kept
        return mod

    # Pre-build the 8 or 16 run modules once
    run_modules: Dict[int, dspy.Module] = {}
    for run in design.runs:
        nl = {fid_to_name[fid]: int(run.factor_levels[fid]) for fid in design_fids}
        run_modules[int(run.run_id)] = _build_mod_from_groups(nl)

    def _eval_screen_job(
        job: Tuple[int, Dict[str, int], int, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        run_id, named_levels, _, item = job
        mod = run_modules[run_id]
        inp, _ = item_to_dspy_inputs(task_id, item, env_id="ref")
        msgs = _dspy_prompt_to_messages(mod, inp)
        resp = client.generate(msgs)
        if resp.error:
            sc = 0.0
        else:
            parsed_ans = _parse_dspy_answer(mod, resp.content)
            sc = float(score_task_response(task_id, parsed_ans, item))
        usage = resp.token_usage or {}
        rec: Dict[str, Any] = {
            "run_id": run_id,
            "sample_id": str(item["sample_id"]),
            "score": sc,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }
        rec.update(named_levels)
        return rec

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_eval_screen_job, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                rows.append(rec)

    df_screen = pd.DataFrame(rows)
    # Measure exact empirical input token delta for each DSPy component
    token_costs: Dict[str, int] = {}
    for fname in factor_names:
        m1 = float(df_screen.loc[df_screen[fname] == 1, "prompt_tokens"].mean())
        m0 = float(df_screen.loc[df_screen[fname] == 0, "prompt_tokens"].mean())
        token_costs[fname] = max(0, int(round(m1 - m0)))

    audit: CalibratedAttributionAudit = run_calibrated_attribution_audit(
        data=df_screen,
        factor_cols=factor_names,
        target_col="score",
        sample_col="sample_id",
        alpha=0.05,
        maximize=True,
        equivalence_margin=equivalence_margin,
        token_costs=token_costs,
        n_sign_flip=2000,
        seed=42,
    )
    (out_dir / "track_b_dspy_audit.md").write_text(
        audit.to_markdown() + "\n", encoding="utf-8"
    )
    (out_dir / "track_b_dspy_audit.json").write_text(
        audit.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    verdict_by_factor = {f.factor_id: f.pruning_verdict for f in audit.factors}
    pruned_levels = {
        fname: (
            0 if verdict_by_factor[fname] in {"REMOVE_HARMFUL", "SAFE_TO_PRUNE"} else 1
        )
        for fname in factor_names
    }
    removed_components = [f for f, lv in pruned_levels.items() if lv == 0]
    pruned_module = _build_mod_from_groups(pruned_levels)

    # Validate Pruned DSPy module on untouched clean_confirmation + paired perturbations
    pruned_eval = None
    pruned_cert = None
    if removed_components:
        print(
            f"[{task_id}][Track B] Pruned {len(removed_components)}/{len(factor_names)} "
            f"DSPy component(s) ({removed_components}); validating on 240 confirmation x 6 envs...",
            flush=True,
        )
        pruned_eval = evaluate_dspy_module_on_envs(
            task_id=task_id,
            module=pruned_module,
            items=confirmation_items,
            env_ids=PAIRED_ENV_IDS,
            client=client,
            max_workers=max_workers,
        )
        pruned_cert = compute_stability_cert(
            task_id=task_id,
            system_label="dspy_pruned_by_promptprism",
            eval_out=pruned_eval,
            items=confirmation_items,
        )
    else:
        print(
            f"[{task_id}][Track B] Zero components pruned (all KEEP or INCONCLUSIVE).",
            flush=True,
        )

    return {
        "skipped": False,
        "design_name": design_label,
        "num_runs": len(design.runs),
        "screen_calls": len(rows),
        "factor_names": factor_names,
        "token_costs": token_costs,
        "noise_floor": audit.noise_floor.model_dump(),
        "estimated_prunable_tokens": audit.estimated_prunable_tokens,
        "factors": [
            {
                "factor": f.factor_id,
                "delta": f.effect_delta,
                "se": f.se_item_clustered,
                "q_bh": f.q_value_bh,
                "p_flip": f.p_value_sign_flip,
                "tost_p": f.tost_p_value,
                "verdict": f.pruning_verdict,
                "token_cost": f.token_cost,
            }
            for f in audit.factors
        ],
        "removed_components": removed_components,
        "pruned_levels": pruned_levels,
        "pruned_eval": pruned_eval,
        "pruned_cert": pruned_cert,
    }


# ---------------------------------------------------------------------------
# Bonus Edge-Case Guardrail Audit (Frozen Synthetic OOS / Unanswerable Cases)
# ---------------------------------------------------------------------------


def evaluate_edge_case_guardrail(
    *,
    task_id: str,
    dspy_module: dspy.Module,
    prism_levels: Mapping[str, int],
    client: GeminiMultiTurnClient,
    max_workers: int = 10,
) -> Optional[Dict[str, Any]]:
    """Evaluate DSPy compiled prompt vs. PromptPrism (+ guardrail) on frozen synthetic edge cases."""
    ts_dir = TESTGEN_PILOT_DIR / task_id / "testset"
    if not (ts_dir / "manifest.json").exists():
        return None
    ts = TestSet.load(ts_dir)
    # Use the held-out confirmation targeted cases for factor E (grounding_guardrail)
    edge_cases = [
        c
        for c in ts.cases_for_split("confirmation")
        if "E" in (c.targets_factors or []) and c.intended_factor == "E"
    ]
    if not edge_cases:
        return None

    metric = CheckSuiteMetric(allow_custom_checks=False)
    options_b77: Optional[List[str]] = None
    if task_id == "banking77_intent":
        sample_split = load_jsonl(SPLITS_DIR / "banking77_intent_pilot.jsonl")
        options_b77 = list(sample_split[0]["options"])

    def _eval_dspy_case(case) -> float:
        if task_id == "banking77_intent":
            inp = {
                "allowed_intents": ", ".join(options_b77 or []),
                "query": str(case.inputs["query"]),
            }
        else:
            inp = {
                "context": str(case.inputs["context"]),
                "question": str(case.inputs["question"]),
            }
        msgs = _dspy_prompt_to_messages(dspy_module, inp)
        resp = client.generate(msgs)
        if resp.error:
            return 0.0
        ans = _parse_dspy_answer(dspy_module, resp.content)
        ctx = {"checks": [chk.model_dump(mode="json") for chk in case.checks]}
        return float(metric.compute(ans, target=case.target, input_data=ctx))

    def _eval_prism_case(case, levels: Mapping[str, int]) -> float:
        if task_id == "banking77_intent":
            fake_item = {
                "sample_id": case.id,
                "input_text": str(case.inputs["query"]),
                "options": options_b77 or [],
                "target": str(case.target or "out_of_scope"),
            }
        else:
            fake_item = {
                "sample_id": case.id,
                "context": str(case.inputs["context"]),
                "question": str(case.inputs["question"]),
                "target": [],
            }
        inst = render_prompt_instance(task_id, fake_item, levels, env_id="ref")
        # Append explicit banking77 out_of_scope clause when E=1 (matching testgen_real_pilot)
        user_txt = inst.user_text
        if task_id == "banking77_intent" and int(levels.get("E", 0)) == 1:
            e_old = "Rely strictly on the provided input and never invent unsupported facts, labels, or assumptions."
            e_new = (
                "Rely strictly on the provided input and never invent unsupported facts, labels, or assumptions. "
                "If the inquiry is completely outside banking or does not match any allowed intent, reply FINAL_ANSWER: out_of_scope."
            )
            if e_old in user_txt:
                user_txt = user_txt.replace(e_old, e_new, 1)
            else:
                user_txt = f"{e_new}\n\n{user_txt}"
        msgs = [
            {"role": "system", "content": inst.system_text},
            {"role": "user", "content": user_txt},
        ]
        resp = client.generate(msgs)
        if resp.error:
            return 0.0
        ctx = {"checks": [chk.model_dump(mode="json") for chk in case.checks]}
        return float(metric.compute(resp.content, target=case.target, input_data=ctx))

    prism_with_e = dict(prism_levels)
    prism_with_e["E"] = 1

    dspy_scores: List[float] = [0.0] * len(edge_cases)
    prism_win_scores: List[float] = [0.0] * len(edge_cases)
    prism_guard_scores: List[float] = [0.0] * len(edge_cases)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        f_d = {pool.submit(_eval_dspy_case, c): i for i, c in enumerate(edge_cases)}
        f_p = {
            pool.submit(_eval_prism_case, c, prism_levels): i
            for i, c in enumerate(edge_cases)
        }
        f_g = {
            pool.submit(_eval_prism_case, c, prism_with_e): i
            for i, c in enumerate(edge_cases)
        }
        for fut, i in f_d.items():
            dspy_scores[i] = fut.result()
        for fut, i in f_p.items():
            prism_win_scores[i] = fut.result()
        for fut, i in f_g.items():
            prism_guard_scores[i] = fut.result()

    return {
        "n_edge_cases": len(edge_cases),
        "testset_sha256": ts.manifest.cases_sha256,
        "dspy_compiled_pass_rate": float(np.mean(dspy_scores)),
        "promptprism_v4_winner_pass_rate": float(np.mean(prism_win_scores)),
        "promptprism_v4_winner_E_level": int(prism_levels.get("E", 0)),
        "promptprism_coverage_gated_E1_pass_rate": float(np.mean(prism_guard_scores)),
    }


# ---------------------------------------------------------------------------
# End-to-End Single-Task Runner
# ---------------------------------------------------------------------------


def run_task_benchmark(
    *,
    task_id: str,
    out_root: Path,
    num_candidates: int = 10,
    num_trials: int = 20,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    max_workers: int = 10,
    seed: int = 42,
    run_track_b: bool = True,
    reuse_compiled: bool = True,
) -> Dict[str, Any]:
    task_dir = out_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required in environment / .env.")

    splits = load_real_v4_splits(task_id)
    screen_items = splits["screen"]
    val_items = splits["validation"]
    conf_items = splits["clean_confirmation"]
    prism_run_id, prism_levels, v4_ref = load_promptprism_v4_winner(task_id)

    student, base_instruction = build_dspy_student(task_id)
    compiled_path = task_dir / "dspy_compiled.json"
    compile_meta_path = task_dir / "dspy_compile_meta.json"

    # --- 1. Compile DSPy MIPROv2 on Real-v4 screen (120) + validation (60) ---
    if reuse_compiled and compiled_path.exists() and compile_meta_path.exists():
        print(
            f"[{task_id}][Track A] Reusing compiled DSPy program from {compiled_path}",
            flush=True,
        )
        compiled_mod = copy.deepcopy(student)
        compiled_mod.load(str(compiled_path))
        compile_meta = json.loads(compile_meta_path.read_text(encoding="utf-8"))
    else:
        print(
            f"[{task_id}][Track A] Compiling DSPy MIPROv2 "
            f"(candidates={num_candidates}, trials={num_trials}, "
            f"train={len(screen_items)}, val={len(val_items)})...",
            flush=True,
        )
        task_lm = dspy.LM(
            "gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.0,
            max_tokens=512,
            reasoning_effort="off",
            cache=False,
        )
        proposer_lm = dspy.LM(
            "gemini/gemini-2.5-pro",
            api_key=api_key,
            temperature=0.7,
            max_tokens=4096,
            reasoning_effort="low",
            cache=False,
        )
        trainset = [item_to_dspy_example(task_id, it) for it in screen_items]
        valset = [item_to_dspy_example(task_id, it) for it in val_items]

        t0 = time.perf_counter()
        u_task_before = snapshot_lm_usage(task_lm)
        u_prop_before = snapshot_lm_usage(proposer_lm)
        with dspy.context(lm=task_lm):
            teleprompter = dspy.MIPROv2(
                metric=make_dspy_metric(task_id),
                prompt_model=proposer_lm,
                task_model=task_lm,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                auto=None,
                num_candidates=num_candidates,
                num_threads=max_workers,
                seed=seed,
                verbose=False,
                track_stats=True,
            )
            compiled_mod = teleprompter.compile(
                student,
                trainset=trainset,
                valset=valset,
                num_trials=num_trials,
                minibatch=True,
                minibatch_size=35,
                minibatch_full_eval_steps=5,
            )
        elapsed_compile = time.perf_counter() - t0
        u_task = diff_usage(snapshot_lm_usage(task_lm), u_task_before)
        u_prop = diff_usage(snapshot_lm_usage(proposer_lm), u_prop_before)

        compiled_mod.save(str(compiled_path))
        _, c_pred = list(compiled_mod.named_predictors())[0]
        compile_meta = {
            "compile_wall_sec": round(elapsed_compile, 2),
            "task_lm_usage": u_task,
            "proposer_lm_usage": u_prop,
            "promptprism_v4_opt_budget_calls": 2640,
            "compiled_instruction": str(c_pred.signature.instructions or ""),
            "base_instruction": base_instruction,
            "num_compiled_demos": len(getattr(c_pred, "demos", []) or []),
        }
        compile_meta_path.write_text(
            json.dumps(compile_meta, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{task_id}][Track A] MIPROv2 compiled in {elapsed_compile:.1f}s "
            f"(task_calls={u_task['calls']}, proposer_calls={u_prop['calls']}, "
            f"demos={compile_meta['num_compiled_demos']})",
            flush=True,
        )

    # --- 2. Live Paired Evaluation on Untouched clean_confirmation (240 x 6 envs) ---
    sut_client = GeminiMultiTurnClient(
        model_id="gemini-2.5-flash",
        api_key=api_key,
        temperature=0.0,
        max_output_tokens=512,
        thinking_budget=0,
    )

    print(
        f"[{task_id}][Track A] Evaluating DSPy MIPROv2 live on 240 confirmation x 6 envs...",
        flush=True,
    )
    dspy_eval = evaluate_dspy_module_on_envs(
        task_id=task_id,
        module=compiled_mod,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        client=sut_client,
        max_workers=max_workers,
    )
    dspy_cert = compute_stability_cert(
        task_id=task_id,
        system_label="dspy_miprov2",
        eval_out=dspy_eval,
        items=conf_items,
    )

    print(
        f"[{task_id}][Track A] Evaluating PromptPrism v4 winner (run={prism_run_id}, "
        f"levels={prism_levels}) live on 240 confirmation x 6 envs...",
        flush=True,
    )
    prism_eval = evaluate_promptprism_on_envs(
        task_id=task_id,
        factor_levels=prism_levels,
        items=conf_items,
        env_ids=PAIRED_ENV_IDS,
        client=sut_client,
        max_workers=max_workers,
    )
    prism_cert = compute_stability_cert(
        task_id=task_id,
        system_label=f"promptprism_run{prism_run_id}",
        eval_out=prism_eval,
        items=conf_items,
    )

    group_ids = [str(it["group_id"]) for it in conf_items]
    head_to_head_clean = paired_item_comparison(
        dspy_eval["scores_by_env"]["ref"],
        prism_eval["scores_by_env"]["ref"],
        group_ids=group_ids,
    )

    # --- 3. Track B: Audit & Prune DSPy's Compiled Prompt with PromptPrism ---
    track_b_out: Optional[Dict[str, Any]] = None
    track_b_vs_dspy_full: Optional[Dict[str, Any]] = None
    if run_track_b:
        track_b_out = run_track_b_audit_and_prune(
            task_id=task_id,
            compiled_module=compiled_mod,
            base_instruction=base_instruction,
            screen_items=screen_items,
            confirmation_items=conf_items,
            client=sut_client,
            out_dir=task_dir,
            max_workers=max_workers,
        )
        if (
            track_b_out
            and not track_b_out.get("skipped")
            and track_b_out.get("pruned_eval") is not None
        ):
            track_b_vs_dspy_full = paired_item_comparison(
                track_b_out["pruned_eval"]["scores_by_env"]["ref"],
                dspy_eval["scores_by_env"]["ref"],
                group_ids=group_ids,
            )

    # --- 4. Bonus Edge-Case Guardrail Audit (when frozen synthetic test set exists) ---
    edge_case_out = evaluate_edge_case_guardrail(
        task_id=task_id,
        dspy_module=compiled_mod,
        prism_levels=prism_levels,
        client=sut_client,
        max_workers=max_workers,
    )

    # Strip raw per-item score arrays from compact JSON summary (save raw separately)
    raw_scores_payload = {
        "sample_ids": [str(it["sample_id"]) for it in conf_items],
        "group_ids": group_ids,
        "dspy_scores_by_env": dspy_eval["scores_by_env"],
        "promptprism_scores_by_env": prism_eval["scores_by_env"],
        "dspy_pruned_scores_by_env": (
            track_b_out["pruned_eval"]["scores_by_env"]
            if (track_b_out and track_b_out.get("pruned_eval"))
            else None
        ),
    }
    (task_dir / "confirmation_scores.json").write_text(
        json.dumps(raw_scores_payload, indent=2) + "\n", encoding="utf-8"
    )

    track_b_compact = None
    if track_b_out is not None:
        track_b_compact = {k: v for k, v in track_b_out.items() if k != "pruned_eval"}
        if track_b_out.get("pruned_eval") is not None:
            track_b_compact["pruned_mean_prompt_tokens_ref"] = track_b_out[
                "pruned_eval"
            ]["mean_prompt_tokens_ref"]
            track_b_compact["pruned_mean_completion_tokens_ref"] = track_b_out[
                "pruned_eval"
            ]["mean_completion_tokens_ref"]
            track_b_compact["pruned_mean_latency_ms_ref"] = track_b_out["pruned_eval"][
                "mean_latency_ms_ref"
            ]
            track_b_compact["vs_dspy_full_clean"] = track_b_vs_dspy_full

    summary = {
        "task_id": task_id,
        "sut_model_id": "gemini-2.5-flash",
        "proposer_model_id": "gemini-2.5-pro",
        "promptprism_v4_winner": {
            "run_id": prism_run_id,
            "factor_levels": prism_levels,
            "historical_v4_ref": v4_ref,
            "live_eval_ref_mean": prism_eval["env_means"]["ref"],
            "mean_prompt_tokens_ref": prism_eval["mean_prompt_tokens_ref"],
            "mean_completion_tokens_ref": prism_eval["mean_completion_tokens_ref"],
            "mean_latency_ms_ref": prism_eval["mean_latency_ms_ref"],
            "stability_cert": prism_cert,
        },
        "dspy_miprov2": {
            "compile_meta": compile_meta,
            "live_eval_ref_mean": dspy_eval["env_means"]["ref"],
            "mean_prompt_tokens_ref": dspy_eval["mean_prompt_tokens_ref"],
            "mean_completion_tokens_ref": dspy_eval["mean_completion_tokens_ref"],
            "mean_latency_ms_ref": dspy_eval["mean_latency_ms_ref"],
            "stability_cert": dspy_cert,
        },
        "track_a_head_to_head_clean": head_to_head_clean,
        "track_b_dspy_audit_and_prune": track_b_compact,
        "edge_case_guardrail_audit": edge_case_out,
        "live_eval_flash_client_usage": {
            "calls": sut_client.total_calls,
            "prompt_tokens": sut_client.total_input_tokens,
            "completion_tokens": sut_client.total_output_tokens,
        },
    }
    (task_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Run Track A + Track B PromptPrism vs. DSPy MIPROv2 benchmark."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["banking77_intent", "squad2_grounded_qa", "gsm8k_math"],
        choices=(
            "banking77_intent",
            "squad2_grounded_qa",
            "gsm8k_math",
            "re3d_entity_extraction",
            "sciq_verification",
        ),
    )
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-track-b", action="store_true")
    parser.add_argument("--no-reuse-compiled", action="store_true")
    parser.add_argument("--guardrail-only", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "research" / "results" / "dspy_benchmark"),
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.guardrail_only:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required in environment / .env.")
        client = GeminiMultiTurnClient(
            "gemini-2.5-flash",
            api_key=api_key,
            temperature=0.0,
            max_output_tokens=512,
            thinking_budget=0,
        )
        comb_path = out_root / "combined_summary.json"
        combined = (
            json.loads(comb_path.read_text(encoding="utf-8"))
            if comb_path.exists()
            else {}
        )
        for task_id in args.tasks:
            task_dir = out_root / task_id
            compiled_path = task_dir / "dspy_compiled.json"
            summary_path = task_dir / "benchmark_summary.json"
            student, _ = build_dspy_student(task_id)
            compiled_mod = copy.deepcopy(student)
            compiled_mod.load(str(compiled_path))
            _, prism_levels, _ = load_promptprism_v4_winner(task_id)
            edge_res = evaluate_edge_case_guardrail(
                task_id=task_id,
                dspy_module=compiled_mod,
                prism_levels=prism_levels,
                client=client,
                max_workers=args.max_workers,
            )
            print(
                f"[{task_id}] Refreshed guardrail audit: {json.dumps(edge_res)}",
                flush=True,
            )
            if summary_path.exists():
                s_obj = json.loads(summary_path.read_text(encoding="utf-8"))
                s_obj["edge_case_guardrail_audit"] = edge_res
                summary_path.write_text(
                    json.dumps(s_obj, indent=2) + "\n", encoding="utf-8"
                )
            if task_id in combined:
                combined[task_id]["edge_case_guardrail_audit"] = edge_res
        if combined:
            comb_path.write_text(
                json.dumps(combined, indent=2) + "\n", encoding="utf-8"
            )
        return

    all_summaries: Dict[str, Any] = {}
    for task_id in args.tasks:
        summary = run_task_benchmark(
            task_id=task_id,
            out_root=out_root,
            num_candidates=args.num_candidates,
            num_trials=args.num_trials,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            max_workers=args.max_workers,
            seed=args.seed,
            run_track_b=not args.skip_track_b,
            reuse_compiled=not args.no_reuse_compiled,
        )
        all_summaries[task_id] = summary
        print(
            f"\n=== SUMMARY [{task_id}] ===\n"
            + json.dumps(
                {
                    "track_a_clean": summary["track_a_head_to_head_clean"],
                    "prism_cert": summary["promptprism_v4_winner"]["stability_cert"],
                    "dspy_cert": summary["dspy_miprov2"]["stability_cert"],
                    "prism_tokens": summary["promptprism_v4_winner"][
                        "mean_prompt_tokens_ref"
                    ],
                    "dspy_tokens": summary["dspy_miprov2"]["mean_prompt_tokens_ref"],
                    "track_b_removed": (
                        summary["track_b_dspy_audit_and_prune"].get(
                            "removed_components"
                        )
                        if summary["track_b_dspy_audit_and_prune"]
                        else None
                    ),
                    "edge_case": summary["edge_case_guardrail_audit"],
                },
                indent=2,
            ),
            flush=True,
        )

    (out_root / "combined_summary.json").write_text(
        json.dumps(all_summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved combined benchmark summary to {out_root / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
