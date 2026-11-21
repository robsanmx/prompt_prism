"""
Per-Item Declarative CheckSuiteMetric for PromptPrism TestSets.

Evaluates deterministic checks (`exact`, `regex`, `contains`, `not_contains`, `json_schema`,
and opt-in `custom`) alongside optional `llm_judge` rubrics backed by either DeepEval
(`deepeval_metrics.py` + `JudgeCache`) or a lightweight `LLMClient` fallback for offline
`MockLLM` testing.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from ..runner.client import CallableLLM, LLMClient
from ..testgen.models import RESERVED_TEST_CASE_KEYS, CheckSpec
from .deepeval_metrics import DeepEvalMetric, deepeval_metric
from .judge_cache import JudgeCache
from .metrics import CustomMetric, ExactMatch, JSONValidation, Metric

_CLEAN_INPUT_KEY = "__check_suite_clean_input__"
_NON_INPUT_CONTEXT_KEYS = RESERVED_TEST_CASE_KEYS | {_CLEAN_INPUT_KEY}


def _extract_clean_input(input_data: Optional[Dict[str, Any]]) -> str:
    """Build an input-only string for LLM judges that NEVER reads `__prompt__`."""
    data = input_data or {}
    if "__test_inputs__" in data and data["__test_inputs__"] is not None:
        return str(data["__test_inputs__"])
    clean = {
        k: v
        for k, v in data.items()
        if k not in _NON_INPUT_CONTEXT_KEYS and not str(k).startswith("__")
    }
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)


def _parse_judge_client_response(raw_text: str) -> Tuple[float, str]:
    """Parse an LLMClient judge response into `(score_in_0_1, reason)`."""
    text = str(raw_text or "").strip()
    if not text:
        raise RuntimeError("Empty response from judge client.")

    # 1. Strip optional fenced JSON block
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = fenced.group(1).strip() if fenced else text

    parsed: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    parsed = obj
            except Exception:
                parsed = None

    if parsed is not None:
        reason = str(parsed.get("reason", "") or "")
        if "score" in parsed and parsed["score"] is not None:
            score = float(parsed["score"])
            if not math.isfinite(score):
                raise RuntimeError(f"Judge returned non-finite score: {score!r}")
            return max(0.0, min(1.0, score)), reason
        if "pass" in parsed and isinstance(parsed["pass"], bool):
            return (1.0 if parsed["pass"] else 0.0), reason
        if "passed" in parsed and isinstance(parsed["passed"], bool):
            return (1.0 if parsed["passed"] else 0.0), reason

    # 2. Fallback: numeric literal or PASS/FAIL token
    lowered = candidate.lower().strip()
    if lowered in {"pass", "passed", "yes", "true"}:
        return 1.0, ""
    if lowered in {"fail", "failed", "no", "false"}:
        return 0.0, ""
    try:
        val = float(lowered)
        if math.isfinite(val):
            return max(0.0, min(1.0, val)), ""
    except ValueError:
        pass

    raise RuntimeError(f"Unable to parse judge score from response: {text[:200]!r}")


def _resolve_custom_callable(ref: str) -> Callable[..., Any]:
    """Import and return callable referenced as `'module.path:attr'`."""
    if ":" not in ref:
        raise ValueError(
            f"Invalid custom check reference {ref!r}; expected 'module:attr'."
        )
    mod_name, attr_path = ref.split(":", 1)
    module = importlib.import_module(mod_name.strip())
    obj: Any = module
    for part in attr_path.strip().split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"Custom check target {ref!r} is not callable.")
    return obj


def _coerce_checks(raw_checks: Any) -> List[CheckSpec]:
    """Normalize `input_data['checks']` into a list of validated `CheckSpec` objects."""
    if raw_checks is None:
        return []
    if isinstance(raw_checks, str):
        s = raw_checks.strip()
        if not s:
            return []
        raw_checks = json.loads(s)
    if not isinstance(raw_checks, Sequence):
        raise TypeError(f"Expected sequence of checks, got {type(raw_checks).__name__}")
    out: List[CheckSpec] = []
    for item in raw_checks:
        if isinstance(item, CheckSpec):
            out.append(item)
        elif isinstance(item, dict):
            out.append(CheckSpec.model_validate(item))
        else:
            raise TypeError(
                f"Each check must be a CheckSpec or dict, got {type(item).__name__}"
            )
    return out


class CheckSuiteMetric(Metric):
    """Scores each item against its declarative `input_data['checks']` list.

    Returns the weighted pass fraction in ``[0.0, 1.0]``.
    - Deterministic checks (`exact`, `regex`, `contains`, `not_contains`, `json_schema`)
      execute locally at zero cost.
    - `custom` checks (`"module:function"`) are disabled by default
      (`allow_custom_checks=False`) to prevent arbitrary code execution from untrusted
      or LLM-generated test files.
    - `llm_judge` checks delegate either to `judge_client` (any `LLMClient` or callable,
      enabling offline `MockLLM` testing without the `deepeval` extra) or to PromptPrism's
      DeepEval `GEval` adapter (`deepeval_metrics.py`), always using `JudgeCache` when
      supplied and isolating judge input via `__test_inputs__` so toggled prompt clauses
      never leak into judge prompts or bust the cache.
    """

    name: str = "check_suite"
    higher_is_better: bool = True
    is_llm_judge: bool = False  # Dynamic per-item; counted via estimate_judge_calls()

    def __init__(
        self,
        *,
        name: str = "check_suite",
        judge_client: Optional[Union[LLMClient, Callable[..., Any]]] = None,
        judge_model: Optional[Any] = None,
        judge_model_id: Optional[str] = None,
        judge_cache: Optional[JudgeCache] = None,
        judge_threshold: float = 0.5,
        judge_mode: str = "score",
        allow_custom_checks: bool = False,
        case_sensitive: bool = False,
    ):
        if judge_mode.lower() not in ("score", "pass"):
            raise ValueError(
                f"judge_mode must be 'score' or 'pass', got {judge_mode!r}"
            )
        self.name = name
        self.judge_client: Optional[LLMClient] = (
            None
            if judge_client is None
            else (
                judge_client
                if isinstance(judge_client, LLMClient)
                else CallableLLM(judge_client)
            )
        )
        self.judge_model = judge_model
        if judge_model_id is not None:
            self.judge_model_id = str(judge_model_id)
        elif self.judge_client is not None and self.judge_client.model_id:
            self.judge_model_id = str(self.judge_client.model_id)
        elif isinstance(judge_model, str):
            self.judge_model_id = judge_model
        else:
            self.judge_model_id = "default"
        self.judge_cache = judge_cache
        self.judge_threshold = float(judge_threshold)
        self.judge_mode = judge_mode.lower()
        self.allow_custom_checks = bool(allow_custom_checks)
        self.case_sensitive = bool(case_sensitive)

        self._exact_metric = ExactMatch(case_sensitive=self.case_sensitive, strip=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._regex_cache: Dict[str, re.Pattern[str]] = {}
        self._schema_cache: Dict[str, JSONValidation] = {}
        self._custom_cache: Dict[
            str, Callable[[Any, Any, Optional[Dict[str, Any]]], float]
        ] = {}
        self._deepeval_cache: Dict[str, DeepEvalMetric] = {}

    def pop_reason(self) -> Optional[str]:
        reason = getattr(self._local, "reason", None)
        self._local.reason = None
        return reason

    def _get_regex(self, pattern: str) -> re.Pattern[str]:
        with self._lock:
            compiled = self._regex_cache.get(pattern)
            if compiled is None:
                compiled = re.compile(pattern)
                self._regex_cache[pattern] = compiled
            return compiled

    def _get_schema_validator(self, schema: Dict[str, Any]) -> JSONValidation:
        key = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        with self._lock:
            validator = self._schema_cache.get(key)
            if validator is None:
                validator = JSONValidation(schema=schema)
                self._schema_cache[key] = validator
            return validator

    def _get_custom_caller(
        self, ref: str
    ) -> Callable[[Any, Any, Optional[Dict[str, Any]]], float]:
        with self._lock:
            caller = self._custom_cache.get(ref)
            if caller is None:
                fn = _resolve_custom_callable(ref)
                caller = CustomMetric._compile_caller(fn)
                self._custom_cache[ref] = caller
            return caller

    def _get_deepeval_judge(self, rubric: str) -> DeepEvalMetric:
        with self._lock:
            metric = self._deepeval_cache.get(rubric)
            if metric is None:
                try:
                    try:
                        from deepeval.test_case import SingleTurnParams as _EvalParams
                    except ImportError:
                        from deepeval.test_case import LLMTestCaseParams as _EvalParams
                except ImportError as exc:
                    raise ImportError(
                        "LLM judge checks without a `judge_client` require the 'deepeval' extra.\n"
                        "Install it with: pip install prompt-prism[deepeval], or pass `judge_client=...`."
                    ) from exc

                metric = deepeval_metric(
                    "g_eval",
                    name=f"{self.name}_geval",
                    criteria=rubric,
                    evaluation_params=[
                        _EvalParams.INPUT,
                        _EvalParams.ACTUAL_OUTPUT,
                    ],
                    model=self.judge_model,
                    judge_model_id=self.judge_model_id,
                    input_key=_CLEAN_INPUT_KEY,
                    mode=self.judge_mode,
                    threshold=self.judge_threshold,
                    cache=self.judge_cache,
                )
                self._deepeval_cache[rubric] = metric
            return metric

    def _evaluate_llm_judge(
        self,
        rubric: str,
        prediction: Any,
        target: Any,
        input_data: Optional[Dict[str, Any]],
    ) -> Tuple[float, Optional[str]]:
        clean_input = _extract_clean_input(input_data)
        actual_out = str(prediction or "")
        expected_out = str(target) if target is not None else ""

        if self.judge_client is not None:
            config_str = json.dumps(
                {
                    "rubric": rubric,
                    "mode": self.judge_mode,
                    "threshold": self.judge_threshold,
                },
                sort_keys=True,
            )
            if self.judge_cache is not None:
                cached = self.judge_cache.get(
                    metric_name=f"{self.name}:llm_judge",
                    metric_config=config_str,
                    judge_model_id=self.judge_model_id,
                    input_text=clean_input,
                    actual_output=actual_out,
                    expected_output=expected_out,
                )
                if cached is not None:
                    score = (
                        float(cached.score)
                        if self.judge_mode == "score"
                        else (1.0 if cached.success else 0.0)
                    )
                    return score, cached.reason or None

            prompt_parts = [
                "Evaluate the assistant's response against the rubric below.",
                'Return ONLY a JSON object: {"score": <float between 0.0 and 1.0>, "reason": "<brief explanation>"}.',
                f"\nRubric:\n{rubric}",
                f"\nInput:\n{clean_input}",
                f"\nActual Output:\n{actual_out}",
            ]
            if expected_out:
                prompt_parts.append(f"\nReference Output:\n{expected_out}")
            judge_prompt = "\n".join(prompt_parts)

            resp = self.judge_client.generate(judge_prompt)
            if resp.error:
                raise RuntimeError(f"Judge client error: {resp.error}")
            raw_score, reason = _parse_judge_client_response(resp.content)
            is_success = raw_score >= self.judge_threshold

            if self.judge_cache is not None:
                self.judge_cache.set(
                    metric_name=f"{self.name}:llm_judge",
                    metric_config=config_str,
                    judge_model_id=self.judge_model_id,
                    input_text=clean_input,
                    actual_output=actual_out,
                    expected_output=expected_out,
                    score=raw_score,
                    reason=reason,
                    success=is_success,
                )
            score = (
                raw_score
                if self.judge_mode == "score"
                else (1.0 if is_success else 0.0)
            )
            return score, reason or None

        # DeepEval adapter path: strip `__prompt__` and inject isolated clean input key
        de_metric = self._get_deepeval_judge(rubric)
        isolated_data = {
            k: v for k, v in (input_data or {}).items() if k != "__prompt__"
        }
        isolated_data[_CLEAN_INPUT_KEY] = clean_input
        score = de_metric.compute(
            prediction=prediction, target=target, input_data=isolated_data
        )
        return float(score), de_metric.pop_reason()

    def _evaluate_single_check(
        self,
        check: CheckSpec,
        prediction: Any,
        target: Any,
        input_data: Optional[Dict[str, Any]],
    ) -> Tuple[float, Optional[str]]:
        pred_str = str(prediction or "")

        if check.kind == "exact":
            expected = check.value if check.value is not None else target
            return self._exact_metric.compute(prediction, expected), None

        if check.kind == "regex":
            pat = self._get_regex(str(check.value))
            return (1.0 if pat.search(pred_str) else 0.0), None

        if check.kind == "contains":
            needle = str(check.value)
            haystack = pred_str if self.case_sensitive else pred_str.lower()
            needle_cmp = needle if self.case_sensitive else needle.lower()
            return (1.0 if needle_cmp in haystack else 0.0), None

        if check.kind == "not_contains":
            needle = str(check.value)
            haystack = pred_str if self.case_sensitive else pred_str.lower()
            needle_cmp = needle if self.case_sensitive else needle.lower()
            return (0.0 if needle_cmp in haystack else 1.0), None

        if check.kind == "json_schema":
            validator = self._get_schema_validator(check.value)
            return validator.compute(prediction), None

        if check.kind == "custom":
            if not self.allow_custom_checks:
                raise PermissionError(
                    f"Custom check {check.value!r} refused: pass allow_custom_checks=True to opt in."
                )
            caller = self._get_custom_caller(str(check.value))
            val = float(caller(prediction, target, input_data))
            if math.isnan(val):
                return float("nan"), None
            return max(0.0, min(1.0, val)), None

        if check.kind == "llm_judge":
            return self._evaluate_llm_judge(
                str(check.value), prediction, target, input_data
            )

        raise ValueError(f"Unsupported check kind: {check.kind!r}")

    def compute(
        self,
        prediction: Any,
        target: Any = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute weighted pass fraction over `input_data['checks']` in `[0.0, 1.0]`."""
        self._local.reason = None
        checks = _coerce_checks((input_data or {}).get("checks"))

        if not checks:
            if target is not None:
                return self._exact_metric.compute(prediction, target)
            raise ValueError(
                "CheckSuiteMetric requires non-empty 'checks' in input_data or a 'target' value."
            )

        total_weight = 0.0
        weighted_sum = 0.0
        reasons: List[str] = []

        for chk in checks:
            if chk.kind == "llm_judge":
                try:
                    score, reason = self._evaluate_single_check(
                        chk, prediction, target, input_data
                    )
                except ImportError:
                    raise
                except Exception as exc:
                    reasons.append(f"[llm_judge error] {exc}")
                    self._local.reason = "; ".join(reasons)
                    return float("nan")
            else:
                # Let PermissionError propagate for security; other deterministic errors fail check
                try:
                    score, reason = self._evaluate_single_check(
                        chk, prediction, target, input_data
                    )
                except PermissionError:
                    raise
                except Exception:
                    score, reason = 0.0, None

            if reason:
                reasons.append(reason)
            if math.isnan(score):
                self._local.reason = "; ".join(reasons) if reasons else None
                return float("nan")

            weighted_sum += chk.weight * score
            total_weight += chk.weight

        if reasons:
            self._local.reason = "; ".join(reasons)
        return float(weighted_sum / total_weight) if total_weight > 0.0 else 0.0

    def estimate_judge_calls(self, dataset: Any) -> int:
        """Count exact number of `llm_judge` checks across all items for a single run."""
        # Accept TestSet, DataFrame, or Sequence[Dict]
        if hasattr(dataset, "to_dataset") and callable(dataset.to_dataset):
            items: Sequence[Dict[str, Any]] = dataset.to_dataset()
        elif isinstance(dataset, pd.DataFrame):
            items = dataset.to_dict(orient="records")
        else:
            items = list(dataset)

        total_judge_checks = 0
        for item in items:
            raw_checks = item.get("checks") if isinstance(item, dict) else None
            if not raw_checks:
                continue
            if isinstance(raw_checks, str):
                try:
                    raw_checks = json.loads(raw_checks)
                except Exception:
                    continue
            for chk in raw_checks:
                kind = (
                    chk.kind
                    if isinstance(chk, CheckSpec)
                    else (chk.get("kind") if isinstance(chk, dict) else None)
                )
                if kind == "llm_judge":
                    total_judge_checks += 1
        return total_judge_checks
