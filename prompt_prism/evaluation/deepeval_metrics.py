"""
DeepEval Metric Adapter: Enables LLM-as-a-Judge evaluation metrics as first-class PromptPrism Metrics.
"""

from __future__ import annotations

import enum
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from .judge_cache import JudgeCache
from .metrics import Metric

# Score directions below encode deepeval >= 4.2.0, which unified every metric on
# "1 is a pass, 0 is a failure". Through 4.1.10, hallucination/toxicity/bias scored the
# proportion of violations instead (lower was better, and `threshold` was a maximum);
# 4.2.0 reversed them to score the *absence* of the problem. The extra is pinned
# >=4.2.0,<5 in pyproject.toml so these constants cannot be read against the old
# semantics - getting this wrong makes the optimizer recommend the more toxic prompt.
DEEPEVAL_METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "answer_relevancy": {
        "class_name": "AnswerRelevancyMetric",
        "higher_is_better": True,
        "default_name": "deepeval_answer_relevancy",
    },
    "faithfulness": {
        "class_name": "FaithfulnessMetric",
        "higher_is_better": True,
        "default_name": "deepeval_faithfulness",
    },
    "hallucination": {
        "class_name": "HallucinationMetric",
        # deepeval >= 4.2.0: scores the absence of the problem, so higher is better.
        "higher_is_better": True,
        "default_name": "deepeval_hallucination",
    },
    "toxicity": {
        "class_name": "ToxicityMetric",
        # deepeval >= 4.2.0: scores the absence of the problem, so higher is better.
        "higher_is_better": True,
        "default_name": "deepeval_toxicity",
    },
    "bias": {
        "class_name": "BiasMetric",
        # deepeval >= 4.2.0: scores the absence of the problem, so higher is better.
        "higher_is_better": True,
        "default_name": "deepeval_bias",
    },
    "contextual_precision": {
        "class_name": "ContextualPrecisionMetric",
        "higher_is_better": True,
        "default_name": "deepeval_contextual_precision",
    },
    "contextual_recall": {
        "class_name": "ContextualRecallMetric",
        "higher_is_better": True,
        "default_name": "deepeval_contextual_recall",
    },
    "contextual_relevancy": {
        "class_name": "ContextualRelevancyMetric",
        "higher_is_better": True,
        "default_name": "deepeval_contextual_relevancy",
    },
    "summarization": {
        "class_name": "SummarizationMetric",
        "higher_is_better": True,
        "default_name": "deepeval_summarization",
    },
    "json_correctness": {
        "class_name": "JsonCorrectnessMetric",
        "higher_is_better": True,
        "default_name": "deepeval_json_correctness",
    },
    "g_eval": {
        "class_name": "GEval",
        "higher_is_better": True,
        "default_name": "deepeval_g_eval",
    },
}


INPUT_FALLBACK_CANDIDATES: Sequence[str] = ("input", "question", "query")
EXPECTED_OUTPUT_FALLBACK_CANDIDATES: Sequence[str] = (
    "target",
    "expected_output",
    "ground_truth",
    "gold_answer",
)
CONTEXT_FALLBACK_CANDIDATES: Sequence[str] = ("context", "documents")
RETRIEVAL_CONTEXT_FALLBACK_CANDIDATES: Sequence[str] = (
    "retrieval_context",
    "context",
    "documents",
)


def _first_present(
    data: Dict[str, Any],
    explicit_key: Optional[str],
    candidates: Sequence[str],
) -> Optional[Any]:
    if explicit_key:
        if explicit_key in data and data[explicit_key] is not None:
            return data[explicit_key]
        return None
    for k in candidates:
        if k in data and data[k] is not None:
            return data[k]
    return None


def _stable_config_value(val: Any) -> Any:
    """Reduce a config value to a deterministic, JSON-serializable form.

    This feeds the judge cache key, so it must be stable *across processes*: a default
    ``repr`` carrying a memory address would make every run miss the cache, and dropping a
    value entirely would make two differently-configured metrics collide. Classes reduce to
    their qualified name plus field signature (so a changed pydantic schema changes the key);
    instances reduce to their class plus any declared model name.
    """
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, enum.Enum):
        return f"{type(val).__qualname__}.{val.name}"
    if isinstance(val, (list, tuple)):
        return [_stable_config_value(v) for v in val]
    if isinstance(val, (set, frozenset)):
        return sorted(str(_stable_config_value(v)) for v in val)
    if isinstance(val, dict):
        return {
            str(k): _stable_config_value(v)
            for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(val, type):
        label = f"{val.__module__}.{val.__qualname__}"
        fields = getattr(val, "model_fields", None)
        if isinstance(fields, dict):
            sig = ",".join(
                f"{k}:{getattr(f, 'annotation', None)}"
                for k, f in sorted(fields.items())
            )
            label = f"{label}({sig})"
        return label
    cls = type(val)
    label = f"{cls.__module__}.{cls.__qualname__}"
    for attr in ("model_name", "name"):
        named = getattr(val, attr, None)
        if isinstance(named, str):
            return f"{label}({attr}={named})"
    return label


def _as_list(val: Optional[Any]) -> Optional[List[str]]:
    if val is None:
        return None
    if isinstance(val, list):
        return [str(item) for item in val]
    return [str(val)]


class DeepEvalMetric(Metric):
    """
    Adapter metric wrapping any deepeval LLM-as-a-judge metric.

    Maps trial prediction, target, and input_data to an LLMTestCase, executes
    the judge evaluation, and returns the continuous score.

    Args:
        metric_factory: Callable returning an initialized deepeval BaseMetric instance.
        name: Metric identifier for dataframes and reports.
        input_key: Field name in input_data to use for LLMTestCase.input.
        context_key: Field name in input_data to use for LLMTestCase.context.
        retrieval_context_key: Field name in input_data to use for LLMTestCase.retrieval_context.
        expected_output_key: Field name in input_data to use for LLMTestCase.expected_output.
        mode: "score" (continuous [0.0, 1.0]) or "pass" (binary 0.0/1.0 based on threshold).
        cache: Optional JudgeCache instance to eliminate redundant judge LLM calls.
        judge_model_id: Identifier of the backing judge model for caching.
        higher_is_better: Whether higher score represents better performance.
        capture_reason: Whether to capture judge explanations into trial metadata.
    """

    is_llm_judge: bool = True

    def __init__(
        self,
        metric_factory: Callable[[], Any],
        name: str = "deepeval_metric",
        input_key: Optional[str] = None,
        context_key: Optional[str] = None,
        retrieval_context_key: Optional[str] = None,
        expected_output_key: Optional[str] = None,
        mode: str = "score",
        cache: Optional[JudgeCache] = None,
        judge_model_id: str = "default",
        higher_is_better: bool = True,
        capture_reason: bool = True,
        config_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.metric_factory = metric_factory
        self.name = name
        self.input_key = input_key
        self.context_key = context_key
        self.retrieval_context_key = retrieval_context_key
        self.expected_output_key = expected_output_key
        self.mode = mode.lower()
        self.cache = cache
        self.judge_model_id = judge_model_id
        self.higher_is_better = higher_is_better
        self.capture_reason = capture_reason
        self.config_metadata = config_metadata or {}
        # Judge explanations travel on a thread-local, collected by Evaluator.last_reasons.
        # The runner executes trials in a thread pool, so a plain attribute would race.
        self._local = threading.local()

    def pop_reason(self) -> Optional[str]:
        reason = getattr(self._local, "reason", None)
        self._local.reason = None
        return reason

    def _record_reason(self, reason: str) -> None:
        if self.capture_reason:
            self._local.reason = reason

    def _resolve_fields(
        self,
        prediction: Any,
        target: Any,
        input_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        data = input_data or {}

        # 1. Input resolution
        raw_inp = _first_present(data, self.input_key, INPUT_FALLBACK_CANDIDATES)
        inp = str(raw_inp) if raw_inp is not None else str(data.get("__prompt__") or "")

        # 2. Actual output
        act_out = str(prediction or "")

        # 3. Expected output resolution
        if target is not None:
            exp_out = str(target)
        else:
            raw_exp = _first_present(
                data,
                self.expected_output_key,
                EXPECTED_OUTPUT_FALLBACK_CANDIDATES,
            )
            exp_out = str(raw_exp) if raw_exp is not None else None

        # 4. Context resolution
        raw_ctx = _first_present(data, self.context_key, CONTEXT_FALLBACK_CANDIDATES)
        ctx = _as_list(raw_ctx)

        # 5. Retrieval context resolution
        raw_ret_ctx = _first_present(
            data,
            self.retrieval_context_key,
            RETRIEVAL_CONTEXT_FALLBACK_CANDIDATES,
        )
        ret_ctx = _as_list(raw_ret_ctx)

        return {
            "input": inp,
            "actual_output": act_out,
            "expected_output": exp_out,
            "context": ctx,
            "retrieval_context": ret_ctx,
        }

    def _build_test_case(
        self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]]
    ) -> Any:
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as e:
            raise ImportError(
                "DeepEval metrics require the 'deepeval' extra.\n"
                "Install it using: pip install prompt-prism[deepeval]"
            ) from e

        fields = self._resolve_fields(prediction, target, input_data)
        return LLMTestCase(
            input=fields["input"],
            actual_output=fields["actual_output"],
            expected_output=fields["expected_output"],
            context=fields["context"],
            retrieval_context=fields["retrieval_context"],
        )

    def _get_config_str(self) -> str:
        cfg = _stable_config_value({"mode": self.mode, **self.config_metadata})
        return json.dumps(cfg, sort_keys=True)

    def compute(
        self,
        prediction: Any,
        target: Any = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Execute the judge evaluation, checking cache and returning score."""
        fields = self._resolve_fields(prediction, target, input_data)
        config_str = self._get_config_str()

        # Check judge cache
        if self.cache:
            cached_val = self.cache.get(
                metric_name=self.name,
                metric_config=config_str,
                judge_model_id=self.judge_model_id,
                input_text=fields["input"],
                actual_output=fields["actual_output"],
                expected_output=fields["expected_output"] or "",
                context=fields["context"],
                retrieval_context=fields["retrieval_context"],
            )
            if cached_val is not None:
                self._record_reason(cached_val.reason)
                return float(
                    cached_val.score
                    if self.mode == "score"
                    else (1.0 if cached_val.success else 0.0)
                )

        # Instantiate metric instance
        try:
            metric_inst = self.metric_factory()
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate DeepEval metric '{self.name}': {e}"
            ) from e

        test_case = self._build_test_case(prediction, target, input_data)

        # Execute evaluation measurement
        metric_inst.measure(test_case)

        score = float(metric_inst.score)
        reason = str(getattr(metric_inst, "reason", "") or "")
        if hasattr(metric_inst, "is_successful"):
            is_success = bool(metric_inst.is_successful())
        else:
            is_success = bool(getattr(metric_inst, "success", True))

        # Save to judge cache
        if self.cache:
            self.cache.set(
                metric_name=self.name,
                metric_config=config_str,
                judge_model_id=self.judge_model_id,
                input_text=fields["input"],
                actual_output=fields["actual_output"],
                expected_output=fields["expected_output"] or "",
                context=fields["context"],
                retrieval_context=fields["retrieval_context"],
                score=score,
                reason=reason,
                success=is_success,
            )

        self._record_reason(reason)

        if self.mode == "pass":
            return 1.0 if is_success else 0.0
        return score


def deepeval_metric(
    kind: str,
    name: Optional[str] = None,
    threshold: float = 0.5,
    criteria: Optional[str] = None,
    evaluation_steps: Optional[Sequence[str]] = None,
    evaluation_params: Optional[Sequence[Any]] = None,
    model: Optional[Any] = None,
    judge_model_id: Optional[str] = None,
    input_key: Optional[str] = None,
    context_key: Optional[str] = None,
    retrieval_context_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    mode: str = "score",
    cache: Optional[JudgeCache] = None,
    higher_is_better: Optional[bool] = None,
    capture_reason: bool = True,
    **kwargs,
) -> DeepEvalMetric:
    """
    Factory function constructing a configured DeepEvalMetric instance.

    Every supported kind scores in the same direction under deepeval >= 4.2.0: higher is
    better, including hallucination, toxicity and bias, which score the *absence* of the
    problem. See DEEPEVAL_METRIC_SPECS for the per-kind direction and `higher_is_better`
    to override it.

    Supported kinds:
        - "answer_relevancy": AnswerRelevancyMetric
        - "faithfulness": FaithfulnessMetric
        - "contextual_precision": ContextualPrecisionMetric
        - "contextual_recall": ContextualRecallMetric
        - "contextual_relevancy": ContextualRelevancyMetric
        - "hallucination": HallucinationMetric
        - "bias": BiasMetric
        - "toxicity": ToxicityMetric
        - "summarization": SummarizationMetric
        - "g_eval": GEval (custom criteria / rubric)
        - "json" / "json_correctness": JsonCorrectnessMetric

    Args:
        kind: Preset metric identifier.
        name: Custom metric name. Defaults to kind.
        threshold: Minimum passing score (default 0.5). Under the supported deepeval
            (>=4.2.0) this is a minimum for every kind, including hallucination,
            toxicity and bias - older deepeval treated it as a maximum for those three.
        criteria: Evaluation criteria string (for g_eval / summarization).
        evaluation_steps: Step-by-step scoring criteria (for g_eval).
        evaluation_params: Parameter names for scoring (for g_eval).
        model: Backing LLM judge instance (or string identifier).
        judge_model_id: Identifier of judge model for cache partitioning.
        input_key: Override key in dataset row for test case input.
        context_key: Override key in dataset row for test case context.
        retrieval_context_key: Override key in dataset row for retrieval context.
        expected_output_key: Override key in dataset row for golden expected output.
        mode: "score" (default) or "pass".
        cache: JudgeCache instance for caching evaluations.
        higher_is_better: Direction override (defaults to kind's standard direction).
        capture_reason: Whether to capture judge explanations into trial metadata.
        **kwargs: Additional parameters passed directly to the deepeval constructor.
    """
    kind_lower = kind.lower().strip()
    if kind_lower == "json":
        kind_lower = "json_correctness"

    if kind_lower not in DEEPEVAL_METRIC_SPECS:
        raise ValueError(
            f"Unknown DeepEval metric kind '{kind}'. Supported kinds: {list(DEEPEVAL_METRIC_SPECS.keys())}"
        )

    if kind_lower == "json_correctness":
        if "expected_schema" not in kwargs and "schema" not in kwargs:
            raise ValueError(
                "deepeval_metric('json_correctness') requires an explicit 'expected_schema' (or 'schema') pydantic model."
            )

    spec = DEEPEVAL_METRIC_SPECS[kind_lower]
    metric_name = name or spec["default_name"]
    metric_direction = (
        higher_is_better if higher_is_better is not None else spec["higher_is_better"]
    )

    if judge_model_id is not None:
        if not isinstance(judge_model_id, str):
            raise TypeError(
                f"judge_model_id must be a string identifier, got {type(judge_model_id).__name__}"
            )
        model_str = judge_model_id
    elif model is not None:
        if isinstance(model, str):
            model_str = model
        else:
            model_str = (
                getattr(model, "model_name", None)
                or getattr(model, "name", None)
                or "custom_model"
            )
    else:
        model_str = "default"

    # Everything that changes what the judge scores must reach the cache key, or two
    # differently-configured metrics sharing a name collide in a persistent JudgeCache.
    # That includes **kwargs, which carries expected_schema for json_correctness.
    config_metadata: Dict[str, Any] = {
        "threshold": threshold,
        "criteria": criteria,
        "evaluation_steps": evaluation_steps,
        "evaluation_params": evaluation_params,
        "model": model,
        **kwargs,
    }

    def factory():
        try:
            import deepeval.metrics as dm
        except ImportError as e:
            raise ImportError(
                "DeepEval metrics require the 'deepeval' extra.\n"
                "Install it using: pip install prompt-prism[deepeval]"
            ) from e

        cls_name = spec["class_name"]
        metric_cls = getattr(dm, cls_name, None)
        if metric_cls is None and cls_name == "JsonCorrectnessMetric":
            metric_cls = getattr(dm, "JSONCorrectnessMetric", None)
        if metric_cls is None:
            raise AttributeError(f"deepeval.metrics has no class '{cls_name}'")

        m_kwargs = {
            "threshold": threshold,
            "async_mode": False,
            "strict_mode": False,
            **kwargs,
        }
        if model is not None:
            m_kwargs["model"] = model

        if kind_lower == "g_eval":
            if criteria:
                m_kwargs["criteria"] = criteria
            if evaluation_steps:
                m_kwargs["evaluation_steps"] = evaluation_steps
            if evaluation_params:
                m_kwargs["evaluation_params"] = evaluation_params
            return metric_cls(name=metric_name, **m_kwargs)

        if kind_lower == "json_correctness":
            if "expected_schema" not in m_kwargs and "schema" in m_kwargs:
                m_kwargs["expected_schema"] = m_kwargs.pop("schema")

        return metric_cls(**m_kwargs)

    return DeepEvalMetric(
        metric_factory=factory,
        name=metric_name,
        input_key=input_key,
        context_key=context_key,
        retrieval_context_key=retrieval_context_key,
        expected_output_key=expected_output_key,
        mode=mode,
        cache=cache,
        judge_model_id=model_str,
        higher_is_better=metric_direction,
        capture_reason=capture_reason,
        config_metadata=config_metadata,
    )
