"""
DeepEval Metric Adapter: Enables LLM-as-a-Judge evaluation metrics as first-class PromptPrism Metrics.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np

from .judge_cache import JudgeCache
from .metrics import Metric


class DeepEvalMetric(Metric):
    """
    Wraps any DeepEval LLM judge metric as a PromptPrism Metric.
    
    Attributes:
        metric_factory: A zero-argument callable returning a fresh deepeval metric instance per evaluation (thread-safe).
        name: Metric identifier.
        input_key: Key in input_data to map to LLMTestCase.input (defaults to 'input' or '__prompt__').
        actual_output_key: Key in input_data to override prediction if needed.
        expected_output_key: Key in input_data or target to map to LLMTestCase.expected_output.
        context_key: Key in input_data for context list.
        retrieval_context_key: Key in input_data for retrieval_context list.
        mode: "score" (continuous [0.0, 1.0]) or "pass" (binary 0.0/1.0 based on threshold).
        cache: Optional JudgeCache instance to eliminate redundant judge LLM calls.
        judge_model_id: Identifier of the backing judge model for caching.
        higher_is_better: Whether higher score represents better performance.
    """
    is_llm_judge: bool = True
    wants_prompt: bool = True

    def __init__(
        self,
        metric_factory: Callable[[], Any],
        name: str = "deepeval_metric",
        input_key: str = "input",
        context_key: Optional[str] = None,
        retrieval_context_key: Optional[str] = None,
        mode: str = "score",
        cache: Optional[JudgeCache] = None,
        judge_model_id: str = "default",
        higher_is_better: bool = True,
    ):
        self.metric_factory = metric_factory
        self.name = name
        self.input_key = input_key
        self.context_key = context_key
        self.retrieval_context_key = retrieval_context_key
        self.mode = mode.lower()
        self.cache = cache
        self.judge_model_id = judge_model_id
        self.higher_is_better = higher_is_better

    def _build_test_case(self, prediction: Any, target: Any, input_data: Optional[Dict[str, Any]]) -> Any:
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError:
            from collections import namedtuple
            LLMTestCase = namedtuple("LLMTestCase", ["input", "actual_output", "expected_output", "context", "retrieval_context"])

        data = input_data or {}
        # Resolve input query or prompt
        inp = str(data.get(self.input_key) or data.get("__prompt__") or "")
        act_out = str(prediction or "")
        exp_out = str(target or data.get("target") or data.get("expected_output") or "")

        ctx = None
        if self.context_key and self.context_key in data:
            c_val = data[self.context_key]
            ctx = c_val if isinstance(c_val, list) else [str(c_val)]

        ret_ctx = None
        if self.retrieval_context_key and self.retrieval_context_key in data:
            rc_val = data[self.retrieval_context_key]
            ret_ctx = rc_val if isinstance(rc_val, list) else [str(rc_val)]

        return LLMTestCase(
            input=inp,
            actual_output=act_out,
            expected_output=exp_out if exp_out else None,
            context=ctx,
            retrieval_context=ret_ctx,
        )

    def compute(self, prediction: Any, target: Any = None, input_data: Optional[Dict[str, Any]] = None) -> float:
        """Execute the judge evaluation, checking cache and returning score or NaN on failure."""
        data = input_data or {}
        inp_str = str(data.get(self.input_key) or data.get("__prompt__") or "")
        act_str = str(prediction or "")
        exp_str = str(target or data.get("target") or "")

        # Check judge cache
        if self.cache:
            cached_val = self.cache.get(
                metric_name=self.name,
                metric_config=str(self.mode),
                judge_model_id=self.judge_model_id,
                input_text=inp_str,
                actual_output=act_str,
                expected_output=exp_str,
            )
            if cached_val is not None:
                return float(cached_val.score if self.mode == "score" else (1.0 if cached_val.success else 0.0))

        # Instantiate fresh metric instance for thread safety
        try:
            metric_inst = self.metric_factory()
        except ImportError as e:
            raise ImportError(
                "DeepEval metrics require the 'deepeval' extra.\n"
                "Install it using: pip install prompt-prism[deepeval]"
            ) from e
        except Exception:
            return float("nan")

        test_case = self._build_test_case(prediction, target, input_data)

        # Run measurement
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(metric_inst.measure, test_case).result()
            else:
                metric_inst.measure(test_case)

            score_val = float(getattr(metric_inst, "score", float("nan")))
            is_success = bool(getattr(metric_inst, "is_successful", lambda: score_val >= getattr(metric_inst, "threshold", 0.5))())
            reason = getattr(metric_inst, "reason", "")

            # Cache the result
            if self.cache and not np.isnan(score_val):
                self.cache.set(
                    metric_name=self.name,
                    metric_config=str(self.mode),
                    judge_model_id=self.judge_model_id,
                    input_text=inp_str,
                    actual_output=act_str,
                    expected_output=exp_str,
                    score=score_val,
                    reason=reason,
                    success=is_success,
                )

            if self.mode == "pass":
                return 1.0 if is_success else 0.0
            return score_val

        except Exception:
            return float("nan")


def deepeval_metric(
    kind: str,
    *,
    model: Optional[Any] = None,
    threshold: float = 0.5,
    name: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluation_steps: Optional[List[str]] = None,
    evaluation_params: Optional[List[Any]] = None,
    mode: str = "score",
    cache: Optional[JudgeCache] = None,
    **kwargs,
) -> DeepEvalMetric:
    """
    Convenience factory to create a DeepEvalMetric without manually importing deepeval.
    
    Supported kinds:
        - "answer_relevancy"
        - "faithfulness"
        - "hallucination"
        - "toxicity"
        - "bias"
        - "contextual_precision"
        - "contextual_recall"
        - "contextual_relevancy"
        - "summarization"
        - "json_correctness"
        - "g_eval"
    """
    kind_lower = kind.lower().strip()
    metric_name = name or f"deepeval_{kind_lower}"
    model_str = str(model) if model is not None else "default"

    def factory():
        try:
            import deepeval.metrics as dm
        except ImportError as e:
            raise ImportError(
                "DeepEval metrics require the 'deepeval' extra.\n"
                "Install it using: pip install prompt-prism[deepeval]"
            ) from e

        m_kwargs = {"threshold": threshold, **kwargs}
        if model is not None:
            m_kwargs["model"] = model

        if kind_lower == "answer_relevancy":
            return dm.AnswerRelevancyMetric(**m_kwargs)
        elif kind_lower == "faithfulness":
            return dm.FaithfulnessMetric(**m_kwargs)
        elif kind_lower == "hallucination":
            return dm.HallucinationMetric(**m_kwargs)
        elif kind_lower == "toxicity":
            return dm.ToxicityMetric(**m_kwargs)
        elif kind_lower == "bias":
            return dm.BiasMetric(**m_kwargs)
        elif kind_lower == "contextual_precision":
            return dm.ContextualPrecisionMetric(**m_kwargs)
        elif kind_lower == "contextual_recall":
            return dm.ContextualRecallMetric(**m_kwargs)
        elif kind_lower == "contextual_relevancy":
            return dm.ContextualRelevancyMetric(**m_kwargs)
        elif kind_lower == "summarization":
            return dm.SummarizationMetric(**m_kwargs)
        elif kind_lower in {"json", "json_correctness"}:
            return dm.JSONCorrectnessMetric(**m_kwargs)
        elif kind_lower == "g_eval":
            if criteria:
                m_kwargs["criteria"] = criteria
            if evaluation_steps:
                m_kwargs["evaluation_steps"] = evaluation_steps
            if evaluation_params:
                m_kwargs["evaluation_params"] = evaluation_params
            return dm.GEval(name=metric_name, **m_kwargs)
        else:
            raise ValueError(
                f"Unknown DeepEval metric kind '{kind}'. Supported kinds: "
                "answer_relevancy, faithfulness, hallucination, toxicity, bias, "
                "contextual_precision, contextual_recall, contextual_relevancy, "
                "summarization, json_correctness, g_eval."
            )

    return DeepEvalMetric(
        metric_factory=factory,
        name=metric_name,
        mode=mode,
        cache=cache,
        judge_model_id=model_str,
    )
