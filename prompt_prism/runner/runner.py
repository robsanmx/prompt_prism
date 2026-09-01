"""
Multi-threaded Experiment Runner with caching, parallel execution, and automated metrics collection.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from ..core.models import DesignMatrix, ExperimentResults, RunConfig, Trial
from ..evaluation.evaluator import Evaluator
from ..evaluation.metrics import Metric
from ..template.composer import PromptComposer
from .cache import ResponseCache
from .client import CallableLLM, LLMClient, LLMResponse


class ExperimentRunner:
    """
    Executes factorial prompt experiments over a benchmark dataset with parallel execution and caching.
    """

    def __init__(
        self,
        composer: PromptComposer,
        client: Union[LLMClient, Callable[..., Any]],
        evaluator: Union[Evaluator, Sequence[Union[Metric, Callable[..., float]]]],
        cache: Optional[ResponseCache] = None,
        max_workers: int = 4,
        target_col: str = "target",
        id_col: str = "id",
        retry_limit: int = 2,
    ):
        """
        Initialize ExperimentRunner.

        Args:
            composer: PromptComposer for rendering prompts.
            client: LLM client or callable.
            evaluator: Evaluator instance or list of metrics.
            cache: Optional ResponseCache.
            max_workers: Thread pool concurrency for trial execution.
            target_col: Column name for ground truth target.
            id_col: Column name for sample identifier.
            retry_limit: Number of additional attempts after the first on failure (default 2 means up to 3 total calls). Note: governs trial execution / client generation only, not inner judge metric calls.
        """
        self.composer = composer
        self.client = client if isinstance(client, LLMClient) else CallableLLM(client)
        self.evaluator = (
            evaluator if isinstance(evaluator, Evaluator) else Evaluator(evaluator)
        )
        self.cache = cache
        self.max_workers = max(1, max_workers)
        self.target_col = target_col
        self.id_col = id_col
        self.retry_limit = max(0, retry_limit)

    def run(
        self,
        design: DesignMatrix,
        dataset: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        experiment_id: str = "exp_01",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExperimentResults:
        """
        Execute all experimental runs across all dataset samples.
        """
        # Normalize dataset to list of dicts
        if isinstance(dataset, pd.DataFrame):
            data_items = dataset.to_dict(orient="records")
        else:
            data_items = list(dataset)

        trials: List[Trial] = []
        total_tasks = len(design.runs) * len(data_items)
        completed_tasks = 0

        # Build execution queue
        trial_tasks: List[Tuple[RunConfig, int, Dict[str, Any]]] = []
        for run in design.runs:
            for s_idx, item in enumerate(data_items):
                trial_tasks.append((run, s_idx, item))

        def execute_single_trial(task: Tuple[RunConfig, int, Dict[str, Any]]) -> Trial:
            run, s_idx, item = task
            sample_id = item.get(self.id_col, s_idx)
            target_value = item.get(self.target_col, None)

            # Compose prompt for this run condition and sample data
            has_explicit_roles = any(
                sec.role is not None for sec in self.composer.template.sections
            )
            if has_explicit_roles:
                # Chat messages format
                prompt_content = self.composer.compose_messages(
                    run_config=run, data=item
                )
                prompt_key = str(prompt_content)
            else:
                # Plain text format
                prompt_content = self.composer.compose_text(run_config=run, data=item)
                prompt_key = prompt_content

            # Check cache
            cached_resp = self.cache.get(prompt_key) if self.cache else None
            if cached_resp:
                llm_response = cached_resp
            else:
                # Execute with retry_limit attempts
                max_attempts = 1 + self.retry_limit
                llm_response = None
                for attempt in range(max_attempts):
                    try:
                        llm_response = self.client.generate(prompt_content)
                        if not llm_response.error:
                            if self.cache:
                                self.cache.set(prompt_key, llm_response)
                            break
                    except Exception as exc:
                        llm_response = LLMResponse(content="", error=str(exc))

                    if attempt < max_attempts - 1:
                        time.sleep(min(0.05 * (2**attempt), 0.5))

                if llm_response is None:
                    llm_response = LLMResponse(content="", error="No attempts executed")

            # Evaluate metrics with rich context (including prompt and IDs)
            scores = {}
            judge_reasons: Dict[str, str] = {}
            eval_context = {
                **item,
                "__prompt__": prompt_key,
                "__run_id__": run.run_id,
                "__sample_id__": sample_id,
            }
            if not llm_response.error:
                scores = self.evaluator.evaluate(
                    prediction=llm_response.content,
                    target=target_value,
                    context=eval_context,
                )
                # last_reasons is thread-local, so this reads only this trial's judges.
                judge_reasons = dict(self.evaluator.last_reasons)

            trial_metadata: Dict[str, Any] = {"combination": run.combination_string}
            for m_name, reason in judge_reasons.items():
                trial_metadata[f"judge_reasons.{m_name}"] = reason

            return Trial(
                run_id=run.run_id,
                sample_id=sample_id,
                factor_levels=run.factor_levels,
                prompt=prompt_key,
                raw_response=llm_response.content,
                metrics=scores,
                latency_ms=llm_response.latency_ms,
                token_usage=llm_response.token_usage,
                error=llm_response.error,
                metadata=trial_metadata,
            )

        # Threaded parallel execution
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_task = {
                executor.submit(execute_single_trial, task): task
                for task in trial_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    trial = future.result()
                    trials.append(trial)
                except Exception as e:
                    task = future_to_task[future]
                    run, s_idx, item = task
                    trials.append(
                        Trial(
                            run_id=run.run_id,
                            sample_id=item.get(self.id_col, s_idx),
                            factor_levels=run.factor_levels,
                            prompt="",
                            raw_response="",
                            metrics={},
                            error=str(e),
                        )
                    )

                completed_tasks += 1
                if progress_callback:
                    progress_callback(completed_tasks, total_tasks)

        # Sort trials for consistent ordering
        trials.sort(key=lambda t: (t.run_id, str(t.sample_id)))

        return ExperimentResults(
            experiment_id=experiment_id,
            design=design,
            trials=trials,
            metadata={
                "num_runs": len(design.runs),
                "num_samples": len(data_items),
                "total_trials": len(trials),
            },
        )
