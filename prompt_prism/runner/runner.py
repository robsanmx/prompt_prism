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
from ..template.composer import PromptComposer
from .cache import ResponseCache
from .client import CallableLLM, LLMClient, LLMResponse, MockLLM


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
        self.composer = composer
        self.client = client if isinstance(client, LLMClient) else CallableLLM(client)
        self.evaluator = (
            evaluator if isinstance(evaluator, Evaluator) else Evaluator(evaluator)
        )
        self.cache = cache
        self.max_workers = max(1, max_workers)
        self.target_col = target_col
        self.id_col = id_col
        self.retry_limit = retry_limit

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
            if (
                self.composer.template.sections
                and self.composer.template.sections[0].role is not None
            ):
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
                llm_response = self.client.generate(prompt_content)
                if self.cache and not llm_response.error:
                    self.cache.set(prompt_key, llm_response)

            # Evaluate metrics with rich context (including prompt and IDs)
            scores = {}
            if not llm_response.error:
                eval_context = {
                    **item,
                    "__prompt__": prompt_key,
                    "__run_id__": run.run_id,
                    "__sample_id__": sample_id,
                }
                scores = self.evaluator.evaluate(
                    prediction=llm_response.content,
                    target=target_value,
                    context=eval_context,
                )

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
                metadata={"combination": run.combination_string},
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
