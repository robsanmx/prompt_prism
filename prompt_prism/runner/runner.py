"""
Experiment Runner: Executes DoE trials across datasets and evaluates metrics in parallel.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd

from ..core.factors import Factor, FactorSet
from ..core.models import DesignMatrix, ExperimentResults, RunConfig, Trial
from ..evaluation.evaluator import Evaluator
from ..evaluation.metrics import Metric
from ..template.composer import PromptComposer, PromptTemplate
from .cache import ResponseCache
from .client import CallableLLM, LLMClient, LLMResponse


class ExperimentRunner:
    """
    Executes a prompt optimization experiment over a DesignMatrix and test dataset.
    """

    def __init__(
        self,
        composer: PromptComposer,
        client: Union[LLMClient, Callable[..., Any]],
        evaluator: Optional[Union[Evaluator, Sequence[Metric]]] = None,
        cache: Optional[ResponseCache] = None,
        max_workers: int = 4,
        target_col: Optional[str] = "target",
        id_col: Optional[str] = "id",
        as_chat_messages: bool = False,
    ):
        self.composer = composer
        if isinstance(client, LLMClient):
            self.client = client
        elif callable(client):
            self.client = CallableLLM(client)
        else:
            raise TypeError(f"Expected LLMClient or callable, got {type(client)}")

        if isinstance(evaluator, Evaluator):
            self.evaluator = evaluator
        elif evaluator:
            self.evaluator = Evaluator(evaluator)
        else:
            self.evaluator = Evaluator()

        self.cache = cache
        self.max_workers = max_workers
        self.target_col = target_col
        self.id_col = id_col
        self.as_chat_messages = as_chat_messages

    def run(
        self,
        design: DesignMatrix,
        dataset: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        experiment_id: str = "exp_01",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExperimentResults:
        """
        Run the complete DoE experiment across all runs and sample test cases.
        """
        # Convert dataset to list of dicts
        if isinstance(dataset, pd.DataFrame):
            items = dataset.to_dict(orient="records")
        else:
            items = list(dataset)

        total_trials = len(design.runs) * len(items)
        trial_tasks: List[Tuple[RunConfig, int, Dict[str, Any]]] = []

        for run in design.runs:
            for s_idx, item in enumerate(items):
                trial_tasks.append((run, s_idx, item))

        trials: List[Trial] = []
        completed = 0

        def execute_single_trial(task_tuple: Tuple[RunConfig, int, Dict[str, Any]]) -> Trial:
            run, s_idx, item = task_tuple
            sample_id = item.get(self.id_col, s_idx)
            target_value = item.get(self.target_col, None)

            # Compose prompt
            if self.as_chat_messages:
                prompt_content = self.composer.compose_messages(run, data=item)
                prompt_key = str(prompt_content)
            else:
                prompt_content = self.composer.compose_text(run, data=item)
                prompt_key = prompt_content

            # Check cache
            cached_resp = self.cache.get(prompt_key) if self.cache else None
            if cached_resp:
                llm_response = cached_resp
            else:
                llm_response = self.client.generate(prompt_content)
                if self.cache and not llm_response.error:
                    self.cache.set(prompt_key, llm_response)

            # Evaluate metrics
            scores = {}
            if not llm_response.error:
                scores = self.evaluator.evaluate(
                    prediction=llm_response.content,
                    target=target_value,
                    context=item,
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(execute_single_trial, task): task for task in trial_tasks}
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
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_trials)

        # Sort trials by run_id, then sample_id
        trials.sort(key=lambda t: (t.run_id, str(t.sample_id)))

        return ExperimentResults(
            experiment_id=experiment_id,
            design=design,
            trials=trials,
            metadata={"num_runs": len(design.runs), "num_samples": len(items), "total_trials": len(trials)},
        )
