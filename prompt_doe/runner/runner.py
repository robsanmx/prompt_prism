"""
Experiment Runner: Executes DoE trials across datasets and evaluates metrics in parallel.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
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
            sample_id = item.get(self.id_col, s_idx) if self.id_col else s_idx
            trial_id = f"run_{run.run_id}_sample_{sample_id}"

            # 1. Compose Prompt
            if self.as_chat_messages:
                prompt_input = self.composer.compose_messages(run_config=run, data=item)
            else:
                prompt_input = self.composer.compose_text(run_config=run, data=item)

            # 2. Check Cache / Query LLM
            response: Optional[LLMResponse] = None
            if self.cache:
                response = self.cache.get(prompt_input, run.params)

            if response is None:
                response = self.client.generate(prompt_input, **run.params)
                if self.cache:
                    self.cache.set(prompt_input, response, run.params)

            # 3. Evaluate Metrics
            target = item.get(self.target_col) if self.target_col and self.target_col in item else None
            metrics = self.evaluator.evaluate(
                prediction=response.content,
                target=target,
                input_data=item,
            )

            return Trial(
                trial_id=trial_id,
                run_id=run.run_id,
                sample_id=sample_id,
                factor_levels=dict(run.factor_levels),
                prompt=prompt_input,
                raw_response=response.content,
                metrics=metrics,
                latency_ms=response.latency_ms,
                token_usage=response.token_usage,
                error=response.error,
            )

        # Execute concurrently
        if self.max_workers > 1 and len(trial_tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(execute_single_trial, task) for task in trial_tasks]
                for future in concurrent.futures.as_completed(futures):
                    trial = future.result()
                    trials.append(trial)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_trials)
        else:
            for task in trial_tasks:
                trial = execute_single_trial(task)
                trials.append(trial)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_trials)

        # Sort trials by run_id, sample_id for determinism
        trials.sort(key=lambda t: (t.run_id, str(t.sample_id)))

        return ExperimentResults(
            experiment_id=experiment_id,
            design=design,
            trials=trials,
            metadata={"num_runs": len(design.runs), "num_samples": len(items)},
        )
