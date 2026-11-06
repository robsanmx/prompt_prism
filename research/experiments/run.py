"""
Resumable, Interleaved Experiment Execution Engine for Studies B and C.
Executes factorial prompt configurations against models with randomized
block ordering, content-addressed caching, and provenance tracking.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from research.experiments.manifest import (
    ExperimentManifest,
    ProvenanceRecord,
    get_current_git_commit,
    sha256_hash,
)


class ResumableExperimentRunner:
    """
    Executes experiment trials with automatic resumption and caching.
    """

    def __init__(
        self,
        manifest: ExperimentManifest,
        task_id: str,
        model_id: str,
        provider: str,
        evaluator_fn: Callable[[str, Dict[str, Any]], Dict[str, float]],
        client_fn: Callable[[str], str],
        seed: int = 42,
    ):
        self.manifest = manifest
        self.task_id = task_id
        self.model_id = model_id
        self.provider = provider
        self.evaluator_fn = evaluator_fn
        self.client_fn = client_fn
        self.seed = seed
        self.git_commit = get_current_git_commit()

    def run_experiment(
        self,
        configurations: List[Tuple[int, Dict[str, int]]],  # List of (run_id, factor_levels)
        items: List[Dict[str, Any]],                      # List of dataset items
        render_prompt_fn: Callable[[Dict[str, int], Dict[str, Any]], str],
        split: str = "discovery",
        randomize_order: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute trials across configurations and items.
        Checks cache for each trial; only calls client_fn on cache miss.
        """
        rng = np.random.default_rng(self.seed)

        # Create all (config, item) task pairs
        tasks = []
        for run_id, factor_levels in configurations:
            for item in items:
                tasks.append((run_id, factor_levels, item))

        if randomize_order:
            rng.shuffle(tasks)

        total_tasks = len(tasks)
        cache_hits = 0
        executed_calls = 0

        for run_id, factor_levels, item in tasks:
            sample_id = str(item.get("id", item.get("sample_id", "item_unknown")))
            prompt_text = render_prompt_fn(factor_levels, item)

            cache_key = self.manifest.compute_cache_key(
                model_id=self.model_id,
                prompt_text=prompt_text,
                sample_id=sample_id,
                seed=self.seed,
            )

            # Check cache
            cached_rec = self.manifest.get_cached_record(cache_key)
            if cached_rec is not None:
                cache_hits += 1
                continue

            # Execute fresh call
            t0 = time.time()
            error_msg = None
            response_text = ""
            tokens_in = len(prompt_text.split()) * 2  # Approximate or provider reported
            tokens_out = 0

            try:
                response_text = self.client_fn(prompt_text)
                tokens_out = len(response_text.split()) * 2
            except Exception as e:
                error_msg = str(e)

            latency_ms = (time.time() - t0) * 1000.0

            # Evaluate metrics
            scores = {}
            if not error_msg:
                try:
                    scores = self.evaluator_fn(response_text, item)
                except Exception as eval_err:
                    error_msg = f"Evaluation error: {str(eval_err)}"

            # Cost estimation (e.g. $0.15/1M in, $0.60/1M out)
            cost_usd = (tokens_in * 0.15 + tokens_out * 0.60) / 1_000_000.0

            record = ProvenanceRecord(
                trial_id=str(uuid.uuid4()),
                experiment_id=self.manifest.experiment_id,
                git_commit=self.git_commit,
                task_id=self.task_id,
                model_id=self.model_id,
                provider=self.provider,
                run_id=run_id,
                factor_levels=factor_levels,
                sample_id=sample_id,
                split=split,
                prompt_text=prompt_text,
                prompt_hash=sha256_hash(prompt_text),
                response_text=response_text,
                response_hash=sha256_hash(response_text),
                metrics=scores,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                seed=self.seed,
                retry_count=0,
                error=error_msg,
                cache_key=cache_key,
            )

            self.manifest.add_record(record, persist=True)
            executed_calls += 1

        return {
            "total_tasks": total_tasks,
            "cache_hits": cache_hits,
            "executed_calls": executed_calls,
        }
