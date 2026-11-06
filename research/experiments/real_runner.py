"""
Resumable Real-LLM Execution Runner with Interleaved Block Randomization.
Executes real or mock provider adapters, enforces content-addressed caching,
and guarantees atomic provenance recording.
"""

import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from research.experiments.manifest import ExperimentManifest, ProvenanceRecord, get_current_git_commit, sha256_hash
from research.experiments.providers import BaseProviderAdapter, ModelRequest, ModelResponse


class RealExperimentRunner:
    """
    Orchestrates real model inferences across experimental configurations and items.
    """

    def __init__(
        self,
        manifest: ExperimentManifest,
        adapter: BaseProviderAdapter,
        task_id: str,
        evaluator_fn: Callable[[str, Dict[str, Any]], Dict[str, float]],
        seed: int = 42,
    ):
        self.manifest = manifest
        self.adapter = adapter
        self.task_id = task_id
        self.evaluator_fn = evaluator_fn
        self.seed = seed
        self.git_commit = get_current_git_commit()

    def run_cell(
        self,
        configurations: List[Tuple[int, Dict[str, int]]],
        items: List[Dict[str, Any]],
        render_prompt_fn: Callable[[Dict[str, int], Dict[str, Any]], str],
        split: str = "discovery",
        replicates: int = 1,
        randomize_order: bool = True,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute experimental cell across configurations, items, and replicates.
        """
        # Build trial task matrix
        trials = []
        for run_id, f_levels in configurations:
            for item in items:
                for rep in range(replicates):
                    trials.append({
                        "run_id": run_id,
                        "factor_levels": f_levels,
                        "item": item,
                        "replicate": rep,
                    })

        if randomize_order:
            rng = random.Random(self.seed)
            rng.shuffle(trials)

        executed_calls = 0
        cache_hits = 0
        failures = 0

        for trial in trials:
            run_id = trial["run_id"]
            f_levels = trial["factor_levels"]
            item = trial["item"]
            rep = trial["replicate"]

            # Deterministic seed per trial replicate
            trial_seed = self.seed + (run_id * 10000) + rep

            # Render prompt
            prompt_text = render_prompt_fn(f_levels, item)
            cache_key = ExperimentManifest.compute_cache_key(
                model_id=self.adapter.model_id,
                prompt_text=prompt_text,
                sample_id=item["id"],
                seed=trial_seed,
            )

            # Check cache
            cached_rec = self.manifest.get_cached_record(cache_key)
            if cached_rec is not None:
                cache_hits += 1
                continue

            # Execute real inference request
            req = ModelRequest(
                prompt=prompt_text,
                model_id=self.adapter.model_id,
                temperature=0.0,
                seed=trial_seed,
                system_prompt=system_prompt,
            )

            response: ModelResponse = self.adapter.generate(req)
            executed_calls += 1

            if response.error:
                failures += 1
                metrics = {k: 0.0 for k in ["field_f1", "token_f1", "accuracy", "exact_match"]}
            else:
                metrics = self.evaluator_fn(response.raw_text, item)

            record = ProvenanceRecord(
                trial_id=f"trial_{uuid.uuid4().hex[:12]}",
                experiment_id=self.manifest.experiment_id,
                git_commit=self.git_commit,
                task_id=self.task_id,
                model_id=response.model_id,
                provider=self.adapter.__class__.__name__,
                run_id=run_id,
                factor_levels=f_levels,
                sample_id=item["id"],
                split=split,
                prompt_text=prompt_text,
                prompt_hash=sha256_hash(prompt_text),
                response_text=response.raw_text,
                response_hash=sha256_hash(response.raw_text or ""),
                metrics=metrics,
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                timestamp=response.timestamp,
                seed=trial_seed,
                retry_count=response.retry_count,
                error=response.error,
                cache_key=cache_key,
                data_origin=response.data_origin,
                request_id=response.request_id,
                finish_reason=response.finish_reason,
                system_fingerprint=response.system_fingerprint,
            )

            self.manifest.add_record(record, persist=True)

        return {
            "total_trials": len(trials),
            "executed_calls": executed_calls,
            "cache_hits": cache_hits,
            "failures": failures,
        }
