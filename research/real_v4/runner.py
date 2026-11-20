"""Resumable, phase-locked, batch-accelerated multi-environment runner for Real-v4 (Task 10 & Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from research.real_v4.baselines import (
    BalancedRandomTwoStageBaseline,
    SequentialBayesianOptBaseline,
)
from research.real_v4.budget import BudgetLedger, FrozenOperationalBudget
from research.real_v4.locking import (
    PhaseLockError,
    StudyPhase,
    compute_study_fingerprint,
    verify_phase_prerequisites,
    write_phase_lock,
)
from research.real_v4.manifest import (
    AppendOnlyManifest,
    deterministic_call_key,
)
from research.real_v4.perturbations import (
    RenderedPromptInstance,
    render_prompt_instance,
    score_task_response,
)
from research.real_v4.providers import BaseRealV4Adapter, CanonicalAdapterRequest
from research.real_v4.schemas import CURRENT_SCHEMA_VERSION, canonical_sha256
from research.real_v4.selection import (
    PromptPrismRobustSelector,
    run_id_to_levels,
)


class RunnerInterrupted(RuntimeError):
    """Raised intentionally in resumability tests when `max_new_calls` is reached mid-phase."""


@dataclass(frozen=True)
class RunnerPaths:
    protocol_dir: Path
    dataset_manifest_path: Path
    perturbation_registry_path: Path
    model_registry_path: Path
    calibration_config_path: Path
    code_paths: Tuple[Path, ...]
    results_dir: Path

    @property
    def locks_dir(self) -> Path:
        return self.results_dir / "locks"


class ResumableRealV4Runner:
    """Enforces strict phase-lock progression, confirmation quarantine, budget parity, and crash-safe resume."""

    def __init__(
        self,
        *,
        study_id: str,
        paths: RunnerPaths,
        adapters: Mapping[str, BaseRealV4Adapter],
        splits_by_task_and_partition: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
        tasks: Sequence[str],
        models: Sequence[str],
        validation_env_ids: Tuple[str, ...] = ("ref", "fmt_whitespace", "word_paraphrase_1"),
        stability_env_ids: Tuple[str, ...] = (
            "repeat_r2",
            "repeat_r3",
            "fmt_whitespace",
            "fmt_delimiter",
            "word_paraphrase_1",
            "word_paraphrase_2",
            "item_holdout_group",
            "temporal_wave_2",
        ),
        num_baseline_trajectories: int = 10,
        n_bootstrap: int = 200,
    ) -> None:
        self.study_id = study_id
        self.paths = paths
        self.adapters = dict(adapters)
        self.splits = dict(splits_by_task_and_partition)
        self.tasks = tuple(tasks)
        self.models = tuple(models)
        self.validation_env_ids = tuple(validation_env_ids)
        self.stability_env_ids = tuple(stability_env_ids)
        self.num_baseline_trajectories = int(num_baseline_trajectories)
        self.n_bootstrap = int(n_bootstrap)

        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.locks_dir.mkdir(parents=True, exist_ok=True)

        self.prism_selector = PromptPrismRobustSelector(
            pad_to_max=True,
            n_bootstrap_screen=self.n_bootstrap,
            n_bootstrap_validation=self.n_bootstrap,
        )
        self.random_baseline = BalancedRandomTwoStageBaseline(
            num_designs=self.num_baseline_trajectories,
            n_bootstrap_validation=self.n_bootstrap,
        )
        self.bo_baseline = SequentialBayesianOptBaseline(
            num_trajectories=self.num_baseline_trajectories,
            n_bootstrap_validation=self.n_bootstrap,
        )

    def current_fingerprint(self) -> Dict[str, Any]:
        return compute_study_fingerprint(
            protocol_dir=self.paths.protocol_dir,
            dataset_manifest_path=self.paths.dataset_manifest_path,
            perturbation_registry_path=self.paths.perturbation_registry_path,
            model_registry_path=self.paths.model_registry_path,
            calibration_config_path=self.paths.calibration_config_path,
            code_paths=self.paths.code_paths,
        )

    def _verify_for(self, phase: StudyPhase) -> Tuple[Dict[str, Any], Dict[StudyPhase, Dict[str, Any]]]:
        fp = self.current_fingerprint()
        verified = verify_phase_prerequisites(
            target_phase=phase,
            locks_dir=self.paths.locks_dir,
            artifacts_dir=self.paths.results_dir,
            current_fingerprint=fp,
        )
        return fp, verified

    def _execute_or_replay_batch(
        self,
        manifest: AppendOnlyManifest,
        *,
        phase: StudyPhase,
        task_id: str,
        model_id: str,
        env_id: str,
        run_id: int,
        items: Sequence[Mapping[str, Any]],
        repeat_idx: int = 0,
        new_calls_counter: List[int],
        max_new_calls: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Optional[Dict[str, Any]]] = [None] * len(items)
        missing_meta: List[Tuple[int, str, Mapping[str, Any], RenderedPromptInstance, CanonicalAdapterRequest]] = []
        levels = run_id_to_levels(run_id)
        hit_interrupt = False

        for idx, item in enumerate(items):
            sample_id = str(item["sample_id"])
            ck = deterministic_call_key(
                study_id=self.study_id,
                phase=phase.value,
                task_id=task_id,
                model_id=model_id,
                env_id=env_id,
                run_id=run_id,
                sample_id=sample_id,
                repeat_idx=repeat_idx,
            )
            if manifest.has_key(ck):
                results[idx] = manifest.get_record(ck)
                continue

            if max_new_calls is not None and (new_calls_counter[0] + len(missing_meta)) >= max_new_calls:
                hit_interrupt = True
                break

            rendered = render_prompt_instance(task_id, item, levels, env_id=env_id)
            req = CanonicalAdapterRequest(
                task_id=task_id,
                sample_id=sample_id,
                env_id=env_id,
                system_text=rendered.system_text,
                user_text=rendered.user_text,
            )
            missing_meta.append((idx, ck, item, rendered, req))

        if missing_meta:
            adapter = self.adapters[model_id]
            resps = adapter.execute_batch([m[4] for m in missing_meta])
            batch_records: List[Dict[str, Any]] = []
            for (idx, ck, item, rendered, _req), resp in zip(missing_meta, resps):
                new_calls_counter[0] += 1
                is_terminal_err = bool(resp.error is not None or resp.provenance.finish_reason == "HTTP_ERROR")
                score = 0.0 if is_terminal_err else score_task_response(task_id, resp.raw_text, item)
                prov_dump = resp.provenance.model_dump(mode="json")
                resp_id = "" if is_terminal_err else f"{prov_dump['response_id']}::{ck[:12]}"

                record = {
                    "call_key": ck,
                    "study_id": self.study_id,
                    "phase": phase.value,
                    "task_id": task_id,
                    "model_id": model_id,
                    "env_id": env_id,
                    "run_id": run_id,
                    "sample_id": str(item["sample_id"]),
                    "group_id": str(item["group_id"]),
                    "repeat_idx": repeat_idx,
                    "prompt_sha256": rendered.rendered_sha256,
                    "parent_prompt_sha256": rendered.parent_prompt_sha256,
                    "raw_text": resp.raw_text,
                    "score": float(score),
                    "status": "terminal_failure" if is_terminal_err else "ok",
                    "response_id": resp_id,
                    "returned_model_version": prov_dump["returned_model_version"],
                    "within_pinned_envelope": bool(prov_dump["within_pinned_envelope"]),
                    "provenance": prov_dump,
                }
                batch_records.append(record)
                results[idx] = record

                if new_calls_counter[0] % 480 == 0:
                    print(
                        f"[{phase.value}] completed {new_calls_counter[0]} new calls "
                        f"(latest: {task_id} | {model_id} | env={env_id} | run={run_id})",
                        flush=True,
                    )
            manifest.append_many(batch_records)

        if hit_interrupt:
            raise RunnerInterrupted(f"Interrupted cleanly after {new_calls_counter[0]} new calls in {phase.value}")

        return [r for r in results if r is not None]

    def freeze_protocol(self) -> Dict[str, Any]:
        fp = self.current_fingerprint()
        return write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.PROTOCOL,
            fingerprint=fp,
            artifact_path=None,
            prior_phase=None,
        )

    def run_pilot(
        self,
        *,
        pilot_repeats: int = 1,
        pilot_run_id: int = 0,
        max_new_calls: Optional[int] = None,
    ) -> Dict[str, Any]:
        if pilot_repeats < 1:
            raise ValueError("pilot_repeats must be >= 1")
        fp, _ = self._verify_for(StudyPhase.PILOT)
        manifest_path = self.paths.results_dir / "pilot_manifest.jsonl"
        manifest = AppendOnlyManifest(manifest_path)
        counter = [0]
        expected_keys: List[str] = []

        for model_id in self.models:
            for task_id in self.tasks:
                items = list(self.splits[(task_id, "pilot")])
                for rep_idx in range(pilot_repeats):
                    recs = self._execute_or_replay_batch(
                        manifest,
                        phase=StudyPhase.PILOT,
                        task_id=task_id,
                        model_id=model_id,
                        env_id="ref",
                        run_id=pilot_run_id,
                        items=items,
                        repeat_idx=rep_idx,
                        new_calls_counter=counter,
                        max_new_calls=max_new_calls,
                    )
                    expected_keys.extend(str(r["call_key"]) for r in recs)
            self.adapters[model_id].release_memory()

        audit = manifest.replay_audit(expected_keys)
        return write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.PILOT,
            fingerprint=fp,
            artifact_path=manifest_path,
            prior_phase=StudyPhase.PROTOCOL,
            extra_metadata={
                "pilot_repeats": int(pilot_repeats),
                "pilot_run_id": int(pilot_run_id),
                "replay_audit": audit,
            },
        )

    def run_screen_and_lock_candidates(
        self,
        *,
        max_new_calls: Optional[int] = None,
        include_research_full_cube_audit: bool = False,
    ) -> Dict[str, Any]:
        fp, _ = self._verify_for(StudyPhase.SCREEN)
        manifest_path = self.paths.results_dir / "screen_manifest.jsonl"
        manifest = AppendOnlyManifest(manifest_path)
        counter = [0]
        expected_keys: Set[str] = set()

        candidate_payload: Dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "study_id": self.study_id,
            "cells": {},
        }

        # Outer loop by model_id so each local MLX model stays resident in GPU memory across all 6 tasks
        for model_id in self.models:
            for task_id in self.tasks:
                scr_items = list(self.splits[(task_id, "screen")])
                scr_groups = [str(it["group_id"]) for it in scr_items]
                scr_item_ids = [str(it["sample_id"]) for it in scr_items]
                val_items = list(self.splits[(task_id, "validation")])
                cell_id = f"{task_id}::{model_id}"

                def _eval_screen_run(rid: int) -> np.ndarray:
                    recs = self._execute_or_replay_batch(
                        manifest,
                        phase=StudyPhase.SCREEN,
                        task_id=task_id,
                        model_id=model_id,
                        env_id="ref",
                        run_id=rid,
                        items=scr_items,
                        repeat_idx=0,
                        new_calls_counter=counter,
                        max_new_calls=max_new_calls,
                    )
                    expected_keys.update(str(r["call_key"]) for r in recs)
                    return np.array([float(r["score"]) for r in recs], dtype=float)

                ledger = BudgetLedger(
                    FrozenOperationalBudget(
                        screen_configs=16,
                        screen_items=len(scr_items),
                        screen_repeats=1,
                        validation_candidates=4,
                        validation_items=len(val_items),
                        validation_env_ids=self.validation_env_ids,
                        validation_repeats=1,
                    )
                )

                # 1. PromptPrism 16-run screen
                prism_scr = {}
                for rid in self.prism_selector.screen_run_ids():
                    prism_scr[rid] = _eval_screen_run(rid)
                    ledger.record_screen_batch(
                        cell_id=cell_id,
                        method_id="promptprism",
                        trajectory_id="primary",
                        run_id=rid,
                        item_ids=scr_item_ids,
                    )
                prism_cands = self.prism_selector.generate_candidates(
                    prism_scr, group_ids=scr_groups, seed=20260402
                )

                # 2. Balanced Random screen (all trajectories)
                rand_cands_by_traj: Dict[str, List[Dict[str, Any]]] = {}
                rand_screen_runs: Dict[str, List[int]] = {}
                for d_idx, s_runs in enumerate(self.random_baseline.frozen_designs):
                    traj_id = f"design_{d_idx:02d}"
                    means = []
                    for rid in s_runs:
                        arr = _eval_screen_run(rid)
                        ledger.record_screen_batch(
                            cell_id=cell_id,
                            method_id="balanced_random",
                            trajectory_id=traj_id,
                            run_id=rid,
                            item_ids=scr_item_ids,
                        )
                        means.append((float(np.mean(arr)), rid))
                    means.sort(key=lambda pair: (-round(pair[0] / 1e-12), pair[1]))
                    rand_screen_runs[traj_id] = list(s_runs)
                    rand_cands_by_traj[traj_id] = [
                        {
                            "run_id": rid,
                            "factor_levels": run_id_to_levels(rid),
                            "source_rule": "random_screen_top_k",
                            "screen_score": float(np.clip(mu, 0.0, 1.0)),
                            "rank_in_screen": k_i,
                        }
                        for k_i, (mu, rid) in enumerate(means[:4], start=1)
                    ]

                # 3. Sequential Bayesian Opt screen
                bo_cands_by_traj: Dict[str, List[Dict[str, Any]]] = {}
                bo_screen_runs: Dict[str, List[int]] = {}
                dummy_val = {e: {r: np.zeros(len(val_items)) for r in range(64)} for e in self.validation_env_ids}
                bo_trajs = self.bo_baseline.run_all_trajectories(
                    cell_id=cell_id,
                    screen_evaluator=_eval_screen_run,
                    validation_evaluator=lambda rid, env_id: dummy_val[env_id][rid],
                    screen_item_ids=scr_item_ids,
                    validation_item_ids=[str(it["sample_id"]) for it in val_items],
                    validation_env_ids=self.validation_env_ids,
                    ledger=None,
                )
                for tr in bo_trajs:
                    for rid in tr.screen_run_ids:
                        ledger.record_screen_batch(
                            cell_id=cell_id,
                            method_id="bayesian_opt",
                            trajectory_id=tr.trajectory_id,
                            run_id=rid,
                            item_ids=scr_item_ids,
                        )
                    bo_screen_runs[tr.trajectory_id] = list(tr.screen_run_ids)
                    bo_cands_by_traj[tr.trajectory_id] = [c.model_dump(mode="json") for c in tr.candidates]

                # Optional research-only 64-run screening audit overhead (outside operational BudgetLedger)
                if include_research_full_cube_audit:
                    for rid_audit in range(64):
                        _eval_screen_run(rid_audit)

                candidate_payload["cells"][cell_id] = {
                    "promptprism": {
                        "primary": {
                            "screen_run_ids": list(self.prism_selector.screen_run_ids()),
                            "candidates": [c.model_dump(mode="json") for c in prism_cands],
                        }
                    },
                    "balanced_random": {
                        tid: {"screen_run_ids": rand_screen_runs[tid], "candidates": rand_cands_by_traj[tid]}
                        for tid in sorted(rand_cands_by_traj)
                    },
                    "bayesian_opt": {
                        tid: {"screen_run_ids": bo_screen_runs[tid], "candidates": bo_cands_by_traj[tid]}
                        for tid in sorted(bo_cands_by_traj)
                    },
                }
            self.adapters[model_id].release_memory()

        audit = manifest.replay_audit(expected_keys)
        write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.SCREEN,
            fingerprint=fp,
            artifact_path=manifest_path,
            prior_phase=StudyPhase.PILOT,
            extra_metadata={
                "include_research_full_cube_audit": bool(include_research_full_cube_audit),
                "replay_audit": audit,
            },
        )

        cand_path = self.paths.results_dir / "candidate_lock.json"
        candidate_payload["candidate_lock_sha256"] = canonical_sha256(candidate_payload)
        cand_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.CANDIDATE_LOCK,
            fingerprint=fp,
            artifact_path=cand_path,
            prior_phase=StudyPhase.SCREEN,
        )

    def run_validation_and_lock_final_selections(self, *, max_new_calls: Optional[int] = None) -> Dict[str, Any]:
        fp, _ = self._verify_for(StudyPhase.VALIDATION)
        cand_lock_data = json.loads((self.paths.results_dir / "candidate_lock.json").read_text(encoding="utf-8"))

        manifest_path = self.paths.results_dir / "validation_manifest.jsonl"
        manifest = AppendOnlyManifest(manifest_path)
        counter = [0]
        expected_keys: Set[str] = set()

        final_selections: Dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "study_id": self.study_id,
            "cells": {},
        }

        from research.real_v4.schemas import CandidateProvenance

        for model_id in self.models:
            for task_id in self.tasks:
                scr_items = list(self.splits[(task_id, "screen")])
                scr_item_ids = [str(it["sample_id"]) for it in scr_items]
                val_items = list(self.splits[(task_id, "validation")])
                val_item_ids = [str(it["sample_id"]) for it in val_items]
                val_groups = [str(it["group_id"]) for it in val_items]

                cell_id = f"{task_id}::{model_id}"
                cell_cands = cand_lock_data["cells"][cell_id]

                ledger = BudgetLedger(
                    FrozenOperationalBudget(
                        screen_configs=16,
                        screen_items=len(scr_items),
                        screen_repeats=1,
                        validation_candidates=4,
                        validation_items=len(val_items),
                        validation_env_ids=self.validation_env_ids,
                        validation_repeats=1,
                    )
                )

                def _eval_val(rid: int, env_id: str) -> np.ndarray:
                    recs = self._execute_or_replay_batch(
                        manifest,
                        phase=StudyPhase.VALIDATION,
                        task_id=task_id,
                        model_id=model_id,
                        env_id=env_id,
                        run_id=rid,
                        items=val_items,
                        repeat_idx=0,
                        new_calls_counter=counter,
                        max_new_calls=max_new_calls,
                    )
                    expected_keys.update(str(r["call_key"]) for r in recs)
                    return np.array([float(r["score"]) for r in recs], dtype=float)

                cell_selected: Dict[str, Dict[str, int]] = {}
                required_trajs: List[Tuple[str, str]] = []

                for method_id in ("promptprism", "balanced_random", "bayesian_opt"):
                    cell_selected[method_id] = {}
                    for traj_id, t_info in sorted(cell_cands[method_id].items()):
                        required_trajs.append((method_id, traj_id))
                        for s_rid in t_info["screen_run_ids"]:
                            ledger.record_screen_batch(
                                cell_id=cell_id,
                                method_id=method_id,
                                trajectory_id=traj_id,
                                run_id=int(s_rid),
                                item_ids=scr_item_ids,
                            )
                        c_objs = [CandidateProvenance.model_validate(c) for c in t_info["candidates"]]
                        val_by_env: Dict[str, Dict[int, np.ndarray]] = {e: {} for e in self.validation_env_ids}
                        for c_obj in c_objs:
                            for env_id in self.validation_env_ids:
                                val_by_env[env_id][c_obj.run_id] = _eval_val(c_obj.run_id, env_id)
                                ledger.record_validation_batch(
                                    cell_id=cell_id,
                                    method_id=method_id,
                                    trajectory_id=traj_id,
                                    run_id=c_obj.run_id,
                                    env_id=env_id,
                                    item_ids=val_item_ids,
                                )
                        winner, _, _ = self.prism_selector.select_final_prompt(
                            c_objs, val_by_env, group_ids=val_groups, seed=20260403
                        )
                        cell_selected[method_id][traj_id] = winner.run_id

                # Reconcile exact operational budgets BEFORE issuing confirmation unlock
                ledger.assert_equal_operational_budgets(cell_id, required_trajs)
                final_selections["cells"][cell_id] = cell_selected
            self.adapters[model_id].release_memory()

        audit = manifest.replay_audit(expected_keys)
        write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.VALIDATION,
            fingerprint=fp,
            artifact_path=manifest_path,
            prior_phase=StudyPhase.CANDIDATE_LOCK,
            extra_metadata={"replay_audit": audit},
        )

        final_path = self.paths.results_dir / "final_selection_lock.json"
        final_selections["final_selection_sha256"] = canonical_sha256(final_selections)
        final_path.write_text(json.dumps(final_selections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.FINAL_SELECTION_LOCK,
            fingerprint=fp,
            artifact_path=final_path,
            prior_phase=StudyPhase.VALIDATION,
        )

    def _locked_run_id_union_for_cell(self, final_lock_data: Mapping[str, Any], cell_id: str) -> List[int]:
        cell_map = final_lock_data["cells"][cell_id]
        run_ids: Set[int] = set()
        for method_map in cell_map.values():
            for rid in method_map.values():
                run_ids.add(int(rid))
        return sorted(run_ids)

    def run_clean_confirmation(
        self,
        *,
        max_new_calls: Optional[int] = None,
        extra_unregistered_run_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        fp, _ = self._verify_for(StudyPhase.CLEAN_CONFIRMATION)
        if extra_unregistered_run_id is not None:
            raise PhaseLockError(
                f"Refusing to run unlocked/unregistered confirmation run_id={extra_unregistered_run_id}"
            )
        final_data = json.loads((self.paths.results_dir / "final_selection_lock.json").read_text(encoding="utf-8"))
        manifest_path = self.paths.results_dir / "confirmation_manifest.jsonl"
        manifest = AppendOnlyManifest(manifest_path)
        counter = [0]
        expected_keys: Set[str] = set()

        for model_id in self.models:
            for task_id in self.tasks:
                conf_items = list(self.splits[(task_id, "clean_confirmation")])
                cell_id = f"{task_id}::{model_id}"
                locked_runs = self._locked_run_id_union_for_cell(final_data, cell_id)
                for rid in locked_runs:
                    recs = self._execute_or_replay_batch(
                        manifest,
                        phase=StudyPhase.CLEAN_CONFIRMATION,
                        task_id=task_id,
                        model_id=model_id,
                        env_id="ref",
                        run_id=rid,
                        items=conf_items,
                        repeat_idx=0,
                        new_calls_counter=counter,
                        max_new_calls=max_new_calls,
                    )
                    expected_keys.update(str(r["call_key"]) for r in recs)
            self.adapters[model_id].release_memory()

        audit = manifest.replay_audit(expected_keys)
        return write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.CLEAN_CONFIRMATION,
            fingerprint=fp,
            artifact_path=manifest_path,
            prior_phase=StudyPhase.FINAL_SELECTION_LOCK,
            extra_metadata={"replay_audit": audit},
        )

    def run_stability_wave_and_lock_raw_results(
        self,
        *,
        requested_env_ids: Optional[Sequence[str]] = None,
        max_new_calls: Optional[int] = None,
        write_final_locks: bool = True,
    ) -> Dict[str, Any]:
        fp, verified = self._verify_for(StudyPhase.STABILITY_WAVE)
        protocol_registered_envs = set(verified[StudyPhase.PROTOCOL]["registered_env_ids"])
        envs_to_run = tuple(requested_env_ids) if requested_env_ids is not None else self.stability_env_ids

        for env_id in envs_to_run:
            if env_id not in protocol_registered_envs or env_id == "ref":
                raise PhaseLockError(
                    f"Stability environment {env_id!r} was not registered in protocol lock {sorted(protocol_registered_envs)}"
                )

        final_data = json.loads((self.paths.results_dir / "final_selection_lock.json").read_text(encoding="utf-8"))
        manifest_path = self.paths.results_dir / "stability_manifest.jsonl"
        manifest = AppendOnlyManifest(manifest_path)
        counter = [0]
        expected_keys: Set[str] = set()

        for model_id in self.models:
            for env_id in envs_to_run:
                part_name = "stability_confirmation" if env_id == "item_holdout_group" else "clean_confirmation"
                rep_idx = 1 if env_id == "repeat_r2" else (2 if env_id == "repeat_r3" else 0)
                for task_id in self.tasks:
                    items = list(self.splits[(task_id, part_name)])
                    cell_id = f"{task_id}::{model_id}"
                    locked_runs = self._locked_run_id_union_for_cell(final_data, cell_id)
                    for rid in locked_runs:
                        recs = self._execute_or_replay_batch(
                            manifest,
                            phase=StudyPhase.STABILITY_WAVE,
                            task_id=task_id,
                            model_id=model_id,
                            env_id=env_id,
                            run_id=rid,
                            items=items,
                            repeat_idx=rep_idx,
                            new_calls_counter=counter,
                            max_new_calls=max_new_calls,
                        )
                        expected_keys.update(str(r["call_key"]) for r in recs)
            self.adapters[model_id].release_memory()

        if not write_final_locks:
            return {"status": "partial_wave_complete", "new_calls": counter[0], "envs_executed": list(envs_to_run)}

        audit = manifest.replay_audit(expected_keys)
        write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.STABILITY_WAVE,
            fingerprint=fp,
            artifact_path=manifest_path,
            prior_phase=StudyPhase.CLEAN_CONFIRMATION,
            extra_metadata={"replay_audit": audit},
        )

        raw_lock_path = self.paths.results_dir / "raw_results_lock.json"
        raw_payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "study_id": self.study_id,
            "confirmation_manifest_sha256": verified[StudyPhase.CLEAN_CONFIRMATION]["artifact_sha256"],
            "stability_manifest_replay": audit,
        }
        raw_payload["raw_results_sha256"] = canonical_sha256(raw_payload)
        raw_lock_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return write_phase_lock(
            locks_dir=self.paths.locks_dir,
            phase=StudyPhase.RAW_RESULTS_LOCK,
            fingerprint=fp,
            artifact_path=raw_lock_path,
            prior_phase=StudyPhase.STABILITY_WAVE,
        )
