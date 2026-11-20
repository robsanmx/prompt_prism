"""Operational budget ledger enforcing strict call, candidate, item, and repeat parity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set, Tuple

from research.real_v4.schemas import MethodBudgetSpec


class BudgetViolationError(RuntimeError):
    """Raised whenever a method violates the frozen operational budget or parity contract."""


@dataclass(frozen=True)
class FrozenOperationalBudget:
    screen_configs: int = 16
    screen_items: int = 120
    screen_repeats: int = 1
    validation_candidates: int = 4
    validation_items: int = 60
    validation_env_ids: Tuple[str, ...] = ("ref", "fmt_whitespace", "word_paraphrase_1")
    validation_repeats: int = 1

    @property
    def expected_screen_calls(self) -> int:
        return self.screen_configs * self.screen_items * self.screen_repeats

    @property
    def expected_validation_calls(self) -> int:
        return (
            self.validation_candidates
            * self.validation_items
            * len(self.validation_env_ids)
            * self.validation_repeats
        )

    @property
    def expected_total_operational_calls(self) -> int:
        return self.expected_screen_calls + self.expected_validation_calls


@dataclass
class _TrajectoryTracker:
    screen_runs: Set[int] = field(default_factory=set)
    screen_items: Set[str] = field(default_factory=set)
    screen_repeats: Set[int] = field(default_factory=set)
    screen_calls: Set[Tuple[int, str, int]] = field(default_factory=set)

    val_runs: Set[int] = field(default_factory=set)
    val_items: Set[str] = field(default_factory=set)
    val_envs: Set[str] = field(default_factory=set)
    val_repeats: Set[int] = field(default_factory=set)
    val_calls: Set[Tuple[int, str, str, int]] = field(default_factory=set)


class BudgetLedger:
    """Append-only operational budget ledger separating method budget from oracle research overhead."""

    def __init__(self, frozen_budget: FrozenOperationalBudget) -> None:
        self.frozen_budget = frozen_budget
        self._trackers: Dict[Tuple[str, str, str], _TrajectoryTracker] = {}
        self._oracle_audit_calls: Dict[str, Set[Tuple[int, str, int]]] = {}

    def record_screen_batch(
        self,
        *,
        cell_id: str,
        method_id: str,
        trajectory_id: str,
        run_id: int,
        item_ids: Iterable[str],
        repeat_idx: int = 0,
        partition_tag: str = "screen",
    ) -> None:
        if "confirm" in partition_tag.lower() or "val" in partition_tag.lower():
            raise BudgetViolationError(f"Illegal partition_tag={partition_tag!r} in record_screen_batch")
        if repeat_idx < 0 or repeat_idx >= self.frozen_budget.screen_repeats:
            raise BudgetViolationError(
                f"Screen repeat_idx={repeat_idx} exceeds frozen screen_repeats={self.frozen_budget.screen_repeats}"
            )
        tracker = self._trackers.setdefault((cell_id, method_id, trajectory_id), _TrajectoryTracker())
        if run_id not in tracker.screen_runs and len(tracker.screen_runs) >= self.frozen_budget.screen_configs:
            raise BudgetViolationError(
                f"{method_id}[{trajectory_id}] exceeded frozen screen_configs={self.frozen_budget.screen_configs}"
            )
        tracker.screen_runs.add(run_id)
        tracker.screen_repeats.add(repeat_idx)

        for item_id in item_ids:
            if item_id not in tracker.screen_items and len(tracker.screen_items) >= self.frozen_budget.screen_items:
                raise BudgetViolationError(
                    f"{method_id}[{trajectory_id}] exceeded frozen screen_items={self.frozen_budget.screen_items}"
                )
            tracker.screen_items.add(item_id)
            call_key = (run_id, item_id, repeat_idx)
            if call_key in tracker.screen_calls:
                raise BudgetViolationError(
                    f"Duplicate screen call {call_key} for {method_id}[{trajectory_id}] in {cell_id}"
                )
            tracker.screen_calls.add(call_key)

    def record_validation_batch(
        self,
        *,
        cell_id: str,
        method_id: str,
        trajectory_id: str,
        run_id: int,
        env_id: str,
        item_ids: Iterable[str],
        repeat_idx: int = 0,
        partition_tag: str = "validation",
    ) -> None:
        if "confirm" in partition_tag.lower():
            raise BudgetViolationError(f"Illegal partition_tag={partition_tag!r} in record_validation_batch")
        if env_id not in self.frozen_budget.validation_env_ids:
            raise BudgetViolationError(
                f"Unregistered validation environment {env_id!r}; allowed={self.frozen_budget.validation_env_ids}"
            )
        if repeat_idx < 0 or repeat_idx >= self.frozen_budget.validation_repeats:
            raise BudgetViolationError(
                f"Validation repeat_idx={repeat_idx} exceeds frozen validation_repeats={self.frozen_budget.validation_repeats}"
            )
        tracker = self._trackers.setdefault((cell_id, method_id, trajectory_id), _TrajectoryTracker())
        if run_id not in tracker.val_runs and len(tracker.val_runs) >= self.frozen_budget.validation_candidates:
            raise BudgetViolationError(
                f"{method_id}[{trajectory_id}] exceeded frozen validation_candidates={self.frozen_budget.validation_candidates}"
            )
        tracker.val_runs.add(run_id)
        tracker.val_envs.add(env_id)
        tracker.val_repeats.add(repeat_idx)

        for item_id in item_ids:
            if item_id not in tracker.val_items and len(tracker.val_items) >= self.frozen_budget.validation_items:
                raise BudgetViolationError(
                    f"{method_id}[{trajectory_id}] exceeded frozen validation_items={self.frozen_budget.validation_items}"
                )
            tracker.val_items.add(item_id)
            call_key = (run_id, env_id, item_id, repeat_idx)
            if call_key in tracker.val_calls:
                raise BudgetViolationError(
                    f"Duplicate validation call {call_key} for {method_id}[{trajectory_id}] in {cell_id}"
                )
            tracker.val_calls.add(call_key)

    def record_oracle_audit_batch(
        self,
        *,
        cell_id: str,
        run_id: int,
        item_ids: Iterable[str],
        repeat_idx: int = 0,
    ) -> None:
        """Record research-only 64-run screening cube audit calls outside method operational budgets."""
        bucket = self._oracle_audit_calls.setdefault(cell_id, set())
        for item_id in item_ids:
            bucket.add((run_id, item_id, repeat_idx))

    def to_budget_spec(self, cell_id: str, method_id: str, trajectory_id: str) -> MethodBudgetSpec:
        key = (cell_id, method_id, trajectory_id)
        if key not in self._trackers:
            raise BudgetViolationError(f"No recorded operational budget for {key}")
        tr = self._trackers[key]
        oracle_count = len(self._oracle_audit_calls.get(cell_id, set()))
        if len(tr.screen_calls) != self.frozen_budget.expected_screen_calls:
            raise BudgetViolationError(
                f"{key} screen calls={len(tr.screen_calls)} != expected={self.frozen_budget.expected_screen_calls}"
            )
        if len(tr.val_calls) != self.frozen_budget.expected_validation_calls:
            raise BudgetViolationError(
                f"{key} validation calls={len(tr.val_calls)} != expected={self.frozen_budget.expected_validation_calls}"
            )
        return MethodBudgetSpec(
            method_id=method_id,
            cell_id=cell_id,
            screen_configs=len(tr.screen_runs),
            screen_items=len(tr.screen_items),
            screen_repeats=len(tr.screen_repeats),
            validation_candidates=len(tr.val_runs),
            validation_items=len(tr.val_items),
            validation_envs=len(tr.val_envs),
            validation_repeats=len(tr.val_repeats),
            oracle_audit_calls=oracle_count,
        )

    def assert_equal_operational_budgets(
        self,
        cell_id: str,
        required_method_trajectories: Iterable[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], MethodBudgetSpec]:
        """Verify byte-exact operational budget parity across all primary method trajectories in a cell."""
        specs: Dict[Tuple[str, str], MethodBudgetSpec] = {}
        expected_sig = (
            self.frozen_budget.screen_configs,
            self.frozen_budget.screen_items,
            self.frozen_budget.screen_repeats,
            self.frozen_budget.validation_candidates,
            self.frozen_budget.validation_items,
            len(self.frozen_budget.validation_env_ids),
            self.frozen_budget.validation_repeats,
        )
        for method_id, traj_id in required_method_trajectories:
            spec = self.to_budget_spec(cell_id, method_id, traj_id)
            if spec.operational_signature() != expected_sig:
                raise BudgetViolationError(
                    f"Operational budget mismatch for {method_id}[{traj_id}] in {cell_id}: "
                    f"got {spec.operational_signature()}, expected {expected_sig}"
                )
            specs[(method_id, traj_id)] = spec
        return specs
