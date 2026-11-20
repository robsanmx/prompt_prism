"""Unit and budget-parity tests for equal-budget Random and Bayesian baselines (Task 5)."""

from __future__ import annotations

import numpy as np
import pytest

from research.real_v4.baselines import (
    BalancedRandomTwoStageBaseline,
    SequentialBayesianOptBaseline,
    ofat_reference_run_ids,
    untuned_reference_run_id,
)
from research.real_v4.budget import (
    BudgetLedger,
    BudgetViolationError,
    FrozenOperationalBudget,
)
from research.real_v4.selection import (
    PromptPrismRobustSelector,
    run_id_to_levels,
)


def test_equal_budget_parity_deterministic_replay_and_full_trajectory_retention():
    frozen_budget = FrozenOperationalBudget(
        screen_configs=16,
        screen_items=40,
        screen_repeats=1,
        validation_candidates=4,
        validation_items=30,
        validation_env_ids=("ref", "fmt_whitespace", "word_paraphrase_1"),
        validation_repeats=1,
    )
    ledger = BudgetLedger(frozen_budget)
    cell_id = "gsm8k_math::gemini-2.5-flash"

    screen_item_ids = [f"scr_{i:03d}" for i in range(40)]
    val_item_ids = [f"val_{i:03d}" for i in range(30)]
    val_envs = frozen_budget.validation_env_ids

    rng_cube = np.random.default_rng(77)
    screen_cube = {}
    val_cube = {e: {} for e in val_envs}
    for rid in range(64):
        lv = run_id_to_levels(rid)
        mu = 0.50 + 0.06 * lv["A"] + 0.05 * lv["C"] + 0.04 * lv["D"] - 0.02 * lv["F"]
        screen_cube[rid] = np.clip(mu + rng_cube.normal(0.0, 0.05, size=40), 0.0, 1.0)
        for e_idx, env_id in enumerate(val_envs):
            drop = 0.0 if env_id == "ref" else (0.002 * (e_idx + 1))
            val_cube[env_id][rid] = np.clip(
                mu - drop + rng_cube.normal(0.0, 0.03, size=30), 0.0, 1.0
            )

    # Record 64-run research-only oracle audit (must NOT affect operational method budgets)
    for rid in range(64):
        ledger.record_oracle_audit_batch(
            cell_id=cell_id, run_id=rid, item_ids=screen_item_ids
        )

    # 1. Run PromptPrism under the same ledger
    prism = PromptPrismRobustSelector(
        pad_to_max=True, n_bootstrap_screen=200, n_bootstrap_validation=200
    )
    prism_screen = {}
    for rid in prism.screen_run_ids():
        prism_screen[rid] = screen_cube[rid]
        ledger.record_screen_batch(
            cell_id=cell_id,
            method_id="promptprism",
            trajectory_id="primary",
            run_id=rid,
            item_ids=screen_item_ids,
        )
    prism_cands = prism.generate_candidates(prism_screen, seed=101)
    prism_val = {e: {} for e in val_envs}
    for cand in prism_cands:
        for env_id in val_envs:
            prism_val[env_id][cand.run_id] = val_cube[env_id][cand.run_id]
            ledger.record_validation_batch(
                cell_id=cell_id,
                method_id="promptprism",
                trajectory_id="primary",
                run_id=cand.run_id,
                env_id=env_id,
                item_ids=val_item_ids,
            )
    prism_winner, _, _ = prism.select_final_prompt(prism_cands, prism_val, seed=102)

    # 2. Run all 20 frozen Balanced Random designs
    rand_base = BalancedRandomTwoStageBaseline(
        num_designs=20, n_bootstrap_validation=200
    )
    rand_trajs = rand_base.run_all_trajectories(
        cell_id=cell_id,
        screen_evaluator=lambda rid: screen_cube[rid],
        validation_evaluator=lambda rid, env_id: val_cube[env_id][rid],
        screen_item_ids=screen_item_ids,
        validation_item_ids=val_item_ids,
        validation_env_ids=val_envs,
        ledger=ledger,
    )
    assert len(rand_trajs) == 20  # Full distribution retained, zero post-hoc selection

    # 3. Run all 20 frozen Sequential Bayesian Optimization trajectories
    bo_base = SequentialBayesianOptBaseline(
        num_trajectories=20, n_bootstrap_validation=200
    )
    bo_trajs = bo_base.run_all_trajectories(
        cell_id=cell_id,
        screen_evaluator=lambda rid: screen_cube[rid],
        validation_evaluator=lambda rid, env_id: val_cube[env_id][rid],
        screen_item_ids=screen_item_ids,
        validation_item_ids=val_item_ids,
        validation_env_ids=val_envs,
        ledger=ledger,
    )
    assert len(bo_trajs) == 20

    # Assert exact operational budget parity across PromptPrism + 20 Random + 20 BO trajectories
    required = (
        [("promptprism", "primary")]
        + [("balanced_random", f"design_{i:02d}") for i in range(20)]
        + [("bayesian_opt", f"bo_seed_{i:02d}") for i in range(20)]
    )
    specs = ledger.assert_equal_operational_budgets(cell_id, required)
    for spec in specs.values():
        assert spec.total_screen_calls == 16 * 40
        assert spec.total_validation_calls == 4 * 30 * 3
        assert spec.total_operational_discovery_calls == 640 + 360
        assert spec.oracle_audit_calls == 64 * 40

    # Deterministic replay reproduces identical BO and Random trajectories & winners
    rand_replay = rand_base.run_all_trajectories(
        cell_id=cell_id,
        screen_evaluator=lambda rid: screen_cube[rid],
        validation_evaluator=lambda rid, env_id: val_cube[env_id][rid],
        screen_item_ids=screen_item_ids,
        validation_item_ids=val_item_ids,
        validation_env_ids=val_envs,
        ledger=None,
    )
    bo_replay = bo_base.run_all_trajectories(
        cell_id=cell_id,
        screen_evaluator=lambda rid: screen_cube[rid],
        validation_evaluator=lambda rid, env_id: val_cube[env_id][rid],
        screen_item_ids=screen_item_ids,
        validation_item_ids=val_item_ids,
        validation_env_ids=val_envs,
        ledger=None,
    )
    assert [t.selected_candidate.run_id for t in rand_replay] == [
        t.selected_candidate.run_id for t in rand_trajs
    ]
    assert [t.screen_run_ids for t in bo_replay] == [t.screen_run_ids for t in bo_trajs]
    assert [t.selected_candidate.run_id for t in bo_replay] == [
        t.selected_candidate.run_id for t in bo_trajs
    ]
    assert prism_winner.run_id >= 0
    assert len(ofat_reference_run_ids()) == 7
    assert untuned_reference_run_id() == 0


def test_budget_ledger_fails_on_single_extra_call_item_repeat_or_candidate():
    budget = FrozenOperationalBudget(
        screen_configs=16,
        screen_items=10,
        screen_repeats=1,
        validation_candidates=4,
        validation_items=8,
        validation_env_ids=("ref", "fmt_whitespace"),
        validation_repeats=1,
    )
    ledger = BudgetLedger(budget)
    cell = "c1"
    s_items = [f"s{i}" for i in range(10)]
    v_items = [f"v{i}" for i in range(8)]

    for rid in range(16):
        ledger.record_screen_batch(
            cell_id=cell,
            method_id="m",
            trajectory_id="t0",
            run_id=rid,
            item_ids=s_items,
        )
    for rid in range(4):
        for env_id in ("ref", "fmt_whitespace"):
            ledger.record_validation_batch(
                cell_id=cell,
                method_id="m",
                trajectory_id="t0",
                run_id=rid,
                env_id=env_id,
                item_ids=v_items,
            )

    # 1. Single duplicate call fails
    with pytest.raises(BudgetViolationError, match="Duplicate screen call"):
        ledger.record_screen_batch(
            cell_id=cell,
            method_id="m",
            trajectory_id="t0",
            run_id=0,
            item_ids=[s_items[0]],
        )

    # 2. Single extra candidate (5th validation candidate) fails
    with pytest.raises(BudgetViolationError, match="validation_candidates"):
        ledger.record_validation_batch(
            cell_id=cell,
            method_id="m",
            trajectory_id="t0",
            run_id=5,
            env_id="ref",
            item_ids=[v_items[0]],
        )

    # 3. Single extra screen config (17th run) fails
    with pytest.raises(BudgetViolationError, match="screen_configs"):
        ledger.record_screen_batch(
            cell_id=cell,
            method_id="m",
            trajectory_id="t0",
            run_id=16,
            item_ids=[s_items[0]],
        )

    # 4. Single extra screen item fails
    with pytest.raises(BudgetViolationError, match="screen_items"):
        ledger.record_screen_batch(
            cell_id=cell,
            method_id="m",
            trajectory_id="t0",
            run_id=0,
            item_ids=["s_extra"],
        )

    # 5. Single extra repeat fails
    with pytest.raises(BudgetViolationError, match="screen_repeats"):
        ledger.record_screen_batch(
            cell_id=cell,
            method_id="m",
            trajectory_id="t0",
            run_id=0,
            item_ids=[s_items[0]],
            repeat_idx=1,
        )

    # 6. Incomplete budget fails when converting to MethodBudgetSpec
    ledger.record_screen_batch(
        cell_id=cell,
        method_id="incomplete",
        trajectory_id="t0",
        run_id=0,
        item_ids=s_items,
    )
    with pytest.raises(BudgetViolationError, match="screen calls"):
        ledger.to_budget_spec(cell, "incomplete", "t0")
