"""Unit tests for the four-candidate robust PromptPrism selector (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from research.real_v4.selection import (
    FACTOR_IDS,
    PromptPrismRobustSelector,
    full_cube_configs,
    levels_to_run_id,
    resolution_iv_screen_run_ids,
    run_id_to_levels,
)


def _eval_surface_on_runs(
    run_ids, n_items: int, fn, noise_sd: float = 0.0, seed: int = 42
):
    rng = np.random.default_rng(seed)
    out = {}
    for rid in run_ids:
        lv = run_id_to_levels(rid)
        mu = fn(lv)
        eps = (
            rng.normal(0.0, noise_sd, size=n_items)
            if noise_sd > 0
            else np.zeros(n_items)
        )
        out[rid] = np.clip(mu + eps, 0.0, 1.0)
    return out


def test_resolution_iv_geometry_and_bijective_run_ids():
    cube = full_cube_configs()
    assert len(cube) == 64
    for rid, lv in cube.items():
        assert levels_to_run_id(lv) == rid
        assert run_id_to_levels(rid) == lv

    screen_ids = resolution_iv_screen_run_ids()
    assert len(screen_ids) == 16
    assert 0 in screen_ids
    assert 63 in screen_ids
    # Check orthogonality of main effects on the 16 runs
    x_pm = np.array(
        [[2 * cube[r][f] - 1 for f in FACTOR_IDS] for r in screen_ids], dtype=float
    )
    gram = x_pm.T @ x_pm
    assert np.allclose(gram, 16.0 * np.eye(6))


def test_additive_zero_effect_deduplication_and_deterministic_replay():
    selector_dedup = PromptPrismRobustSelector(pad_to_max=False, n_bootstrap_screen=200)
    selector_padded = PromptPrismRobustSelector(pad_to_max=True, n_bootstrap_screen=200)
    screen_ids = selector_padded.screen_run_ids()

    # 1. Pure zero/constant response surface -> sign=0, screen_best=0, conservative=0, weakest flips A -> run 32!
    flat_scores = _eval_surface_on_runs(screen_ids, 60, lambda _lv: 0.50, noise_sd=0.0)
    cands_dedup = selector_dedup.generate_candidates(flat_scores, seed=101)
    assert [c.run_id for c in cands_dedup] == [0, 32]
    assert len({c.run_id for c in cands_dedup}) == len(cands_dedup) <= 4

    cands_pad1 = selector_padded.generate_candidates(flat_scores, seed=101)
    cands_pad2 = selector_padded.generate_candidates(flat_scores, seed=101)
    assert cands_pad1 == cands_pad2
    assert len(cands_pad1) == 4
    assert len({c.run_id for c in cands_pad1}) == 4

    # 2. Unobserved additive optimum: A=+1, B=+1, C=+1, D=+1, E=+1, F=-1 (run_id = 62, since E=ABC=+1, F=BCD=+1 is 63, so 62 is NOT in screen!)
    assert 62 not in screen_ids

    def additive_fn(lv):
        return (
            0.40
            + 0.08 * lv["A"]
            + 0.07 * lv["B"]
            + 0.06 * lv["C"]
            + 0.05 * lv["D"]
            + 0.04 * lv["E"]
            - 0.03 * lv["F"]
        )

    add_screen = _eval_surface_on_runs(
        screen_ids, 80, additive_fn, noise_sd=0.01, seed=202
    )
    cands_add = selector_padded.generate_candidates(add_screen, seed=202)
    assert 62 in [c.run_id for c in cands_add]

    val_envs = {
        "ref": _eval_surface_on_runs(
            [c.run_id for c in cands_add], 80, additive_fn, noise_sd=0.005, seed=301
        ),
        "fmt_whitespace": _eval_surface_on_runs(
            [c.run_id for c in cands_add],
            80,
            lambda lv: additive_fn(lv) - 0.002,
            noise_sd=0.005,
            seed=302,
        ),
    }
    winner1, score1, _ = selector_padded.select_final_prompt(
        cands_add, val_envs, seed=404
    )
    winner2, score2, _ = selector_padded.select_final_prompt(
        cands_add, val_envs, seed=404
    )
    assert winner1.run_id == 62
    assert winner2.run_id == 62
    assert np.isclose(score1.robust_lower_bound, score2.robust_lower_bound)


def test_interaction_dominated_tie_rule_lower_is_better_and_leakage_rejection():
    selector = PromptPrismRobustSelector(
        pad_to_max=True, n_bootstrap_screen=200, n_bootstrap_validation=200
    )
    screen_ids = selector.screen_run_ids()

    # 1. Interaction-dominated surface where sign_composed fails (epistasis: A=1 alone +0.04, B=1 alone +0.04, A=1&B=1 crashes -0.25!)
    def epistasis_fn(lv):
        if lv["A"] == 1 and lv["B"] == 1:
            return 0.25
        return 0.60 + 0.12 * lv["A"] + 0.11 * lv["B"]

    epi_screen = _eval_surface_on_runs(
        screen_ids, 80, epistasis_fn, noise_sd=0.005, seed=11
    )
    cands_epi = selector.generate_candidates(epi_screen, seed=11)
    # Validation catches the crash of A=1&B=1 and picks the high-performing A=1, B=0 candidate!
    epi_val = {
        "ref": _eval_surface_on_runs(
            [c.run_id for c in cands_epi], 60, epistasis_fn, noise_sd=0.0, seed=12
        ),
        "word_paraphrase_1": _eval_surface_on_runs(
            [c.run_id for c in cands_epi], 60, epistasis_fn, noise_sd=0.0, seed=13
        ),
    }
    winner_epi, w_score, _ = selector.select_final_prompt(cands_epi, epi_val, seed=14)
    assert not (
        winner_epi.factor_levels["A"] == 1 and winner_epi.factor_levels["B"] == 1
    )
    assert w_score.worst_env_mean >= 0.70

    # 2. Exact validation tie breaks deterministically to lowest run_id
    tied_val = {
        "ref": {c.run_id: np.full(40, 0.80) for c in cands_epi},
        "word_paraphrase_1": {c.run_id: np.full(40, 0.79) for c in cands_epi},
    }
    winner_tie, _, _ = selector.select_final_prompt(cands_epi, tied_val, seed=15)
    assert winner_tie.run_id == min(c.run_id for c in cands_epi)

    # 3. Lower-is-better metric (e.g., error rate where 0.10 beats 0.40):
    def loss_fn(lv):
        return (
            0.50 - 0.15 * lv["A"] - 0.10 * lv["D"]
        )  # lowest loss = 0.25 when A=1, D=1

    loss_screen = _eval_surface_on_runs(screen_ids, 60, loss_fn, noise_sd=0.0, seed=21)
    cands_loss = selector.generate_candidates(
        loss_screen, higher_is_better=False, seed=21
    )
    loss_val = {
        "ref": _eval_surface_on_runs(
            [c.run_id for c in cands_loss], 60, loss_fn, noise_sd=0.0, seed=22
        ),
    }
    winner_loss, _, _ = selector.select_final_prompt(
        cands_loss, loss_val, higher_is_better=False, seed=23
    )
    assert winner_loss.factor_levels["A"] == 1
    assert winner_loss.factor_levels["D"] == 1

    # 4. Confirmation partition leakage raises PermissionError immediately
    with pytest.raises(PermissionError, match="screen partition"):
        selector.generate_candidates(epi_screen, partition_tag="clean_confirmation")
    with pytest.raises(PermissionError, match="Confirmation leakage"):
        selector.select_final_prompt(
            cands_epi, epi_val, partition_tag="clean_confirmation"
        )
