"""Fixed-seed replay, gate verification, and boundary hand-fixture tests for Task 7 calibration."""

from __future__ import annotations

import json

from research.real_v4.decisions import DecisionThresholds
from research.real_v4.simulation import (
    CALIBRATION_SUMMARY_PATH,
    make_calibration_envelope,
    run_full_calibration,
    run_single_adaptive_trial,
)


def test_calibration_summary_meets_all_scientific_feasibility_gates():
    assert CALIBRATION_SUMMARY_PATH.exists(), "Run research.real_v4.simulation first"
    summary = json.loads(CALIBRATION_SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["all_scientific_feasibility_gates_passed"] is True
    gates = summary["gate_results"]
    assert gates["null_claim_p_type1_ok"] is True
    assert gates["null_claim_s_boundary_type1_ok"] is True
    assert gates["null_simultaneous_coverage_ok"] is True
    assert gates["beneficial_claim_p_power_ok"] is True
    assert gates["beneficial_claim_s_power_ok"] is True
    assert gates["provider_drift_fails_closed_100pct"] is True
    assert gates["harmful_regime_rejected_100pct"] is True

    # Verify all required regimes are documented in the machine-readable report
    regimes = summary["regimes"]
    for req in (
        "null_flat",
        "beneficial_structured",
        "smooth_additive_bo_parity",
        "harmful_antagonistic",
        "interaction_heavy",
        "heterogeneous_cells",
        "provider_drift",
    ):
        assert req in regimes
        assert regimes[req]["n_trials"] >= 40


def test_fixed_seed_single_trial_deterministic_replay_and_drift_boundary():
    env = make_calibration_envelope()
    th = DecisionThresholds(
        alpha=0.05,
        delta_select=0.01,
        epsilon_stability=0.02,
        n_bootstrap=150,
        seed=777,
    )
    r1 = run_single_adaptive_trial(
        "beneficial_structured",
        envelope=env,
        trial_seed=20261118,
        n_screen=60,
        n_val=30,
        n_conf=100,
        n_holdout=100,
        num_baseline_trajectories=5,
        n_bootstrap_screen=100,
        n_bootstrap_validation=100,
        n_bootstrap_decision=150,
        thresholds=th,
        verify_budget_ledger=True,
    )
    r2 = run_single_adaptive_trial(
        "beneficial_structured",
        envelope=env,
        trial_seed=20261118,
        n_screen=60,
        n_val=30,
        n_conf=100,
        n_holdout=100,
        num_baseline_trajectories=5,
        n_bootstrap_screen=100,
        n_bootstrap_validation=100,
        n_bootstrap_decision=150,
        thresholds=th,
        verify_budget_ledger=False,
    )
    assert r1 == r2

    # Provider drift must fail closed with INVALID/OUT_OF_ENVELOPE
    r_drift = run_single_adaptive_trial(
        "provider_drift",
        envelope=env,
        trial_seed=20261118,
        n_screen=60,
        n_val=30,
        n_conf=100,
        n_holdout=100,
        num_baseline_trajectories=5,
        n_bootstrap_screen=100,
        n_bootstrap_validation=100,
        n_bootstrap_decision=150,
        thresholds=th,
    )
    assert r_drift["certificate_status"] == "INVALID/OUT_OF_ENVELOPE"
    assert r_drift["claim_s_pass"] is False
    assert r_drift["overall_pass"] is False


def test_mini_calibration_config_fixed_seed_reproducibility(tmp_path):
    mini_yaml = tmp_path / "mini_calib.yaml"
    mini_yaml.write_text(
        """
schema_version: "1.0.0"
study_id: "prompt-prism-real-v4-selection-stability-2026"
seed: 12345
sizing_envelope:
  n_tasks: 6
  n_models: 3
  n_cells: 18
  pilot_items_per_task: 10
  screen_items_per_task: 30
  validation_items_per_task: 20
  clean_confirmation_items_per_task: 40
  stability_holdout_items_per_task: 40
  items_per_source_group: 2
  screen_configs: 16
  validation_candidates: 4
  validation_env_ids: ["ref", "fmt_whitespace", "word_paraphrase_1"]
  repeat_count_when_nondeterministic: 3
decision_thresholds:
  alpha: 0.05
  delta_select: 0.01
  epsilon_stability: 0.02
  cell_majority_threshold: 12
operating_characteristic_gates:
  max_type1_claim_p: 0.50
  max_type1_claim_s_boundary: 0.50
  min_power_claim_p_at_design_alt: 0.50
  min_power_claim_s_at_design_alt: 0.50
  min_simultaneous_stability_coverage: 0.50
simulation_settings:
  num_baseline_trajectories: 4
  n_bootstrap_screen: 100
  n_bootstrap_validation: 100
  n_bootstrap_decision: 100
  regimes:
    null_flat:
      n_trials: 2
      description: "mini null"
    beneficial_structured:
      n_trials: 2
      description: "mini ben"
    harmful_antagonistic:
      n_trials: 1
      description: "mini harm"
    provider_drift:
      n_trials: 1
      description: "mini drift"
""",
        encoding="utf-8",
    )
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    s1 = run_full_calibration(config_path=mini_yaml, output_path=out1)
    s2 = run_full_calibration(config_path=mini_yaml, output_path=out2)
    assert s1["summary_sha256"] == s2["summary_sha256"]
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
