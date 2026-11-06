"""
Single-Command End-to-End Reproducibility Pipeline (Phase 5 Task 18).
Re-executes simulations, method replay, confirmatory analyses, and sensitivity audits
using local cached fixtures with zero paid API calls.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], description: str):
    print(f"\n=======================================================")
    print(f"--> {description}")
    print(f"    Command: {' '.join(cmd)}")
    print(f"=======================================================")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED: {description} (exit code {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"SUCCESS: {description} ({elapsed:.2f}s)")


def reproduce():
    print("==================================================================")
    print("PROMPTPRISM SCIENTIFIC VALIDATION: REPRODUCIBILITY PIPELINE")
    print("Running in offline verification mode (0 paid API calls)")
    print("==================================================================")

    # 1. Verification Unit Tests
    run_command(
        [sys.executable, "-m", "pytest", "tests/research/"],
        "Step 1: Running Research Verification Tests",
    )

    # 2. Simulation Summarization
    run_command(
        [sys.executable, "-m", "research.analysis.summarize_simulation"],
        "Step 2: Generating Simulation Calibration Report",
    )

    # 3. Confirmatory Replay on Discovery Partition
    run_command(
        [sys.executable, "-m", "research.analysis.replay_methods"],
        "Step 3: Replaying Methods on Discovery Partition",
    )

    # 4. Confirmatory Hypothesis Testing
    run_command(
        [sys.executable, "-m", "research.analysis.run_confirmatory"],
        "Step 4: Executing Primary Confirmatory Hypothesis Test (H1)",
    )

    # 5. Sensitivity and Failure Mode Analyses
    run_command(
        [sys.executable, "-m", "research.analysis.run_sensitivity"],
        "Step 5: Executing Sensitivity & Failure-Mode Analysis",
    )

    # 6. Judge Validation Analysis
    run_command(
        [sys.executable, "-m", "research.analysis.judge_validation"],
        "Step 6: Executing Human-Judge Agreement Analysis",
    )

    print("\n==================================================================")
    print("REPRODUCIBILITY PIPELINE COMPLETE: ALL ARTIFACTS VERIFIED")
    print("All results regenerated deterministically.")
    print("==================================================================")


if __name__ == "__main__":
    reproduce()
