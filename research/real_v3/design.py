"""Deterministic factorial and comparator designs for real-v3."""

from __future__ import annotations

import hashlib
import itertools
import random
from typing import Dict, List

import numpy as np


FACTOR_IDS = ["A", "B", "C", "D", "E", "F"]


def stable_seed(label: str, base: int = 42) -> int:
    """Return a process-independent nonnegative 31-bit seed."""
    digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def all_factorial_configs() -> Dict[int, Dict[str, int]]:
    """Map stable run IDs 1..64 to the full six-factor cube."""
    return {
        run_id: dict(zip(FACTOR_IDS, levels))
        for run_id, levels in enumerate(itertools.product([-1, 1], repeat=6), start=1)
    }


def resolution_iv_run_ids(configs: Dict[int, Dict[str, int]] | None = None) -> List[int]:
    """Return the registered 2^(6-2) fraction with E=ABC and F=BCD."""
    cube = configs or all_factorial_configs()
    return [
        run_id for run_id, row in cube.items()
        if row["E"] == row["A"] * row["B"] * row["C"]
        and row["F"] == row["B"] * row["C"] * row["D"]
    ]


def balanced_random_designs(
    configs: Dict[int, Dict[str, int]],
    count: int,
    seed: int,
) -> List[List[int]]:
    """Generate unique 16-run random designs balanced on every main factor."""
    rng = random.Random(seed)
    run_ids = sorted(configs)
    accepted: List[List[int]] = []
    seen = set()
    attempts = 0
    while len(accepted) < count:
        attempts += 1
        if attempts > 1_000_000:
            raise RuntimeError("Could not generate enough balanced random designs")
        candidate = sorted(rng.sample(run_ids, 16))
        key = tuple(candidate)
        if key in seen:
            continue
        matrix = np.array([[configs[r][f] for f in FACTOR_IDS] for r in candidate])
        if not np.all(matrix.sum(axis=0) == 0):
            continue
        design = np.column_stack([np.ones(len(matrix)), matrix])
        if np.linalg.matrix_rank(design) != len(FACTOR_IDS) + 1:
            continue
        seen.add(key)
        accepted.append(candidate)
    return accepted
