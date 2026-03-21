"""Shared timing helpers for standalone benchmark scripts (run from repo root)."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def median_ms(fn: Callable[[], object], repeats: int = 11, warmup: int = 3) -> float:
    """Median wall time in milliseconds over ``repeats`` runs after ``warmup`` discards."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))
