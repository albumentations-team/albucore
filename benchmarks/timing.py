"""Shared timing helpers for standalone benchmark scripts (run from repo root)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WallTimingMs:
    """Wall-clock timings in milliseconds after warmup."""

    median: float
    mean: float
    std: float
    """Sample standard deviation of timed runs (``ddof=1``); 0 if ``n < 2``."""
    mad: float
    """Median absolute deviation from the median (robust spread)."""
    n: int


def bench_wall_ms(fn: Callable[[], object], repeats: int = 11, warmup: int = 3) -> WallTimingMs:
    """Wall time in ms: ``warmup`` untimed runs, then ``repeats`` timed runs.

    Uses ``time.perf_counter()`` (monotonic, best effort for microbenchmarks).
    """
    if repeats < 1:
        msg = f"repeats must be >= 1, got {repeats}"
        raise ValueError(msg)
    if warmup < 0:
        msg = f"warmup must be >= 0, got {warmup}"
        raise ValueError(msg)

    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    med = float(np.median(arr))
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if repeats > 1 else 0.0
    mad = float(np.median(np.abs(arr - med)))
    return WallTimingMs(median=med, mean=mean, std=std, mad=mad, n=repeats)


def median_ms(fn: Callable[[], object], repeats: int = 11, warmup: int = 3) -> float:
    """Median wall time in milliseconds over ``repeats`` runs after ``warmup`` discards."""
    return bench_wall_ms(fn, repeats=repeats, warmup=warmup).median
