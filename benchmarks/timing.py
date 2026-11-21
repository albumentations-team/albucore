"""Shared timing helpers for standalone benchmark scripts (run from repo root)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class WallTimingMs:
    """Wall-clock timings in milliseconds after warmup."""

    raw: tuple[float, ...]
    """Individual timed samples in milliseconds after warmup."""
    median: float
    mean: float
    std: float
    """Sample standard deviation of timed runs (``ddof=1``); 0 if ``n < 2``."""
    mad: float
    """Median absolute deviation from the median (robust spread)."""
    n: int


def _summarize_wall_ms(samples: list[float]) -> WallTimingMs:
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return WallTimingMs(
        raw=tuple(samples),
        median=median,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(samples) > 1 else 0.0,
        mad=float(np.median(np.abs(values - median))),
        n=len(samples),
    )


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

    return _summarize_wall_ms(times)


def bench_paired_wall_ms(
    first: Callable[[], object],
    second: Callable[[], object],
    repeats: int = 11,
    warmup: int = 3,
    *,
    first_starts: bool = True,
) -> tuple[WallTimingMs, WallTimingMs]:
    """Time two routes in alternating order, returning a timing summary for each."""
    if repeats < 1:
        msg = f"repeats must be >= 1, got {repeats}"
        raise ValueError(msg)
    if warmup < 0:
        msg = f"warmup must be >= 0, got {warmup}"
        raise ValueError(msg)

    calls = (first, second)
    samples: tuple[list[float], list[float]] = ([], [])
    for sample in range(warmup + repeats):
        order = (0, 1) if (sample % 2 == 0) == first_starts else (1, 0)
        for index in order:
            started = time.perf_counter_ns()
            calls[index]()
            if sample >= warmup:
                samples[index].append((time.perf_counter_ns() - started) / 1e6)
    return _summarize_wall_ms(samples[0]), _summarize_wall_ms(samples[1])


def median_ms(fn: Callable[[], object], repeats: int = 11, warmup: int = 3) -> float:
    """Median wall time in milliseconds over ``repeats`` runs after ``warmup`` discards."""
    return bench_wall_ms(fn, repeats=repeats, warmup=warmup).median


def _check_resume_profile(previous: dict[str, Any], current: dict[str, Any]) -> None:
    for key in sorted((previous.keys() | current.keys()) - {"case_list"}):
        if key not in previous or key not in current or previous[key] != current[key]:
            raise RuntimeError(f"Output belongs to a different benchmark {key}")
