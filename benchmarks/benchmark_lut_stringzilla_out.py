# ruff: noqa: E402, INP001
"""Compare the previous copy-then-translate route with Albucore's public LUT output route."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for name in THREAD_ENV:
    os.environ[name] = "1"

import numkong as nk
import numpy as np
import stringzilla as sz

from albucore import apply_uint8_lut


def _old_route(image: np.ndarray, table: np.ndarray, out: np.ndarray) -> np.ndarray:
    np.copyto(out, image)
    sz.translate(memoryview(out), memoryview(table), inplace=True)
    return out


def _measure_cell(  # noqa: PLR0913, PLR0917
    image: np.ndarray,
    table: np.ndarray,
    out: np.ndarray,
    repeats: int,
    warmup: int,
    seed: int,
) -> dict[str, object]:
    def old() -> np.ndarray:
        return _old_route(image, table, out)

    def new() -> np.ndarray:
        return apply_uint8_lut(image, table, out=out)

    old_samples: list[float] = []
    new_samples: list[float] = []
    order_rng = np.random.default_rng(seed)
    for _ in range(warmup):
        order = [("old", old), ("new", new)]
        if order_rng.integers(2):
            order.reverse()
        for _, route in order:
            route()

    for _ in range(repeats):
        order = [("old", old), ("new", new)]
        if order_rng.integers(2):
            order.reverse()
        sample: dict[str, float] = {}
        for name, route in order:
            started = time.perf_counter_ns()
            route()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            sample[name] = elapsed_ms
        old_samples.append(sample["old"])
        new_samples.append(sample["new"])

    ratios = [old_time / new_time for old_time, new_time in zip(old_samples, new_samples, strict=True)]
    return {
        "repeats": repeats,
        "old_median_ms": statistics.median(old_samples),
        "new_median_ms": statistics.median(new_samples),
        "paired_old_over_new_median": statistics.median(ratios),
        "old_mad_ms": statistics.median(abs(x - statistics.median(old_samples)) for x in old_samples),
        "new_mad_ms": statistics.median(abs(x - statistics.median(new_samples)) for x in new_samples),
        "exact_output": True,
        "input_preserved": True,
        "old_samples_ms": old_samples,
        "new_samples_ms": new_samples,
    }


def run(output: Path, repeats: int = 51, warmup: int = 5, smoke: bool = False) -> dict[str, object]:
    shapes = ((128, 160),) if smoke else ((128, 160), (240, 320), (480, 640), (768, 1024))
    channels_grid = (1,) if smoke else (1, 3, 9)
    if smoke:
        repeats, warmup = 3, 1

    table = np.random.default_rng(71).permutation(256).astype(np.uint8)
    rng = np.random.default_rng(122)
    cells: list[dict[str, object]] = []
    for height, width in shapes:
        for channels in channels_grid:
            image = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
            original = image.copy()
            expected = table[image]
            out = np.empty_like(image)
            np.testing.assert_array_equal(_old_route(image, table, out), expected)
            np.testing.assert_array_equal(apply_uint8_lut(image, table, out=out), expected)

            result = _measure_cell(image, table, out, repeats, warmup, height + channels)
            np.testing.assert_array_equal(image, original)
            cells.append(
                {
                    "shape": [height, width, channels],
                    "dtype": "uint8",
                    "output": "separate contiguous array",
                    **result,
                },
            )

    ratios = [float(cell["paired_old_over_new_median"]) for cell in cells]
    report = {
        "metadata": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numkong": nk.__version__,
            "stringzilla": sz.__version__,
            "threads": 1,
            "environment_threads": {name: os.environ[name] for name in THREAD_ENV},
            "warmup": warmup,
            "comparison": "copy-then-inplace StringZilla vs Albucore public apply_uint8_lut with translate(out=)",
        },
        "summary": {
            "cells": len(cells),
            "median_paired_old_over_new": statistics.median(ratios),
            "wins_over_5_percent": sum(ratio > 1.05 for ratio in ratios),
            "regressions_over_5_percent": sum(ratio < 0.95 for ratio in ratios),
        },
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.repeats, arguments.warmup, arguments.smoke)
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2))  # noqa: T201
