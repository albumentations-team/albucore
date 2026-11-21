"""Paired global float32 statistics benchmark for NumPy and OpenCV."""

# ruff: noqa: E402, INP001
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from functools import partial
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_output import output_path, write_json_atomic

from albucore.stats import DEFAULT_EPS, mean_std, std

RELATIVE_VARIANCE_FLOOR = 1e-4


def numpy_mean_std(image: np.ndarray) -> tuple[np.float64, np.float64]:
    mean_keepdims = np.asarray(np.mean(image, dtype=np.float64, keepdims=True), dtype=np.float64)
    mean_value = np.float64(mean_keepdims.reshape(())[()])
    std_value = np.float64(np.std(image, dtype=np.float64, mean=mean_keepdims)) + DEFAULT_EPS
    return mean_value, std_value


def numpy_std(image: np.ndarray) -> np.float64:
    return np.float64(np.std(image, dtype=np.float64)) + DEFAULT_EPS


def cv2_candidate(image: np.ndarray) -> tuple[float, float, bool]:
    mean_array, std_array = cv2.meanStdDev(image.reshape(-1, 1))
    mean_value = float(mean_array[0, 0])
    std_value = float(std_array[0, 0])
    fallback = (
        not math.isfinite(mean_value)
        or not math.isfinite(std_value)
        or std_value * std_value <= RELATIVE_VARIANCE_FLOOR * mean_value * mean_value
    )
    if fallback and std_value == 0 and np.ptp(image) == 0:
        fallback = False
    if fallback:
        mean_keepdims = np.asarray(np.mean(image, dtype=np.float64, keepdims=True), dtype=np.float64)
        mean_value = float(mean_keepdims.reshape(())[()])
        std_value = float(np.std(image, dtype=np.float64, mean=mean_keepdims))
    return mean_value, std_value + DEFAULT_EPS, fallback


def paired_times(
    base: Callable[[], object],
    candidate: Callable[[], object],
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    samples: tuple[list[float], list[float]] = ([], [])
    calls: tuple[Callable[[], object], Callable[[], object]] = (base, candidate)
    for sample in range(warmup + repeats):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            started = perf_counter_ns()
            calls[index]()
            elapsed_ms = (perf_counter_ns() - started) / 1e6
            if sample >= warmup:
                samples[index].append(elapsed_ms)
    return statistics.median(samples[0]), statistics.median(samples[1])


def check_values(image: np.ndarray) -> dict[str, object]:
    reference = (float(np.mean(image, dtype=np.float64)), float(np.std(image, dtype=np.float64)) + DEFAULT_EPS)
    actual_mean, actual_std, fallback = cv2_candidate(image)
    public_mean, public_std = mean_std(image, "global")
    public_mean, public_std = float(public_mean), float(public_std)
    public_std_only = float(std(image, "global"))
    for label, values in (
        ("cv2", (actual_mean, actual_std)),
        ("public", (public_mean, public_std)),
    ):
        if not math.isclose(values[0], reference[0], rel_tol=5e-10, abs_tol=1e-12):
            raise AssertionError(f"{label} mean mismatch: {values[0]} != {reference[0]}")
        if not math.isclose(values[1], reference[1], rel_tol=5e-10, abs_tol=1e-12):
            raise AssertionError(f"{label} std mismatch: {values[1]} != {reference[1]}")
    if not math.isclose(public_std_only, reference[1], rel_tol=5e-10, abs_tol=1e-12):
        raise AssertionError(f"public std mismatch: {public_std_only} != {reference[1]}")
    return {
        "shape": list(image.shape),
        "reference_mean": reference[0],
        "reference_std": reference[1],
        "cv2_fallback": fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)

    rng = np.random.default_rng(8043)
    rows = []
    for shape in (
        (1, 2, 1),
        (2, 2, 1),
        (4, 4, 1),
        (8, 8, 1),
        (16, 16, 1),
        (32, 32, 3),
        (64, 64, 3),
        (128, 160, 1),
        (128, 160, 3),
        (240, 320, 3),
        (480, 640, 3),
        (768, 1024, 3),
        (4, 128, 160, 3),
    ):
        image = rng.random(shape, dtype=np.float32)
        correctness = check_values(image)
        baseline = partial(numpy_mean_std, image)
        candidate = partial(mean_std, image, "global")
        baseline_ms, candidate_ms = paired_times(baseline, candidate, args.repeats, args.warmup)
        std_baseline_ms, std_candidate_ms = paired_times(
            partial(numpy_std, image),
            partial(std, image, "global"),
            args.repeats,
            args.warmup,
        )
        rows.append(
            {
                **correctness,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "std_baseline_ms": std_baseline_ms,
                "std_candidate_ms": std_candidate_ms,
                "std_speedup": std_baseline_ms / std_candidate_ms,
            },
        )

    shifted = np.full((128, 160, 3), np.float32(1e6), dtype=np.float32)
    shifted[::3] = np.nextafter(np.float32(1e6), np.float32(np.inf))
    near_constant = np.full((128, 160, 3), np.float32(0.5), dtype=np.float32)
    near_constant.flat[0] = np.nextafter(np.float32(0.5), np.float32(np.inf))
    guard_cases = []
    constant_cases = tuple(
        (f"constant_{math.prod(shape)}", np.full(shape, np.float32(0.5), dtype=np.float32))
        for shape in ((1, 2, 1), (4, 4, 1), (16, 16, 1), (128, 160, 3))
    )
    for name, image in (
        ("high_offset", shifted),
        ("near_constant", near_constant),
        *constant_cases,
    ):
        correctness = check_values(image)
        baseline_ms, candidate_ms = paired_times(
            partial(numpy_mean_std, image),
            partial(mean_std, image, "global"),
            args.repeats,
            args.warmup,
        )
        guard_cases.append(
            {
                "name": name,
                **correctness,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
            },
        )
    precision_cases = []
    for offset, relative_std in ((1.0, 0.02), (100.0, 0.02), (10_000.0, 0.02), (1e6, 0.02), (1e8, 0.1)):
        image = rng.normal(offset, offset * relative_std, size=(128, 160, 3)).astype(np.float32)
        precision_cases.append(
            {
                "offset": offset,
                "relative_std": relative_std,
                **check_values(image),
            },
        )
    smallest = np.nextafter(np.float32(0), np.float32(1))
    subnormal_constant = np.full((128, 160, 3), smallest, dtype=np.float32)
    subnormal_mixed = subnormal_constant.copy()
    subnormal_mixed.flat[0] = -smallest
    precision_cases.extend(
        [
            {"name": "subnormal_constant", **check_values(subnormal_constant)},
            {"name": "subnormal_mixed", **check_values(subnormal_mixed)},
        ],
    )
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "benchmark": {"repeats": args.repeats, "warmup": args.warmup, "paired": True},
        "guard": {
            "relative_variance_floor": RELATIVE_VARIANCE_FLOOR,
            "fallback_cases": guard_cases,
            "well_conditioned_offsets": precision_cases,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
