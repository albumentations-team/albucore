#!/usr/bin/env python3
"""Benchmark the public ``add_weighted`` router and its float32 backends.

Run from the repository root:

    uv run python benchmarks/benchmark_add_weighted.py

The issue #130 matrix uses contiguous and strided float32 HWC arrays in
``[0, 255]``, weights 0.5 / 0.5, spatial sizes 256 / 512 / 1024, and 1 / 3 / 5
channels. The default run also includes Albucore's canonical non-square sizes
and 9-channel inputs.
"""

from __future__ import annotations

import argparse
import platform
import random
import time
from collections.abc import Callable
from importlib.metadata import version

import cv2
import numpy as np

from albucore import add_weighted
from albucore.arithmetic import add_weighted_numpy, add_weighted_opencv
from albucore.weighted import add_weighted_numkong
from timing import WallTimingMs


def format_timing(timing: WallTimingMs) -> str:
    return f"{timing.median:.4f} ± {timing.mad:.4f}"


def benchmark_interleaved(
    candidates: dict[str, Callable[[], np.ndarray]],
    repeats: int,
    warmup: int,
    seed: int,
) -> dict[str, WallTimingMs]:
    """Time candidates in a shuffled order on every repeat to reduce ordering bias."""
    order = list(candidates)
    order_rng = random.Random(seed)
    for _ in range(warmup):
        order_rng.shuffle(order)
        for name in order:
            candidates[name]()

    samples: dict[str, list[float]] = {name: [] for name in candidates}
    for _ in range(repeats):
        order_rng.shuffle(order)
        for name in order:
            start = time.perf_counter()
            candidates[name]()
            samples[name].append((time.perf_counter() - start) * 1_000)

    result: dict[str, WallTimingMs] = {}
    for name, values in samples.items():
        data = np.asarray(values, dtype=np.float64)
        median = float(np.median(data))
        result[name] = WallTimingMs(
            median=median,
            mean=float(data.mean()),
            std=float(data.std(ddof=1)) if repeats > 1 else 0.0,
            mad=float(np.median(np.abs(data - median))),
            n=repeats,
        )
    return result


def make_inputs(
    rng: np.random.Generator,
    height: int,
    width: int,
    channels: int,
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    storage_width = width if layout == "contiguous" else width * 2
    shape = (height, storage_width, channels)
    img1 = rng.random(shape, dtype=np.float32) * np.float32(255)
    img2 = rng.random(shape, dtype=np.float32) * np.float32(255)
    if layout == "strided":
        return img1[:, ::2, :], img2[:, ::2, :]
    return img1, img2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", choices=("all", "issue", "canonical"), default="all")
    parser.add_argument("--channels", type=int, nargs="+", default=(1, 3, 5, 9))
    parser.add_argument(
        "--layouts",
        choices=("contiguous", "strided"),
        nargs="+",
        default=("contiguous", "strided"),
    )
    args = parser.parse_args()

    cv2.setNumThreads(0)
    rng = np.random.default_rng(args.seed)
    weight1, weight2 = 0.5, 0.5
    issue_hw = ((256, 256), (512, 512), (1024, 1024))
    canonical_hw = ((128, 160), (240, 320), (480, 640), (768, 1024))
    if args.grid == "issue":
        sizes = issue_hw
    elif args.grid == "canonical":
        sizes = canonical_hw
    else:
        sizes = (*issue_hw, *canonical_hw)

    print("# `add_weighted` float32 backend benchmark")
    print()
    print(
        f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, "
        f"NumPy {np.__version__}, OpenCV {cv2.__version__}, NumKong {version('numkong')}. "
        f"Median ± MAD in ms; {args.repeats} repeats after {args.warmup} warmups; OpenCV threads: "
        f"{cv2.getNumThreads()}.",
    )
    print()
    print("Candidates are timed in a shuffled order on every repeat.")
    print()
    print(
        "| Layout | Shape | Public router | NumPy | OpenCV | NumKong | Fastest backend | Router / best | "
        "Max abs diff |",
    )
    print("|---|---|---:|---:|---:|---:|---|---:|---:|")

    for layout_index, layout in enumerate(args.layouts):
        for height, width in sizes:
            for channels in args.channels:
                shape = (height, width, channels)
                img1, img2 = make_inputs(rng, height, width, channels, layout)
                expected = img1 * weight1 + img2 * weight2

                candidates: dict[str, Callable[[], np.ndarray]] = {
                    "Public router": lambda: add_weighted(img1, weight1, img2, weight2),
                    "NumPy": lambda: add_weighted_numpy(img1, weight1, img2, weight2),
                    "OpenCV": lambda: add_weighted_opencv(img1, weight1, img2, weight2),
                    "NumKong": lambda: add_weighted_numkong(img1, weight1, img2, weight2),
                }
                outputs = {name: candidate() for name, candidate in candidates.items()}
                for output in outputs.values():
                    assert output.shape == shape
                    assert output.dtype == np.float32
                    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)

                timings = benchmark_interleaved(
                    candidates,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    seed=args.seed + layout_index * 100 + height + width + channels,
                )
                backend_timings = {name: timings[name] for name in ("NumPy", "OpenCV", "NumKong")}
                fastest = min(backend_timings, key=lambda name: backend_timings[name].median)
                best_ms = backend_timings[fastest].median
                ratio = timings["Public router"].median / best_ms if best_ms > 0 else float("inf")
                max_abs_diff = max(float(np.max(np.abs(output - expected))) for output in outputs.values())

                print(
                    f"| {layout} | {height}×{width}×{channels} | {format_timing(timings['Public router'])} | "
                    f"{format_timing(timings['NumPy'])} | {format_timing(timings['OpenCV'])} | "
                    f"{format_timing(timings['NumKong'])} | {fastest} | {ratio:.2f}× | {max_abs_diff:.3g} |",
                )


if __name__ == "__main__":
    main()
