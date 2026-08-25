#!/usr/bin/env python3
"""Benchmark the public ``add_weighted`` router and its float32 backends.

Run from the repository root:

    uv run python benchmarks/benchmark_add_weighted.py

The issue #130 matrix uses contiguous and strided float32 HWC arrays in
``[0, 255]``, weights 0.5 / 0.5, spatial sizes 256 / 512 / 1024, and 1 / 3 / 5
channels. The default run also includes Albucore's canonical non-square sizes
and 9-channel inputs. Pass ``--grid rank`` to compare rank-3/4 inputs and
asymmetric input layouts without running the larger HWC grids.
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
from timing import WallTimingMs

from albucore import add_weighted
from albucore.arithmetic import add_weighted_numpy, add_weighted_opencv
from albucore.weighted import add_weighted_numkong


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
            raw=tuple(values),
            median=median,
            mean=float(data.mean()),
            std=float(data.std(ddof=1)) if repeats > 1 else 0.0,
            mad=float(np.median(np.abs(data - median))),
            n=repeats,
        )
    return result


def make_inputs(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    layout: str,
) -> tuple[np.ndarray, np.ndarray]:
    layout1, layout2 = {
        "contiguous": ("contiguous", "contiguous"),
        "strided": ("strided", "strided"),
        "contiguous-strided": ("contiguous", "strided"),
        "strided-contiguous": ("strided", "contiguous"),
    }[layout]

    def make_input(input_layout: str) -> np.ndarray:
        if input_layout == "contiguous":
            return rng.random(shape, dtype=np.float32) * np.float32(255)
        storage_shape = (*shape[:-2], shape[-2] * 2, shape[-1])
        storage = rng.random(storage_shape, dtype=np.float32) * np.float32(255)
        return storage[..., ::2, :]

    return make_input(layout1), make_input(layout2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", choices=("all", "issue", "canonical", "rank"), default="all")
    parser.add_argument("--ranks", type=int, choices=(3, 4), nargs="+")
    parser.add_argument("--channels", type=int, nargs="+", default=(1, 3, 5, 9))
    parser.add_argument(
        "--layouts",
        choices=("contiguous", "strided", "contiguous-strided", "strided-contiguous"),
        nargs="+",
    )
    args = parser.parse_args()

    ranks = args.ranks or ((3, 4) if args.grid == "rank" else (3,))
    layouts = args.layouts or (
        ("contiguous", "strided", "contiguous-strided", "strided-contiguous")
        if args.grid == "rank"
        else ("contiguous", "strided")
    )

    cv2.setNumThreads(0)
    rng = np.random.default_rng(args.seed)
    weight1, weight2 = 0.5, 0.5
    issue_hw = ((256, 256), (512, 512), (1024, 1024))
    canonical_hw = ((128, 160), (240, 320), (480, 640), (768, 1024))
    if args.grid == "issue":
        sizes = issue_hw
    elif args.grid == "canonical":
        sizes = canonical_hw
    elif args.grid == "rank":
        sizes = ((128, 160),)
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

    for layout_index, layout in enumerate(layouts):
        for rank in ranks:
            prefix = {3: (), 4: (4,)}[rank]
            for height, width in sizes:
                for channels in args.channels:
                    shape = (*prefix, height, width, channels)
                    img1, img2 = make_inputs(rng, shape, layout)
                    expected = img1 * weight1 + img2 * weight2

                    candidates: dict[str, Callable[[], np.ndarray]] = {
                        "Public router": lambda first=img1, second=img2: add_weighted(
                            first,
                            weight1,
                            second,
                            weight2,
                        ),
                        "NumPy": lambda first=img1, second=img2: add_weighted_numpy(
                            first,
                            weight1,
                            second,
                            weight2,
                        ),
                        "OpenCV": lambda first=img1, second=img2: add_weighted_opencv(
                            first,
                            weight1,
                            second,
                            weight2,
                        ),
                        "NumKong": lambda first=img1, second=img2: add_weighted_numkong(
                            first,
                            weight1,
                            second,
                            weight2,
                        ),
                    }
                    max_abs_diff = 0.0
                    for candidate in candidates.values():
                        output = candidate()
                        assert output.shape == shape
                        assert output.dtype == np.float32
                        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
                        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(output - expected))))
                        del output

                    timings = benchmark_interleaved(
                        candidates,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        seed=args.seed + layout_index * 100 + sum(shape),
                    )
                    backend_timings = {name: timings[name] for name in ("NumPy", "OpenCV", "NumKong")}
                    fastest = min(backend_timings, key=lambda name: backend_timings[name].median)
                    best_ms = backend_timings[fastest].median
                    ratio = timings["Public router"].median / best_ms if best_ms > 0 else float("inf")

                    shape_label = "×".join(str(dimension) for dimension in shape)
                    print(
                        f"| {layout} | {shape_label} | {format_timing(timings['Public router'])} | "
                        f"{format_timing(timings['NumPy'])} | {format_timing(timings['OpenCV'])} | "
                        f"{format_timing(timings['NumKong'])} | {fastest} | {ratio:.2f}× | {max_abs_diff:.3g} |",
                    )


if __name__ == "__main__":
    main()
