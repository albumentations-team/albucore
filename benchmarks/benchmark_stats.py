"""Timing sweep for albucore.stats vs plain NumPy.

Covers global and per_channel reductions for mean_std, std, reduce_sum
across HWC, NHWC layouts and C<=4 / C>4 to exercise all routing branches.

Run from repo root::

    uv run python benchmarks/benchmark_stats.py
"""

from __future__ import annotations

import argparse

import numpy as np
from timing import median_ms

from albucore.stats import mean_std, reduce_sum, std


def _bench(label: str, fn_albu: object, fn_np: object, r: int, w: int) -> None:
    t_a = median_ms(fn_albu, r, w)  # type: ignore[arg-type]
    t_n = median_ms(fn_np, r, w)  # type: ignore[arg-type]
    faster = "albucore" if t_a <= t_n else "numpy"
    ratio = max(t_a, t_n) / max(min(t_a, t_n), 1e-12)
    print(f"  {label}")  # noqa: T201
    print(f"    albucore {t_a:.4f} ms | numpy {t_n:.4f} ms → {faster} ({ratio:.2f}x)")  # noqa: T201


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=41)
    p.add_argument("--warmup", type=int, default=12)
    args = p.parse_args()
    r, w = args.repeats, args.warmup
    rng = np.random.default_rng(0)

    cases_hwc = [
        ((256, 256, 3), "HWC 256x256x3"),
        ((512, 512, 3), "HWC 512x512x3"),
        ((512, 512, 9), "HWC 512x512x9"),  # C>4: exercises NumPy fallback in per_channel
        ((1024, 1024, 3), "HWC 1024x1024x3"),
    ]
    cases_nhwc = [
        ((4, 256, 256, 3), "NHWC 4x256x256x3"),
        ((4, 256, 256, 9), "NHWC 4x256x256x9"),
    ]

    for dtype, dname in [(np.uint8, "uint8"), (np.float32, "float32")]:
        print(f"\n=== {dname} ===")  # noqa: T201

        print("\n--- mean_std(global) ---")  # noqa: T201
        for shape, label in cases_hwc + cases_nhwc:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            _bench(
                label,
                lambda a=arr: mean_std(a, "global"),
                lambda a=arr: (float(np.mean(a, dtype=np.float64)), float(np.std(a, dtype=np.float64)) + 1e-4),
                r,
                w,
            )

        print("\n--- mean_std(per_channel) ---")  # noqa: T201
        for shape, label in cases_hwc + cases_nhwc:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            axes = tuple(range(arr.ndim - 1))
            _bench(
                label,
                lambda a=arr: mean_std(a, "per_channel"),
                lambda a=arr, ax=axes: (
                    a.mean(axis=ax, dtype=np.float64),
                    a.std(axis=ax, dtype=np.float64) + 1e-4,
                ),
                r,
                w,
            )

        print("\n--- std(global) ---")  # noqa: T201
        for shape, label in cases_hwc:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            _bench(
                label,
                lambda a=arr: std(a, "global"),
                lambda a=arr: float(np.std(a, dtype=np.float64)) + 1e-4,
                r,
                w,
            )

        print("\n--- std(per_channel) ---")  # noqa: T201
        for shape, label in cases_hwc + cases_nhwc:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            axes = tuple(range(arr.ndim - 1))
            _bench(
                label,
                lambda a=arr: std(a, "per_channel"),
                lambda a=arr, ax=axes: a.std(axis=ax, dtype=np.float64) + 1e-4,
                r,
                w,
            )

        print("\n--- reduce_sum(global) ---")  # noqa: T201
        acc = np.uint64 if dtype == np.uint8 else np.float64
        for shape, label in cases_hwc + cases_nhwc:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            _bench(
                label,
                lambda a=arr: reduce_sum(a, "global"),
                lambda a=arr, ac=acc: np.sum(a, dtype=ac),
                r,
                w,
            )

        print("\n--- reduce_sum(per_channel) ---")  # noqa: T201
        for shape, label in cases_hwc + cases_nhwc:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            axes = tuple(range(arr.ndim - 1))
            _bench(
                label,
                lambda a=arr: reduce_sum(a, "per_channel"),
                lambda a=arr, ax=axes, ac=acc: np.sum(a, axis=ax, dtype=ac),
                r,
                w,
            )


if __name__ == "__main__":
    main()
