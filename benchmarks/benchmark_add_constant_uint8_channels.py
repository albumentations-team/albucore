#!/usr/bin/env python3
"""
``add_constant`` uint8: compare **OpenCV** (scalar if C<=4 else ``np.full`` + cv2), **LUT**/stringzilla,
**NumKong** ``nk.scale``, **NumPy** saturated int16 add, and the production **@clipped** ``add_constant`` wrapper.

Sweep C in 5..9 at several spatial sizes. Production: **C≤4** OpenCV scalar/tensor prep, **C≥5** NumKong
``add_constant_numkong`` (see ``albucore.arithmetic.add_constant``).

Run from repo root::

    uv run python benchmarks/benchmark_add_constant_uint8_channels.py --repeats 41 --warmup 12
"""

from __future__ import annotations

import argparse

import numpy as np

from timing import bench_wall_ms

from albucore.arithmetic import add_constant, add_lut, add_opencv
from albucore.weighted import add_constant_numkong


def numpy_uint8_add_constant(img: np.ndarray, scalar: int) -> np.ndarray:
    """Saturated uint8 + scalar (reference; not production API)."""
    return np.clip(np.add(img.astype(np.int16, copy=False), scalar), 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=31)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    scalar = 3
    spatial = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    channels = list(range(5, 10))

    print(
        "| H×W | C | OpenCV | LUT | NumKong | NumPy | `add_constant` | fastest |",
    )
    print("|-----|---|-------:|----:|--------:|------:|---------------:|---------|")
    for h, w in spatial:
        for c in channels:
            base = rng.integers(0, 256, (h, w, c), dtype=np.uint8)

            def run_cv() -> None:
                x = np.ascontiguousarray(base.copy())
                add_opencv(x, scalar)

            def run_lut() -> None:
                x = np.ascontiguousarray(base.copy())
                add_lut(x, scalar)

            def run_nk() -> None:
                x = np.ascontiguousarray(base.copy())
                add_constant_numkong(x, float(scalar))

            def run_np() -> None:
                x = base.copy()
                numpy_uint8_add_constant(x, scalar)

            def run_wrap() -> None:
                x = np.ascontiguousarray(base.copy())
                add_constant(x, float(scalar))

            ref = add_opencv(np.ascontiguousarray(base.copy()), scalar)
            for name, fn in [
                ("LUT", run_lut),
                ("NumKong", run_nk),
                ("NumPy", run_np),
            ]:
                got = {
                    "LUT": add_lut(np.ascontiguousarray(base.copy()), scalar),
                    "NumKong": add_constant_numkong(np.ascontiguousarray(base.copy()), float(scalar)),
                    "NumPy": numpy_uint8_add_constant(base.copy(), scalar),
                }[name]
                if not np.array_equal(ref, got):
                    msg = f"{name} mismatch H={h} W={w} C={c}"
                    raise AssertionError(msg)

            t_cv = bench_wall_ms(run_cv, repeats=args.repeats, warmup=args.warmup).median
            t_lt = bench_wall_ms(run_lut, repeats=args.repeats, warmup=args.warmup).median
            t_nk = bench_wall_ms(run_nk, repeats=args.repeats, warmup=args.warmup).median
            t_np = bench_wall_ms(run_np, repeats=args.repeats, warmup=args.warmup).median
            t_wr = bench_wall_ms(run_wrap, repeats=args.repeats, warmup=args.warmup).median

            times = {
                "OpenCV": t_cv,
                "LUT": t_lt,
                "NumKong": t_nk,
                "NumPy": t_np,
                "wrapper": t_wr,
            }
            fastest = min(times, key=times.get)
            print(
                f"| {h}×{w} | {c} | {t_cv:.4f} | {t_lt:.4f} | {t_nk:.4f} | {t_np:.4f} | {t_wr:.4f} | {fastest} |",
            )

    print()
    print(
        "Production uint8 `add_constant`: **C≤4** → OpenCV; **C≥5** → NumKong (wrapper ≈ NumKong ms for C≥5). "
        "This table still compares all backends for the same shapes."
    )


if __name__ == "__main__":
    main()
