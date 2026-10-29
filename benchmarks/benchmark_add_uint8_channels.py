#!/usr/bin/env python3
"""
``add`` uint8: compare **OpenCV** (scalar if C<=4 else ``np.full`` + cv2), **LUT**/stringzilla,
**NumKong** ``nk.scale``, **NumPy** saturated int16 add, and the production saturating ``add`` router.

Sweep C in 5..9 at several spatial sizes. Production uses OpenCV scalar/tensor preparation for every channel count
(see ``albucore.arithmetic.add``).

Run from repo root::

    uv run python benchmarks/benchmark_add_uint8_channels.py --repeats 41 --warmup 12
"""

from __future__ import annotations

import argparse

import numpy as np

from shape_grids import SQUARE_BASE_HW
from timing import bench_wall_ms

from albucore.arithmetic import add, _apply_lut, _add_uint8_opencv
from albucore.arithmetic import _scale_numkong


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
    channels = list(range(5, 10))

    print(
        "| H×W | C | OpenCV | LUT | NumKong | NumPy | `add` | fastest |",
    )
    print("|-----|---|-------:|----:|--------:|------:|---------------:|---------|")
    for h, w in SQUARE_BASE_HW:
        for c in channels:
            base = rng.integers(0, 256, (h, w, c), dtype=np.uint8)

            def run_cv() -> None:
                x = np.ascontiguousarray(base.copy())
                _add_uint8_opencv(x, scalar)

            def run_lut() -> None:
                x = np.ascontiguousarray(base.copy())
                _apply_lut(x, scalar, 'add')

            def run_nk() -> None:
                x = np.ascontiguousarray(base.copy())
                _scale_numkong(x, 1.0, float(scalar))

            def run_np() -> None:
                x = base.copy()
                numpy_uint8_add_constant(x, scalar)

            def run_wrap() -> None:
                x = np.ascontiguousarray(base.copy())
                add(x, float(scalar))

            ref = _add_uint8_opencv(np.ascontiguousarray(base.copy()), scalar)
            for name, fn in [
                ("LUT", run_lut),
                ("NumKong", run_nk),
                ("NumPy", run_np),
            ]:
                got = {
                    "LUT": _apply_lut(np.ascontiguousarray(base.copy()), scalar, 'add'),
                    "NumKong": _scale_numkong(np.ascontiguousarray(base.copy()), 1.0, float(scalar)),
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
        "Production uint8 `add`: **C≤4** → OpenCV; **C≥5** → NumKong (wrapper ≈ NumKong ms for C≥5). "
        "This table still compares all backends for the same shapes."
    )


if __name__ == "__main__":
    main()
