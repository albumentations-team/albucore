# ruff: noqa: T201, B023, INP001
"""``albucore.stats.reduce_sum`` (NumKong uint8 routing) vs raw NumPy ``sum``.

Run from repo root::

    uv run python benchmarks/benchmark_reduce_sum.py
"""

from __future__ import annotations

import argparse

import numpy as np
from timing import median_ms

from albucore.stats import reduce_sum


def _bench_pair(
    label: str,
    fn_albu: object,
    fn_np: object,
    repeats: int,
    warmup: int,
) -> None:
    t_a = median_ms(fn_albu, repeats, warmup)  # type: ignore[arg-type]
    t_n = median_ms(fn_np, repeats, warmup)  # type: ignore[arg-type]
    faster = "reduce_sum" if t_a <= t_n else "numpy"
    ratio = max(t_a, t_n) / max(min(t_a, t_n), 1e-12)
    print(f"{label}")
    print(f"  reduce_sum {t_a:.4f} ms  |  numpy {t_n:.4f} ms  →  {faster} ({ratio:.2f}x)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    r, w = args.repeats, args.warmup
    rng = np.random.default_rng(0)

    print("Median ms (lower better). Machine-specific.\n")

    for h, w_, c in [(256, 256, 3), (512, 512, 3)]:
        for dtype, name in [(np.uint8, "uint8"), (np.float32, "float32")]:
            if dtype == np.uint8:
                img = rng.integers(0, 256, size=(h, w_, c), dtype=np.uint8)
            else:
                img = rng.random((h, w_, c), dtype=np.float32)
            axes = tuple(range(img.ndim - 1))

            def albu_g() -> None:
                reduce_sum(img, "global")

            def np_g() -> None:
                np.sum(img, dtype=np.uint64 if dtype == np.uint8 else np.float64)

            _bench_pair(
                f"{name} global  {h}x{w_}x{c}  ({img.size} elems)",
                albu_g,
                np_g,
                r,
                w,
            )

            def albu_pc() -> None:
                reduce_sum(img, "per_channel")

            def np_pc() -> None:
                np.sum(img, axis=axes, dtype=np.uint64 if dtype == np.uint8 else np.float64)

            _bench_pair(
                f"{name} per_channel  {h}x{w_}x{c}",
                albu_pc,
                np_pc,
                r,
                w,
            )
            print()


if __name__ == "__main__":
    main()
