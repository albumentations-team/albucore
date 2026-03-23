#!/usr/bin/env python3
"""Micro-benchmarks: NumKong vs NumPy/OpenCV — detailed timings and speedups.

Run from repo root::

    uv run python benchmarks/benchmark_numkong.py
    uv run python benchmarks/benchmark_numkong.py --repeats 15

Median time in milliseconds; **speedup** = slower_median / faster_median (≥1).
Results are **machine-specific** (CPU, thermals, BLAS). Use this to choose backends,
not as absolute truth across hardware.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numkong as nk
import numpy as np

from timing import median_ms as _median_ms

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BenchRow:
    """One comparison row for the report."""

    category: str
    case: str
    winner: str
    times_ms: dict[str, float]
    note: str = ""

    def speedup_line(self) -> str:
        vals = {k: v for k, v in self.times_ms.items() if not math.isnan(v) and v > 0}
        if len(vals) < 2:
            return ""
        fastest = min(vals, key=vals.get)
        slowest = max(vals, key=vals.get)
        if fastest == slowest:
            return ""
        ratio = vals[slowest] / vals[fastest]
        return f"{slowest} is {ratio:.2f}x slower than {fastest}"


def _print_table(rows: list[BenchRow]) -> None:
    print()
    print("=" * 100)
    print("DETAILED RESULTS (median ms; lower is better)")
    print("=" * 100)

    current_cat = ""
    for r in rows:
        if r.category != current_cat:
            current_cat = r.category
            print()
            print(f"## {current_cat}")
            print("-" * 100)

        parts = [f"  {r.case}"]
        for name in sorted(r.times_ms.keys()):
            t = r.times_ms[name]
            parts.append(f"  {name}={t:.4f}ms" if not math.isnan(t) else f"  {name}=n/a")
        print("".join(parts))
        print(f"      → WINNER: {r.winner}")
        sl = r.speedup_line()
        if sl:
            print(f"      → {sl}")
        if r.note:
            print(f"      → note: {r.note}")

    print()
    print("=" * 100)
    print("SUMMARY — what is faster when (on this machine)")
    print("=" * 100)
    for r in rows:
        sl = r.speedup_line()
        extra = f" ({sl})" if sl else ""
        print(f"  • [{r.category}] {r.case}: **{r.winner}**{extra}")
    print()


def bench_blend_sweep(repeats: int, warmup: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(0)
    w1, w2 = 0.6, 0.4

    for h, w, c in [(128, 128, 1), (128, 128, 3), (256, 256, 3), (512, 512, 3)]:
        img1_u8 = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        img2_u8 = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        a1, a2 = img1_u8.reshape(-1), img2_u8.reshape(-1)

        def nk_b() -> None:
            nk.blend(a1, a2, alpha=w1, beta=w2)

        def np_b() -> None:
            np.clip(img1_u8.astype(np.float32) * w1 + img2_u8.astype(np.float32) * w2, 0, 255).astype(np.uint8)

        t_nk = _median_ms(nk_b, repeats, warmup)
        t_np = _median_ms(np_b, repeats, warmup)
        t_cv = float("nan")
        if cv2 is not None:

            def cv_b() -> None:
                cv2.addWeighted(img1_u8, w1, img2_u8, w2, 0)

            t_cv = _median_ms(cv_b, repeats, warmup)

        times = {"numkong_blend": t_nk, "numpy": t_np}
        if cv2 is not None:
            times["opencv_addWeighted"] = t_cv
        winner = min(times, key=times.get)
        rows.append(
            BenchRow(
                "addWeighted / blend (uint8)",
                f"{h}x{w}x{c} pixels ({a1.size} elems raveled)",
                winner,
                times,
                note="albucore add_weighted uses NumKong blend on this path.",
            ),
        )

    for h, w, c in [(256, 256, 3), (512, 512, 3)]:
        img1_f = rng.random((h, w, c), dtype=np.float32)
        img2_f = rng.random((h, w, c), dtype=np.float32)
        f1, f2 = img1_f.reshape(-1), img2_f.reshape(-1)

        def nk_bf() -> None:
            nk.blend(f1, f2, alpha=w1, beta=w2)

        def np_bf() -> None:
            img1_f * w1 + img2_f * w2

        t_nk = _median_ms(nk_bf, repeats, warmup)
        t_np = _median_ms(np_bf, repeats, warmup)
        t_cv = float("nan")
        if cv2 is not None:

            def cv_bf() -> None:
                cv2.addWeighted(img1_f, w1, img2_f, w2, 0)

            t_cv = _median_ms(cv_bf, repeats, warmup)

        times = {"numkong_blend": t_nk, "numpy": t_np}
        if cv2 is not None:
            times["opencv_addWeighted"] = t_cv
        winner = min(times, key=times.get)
        rows.append(
            BenchRow(
                "addWeighted / blend (float32)",
                f"{h}x{w}x{c}",
                winner,
                times,
            ),
        )

    return rows


def bench_cdist_sweep(repeats: int, warmup: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(1)
    d = 2

    configs = [
        (5, 5),
        (10, 10),
        (20, 20),
        (32, 32),
        (50, 50),
        (100, 10),
        (100, 100),
    ]
    for n, m in configs:
        p1 = rng.standard_normal((n, d), dtype=np.float32)
        p2 = rng.standard_normal((m, d), dtype=np.float32)
        prod = n * m

        def nk_c() -> None:
            nk.cdist(p1, p2, metric="sqeuclidean")

        def np_c() -> None:
            p1_sq = (p1**2).sum(axis=1, keepdims=True)
            p2_sq = (p2**2).sum(axis=1)[None, :]
            out = p1_sq + p2_sq - 2 * (p1 @ p2.T)
            np.maximum(out, 0.0, out=out)

        t_nk = _median_ms(nk_c, repeats, warmup)
        t_np = _median_ms(np_c, repeats, warmup)
        winner = "numkong_cdist" if t_nk <= t_np else "numpy_formula"
        route = "albucore: NumKong (n*m < 1000)" if prod < 1000 else "albucore: NumPy (n*m >= 1000)"
        rows.append(
            BenchRow(
                "pairwise_distances_sq (cdist)",
                f"n={n}, m={m}, d={d} → n*m={prod}; {route}",
                winner,
                {"numkong_cdist": t_nk, "numpy_formula": t_np},
                note="Threshold 1000 matches albucore.functions.pairwise_distances_squared.",
            ),
        )

    return rows


def bench_reductions_sweep(repeats: int, warmup: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(2)

    for h, w, c in [(256, 256, 3), (512, 512, 3)]:
        img_u8 = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        flat = img_u8.reshape(-1)
        t = nk.Tensor(flat)

        def nk_s() -> None:
            t.sum()

        def np_s() -> None:
            flat.sum()

        t_nk = _median_ms(nk_s, repeats, warmup)
        t_np = _median_ms(np_s, repeats, warmup)
        winner = "Tensor.sum" if t_nk <= t_np else "np.sum"
        rows.append(
            BenchRow(
                "Global sum (uint8 ravel)",
                f"{h}x{w}x{c} → {flat.size} elems",
                winner,
                {"Tensor.sum": t_nk, "np.sum": t_np},
            ),
        )

        def nk_m() -> None:
            nk.moments(nk.Tensor(flat))

        def np_m() -> None:
            u = flat.astype(np.uint64)
            u.sum()
            (u * u).sum()

        t_nkm = _median_ms(nk_m, repeats, warmup)
        t_npm = _median_ms(np_m, repeats, warmup)
        winner = "nk.moments" if t_nkm <= t_npm else "numpy_uint64_x2"
        rows.append(
            BenchRow(
                "sum + sumsq (uint8 — like stats for var)",
                f"{h}x{w}x{c} → {flat.size} elems",
                winner,
                {"nk.moments": t_nkm, "numpy_uint64_2pass": t_npm},
                note="albucore uses nk.moments for global uint8 mean/std in some LUT paths.",
            ),
        )

    # float32: full tensor sum vs np.mean over spatial axes (different math, comparable cost order)
    h, w, c = 512, 512, 3
    img_f = rng.random((h, w, c), dtype=np.float32)
    tf = nk.Tensor(img_f)
    axes = (0, 1)

    def nk_all() -> None:
        tf.sum()

    def np_mean_axes() -> None:
        img_f.mean(axis=axes)

    t_nk = _median_ms(nk_all, repeats, warmup)
    t_np = _median_ms(np_mean_axes, repeats, warmup)
    winner = "Tensor.sum_all" if t_nk <= t_np else "np.mean(axis=spatial)"
    rows.append(
        BenchRow(
            "Reduction shape (float32)",
            f"{img_f.shape}: Tensor.sum() over all vs np.mean(axis=(0,1))",
            winner,
            {"Tensor.sum_whole": t_nk, "np.mean_H_W": t_np},
            note="Not identical reductions; shows cost of ‘one big sum’ vs ‘per-channel spatial mean’.",
        ),
    )

    return rows


def bench_minmax_sweep(repeats: int, warmup: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(3)

    for n in [65_536, 262_144, 786_432, 2_097_152]:
        flat = rng.random(n, dtype=np.float32)
        t = nk.Tensor(flat)

        def nk_mm() -> None:
            t.minmax()

        def np_mm() -> None:
            flat.min()
            flat.max()

        t_nk = _median_ms(nk_mm, repeats, warmup)
        t_np = _median_ms(np_mm, repeats, warmup)
        winner = "Tensor.minmax" if t_nk <= t_np else "np.min+np.max"
        rows.append(
            BenchRow(
                "min + max (float32 contiguous 1D)",
                f"{n} elements",
                winner,
                {"Tensor.minmax_1pass": t_nk, "np_min_plus_max": t_np},
                note="NumPy often wins on large contiguous float32 (optimized separately); verify on your CPU.",
            ),
        )

    return rows


def bench_scale_fma_sweep(repeats: int, warmup: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rng = np.random.default_rng(4)

    for n in [196_608, 786_432]:
        x = rng.random(n, dtype=np.float32)
        y = rng.random(n, dtype=np.float32)
        def nk_sc() -> None:
            nk.scale(x, alpha=1.5, beta=0.25)

        def np_sc() -> None:
            x * 1.5 + 0.25

        def nk_fm() -> None:
            nk.fma(x, y, x, alpha=1.0, beta=1.0)

        def np_fm() -> None:
            x * y + x

        t_nks = _median_ms(nk_sc, repeats, warmup)
        t_nps = _median_ms(np_sc, repeats, warmup)
        t_nkf = _median_ms(nk_fm, repeats, warmup)
        t_npf = _median_ms(np_fm, repeats, warmup)

        rows.append(
            BenchRow(
                "scale (float32 1D)",
                f"{n} elems",
                "nk.scale" if t_nks <= t_nps else "numpy",
                {"nk.scale": t_nks, "numpy_a*x+b": t_nps},
            ),
        )
        rows.append(
            BenchRow(
                "fma-style (float32 1D)",
                f"{n} elems",
                "nk.fma" if t_nkf <= t_npf else "numpy",
                {"nk.fma": t_nkf, "numpy_x*y+x": t_npf},
            ),
        )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="NumKong vs NumPy/OpenCV micro-benchmarks")
    p.add_argument("--repeats", type=int, default=11, help="Timed iterations (median taken)")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing")
    args = p.parse_args()

    print("Albucore NumKong benchmark")
    print(f"  repeats={args.repeats}, warmup={args.warmup}")
    print(f"  OpenCV (cv2): {'available' if cv2 is not None else 'NOT installed — OpenCV rows only for NumKong vs NumPy'}")

    all_rows: list[BenchRow] = []
    all_rows.extend(bench_blend_sweep(args.repeats, args.warmup))
    all_rows.extend(bench_cdist_sweep(args.repeats, args.warmup))
    all_rows.extend(bench_reductions_sweep(args.repeats, args.warmup))
    all_rows.extend(bench_minmax_sweep(args.repeats, args.warmup))
    all_rows.extend(bench_scale_fma_sweep(args.repeats, args.warmup))

    _print_table(all_rows)

    print()
    print(
        "Note: NumKong 7.x Python has `Tensor.sum`, `moments`, etc.; no `Tensor.mean` here — "
        "use sum/size or moments-derived mean for benchmarks.",
    )


if __name__ == "__main__":
    main()
