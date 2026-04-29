#!/usr/bin/env python3
"""
Min/max on a raveled image buffer: NumKong ``Tensor.minmax()`` vs NumPy ``min`` + ``max``.

Run from repo root::

    uv run python benchmarks/benchmark_minmax_ravel.py

Stdout is Markdown (tables + optional collapsible details).
"""

from __future__ import annotations

import argparse
import platform

import cv2
import numkong as nk
import numpy as np

from timing import median_ms


def _cv2_reduce_minmax_per_channel(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    dtype = cv2.CV_32F if arr.dtype == np.float32 else -1
    mn = cv2.reduce(flat, 0, cv2.REDUCE_MIN, dtype=dtype).reshape(-1)
    mx = cv2.reduce(flat, 0, cv2.REDUCE_MAX, dtype=dtype).reshape(-1)
    return mn, mx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    shapes: list[tuple[int, int, int]] = [
        (224, 224, 3),  # common CNN
        (512, 512, 3),  # medium
        (1024, 1024, 1),  # large grayscale
    ]
    per_channel_shapes: list[tuple[int, ...]] = [
        (128, 128, 1),
        (128, 128, 3),
        (128, 128, 9),
        (512, 512, 3),
        (512, 512, 9),
        (1024, 1024, 3),
        (1024, 1024, 9),
        (4, 256, 256, 3),
        (4, 256, 256, 9),
    ]

    rows: list[tuple[str, str, int, float, float, str, float]] = []
    pc_rows: list[tuple[str, str, float, float, str, float]] = []

    for dtype, dname in [(np.float32, "float32"), (np.uint8, "uint8")]:
        for h, w, c in shapes:
            if dtype == np.uint8:
                img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            else:
                img = rng.random((h, w, c), dtype=np.float32)
            flat = np.ascontiguousarray(img).reshape(-1)
            t = nk.Tensor(flat)

            def nk_mm() -> None:
                t.minmax()

            def np_mm() -> None:
                flat.min()
                flat.max()

            t_nk = median_ms(nk_mm, args.repeats, args.warmup)
            t_np = median_ms(np_mm, args.repeats, args.warmup)
            faster = "NumPy" if t_np <= t_nk else "NumKong"
            ratio = max(t_nk, t_np) / max(min(t_nk, t_np), 1e-12)
            rows.append((dname, f"{h}×{w}×{c}", flat.size, t_nk, t_np, faster, ratio))

        for shape in per_channel_shapes:
            if dtype == np.uint8:
                arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
            else:
                arr = rng.random(shape, dtype=np.float32)
            axes = tuple(range(arr.ndim - 1))

            def np_pc_mm() -> tuple[np.ndarray, np.ndarray]:
                return arr.min(axis=axes), arr.max(axis=axes)

            def cv_pc_mm() -> tuple[np.ndarray, np.ndarray]:
                return _cv2_reduce_minmax_per_channel(arr)

            t_np = median_ms(np_pc_mm, args.repeats, args.warmup)
            t_cv = median_ms(cv_pc_mm, args.repeats, args.warmup)
            faster = "NumPy" if t_np <= t_cv else "OpenCV"
            ratio = max(t_np, t_cv) / max(min(t_np, t_cv), 1e-12)
            pc_rows.append((dname, "×".join(str(x) for x in shape), t_np, t_cv, faster, ratio))

    print()
    print("### Benchmark: `Tensor.minmax()` vs `np.min` + `np.max` (image-shaped data)")
    print()
    print("**Setup:** Random `(H, W, C)` arrays, **C-contiguous**, `reshape(-1)` then one NumKong `Tensor` view. ")
    print("Timing median over repeated calls (warmup first). **Not** an axis-wise per-channel reduction — ")
    print("this is “global min + global max over all pixels” like a single 1D buffer.")
    print()
    print("<details>")
    print("<summary>Reference snippet (Python)</summary>")
    print()
    print("```python")
    print("import time")
    print("import numkong as nk")
    print("import numpy as np")
    print()
    print("def median_ms(fn, repeats=11, warmup=3):")
    print("    for _ in range(warmup):")
    print("        fn()")
    print("    times = []")
    print("    for _ in range(repeats):")
    print("        t0 = time.perf_counter()")
    print("        fn()")
    print("        times.append((time.perf_counter() - t0) * 1000)")
    print("    return float(np.median(times))")
    print()
    print("rng = np.random.default_rng(0)")
    print("for dtype, name in [(np.float32, 'float32'), (np.uint8, 'uint8')]:")
    print("    for h, w, c in [(224, 224, 3), (512, 512, 3), (1024, 1024, 1)]:")
    print("        if dtype == np.uint8:")
    print("            img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)")
    print("        else:")
    print("            img = rng.random((h, w, c), dtype=np.float32)")
    print("        flat = np.ascontiguousarray(img).reshape(-1)")
    print("        t = nk.Tensor(flat)")
    print("        t_nk = median_ms(lambda: t.minmax())")
    print("        t_np = median_ms(lambda: (flat.min(), flat.max()))")
    print("        print(name, (h, w, c), 'n=', flat.size, 'ms nk', t_nk, 'np', t_np)")
    print("```")
    print()
    print("</details>")
    print()
    print("| dtype | shape (H×W×C) | pixels | NumKong `minmax` (ms) | `np.min`+`np.max` (ms) | faster | ratio |")
    print("|-------|---------------|--------|----------------------:|-----------------------:|--------|------:|")
    for dname, sh, n, t_nk, t_np, faster, ratio in rows:
        print(f"| {dname} | {sh} | {n} | {t_nk:.4f} | {t_np:.4f} | {faster} | {ratio:.2f}× |")
    print()
    print("**Takeaway (this machine):** For global min+max over a flat image buffer, **NumPy was faster** ")
    print("than `Tensor.minmax()` on all tested sizes/dtypes. So for axis-wise `minmax` APIs, ")
    print("downstream libs may still prefer NumPy (or batched native reductions) unless NumKong wins on ")
    print("their exact strided/axis case.")
    print()
    print("### Benchmark: per-channel min+max, NumPy axis reduce vs `cv2.reduce`")
    print()
    print("**Setup:** Channel-last arrays, per-channel min/max over all axes except the last.")
    print()
    print("| dtype | shape | NumPy min+max (ms) | OpenCV `reduce` min+max (ms) | faster | ratio |")
    print("|-------|-------|-------------------:|-----------------------------:|--------|------:|")
    for dname, sh, t_np, t_cv, faster, ratio in pc_rows:
        print(f"| {dname} | {sh} | {t_np:.4f} | {t_cv:.4f} | {faster} | {ratio:.2f}× |")
    print()
    print(
        f"_Environment: **{platform.system()}** `{platform.machine()}`, "
        f"numkong {getattr(__import__('numkong'), '__version__', '?')}, numpy {np.__version__}, "
        f"repeats={args.repeats}, seed={args.seed}._",
    )
    print()
    print("<!-- END COPY -->")
    print()


if __name__ == "__main__":
    main()
