#!/usr/bin/env python3
"""Print Markdown comparing NumPy vs NumKong global sum/mean/std on `(H,W,C)` arrays."""

from __future__ import annotations

import platform
from pathlib import Path

import numkong as nk
import numpy as np

from timing import median_ms


def main() -> None:
    rng = np.random.default_rng(42)
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    channels = [1, 3, 9]
    eps = 1e-4

    print("### Sum / mean / std: NumPy vs NumKong (image-shaped `(H,W,C)`)")
    print()
    print(
        "**Summary (one machine; regenerate to compare hardware):**\n"
        "- **float32:** **NumPy** `img.sum()` / `img.mean()` / `img.std()` is usually faster than NumKong "
        "`Tensor.sum` / `moments`-derived mean & std on a contiguous ravel.\n"
        "- **uint8:** **NumKong `moments`** (wide accumulator) is much faster than NumPy "
        "`float(img.mean())` / `float(img.std())` for global stats over the full image "
        "(avoids naive uint8 overflow in reductions).",
    )
    print()
    fence = "`" * 3
    print("<details>")
    print(
        "<summary>Source: <code>benchmarks/benchmark_sum_mean_std_ravel.py</code></summary>",
    )
    print()
    print(fence + "python")
    print(Path(__file__).resolve().read_text(encoding="utf-8").rstrip("\n"))
    print(fence)
    print()
    print("</details>")
    print()
    print("#### float32 — NumPy on `img` vs NumKong on contiguous ravel")
    print()
    print(
        "NumKong: `Tensor.sum()`; mean/std from one `moments` call per timing "
        f"(`mean = s/n`, `std = sqrt(max(s2/n - mean**2, 0)) + {eps}`). "
        "NumPy: `img.sum()`, `img.mean()`, `img.std() + eps`.",
    )
    print()
    print("| H×W | C | pixels | op | NumPy (ms) | NumKong (ms) | faster |")
    print("|-----|---|--------|----|-----------:|-------------:|--------|")

    for h, w in sizes:
        for c in channels:
            img = rng.random((h, w, c), dtype=np.float32)
            n = h * w * c

            t_np = median_ms(lambda: float(img.sum()))
            t_nk = median_ms(lambda: float(nk.sum(img)))
            wn = "NumPy" if t_np <= t_nk else "NumKong"
            print(f"| {h}×{w} | {c} | {n} | sum | {t_np:.4f} | {t_nk:.4f} | {wn} |")

            t_np = median_ms(lambda: float(img.mean()))

            def nk_mean(a=img) -> None:
                s, _ = nk.moments(a)
                float(s) / a.size

            t_nk = median_ms(nk_mean)
            wn = "NumPy" if t_np <= t_nk else "NumKong"
            print(f"| {h}×{w} | {c} | {n} | mean | {t_np:.4f} | {t_nk:.4f} | {wn} |")

            t_np = median_ms(lambda: float(img.std()) + eps)

            def nk_std(a=img) -> None:
                s, s2 = nk.moments(a)
                m = float(s) / a.size
                v = max(float(s2) / a.size - m * m, 0.0)
                float(np.sqrt(v)) + eps

            t_nk = median_ms(nk_std)
            wn = "NumPy" if t_np <= t_nk else "NumKong"
            print(f"| {h}×{w} | {c} | {n} | std | {t_np:.4f} | {t_nk:.4f} | {wn} |")

    print()
    print("#### uint8 — NumPy global `mean`/`std` vs NumKong `moments` (global over all pixels)")
    print()
    print("| H×W | C | pixels | NumPy mean+std (ms) | NumKong moments+formula (ms) | faster |")
    print("|-----|---|--------|--------------------:|-----------------------------:|--------|")

    for h, w in sizes:
        for c in channels:
            img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            n = h * w * c

            def np_two(a=img) -> None:
                float(a.mean())
                float(a.std()) + eps

            def nk_two(a=img) -> None:
                s, s2 = nk.moments(a)
                m = float(s) / a.size
                v = max(float(s2) / a.size - m * m, 0.0)
                float(np.sqrt(v)) + eps

            t_np = median_ms(np_two)
            t_nk = median_ms(nk_two)
            wn = "NumPy" if t_np <= t_nk else "NumKong"
            print(f"| {h}×{w} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {wn} |")

    print()
    print(
        "_Environment:_ ",
        f"`{platform.system()}` `{platform.machine()}`, ",
        f"numkong {getattr(__import__('numkong'), '__version__', '?')}, ",
        f"numpy {np.__version__}. ",
        "Median ms, 11 repeats, 3 warmup, seed=42.",
    )


if __name__ == "__main__":
    main()
