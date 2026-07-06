### Sum / mean / std: NumPy vs NumKong (image-shaped `(H,W,C)`)

**Summary (one machine; regenerate to compare hardware):**
- **float32:** **NumPy** `img.sum()` / `img.mean()` / `img.std()` is usually faster than NumKong `Tensor.sum` / `moments`-derived mean & std on a contiguous ravel.
- **uint8:** **NumKong `moments`** (wide accumulator) is much faster than NumPy `float(img.mean())` / `float(img.std())` for global stats over the full image (avoids naive uint8 overflow in reductions).

<details>
<summary>Source: <code>benchmarks/benchmark_sum_mean_std_ravel.py</code></summary>

```python
#!/usr/bin/env python3
"""Print Markdown comparing NumPy vs NumKong global sum/mean/std on `(H,W,C)` arrays."""

from __future__ import annotations

import platform
from pathlib import Path

import numkong as nk
import numpy as np

from shape_grids import LARGE_SQUARE_HW
from timing import median_ms


def main() -> None:
    rng = np.random.default_rng(42)
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

    for h, w in LARGE_SQUARE_HW:
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

    for h, w in LARGE_SQUARE_HW:
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
```

</details>

#### float32 — NumPy on `img` vs NumKong on contiguous ravel

NumKong: `Tensor.sum()`; mean/std from one `moments` call per timing (`mean = s/n`, `std = sqrt(max(s2/n - mean**2, 0)) + 0.0001`). NumPy: `img.sum()`, `img.mean()`, `img.std() + eps`.

| H×W | C | pixels | op | NumPy (ms) | NumKong (ms) | faster |
|-----|---|--------|----|-----------:|-------------:|--------|
| 256×256 | 1 | 65536 | sum | 0.0082 | 0.0296 | NumPy |
| 256×256 | 1 | 65536 | mean | 0.0119 | 0.0294 | NumPy |
| 256×256 | 1 | 65536 | std | 0.0303 | 0.0297 | NumKong |
| 256×256 | 3 | 196608 | sum | 0.0222 | 0.0873 | NumPy |
| 256×256 | 3 | 196608 | mean | 0.0245 | 0.0933 | NumPy |
| 256×256 | 3 | 196608 | std | 0.0766 | 0.0909 | NumPy |
| 256×256 | 9 | 589824 | sum | 0.0653 | 0.2612 | NumPy |
| 256×256 | 9 | 589824 | mean | 0.0665 | 0.2612 | NumPy |
| 256×256 | 9 | 589824 | std | 0.2190 | 0.2590 | NumPy |
| 512×512 | 1 | 262144 | sum | 0.0287 | 0.1151 | NumPy |
| 512×512 | 1 | 262144 | mean | 0.0369 | 0.1165 | NumPy |
| 512×512 | 1 | 262144 | std | 0.1013 | 0.1170 | NumPy |
| 512×512 | 3 | 786432 | sum | 0.0864 | 0.3485 | NumPy |
| 512×512 | 3 | 786432 | mean | 0.0863 | 0.3593 | NumPy |
| 512×512 | 3 | 786432 | std | 0.2833 | 0.3525 | NumPy |
| 512×512 | 9 | 2359296 | sum | 0.2581 | 1.0966 | NumPy |
| 512×512 | 9 | 2359296 | mean | 0.2593 | 1.0487 | NumPy |
| 512×512 | 9 | 2359296 | std | 0.8407 | 1.0303 | NumPy |
| 1024×1024 | 1 | 1048576 | sum | 0.1244 | 0.4773 | NumPy |
| 1024×1024 | 1 | 1048576 | mean | 0.1332 | 0.4810 | NumPy |
| 1024×1024 | 1 | 1048576 | std | 0.3814 | 0.4778 | NumPy |
| 1024×1024 | 3 | 3145728 | sum | 0.3521 | 1.4435 | NumPy |
| 1024×1024 | 3 | 3145728 | mean | 0.3745 | 1.4478 | NumPy |
| 1024×1024 | 3 | 3145728 | std | 1.2073 | 1.4256 | NumPy |
| 1024×1024 | 9 | 9437184 | sum | 1.0680 | 4.2717 | NumPy |
| 1024×1024 | 9 | 9437184 | mean | 1.0899 | 4.3332 | NumPy |
| 1024×1024 | 9 | 9437184 | std | 3.5365 | 4.2605 | NumPy |

#### uint8 — NumPy global `mean`/`std` vs NumKong `moments` (global over all pixels)

| H×W | C | pixels | NumPy mean+std (ms) | NumKong moments+formula (ms) | faster |
|-----|---|--------|--------------------:|-----------------------------:|--------|
| 256×256 | 1 | 65536 | 0.1002 | 0.0037 | NumKong |
| 256×256 | 3 | 196608 | 0.2842 | 0.0103 | NumKong |
| 256×256 | 9 | 589824 | 0.8049 | 0.0321 | NumKong |
| 512×512 | 1 | 262144 | 0.3554 | 0.0130 | NumKong |
| 512×512 | 3 | 786432 | 1.1000 | 0.0382 | NumKong |
| 512×512 | 9 | 2359296 | 3.2359 | 0.1095 | NumKong |
| 1024×1024 | 1 | 1048576 | 1.4541 | 0.0505 | NumKong |
| 1024×1024 | 3 | 3145728 | 4.2880 | 0.1478 | NumKong |
| 1024×1024 | 9 | 9437184 | 13.3035 | 0.4497 | NumKong |

_Environment:_  `Darwin` `arm64`,  numkong 7.7.0,  numpy 2.2.6.  Median ms, 11 repeats, 3 warmup, seed=42.
