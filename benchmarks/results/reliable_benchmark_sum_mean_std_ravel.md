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
            flat = np.ascontiguousarray(img.reshape(-1))
            t = nk.Tensor(flat)

            t_np = median_ms(lambda: float(img.sum()))
            t_nk = median_ms(lambda: float(t.sum()))
            wn = "NumPy" if t_np <= t_nk else "NumKong"
            print(f"| {h}×{w} | {c} | {n} | sum | {t_np:.4f} | {t_nk:.4f} | {wn} |")

            t_np = median_ms(lambda: float(img.mean()))

            def nk_mean() -> None:
                s, _ = nk.moments(nk.Tensor(flat))
                float(s) / n

            t_nk = median_ms(nk_mean)
            wn = "NumPy" if t_np <= t_nk else "NumKong"
            print(f"| {h}×{w} | {c} | {n} | mean | {t_np:.4f} | {t_nk:.4f} | {wn} |")

            t_np = median_ms(lambda: float(img.std()) + eps)

            def nk_std() -> None:
                s, s2 = nk.moments(nk.Tensor(flat))
                m = float(s) / n
                v = max(float(s2) / n - m * m, 0.0)
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
            flat = np.ascontiguousarray(img.reshape(-1))

            def np_two() -> None:
                float(img.mean())
                float(img.std()) + eps

            def nk_two() -> None:
                s, s2 = nk.moments(nk.Tensor(flat))
                m = float(s) / n
                v = max(float(s2) / n - m * m, 0.0)
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
| 256×256 | 1 | 65536 | sum | 0.0084 | 0.0285 | NumPy |
| 256×256 | 1 | 65536 | mean | 0.0103 | 0.0322 | NumPy |
| 256×256 | 1 | 65536 | std | 0.0301 | 0.0324 | NumPy |
| 256×256 | 3 | 196608 | sum | 0.0235 | 0.0851 | NumPy |
| 256×256 | 3 | 196608 | mean | 0.0254 | 0.0952 | NumPy |
| 256×256 | 3 | 196608 | std | 0.0766 | 0.0954 | NumPy |
| 256×256 | 9 | 589824 | sum | 0.0726 | 0.2551 | NumPy |
| 256×256 | 9 | 589824 | mean | 0.0749 | 0.3482 | NumPy |
| 256×256 | 9 | 589824 | std | 0.3005 | 0.3480 | NumPy |
| 512×512 | 1 | 262144 | sum | 0.0310 | 0.1135 | NumPy |
| 512×512 | 1 | 262144 | mean | 0.0330 | 0.1267 | NumPy |
| 512×512 | 1 | 262144 | std | 0.0990 | 0.1272 | NumPy |
| 512×512 | 3 | 786432 | sum | 0.0930 | 0.3401 | NumPy |
| 512×512 | 3 | 786432 | mean | 0.0947 | 0.4961 | NumPy |
| 512×512 | 3 | 786432 | std | 0.4258 | 0.4893 | NumPy |
| 512×512 | 9 | 2359296 | sum | 0.3002 | 0.9920 | NumPy |
| 512×512 | 9 | 2359296 | mean | 0.3045 | 1.1043 | NumPy |
| 512×512 | 9 | 2359296 | std | 0.8933 | 1.1375 | NumPy |
| 1024×1024 | 1 | 1048576 | sum | 0.1231 | 0.4481 | NumPy |
| 1024×1024 | 1 | 1048576 | mean | 0.1245 | 0.6655 | NumPy |
| 1024×1024 | 1 | 1048576 | std | 0.5413 | 0.6740 | NumPy |
| 1024×1024 | 3 | 3145728 | sum | 0.3685 | 1.3330 | NumPy |
| 1024×1024 | 3 | 3145728 | mean | 0.3741 | 1.5041 | NumPy |
| 1024×1024 | 3 | 3145728 | std | 1.1425 | 1.4809 | NumPy |
| 1024×1024 | 9 | 9437184 | sum | 1.2081 | 3.8190 | NumPy |
| 1024×1024 | 9 | 9437184 | mean | 1.2664 | 4.4498 | NumPy |
| 1024×1024 | 9 | 9437184 | std | 3.6869 | 4.3942 | NumPy |

#### uint8 — NumPy global `mean`/`std` vs NumKong `moments` (global over all pixels)

| H×W | C | pixels | NumPy mean+std (ms) | NumKong moments+formula (ms) | faster |
|-----|---|--------|--------------------:|-----------------------------:|--------|
| 256×256 | 1 | 65536 | 0.0912 | 0.0051 | NumKong |
| 256×256 | 3 | 196608 | 0.3033 | 0.0125 | NumKong |
| 256×256 | 9 | 589824 | 0.9103 | 0.0352 | NumKong |
| 512×512 | 1 | 262144 | 0.4225 | 0.0164 | NumKong |
| 512×512 | 3 | 786432 | 1.2969 | 0.0467 | NumKong |
| 512×512 | 9 | 2359296 | 3.0598 | 0.2028 | NumKong |
| 1024×1024 | 1 | 1048576 | 1.6779 | 0.0621 | NumKong |
| 1024×1024 | 3 | 3145728 | 5.0078 | 0.2705 | NumKong |
| 1024×1024 | 9 | 9437184 | 16.7442 | 0.5605 | NumKong |

_Environment:_  `Darwin` `arm64`,  numkong 7.0.0,  numpy 2.4.2.  Median ms, 11 repeats, 3 warmup, seed=42.
