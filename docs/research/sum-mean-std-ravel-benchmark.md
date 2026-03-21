### Sum / mean / std: NumPy vs NumKong (image-shaped `(H,W,C)`)

**Summary (one archived run):**
- **float32:** **NumPy** `img.sum()` / `img.mean()` / `img.std()` is faster than NumKong `Tensor.sum` / `moments`-derived mean & std on a contiguous ravel — on the machine below, **sum** and **mean** are NumPy-faster for every cell; **std** is usually NumPy but can flip on some shapes (noise / backend overlap).
- **uint8:** **NumKong `moments`** (wide accumulator) is much faster than NumPy `float(img.mean())` / `float(img.std())` for global stats over the full image (same semantics we need for normalization without uint8 overflow in naive reductions).

<details>
<summary>Source: <code>benchmarks/benchmark_sum_mean_std_ravel.py</code></summary>

```python
#!/usr/bin/env python3
"""Print Markdown comparing NumPy vs NumKong global sum/mean/std on `(H,W,C)` arrays."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Callable

import numkong as nk
import numpy as np


def median_ms(fn: Callable[[], object], repeats: int = 11, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


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
    print()


if __name__ == "__main__":
    main()
```

</details>

#### float32 — NumPy on `img` vs NumKong on contiguous ravel

NumKong: `Tensor.sum()`; mean/std from one `moments` call per timing (`mean = s/n`, `std = sqrt(max(s2/n - mean**2, 0)) + 0.0001`). NumPy: `img.sum()`, `img.mean()`, `img.std() + eps`.

| H×W | C | pixels | op | NumPy (ms) | NumKong (ms) | faster |
|-----|---|--------|----|-----------:|-------------:|--------|
| 256×256 | 1 | 65536 | sum | 0.0077 | 0.0305 | NumPy |
| 256×256 | 1 | 65536 | mean | 0.0097 | 0.0322 | NumPy |
| 256×256 | 1 | 65536 | std | 0.0269 | 0.0362 | NumPy |
| 256×256 | 3 | 196608 | sum | 0.0224 | 0.0816 | NumPy |
| 256×256 | 3 | 196608 | mean | 0.0238 | 0.0956 | NumPy |
| 256×256 | 3 | 196608 | std | 0.0738 | 0.0977 | NumPy |
| 256×256 | 9 | 589824 | sum | 0.0805 | 0.2476 | NumPy |
| 256×256 | 9 | 589824 | mean | 0.0688 | 0.3705 | NumPy |
| 256×256 | 9 | 589824 | std | 0.3515 | 0.3758 | NumPy |
| 512×512 | 1 | 262144 | sum | 0.0301 | 0.1045 | NumPy |
| 512×512 | 1 | 262144 | mean | 0.0370 | 0.1247 | NumPy |
| 512×512 | 1 | 262144 | std | 0.0850 | 0.1216 | NumPy |
| 512×512 | 3 | 786432 | sum | 0.0935 | 0.3176 | NumPy |
| 512×512 | 3 | 786432 | mean | 0.0921 | 0.5137 | NumPy |
| 512×512 | 3 | 786432 | std | 0.4374 | 0.4891 | NumPy |
| 512×512 | 9 | 2359296 | sum | 0.2907 | 0.9783 | NumPy |
| 512×512 | 9 | 2359296 | mean | 0.2824 | 1.1411 | NumPy |
| 512×512 | 9 | 2359296 | std | 0.8898 | 1.1141 | NumPy |
| 1024×1024 | 1 | 1048576 | sum | 0.1297 | 0.4112 | NumPy |
| 1024×1024 | 1 | 1048576 | mean | 0.1201 | 0.6083 | NumPy |
| 1024×1024 | 1 | 1048576 | std | 0.5334 | 0.6320 | NumPy |
| 1024×1024 | 3 | 3145728 | sum | 0.3550 | 1.3315 | NumPy |
| 1024×1024 | 3 | 3145728 | mean | 0.3967 | 1.5113 | NumPy |
| 1024×1024 | 3 | 3145728 | std | 1.1522 | 1.5584 | NumPy |
| 1024×1024 | 9 | 9437184 | sum | 1.1848 | 3.9976 | NumPy |
| 1024×1024 | 9 | 9437184 | mean | 1.1929 | 4.7008 | NumPy |
| 1024×1024 | 9 | 9437184 | std | 3.6213 | 4.5765 | NumPy |

#### uint8 — NumPy global `mean`/`std` vs NumKong `moments` (global over all pixels)

| H×W | C | pixels | NumPy mean+std (ms) | NumKong moments+formula (ms) | faster |
|-----|---|--------|--------------------:|-----------------------------:|--------|
| 256×256 | 1 | 65536 | 0.1162 | 0.0052 | NumKong |
| 256×256 | 3 | 196608 | 0.3388 | 0.0112 | NumKong |
| 256×256 | 9 | 589824 | 1.0550 | 0.0316 | NumKong |
| 512×512 | 1 | 262144 | 0.4728 | 0.0148 | NumKong |
| 512×512 | 3 | 786432 | 1.2888 | 0.0425 | NumKong |
| 512×512 | 9 | 2359296 | 3.0038 | 0.2653 | NumKong |
| 1024×1024 | 1 | 1048576 | 1.7859 | 0.0583 | NumKong |
| 1024×1024 | 3 | 3145728 | 6.1183 | 0.3005 | NumKong |
| 1024×1024 | 9 | 9437184 | 18.2352 | 0.5224 | NumKong |

_Environment:_  `Darwin` `arm64`,  numkong 7.0.0,  numpy 2.4.2.  Median ms, 11 repeats, 3 warmup, seed=42.
