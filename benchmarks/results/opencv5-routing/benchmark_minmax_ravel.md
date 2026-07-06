
### Benchmark: `Tensor.minmax()` vs `np.min` + `np.max` (image-shaped data)

**Setup:** Random `(H, W, C)` arrays, **C-contiguous**, `reshape(-1)` then one NumKong `Tensor` view.
Timing median over repeated calls (warmup first). **Not** an axis-wise per-channel reduction —
this is “global min + global max over all pixels” like a single 1D buffer.

<details>
<summary>Reference snippet (Python)</summary>

```python
import time
import numkong as nk
import numpy as np
from shape_grids import MINMAX_GLOBAL_HWC_SHAPES

def median_ms(fn, repeats=11, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))

rng = np.random.default_rng(0)
for dtype, name in [(np.float32, 'float32'), (np.uint8, 'uint8')]:
    for h, w, c in MINMAX_GLOBAL_HWC_SHAPES:
        if dtype == np.uint8:
            img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        else:
            img = rng.random((h, w, c), dtype=np.float32)
        flat = np.ascontiguousarray(img).reshape(-1)
        t = nk.Tensor(flat)
        t_nk = median_ms(lambda: t.minmax())
        t_np = median_ms(lambda: (flat.min(), flat.max()))
        print(name, (h, w, c), 'n=', flat.size, 'ms nk', t_nk, 'np', t_np)
```

</details>

| dtype | shape (H×W×C) | pixels | NumKong `minmax` (ms) | `np.min`+`np.max` (ms) | faster | ratio |
|-------|---------------|--------|----------------------:|-----------------------:|--------|------:|
| float32 | 224×224×3 | 150528 | 0.0388 | 0.0131 | NumPy | 2.96× |
| float32 | 512×512×3 | 786432 | 0.2026 | 0.0636 | NumPy | 3.18× |
| float32 | 1024×1024×1 | 1048576 | 0.2628 | 0.0848 | NumPy | 3.10× |
| uint8 | 224×224×3 | 150528 | 0.0053 | 0.0048 | NumPy | 1.11× |
| uint8 | 512×512×3 | 786432 | 0.0269 | 0.0191 | NumPy | 1.41× |
| uint8 | 1024×1024×1 | 1048576 | 0.0360 | 0.0253 | NumPy | 1.42× |

**Takeaway (this machine):** For global min+max over a flat image buffer, **NumPy was faster**
than `Tensor.minmax()` on all tested sizes/dtypes. So for axis-wise `minmax` APIs,
downstream libs may still prefer NumPy (or batched native reductions) unless NumKong wins on
their exact strided/axis case.

### Benchmark: per-channel min+max, NumPy axis reduce vs `cv2.reduce`

**Setup:** Channel-last arrays, per-channel min/max over all axes except the last.

| dtype | shape | NumPy min+max (ms) | OpenCV `reduce` min+max (ms) | faster | ratio |
|-------|-------|-------------------:|-----------------------------:|--------|------:|
| float32 | 128×128×1 | 0.0021 | 0.0184 | NumPy | 8.82× |
| float32 | 128×128×3 | 0.2701 | 0.1321 | OpenCV | 2.05× |
| float32 | 128×128×9 | 0.2628 | 0.1320 | OpenCV | 1.99× |
| float32 | 512×512×3 | 4.2553 | 1.6818 | OpenCV | 2.53× |
| float32 | 512×512×9 | 4.2713 | 1.4829 | OpenCV | 2.88× |
| float32 | 1024×1024×3 | 17.1767 | 6.2731 | OpenCV | 2.74× |
| float32 | 1024×1024×9 | 17.1557 | 5.7971 | OpenCV | 2.96× |
| float32 | 4×256×256×3 | 4.2949 | 1.6659 | OpenCV | 2.58× |
| float32 | 4×256×256×9 | 4.3371 | 1.1385 | OpenCV | 3.81× |
| uint8 | 128×128×1 | 0.0014 | 0.0610 | NumPy | 43.08× |
| uint8 | 128×128×3 | 0.1710 | 0.1029 | OpenCV | 1.66× |
| uint8 | 128×128×9 | 0.2207 | 0.1620 | OpenCV | 1.36× |
| uint8 | 512×512×3 | 2.8779 | 1.0377 | OpenCV | 2.77× |
| uint8 | 512×512×9 | 3.6188 | 1.6285 | OpenCV | 2.22× |
| uint8 | 1024×1024×3 | 11.4569 | 4.0816 | OpenCV | 2.81× |
| uint8 | 1024×1024×9 | 14.4934 | 5.1937 | OpenCV | 2.79× |
| uint8 | 4×256×256×3 | 2.8684 | 1.0504 | OpenCV | 2.73× |
| uint8 | 4×256×256×9 | 3.5908 | 1.4685 | OpenCV | 2.45× |

_Environment: **Darwin** `arm64`, numkong 7.7.0, numpy 2.2.6, repeats=11, seed=0._

<!-- END COPY -->
