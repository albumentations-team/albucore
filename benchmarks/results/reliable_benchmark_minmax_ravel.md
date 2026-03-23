
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
    for h, w, c in [(224, 224, 3), (512, 512, 3), (1024, 1024, 1)]:
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
| float32 | 224×224×3 | 150528 | 0.0379 | 0.0125 | NumPy | 3.02× |
| float32 | 512×512×3 | 786432 | 0.1972 | 0.0633 | NumPy | 3.12× |
| float32 | 1024×1024×1 | 1048576 | 0.2628 | 0.0810 | NumPy | 3.24× |
| uint8 | 224×224×3 | 150528 | 0.0054 | 0.0041 | NumPy | 1.32× |
| uint8 | 512×512×3 | 786432 | 0.0273 | 0.0163 | NumPy | 1.68× |
| uint8 | 1024×1024×1 | 1048576 | 0.0366 | 0.0213 | NumPy | 1.72× |

**Takeaway (this machine):** For global min+max over a flat image buffer, **NumPy was faster**
than `Tensor.minmax()` on all tested sizes/dtypes. So for axis-wise `minmax` APIs,
downstream libs may still prefer NumPy (or batched native reductions) unless NumKong wins on
their exact strided/axis case.

_Environment: **Darwin** `arm64`, numkong 7.0.0, numpy 2.4.2, repeats=41, seed=0._

<!-- END COPY -->
