### Benchmark: `Tensor.minmax()` vs `np.min` + `np.max` on image-shaped data

**Context:** Evaluation for image augmentation / vision pipelines (e.g. Albucore). We compared NumKong `Tensor.minmax()` to two-pass NumPy (`min` + `max`) on **realistic `(H, W, C)`** layouts (channel-last), **C-contiguous**, flattened to 1D — i.e. **global** min and max over all pixels, not per-channel axis reductions.

**Takeaway (below hardware):** On this setup, **NumPy was faster** for every shape/dtype tested. We are **not** planning to route global min+max through NumKong `minmax()` for contiguous flat buffers unless future versions or other CPUs change that.

---

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
for dtype, name in [(np.float32, "float32"), (np.uint8, "uint8")]:
    for h, w, c in [(224, 224, 3), (512, 512, 3), (1024, 1024, 1)]:
        if dtype == np.uint8:
            img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        else:
            img = rng.random((h, w, c), dtype=np.float32)
        flat = np.ascontiguousarray(img).reshape(-1)
        t = nk.Tensor(flat)
        t_nk = median_ms(lambda: t.minmax())
        t_np = median_ms(lambda: (flat.min(), flat.max()))
        ratio = max(t_nk, t_np) / min(t_nk, t_np)
        winner = "NumPy" if t_np <= t_nk else "NumKong"
        print(f"{name}  {h}x{w}x{c}  n={flat.size}  nk={t_nk:.4f}ms  np={t_np:.4f}ms  -> {winner}  {ratio:.2f}x")
```

</details>

---

#### Results (median ms, 11 repeats after 3 warmup, seed=0)

| dtype   | shape (H×W×C)   | pixels | NumKong `minmax` (ms) | `np.min` + `np.max` (ms) | faster | ratio |
|---------|-----------------|--------|----------------------:|-------------------------:|--------|------:|
| float32 | 224 × 224 × 3   | 150528 | 0.0336                | 0.0112                   | NumPy  | 3.00× |
| float32 | 512 × 512 × 3   | 786432 | 0.1746                | 0.0543                   | NumPy  | 3.22× |
| float32 | 1024 × 1024 × 1 | 1048576 | 0.2328               | 0.0728                   | NumPy  | 3.20× |
| uint8   | 224 × 224 × 3   | 150528 | 0.0048                | 0.0038                   | NumPy  | 1.24× |
| uint8   | 512 × 512 × 3   | 786432 | 0.0270                | 0.0145                   | NumPy  | 1.86× |
| uint8   | 1024 × 1024 × 1 | 1048576 | 0.0325               | 0.0210                   | NumPy  | 1.55× |

---

**Environment**

- OS: Darwin, `arm64`
- CPU: Apple M4 Max
- `numkong` 7.0.0
- `numpy` 2.4.2

**Caveat:** This does **not** benchmark axis-wise `minmax` (e.g. per column / per channel along spatial axes). NumKong may still win on other access patterns (e.g. strided column reductions); this table is only **global min+max on a contiguous 1D view of an image tensor**.
