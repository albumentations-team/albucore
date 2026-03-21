# NumKong in albucore — what ships, what’s next, what we skip

Albucore picks backends by **measured** speed (OpenCV, NumPy, LUT, NumKong). This page maps **NumKong** to that policy: what is **already wired**, what we **may still add**, and where NumKong is **slower** or a **bad semantic match** so we **do not** route there.

**Regenerate numbers** (OpenCV required for `add_weighted` / `meanStdDev` baselines):

```bash
uv sync --extra headless
uv run python benchmarks/benchmark_numkong_vs_albucore_backends.py
uv run python benchmarks/benchmark_multiply_add_numkong.py   # multiply/add vs scale, fma, blend
uv run python benchmarks/benchmark_stats.py                  # mean_std vs NumPy reference (smoke timings)
```

**Reference run** (embedded tables below): Darwin `arm64`, Apple M4 Max, numkong 7.0.0, numpy 2.4.x, opencv-python-headless 4.13.x — median ms, 9 repeats, 3 warmup, seed **42**.

---

## Operation map

| Operation | Layouts in bench | dtypes | Status |
|-----------|------------------|--------|--------|
| `add_weighted` | Image `(H,W,C)`; batch `(N,H,W,C)` | uint8, float32 | **uint8** → `nk.blend`; **float32** → **`cv2.addWeighted`** (router vs 0.0.40 SimSimd `wsum`). |
| `pairwise_distances_squared` | Small `n1×n2` | float32 | **`nk.cdist`** if `n1*n2 < 1000`; else NumPy. Still slower than 0.0.40 **SimSimd** `cdist` on some sizes (no simsimd dep). |
| Global **mean** / **std** / **mean_std** | HWC, NHWC, NDHWC | uint8 | **Shipped** — [`albucore.stats`](../albucore/stats.py): global reduction uses **`nk.moments`** on a contiguous ravel (one pass for `mean_std`). |
| Global mean / std / both | same | float32 | **NumPy** in `stats` (`np.mean` / `np.std`, float64 accumulators); not routed to NumKong. |
| Per-channel **mean** / **std** / **mean_std** | `(H,W,C)`, `(N,H,W,C)`, … | uint8, float32 | **Shipped** in `stats` — 3D, `keepdims=False`: **`cv2.mean`** for **mean** only; **`cv2.meanStdDev`** for **std** / **mean_std** (joint mean+std); higher rank or `keepdims=True` → NumPy axis reduction. Normalization calls this via `_compute_per_channel_stats_opencv` → `mean_std(..., "per_channel")`. |
| **min** (global on ravel) | `(H,W,C)` | uint8, float32 | **Not NumKong** — NumPy faster ([`research/minmax-ravel-benchmark.md`](research/minmax-ravel-benchmark.md)). |
| **max** | same | same | **Not NumKong** — same bench as min. |
| **minmax** (`Tensor.minmax`) | same | same | **Not NumKong** — see [`research/minmax-ravel-benchmark.md`](research/minmax-ravel-benchmark.md). |
| `multiply_by_constant` | same | uint8, float32 | **float32 → NumPy** (`multiply_numpy`, same as 0.0.40); uint8 LUT. **`multiply_by_constant_numkong`** for microbenches only. |
| `add_constant` | same | uint8, float32 | **Keep OpenCV** — **`nk.scale`(1,β)** rarely wins (§2). |
| `multiply_by_array` | same | uint8, float32 | **Not `fma`** — **NumPy** `multiply_numpy` fastest vs OpenCV and `nk.fma` here (§2). |
| `add_array` | same | uint8, float32 | **Shipped** — `add_array` uses **`add_array_numkong`** when shapes/dtypes match and `inplace=False`; else OpenCV. |
| `multiply_by_vector` / `add_vector` | same | uint8, float32 | **Keep LUT/OpenCV** — channel-wise **`scale` loop** mixed vs one prod pass (§2). |

Scripts: **[`benchmarks/benchmark_numkong_vs_albucore_backends.py`](../benchmarks/benchmark_numkong_vs_albucore_backends.py)** (tables in §1–§3), **[`benchmarks/benchmark_multiply_add_numkong.py`](../benchmarks/benchmark_multiply_add_numkong.py)** (multiply/add vs `scale` / `fma` / `blend`), **[`benchmarks/benchmark_numkong.py`](../benchmarks/benchmark_numkong.py)** (`cdist` / blend / scale-fma microbenches), **[`benchmarks/benchmark_minmax_ravel.py`](../benchmarks/benchmark_minmax_ravel.py)**.

---

## 1. Already using NumKong (production)

### `add_weighted` — `nk.blend` (uint8) vs `cv2.addWeighted` (float32)

Production: **uint8** uses **`add_weighted_numkong`**; **float32** uses **`add_weighted_opencv`** (router vs 0.0.40 SimSimd). Microbench tables below still compare NK / OpenCV / NumPy.

Weights **0.5 / 0.5**. **Image** `(H,W,C)` — H×W ∈ {256, 512, 1024}, C ∈ {1, 3, 9}. **Batch / video** `(N,H,W,C)` with **N=4**, H=W=256, same C.

#### Image — uint8 (incl. LUT baseline)

| H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |
|-----|---|--------|--------:|-------:|------:|----:|------------:|---------------:|
| 256×256 | 1 | 65536 | 0.0047 | 0.0133 | 0.0507 | 0.0422 | OpenCV (0.0133) | NK 2.84× faster than OpenCV |
| 256×256 | 3 | 196608 | 0.0118 | 0.0299 | 0.1642 | 0.1631 | OpenCV (0.0299) | NK 2.53× faster than OpenCV |
| 256×256 | 9 | 589824 | 0.0375 | 0.0954 | 0.9108 | 0.6250 | OpenCV (0.0954) | NK 2.54× faster than OpenCV |
| 512×512 | 1 | 262144 | 0.0176 | 0.0420 | 0.2386 | 0.1404 | OpenCV (0.0420) | NK 2.39× faster than OpenCV |
| 512×512 | 3 | 786432 | 0.0501 | 0.1185 | 1.1933 | 0.4463 | OpenCV (0.1185) | NK 2.37× faster than OpenCV |
| 512×512 | 9 | 2359296 | 0.2845 | 0.4193 | 1.7477 | 0.6071 | OpenCV (0.4193) | NK 1.47× faster than OpenCV |
| 1024×1024 | 1 | 1048576 | 0.0668 | 0.1583 | 1.7400 | 0.8280 | OpenCV (0.1583) | NK 2.37× faster than OpenCV |
| 1024×1024 | 3 | 3145728 | 0.3116 | 0.6182 | 2.3463 | 0.6930 | OpenCV (0.6182) | NK 1.98× faster than OpenCV |
| 1024×1024 | 9 | 9437184 | 0.6007 | 1.3652 | 10.0505 | 3.0929 | OpenCV (1.3652) | NK 2.27× faster than OpenCV |

#### Image — float32 (no LUT)

| H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |
|-----|---|--------|--------:|-------:|------:|------------:|---------------:|
| 256×256 | 1 | 65536 | 0.0070 | 0.0132 | 0.0140 | OpenCV (0.0132) | NK 1.87× faster than OpenCV |
| 256×256 | 3 | 196608 | 0.0212 | 0.0345 | 0.0436 | OpenCV (0.0345) | NK 1.63× faster than OpenCV |
| 256×256 | 9 | 589824 | 0.1323 | 0.1997 | 0.3861 | OpenCV (0.1997) | NK 1.51× faster than OpenCV |
| 512×512 | 1 | 262144 | 0.0251 | 0.0468 | 0.0573 | OpenCV (0.0468) | NK 1.86× faster than OpenCV |
| 512×512 | 3 | 786432 | 0.2334 | 0.2574 | 0.3950 | OpenCV (0.2574) | NK 1.10× faster than OpenCV |
| 512×512 | 9 | 2359296 | 0.2433 | 0.4349 | 0.4399 | OpenCV (0.4349) | NK 1.79× faster than OpenCV |
| 1024×1024 | 1 | 1048576 | 0.3186 | 0.4156 | 0.7695 | OpenCV (0.4156) | NK 1.30× faster than OpenCV |
| 1024×1024 | 3 | 3145728 | 0.2649 | 0.5590 | 0.6670 | OpenCV (0.5590) | NK 2.11× faster than OpenCV |
| 1024×1024 | 9 | 9437184 | 3.7712 | 1.6452 | 5.0307 | OpenCV (1.6452) | OpenCV 2.29× faster than NK |

#### Batch / video `(N,H,W,C)`, N=4, H×W=256×256

| N×H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |
|-------|---|--------|--------:|-------:|------:|----:|------------:|---------------:|
| 4×256×256 | 1 | 262144 | 0.0173 | 0.0405 | 0.2700 | 0.2410 | OpenCV (0.0405) | NK 2.34× faster than OpenCV |
| 4×256×256 | 3 | 786432 | 0.0498 | 0.1088 | 1.3242 | 0.9672 | OpenCV (0.1088) | NK 2.19× faster than OpenCV |
| 4×256×256 | 9 | 2359296 | 0.3093 | 0.4079 | 1.7228 | 1.4578 | OpenCV (0.4079) | NK 1.32× faster than OpenCV |

| N×H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |
|-------|---|--------|--------:|-------:|------:|------------:|---------------:|
| 4×256×256 | 1 | 262144 | 0.0284 | 0.0468 | 0.0665 | OpenCV (0.0468) | NK 1.65× faster than OpenCV |
| 4×256×256 | 3 | 786432 | 0.2083 | 0.2816 | 0.4767 | OpenCV (0.2816) | NK 1.35× faster than OpenCV |
| 4×256×256 | 9 | 2359296 | 0.2665 | 0.4092 | 0.5289 | OpenCV (0.4092) | NK 1.54× faster than OpenCV |

---

### `pairwise_distances_squared` — `nk.cdist` (small point sets)

Albucore uses **`nk.cdist`**, metric **`sqeuclidean`**, when **`n1 * n2 < 1000`**; otherwise the existing NumPy vectorized formula. OpenCV is not on this hot path. Size sweep: **[`benchmarks/benchmark_numkong.py`](../benchmarks/benchmark_numkong.py)** (`cdist` section).

---

### Global uint8 statistics — `nk.moments` (scalar over all pixels)

**Implemented** in [`albucore.stats`](../albucore/stats.py): global uint8 **`mean`**, **`std`**, and **`mean_std`** use **`nk.moments`** on a contiguous ravel (**any** `ndim`). [`normalize._compute_image_stats_opencv`](../albucore/normalize.py) is a thin wrapper around **`mean_std(img, "global")`**. Below: **mean-only** and **std-only** timed separately; **OpenCV** `meanStdDev` is a valid global scalar baseline only for **C=1** (for **C>1** it is per-channel → N/A). OpenCV always computes mean+std internally, so **C=1** shows the **same full-call cost** in both tables.

#### Image — global mean only (uint8)

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0251 | 0.0046 | 0.0117 | NumKong |
| 256×256 | 3 | 196608 | 0.0728 | 0.0127 | N/A | NumKong |
| 256×256 | 9 | 589824 | 0.2195 | 0.0315 | N/A | NumKong |
| 512×512 | 1 | 262144 | 0.1005 | 0.0163 | 0.0449 | NumKong |
| 512×512 | 3 | 786432 | 0.2971 | 0.0559 | N/A | NumKong |
| 512×512 | 9 | 2359296 | 0.8730 | 0.2435 | N/A | NumKong |
| 1024×1024 | 1 | 1048576 | 0.3688 | 0.0577 | 0.1836 | NumKong |
| 1024×1024 | 3 | 3145728 | 1.2061 | 0.2863 | N/A | NumKong |
| 1024×1024 | 9 | 9437184 | 3.6769 | 0.5555 | N/A | NumKong |

#### Image — global std only (uint8)

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0703 | 0.0051 | 0.0128 | NumKong |
| 256×256 | 3 | 196608 | 0.2796 | 0.0113 | N/A | NumKong |
| 256×256 | 9 | 589824 | 0.8127 | 0.0395 | N/A | NumKong |
| 512×512 | 1 | 262144 | 0.3625 | 0.0182 | 0.0451 | NumKong |
| 512×512 | 3 | 786432 | 1.0513 | 0.0434 | N/A | NumKong |
| 512×512 | 9 | 2359296 | 2.3024 | 0.2152 | N/A | NumKong |
| 1024×1024 | 1 | 1048576 | 1.4475 | 0.0578 | 0.1760 | NumKong |
| 1024×1024 | 3 | 3145728 | 5.1997 | 0.2963 | N/A | NumKong |
| 1024×1024 | 9 | 9437184 | 15.7190 | 0.5626 | N/A | NumKong |

**Global mean + std together:** one **`mean_std`** call → `_global_mean_std_uint8` (one `moments` pass) — not a separate row in the script; both scalars come from the same kernel.

#### Batch / video — global mean & std (uint8), `4×256×256×C`

Same semantics: one scalar mean/std over **all** elements.

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.0989 | 0.0150 | 0.0452 | NumKong |
| 4×256×256 | 3 | 786432 | 0.3065 | 0.0548 | N/A | NumKong |
| 4×256×256 | 9 | 2359296 | 0.8804 | 0.2587 | N/A | NumKong |

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.3914 | 0.0150 | 0.0451 | NumKong |
| 4×256×256 | 3 | 786432 | 1.0375 | 0.0447 | N/A | NumKong |
| 4×256×256 | 9 | 2359296 | 2.3299 | 0.2016 | N/A | NumKong |

---

## 2. Planned / under review (benchmarks first)

- **Per-channel stats → NumKong:** production uses OpenCV / NumPy via **`stats.mean` / `stats.std` / `stats.mean_std`** (`axis='per_channel'`). Benchmarks show **C × `moments`** can win on **uint8** for some shapes; no default switch until winners are pinned per layout/dtype.
- **Float32 global `mean_std`:** still **two NumPy passes** (`np.mean` + `np.std`); a single-pass alternative would need benchmarking vs accuracy requirements.
- **Multiply / add (NumKong):** **`add_array_numkong`** uses **`blend`**; **`multiply_by_constant_numkong`** / **`add_constant_numkong`** use **`nk.scale`**. Production **`multiply_by_constant`** (float32) uses **`multiply_numpy`** (0.0.40 baseline); **`add_constant`** stays **OpenCV**. Raw **`fma`** is rarely the win for full-array multiply (see table below).

### Multiply / add — `nk.scale`, `nk.fma`, vs `nk.blend`

**Question:** Should `multiply_by_constant`, `add_constant`, `multiply_by_array`, `add_array`, `multiply_by_vector`, `add_vector` use NumKong via **`scale`** (α·x+β), **`fma`** (α·a·b+β·c), or **`blend`** (`add_weighted_numkong`–style helpers)?

**Benchmark:** [`benchmarks/benchmark_multiply_add_numkong.py`](../benchmarks/benchmark_multiply_add_numkong.py) — same H×W / C grid as the main tables; compares **production** `@clipped` APIs vs NumKong (including `multiply_by_constant_numkong` and `add_array_numkong`).

| Public-style op | NumKong mapping | Verdict (reference Mac) |
|-----------------|-----------------|-------------------------|
| **`multiply_by_constant`** | **`multiply_by_constant_numkong`** → **`nk.scale`(α=value, β=0)** | **Production: NumPy** (`multiply_numpy`) — matches 0.0.40; NK helper in **`weighted`** for benches. |
| **`add_constant`** | `nk.scale(α=1, β=scalar)` | **Keep OpenCV** — no reliable `scale` win. |
| **`multiply_by_array`** | `nk.fma` vs OpenCV vs **NumPy** | **Do not use `fma`** — **NumPy** wins; consider OpenCV→NumPy separately, not NumKong. |
| **`add_array`** | **`add_array_numkong`** (`blend`) | **Shipped** when `value.shape == img.shape`, dtypes match, `inplace=False`, and dtype is uint8/float32; else OpenCV. |
| **`multiply_by_vector`**, **`add_vector`** | C× **`scale`** loop vs LUT/OpenCV | **Keep production** — NK loop is mixed and loses on large float paths. |

**Note:** [`benchmarks/benchmark_numkong.py`](../benchmarks/benchmark_numkong.py) also has a small 1D **`scale` / `fma`** sweep; the multiply/add script uses **real `(H,W,C)`** shapes.

### Per-channel — mean only, std only, mean+std (same table)

Reduce over all axes except **channel** (`shape[-1]`). Columns: **NP mean**, **NP std**, **NP both** (one block), **albucore** (`mean_std(..., "per_channel")`, same path as `normalize`’s stats helpers), **NK** (one `moments` per channel). Layouts: image `(H,W,C)`; video **`4×256×256×C`**; volume **`2×4×64×64×C`**.

| dtype | layout (…×C) | C | pixels | NP mean | NP std | NP both | albucore | NK (C×moments) |
|-------|----------------|---|--------|--------:|-------:|--------:|---------:|---------------:|
| uint8 | 256×256×1 | 1 | 65536 | 0.0244 | 0.0707 | 0.1610 | 0.0136 | 0.0065 |
| float32 | 256×256×1 | 1 | 65536 | 0.0094 | 0.0275 | 0.0408 | 0.0537 | 0.0298 |
| uint8 | 256×256×3 | 3 | 196608 | 0.3881 | 1.1473 | 1.5251 | 0.0332 | 0.0767 |
| float32 | 256×256×3 | 3 | 196608 | 0.3359 | 0.8984 | 1.2569 | 0.0653 | 0.1820 |
| uint8 | 256×256×9 | 9 | 589824 | 0.5077 | 1.6791 | 2.1655 | 0.1265 | 0.1804 |
| float32 | 256×256×9 | 9 | 589824 | 0.3298 | 1.1115 | 1.4401 | 0.2026 | 0.4877 |
| uint8 | 512×512×1 | 1 | 262144 | 0.1100 | 0.3956 | 0.5133 | 0.0455 | 0.0157 |
| float32 | 512×512×1 | 1 | 262144 | 0.0332 | 0.1000 | 0.1349 | 0.2113 | 0.1179 |
| uint8 | 512×512×3 | 3 | 786432 | 1.5236 | 4.5968 | 6.2454 | 0.1282 | 0.2700 |
| float32 | 512×512×3 | 3 | 786432 | 1.3319 | 3.8765 | 5.2197 | 0.2583 | 0.6404 |
| uint8 | 512×512×9 | 9 | 2359296 | 2.0401 | 5.5273 | 7.5120 | 0.4625 | 0.8221 |
| float32 | 512×512×9 | 9 | 2359296 | 1.3100 | 3.9852 | 5.2486 | 0.6926 | 2.3054 |
| uint8 | 1024×1024×1 | 1 | 1048576 | 0.4213 | 1.5304 | 1.8497 | 0.1771 | 0.0605 |
| float32 | 1024×1024×1 | 1 | 1048576 | 0.1291 | 0.5658 | 0.6907 | 0.9167 | 0.7440 |
| uint8 | 1024×1024×3 | 3 | 3145728 | 6.2873 | 18.8581 | 25.1687 | 0.5175 | 1.0843 |
| float32 | 1024×1024×3 | 3 | 3145728 | 5.4393 | 14.2499 | 19.7864 | 0.9618 | 3.7320 |
| uint8 | 1024×1024×9 | 9 | 9437184 | 8.1107 | 28.3710 | 36.4098 | 1.8178 | 3.3476 |
| float32 | 1024×1024×9 | 9 | 9437184 | 5.2167 | 19.4685 | 24.6047 | 2.7157 | 14.0420 |
| uint8 | 4×256×256×1 | 1 | 262144 | 0.1003 | 0.3490 | 0.5089 | 0.4682 | 0.0171 |
| float32 | 4×256×256×1 | 1 | 262144 | 0.0460 | 0.1012 | 0.1345 | 0.1391 | 0.1258 |
| uint8 | 4×256×256×3 | 3 | 786432 | 1.5297 | 4.6668 | 6.2846 | 6.1552 | 0.2754 |
| float32 | 4×256×256×3 | 3 | 786432 | 1.3325 | 3.9888 | 5.2042 | 5.2435 | 0.6796 |
| uint8 | 4×256×256×9 | 9 | 2359296 | 2.0325 | 5.4815 | 7.5856 | 7.6900 | 0.8759 |
| float32 | 4×256×256×9 | 9 | 2359296 | 1.2910 | 3.8923 | 5.2229 | 5.1452 | 2.0057 |
| uint8 | 2×4×64×64×1 | 1 | 32768 | 0.0131 | 0.0424 | 0.0581 | 0.0706 | 0.0030 |
| float32 | 2×4×64×64×1 | 1 | 32768 | 0.0062 | 0.0188 | 0.0249 | 0.0254 | 0.0212 |
| uint8 | 2×4×64×64×3 | 3 | 98304 | 0.1863 | 0.5692 | 0.7377 | 0.7546 | 0.0353 |
| float32 | 2×4×64×64×3 | 3 | 98304 | 0.1669 | 0.4373 | 0.6297 | 0.6131 | 0.1140 |
| uint8 | 2×4×64×64×9 | 9 | 294912 | 0.2540 | 0.8369 | 1.0639 | 1.1000 | 0.1047 |
| float32 | 2×4×64×64×9 | 9 | 294912 | 0.1650 | 0.4739 | 0.6517 | 0.6445 | 0.3531 |

---

## 3. Not using NumKong (slower or wrong tool)

### Global float32 — mean only, std only (image + batch)

**Keep NumPy** for global float statistics. OpenCV note same as above (C=1 only for global scalar; joint mean+std cost).

#### Image — global mean only (float32)

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0096 | 0.0289 | 0.0533 | NumPy |
| 256×256 | 3 | 196608 | 0.0265 | 0.0873 | N/A | NumPy |
| 256×256 | 9 | 589824 | 0.0780 | 0.3241 | N/A | NumPy |
| 512×512 | 1 | 262144 | 0.0339 | 0.1317 | 0.2335 | NumPy |
| 512×512 | 3 | 786432 | 0.0970 | 0.5475 | N/A | NumPy |
| 512×512 | 9 | 2359296 | 0.2952 | 1.1287 | N/A | NumPy |
| 1024×1024 | 1 | 1048576 | 0.1202 | 0.6659 | 0.8805 | NumPy |
| 1024×1024 | 3 | 3145728 | 0.3735 | 1.4854 | N/A | NumPy |
| 1024×1024 | 9 | 9437184 | 1.1840 | 4.5857 | N/A | NumPy |

#### Image — global std only (float32)

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0302 | 0.0334 | 0.0640 | NumPy |
| 256×256 | 3 | 196608 | 0.0772 | 0.0955 | N/A | NumPy |
| 256×256 | 9 | 589824 | 0.3234 | 0.3932 | N/A | NumPy |
| 512×512 | 1 | 262144 | 0.1022 | 0.1206 | 0.2267 | NumPy |
| 512×512 | 3 | 786432 | 0.4366 | 0.4985 | N/A | NumPy |
| 512×512 | 9 | 2359296 | 0.9103 | 1.0887 | N/A | NumPy |
| 1024×1024 | 1 | 1048576 | 0.5625 | 0.7188 | 0.8525 | NumPy |
| 1024×1024 | 3 | 3145728 | 1.1457 | 1.4686 | N/A | NumPy |
| 1024×1024 | 9 | 9437184 | 6.5964 | 4.5752 | N/A | NumKong |

*(Last cell: single NumKong win at **1024²×9** std — treat as noise unless you care about that slice; default stays **NumPy**.)*

#### Batch — global mean / std only (float32), `4×256×256×C`

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.0309 | 0.1165 | 0.2238 | NumPy |
| 4×256×256 | 3 | 786432 | 0.0971 | 0.5036 | N/A | NumPy |
| 4×256×256 | 9 | 2359296 | 0.3293 | 1.0948 | N/A | NumPy |

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.1066 | 0.1140 | 0.2153 | NumPy |
| 4×256×256 | 3 | 786432 | 0.3841 | 0.4982 | N/A | NumPy |
| 4×256×256 | 9 | 2359296 | 0.9177 | 1.1164 | N/A | NumPy |

---

### `add_weighted` — exception on default path

For **float32**, **1024×1024**, **C=9**, **OpenCV** beats NumKong on this run (see §1 image float32 table). We still **default to NumKong** everywhere in code today; **optional** future work is **shape-aware routing** to OpenCV for that corner (benchmark-driven).

---

### Global min, max, combined minmax on image ravel

**NumPy** `min` / `max` / paired min+max beat **NumKong `Tensor.minmax()`** on contiguous ravels in our tests — write-up [`research/minmax-ravel-benchmark.md`](research/minmax-ravel-benchmark.md); generator **[`benchmarks/benchmark_minmax_ravel.py`](../benchmarks/benchmark_minmax_ravel.py)**.

---

### `pairwise_distances_squared` — large `n1×n2`

For **`n1 * n2 ≥ 1000`**, albucore keeps the **NumPy** vectorized path (not NumKong). Small-set timings: **[`benchmarks/benchmark_numkong.py`](../benchmarks/benchmark_numkong.py)**.

---

## Related

- [Performance optimization](performance-optimization.md)
- [Image conventions](image-conventions.md)
