# Plan: address synthetic-router regressions vs 0.0.40

This doc ties **observed** `benchmarks/compare_router_json.py` deltas (ratio = `old_ms / new_ms`; **&lt;1 ⇒ new slower**) to **code paths** and next steps.

## 1. Float32 `normalize_per_image(..., "image")` — large regression

**Symptom:** e.g. `(512,512,3)` float32 new ~1.42 ms vs old ~0.95 ms (ratio ~0.67).

**Cause:** Router sends float32 **`"image"`** to `normalize_per_image_numpy` → `mean_std` global → **two full `np.mean` / `np.std` passes** + Python arithmetic. Older stack used **`normalize_per_image_opencv`** → `cv2.meanStdDev` + `cv2` divide pipeline on 3D.

**Plan:**

1. Re-bench **OpenCV path vs current NumPy+stats** for float32, 3D only, all `H×W×C` in the synthetic grid (including 1024²).
2. If OpenCV wins on reference hardware, **route float32 + `ndim==3` + `"image"`** back to `normalize_per_image_opencv` (keep `stats` for uint8 / higher rank).
3. Optional: **fused** float32 global mean+var in one pass (Welford or single `np.add.reduce` on squares) *only if* it beats OpenCV when measured.

## 2. Float32 `multiply` / `multiply_by_constant` — regressions on some shapes

**Symptom:** e.g. `(256,256,3)` float ratio ~0.45 (new slower); uint8 often fine (LUT).

**Cause:** New path uses **NumKong `scale`** on ravel; old used **SimSimd `wsum`**-style multiply. Different constants, alignment, and backend tuning.

**Plan:**

1. Extend `benchmark_multiply_add_numkong.py` / router bench to include **1024²** and **C∈{1,3,9}**.
2. Add **shape-aware routing**: if `nk.scale` loses to OpenCV/NumPy on a cell, branch (thresholds from JSON, not guesses).
3. Consider **`out=` / in-place** where `multiply_by_constant(..., inplace=True)` is safe and avoids alloc (micro-gain; measure first).

## 3. `pairwise_distances_squared` — small regression on benchmark size

**Symptom:** Single row ratio ~0.76.

**Cause:** **24×16 = 384 &lt; 1000** → **NumKong `cdist`**; older code may have used **cv2** or pure NumPy for that size.

**Plan:**

1. Sweep **n1×n2** around 1000; adjust threshold or use fastest-of-three for small grids.
2. Ensure **contiguous float32** input (already) and avoid extra copies in hot path.

## 4. `hflip` noise / possible regression on large float RGB

**Symptom:** e.g. `(512,512,3)` float ratio ~0.54 (inconsistent with other cells).

**Plan:** Treat as **suspect** until repeated with fixed clock, more repeats, and **cold vs warm cache**. If real: check **contiguity** before `cv2.flip` (router already uses OpenCV); compare **`hflip_numpy`** (slice) vs **`cv2.flip`** on that shape.

## 5. UInt8 `normalize_per_image` — large *wins* (not regressions)

**Symptom:** ratio ≫1 (e.g. 6–9×).

**Cause:** **LUT** path vs older OpenCV-heavy stats+normalize.

**Plan:** Keep; use as **guardrail** when touching LUT or `stats` for uint8.

## 6. Engineering checklist (any change)

- [ ] Re-run `benchmark_router_synthetic.py` → JSON for **editable** and **pinned old**.
- [ ] `compare_router_json.py` → inspect **Summary** regressions block.
- [ ] Re-run targeted scripts: `benchmark_numkong_vs_albucore_backends.py`, `benchmark_multiply_add_numkong.py`, `benchmark_stats.py`.

## 7. In-place / vectorization (general)

- Prefer **OpenCV in-place** (`dst=`) only when API guarantees and **no** read-after-alias bugs.
- **Vectorize** per-channel Python loops (e.g. multiple `moments` calls) only after proving **batched** NumKong or NumPy beats current on target shapes.
- **Avoid** redundant `astype` / `copy` on full tensors; profile with **same** benchmark seeds.
