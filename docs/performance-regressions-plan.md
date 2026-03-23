# Plan: address synthetic-router regressions vs 0.0.40

This doc ties **observed** `benchmarks/compare_router_json.py` deltas (ratio = `old_ms / new_ms`; **&lt;1 ⇒ new slower**) to **code paths** and next steps.

## 1. Float32 `normalize_per_image(..., "image")`

**Cause (fixed in tree):** `normalize_per_image_numpy` used **`stats.mean_std`** global for float32; 0.0.40 used raw **`img.mean()` / `img.std()`** + divide.

**Done:** **`normalize_per_image_numpy`** `"image"` branch restored to **raw ndarray mean/std**. **`_compute_image_stats_opencv`** for 3D global stats restored to **`img.mean()` / `img.std()`** (same as 0.0.40), not `mean_std`.

**Note:** Routing **all** float32 `normalize_per_image` through OpenCV **hurt** some router cells vs the NumPy `"image"` path — keep **NumPy** for `"image"` only.

## 2. Float32 `multiply` / `multiply_by_constant`

**Cause:** 0.0.40 used **`multiply_numpy`** for float32 scalars. Intermediate releases used NumKong / OpenCV and regressed vs that baseline.

**Done:** **`multiply_by_constant`** float32 → **`multiply_numpy`** (matches 0.0.40). **`multiply_by_constant_numkong`** remains for **`benchmark_multiply_add_numkong.py`**.

**Micro:** **`out=` / in-place** for `multiply_by_constant(..., inplace=True)` — measure first.

## 3. `pairwise_distances_squared` — vs 0.0.40 SimSimd

**Cause:** 0.0.40 used **SimSimd `cdist`** for `n1*n2 < 1000`. Current stack uses **NumKong `cdist`** (pure NumPy alone was slower on the router’s small case).

**Open:** Optional **`simsimd`** dependency + branch for small `cdist`, or accept gap until NumKong catches up on that shape class.

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
