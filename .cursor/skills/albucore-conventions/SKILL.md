---
name: albucore-conventions
description: Albucore image processing conventions - shapes (H,W,C), dtypes (uint8/float32), benchmark-driven backend routing (OpenCV, NumPy, LUT, NumKong). Use when implementing or modifying albucore modules, writing tests, or reviewing image-processing code.
---

# Albucore Conventions

## When to Apply

- Implementing or changing code under **`albucore/`** (split modules: `arithmetic`, `normalize`, `convert`, `geometric`, `ops_misc`, `stats`, `weighted`, etc.; **`functions.py`** re-exports).
- Adding or modifying tests
- Reviewing image-processing code
- Questions about shapes, dtypes, or backend choice

## Critical Rules

### 1. Image Shapes — Always Explicit Channel Dimension

```python
# ✅ CORRECT
grayscale = (H, W, 1)
rgb = (H, W, 3)
batch = (N, H, W, C)
volume = (D, H, W, C)
batch_volumes = (N, D, H, W, C)

# ❌ WRONG — never use implicit channel
gray = (H, W)           # Missing channel!
gray_vol = (D, H, W)   # Missing channel!
```

**Dimension indexing (always valid):**

```python
num_channels = image.shape[-1]
width = image.shape[-2]
height = image.shape[-3]
```

### 2. Supported Dtypes — uint8 and float32 Only

No float64 in public paths. Raise `ValueError` for unsupported dtypes.

### 3. Backend Routing — Benchmark-Driven Only

- **Do not assume** LUT is fastest for uint8 — benchmark.
- **NumKong** is used where benches win (`blend`, `moments`, `scale`, `cdist`, etc.); see `docs/numkong-performance.md`.
- OpenCV has a **4-channel** limit for many ops: use `MAX_OPENCV_WORKING_CHANNELS`, fall back to NumPy (or chunking in `geometric.py`) for >4 channels.
- Route from **`benchmarks/`** scripts and `docs/numkong-performance.md`, not by convention.

### 4. OpenCV LUT — Inputs vs table dtype

- **`cv2.LUT` source image:** On the uint8 LUT path, pass a **uint8** `(H, W, C)` image (Albucore convention).
- **`cv2.LUT` lookup table:** For float LUTs, the table must be **float32**, not float64 — OpenCV stats often promote to float64; cast the small LUT to float32 so the **output** is not float64-wide. See `docs/performance-optimization.md`.

### 5. Normalize / float work — float32 only

Keep intermediate buffers **float32** (no float64 wideners) unless a benchmark proves otherwise. Public API supports **uint8** and **float32** only.

### 6. Utilities & Decorators

- `get_num_channels`, `convert_value`, `clip`, … — `albucore.utils`
- `@preserve_channel_dim`, `@contiguous`, `@clipped`, `@batch_transform`, … — `albucore.decorators`

### 7. Tests

- Test **uint8** and **float32** only
- Test single image, batch, volume, batch-of-volumes where the router supports it
- Edge cases: 1 channel, >4 channels

## Quick Ref

| Convention | Rule |
|------------|------|
| Grayscale shape | `(H, W, 1)` never `(H, W)` |
| Dtypes | uint8, float32 only |
| Backend | Choose by benchmark |
| LUT | uint8 image in; float32 LUT table (not float64) |
| Normalize / math | float32 buffers; no float64 in public paths |
| OpenCV limit | 4 channels unless chunked / NumPy |
| Benchmarks | `benchmarks/` (see skill **albucore-benchmarks**) |
