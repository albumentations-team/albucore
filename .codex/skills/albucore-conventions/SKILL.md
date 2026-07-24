---
name: albucore-conventions
description: Albucore image processing conventions - shapes (H,W,C), dtypes (uint8/float32), benchmark-driven backend routing (OpenCV, NumPy, LUT, NumKong), tests, and lockfile discipline. Use when implementing or modifying albucore modules, writing tests, or reviewing image-processing code.
---

# Albucore Conventions

For runtime implementation or review, also read `../performance-optimization/SKILL.md` and its canonical reference
completely before acting.

## When to Apply

- Implementing or changing code under `albucore/`.
- Adding or modifying tests.
- Reviewing image-processing code.
- Answering questions about shapes, dtypes, backend choice, or dependency lock consistency.

## Critical Rules

### 1. Image Shapes - Always Explicit Channel Dimension

```python
# Correct
grayscale = (H, W, 1)
rgb = (H, W, 3)
batch = (N, H, W, C)
volume = (D, H, W, C)
batch_volumes = (N, D, H, W, C)

# Wrong - never use implicit channels
gray = (H, W)
gray_vol = (D, H, W)
```

Dimension indexing is always:

```python
num_channels = image.shape[-1]
width = image.shape[-2]
height = image.shape[-3]
```

### 2. Supported Dtypes - uint8 and float32 Only

No float64 in public paths. Raise `ValueError` for unsupported dtypes.

### 3. Backend Routing - Benchmark-Driven Only

- Do not assume LUT is fastest for uint8; benchmark.
- NumKong is used where benchmarks win (`blend`, `moments`, `scale`, `cdist`, etc.); see `docs/numkong-performance.md`.
- StringZilla is a candidate for uint8 translation paths; compare it with the public LUT router and OpenCV.
- OpenCV has a 4-channel limit for many ops. Use `MAX_OPENCV_WORKING_CHANNELS`, then fall back to NumPy or chunking for more channels.
- Route from benchmark evidence in `benchmarks/` and `docs/numkong-performance.md`, not convention.

### 4. OpenCV LUT - Source vs Table Dtype

- `cv2.LUT` source image: for uint8 LUT paths, pass a uint8 `(H, W, C)` image.
- `cv2.LUT` lookup table: for float outputs, the table must be float32, not float64. OpenCV stats often promote to float64; cast the small LUT to float32 so the output does not widen.

### 5. Normalize / Float Work - float32 Only

Keep intermediate buffers float32 unless a benchmark proves otherwise. Public API supports uint8 and float32 only.

### 6. Utilities and Decorators

- Utilities: `get_num_channels`, `convert_value`, `clip`, etc. in `albucore.utils`.
- Decorators: `@preserve_channel_dim`, `@contiguous`, `@clipped`, `@batch_transform`, etc. in `albucore.decorators`.

### 7. Tests

- Test uint8 and float32 only.
- Test single images, batches, volumes, and batch-of-volumes where the router supports them.
- Cover 1-channel and >4-channel edge cases.

### 8. Dependency Lock Consistency

- When changing dependencies in `pyproject.toml`, update `uv.lock` in the same PR.
- Validate with `uv lock --check`.
- Release flow uses `uv export --frozen`; stale `uv.lock` can break release artifact generation.

## Quick Reference

| Convention | Rule |
|------------|------|
| Grayscale shape | `(H, W, 1)` never `(H, W)` |
| Dtypes | uint8, float32 only |
| Backend | Choose by benchmark |
| LUT | uint8 image in; float32 LUT table when output is float32 |
| Normalize / math | float32 buffers; no float64 in public paths |
| OpenCV limit | 4 channels unless chunked or using NumPy |
| Benchmarks | `benchmarks/` and `docs/numkong-performance.md` |
| Lockfile | Keep `uv.lock` in sync with `pyproject.toml` |
