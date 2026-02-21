---
name: albucore-conventions
description: Albucore image processing conventions - shapes (H,W,C), dtypes (uint8/float32), backend routing. Use when implementing or modifying albucore functions, writing tests, or reviewing image-processing code.
---

# Albucore Conventions

## When to Apply

- Implementing new functions in `albucore/functions.py` or `albucore/geometric.py`
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

No float64. Raise `ValueError` for unsupported dtypes.

### 3. Backend Routing — Benchmark-Driven Only

- **Do not assume** LUT is fastest for uint8 — benchmark.
- OpenCV has 4-channel limit: use `MAX_OPENCV_WORKING_CHANNELS`, fall back to NumPy for >4 channels.
- Route based on `./benchmark.sh` results, not convention.

### 4. LUT Operations — Keep float32

When building LUTs from `cv2.meanStdDev` (returns float64), cast explicitly:

```python
# BAD: numpy promotes to float64
lut = (np.arange(256, dtype=np.float32) - mean) / std

# GOOD
lut = ((np.arange(256, dtype=np.float32) - mean) / std).astype(np.float32)
```

### 5. Utilities

- `get_num_channels(img)` — from `albucore.utils`
- `convert_value(value, num_channels)` — for per-channel values
- `@preserve_channel_dim`, `@contiguous`, `@clipped` — from `albucore.decorators`

### 6. Tests

- Test uint8 and float32 only
- Test single, batch, volume, batch-of-volumes
- Edge cases: 1 channel, >4 channels
- Run `./benchmark.sh` before changing backend routing

## Quick Ref

| Convention | Rule |
|------------|------|
| Grayscale shape | `(H, W, 1)` never `(H, W)` |
| Dtypes | uint8, float32 only |
| Backend | Choose by benchmark |
| LUT dtype | Always float32 |
| OpenCV limit | 4 channels max |
