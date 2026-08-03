---
name: albucore-conventions
description: Albucore image processing conventions - shapes (H,W,C), dtypes (uint8/float32), benchmark-driven backend routing (OpenCV, NumPy, Torch CPU, LUT, NumKong), tests, and lockfile discipline. Use when implementing or modifying albucore modules, writing tests, or reviewing image-processing code.
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

An API that explicitly accepts a `torch.Tensor` defines its layout independently; never infer a Tensor layout from its
rank. `resize3d` declares NumPy `DHWC` and Torch `CDHW`. `warp_affine3d` uses the same prevalidated single-volume
layouts. AlbumentationsX checks CPU, strided layout, eager (`requires_grad=False`) execution, and all control data
before the call. It does not accept `NDHWC` or `NCDHW` batch layouts.

### 2. Supported Dtypes - uint8 and float32 Only

No float64 in public paths. Raise `ValueError` for unsupported dtypes.

### 3. Backend Routing - Benchmark-Driven Only

- Do not assume LUT is fastest for uint8; benchmark.
- NumKong is used where benchmarks win (`blend`, `moments`, `scale`, `cdist`, etc.); see `docs/numkong-performance.md`.
- StringZilla is a candidate for uint8 translation paths; compare it with the public LUT router and OpenCV.
- OpenCV has a 4-channel limit for many ops. Use `MAX_OPENCV_WORKING_CHANNELS`, then fall back to NumPy or chunking for more channels.
- Route from benchmark evidence in `benchmarks/` and `docs/numkong-performance.md`, not convention.

### 4. Torch CPU Is a Mandatory Backend

- `torch>=2.13.0` is a required runtime dependency, not an optional import. Use direct imports rather than
  `sys.modules` checks, class-name heuristics, or lazy imports.
- Training callers are expected to have imported Torch already. Do not optimize public CPU routing around deferred
  import cost.
- A public Tensor path must state its layout. For caller-prevalidated routers such as `resize3d` and `warp_affine3d`,
  AlbumentationsX owns CPU, strided-layout, and `requires_grad=False` validation. Neither router silently detaches or
  moves data.
- Benchmark NumPy-to-Torch routes end-to-end: wrapper creation, permutations, dtype casts, kernel execution, and
  returned NumPy layout all belong inside the timed region.
- For `resize3d`, benchmark direct Tensor and zero-copy Tensor→NumPy→Tensor routes separately. A linear all-axis
  upscale may select the bridge for speed; preserve its documented float32 tolerance and uint8 delta bound.
- For `warp_affine3d`, benchmark the full single-volume path: matrix conversion, affine grid, sampling, nonzero-fill
  correction, uint8 conversion, NumPy/Torch views, and output materialization. Do not introduce a batch layout or an
  unmeasured manual-grid/native fallback.

### 5. OpenCV LUT - Source vs Table Dtype

- `cv2.LUT` source image: for uint8 LUT paths, pass a uint8 `(H, W, C)` image.
- `cv2.LUT` lookup table: for float outputs, the table must be float32, not float64. OpenCV stats often promote to float64; cast the small LUT to float32 so the output does not widen.

### 6. Normalize / Float Work - float32 Only

Keep intermediate buffers float32 unless a benchmark proves otherwise. Public API supports uint8 and float32 only.

### 7. Utilities and Decorators

- Utilities: `get_num_channels`, `convert_value`, `clip`, etc. in `albucore.utils`.
- Decorators: `@preserve_channel_dim`, `@contiguous`, `@clipped`, `@batch_transform`, etc. in `albucore.decorators`.

### 8. Tests

- Test uint8 and float32 only.
- Test single images, batches, volumes, and batch-of-volumes where the router supports them.
- Cover 1-channel and >4-channel edge cases.

### 9. Dependency Lock Consistency

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
| Torch Tensor route | Explicit layout; AlbumentationsX prevalidates CPU/strides/autograd before `warp_affine3d` |
| Benchmarks | `benchmarks/` and `docs/numkong-performance.md` |
| Lockfile | Keep `uv.lock` in sync with `pyproject.toml` |
