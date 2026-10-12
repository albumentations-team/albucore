# Albucore public API classification

`from albucore import *` exposes only **routers** and shared helpers listed in [`albucore/__init__.py`](../albucore/__init__.py) `__all__`.

## Precondition boundary

Public routers are low-level kernels and receive prevalidated inputs. Upstream callers own container,
rank/layout, dtype, device, contiguity, autograd, and operation-control validation. Albucore does not
repeat those checks on the hot path; NumPy/Torch dispatch selects the implementation, and backend code
may still normalize values or allocate the buffers required by its kernel.

## Containers, layouts, and dtypes

NumPy uses channel-last layouts: HWC/NHWC for images and DHWC/NDHWC for volumes. Eager CPU Torch uses channel-first
layouts: CHW/NCHW for images and CDHW/NCDHW for volumes. Common Tensor operations with channel-dependent controls
take keyword-only `layout=`; do not infer layout from rank. Spatial APIs define their layout by operation.

Image inputs support `uint8`, `float16`, and `float32`; float64 is not a supported image input. Most arithmetic and
geometric operations preserve the input dtype. `normalize`, `normalize_per_image`, and `to_float` return float16 for
float16 input and float32 for uint8/float32 input; `exp`, `log`, and `sqrt` return float32 for uint8 input. Statistics
use wide accumulators and may return float64. `apply_uint8_lut` and `sz_lut` remain uint8-only. A backend may compute
in float32 and restore float16 when needed; include those conversions in the full-path benchmark.

## Routers (package `__all__`)

User-facing entry points with benchmark-driven backends inside:

- **Arithmetic / weighted:** `add`, `add_constant`, `add_array`, `add_vector`, `add_weighted`, `multiply`, `multiply_by_constant`, `multiply_by_vector`, `multiply_by_array`, `multiply_add`, `power`, `normalize`, `normalize_per_image`
- **Elementwise math:** `exp`, `log`, `sqrt` (float16/float32 inputs, uint8 with float32 output, NumPy-compatible special values, exact shape preservation, safe owned-buffer `inplace`)
- **Uint8 LUT (`albucore.lut`):** `sz_lut`, `apply_uint8_lut` (StringZilla + OpenCV routing; re-exported on `albucore` for star imports)
- **I/O:** `to_float`, `from_float`
  Floating `from_float` targets cast without intensity scaling; Tensor inputs follow the prevalidated CPU,
  `requires_grad=False`, float16/float32 contract above. Large contiguous NumPy FP16↔FP32 casts use NumKong
  on the measured AVX-512 region when NumPy's floating-error policy is preserved; other cases retain NumPy.
  Large contiguous macOS ARM Tensor casts use NumPy
  storage conversion and zero-copy Tensor wrapping. Same-dtype calls preserve the input object.
- **Geometry / misc:** `hflip`, `vflip`, `median_blur`, `matmul`, `pairwise_distances_squared`
- **Stats:** `mean`, `std`, `mean_std`, `reduce_sum` (from `albucore.stats`)
- **Decorators:** see `decorators.__all__` in [`albucore/decorators.py`](../albucore/decorators.py)
- **Geometric:** `copy_make_border`, `flip_volume`, `gaussian_blur3d`, `remap`, `resize`, `resize3d`, `rot90_volume`, `separable_filter3d`, `transpose_volume`, `warp_affine`, `warp_affine3d`, `warp_perspective`.
  NumPy image layouts are HWC/NHWC and volume layouts are DHWC/NDHWC; CPU Tensor image layouts are CHW/NCHW and volume layouts are CDHW/NCDHW.
  Only `warp_affine3d` and `remap3d` accept volume batches, applying one shared matrix or grid to each item. Other 3D routers accept one volume per call.
- **3D dense resampling:** `remap3d` applies a caller-provided float32 `(D_out, H_out, W_out, 3)` normalized `align_corners=False` pull grid in `(x, y, z)` order to one NumPy `DHWC`/`NDHWC` array or CPU Torch `CDHW`/`NCDHW` tensor. For rank 5 the same grid is applied to every volume. The grid can be NumPy or Torch, its spatial shape defines the output, and coordinates outside `[-1, 1]` use constant or replicate borders. The router preserves the input container, dtype, channel count, and public layout. `remap3d` does not scan the dense grid for identity or convert a voxel-coordinate map.
- `warp_affine3d` and `remap3d` retain their `DHWC`/`CDHW` single-volume calls and also accept NumPy `NDHWC` and CPU Torch `NCDHW` batches. Other 3D routers continue to accept one volume per call.
- **3D constant padding:** `pad3d` takes NumPy DHWC or CPU Torch CDHW and spatial padding `(front, back, top, bottom, left, right)`. It supports uint8/float16/float32 volumes and int16 masks, preserves dtype and channel count, and follows NumPy fill semantics.
- **Utils:** see `utils.__all__` in [`albucore/utils.py`](../albucore/utils.py)
- **Types / constants:** `ImageInput`, `ImageType`, `ImageUInt8`, `ImageFloat16`, `ImageFloat32`, `SupportedDType`, `TensorLayout`, `NormalizationType`, `ValueType`, and shared routing constants.
- **Metadata:** `__version__`, `__author__`, `__maintainer__`

## Backend-specific imports (submodule-only)

Backend implementations stay outside the package star-import API. Import one explicitly when a test or specialized integration must select a backend:

- Backend-specific: `*_numpy`, `*_opencv`, `*_lut`, `*_cv2`, `hflip_numpy`, `vflip_numpy`; this includes `exp_numpy`/`exp_opencv`, `log_numpy`/`log_opencv`, and `sqrt_numpy`/`sqrt_opencv`
- NumKong helpers: `add_weighted_numkong`, `add_array_numkong`, `multiply_by_constant_numkong` (`albucore.functions`); `add_constant_numkong` lives on [`albucore.weighted`](../albucore/weighted.py) only
- LUT plumbing: `create_lut_array`, `apply_lut`, `prepare_value_opencv`, `apply_numpy`

## Internal

Names prefixed with `_` or used only inside albucore (not stable API).
