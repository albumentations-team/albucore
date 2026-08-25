# Albucore public API classification

`from albucore import *` exposes only **routers** and shared helpers listed in [`albucore/__init__.py`](../albucore/__init__.py) `__all__`.

## Precondition boundary

Public routers are low-level kernels and receive prevalidated inputs. Upstream callers own container,
rank/layout, dtype, device, contiguity, autograd, and operation-control validation. Albucore does not
repeat those checks on the hot path; NumPy/Torch dispatch selects the implementation, and backend code
may still normalize values or allocate the buffers required by its kernel.

## Routers (package `__all__`)

User-facing entry points with benchmark-driven backends inside:

- **Arithmetic / weighted:** `add`, `add_constant`, `add_array`, `add_vector`, `add_weighted`, `multiply`, `multiply_by_constant`, `multiply_by_vector`, `multiply_by_array`, `multiply_add`, `power`, `normalize`, `normalize_per_image`
- **Elementwise float32:** `exp`, `log`, `sqrt` (NumPy-compatible special values, exact shape preservation, safe owned-buffer `inplace`)
- **Uint8 LUT (`albucore.lut`):** `sz_lut`, `apply_uint8_lut` (StringZilla + OpenCV routing; re-exported on `albucore` for star imports)
- **I/O:** `to_float`, `from_float`
- **Geometry / misc:** `hflip`, `vflip`, `median_blur`, `matmul`, `pairwise_distances_squared`
- **Stats:** `mean`, `std`, `mean_std`, `reduce_sum` (from `albucore.stats`)
- **Decorators:** see `decorators.__all__` in [`albucore/decorators.py`](../albucore/decorators.py)
- **Geometric:** `copy_make_border`, `flip_volume`, `gaussian_blur3d`, `remap`, `resize`, `resize3d`, `rot90_volume`, `separable_filter3d`, `transpose_volume`, `warp_affine`, `warp_affine3d`, `warp_perspective`. `flip_volume`, `transpose_volume`, and `rot90_volume` apply prevalidated NumPy `DHWC` or CPU Torch `CDHW` volume geometry primitives. Their integer axes are passed directly to the selected NumPy or Torch operation: `flip_volume` takes one or more axes, `transpose_volume` takes two, and `rot90_volume` takes an ordered pair. The native calls preserve container, supported uint8/float32 dtype, and native NumPy/Torch stride semantics. For `transpose_volume` and `rot90_volume`, caller-selected raw axes determine output shape and layout. Contiguity is the caller's decision. `gaussian_blur3d` and `separable_filter3d` filter one prevalidated NumPy `DHWC` or CPU Torch `CDHW` volume through three float32 D/H/W passes with fixed `BORDER_REFLECT_101` semantics, then restore uint8 once. They preserve their supported uint8 and float32 dtypes; callers reject other dtypes before dispatch. `gaussian_blur3d` accepts scalar or `(D,H,W)` sigma and kernel-size controls; zero sigma skips that axis. Target-level mask behavior is outside these single-volume primitives. `resize3d` expects prevalidated NumPy `DHWC` or Torch `CDHW`, supports uint8/float32, and runs Tensor kernels in inference mode. Large linear all-axis Tensor upscales (at least 10,000 output elements) use a zero-copy NumPy/OpenCV bridge selected by benchmark; float32 remains within `rtol=2e-4`, `atol=3e-5` of native Torch and uint8 differs by at most one value. `warp_affine3d` applies one forward voxel-space `(x, y, z)` matrix to one prevalidated NumPy `DHWC` or CPU Torch `CDHW` volume. The upstream caller supplies uint8/float32 data, a valid matrix, size, interpolation, border mode, and fill; masks use the same dtype rule, so AlbumentationsX rejects int64 before this call. It does not repeat CPU, layout, or autograd validation and never moves or detaches Tensor data. All seven routers preserve container and their documented supported dtypes.
- **3D dense resampling:** `remap3d` applies a caller-provided float32 `(D_out, H_out, W_out, 3)` normalized `align_corners=False` pull grid in `(x, y, z)` order to one prevalidated NumPy `DHWC` volume or CPU Torch `CDHW` tensor. The grid can be NumPy or Torch, its spatial shape defines the output, and coordinates outside `[-1, 1]` use constant or replicate borders. The router preserves the input container, dtype, channel count, and public layout. Tensor calls use the measured zero-copy Tensor-to-NumPy-to-Tensor bridge; the direct CPU sampler remains a rejected benchmark candidate, not a heuristic route. `remap3d` does not scan the dense grid for identity or convert a voxel-coordinate map.
- **Utils:** see `utils.__all__` in [`albucore/utils.py`](../albucore/utils.py)
- **Types / constants:** `ImageType`, `ImageUInt8`, `ImageFloat32`, `SupportedDType`, `NormalizationType`, `ValueType`, `MAX_OPENCV_WORKING_CHANNELS`, `get_opencv_max_channels`, etc.
- **Metadata:** `__version__`, `__author__`, `__maintainer__`

## Shims (submodule-only, not star-exported)

Import explicitly from `albucore.functions` for tests and golden references:

- Backend-specific: `*_numpy`, `*_opencv`, `*_lut`, `*_cv2`, `hflip_numpy`, `vflip_numpy`; this includes `exp_numpy`/`exp_opencv`, `log_numpy`/`log_opencv`, and `sqrt_numpy`/`sqrt_opencv`
- NumKong helpers: `add_weighted_numkong`, `add_array_numkong`, `multiply_by_constant_numkong` (`albucore.functions`); `add_constant_numkong` lives on [`albucore.weighted`](../albucore/weighted.py) only
- LUT plumbing: `create_lut_array`, `apply_lut`, `prepare_value_opencv`, `apply_numpy`

## Internal

Names prefixed with `_` or used only inside albucore (not stable API).

## Migration (package `__all__`)

- `from albucore import *` no longer exposes backend-specific helpers (`add_opencv`, `normalize_per_image_numpy`, …). Import them explicitly: `from albucore.functions import add_opencv`.
- Deprecated **SimSimd** aliases are removed; use `add_weighted_numkong` (and other `*_numkong` helpers) from `albucore.functions` if needed.
- New stats entrypoints: `from albucore.stats import mean, std, mean_std, reduce_sum` (also re-exported on `albucore` for star-import users).
- New float32 elementwise entrypoints: `from albucore import exp, log, sqrt`.
