# Albucore public API classification

`from albucore import *` exposes only **routers** and shared helpers listed in [`albucore/__init__.py`](../albucore/__init__.py) `__all__`.

## Routers (package `__all__`)

User-facing entry points with benchmark-driven backends inside:

- **Arithmetic / weighted:** `add`, `add_constant`, `add_array`, `add_vector`, `add_weighted`, `multiply`, `multiply_by_constant`, `multiply_by_vector`, `multiply_by_array`, `multiply_add`, `power`, `normalize`, `normalize_per_image`, `sz_lut` (uint8 LUT via stringzilla; used by Albumentations)
- **I/O:** `to_float`, `from_float`
- **Geometry / misc:** `hflip`, `vflip`, `median_blur`, `matmul`, `pairwise_distances_squared`
- **Stats:** `mean`, `std`, `mean_std` (from `albucore.stats`)
- **Decorators:** see `decorators.__all__` in [`albucore/decorators.py`](../albucore/decorators.py)
- **Geometric:** `copy_make_border`, `remap`, `resize`, `warp_affine`, `warp_perspective`
- **Utils:** see `utils.__all__` in [`albucore/utils.py`](../albucore/utils.py)
- **Types / constants:** `ImageType`, `ImageUInt8`, `ImageFloat32`, `SupportedDType`, `NormalizationType`, `ValueType`, `MAX_OPENCV_WORKING_CHANNELS`, etc.
- **Metadata:** `__version__`, `__author__`, `__maintainer__`

## Shims (submodule-only, not star-exported)

Import explicitly from `albucore.functions` for tests and golden references:

- Backend-specific: `*_numpy`, `*_opencv`, `*_lut`, `*_cv2`, `hflip_numpy`, `vflip_numpy`
- NumKong helpers: `add_weighted_numkong`, `add_array_numkong`, `multiply_by_constant_numkong` (`albucore.functions`); `add_constant_numkong` lives on [`albucore.weighted`](../albucore/weighted.py) only
- LUT plumbing: `create_lut_array`, `apply_lut`, `prepare_value_opencv`, `apply_numpy`

## Internal

Names prefixed with `_` or used only inside albucore (not stable API).

## Migration (package `__all__`)

- `from albucore import *` no longer exposes backend-specific helpers (`add_opencv`, `normalize_per_image_numpy`, …). Import them explicitly: `from albucore.functions import add_opencv`.
- Deprecated **SimSimd** aliases are removed; use `add_weighted_numkong` (and other `*_numkong` helpers) from `albucore.functions` if needed.
- New stats entrypoints: `from albucore.stats import mean, std, mean_std` (also re-exported on `albucore` for star-import users).
