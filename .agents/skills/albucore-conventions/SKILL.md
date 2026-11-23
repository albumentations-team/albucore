---
name: albucore-conventions
description: Albucore shape, layout, dtype, routing, and test conventions for NumPy and Torch CPU image code. Use when writing or changing code or tests under albucore/ or tests/, or answering questions about image shapes, channel layouts, supported dtypes, or OpenCV channel limits ("какой layout", "почему float16", "HWC или CHW").
---

# Albucore Conventions

`AGENTS.md` already defines explicit channel dimensions, supported dtypes, the caller-validated router boundary,
output ownership, and lockfile rules. This skill adds the details needed while writing code. For runtime
implementation or review, also load the `performance-optimization` skill.

## Shapes and layouts

- NumPy images are channel-last: `HWC`, `NHWC`, `DHWC`, `NDHWC`. Grayscale is `(H, W, 1)`, never `(H, W)`; a
  grayscale volume is `(D, H, W, 1)`, never `(D, H, W)`.
- A Tensor API declares its layout (`CHW`, `NCHW`, `CDHW`, `NCDHW`) through `layout=`. Never infer a Tensor layout
  from rank or axis sizes.
- `resize3d` declares NumPy `DHWC` and Torch `CDHW`. `warp_affine3d` and `remap3d` also accept `NDHWC` / `NCDHW`
  batches with one matrix or grid shared by every item. Other 3D routers accept one volume.
- AlbumentationsX checks CPU device, layout, strides, `requires_grad=False`, and control data before the call.
  Routers do not silently detach or move data, and do not re-check these preconditions. Document the boundary in
  public-router docstrings.

## Dtypes

- Image routers support `uint8`, `float16`, `float32`. Statistics may return wide `float64` or integer results.
  `apply_uint8_lut` is `uint8`-only.
- Choose input, working, accumulator, and output dtypes per operation. Float16 output preserves float16 input where
  documented; a float32 working buffer is allowed for precision or backend support. Do not widen full arrays by
  default when a measured kernel writes the final half output directly.
- `cv2.LUT`: pass a `uint8` `(H, W, C)` source; build the table in the output dtype. Cast float64 statistics to
  float32 when the output is float32 so the whole result stays float32.

## Backend routing

- Route only from benchmark evidence, measured per dtype on Linux x86-64 and macOS ARM64 independently. Do not
  assume LUT is fastest for `uint8`.
- Current routes, thresholds, and the measurements behind them live in `docs/numkong-performance.md` (NumKong,
  OpenCV, NumPy) and `docs/torch-performance-optimization.md` (Torch). Read the entry for the operation you change
  and keep its measured region; do not copy route measurements into skills.
- OpenCV supports at most `MAX_OPENCV_WORKING_CHANNELS = 4` for many ops; above that, fall back to NumPy or chunk.
- StringZilla backs shared `uint8 -> uint8` lookup in `apply_uint8_lut`. For distinct per-channel tables, compare
  interleaved OpenCV with StringZilla on planes, including split, merge, and copy costs.
- Runtime minimums: `numkong>=7.8.7`, `stringzilla>=5.3.2`, `torch>=2.14.1`.

## Torch CPU

- Torch is a required runtime dependency. Import it directly; no `sys.modules` checks, class-name heuristics, or
  lazy imports, and do not optimize routing around deferred import cost.
- Benchmark NumPy-to-Torch routes end to end: wrapper creation, permutations, dtype casts, kernel, and the returned
  NumPy layout all belong in the timed region. Load the `torch-performance-optimization` skill for Torch paths.

## Operation contracts

- Types have one owner, `albucore._types`; consumers use the types exported by `albucore`. Utilities such as
  `get_num_channels` and `convert_value` live in `albucore.utils`.
- Dtype conversion and clipping belong in the owning operation. C=1 OpenCV operations keep `(H, W, 1)` through an
  explicit singleton-channel destination when the backend supports it. Do not add wrappers.
- Per-operation `out=`, alias, and dtype contracts (`clip`, `add`, `multiply`, `multiply_add`, `add_weighted`,
  `normalize`, `normalize_per_image`, `power`, `convert_dtype`) are in `docs/public-api.md`; load the
  `albucore-public-api` skill when changing them.

## Tests

- Test `uint8`, `float16`, and `float32` for every supported image router; keep `uint8` LUT APIs `uint8`-only.
- Test NumPy channel-last and declared Tensor layouts, plus image batches and volumes where supported.
- Cover 1-channel, >4-channel, and extreme-value cases. Tests and benchmarks use valid, caller-prevalidated inputs.
