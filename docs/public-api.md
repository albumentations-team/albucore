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
use wide accumulators and may return float64. `apply_uint8_lut` remains uint8-only. A backend may compute
in float32 and restore float16 when needed; include those conversions in the full-path benchmark.

## Explicit output buffers

`clip(image)` saturates to the input dtype range without writing to it. `clip(image, out=destination)` uses the
destination dtype and returns that exact object; `out=image` requests mutation. Pass `dtype` explicitly when choosing
a conversion independently of optional output storage. An explicit dtype must match the supplied destination.
Clipping does not scale intensities: a uint8 destination clips to [0,255] and truncates fractions; floating destinations
clip to [0,1]. The caller validates output metadata and overlap. Dtype changes may require working storage.

`exp`, `log`, and `sqrt` use the same explicit-output contract. Their output dtype is float16 for float16 input
and float32 for uint8/float32 input. Writable views may be supplied as output; AX owns protective copies required
when that storage must remain unchanged. No dtype-specific ownership heuristic silently refuses an explicit output.

`normalize(image, mean, denominator, out=destination)` uses fixed caller statistics. Its destination is float16
for float16 input and float32 for uint8/float32 input. Floating output may alias the complete input; uint8 input
requires a separate float32 result. Tensor calls require layout. Without `out`, input is unchanged and the existing
approximate identity policy may return it directly. Output strides do not change the selected arithmetic kernel;
an unsupported destination uses a temporary followed by a copy. Adaptive `normalize_per_image` has separate statistics
and clipping policies.

`normalize_per_image(image, mode, out=destination)` estimates its statistics from the input. Its result dtype and
output alias rules match fixed `normalize`. Statistics are computed before writing an alias. Mean/std modes add
`1e-4` to the standard deviation and clip to [-20,20]. Min/max policy is:

| Mode | Epsilon added to the range |
| --- | --- |
| `min_max` | None |
| `min_max_per_channel`, uint8 | `1e-4`, including C=1 |
| `min_max_per_channel`, floating C=1 | None |
| `min_max_per_channel`, floating C>1 | `1e-4` |

NumPy global modes reduce the whole array; channel modes reduce every axis except the last. Tensor NCHW/NCDHW
normalize each batch item independently; CHW/CDHW span one image or volume. Tensor layout is required. Empty input
returns an empty result without estimating statistics. Strided outputs can require a temporary.

`add` and `multiply` preserve the input dtype and accept keyword-only `out` with that dtype and container.
The destination has the image shape and layout. Exact image or full-field operand aliases are supported;
other overlaps, including a broadcast control sharing output storage, are outside the caller-validated contract.
Without `out`, neither input is written. Tensor calls require an explicit `layout=`.

`multiply_add` also uses keyword-only, dtype/container-matching `out` and requires a Tensor layout.
Uint8 controls are scalars or channel vectors; floating inputs also accept broadcastable fields.
Exact image and full-field control aliases are supported, including an output that is also the bias field.
The two-step half routes retain product rounding before addition; selected NumKong scalar routes retain fused rounding.
Supplying output may still require an intermediate to preserve that numerical policy and read-before-write order.

`apply_uint8_lut` maps uint8 images through a shared `(256,)` or per-channel `(256, 1, C)` table.
It accepts writable strided `out` buffers and `out=image`, returning that exact object. Other overlapping
views and overlap with the table are unsupported. Shared Tensor tables do not require layout; per-channel
Tensor calls declare `layout=`. Strided paths may use a temporary. Values match integer table lookup exactly.

## Static return types

Overloads preserve the relationship between input and output container and dtype. Dtype-preserving NumPy calls retain
that array type; Tensor calls return Tensor. Normalization and conversion follow their explicit output-dtype contracts.
The implementation may still accept a union of inputs without exposing that entire union as every caller's result.

Concrete NumPy dtypes use `NDArray[np.uint8]`, `NDArray[np.float16]`, and `NDArray[np.float32]` directly.
These annotations constrain dtype, not rank; the same types describe images, videos and volumes.
Shape and layout follow the operation's documented contract. Shared unions and TypeVars express supported
input sets and input/output relationships rather than abbreviating a single dtype.

The checked generic `np.dtype[T]` path in `clip` and `from_float` uses `T` bounded by
`np.uint8 | np.float16 | np.float32` and retains `NDArray[T]` without a destination. A dynamically selected
`SupportedDType` returns the union of the three concrete array types. Inferred `clip` output follows the supplied
buffer; an explicit target dtype must match that buffer. Generic overloads must not hide a conflicting destination.

`ImageInputT` describes a helper that preserves both container and image dtype. Its bound includes the three supported
NumPy image dtypes and Tensor; Tensor dtype and CPU placement remain runtime preconditions. Operations that change
dtype or accept different operand types by container use their concrete overloads.

`float32_io` and `uint8_io` preserve NumPy input types and required callback arguments, including `img=` calls.
Bind a Tensor layout before calling a decorated function: `kernel.with_layout("CHW")(tensor, amount=1.5)`.
The bound callable accepts NumPy/Tensor unions and preserves the input container and dtype. Bind once and reuse the
callable for repeated calls. Both forms check callback argument names and types; omitting a required argument is a
static error. Binding requires a non-None `CHW`, `NCHW`, `CDHW`, or `NCDHW` layout.

Stacking `clipped` inside or outside either I/O decorator preserves those signatures and the clipping behavior of the
chosen order. A plain `clipped` callback that intentionally returns uint8 keeps that output contract.

For these I/O decorators, Tensor calls require `.with_layout()`; their direct call signature accepts NumPy arrays.
Checked consumer examples live in
[`tests/typing/router_calls.py`](../tests/typing/router_calls.py); the mypy hook and runtime-stub CI check compile them.

## Routers (package `__all__`)

User-facing entry points with benchmark-driven backends inside:

`add` and `multiply` accept scalar, channel-vector, and full-field operands. A valid channel vector retains its
dtype and vector semantics for C=1. It is not silently converted to a Python scalar. Full-size floating Tensor
operands use the native CPU operation when that preserves the declared result dtype.

- **Arithmetic / weighted:** `add`, `add_weighted`, `multiply`, `multiply_add`, `power`, `normalize`, `normalize_per_image`
  `power` accepts keyword-only `out` and returns it by identity. Without `out`, input storage is unchanged.
  Writable strided outputs and exact input aliasing are supported; other overlaps are outside the contract.
  Uint8 accepts scalar/channel exponents and saturates/truncates a float32 table. Floating inputs also accept
  broadcastable exponent fields. Tensor calls require an explicit layout. Scalar half powers compute in float32
  and round once; half channel/field routes match the wide result within one half ULP.
- **Elementwise math:** `exp`, `log`, `sqrt` (float16/float32 inputs, uint8 with float32 output, NumPy-compatible special values, exact shape preservation, explicit writable `out` buffers)
- **Uint8 LUT (`albucore.lut`):** `apply_uint8_lut` (shared or per-channel tables, with explicit `out`)
- **I/O:** `to_float`, `from_float`
  Both accept keyword-only `out` with the result dtype and return that destination by identity. Without `out`,
  input storage is unchanged; same-dtype floating conversions return by identity. Writable strided destinations
  are supported; exact input aliasing is allowed for same-dtype casts. Other overlapping views are outside the
  contract. `to_float` divides uint8 by `max_value` (default 255), while floating inputs ignore that divisor.
  Floating `from_float` targets cast without intensity scaling. Uint8 targets scale, round ties to even, saturate,
  and map NaN to zero. NumPy half input computes in float32; contiguous half Tensors may compute in half for exactly
  representable scales, within one uint8 unit of the wide result. Tensor inputs follow the prevalidated CPU,
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
- **Types / constants:** `ImageInput`, `ImageInputT`, `ImageType`, `SupportedDType`, `TensorLayout`, `NormalizationType`, `ValueType`, and shared routing constants.
- **Metadata:** `__version__`, `__author__`, `__maintainer__`

`add_weighted(img1, weight1, img2, weight2, *, out=...)` preserves the matching input dtype and container.
The second image has the same shape, dtype and container as the first. Tensor calls require a layout. Without
`out`, both inputs remain unchanged; a writable output is returned by identity and may alias either complete input.
Strided outputs may require a temporary. The existing NumKong/OpenCV region keeps its rounding and non-finite
policy; zero weights do not add a new identity shortcut. Caller prevalidation owns metadata and overlap checks.

## Backend-specific imports (submodule-only)

Public calls use `from albucore import <operation>`. Package `__all__` is explicit; the implementation is imported
directly from its owning module. Backend functions are implementation details for internal tests and benchmarks:

- Backend-specific: `*_numpy`, `*_opencv`, `*_lut`, `*_cv2`, `hflip_numpy`, `vflip_numpy`
- Weighted sums and private blend kernels live in [`albucore.weighted`](../albucore/weighted.py). Scalar NumKong scale and arithmetic kernels live in
  [`albucore.arithmetic`](../albucore/arithmetic.py); conversion kernels in
  [`albucore.convert`](../albucore/convert.py); fixed-statistics and per-image normalization in
  [`albucore.normalize`](../albucore/normalize.py); elementwise kernels in
  [`albucore.elementwise`](../albucore/elementwise.py); flips in
  [`albucore.ops_misc`](../albucore/ops_misc.py).
- Arithmetic LUT construction, operand preparation, and NumPy/OpenCV kernels are private implementation details.

## Internal

Names prefixed with `_` or used only inside albucore (not stable API).
