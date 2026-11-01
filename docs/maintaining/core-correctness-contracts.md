# Core Correctness Contracts

Albucore public routers are low-level image kernels. Their contracts are intentionally small and
strict.

## Caller-validated preconditions

The caller validates container type, rank, layout, explicit channel dimension, dtype, device,
contiguity, autograd state, and operation-specific control data before entering Albucore. The
router keeps only backend dispatch and kernel work; it does not repeat those checks. Direct calls
with invalid metadata are outside the low-level contract and are not part of the correctness matrix.

## Image Layout

Image-like inputs always have an explicit channel dimension:

- NumPy images/batches: HWC / NHWC
- NumPy volumes/batches: DHWC / NDHWC
- CPU Tensor images/batches: CHW / NCHW
- CPU Tensor volumes/batches: CDHW / NCDHW

The channel dimension is explicit even for grayscale (`(H, W, 1)` for NumPy); channel-first Tensor operations that
need channel semantics receive an explicit `layout=`. Never infer `NCHW` versus `CDHW` from rank or axis sizes.

For image arrays, dimension indexing is always:

- `shape[-1]`: channels
- `shape[-2]`: width
- `shape[-3]`: height

For NumPy volumes, `shape[-4]` is depth; the last three spatial/channel axes retain the same order.

Tests and benchmarks must include non-square H/W cases so height-width swaps fail visibly.

## Dtype

Public image operations support:

- `uint8`
- `float16`
- `float32`

Other dtypes are invalid for the validated public image contract unless a function explicitly documents
a different array contract. Full image outputs must not silently widen to `float64`.

Stats and reduction helpers document their accumulator dtypes separately.

## Shape Preservation

Unless a router explicitly changes spatial size:

- HWC remains HWC
- NHWC remains NHWC
- DHWC remains DHWC
- NDHWC remains NDHWC
- CPU Tensor layout is preserved
- `(H, W, 1)` remains `(H, W, 1)`

Public routers must hide OpenCV channel-dropping behavior.

## Clipping And Bounds

- `uint8` arithmetic output stays in `[0, 255]`.
- Operations whose contract clips floating output constrain float16/float32 values to `[0.0, 1.0]`.
- Stats functions do not clip.

## Values

Scalar, per-channel vector, and image-shaped array inputs must follow documented conversion rules.
Short-vector behavior follows `convert_value` and must be covered by router contracts where relevant.

## Aliasing

- Without `out`, a router does not write input storage; returning an identity or view is allowed.
- `out=` explicitly requests a write and the supplied destination is returned by identity.
- Exact aliases and other overlaps are supported only where the operation contract names them; the caller validates them.
- Intentional no-op alias behavior must be recorded in `tests/router_contracts.py`.

## Contiguity

Routers that need a contiguous input normalize it at their own backend boundary.

## Backend Routing

Acceptance tests run on Python 3.11 with the newest stable dependencies available for Python 3.11 and each target
platform. See the [canonical performance guide](../performance-optimization.md) for the current version profile and
evidence requirements.

Routing is implementation detail, but routing changes require benchmark evidence. Public outputs
must match reference semantics regardless of whether the selected backend is NumPy, OpenCV,
StringZilla, or NumKong.
