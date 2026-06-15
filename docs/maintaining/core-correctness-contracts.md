# Core Correctness Contracts

Albucore public routers are low-level image kernels. Their contracts are intentionally small and
strict.

## Image Layout

Image-like inputs always have an explicit channel dimension:

- HWC: `(H, W, C)`
- XHWC: `(X, H, W, C)`
- NDHWC: `(N, D, H, W, C)`

Grayscale is `(H, W, 1)` or `(..., H, W, 1)`. Implicit-channel grayscale arrays such as `(H, W)` or
`(D, H, W)` are invalid image inputs.

Dimension indexing is always:

- `shape[-1]`: channels
- `shape[-2]`: width
- `shape[-3]`: height

Tests and benchmarks must include non-square H/W cases so height-width swaps fail visibly.

## Dtype

Public image operations support:

- `uint8`
- `float32`

Other dtypes are invalid for public image kernels unless a function explicitly documents a different
array contract. Full image outputs must not silently widen to `float64`.

Stats and reduction helpers document their accumulator dtypes separately.

## Shape Preservation

Unless a router explicitly changes spatial size:

- HWC remains HWC
- XHWC remains XHWC
- NDHWC remains NDHWC
- `(H, W, 1)` remains `(H, W, 1)`

Public routers must hide OpenCV channel-dropping behavior.

## Clipping And Bounds

- `uint8` arithmetic output stays in `[0, 255]`.
- Clipped `float32` output stays in `[0.0, 1.0]`.
- Stats functions do not clip.

## Values

Scalar, per-channel vector, and image-shaped array inputs must follow documented conversion rules.
Short-vector behavior follows `convert_value` and must be covered by router contracts where relevant.

## Aliasing

- `inplace=True` mutates only where documented.
- Functions without an `inplace` parameter must not mutate inputs.
- Intentional no-op alias behavior must be recorded in `tests/router_contracts.py`.

## Contiguity

Routers should document whether non-contiguous input is accepted, copied, or rejected. Decorators
that enforce contiguity need direct tests.

## Backend Routing

Routing is implementation detail, but routing changes require benchmark evidence. Public outputs
must match reference semantics regardless of whether the selected backend is NumPy, OpenCV,
StringZilla, or NumKong.
