# Codex AI Development Guidelines for Albucore

This document provides guidelines and conventions for AI assistants (particularly Codex) when working on the Albucore codebase.

## Project Overview

Albucore is a high-performance image processing library that provides optimized atomic functions for image manipulation. It serves as the foundation for AlbumentationsX and focuses on:

- Maximum performance through multiple backend implementations (NumPy, OpenCV, custom)
- **Supported image dtypes: uint8, float16, and float32** (no float64 image inputs)
- **Backend routing is based ONLY on performance** — benchmark before choosing. If LUT is slower, don't use it.
- Consistent API across different image types and shapes

## Core Documentation

Please familiarize yourself with these key documents:

1. **[Image Shape Conventions](docs/image-conventions.md)** - **CRITICAL** rules about image shapes
   - All images must have an explicit channel dimension
   - NumPy uses channel-last; Torch CPU uses an explicit channel-first layout
   - Grayscale images use `(H, W, 1)` or a declared layout with `C=1`

2. **[Performance Optimization](docs/performance-optimization.md)** - Performance best practices
   - LUT operations and float32 dtype management
   - Backend selection strategies
   - Memory layout considerations
   - Benchmarking guidelines

3. **[Public API](docs/public-api.md)** - Public operations, explicit root exports, and owning modules

4. **[PyTorch Optimization](docs/torch-performance-optimization.md)** - Eager CPU Torch contracts, full-path benchmarks,
   Tensor/NumPy bridge costs, profiling, allocation, layout, and threading decisions

5. **[NumKong Performance](docs/numkong-performance.md)** - NumKong vs OpenCV/NumPy/LUT benchmark tables and methodology

## Repo-Local Codex Skills

Repo-specific Codex skills live in `.codex/skills/`:

- `albucore-conventions` - Apply when implementing or reviewing image-processing code, tests, shape handling, dtype handling, and backend routing.
- `albucore-benchmarks` - Apply when adding benchmarks, comparing versions, or documenting benchmark workflow.
- `performance-optimization` - Apply for every runtime implementation, review, profile, or optimization. It loads
  `docs/performance-optimization.md` completely and runs the delete-first, vectorization, grouped-reduction, LUT, RNG,
  backend, extraction, and explicit-output/alias audit.
- `torch-performance-optimization` - Apply for every eager CPU-only Torch runtime implementation, review, profile,
  routing, layout, allocation, threading, or benchmark decision. It loads `docs/torch-performance-optimization.md` and adds
  Torch-specific contract, profiler, bridge, and full-path benchmark checks.
- `albucore-public-api` - Apply when changing exports, documenting API status, or deciding what belongs in package `__all__`.

Keep these skills aligned with `AGENTS.md` and the docs above when conventions change.

## Development Principles

### 0. Caller-validated router boundary

Albucore receives prevalidated inputs from AlbumentationsX or another upstream caller. Container type, rank, layout,
explicit channel dimension, dtype, device, contiguity, autograd state, and operation-specific control data are checked
before dispatch. Do not duplicate those checks in Albucore routers; keep only backend dispatch and kernel-required
normalization. Invalid direct calls are outside the low-level contract.

### Output ownership

Operations with `out` write input storage only when the caller explicitly supplies it as output. Without `out`,
do not write input data; returning a view or the input object is allowed. Return the supplied destination object
when `out` is present. The caller validates its shape, dtype, device, strides, writeability, and supported overlap.

AX owns reusable buffers and any protective copies needed before writing shared storage. Include those copies in
AX timing and memory only when the user explicitly requests AX performance validation. Albucore performance checks
cover Core public operations only. Do not add hidden pools or caches of large arrays. `clip` follows this
contract through keyword-only `out`.
`multiply_add` follows the same explicit output contract; Tensor calls require layout and compound control aliases must preserve read-before-write order.
`normalize` writes result-dtype `out`: float16 for float16 input, float32 for uint8/float32. Exact floating source
aliases are supported. Preserve its backend arithmetic order and approximate identity policy; destination strides
must not select a different numerical kernel. Fixed and adaptive normalization share the normalize module.
`normalize_per_image` also uses result-dtype `out` and required Tensor layout. Compute all statistics before an alias
write. Preserve mode-specific epsilon/clipping and NumPy whole-array versus Tensor per-item batch statistics.



### 1. Image Shape Convention

**This is the most important rule:**

```python
# ALWAYS use an explicit channel dimension
grayscale_image = np.array(..., shape=(H, W, 1))

# NEVER use an implicit channel dimension
grayscale_image = np.array(..., shape=(H, W))
```

For NumPy image arrays, you can always assume:
- `num_channels = image.shape[-1]`
- `width = image.shape[-2]`
- `height = image.shape[-3]`

Torch image calls use explicit `CHW`, `NCHW`, `CDHW`, or `NCDHW` layout where channel-axis semantics are needed.
Never infer `NCHW` versus `CDHW` from rank or axis sizes.

### 2. Performance First — Routing by Benchmark Only

- **Choose implementations ONLY based on benchmark results.** No conventions. If LUT is slower for a given case, use something else.
- Always benchmark when adding new implementations or changing routing
- For local macOS ARM64 benchmarks that depend on NumKong, run the process outside the Codex sandbox with
  `sandbox_permissions="require_escalated"`; verify the NumKong `neon` capability before accepting results. Never force
  backend flags to conceal a failed hardware probe.
- OpenCV, NumPy, LUT — pick whichever is fastest for the target dtype/shape

### 3. Type Safety

- Use type hints consistently
- Use `NDArray[np.uint8]`, `NDArray[np.float16]`, and `NDArray[np.float32]` for concrete NumPy dtypes.
- Keep shared unions and TypeVars where they define supported inputs or preserve input/output type relationships.
- Ensure dtype preservation or explicit conversion

### 4. Multiple Implementations Pattern

Route based on **benchmarked performance**, not convention:

```python
def operation(
    img: ImageInput,
    value: ValueType,
    *,
    out: ImageInput | None = None,
    layout: TensorLayout | None = None,
) -> ImageInput:
    if img.dtype == np.uint8:
        return operation_uint8(img, value, out=out)
    if img.dtype == np.float16:
        return operation_float16(img, value, out=out)
    return operation_float32(img, value, out=out)
```

Public overloads must still preserve concrete dtype/container relationships and require `layout` for channel-dependent
Tensor semantics. This sketch shows only the measured dispatch boundary; the caller validates inputs and controls.

### 5. Testing

- Test uint8, float16, and float32; LUT-specialized APIs remain uint8-only
- Test NumPy channel-last and declared Torch CPU layouts where supported
- Test single images, image batches, and single volumes; test volume batches for `warp_affine3d` and `remap3d`
- Test edge cases: single-channel, many channels (>4), extreme values
- Include performance benchmarks when relevant

Final correctness, performance, and memory acceptance for Albucore and AX runs on Python 3.11 with the newest stable
third-party dependencies that support Python 3.11 and the target platform. The profile recorded on 2026-10-31 used
PyTorch 2.15.0 (`2.15.0+cpu` on Linux), NumPy 2.4.6, OpenCV 5.0.0.93, NumKong 7.8.6, and StringZilla 5.3.2. Re-resolve
before future acceptance runs and record the complete package inventory and imported source paths. A newer Python
profile or stale lockfile does not replace this acceptance profile. Keep runtime minimums separate from test versions.

### 6. Dependency Lock Consistency

- When changing dependencies in `pyproject.toml`, update `uv.lock` in the same PR.
- Validate lock consistency with `uv lock --check`.
- Release flow uses `uv export --frozen`; stale `uv.lock` can break release artifact generation.

## Code Style

### Imports

```python
from collections.abc import Callable
from typing import Any, Literal

import cv2
import numpy as np
import numkong as nk
import stringzilla as sz

from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    get_num_channels,
    # ... other imports
)
```

### Function Documentation

Use clear docstrings with examples:

```python
def function_name(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    param: ValueType,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    """Brief description.

    Detailed explanation of what the function does and when to use it.

    Args:
        img: Input image as numpy array with shape (H, W, C).
        param: Description of parameter.

    Returns:
        Processed image with same shape as input.

    Notes:
        - Any important performance considerations
        - Supported dtypes
        - Special behaviors
    """
```

## Common Patterns

### 1. Value Conversion

```python
num_channels = get_num_channels(img)
value = convert_value(value, num_channels)

if isinstance(value, (float, int)):
    # Scalar operation
elif isinstance(value, np.ndarray) and value.ndim == 1:
    # Per-channel operation
else:
    # Array operation
```

### 2. Dtype Handling

```python
original_dtype = img.dtype

# Process
result = process_image(img)

# Ensure output dtype matches input
return result.astype(original_dtype, copy=False)
```

### 3. Channel Limit Handling

OpenCV has a 4-channel limit for many operations:

```python
num_channels = get_num_channels(img)

if num_channels > MAX_OPENCV_WORKING_CHANNELS:
    # Use NumPy fallback
    return operation_numpy(img, value)

# Use OpenCV
return operation_opencv(img, value)
```

## Performance Guidelines

Read `docs/performance-optimization.md` and use the `performance-optimization` skill for the complete required pass.

1. Delete redundant work and full-array passes before selecting a backend.
2. Vectorize loops when the benchmark supports the resulting memory layout.
3. Consider `np.bincount` for grouped reductions over dense non-negative integer labels.
4. Compare applicable NumPy, OpenCV, NumKong, StringZilla, LUT, and small Python implementations end to end.
5. Compare Python, NumPy, and OpenCV random generation without breaking seeded isolation or replay.
6. Route by benchmarked dtype, shape, channel, size, contiguity, and allocation regions.
7. Cast float LUTs to float32 and keep full working arrays from widening to float64.
8. Use explicit `out=` when ownership, overlap rules, and benchmarks show the destination removes work.

## Questions to Ask

When implementing a new function, consider:

1. **Supported image dtypes: uint8, float16, and float32; LUT-specialized APIs are uint8-only**
2. Can any work, full-array pass, conversion, or allocation be deleted?
3. Can loops be vectorized or dense labels be reduced with `np.bincount`?
4. Is the operation a LUT, reduction, fused arithmetic kernel, or existing Albucore atom?
5. Which NumPy, OpenCV, NumKong, StringZilla, LUT, Python, or random-generation candidates apply?
6. Does OpenCV support the channel count, rank, layout, and aliasing mode?
7. Which documented array ranks, including single volumes and volume batches where supported, should this support?
8. What should the output dtype and mutation contract be?
9. Does `out=` safely reuse storage without adding temporaries or changing read-before-write order?

## Resources

- Operation owners: `albucore/arithmetic.py`, `convert.py`, `normalize.py`, `stats.py`, and geometric modules
- Public exports: `albucore/__init__.py`
- Utilities: `albucore/utils.py`; type definitions: `albucore/_types.py`
- Tests: `tests/`
- Benchmarks: `./benchmark.sh` (dataset I/O); Python micro-benchmarks: [`benchmarks/README.md`](benchmarks/README.md)
- Public API: `docs/public-api.md`
- Repo-local skills: `.codex/skills/`

## Getting Help

If you're unsure about:
- Performance characteristics: Check existing benchmarks or write new ones
- Expected behavior: Look at similar functions in the codebase
- Edge cases: Check the test suite for examples

## Summary

The key to working with Albucore is:
1. **Always use explicit channel dimensions** - `(H, W, C)` not `(H, W)`
2. **Performance matters** - choose the right backend
3. **Type safety** - handle dtypes carefully
4. **Test thoroughly** - many dtypes, shapes, and edge cases
5. **Follow patterns** - consistency makes the codebase maintainable

## Generated benchmark artifacts

Keep generated benchmark reports, CSV/JSON/GZ data, profiler output and run logs outside Git. Local outputs may
use ignored results directories. Never force-add them, including when publishing a PR or preserving evidence.
Keep reproducible benchmark code and concise current route decisions in the existing documentation. Preserve raw
measurements as external artifacts and link them from PR descriptions or the archive index in benchmarks/README.md.
