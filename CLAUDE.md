# Claude AI Development Guidelines for Albucore

This document provides guidelines and conventions for AI assistants (particularly Claude) when working on the Albucore codebase.

## Project Overview

Albucore is a high-performance image processing library that provides optimized atomic functions for image manipulation. It serves as the foundation for AlbumentationsX and focuses on:

- Maximum performance through multiple backend implementations (NumPy, OpenCV, custom)
- **Supported dtypes: uint8 and float32 only** (no float64)
- **Backend routing is based ONLY on performance** — benchmark before choosing. If LUT is slower, don't use it.
- Consistent API across different image types and shapes

## Core Documentation

Please familiarize yourself with these key documents:

1. **[Image Shape Conventions](docs/image-conventions.md)** - **CRITICAL** rules about image shapes
   - All images must have an explicit channel dimension
   - Grayscale images use shape `(H, W, 1)`, not `(H, W)`
   - Consistent dimension indexing: `shape[-1]` is channels, `shape[-2]` is width, `shape[-3]` is height

2. **[Decorators](docs/decorators.md)** - Decorator usage and patterns
   - `@preserve_channel_dim` - Maintains single-channel shape
   - `@contiguous` - Ensures C-contiguous memory layout
   - `@clipped` - Clips to valid dtype range
   - `@float32_io` / `@uint8_io` - Type conversion wrappers
   - `@batch_transform` - Batch processing patterns

3. **[Performance Optimization](docs/performance-optimization.md)** - Performance best practices
   - LUT operations and float32 dtype management
   - Backend selection strategies
   - Memory layout considerations
   - Benchmarking guidelines

## Development Principles

### 1. Image Shape Convention

**This is the most important rule:**

```python
# ✅ ALWAYS use explicit channel dimension
grayscale_image = np.array(..., shape=(H, W, 1))

# ❌ NEVER use implicit channel dimension
grayscale_image = np.array(..., shape=(H, W))  # WRONG!
```

Given any image, you can always assume:
- `num_channels = image.shape[-1]`
- `width = image.shape[-2]`
- `height = image.shape[-3]`

### 2. Performance First — Routing by Benchmark Only

- **Choose implementations ONLY based on benchmark results.** No conventions. If LUT is slower for a given case, use something else.
- Always benchmark when adding new implementations or changing routing
- OpenCV, NumPy, LUT — pick whichever is fastest for the target dtype/shape

### 3. Type Safety

- Use type hints consistently
- Leverage the provided type aliases: `ImageType`, `ImageUInt8`, `ImageFloat32`, `ValueType`
- Ensure dtype preservation or explicit conversion

### 4. Multiple Implementations Pattern

Route based on **benchmarked performance**, not convention:

```python
@clipped
def operation(img: ImageType, value: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    # Route based on what benchmarks show is fastest for this dtype/shape
    if img.dtype == np.uint8:
        return operation_lut(img, value, inplace)  # Only if LUT wins the benchmark

    if img.dtype == np.float32:
        return operation_numpy(img, value)  # Or opencv — whichever is faster

    raise ValueError(f"Unsupported dtype {img.dtype}. Albucore supports only uint8 and float32.")
```

### 5. Use Decorators

Albucore provides several useful decorators:

- `@preserve_channel_dim` - Maintains `(H, W, 1)` shape when OpenCV might drop it
- `@contiguous` - Ensures C-contiguous memory layout
- `@clipped` - Clips results to valid dtype range
- `@float32_io` / `@uint8_io` - Type conversion wrappers

### 6. Testing

- Write tests for uint8 and float32 only
- Test single images, batches, volumes, and batch of volumes
- Test edge cases: single-channel, many channels (>4), extreme values
- Include performance benchmarks when relevant

## Code Style

### Imports

```python
from collections.abc import Callable
from typing import Any, Literal

import cv2
import numpy as np
import simsimd as ss
import stringzilla as sz

from albucore.decorators import contiguous, preserve_channel_dim
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
def function_name(img: ImageType, param: ValueType) -> ImageType:
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

1. **Routing is benchmark-driven** — LUT, OpenCV, NumPy: use whichever is fastest. Don't assume LUT wins for uint8.
2. **NumPy** is best for >4 channels (OpenCV limit)
3. **Always cast LUTs to float32** to avoid dtype promotion issues
4. **Use in-place operations** when safe to reduce memory allocation

## Questions to Ask

When implementing a new function, consider:

1. **Supported dtypes: uint8 and float32 only**
2. Benchmark LUT vs OpenCV vs NumPy — use the fastest. Don't assume LUT wins.
3. Does OpenCV support this operation? What's the behavior for >4 channels?
5. Should this support batches/volumes?
6. Are there any edge cases with single-channel images?
7. What should the output dtype be?
8. Is an in-place option appropriate?

## Resources

- Main implementation: `albucore/functions.py`
- Utilities and types: `albucore/utils.py`
- Decorators: `albucore/decorators.py`
- Tests: `tests/`
- Benchmarks: Run `./benchmark.sh`

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
