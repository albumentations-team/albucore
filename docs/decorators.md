# Decorators

Albucore provides several decorators to handle common image processing patterns and edge cases.

## Core Decorators

### `@preserve_channel_dim`

Ensures single-channel images maintain their shape `(H, W, 1)` when OpenCV operations might drop the channel dimension.

**Problem**: OpenCV functions like `cv2.flip()`, `cv2.LUT()` may drop the channel dimension for single-channel images, converting `(H, W, 1)` to `(H, W)`.

**Solution**: This decorator automatically restores the channel dimension if it was dropped.

```python
from albucore.decorators import preserve_channel_dim

@preserve_channel_dim
def my_function(img: ImageType) -> ImageType:
    # OpenCV operation that might drop channel dimension
    return cv2.flip(img, 1)
```

### `@clipped`

Clips the result to the valid range for the input dtype.

**Use case**: After arithmetic operations that might produce out-of-range values.

```python
from albucore.utils import clipped

@clipped
def multiply_by_constant(img: ImageType, value: float, inplace: bool) -> ImageType:
    # Multiplication might produce values outside valid range
    if img.dtype == np.uint8:
        return multiply_lut(img, value, inplace)
    return multiply_opencv(img, value)
```

## Type Conversion Decorators

### `@float32_io`

Converts input to float32, processes it, and converts back to the original dtype.

**Use case**: When your function works best with float32 but needs to support other dtypes.

```python
from albucore.functions import float32_io

@float32_io
def some_image_function(img: np.ndarray) -> np.ndarray:
    # Function implementation assuming float32
    return img * 1.5 + 10
```

### `@uint8_io`

Converts input to uint8, processes it, and converts back to the original dtype.

**Use case**: When your function requires uint8 input (e.g., LUT operations) but needs to support other dtypes.

```python
from albucore.functions import uint8_io

@uint8_io
def some_image_function(img: np.ndarray) -> np.ndarray:
    # Function implementation assuming uint8
    return uint8_only_backend(img)
```

## Usage Patterns

### Single-Channel Image Handling

```python
@preserve_channel_dim
def normalize_lut(img: ImageUInt8, mean: float, std: float) -> ImageFloat32:
    lut = ((np.arange(0, 256, dtype=np.float32) - mean) / std).astype(np.float32)
    # cv2.LUT might drop the channel dimension for single-channel images
    return cv2.LUT(img, lut)
```

### Type-Safe Operations

```python
@clipped
@preserve_channel_dim
def add_opencv(img: ImageType, value: np.ndarray | float) -> ImageType:
    prepared_value = prepare_value_opencv(img, value, "add")
    return cv2.add(img, prepared_value)
```

### Views

```python
def hflip_numpy(img: ImageType) -> ImageType:
    return img[:, ::-1, ...]
```

NumPy slicing returns a view with its native strides. A later backend that requires contiguous input performs that repair at its own boundary.

## Common Gotchas

1. **Don't over-use decorators**: Only apply decorators when they're actually needed for your specific operation.

2. **`@preserve_channel_dim` is not needed for NumPy operations**: NumPy maintains dimensions correctly; this is primarily for OpenCV.

3. **Type conversion decorators**: `@float32_io` and `@uint8_io` involve conversions that have cost. Use them judiciously.
