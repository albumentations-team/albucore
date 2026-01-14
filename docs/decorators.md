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

### `@contiguous`

Ensures arrays are C-contiguous for optimal performance with certain operations (e.g., stringzilla).

**Why**: Some operations require C-contiguous memory layout. Fortran-contiguous arrays will be converted, which may involve copying data.

```python
from albucore.decorators import contiguous

@contiguous
def sz_lut(img: ImageUInt8, lut: ImageUInt8, inplace: bool = True) -> ImageUInt8:
    """Apply lookup table using stringzilla."""
    if not inplace:
        img = img.copy()
    sz.translate(memoryview(img), memoryview(lut), inplace=True)
    return img
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
    return apply_lut(img, some_lut, inplace=False)
```

## Batch Processing Decorator

### `@batch_transform`

Handles batch transformations by reshaping data appropriately for different transform types.

```python
from albucore.decorators import batch_transform, BatchTransformType

@batch_transform(transform_type="spatial")
def apply_spatial_transform(self, img: np.ndarray, ...) -> np.ndarray:
    # Transform is applied to reshaped data
    # Decorator handles (N,H,W,C) -> (H,W,N*C) conversion
    return transformed_img
```

**Transform types**:
- `"spatial"`: For transforms that modify spatial dimensions (H, W)
- `"channel"`: For transforms that modify channel dimension
- `"full"`: No reshaping, process the array as-is

**Additional parameter**:
- `keep_depth_dim=True`: For 3D volumes `(N,D,H,W,C)`, preserves depth dimension during processing

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

### Memory Layout Enforcement

```python
@contiguous
def hflip_numpy(img: ImageType) -> ImageType:
    # Slicing might produce non-contiguous arrays
    return img[:, ::-1, ...]
```

## Decorator Ordering

When using multiple decorators, apply them in this order (from innermost to outermost):

1. `@contiguous` - Ensure proper memory layout first
2. `@preserve_channel_dim` - Handle OpenCV quirks
3. `@clipped` - Clip final results

```python
@clipped
@preserve_channel_dim
@contiguous
def my_function(img: ImageType) -> ImageType:
    # Implementation
    return result
```

## Common Gotchas

1. **Don't over-use decorators**: Only apply decorators when they're actually needed for your specific operation.

2. **`@preserve_channel_dim` is not needed for NumPy operations**: NumPy maintains dimensions correctly; this is primarily for OpenCV.

3. **`@contiguous` adds overhead**: Only use it when required by the underlying library (e.g., stringzilla).

4. **Type conversion decorators**: `@float32_io` and `@uint8_io` involve conversions that have cost. Use them judiciously.
