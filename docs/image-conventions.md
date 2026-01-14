# Image Shape Conventions

## Overview

Albucore expects images to follow specific shape conventions where **the channel dimension is always present**, even for grayscale images.

## Shape Formats

### Single Images and Volumes

- **RGB/Multi-channel image**: `(H, W, C)` - Height, Width, Channels
- **Grayscale image**: `(H, W, 1)` - Height, Width, 1 channel
- **Volume**: `(D, H, W, C)` - Depth, Height, Width, Channels
- **Grayscale volume**: `(D, H, W, 1)` - Depth, Height, Width, 1 channel

### Batches

- **Batch of images**: `(N, H, W, C)` - Number of images, Height, Width, Channels
- **Batch of grayscale images**: `(N, H, W, 1)`
- **Batch of volumes**: `(N, D, H, W, C)` - Number of volumes, Depth, Height, Width, Channels
- **Batch of grayscale volumes**: `(N, D, H, W, 1)`

## Accessing Dimensions

Given any image array, you can always rely on:

```python
num_channels = image.shape[-1]
width = image.shape[-2]
height = image.shape[-3]
```

This consistency simplifies code and makes it more robust across different input types.

## Important Notes

### ✅ Correct

```python
import numpy as np

# Grayscale image with explicit channel dimension
gray_image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)

# RGB image
rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Multi-channel image (e.g., multispectral)
multi_channel = np.random.randint(0, 256, (100, 100, 10), dtype=np.uint8)
```

### ❌ Incorrect

```python
# DO NOT use 2D shape for grayscale images
gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Missing channel dimension!

# DO NOT use 3D shape for grayscale volumes
gray_volume = np.random.randint(0, 256, (20, 100, 100), dtype=np.uint8)  # Missing channel dimension!
```

## Examples

```python
import numpy as np
import albucore

# Single grayscale image
gray = np.random.randint(0, 256, (256, 256, 1), dtype=np.uint8)
result = albucore.multiply(gray, 1.5)

# Single RGB image
rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
result = albucore.multiply(rgb, 1.5)

# Batch of 10 RGB images
batch = np.random.randint(0, 256, (10, 256, 256, 3), dtype=np.uint8)
result = albucore.multiply(batch, 1.5)

# 3D volume with 20 slices (grayscale)
volume = np.random.randint(0, 256, (20, 256, 256, 1), dtype=np.uint8)
result = albucore.multiply(volume, 1.5)

# Batch of 5 RGB volumes, each with 20 slices
batch_volumes = np.random.randint(0, 256, (5, 20, 256, 256, 3), dtype=np.uint8)
result = albucore.multiply(batch_volumes, 1.5)
```

## Why This Convention?

1. **Consistency**: Uniform interface regardless of the number of channels
2. **Simplicity**: No special cases for grayscale vs. color images
3. **Compatibility**: Works seamlessly with libraries that expect channel dimensions
4. **Type Safety**: Shape information is preserved and predictable
5. **Broadcasting**: Enables efficient vectorized operations across channels
