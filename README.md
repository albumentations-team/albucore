# Albucore: High-Performance Image Processing Functions

Albucore is a library of optimized atomic functions designed for efficient image processing. These functions serve as the foundation for [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX), a popular image augmentation library.

## Overview

Image processing operations can be implemented in various ways, each with its own performance characteristics depending on the image type, size, and number of channels. Albucore aims to provide the fastest implementation for each operation by leveraging different backends such as NumPy, OpenCV, and custom optimized code.

Key features:

- Optimized atomic image processing functions
- Automatic selection of the fastest implementation based on input image characteristics
- Seamless integration with Albumentations
- Extensive benchmarking for performance validation

## GitAds Sponsored
[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=albumentations-team/albucore@github)](https://gitads.dev/v1/ad-track?source=albumentations-team/albucore@github)


## Installation

```bash
pip install albucore
```

## Usage

```python
import numpy as np
import albucore

# Create a sample RGB image
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Apply a function
result = albucore.multiply(image, 1.5)

# For grayscale images, ensure the channel dimension is present
gray_image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
gray_result = albucore.multiply(gray_image, 1.5)
```

Albucore automatically selects the most efficient implementation based on the input image type and characteristics.

## Shape Conventions

Albucore expects images to follow specific shape conventions, with the channel dimension always present:

- **Single image**: `(H, W, C)` - Height, Width, Channels
- **Grayscale image**: `(H, W, 1)` - Height, Width, 1 channel
- **Batch of images**: `(N, H, W, C)` - Number of images, Height, Width, Channels
- **3D volume**: `(D, H, W, C)` - Depth, Height, Width, Channels
- **Batch of volumes**: `(N, D, H, W, C)` - Number of volumes, Depth, Height, Width, Channels

### Important Notes:

1. **Channel dimension is always required**, even for grayscale images (use shape `(H, W, 1)`)
2. Single-channel images should have shape `(H, W, 1)` not `(H, W)`
3. Batches and volumes are treated uniformly - a 4D array `(N, H, W, C)` can represent either a batch of images or a 3D volume

### Examples:

```python
import numpy as np
import albucore

# Grayscale image - MUST have explicit channel dimension
gray_image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)

# RGB image
rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Batch of 10 grayscale images
batch_gray = np.random.randint(0, 256, (10, 100, 100, 1), dtype=np.uint8)

# 3D volume with 20 slices
volume = np.random.randint(0, 256, (20, 100, 100, 1), dtype=np.uint8)

# Batch of 5 RGB volumes, each with 20 slices
batch_volumes = np.random.randint(0, 256, (5, 20, 100, 100, 3), dtype=np.uint8)
```

## Functions

Albucore includes optimized implementations for various image processing operations, including:

- Arithmetic operations (add, multiply, power)
- Normalization (per-channel, global)
- Geometric transformations (vertical flip, horizontal flip)
- Helper decorators (to_float, to_uint8)

### Batch Processing

Many functions in Albucore support batch processing out of the box. The library automatically handles different input shapes:

- Single images: `(H, W, C)`
- Batches: `(N, H, W, C)`
- Volumes: `(D, H, W, C)`
- Batch of volumes: `(N, D, H, W, C)`

Functions will preserve the input shape structure, applying operations efficiently across all images/slices in the batch.

### Decorators

Albucore provides several useful decorators:

- `@preserve_channel_dim`: Ensures single-channel images maintain their shape `(H, W, 1)` when OpenCV operations might drop the channel dimension
- `@contiguous`: Ensures arrays are C-contiguous for optimal performance
- `@uint8_io` and `@float32_io`: Handle automatic type conversions for functions that work best with specific data types

See [docs/decorators.md](docs/decorators.md) for detailed documentation on all decorators.

## Performance

Albucore uses a combination of techniques to achieve high performance:

1. **Multiple Implementations**: Each function may have several implementations using different backends (NumPy, OpenCV, custom code).
2. **Automatic Selection**: The library automatically chooses the fastest implementation based on the input image type, size, and number of channels.
3. **Optimized Algorithms**: Custom implementations are optimized for specific use cases, often outperforming general-purpose libraries.

See [docs/performance-optimization.md](docs/performance-optimization.md) for detailed performance guidelines and best practices.

## Documentation

- [CLAUDE.md](CLAUDE.md) - AI development guidelines for working with this codebase
- [docs/image-conventions.md](docs/image-conventions.md) - Image shape conventions and requirements
- [docs/decorators.md](docs/decorators.md) - Decorator usage and patterns
- [docs/performance-optimization.md](docs/performance-optimization.md) - Performance optimization guidelines

## License

MIT

## Acknowledgements

Albucore is part of the [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX) project. We'd like to thank all contributors to [AlbumentationsX](https://albumentations.ai/people) and the broader computer vision community for their inspiration and support.

<!-- GitAds-Verify: 1LSAKH1Y2GKIISALRDIFCG2T9YYNR5WD -->
