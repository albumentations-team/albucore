# Albucore: High-Performance Image Processing Functions

Albucore is a library of optimized atomic functions designed for efficient image processing. These functions serve as the foundation for [Albumentations](https://github.com/albumentations-team/albumentations), a popular image augmentation library.

## Overview

Image processing operations can be implemented in various ways, each with its own performance characteristics depending on the image type, size, and number of channels. Albucore aims to provide the fastest implementation for each operation by leveraging different backends such as NumPy, OpenCV, and custom optimized code.

Key features:

- Optimized atomic image processing functions
- Automatic selection of the fastest implementation based on input image characteristics
- Seamless integration with Albumentations
- Extensive benchmarking for performance validation

## Installation

```bash
pip install albucore
```

## Usage

```python
import numpy as np
import albucore
# Create a sample image
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
# Apply a function
result = albucore.multiply(image, 1.5)
```

Albucore automatically selects the most efficient implementation based on the input image type and characteristics.

## Functions

Albucore includes optimized implementations for various image processing operations, including:

- Arithmetic operations (add, multiply, power)
- Normalization (per-channel, global)
- Geometric transformations (vertical flip, horizontal flip)
- Helper decorators (to_float, to_uint8)

## Performance

Albucore uses a combination of techniques to achieve high performance:

1. **Multiple Implementations**: Each function may have several implementations using different backends (NumPy, OpenCV, custom code).
2. **Automatic Selection**: The library automatically chooses the fastest implementation based on the input image type, size, and number of channels.
3. **Optimized Algorithms**: Custom implementations are optimized for specific use cases, often outperforming general-purpose libraries.

### Benchmarks

We maintain an extensive benchmark suite to ensure Albucore's performance across various scenarios. You can find the benchmarks and their results in the [benchmarks](./benchmarks/README.md) directory.

## License

MIT

## Acknowledgements

Albucore is part of the [Albumentations](https://github.com/albumentations-team/albumentations) project. We'd like to thank all contributors to [Albumentations](https://albumentations.ai/) and the broader computer vision community for their inspiration and support.
