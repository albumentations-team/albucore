# Albucore

Albucore is a high-performance image processing library designed to optimize operations on images using Python and OpenCV, building upon the foundations laid by the popular Albumentations library. It offers specialized optimizations for different image data types and aims to provide faster processing times through efficient algorithm implementations.

## Features

- Optimized image multiplication operations for both `uint8` and `float32` data types.
- Support for single-channel and multi-channel images.
- Custom decorators to manage channel dimensions and output constraints.

## Installation

Install Albucore using pip:

```bash
pip install -U albucore
```

## Example

Here's how you can use Albucore to multiply an image by a constant or a vector:

```python
import cv2
import numpy as np
from albucore import multiply

# Load an image
img = cv2.imread('path_to_your_image.jpg')

# Multiply by a constant
multiplied_image = multiply(img, 1.5)

# Multiply by a vector
multiplier = [1.5, 1.2, 0.9]  # Different multiplier for each channel
multiplied_image = multiply(img, multiplier)
```

## Benchmarks

### Benchmark Results for 1000 Images of `float32` Type (256, 256, 1)

|                  | albucore     | opencv           | numpy            |
| ---------------- | ------------ | ---------------- | ---------------- |
| MultiplyConstant | 12925 ± 1237 | 10963 ± 1053     | **14040 ± 2063** |
| MultiplyVector   | 3832 ± 512   | **10824 ± 1005** | 8986 ± 511       |

### Benchmark Results for 1000 Images of `uint8` Type (256, 256, 1)

|                  | albucore         | opencv      | numpy      |
| ---------------- | ---------------- | ----------- | ---------- |
| MultiplyConstant | **24131 ± 1129** | 11622 ± 175 | 6969 ± 643 |
| MultiplyVector   | **24279 ± 908**  | 11756 ± 152 | 6936 ± 408 |

Albucore provides significant performance improvements for image processing tasks. Here are some benchmark results comparing Albucore with OpenCV and Numpy:

For more detailed benchmark results, including other configurations and data types, refer to the [Benchmark](benchmark/results/) in the repository.

## License

Distributed under the MIT License. See LICENSE for more information.
