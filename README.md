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

For detailed benchmark results, including other configurations and data types, refer to the [Benchmark](benchmark/results/) in the repository.

## License

Distributed under the MIT License. See LICENSE for more information.
