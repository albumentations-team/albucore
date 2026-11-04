"""New image kernels preserve the caller's concrete NumPy dtype."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from albucore import box_blur, convolve, dilate, erode, linear_transformation_rgb, unsharp_mask


def filtering_results(image: NDArray[np.uint8], half: NDArray[np.float16], wide: NDArray[np.float32]) -> None:
    kernel = np.ones((3, 3), dtype=np.uint8)
    matrix = np.eye(3, dtype=np.float32)
    assert_type(box_blur(image, 3), NDArray[np.uint8])
    assert_type(box_blur(half, 3), NDArray[np.float16])
    assert_type(box_blur(wide, 3), NDArray[np.float32])
    assert_type(convolve(image, kernel), NDArray[np.uint8])
    assert_type(convolve(half, kernel), NDArray[np.float16])
    assert_type(convolve(wide, kernel), NDArray[np.float32])
    assert_type(dilate(image, kernel), NDArray[np.uint8])
    assert_type(dilate(half, kernel), NDArray[np.float16])
    assert_type(dilate(wide, kernel), NDArray[np.float32])
    assert_type(erode(image, kernel), NDArray[np.uint8])
    assert_type(erode(half, kernel), NDArray[np.float16])
    assert_type(erode(wide, kernel), NDArray[np.float32])
    assert_type(linear_transformation_rgb(image, matrix), NDArray[np.uint8])
    assert_type(linear_transformation_rgb(half, matrix), NDArray[np.float16])
    assert_type(linear_transformation_rgb(wide, matrix), NDArray[np.float32])
    assert_type(unsharp_mask(image, 3, 1.0, 0.5, 10), NDArray[np.uint8])
    assert_type(unsharp_mask(half, 3, 1.0, 0.5, 10), NDArray[np.float16])
    assert_type(unsharp_mask(wide, 3, 1.0, 0.5, 10), NDArray[np.float32])
