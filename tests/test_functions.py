from albucore.utils import MAX_VALUES_BY_DTYPE
import numpy as np
import pytest
import cv2

import albucore

UNIT8_IMAGE = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
FLOAT32_IMAGE = np.random.random((10, 10, 3)).astype(np.float32)

@pytest.mark.parametrize("image", [UNIT8_IMAGE, FLOAT32_IMAGE])
def test_multiply_by_constant(image):
    value = 1.5
    result = albucore.multiply_by_constant(image, value)

    numpy_result = image * value

    if image.dtype == np.uint8:
        numpy_result = np.clip(numpy_result, 0, MAX_VALUES_BY_DTYPE[image.dtype]).astype(np.uint8)

    opencv_result = cv2.multiply(image, value)

    np.array_equal(result, numpy_result)
    np.array_equal(result, opencv_result)
