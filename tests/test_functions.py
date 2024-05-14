import numpy as np
import pytest
import cv2
from albucore.utils import MAX_VALUES_BY_DTYPE, NPDTYPE_TO_OPENCV_DTYPE, clip
import albucore

# Pre-defined images for testing
UINT8_IMAGE = np.random.randint(0, 255, (3, 3, 3), dtype=np.uint8)
FLOAT32_IMAGE = np.random.random((3, 3, 3)).astype(np.float32)

# Test data combining both scalar and vector multipliers
test_data = [
    (UINT8_IMAGE, 1.5),
    (FLOAT32_IMAGE, 1.5),
    (UINT8_IMAGE, np.array([1.5, 0.75, 1.1], dtype=np.float32)),
    (FLOAT32_IMAGE, np.array([1.5, 0.75, 1.1], dtype=np.float32))
]

@pytest.mark.parametrize("image, multiplier", test_data)
def test_multiply(image, multiplier):
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    result = albucore.multiply(image, multiplier)

    opencv_dtype = NPDTYPE_TO_OPENCV_DTYPE[image.dtype]

    opencv_result = np.clip(0, max_value, cv2.multiply(image, multiplier, dtype=opencv_dtype)).astype(image.dtype)

    print(opencv_result)
    print(result)

    numpy_result = image * multiplier
    numpy_result = np.clip(0, max_value, numpy_result).astype(image.dtype)

    # Check that all results match expected dtype and value arrays
    assert result.dtype == image.dtype, "Result dtype does not match expected dtype."
    assert np.array_equal(result, numpy_result), "Result does not match NumPy result."
    assert np.array_equal(result, opencv_result), "Result does not match OpenCV result."


@pytest.mark.parametrize("input_img, dtype, expected", [
    (np.array([[-300, 0], [100, 400]], dtype=np.float32), np.uint8, np.array([[0, 0], [100, 255]], dtype=np.float32)),
    (np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32), np.float32, np.array([[0, 0], [0.5, 1.0]], dtype=np.float32))
])
def test_clip(input_img, dtype, expected):
    clipped = clip(input_img, dtype=dtype)
    assert np.array_equal(clipped, expected)
