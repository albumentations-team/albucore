import numpy as np
import pytest
import cv2
from albucore.utils import MAX_VALUES_BY_DTYPE, NPDTYPE_TO_OPENCV_DTYPE, clip
import albucore


@pytest.mark.parametrize("input_img, dtype, expected", [
    (np.array([[-300, 0], [100, 400]], dtype=np.float32), np.uint8, np.array([[0, 0], [100, 255]], dtype=np.float32)),
    (np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32), np.float32, np.array([[0, 0], [0.5, 1.0]], dtype=np.float32))
])
def test_clip(input_img, dtype, expected):
    clipped = clip(input_img, dtype=dtype)
    assert np.array_equal(clipped, expected)

valid_cv2_types = {
    cv2.CV_8U, cv2.CV_16U, cv2.CV_32F, cv2.CV_64F
}

@pytest.mark.parametrize("dtype, cv_type", NPDTYPE_TO_OPENCV_DTYPE.items())
def test_valid_cv2_types(dtype, cv_type):
    assert cv_type in valid_cv2_types, f"{cv_type} is not a valid cv2 type"
