import numpy as np
import pytest
import cv2
from albucore.utils import NPDTYPE_TO_OPENCV_DTYPE, clip, convert_value, get_opencv_dtype_from_numpy, contiguous


@pytest.mark.parametrize("input_img, dtype, expected", [
    (np.array([[-300, 0], [100, 400]], dtype=np.float32), np.uint8, np.array([[0, 0], [100, 255]], dtype=np.float32)),
    (np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32), np.float32, np.array([[0, 0], [0.5, 1.0]], dtype=np.float32))
])
def test_clip(input_img, dtype, expected):
    clipped = clip(input_img, dtype=dtype)
    assert np.array_equal(clipped, expected)

valid_cv2_types = {
    cv2.CV_8U, cv2.CV_16U, cv2.CV_32F, cv2.CV_64F, cv2.CV_32S
}

@pytest.mark.parametrize("cv_type", NPDTYPE_TO_OPENCV_DTYPE.values())
def test_valid_cv2_types(cv_type):
    assert cv_type in valid_cv2_types, f"{cv_type} is not a valid cv2 type"


def test_cv_dtype_from_np():
    assert get_opencv_dtype_from_numpy(np.uint8) == cv2.CV_8U
    assert get_opencv_dtype_from_numpy(np.uint16) == cv2.CV_16U
    assert get_opencv_dtype_from_numpy(np.float32) == cv2.CV_32F
    assert get_opencv_dtype_from_numpy(np.float64) == cv2.CV_64F
    assert get_opencv_dtype_from_numpy(np.int32) == cv2.CV_32S

    assert get_opencv_dtype_from_numpy(np.dtype("uint8")) == cv2.CV_8U
    assert get_opencv_dtype_from_numpy(np.dtype("uint16")) == cv2.CV_16U
    assert get_opencv_dtype_from_numpy(np.dtype("float32")) == cv2.CV_32F
    assert get_opencv_dtype_from_numpy(np.dtype("float64")) == cv2.CV_64F
    assert get_opencv_dtype_from_numpy(np.dtype("int32")) == cv2.CV_32S


@pytest.mark.parametrize(
    "value, num_channels, expected",
    [
        ((1.5), 1, 1.5),
        (np.array([1.5]), 3, 1.5),
        (np.array([1.5, 2.5]), 1, 1.5),
        (np.array([1.5, 2.5, 0.5]), 2, np.array([1.5, 2.5,], dtype=np.float32)),
        (3, 2, 3),
        (np.array((1.5)), 2, 1.5),
        (np.reciprocal(np.array(0.2, dtype=np.float64)), 2, 5),
    ]
)
def test_convert_value(value, num_channels, expected):
    result = convert_value(value, num_channels)
    if isinstance(expected, np.ndarray):
        assert np.array_equal(result, expected)
    else:
         assert result == expected

@contiguous
def process_image(img: np.ndarray) -> np.ndarray:
    # For demonstration, let's just return a non-contiguous view
    return img[::-1, ::-1]

@pytest.mark.parametrize(
    "input_array",
    [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # C-contiguous array
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[::-1, ::-1],  # Non-contiguous view
        np.arange(100).reshape(10, 10),  # Another C-contiguous array
        np.ones([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)  # 3D array with transpose
    ]
)
def test_contiguous_decorator(input_array):
    # Check if input is made contiguous
    contiguous_input = np.require(input_array, requirements=["C_CONTIGUOUS"])

    assert contiguous_input.flags["C_CONTIGUOUS"], "Input array is not C-contiguous"

    # Process the array using the decorated function
    output_array = process_image(input_array)

    # Check if output is contiguous
    assert output_array.flags["C_CONTIGUOUS"], "Output array is not C-contiguous"

    non_contiguous_array = np.asfortranarray(input_array)

    output_array = process_image(non_contiguous_array)
    assert output_array.flags["C_CONTIGUOUS"], "Output array is not C-contiguous"

    # Check if the content is correct (same as reversing the original array)
    expected_output = input_array[::-1, ::-1]
    np.testing.assert_array_equal(output_array, expected_output), "Output array content is not as expected"
