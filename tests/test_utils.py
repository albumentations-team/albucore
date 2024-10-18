import numpy as np
import pytest
import cv2
from albucore.decorators import contiguous
from albucore.functions import float32_io, from_float, to_float, uint8_io
from albucore.utils import NPDTYPE_TO_OPENCV_DTYPE, clip, convert_value, get_opencv_dtype_from_numpy


@pytest.mark.parametrize("input_img, dtype, expected", [
    (np.array([[-300, 0], [100, 400]], dtype=np.float32), np.uint8, np.array([[0, 0], [100, 255]], dtype=np.float32)),
    (np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32), np.float32, np.array([[0, 0], [0.5, 1.0]], dtype=np.float32))
])
def test_clip(input_img, dtype, expected):
    clipped = clip(input_img, dtype=dtype)
    np.testing.assert_array_equal(clipped, expected)

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
        np.testing.assert_array_equal(result, expected)
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


# Sample functions to be wrapped
@float32_io
def dummy_float32_func(img):
    return img * 2

@uint8_io
def dummy_uint8_func(img):
    return np.clip(img + 10, 0, 255).astype(np.uint8)

# Test data
@pytest.fixture(params=[
    np.uint8, np.float32
])
def test_image(request):
    dtype = request.param
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 256, (10, 10, 3), dtype=dtype)
    else:
        return np.random.rand(10, 10, 3).astype(dtype)

# Tests
@pytest.mark.parametrize("wrapper,func, image", [
    (float32_io, dummy_float32_func, np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)),
    (uint8_io, dummy_uint8_func, np.random.rand(10, 10, 3).astype(np.float32))
])
def test_io_wrapper(wrapper, func, image):
    input_dtype = image.dtype
    result = func(image)

    # Check if the output dtype matches the input dtype
    assert result.dtype == input_dtype

    # Check if the function was actually applied
    if wrapper == float32_io:
        expected = from_float(to_float(image) * 2, input_dtype)
    else:  # uint8_io
        expected = to_float(from_float(image, np.uint8) + 10)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("wrapper,func,expected_intermediate_dtype", [
    (float32_io, dummy_float32_func, np.float32),
    (uint8_io, dummy_uint8_func, np.uint8)
])
def test_intermediate_dtype(wrapper, func, expected_intermediate_dtype, test_image):
    original_func = func.__wrapped__  # Access the original function

    def check_dtype(img):
        assert img.dtype == expected_intermediate_dtype
        return original_func(img)

    wrapped_func = wrapper(check_dtype)
    wrapped_func(test_image)  # This will raise an assertion error if the intermediate dtype is incorrect

def test_float32_io_preserves_float32(test_image):
    if test_image.dtype == np.float32:
        result = dummy_float32_func(test_image)
        assert result.dtype == np.float32

def test_uint8_io_preserves_uint8(test_image):
    if test_image.dtype == np.uint8:
        result = dummy_uint8_func(test_image)
        assert result.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_float32_io_does_not_modify_input(dtype):
    @float32_io
    def identity(img):
        return img

    shape = (10, 10, 3)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, np.iinfo(dtype).max, shape).astype(dtype)
    else:
        original = np.random.rand(*shape).astype(dtype)

    original_copy = original.copy()
    _ = identity(original)

    np.testing.assert_array_equal(original, original_copy)

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_uint8_io_does_not_modify_input(dtype):
    @uint8_io
    def identity(img):
        return img

    shape = (10, 10, 3)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, np.iinfo(dtype).max, shape).astype(dtype)
    else:
        original = np.random.rand(*shape).astype(dtype)

    original_copy = original.copy()
    _ = identity(original)

    np.testing.assert_array_equal(original, original_copy)

@pytest.mark.parametrize("wrapper", [float32_io, uint8_io])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_wrapper_preserves_dtype(wrapper, dtype):
    @wrapper
    def identity(img):
        return img

    shape = (10, 10, 3)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, np.iinfo(dtype).max, shape).astype(dtype)
    else:
        original = np.random.rand(*shape).astype(dtype)

    result = identity(original)

    assert result.dtype == dtype

@pytest.mark.parametrize("wrapper", [float32_io, uint8_io])
def test_wrapper_intermediate_dtype(wrapper):
    intermediate_dtype = np.float32 if wrapper == float32_io else np.uint8

    @wrapper
    def check_dtype(img):
        assert img.dtype == intermediate_dtype
        return img

    original = np.random.rand(10, 10, 3).astype(np.float32)
    _ = check_dtype(original)
