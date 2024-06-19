import pytest
import numpy as np


from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, clip

from albucore import (
    multiply_lut,
    multiply_numpy,
    multiply_opencv,
    multiply,
    convert_value,
)


@pytest.mark.parametrize(
    "img, multiplier, expected_output",
    [
        # Test case 1: Multiplier as a float, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            2.0,
            np.array([[[2, 4, 6], [8, 10, 12]], [[14, 16, 18], [20, 22, 24]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            -2,
            np.zeros((2, 2, 3)),
        ),
        # Test case 2: Multiplier as a vector, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[0, 3, 6], [2, 7, 12]], [[3, 12, 18], [5, 16, 24]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([-0.5, -1.5, -2.0], dtype=np.float32),
            np.zeros((2, 2, 3)),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([-0.5, 1.5, -2.0], dtype=np.float32),
            np.array([[[0, 3, 0], [0, 7, 0]], [[0, 12, 0], [0, 16, 0]]], dtype=np.uint8),
        ),
        # Test case 3: Multiplier as an array, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, 0.5, 0.25], [0.5, 1, 1.5]], [[1.5, 2, 0.5], [0.25, 0.75, 1]]], dtype=np.float32),
            np.array([[[1, 1, 0], [2, 5, 9]], [[10, 16, 4], [2, 8, 12]]], dtype=np.uint8),
        ),
        # Test case 4: Multiplier as a float, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            2.0,
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 5: Multiplier as a vector, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[0.05, 0.3, 0.6], [0.2, 0.75, 1.0]], [[0.35, 1.0, 1.0], [0.5, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 6: Multiplier as an array, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[1.0, 0.5, 0.25], [0.5, 1.0, 1.5]], [[1.5, 2.0, 0.5], [0.25, 0.75, 1.0]]], dtype=np.float32),
            np.array([[[0.1, 0.1, 0.075], [0.2, 0.5, 0.9]], [[1.0, 1.0, 0.45], [0.25, 0.825, 1.0]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[-1.0, 0.5, -0.25], [0.5, -1.0, 1.5]], [[-1.5, 2.0, -0.5], [0.25, -0.75, 1.0]]], dtype=np.float32),
            np.array([[[0, 0.1, 0], [0.2, 0, 0.9]], [[0, 1.0, 0], [0.25, 0, 1.0]]], dtype=np.float32),
        ),
        # Clipping effect test for uint8
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            2.0,
            np.array([[[200, 255, 255], [255, 255, 200]], [[100, 150, 250], [255, 255, 255]]], dtype=np.uint8),
        ),
        # Clipping effect test for float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            2.0,
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # New test case 7: Multiplier as a vector, image of type uint8 with 4 channels
        (
            np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0, 2.5], dtype=np.float32),
            np.array([[[0, 3, 6, 10], [2, 9, 14, 20]], [[4, 15, 22, 30], [6, 21, 30, 40]]], dtype=np.uint8),
        ),
                (
            np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.uint8),
            np.array([-0.5, 1.5, 2.0, -2.5], dtype=np.float32),
            np.array([[[0, 3, 6, 0], [0, 9, 14, 0]], [[0, 15, 22, 0], [0, 21, 30, 0]]], dtype=np.uint8),
        ),
        # New test case 8: Multiplier as a vector, image of type float32 with 4 channels
        (
            np.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0, 2.5], dtype=np.float32),
            np.array([[[0.05, 0.3, 0.6, 1.0], [0.25, 0.9, 1.0, 1.0]], [[0.45, 1.0, 1.0, 1.0], [0.65, 1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 1: Multiplier as a float, image of type uint8, shape (height, width)
        (
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            2.0,
            np.array([[2, 4, 6], [8, 10, 12]], dtype=np.uint8),
        ),
        # Test case 2: Multiplier as a float, image of type uint8, shape (height, width, 1)
        (
            np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8),
            2.0,
            np.array([[[2], [4]], [[6], [8]]], dtype=np.uint8),
        ),
        # Test case 3: Multiplier as a float, image of type float32, shape (height, width)
        (
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            2.0,
            np.array([[0.2, 0.4, 0.6], [0.8, 1.0, 1.0]], dtype=np.float32),
        ),
        # Test case 4: Multiplier as a float, image of type float32, shape (height, width, 1)
        (
            np.array([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=np.float32),
            2.0,
            np.array([[[0.2], [0.4]], [[0.6], [0.8]]], dtype=np.float32),
        ),
        # Test case 7: Multiplier as a vector, image of type uint8 with 5 channels
        (
            np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0, 2.5, 3.0], dtype=np.float32),
            np.array([[[0, 3, 6, 10, 15], [3, 10, 16, 22, 30]], [[5, 18, 26, 35, 45], [8, 25, 36, 47, 60]]], dtype=np.uint8),
        ),
        # Test case 8: Multiplier as a vector, image of type float32 with 5 channels
        (
            np.array([[[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], [[1.1, 1.2, 1.3, 1.4, 1.5], [1.6, 1.7, 1.8, 1.9, 2.0]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0, 2.5, 3.0], dtype=np.float32),
            np.array([[[0.05, 0.3, 0.6, 1.0, 1.0], [0.3, 1.0, 1.0, 1.0, 1.0]], [[0.55, 1.0, 1.0, 1.0, 1.0], [0.8, 1.0, 1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        (
            np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8),
            2.0,
            np.array([[[2], [4]], [[6], [8]]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            2.0,
            np.array([[2, 4, 6], [8, 10, 12]], dtype=np.uint8),
        ),
    ],
)
def test_multiply_with_numpy(img, multiplier, expected_output):
    result_numpy = clip(multiply_numpy(img, multiplier), img.dtype)
    assert np.allclose(result_numpy, expected_output, atol=1e-6)

    result_opencv = clip(multiply_opencv(img, multiplier), img.dtype)
    assert np.allclose(result_opencv, expected_output, atol=1e-6)

    if img.dtype == np.uint8 and not (isinstance(multiplier, np.ndarray) and multiplier.ndim > 1):
        result_lut = multiply_lut(img, multiplier)
        assert np.allclose(result_lut, expected_output, atol=1e-6)


@pytest.mark.parametrize("img_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize(
    "multiplier",
    [
        1.4,
        np.array([1.5]),
        np.array((1.6)),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1], np.float32),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1, 2.0], np.float32),
    ]
)
@pytest.mark.parametrize("is_contiguous", [True, False])
def test_multiply(img_dtype, num_channels, multiplier, is_contiguous):
    height, width = 9, 11

    if is_contiguous:
        if img_dtype == np.uint8:
            img = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype)
        else:
            img = np.random.rand(height, width, num_channels).astype(img_dtype)
    else:
        if img_dtype == np.uint8:
            img = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0)
        else:
            img = np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)

    original_image = img.copy()

    processed_multiplier = convert_value(multiplier, num_channels)

    result = multiply(img, multiplier)

    assert result.dtype == img.dtype
    assert result.shape == img.shape

    result_numpy = clip(multiply_numpy(img, processed_multiplier), img.dtype)

    assert np.allclose(result, result_numpy, atol=1e-6)

    if img.dtype == np.uint8:
        result_lut = multiply_lut(img, processed_multiplier)
        assert np.array_equal(img, original_image), "Input image was modified"
        assert np.array_equal(result, result_lut), f"Difference {(result - result_lut).mean()}"

    result_opencv = clip(multiply_opencv(img, processed_multiplier), img.dtype)

    assert np.array_equal(img, original_image), "Input image was modified"

    assert np.allclose(result, result_opencv, atol=1e-6), f"Difference {(result - result_opencv).max()}"
