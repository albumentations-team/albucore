import pytest
import numpy as np
import cv2
from albucore.functions import multiply_add, multiply_add_numpy, multiply_add_opencv, multiply_add_lut
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, clip

@pytest.mark.parametrize(
    "img, value, factor, expected_output",
    [
        # Test case 1: Both factor and value are 1, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            1.0,
            np.array([[2, 3], [4, 5]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            -1.0,
            2.0,
            np.array([[1, 3], [5, 7]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            0.0,
            np.ones((2, 2))
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        # Test case 2: Different factor and value, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            2.0,
            0.5,
            np.array([[2, 3], [3, 4]], dtype=np.uint8),
        ),
        # Test case 3: Zero factor and value, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            0.0,
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
        ),
        # Test case 4: Factor 1 and value 0, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        # Test case 5: Both factor and value are 1, image of type float32
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            1.0,
            np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        ),
        # Test case 6: Different factor and value, image of type float32
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            0.5,
            0.5,
            np.array([[0.55, 0.6], [0.65, 0.7]], dtype=np.float32),
        ),
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            -0.1,
            0.5,
            np.array([[0, 0], [0.05, 0.1]], dtype=np.float32),
        ),
    ],
)
def test_multiply_add_numpy(img, value, factor, expected_output):
    result_numpy = clip(multiply_add_numpy(img, factor, value), img.dtype)
    assert np.allclose(result_numpy, expected_output, atol=1e-6)

    result_opencv = clip(multiply_add_opencv(img, factor, value), img.dtype)
    assert np.allclose(result_opencv, expected_output, atol=1e-6)

    if img.dtype == np.uint8:
        result_lut = multiply_add_lut(img, factor, value)
        assert np.allclose(result_lut, expected_output, atol=1e-6)


@pytest.mark.parametrize(
    "img_dtype", [np.uint8, np.float32]
)
@pytest.mark.parametrize(
    "num_channels", [1, 3, 5]
)
@pytest.mark.parametrize(
    "value, factor",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (-0.2, 0.5),
        (1.0, 1.0),
        (2.0, 0.5),
    ]
)
@pytest.mark.parametrize(
    "is_contiguous", [True, False]
)
def test_multiply_add(img_dtype, num_channels, value, factor, is_contiguous):
    np.random.seed(0)
    height, width = 9, 11

    if is_contiguous:
        img = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype) if img_dtype == np.uint8 else np.random.rand(height, width, num_channels).astype(img_dtype)
    else:
        img = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0) if img_dtype == np.uint8 else np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)

    original_img = img.copy()

    result = multiply_add(img, factor, value)

    assert result.shape == original_img.shape
    assert result.dtype == original_img.dtype

    assert np.array_equal(img, original_img), "Input img was modified"

    result_numpy = clip(multiply_add_numpy(img, factor, value), img_dtype)
    assert np.allclose(result, result_numpy, atol=1e-6)


    if img.dtype == np.uint8:
        result_lut = clip(multiply_add_lut(img, factor, value), img.dtype)
        assert np.array_equal(result, result_lut), f"Difference {(result - result_lut).mean()}"

    result_opencv = clip(multiply_add_opencv(img, factor, value), img.dtype)
    assert np.allclose(result, result_opencv, atol=1e-6), f"Difference {(result - result_opencv).max()}"
