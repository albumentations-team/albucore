import pytest
import numpy as np
from albucore.utils import clip
from albucore import (
    add_lut,
    add_numpy,
    add_opencv,
    add,
    convert_value,
)


@pytest.mark.parametrize(
    "img, value, expected_output",
    [
        # Test case 1: Value as a float, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            2.0,
            np.array([[[3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14]]], dtype=np.uint8),
        ),
        (
            np.full((2, 2, 3), 254, dtype=np.uint8),
            2,
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ),
        (
            255 * np.ones((2, 2, 3), dtype=np.uint8),
            2 * np.ones((2, 2, 3), dtype=np.uint8),
            255 * np.ones((2, 2, 3), dtype=np.uint8),
        ),
        (
            254 * np.ones((2, 2, 3), dtype=np.uint8),
            np.array([2, 2, 2]),
            255 * np.ones((2, 2, 3), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 3), dtype=np.uint8),
            -1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 3), dtype=np.uint8),
            np.array([-1, -2, -3]),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 3), dtype=np.uint8),
            -2 * np.ones((2, 2, 3), dtype=np.int8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        (
            254 * np.ones((2, 2), dtype=np.uint8),
            2.0,
            255 * np.ones((2, 2), dtype=np.uint8),
        ),
        (
            np.full((2, 2), 254, dtype=np.uint8),
            3 * np.ones((2, 2), dtype=np.uint8),
            np.full((2, 2), 255, dtype=np.uint8),
        ),
        (
            np.ones((2, 2), dtype=np.uint8),
            -1,
            np.zeros((2, 2), dtype=np.uint8),
        ),
        (
            np.full((2, 2, 5), 254, dtype=np.uint8),
            2.0,
            np.full((2, 2, 5), 255, dtype=np.uint8),
        ),
        (
            np.full((2, 2, 5), 254, dtype=np.uint8),
            (3 * np.ones((2, 2, 5))).astype(np.uint8),
            np.full((2, 2, 5), 255, dtype=np.uint8),
        ),
        (
            np.full((2, 2, 5), 254, dtype=np.uint8),
            np.array([2.0] * 5),
            np.full((2, 2, 5), 255, dtype=np.uint8),
        ),
        (
            (np.ones((2, 2, 5))).astype(np.uint8),
            -1.0,
            np.zeros((2, 2, 5), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 5), dtype=np.uint8),
            np.array([-3.5] * 5),
            np.zeros((2, 2, 5), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 5), dtype=np.uint8),
            -2 * (np.ones((2, 2, 5), dtype=np.int16)),
            np.zeros((2, 2, 5), dtype=np.uint8),
        ),
        # Test case 2: Negative float, image of type uint8
        (
            np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8),
            -5.0,
            np.array([[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 115]]], dtype=np.uint8),
        ),
        # Test case 2: Negative float, image of type uint8
        (
            np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8),
            -5,
            np.array([[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 115]]], dtype=np.uint8),
        ),
        # Test case 2: Value as a vector, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[1, 3, 5], [4, 6, 8]], [[7, 9, 11], [10, 12, 14]]], dtype=np.uint8),
        ),
        # Test case 3: Vector with positive and negative values, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, -1.5, 2.0], dtype=np.float32),
            np.array([[[1, 1, 5], [4, 4, 8]], [[7, 7, 11], [10, 10, 14]]], dtype=np.uint8),
        ),
        # Test case 3: Value as an array, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, 0.5, 0.25], [0.5, 1, 1.5]], [[1.5, 2, 0.5], [0.25, 0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 7]], [[8, 10, 9], [10, 11, 13]]], dtype=np.uint8),
        ),
        # Test case 4: Array with positive and negative values, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, -0.5, 0.25], [-0.5, 1, -1.5]], [[-1.5, 2, 0.5], [0.25, -0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 5]], [[6, 10, 9], [10, 11, 13]]], dtype=np.uint8),
        ),
        # Test case 4: Value as a float, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            0.2,
            np.array([[[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]], [[0.9, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 6: Negative float, image of type float32
        (
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], [[1.4, 1.6, 1.8], [2.0, 2.2, 2.4]]], dtype=np.float32),
            -0.2,
            np.array([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 5: Value as a vector, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[0.6, 1.0, 1.0], [0.9, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 6: Value as an array, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 7: Vector with positive and negative values, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, -0.2, 0.1], dtype=np.float32),
            np.array([[[0.6, 0.0, 0.4], [0.9, 0.3, 0.7]], [[1.0, 0.6, 1.0], [1.0, 0.9, 1.0]]], dtype=np.float32),
        ),
        # Test case 8: Array with positive and negative values, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], [[-0.7, 0.8, -0.9], [1.0, -1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.2, 0.0, 0.6], [0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]], dtype=np.float32),
        ),
        # Clipping effect test for uint8
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            60,
            np.array([[[160, 210, 255], [255, 255, 160]], [[110, 135, 185], [235, 255, 255]]], dtype=np.uint8),
        ),
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            -60,
            np.array([[[40, 90, 140], [190, 195, 40]], [[0, 15, 65], [115, 140, 165]]], dtype=np.uint8),
        ),
        # Clipping effect test for float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            1.0,
            np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
    ],
)
def test_add_consistency(img, value, expected_output):
    result_numpy = clip(add_numpy(img, value), img.dtype)

    assert np.allclose(result_numpy, expected_output, atol=1e-6)

    result_opencv = clip(add_opencv(img, value), img.dtype)
    assert np.allclose(result_opencv, expected_output, atol=1e-6)

    if img.dtype == np.uint8 and not (isinstance(value, np.ndarray) and value.ndim > 1):
        result_lut = add_lut(img, value)
        assert np.allclose(result_lut, expected_output, atol=1e-6)


@pytest.mark.parametrize(
    "img_dtype", [np.uint8, np.float32]
)
@pytest.mark.parametrize(
    "num_channels", [1, 3, 5]
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        np.array([1.4]),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1], np.float32),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1, 2.0], np.float32),
    ]
)
@pytest.mark.parametrize(
    "is_contiguous", [True, False]
)
def test_add(img_dtype, num_channels, value, is_contiguous):
    np.random.seed(0)

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

    processed_value = convert_value(value, num_channels)

    result_numpy = clip(add_numpy(img, processed_value), img_dtype)

    result_opencv = clip(add_opencv(img, processed_value), img.dtype)
    assert np.array_equal(img, original_image), "Input image was modified"

    assert np.allclose(result_opencv, result_numpy, atol=1e-6)

    if img.dtype == np.uint8:
        result_lut = add_lut(img, processed_value)
        assert np.array_equal(result_numpy, result_lut), f"Difference {(result_numpy - result_lut).mean()}"

    result = add(img, value)

    assert result.dtype == img.dtype
    assert result.shape == img.shape

    assert np.allclose(result, result_opencv, atol=1e-6), f"Difference {(result - result_opencv).max()}"

    assert np.array_equal(img, original_image), "Input image was modified"


@pytest.mark.parametrize(
    ["shift_params", "expected"], [[(-10, 0, 10), (117, 127, 137)], [(-200, 0, 200), (0, 127, 255)]]
)
def test_shift_rgb(shift_params, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    r_shift, g_shift, b_shift = shift_params
    img = add(img, np.array((r_shift, g_shift, b_shift)))
    expected_r, expected_g, expected_b = expected
    assert img.dtype == np.dtype("uint8")
    assert (img[:, :, 0] == expected_r).all()
    assert (img[:, :, 1] == expected_g).all()
    assert (img[:, :, 2] == expected_b).all()


@pytest.mark.parametrize(
    ["shift_params", "expected"], [[(-0.1, 0, 0.1), (0.3, 0.4, 0.5)], [(-0.6, 0, 0.6), (0, 0.4, 1.0)]]
)
def test_shift_rgb_float(shift_params, expected):
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    r_shift, g_shift, b_shift = shift_params
    img = add(img, np.array((r_shift, g_shift, b_shift)))
    expected_r, expected_g, expected_b = [
        np.ones((100, 100), dtype=np.float32) * channel_value for channel_value in expected
    ]
    assert img.dtype == np.dtype("float32")
    np.array_equal(img[:, :, 0], expected_r)
    np.array_equal(img[:, :, 1], expected_g)
    np.array_equal(img[:, :, 2], expected_b)
