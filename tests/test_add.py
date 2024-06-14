import pytest
import numpy as np
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, clip
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
        # Test case 2: Value as a vector, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[1, 4, 5], [4, 7, 8]], [[7, 10, 11], [10, 13, 14]]], dtype=np.uint8),
        ),
        # Test case 3: Value as an array, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, 0.5, 0.25], [0.5, 1, 1.5]], [[1.5, 2, 0.5], [0.25, 0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 8]], [[9, 10, 9], [10, 12, 13]]], dtype=np.uint8),
        ),
        # Test case 4: Value as a float, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            0.2,
            np.array([[[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]], [[0.9, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
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
        # Clipping effect test for uint8
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            60,
            np.array([[[160, 210, 255], [255, 255, 160]], [[110, 135, 185], [235, 255, 255]]], dtype=np.uint8),
        ),
        # Clipping effect test for float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            1.0,
            np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
    ],
)
def test_add(img, value, expected_output):
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
        [1.4],
        (1.6),
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
        result_lut = clip(add_lut(img, processed_value), img.dtype)
        assert np.array_equal(result_numpy, result_lut), f"Difference {(result_numpy - result_lut).mean()}"

    result = add(img, value)
    assert np.allclose(result, result_opencv, atol=1e-6), f"Difference {(result - result_opencv).max()}"

    assert np.array_equal(img, original_image), "Input image was modified"

@pytest.mark.parametrize(
    ["shift_params", "expected"], [[(-10, 0, 10), (117, 127, 137)], [(-200, 0, 200), (0, 127, 255)]]
)
def test_shift_rgb(shift_params, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    r_shift, g_shift, b_shift = shift_params
    img = add(img, (r_shift, g_shift, b_shift))
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
    img = add(img, (r_shift, g_shift, b_shift))
    expected_r, expected_g, expected_b = [
        np.ones((100, 100), dtype=np.float32) * channel_value for channel_value in expected
    ]
    assert img.dtype == np.dtype("float32")
    np.array_equal(img[:, :, 0], expected_r)
    np.array_equal(img[:, :, 1], expected_g)
    np.array_equal(img[:, :, 2], expected_b)
