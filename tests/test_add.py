import numpy as np
import pytest

from albucore import add, convert_value


@pytest.mark.parametrize(
    "img, value, expected_output",
    [
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
        ((254 * np.ones((2, 2), dtype=np.uint8))[..., None], 2.0, (255 * np.ones((2, 2), dtype=np.uint8))[..., None]),
        (
            np.full((2, 2), 254, dtype=np.uint8)[..., None],
            (3 * np.ones((2, 2), dtype=np.uint8))[..., None],
            np.full((2, 2), 255, dtype=np.uint8)[..., None],
        ),
        (np.ones((2, 2), dtype=np.uint8)[..., None], -1, np.zeros((2, 2), dtype=np.uint8)[..., None]),
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
        (
            np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8),
            -5.0,
            np.array([[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 115]]], dtype=np.uint8),
        ),
        (
            np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8),
            -5,
            np.array([[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 115]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[1, 3, 5], [4, 6, 8]], [[7, 9, 11], [10, 12, 14]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, -1.5, 2.0], dtype=np.float32),
            np.array([[[1, 1, 5], [4, 4, 8]], [[7, 7, 11], [10, 10, 14]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, 0.5, 0.25], [0.5, 1, 1.5]], [[1.5, 2, 0.5], [0.25, 0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 7]], [[8, 10, 9], [10, 11, 13]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, -0.5, 0.25], [-0.5, 1, -1.5]], [[-1.5, 2, 0.5], [0.25, -0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 5]], [[6, 10, 9], [10, 11, 13]]], dtype=np.uint8),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            0.2,
            np.array([[[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1], [1.2, 1.3, 1.4]]], dtype=np.float32),
        ),
        (
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], [[1.4, 1.6, 1.8], [2.0, 2.2, 2.4]]], dtype=np.float32),
            -0.2,
            np.array([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]], [[1.2, 1.4, 1.6], [1.8, 2.0, 2.2]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[0.6, 1.7, 2.3], [0.9, 2.0, 2.6]], [[1.2, 2.3, 2.9], [1.5, 2.6, 3.2]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], [[1.4, 1.6, 1.8], [2.0, 2.2, 2.4]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, -0.2, 0.1], dtype=np.float32),
            np.array([[[0.6, 0.0, 0.4], [0.9, 0.3, 0.7]], [[1.2, 0.6, 1.0], [1.5, 0.9, 1.3]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], [[-0.7, 0.8, -0.9], [1.0, -1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.2, 0.0, 0.6], [0.0, 1.0, 0.0]], [[0.0, 1.6, 0.0], [2.0, 0.0, 2.4]]], dtype=np.float32),
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
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            1.0,
            np.array([[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]], dtype=np.float32),
        ),
    ],
)
def test_add_consistency(img, value, expected_output):
    original = img.copy()
    result = add(img, value)

    np.testing.assert_allclose(result, expected_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(img, original)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


@pytest.mark.parametrize("img_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        np.array([1.4]),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1], np.float32),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1, 2.0], np.float32),
    ],
)
@pytest.mark.parametrize("is_contiguous", [True, False])
def test_add(img_dtype, num_channels, value, is_contiguous):
    np.random.seed(0)

    height, width = 9, 11

    if is_contiguous:
        if img_dtype == np.uint8:
            img = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype)
        else:
            img = np.random.rand(height, width, num_channels).astype(img_dtype)
    elif img_dtype == np.uint8:
        img = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0)
    else:
        img = np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)

    original_image = img.copy()

    processed_value = convert_value(value, num_channels)

    result = add(img, value)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    control = np.asarray(processed_value, dtype=np.float32)
    if img_dtype == np.uint8:
        control = np.trunc(control)
    expected = np.add(img.astype(np.float32), control)
    if img_dtype == np.uint8:
        expected = np.clip(expected, 0, 255).astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(img, original_image)
