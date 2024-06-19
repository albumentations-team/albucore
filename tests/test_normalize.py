import pytest
import numpy as np

from albucore.functions import normalize, normalize, normalize, normalize_numpy, normalize_opencv, normalize_lut
from albucore.utils import MAX_VALUES_BY_DTYPE, convert_value, get_num_channels
from numpy.testing import assert_array_almost_equal_nulp


@pytest.mark.parametrize("img, factor, shift, expected", [
    (np.array([[1, 2], [3, 4]], dtype=np.float32), 2.0, 1.0, np.array([[4, 6], [8, 10]], dtype=np.float32)),
    (np.array([[0, 1], [2, 3]], dtype=np.float32), np.array([[2, 3], [4, 5]], dtype=np.float32), 1.0, np.array([[2, 6], [12, 20]], dtype=np.float32)),
    (np.array([[1, 2], [3, 4]], dtype=np.float32), 2.0, np.array([[1, 0], [0, 1]], dtype=np.float32), np.array([[4, 4], [6, 10]], dtype=np.float32))
])
def test_normalize(img, factor, shift, expected):
    result = normalize(img, factor, shift)
    result_np = normalize_numpy(img, factor, shift)
    result_cv2 = normalize_opencv(img, factor, shift)
    np.array_equal(result, expected)
    np.array_equal(result_np, expected)
    np.array_equal(result_cv2, expected)


@pytest.mark.parametrize("img, denominator, mean, expected", [
    (np.array([[1, 2], [3, 4]], dtype=np.uint8), 2.0, 1.0, np.array([[0, 2], [4, 6]])),
    (np.array([[1, 2], [3, 4]], dtype=np.uint8), 2.0, np.array([0.5, -1]), np.array([[1, 3], [5, 7]])), # Pick only first element for grayscale image
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8), 2.0, 1.0, np.array([[[0, 2], [4, 6]], [[8, 10], [12, 14]]])),
    (np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8), np.array([2.0, 3.0]), 1.0, np.array([[[-2, 0], [2, 6]], [[6, 12], [10, 18]]])),
])
def test_normalize_lut(img, denominator, mean, expected):
    num_channels = get_num_channels(img)

    converted_denominator = convert_value(denominator, num_channels)
    converted_mean = convert_value(mean, num_channels)

    result = normalize_lut(img, converted_mean, denominator)
    result_lut = normalize_lut(img, converted_mean, converted_denominator)
    result_np = normalize_numpy(img, mean, denominator)
    result_cv2 = normalize_opencv(img, mean, denominator)

    np.array_equal(result, expected)
    np.array_equal(result_lut, expected)
    np.array_equal(result_np, expected)
    np.array_equal(result_cv2, expected)


@pytest.mark.parametrize(
    ["image", "mean", "std"],
    [
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), 0.5, 0.7],
        [np.random.randint(0, 256, [101, 99, 1], dtype=np.uint8), 0.5, 0.7],
        [np.random.randint(0, 256, [101, 99], dtype=np.uint8), 0.5, 0.5],
    ],
)
def test_normalize_np_cv_equal(image, mean, std):
    mean = np.array(mean, dtype=np.float32)
    max_pixel_value = MAX_VALUES_BY_DTYPE[image.dtype]

    denominator = np.reciprocal(np.array(std, dtype=np.float64) * max_pixel_value)

    num_channels = get_num_channels(image)

    converted_denominator = convert_value(denominator, num_channels)
    converted_mean = convert_value(mean, num_channels)

    res3 = normalize_lut(image, converted_mean, converted_denominator)

    res1 = normalize_numpy(image, mean, denominator)
    res2 = normalize_opencv(image, mean, converted_denominator)

    assert np.array_equal(image.shape, res1.shape)
    assert np.array_equal(image.shape, res2.shape)
    assert np.array_equal(image.shape, res3.shape)

    assert np.allclose(res1, res2, atol=1e-7), f"mean: {(res1 - res2).mean()}, max: {(res1 - res2).max()}"
    assert np.allclose(res1, res3, atol=1e-6), f"mean: {(res1 - res3).mean()}, max: {(res1 - res3).max()}"


@pytest.mark.parametrize("dtype", [
    np.uint8,
    np.float32,
])
@pytest.mark.parametrize("shape", [(99, 101, 3), (99, 101, 1), (99, 101)])
def test_normalize(dtype, shape) -> None:
    img = np.ones(shape, dtype=dtype) * 0.4
    mean = np.array(50, dtype=np.float32)
    denominator = np.array(1 / 3, dtype=np.float32)
    normalized = normalize(img, mean=mean, denominator=denominator)

    assert normalized.shape == img.shape
    assert normalized.dtype == np.float32

    expected = (np.ones(img.shape, dtype=np.float32) * 0.4 - 50) / 3
    assert_array_almost_equal_nulp(normalized, expected)
