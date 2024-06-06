import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from albucore.functions import (
    normalize_per_image_opencv,
    normalize_per_image_numpy,
    normalize_per_image_lut,
    normalize_per_image,
)
from albucore.utils import MAX_VALUES_BY_DTYPE, get_num_channels, convert_value


@pytest.mark.parametrize("img, normalization, expected", [
    (np.array([[1, 2], [3, 4]]), "min_max", np.array([[0, 1/3], [2/3, 1]], dtype=np.float32)),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "min_max_per_channel", np.array([[[0, 0], [1/3, 1/3]], [[2/3, 2/3], [1, 1]]], dtype=np.float32)),
    (np.array([[1, 2], [3, 4]]), "image", np.array([[-1.34164079, -0.4472136 ], [ 0.4472136 ,  1.34164079]], dtype=np.float32)),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "image_per_channel", np.array([[[-1.34164079, -1.34164079],
        [-0.4472136 , -0.4472136 ]],
       [[ 0.4472136 ,  0.4472136 ],
        [ 1.34164079,  1.34164079]]], dtype=np.float32))
])
@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_normalize_per_image(img, normalization, expected, dtype):
    img = img.astype(dtype)
    result = normalize_per_image(img, normalization)
    result_np = normalize_per_image_numpy(img, normalization)
    result_cv2 = normalize_per_image_opencv(img, normalization)

    assert np.allclose(result_np, expected, atol=1.5e-4), f"Result - expected {(result_np - expected).max()}"
    assert np.allclose(result_cv2, expected, atol=1.5e-4), f"Result - expected {(result_cv2 - expected).max()}"
    assert np.allclose(result, expected, atol=1.5e-4), f"Result - expected {(result - expected).max()}"

    if img.dtype == np.uint8:
        result_lut = normalize_per_image_lut(img, normalization)
        assert np.allclose(result_lut, expected, atol=1.5e-4), f"Result - expected {(result_lut - expected).max()}"


@pytest.mark.parametrize(
    ["image", "normalization"],
    [
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), "image"],
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), "image_per_channel"],
        [np.random.randint(0, 256, [101, 99, 1], dtype=np.uint8), "min_max"],
        [np.random.randint(0, 256, [101, 99], dtype=np.uint8), "min_max_per_channel"],
    ],
)
def test_normalize_np_cv_equal(image, normalization):
    res1 = normalize_per_image_numpy(image, normalization)
    res2 = normalize_per_image_opencv(image, normalization)
    res3 = normalize_per_image_lut(image, normalization)

    assert np.array_equal(image.shape, res1.shape)
    assert np.array_equal(image.shape, res2.shape)
    assert np.array_equal(image.shape, res3.shape)

    assert np.allclose(res1, res2, atol=1e-7), f"mean: {(res1 - res2).mean()}, max: {(res1 - res2).max()}"
    assert np.allclose(res1, res3, atol=1e-6), f"mean: {(res1 - res3).mean()}, max: {(res1 - res3).max()}"
