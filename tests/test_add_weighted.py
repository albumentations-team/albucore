import pytest
import numpy as np
from albucore.functions import add, add_numpy, add_opencv, add_weighted, add_weighted_numpy, add_weighted_opencv, add_weighted_lut, add_weighted_simsimd
from albucore.utils import clip

@pytest.mark.parametrize(
    "img1, weight1, img2, weight2, expected_output",
    [
        # Test case 1: Both weights are 1, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            1.0,
            np.array([[6, 8], [10, 12]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            -1.0,
            np.zeros((2, 2)),
        ),
        # Test case 2: Different weights, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.5,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            0.5,
            np.array([[3, 4], [5, 6]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.5,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            -0.5,
            np.zeros((2, 2)),
        ),
        # Test case 3: Zero weights, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            0.0,
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
        ),
        # Test case 4: Weight 1 and weight 0, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            0.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        # Test case 5: Both weights are 1, image of type float32
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            1.0,
            np.array([[0.6, 0.8], [1.0, 1.0]], dtype=np.float32),
        ),
        # Test case 6: Different weights, image of type float32
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            0.5,
            np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            0.5,
            np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        ),
        (
            np.array([[1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            np.array([[0.5, 0.6], [0.7, 0.2]], dtype=np.float32),
            -0.5,
            np.array([[0.75, 0], [0, 0.3]], dtype=np.float32),
        ),
    ],
)
def test_add_weighted_numpy(img1, weight1, img2, weight2, expected_output):
    result_numpy = clip(add_weighted_numpy(img1, weight1, img2, weight2), img1.dtype)
    np.testing.assert_array_equal(result_numpy, expected_output)

    result_opencv = clip(add_weighted_opencv(img1, weight1, img2, weight2), img1.dtype)
    np.testing.assert_array_equal(result_opencv, expected_output)

    result_simsimd = clip(add_weighted_simsimd(img1, weight1, img2, weight2), img1.dtype)
    np.testing.assert_array_equal(result_simsimd, expected_output)

    if img1.dtype == np.uint8 and img2.dtype == np.uint8:
        result_lut = add_weighted_lut(img1, weight1, img2, weight2)
        np.testing.assert_array_equal(result_lut, expected_output)


@pytest.mark.parametrize(
    "img_dtype", [np.uint8, np.float32]
)
@pytest.mark.parametrize(
    "num_channels", [1, 3, 5]
)
@pytest.mark.parametrize(
    "weight1, weight2",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 0.5),
        (2, -0.5),
    ]
)
@pytest.mark.parametrize(
    "is_contiguous", [True, False]
)
def test_add_weighted(img_dtype, num_channels, weight1, weight2, is_contiguous):
    np.random.seed(0)
    height, width = 9, 11

    if is_contiguous:
        img1 = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype) if img_dtype == np.uint8 else np.random.rand(height, width, num_channels).astype(img_dtype)
        img2 = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype) if img_dtype == np.uint8 else np.random.rand(height, width, num_channels).astype(img_dtype)
    else:
        img1 = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0) if img_dtype == np.uint8 else np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)
        img2 = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0) if img_dtype == np.uint8 else np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)

    original_img1 = img1.copy()
    original_img2 = img2.copy()

    result = add_weighted(img1, weight1, img2, weight2)

    result_simsimd = clip(add_weighted_simsimd(img1, weight1, img2, weight2), img_dtype)

    if img_dtype == np.uint8:
        np.testing.assert_allclose(result, result_simsimd, atol=1)
    else:
        np.testing.assert_allclose(result, result_simsimd, atol=1e-6)

    result_numpy = clip(add_weighted_numpy(img1, weight1, img2, weight2), img_dtype)
    np.testing.assert_allclose(result, result_numpy, atol=1)

    result_opencv = clip(add_weighted_opencv(img1, weight1, img2, weight2), img1.dtype)
    np.testing.assert_allclose(result, result_opencv, atol=1)

    if img1.dtype == np.uint8 and img2.dtype == np.uint8:
        result_lut = clip(add_weighted_lut(img1, weight1, img2, weight2), img1.dtype)
        np.testing.assert_allclose(result, result_lut, atol=1)

    np.testing.assert_array_equal(img1, original_img1)
    np.testing.assert_array_equal(img2, original_img2)


@pytest.mark.parametrize(
    "img_dtype", [np.uint8, np.float32]
)
@pytest.mark.parametrize(
    "num_channels", [1, 3, 5]
)
@pytest.mark.parametrize(
    "is_contiguous", [True, False]
)
def test_add_weighted_vs_add(img_dtype, num_channels, is_contiguous):
    np.random.seed(0)
    height, width = 9, 11

    if is_contiguous:
        img1 = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype) if img_dtype == np.uint8 else np.random.rand(height, width, num_channels).astype(img_dtype)
        img2 = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype) if img_dtype == np.uint8 else np.random.rand(height, width, num_channels).astype(img_dtype)
    else:
        img1 = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0) if img_dtype == np.uint8 else np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)
        img2 = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0) if img_dtype == np.uint8 else np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)

    original_img1 = img1.copy()
    original_img2 = img2.copy()

    result = add_weighted(img1, 1, img2, 1)

    assert result.dtype == img1.dtype
    assert result.shape == img1.shape

    result_add = add(img1, img2)

    if img_dtype == np.uint8:
        np.testing.assert_array_equal(result, result_add)
    else:
        np.testing.assert_allclose(result, result_add, atol=1e-6)

    result_numpy = clip(add_weighted_numpy(img1, 1, img2, 1), img_dtype)
    result_numpy_add = clip(add_numpy(img1, img2), img_dtype)

    np.testing.assert_array_equal(result, result_numpy)
    np.testing.assert_array_equal(result_numpy, result_numpy_add)

    result_opencv = clip(add_weighted_opencv(img1, 1, img2, 1), img1.dtype)
    result_opencv_add = clip(add_opencv(img1, img2), img1.dtype)

    np.testing.assert_array_equal(result_opencv, result_opencv_add)
    np.testing.assert_array_equal(result, result_opencv)

    if img1.dtype == np.uint8 and img2.dtype == np.uint8:
        result_lut = clip(add_weighted_lut(img1, 1, img2, 1), img1.dtype)

        np.testing.assert_allclose(result, result_lut, atol=1)

    np.testing.assert_array_equal(img1, original_img1)
    np.testing.assert_array_equal(img2, original_img2)
