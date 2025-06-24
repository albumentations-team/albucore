import pytest
import numpy as np

from albucore.functions import normalize, normalize_numpy, normalize_opencv, normalize_lut
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
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result_np, expected)
    np.testing.assert_array_equal(result_cv2, expected)


@pytest.mark.parametrize("img, denominator, mean, expected", [
    (np.array([[1, 2], [3, 4]], dtype=np.uint8), 2.0, 1.0, np.array([[0, 2], [4, 6]])),
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

    np.testing.assert_allclose(result, expected, atol=1e-6)
    np.testing.assert_allclose(result_lut, expected, atol=1e-6)
    np.testing.assert_allclose(result_np, expected, atol=1e-6)
    np.testing.assert_allclose(result_cv2, expected, atol=1e-6)


@pytest.mark.parametrize(
    ["image", "mean", "std"],
    [
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), 0.5, 0.7],
        [np.random.randint(0, 256, [101, 99, 1], dtype=np.uint8), 0.5, 0.7],
        [np.random.randint(0, 256, [101, 99, 1], dtype=np.uint8), 0.5, 0.5],
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

    np.testing.assert_array_equal(image.shape, res1.shape)
    np.testing.assert_array_equal(image.shape, res2.shape)
    np.testing.assert_array_equal(image.shape, res3.shape)

    np.testing.assert_allclose(res1, res2, atol=1e-6)
    np.testing.assert_allclose(res1, res3, atol=1e-6)


@pytest.mark.parametrize("dtype", [
    np.uint8,
    np.float32,
])
@pytest.mark.parametrize("shape", [(99, 101, 3), (99, 101, 1)])
def test_normalize(dtype, shape) -> None:
    img = np.ones(shape, dtype=dtype) * 0.4
    mean = np.array(50, dtype=np.float32)
    denominator = np.array(1 / 3, dtype=np.float32)

    volume = np.stack([img.copy()] * 4, axis=0)  # (4, H, W) or (4, H, W, C)
    images = np.stack([img.copy()] * 3, axis=0)  # (3, H, W) or (3, H, W, C)
    volumes = np.stack([volume.copy()] * 2, axis=0)

    normalized_image = normalize(img, mean=mean, denominator=denominator)
    assert normalized_image.shape == img.shape
    assert normalized_image.dtype == np.float32

    expected_image = (np.ones(img.shape, dtype=np.float32) * 0.4 - 50) / 3
    np.testing.assert_array_almost_equal_nulp(normalized_image, expected_image)

    normalized_images = normalize(images, mean=mean, denominator=denominator)
    assert normalized_images.shape == images.shape
    assert normalized_images.dtype == np.float32

    normalized_volume = normalize(volume, mean=mean, denominator=denominator)
    assert normalized_volume.shape == volume.shape
    assert normalized_volume.dtype == np.float32

    normalized_volumes = normalize(volumes, mean=mean, denominator=denominator)
    assert normalized_volumes.shape == volumes.shape
    assert normalized_volumes.dtype == np.float32

    np.testing.assert_allclose(normalized_image[0], normalized_image[1], atol=4, rtol=1e-5)
    np.testing.assert_allclose(normalized_images[0], normalized_images[1], atol=4, rtol=1e-5)
    np.testing.assert_allclose(normalized_volume[0], normalized_volume[1], atol=4, rtol=1e-5)
    np.testing.assert_allclose(normalized_volumes[0], normalized_volumes[1], atol=4, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalize_with_1d_arrays(dtype):
    """Test normalize function with 1D array mean and denominator for single channel images."""
    # Test 1: Single channel image (H, W, 1) with 1D arrays
    img = np.ones((50, 50, 1), dtype=dtype) * 100
    mean = np.array([50.0], dtype=np.float32)  # 1D array with single element
    denominator = np.array([2.0], dtype=np.float32)  # 1D array with single element

    # Calculate expected BEFORE calling normalize to avoid in-place modification issues
    expected = (img.astype(np.float32) - 50.0) * 2.0
    result = normalize(img, mean, denominator)

    assert result.shape == img.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    # Test 2: Batch of single channel images (N, H, W, 1) with 1D arrays
    batch = np.ones((3, 50, 50, 1), dtype=dtype) * 100  # Create fresh batch
    expected_batch = (batch.astype(np.float32) - 50.0) * 2.0
    result_batch = normalize(batch, mean, denominator)

    assert result_batch.shape == batch.shape
    assert result_batch.dtype == np.float32
    # All images should be normalized identically
    np.testing.assert_allclose(result_batch[0], result_batch[1], rtol=1e-5)
    np.testing.assert_allclose(result_batch[0], result_batch[2], rtol=1e-5)
    np.testing.assert_allclose(result_batch, expected_batch, rtol=1e-5)

    # Test 3: Multi-channel image (H, W, C) with matching 1D arrays
    img_3ch = np.ones((50, 50, 3), dtype=dtype) * np.array([100, 150, 200])
    mean_3ch = np.array([50.0, 75.0, 100.0], dtype=np.float32)
    denominator_3ch = np.array([2.0, 3.0, 4.0], dtype=np.float32)

    expected_3ch = np.zeros_like(img_3ch, dtype=np.float32)
    for c in range(3):
        expected_3ch[..., c] = (img_3ch[..., c].astype(np.float32) - mean_3ch[c]) * denominator_3ch[c]

    result_3ch = normalize(img_3ch, mean_3ch, denominator_3ch)

    assert result_3ch.shape == img_3ch.shape
    assert result_3ch.dtype == np.float32
    np.testing.assert_allclose(result_3ch, expected_3ch, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape", [(100, 100, 1), (100, 100, 3)])
def test_normalize_preserves_original_image(dtype, shape):
    """Test that normalize functions don't modify the original image."""
    # Create test image
    if dtype == np.uint8:
        original_img = np.random.randint(0, 256, size=shape, dtype=dtype)
    else:
        original_img = np.random.randn(*shape).astype(dtype)

    # Make a copy to compare later
    img_copy = original_img.copy()

    # Define normalization parameters
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) if shape[-1] == 3 else np.array(0.5, dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) if shape[-1] == 3 else np.array(0.5, dtype=np.float32)

    # Convert std to denominator
    max_pixel_value = MAX_VALUES_BY_DTYPE[dtype] if dtype == np.uint8 else 1.0
    denominator = np.reciprocal(std * max_pixel_value)

    # Test main normalize function
    _ = normalize(original_img, mean, denominator)
    np.testing.assert_array_equal(original_img, img_copy,
                                  err_msg="normalize() modified the original image")

    # Test normalize_numpy
    original_img = img_copy.copy()
    _ = normalize_numpy(original_img, mean, denominator)
    np.testing.assert_array_equal(original_img, img_copy,
                                  err_msg="normalize_numpy() modified the original image")

    # Test normalize_opencv
    original_img = img_copy.copy()
    _ = normalize_opencv(original_img, mean, denominator)
    np.testing.assert_array_equal(original_img, img_copy,
                                  err_msg="normalize_opencv() modified the original image")

    # Test normalize_lut for uint8 only
    if dtype == np.uint8:
        original_img = img_copy.copy()
        num_channels = get_num_channels(original_img)
        converted_denominator = convert_value(denominator, num_channels)
        converted_mean = convert_value(mean, num_channels)
        _ = normalize_lut(original_img, converted_mean, converted_denominator)
        np.testing.assert_array_equal(original_img, img_copy,
                                      err_msg="normalize_lut() modified the original image")
