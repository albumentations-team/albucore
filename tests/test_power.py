import pytest
import numpy as np
from albucore.functions import power, power_numpy, power_opencv, power_lut
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, convert_value, clip


@pytest.mark.parametrize(
    "img, exponent, expected_output",
    [
        # Test case 1: Exponent as a float, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            2.0,
            np.array([[[1, 4, 9], [16, 25, 36]], [[49, 64, 81], [100, 121, 144]]], dtype=np.uint8),
        ),
        # Test case 2: Exponent as a vector, image of type uint8
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
            np.array([[[1, 8, 81], [16, 125, 255]], [[49, 255, 255], [100, 255, 255]]], dtype=np.uint8),
        ),
        # Clipping effect test for uint8
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            2.0,
            np.array([[[255, 255, 255], [255, 255, 255]], [[255, 255, 255], [255, 255, 255]]], dtype=np.uint8),
        ),
        # Test case 3: Exponent as a float, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            2.0,
            np.array([[[0.01, 0.04, 0.09], [0.16, 0.25, 0.36]], [[0.49, 0.64, 0.81], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
        # Test case 4: Exponent as a vector, image of type float32
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[0.31622776, 0.08944272, 0.09      ], [0.6324555 , 0.35355338, 0.36      ]], [[0.83666   , 0.7155418 , 0.81], [1.0, 1.0, 1.0]]], dtype=np.float32),
        ),
    ],
)
def test_power_with_numpy(img, exponent, expected_output):
    result_numpy = clip(power_numpy(img, exponent), img.dtype)
    assert np.allclose(result_numpy, expected_output, atol=1e-6)

    assert result_numpy.dtype == img.dtype, "Input image was modified"
    assert result_numpy.shape == img.shape

    if isinstance(exponent, (float, int)):
        result_opencv = clip(power_opencv(img, exponent), img.dtype)
        assert np.allclose(result_opencv, expected_output, atol=1e-6)

    if img.dtype == np.uint8:
        result_lut = clip(power_lut(img, exponent), img.dtype)
        assert np.allclose(result_lut, expected_output, atol=1e-6)


@pytest.mark.parametrize(
    "img_dtype", [np.uint8, np.float32]
)
@pytest.mark.parametrize(
    "num_channels", [1, 3, 5]
)
@pytest.mark.parametrize(
    "exponent",
    [
        1.6,
        np.array([1.5]),
        np.array((1.6)),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1], np.float32),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1, 2.0], np.float32),
    ]
)
@pytest.mark.parametrize(
    "is_contiguous", [True, False]
)
def test_power(img_dtype, num_channels, exponent, is_contiguous):
    height, width = 9, 11
    np.random.seed(42)

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

    processed_exponent = convert_value(exponent, num_channels)

    result = power(img, exponent)
    assert result.shape == original_image.shape
    assert result.dtype == original_image.dtype

    result_numpy = clip(power_numpy(img, processed_exponent), img_dtype)

    assert np.array_equal(img, original_image), "Input image was modified"

    assert np.allclose(result, result_numpy, atol=1e-6)

    if img.dtype == np.uint8:
        # result_lut = clip(power_lut(img, processed_exponent), img.dtype)
        result_lut = power_lut(img, processed_exponent)

        assert np.allclose(result, result_lut, atol=1e-6), f"Difference {(result - result_lut).mean()}"

    if isinstance(exponent, (float, int)):
        result_opencv = clip(power_opencv(img, processed_exponent), img.dtype)
        assert np.allclose(result, result_opencv, atol=1e-6), f"Difference {(result - result_opencv).max()} {(result - result_opencv).mean()}"

    assert np.array_equal(img, original_image), "Input image was modified"
