from collections.abc import Callable

import cv2
import numpy as np
import pytest

from albucore.utils import (
    clip,
    convert_value,
    get_num_channels,
    get_opencv_max_channels,
    is_grayscale_image,
    is_multispectral_image,
    is_rgb_image,
)


@pytest.mark.parametrize(
    "input_img, dtype, expected",
    [
        (
            np.array([[-300, 0], [100, 400]], dtype=np.float32),
            np.uint8,
            np.array([[0, 0], [100, 255]], dtype=np.float32),
        ),
        (
            np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32),
            np.float32,
            np.array([[0, 0], [0.5, 1.0]], dtype=np.float32),
        ),
    ],
)
@pytest.mark.parametrize("provided_output", [False, True])
def test_clip(input_img, dtype, expected, provided_output):
    original = input_img.copy()
    destination = np.empty_like(input_img, dtype=dtype) if provided_output else None
    clipped = clip(input_img, dtype=dtype, out=destination)
    np.testing.assert_array_equal(clipped, expected)
    assert clipped.dtype == dtype
    np.testing.assert_array_equal(input_img, original)
    if destination is not None:
        assert clipped is destination


def test_get_opencv_max_channels_matches_matrix_type_encoding():
    max_channels = get_opencv_max_channels()

    assert cv2.CV_MAKETYPE(cv2.CV_8U, max_channels) <= cv2.MAT_TYPE_MASK
    assert cv2.CV_MAKETYPE(cv2.CV_8U, max_channels + 1) > cv2.MAT_TYPE_MASK


@pytest.mark.parametrize(
    "value, num_channels, expected",
    [
        ((1.5), 1, 1.5),
        (np.array([1.5]), 3, 1.5),
        (np.array([1.5, 2.5]), 1, 1.5),
        (np.array([1.5, 2.5, 0.5]), 2, np.array([1.5, 2.5], dtype=np.float32)),
        (3, 2, 3),
        (np.array(1.5), 2, 1.5),
        (np.reciprocal(np.array(0.2, dtype=np.float64)), 2, 5),
    ],
)
def test_convert_value(value, num_channels, expected):
    result = convert_value(value, num_channels)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "shape, expected_channels, description",
    [
        # 3D image with channels (H, W, C)
        ((100, 99, 1), 1, "3D image with 1 channel"),
        ((100, 99, 3), 3, "3D RGB image"),
        ((100, 99, 7), 7, "3D multi-channel image"),
        # 4D batch of images or volume (N/D, H, W, C)
        ((4, 100, 99, 1), 1, "Batch of grayscale images or volume"),
        ((4, 100, 99, 3), 3, "Batch of RGB images or volume"),
        ((10, 224, 224, 3), 3, "Batch of RGB images or volume"),
        # Edge cases
        ((1, 1, 1), 1, "Minimal 3D array"),
    ],
)
def test_get_num_channels(shape, expected_channels, description):
    image = np.zeros(shape)
    assert get_num_channels(image) == expected_channels, f"Failed for {description} with shape {shape}"


@pytest.mark.parametrize(
    "shape, expected_grayscale, description",
    [
        # HWC: shape=(100, 200, C)
        ((100, 200, 3), False, "HWC: RGB image"),
        ((100, 200, 1), True, "HWC: single channel image"),
        ((100, 200, 4), False, "HWC: RGBA image"),
        # XHWC: shape=(X, 100, 200, C) where X is batch or depth
        ((10, 100, 200, 3), False, "XHWC: batch/volume of RGB"),
        ((10, 100, 200, 1), True, "XHWC: batch/volume of single channel"),
    ],
)
def test_is_grayscale_image(shape, expected_grayscale, description):
    image = np.zeros(shape)
    result = is_grayscale_image(image)
    assert result == expected_grayscale, f"Failed for {description} with shape {shape}"


@pytest.mark.parametrize(
    ("shape", "expected_rgb", "expected_multispectral"),
    [
        ((100, 200, 1), False, False),
        ((100, 200, 3), True, False),
        ((100, 200, 4), False, True),
        ((100, 200, 5), False, True),
        ((10, 100, 200, 1), False, False),
        ((10, 100, 200, 3), True, False),
        ((10, 100, 200, 4), False, True),
        ((10, 100, 200, 5), False, True),
    ],
)
def test_image_channel_predicates(
    shape: tuple[int, ...], expected_rgb: bool, expected_multispectral: bool
) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    assert is_rgb_image(image) is expected_rgb
    assert is_multispectral_image(image) is expected_multispectral


@pytest.mark.parametrize(
    "shape",
    [
        # Basic HWC cases
        (100, 200, 1),
        (100, 200, 3),
        # XHWC cases (batch or depth)
        (10, 100, 200, 1),
        (10, 100, 200, 3),
    ],
)
def test_get_num_channels_and_is_grayscale_consistency(shape):
    image = np.zeros(shape)
    num_channels = get_num_channels(image)
    is_grayscale = is_grayscale_image(image)

    assert (num_channels == 1) == is_grayscale, (
        f"Inconsistency for shape {shape}: num_channels={num_channels}, is_grayscale={is_grayscale}"
    )
