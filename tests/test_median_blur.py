"""Tests for the public median_blur router."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from albucore import from_float, median_blur, to_float


def _make_image(channels: int, dtype: type[np.uint8 | np.float32]) -> np.ndarray:
    rng = np.random.default_rng(137)
    if dtype == np.uint8:
        return rng.integers(0, 256, (31, 37, channels), dtype=np.uint8)
    return rng.random((31, 37, channels), dtype=np.float32)


def _repair_channel_dim(result: np.ndarray, channels: int) -> np.ndarray:
    return result[..., np.newaxis] if channels == 1 and result.ndim == 2 else result


def _uint8_reference(image: np.ndarray, kernel_size: int) -> np.ndarray:
    channels = image.shape[-1]
    if kernel_size in (3, 5) or channels <= 4:
        return _repair_channel_dim(cv2.medianBlur(image, kernel_size), channels)
    return np.concatenate(
        [
            _repair_channel_dim(cv2.medianBlur(image[..., index : index + 1], kernel_size), 1)
            for index in range(channels)
        ],
        axis=-1,
    )


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_median_blur_uint8_matches_opencv(channels: int, kernel_size: int) -> None:
    image = _make_image(channels, np.uint8)

    result = median_blur(image, kernel_size)

    np.testing.assert_array_equal(result, _uint8_reference(image, kernel_size))
    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_median_blur_float32_native_matches_opencv_exactly(channels: int, kernel_size: int) -> None:
    image = _make_image(channels, np.float32)

    result = median_blur(image, kernel_size)
    expected = _repair_channel_dim(cv2.medianBlur(image, kernel_size), channels)

    np.testing.assert_array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == np.float32


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_median_blur_float32_native_preserves_sub_uint8_precision(kernel_size: int) -> None:
    image = np.linspace(0.1001, 0.1099, 9 * 11, dtype=np.float32).reshape(9, 11, 1)

    result = median_blur(image, kernel_size)
    native = _repair_channel_dim(cv2.medianBlur(image, kernel_size), 1)
    legacy = to_float(_uint8_reference(from_float(image, np.dtype(np.uint8)), kernel_size))

    np.testing.assert_array_equal(result, native)
    assert not np.array_equal(result, legacy)


@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("kernel_size", [7, 9])
def test_median_blur_float32_large_kernel_matches_quantized_fallback(channels: int, kernel_size: int) -> None:
    image = _make_image(channels, np.float32)
    quantized = from_float(image, np.dtype(np.uint8))
    expected = to_float(_uint8_reference(quantized, kernel_size))

    result = median_blur(image, kernel_size)

    np.testing.assert_array_equal(result, expected)
    assert result.shape == image.shape
    assert result.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_median_blur_accepts_non_contiguous_read_only_input(
    dtype: type[np.uint8 | np.float32],
    kernel_size: int,
) -> None:
    source = _make_image(5, dtype)
    image = source[:, ::2, :]
    image.setflags(write=False)
    before = image.copy()

    result = median_blur(image, kernel_size)

    np.testing.assert_array_equal(image, before)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_median_blur_does_not_mutate_input(
    dtype: type[np.uint8 | np.float32],
    kernel_size: int,
) -> None:
    image = _make_image(3, dtype)
    before = image.copy()

    result = median_blur(image, kernel_size)

    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(result, image)
