"""Tests for median_blur."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from albucore.functions import median_blur

CHANNEL_COUNTS = [1, 3, 4, 5, 8]
DTYPES = [np.uint8, np.float32]
RNG = np.random.default_rng(42)


def _make_img(h: int, w: int, c: int, dtype: type) -> np.ndarray:
    if dtype == np.uint8:
        return RNG.integers(0, 256, (h, w, c), dtype=np.uint8)
    return RNG.random((h, w, c), dtype=np.float32)


@pytest.mark.parametrize("channels", CHANNEL_COUNTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ksize", [3, 5])
def test_median_blur_direct_path(channels: int, dtype: type, ksize: int) -> None:
    """k=3,5 work for any channels (direct path)."""
    img = _make_img(32, 32, channels, dtype)
    result = median_blur(img, ksize)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ksize", [7, 9, 11])
def test_median_blur_direct_path_large_kernel(channels: int, dtype: type, ksize: int) -> None:
    """k>=7 with 1-4ch: direct path. float32 uses uint8_io."""
    img = _make_img(32, 32, channels, dtype)
    result = median_blur(img, ksize)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    if dtype == np.uint8:
        expected = cv2.medianBlur(img, ksize)
        if channels == 1 and expected.ndim == 2:
            expected = np.expand_dims(expected, -1)
        np.testing.assert_array_equal(result, expected)
    else:
        expected = cv2.medianBlur(
            (img * 255).astype(np.uint8), ksize
        ).astype(np.float32) / 255
        if channels == 1 and expected.ndim == 2:
            expected = np.expand_dims(expected, -1)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1 / 255)


@pytest.mark.parametrize("channels", [5, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ksize", [7, 9])
def test_median_blur_chunked_path(channels: int, dtype: type, ksize: int) -> None:
    """k>=7 with 5+ch: chunked path. float32 uses uint8_io."""
    img = _make_img(32, 32, channels, dtype)
    result = median_blur(img, ksize)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    if dtype == np.uint8:
        expected_chunks = []
        for i in range(channels):
            ch = img[:, :, i : i + 1]
            out = cv2.medianBlur(ch, ksize)
            if out.ndim == 2:
                out = np.expand_dims(out, -1)
            expected_chunks.append(out)
        expected = np.concatenate(expected_chunks, axis=-1)
        np.testing.assert_array_equal(result, expected)
    else:
        expected_chunks = []
        for i in range(channels):
            ch_uint8 = (img[:, :, i : i + 1] * 255).astype(np.uint8)
            out = cv2.medianBlur(ch_uint8, ksize)
            if out.ndim == 2:
                out = np.expand_dims(out, -1)
            expected_chunks.append(out.astype(np.float32) / 255)
        expected = np.concatenate(expected_chunks, axis=-1)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1 / 255)


@pytest.mark.parametrize("channels", [1, 3, 8])
@pytest.mark.parametrize("dtype", DTYPES)
def test_median_blur_matches_cv2_for_supported(channels: int, dtype: type) -> None:
    """median_blur matches cv2.medianBlur for k=3,5. float32 via uint8_io (quantization)."""
    img = _make_img(32, 32, channels, dtype)
    for ksize in [3, 5]:
        result = median_blur(img, ksize)
        if dtype == np.uint8:
            expected = cv2.medianBlur(img, ksize)
        else:
            expected = cv2.medianBlur(
                (img * 255).astype(np.uint8), ksize
            ).astype(np.float32) / 255
        if channels == 1 and expected.ndim == 2:
            expected = np.expand_dims(expected, -1)
        if dtype == np.uint8:
            np.testing.assert_array_equal(result, expected)
        else:
            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1 / 255)


def test_median_blur_invalid_ksize() -> None:
    """Reject even or invalid ksize."""
    img = _make_img(16, 16, 3, np.uint8)
    with pytest.raises(ValueError, match="ksize must be odd"):
        median_blur(img, 4)
    with pytest.raises(ValueError, match="ksize must be odd"):
        median_blur(img, 2)
    with pytest.raises(ValueError, match="ksize must be odd"):
        median_blur(img, 1)
