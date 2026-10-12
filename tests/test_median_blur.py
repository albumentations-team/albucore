"""Tests for the public median_blur router."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

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
    quantized_reference = to_float(_uint8_reference(from_float(image, np.dtype(np.uint8)), kernel_size))

    np.testing.assert_array_equal(result, native)
    assert not np.array_equal(result, quantized_reference)


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


@pytest.mark.parametrize("layout", ["numpy", "CHW", "NCHW"])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
def test_median_blur_float16_matches_float32_route(
    layout: str,
    channels: int,
    kernel_size: int,
) -> None:
    batch_size = 2 if layout == "NCHW" else 1
    source = (
        np.random.default_rng(610111 + channels + kernel_size)
        .random((batch_size, 19, 23, channels), dtype=np.float32)
        .astype(np.float16)
    )
    expected = np.stack([median_blur(image.astype(np.float32), kernel_size).astype(np.float16) for image in source])
    before = source.copy()

    if layout == "numpy":
        result = median_blur(source[0], kernel_size)
        actual = result[None]
    elif layout == "CHW":
        tensor = torch.from_numpy(source[0]).permute(2, 0, 1)
        result = median_blur(tensor, kernel_size, layout="CHW")
        actual = result.permute(1, 2, 0).numpy()[None]
        assert result.data_ptr() != tensor.data_ptr()
    else:
        tensor = torch.from_numpy(source).permute(0, 3, 1, 2)
        result = median_blur(tensor, kernel_size, layout="NCHW")
        actual = result.permute(0, 2, 3, 1).numpy()
        assert result.data_ptr() != tensor.data_ptr()

    assert result.dtype == (np.float16 if layout == "numpy" else torch.float16)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)


def test_median_blur_float16_single_channel_k5_matches_nonzero_non_nan_binary16_values() -> None:
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    non_nan_bits = bits[~((exponent == 0x7C00) & (mantissa != 0))]
    nonzero_bits = non_nan_bits[(non_nan_bits & np.uint16(0x7FFF)) != 0]
    image = np.resize(nonzero_bits, (240, 320, 1)).view(np.float16)
    expected = median_blur(image.astype(np.float32), 5).astype(np.float16)
    before = image.copy()

    result = median_blur(image, 5)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
    np.testing.assert_array_equal(image, before)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.shares_memory(result, image)


def test_median_blur_float16_single_channel_k5_nan_uses_float32_semantics() -> None:
    bits = np.resize(np.array([0x7E00, 0x3C00, 0xBC00, 0x7C00, 0xFC00], dtype=np.uint16), (240, 320, 1))
    image = bits.view(np.float16)
    expected = median_blur(image.astype(np.float32), 5).astype(np.float16)

    result = median_blur(image, 5)

    np.testing.assert_array_equal(result, expected)


def test_median_blur_float16_single_channel_k5_preserves_signed_zero_bits() -> None:
    bits = np.resize(np.array([0x0000, 0x8000], dtype=np.uint16), (240, 320, 1))
    image = bits.view(np.float16)
    expected = median_blur(image.astype(np.float32), 5).astype(np.float16)

    result = median_blur(image, 5)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("zero_bits", [0x0000, 0x8000])
def test_median_blur_float16_single_sign_zero_k5_preserves_bits(zero_bits: int) -> None:
    image = np.full((240, 320, 1), zero_bits, dtype=np.uint16).view(np.float16)
    expected = median_blur(image.astype(np.float32), 5).astype(np.float16)

    result = median_blur(image, 5)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
def test_median_blur_float16_tensor_single_channel_k5_matches_float32_route(layout: str) -> None:
    batch = 2 if layout == "NCHW" else 1
    image = np.random.default_rng(640_011).random((batch, 240, 320, 1), dtype=np.float32).astype(np.float16)
    expected = np.stack([median_blur(item.astype(np.float32), 5).astype(np.float16) for item in image])
    if layout == "CHW":
        tensor = torch.from_numpy(image[0]).permute(2, 0, 1).contiguous()
        result = median_blur(tensor, 5, layout=layout)
        actual = result.permute(1, 2, 0).numpy()[None]
    else:
        tensor = torch.from_numpy(image).permute(0, 3, 1, 2).contiguous()
        result = median_blur(tensor, 5, layout=layout)
        actual = result.permute(0, 2, 3, 1).numpy()

    np.testing.assert_array_equal(actual, expected)
    assert result.dtype == torch.float16
    assert result.data_ptr() != tensor.data_ptr()


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
