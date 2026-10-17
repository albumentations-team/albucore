from typing import Any

import numpy as np
import pytest

from albucore import from_float, to_float
from albucore.utils import clip, convert_value, get_max_value

CHANNELS = [1, 3, 5]
DTYPES = [np.uint8, np.float16, np.float32]
BATCHES = [0, 2]


def generate_img(dtype: Any, channels: int, batch: int) -> np.ndarray:
    shape = (*((batch,) if batch else ()), 13, 17, channels)
    rng = np.random.default_rng(137)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("max_value", [None, 128.0, 255.0])
def test_to_float_reference_and_input_preservation(dtype, channels, batch, max_value):
    image = generate_img(dtype, channels, batch)
    original = image.copy()
    expected = (
        image.astype(np.float32) / np.float32(255 if max_value is None else max_value) if dtype == np.uint8 else image
    )

    result = to_float(image, max_value)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    assert result.dtype == (np.float32 if dtype == np.uint8 else dtype)
    assert result.shape == image.shape
    if dtype != np.uint8:
        assert result is image


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("max_value", [None, 128.0, 255.0])
def test_from_float_reference_and_input_preservation(dtype, channels, batch, max_value):
    image = generate_img(np.float32, channels, batch)
    original = image.copy()
    expected = (
        np.clip(np.rint(image * np.float32(255 if max_value is None else max_value)), 0, 255).astype(np.uint8)
        if dtype == np.uint8
        else image.astype(dtype)
    )

    result = from_float(image, dtype, max_value)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    assert result.shape == image.shape
    assert result.dtype == dtype


def test_float16_to_float_preserves_the_input_and_custom_scale() -> None:
    image = np.array([[[-2.0], [0.5], [4.0]]], dtype=np.float16)
    assert to_float(image, max_value=255) is image


def test_from_float_to_float16_casts_without_integer_rounding_or_clipping() -> None:
    image = np.array([[[-2.0], [0.3333], [4.0]]], dtype=np.float32)
    result = from_float(image, np.dtype(np.float16))
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, image.astype(np.float16))


def test_float16_conversion_helpers_use_float_image_range() -> None:
    image = np.array([[[-1.0], [0.5], [2.0]]], dtype=np.float16)
    result = clip(image.copy(), np.dtype(np.float16))
    assert get_max_value(image.dtype) == 1.0
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, np.array([[[0.0], [0.5], [1.0]]], dtype=np.float16))
    assert convert_value(np.float16(0.3333), 3) == float(np.float16(0.3333))


@pytest.mark.parametrize("shape", [(1, 257, 1), (128, 160, 3)])
def test_from_float16_to_uint8_matches_promoted_reference(shape) -> None:
    image = np.linspace(0, 1, num=np.prod(shape), dtype=np.float32).reshape(shape).astype(np.float16)
    expected = np.clip(np.rint(image.astype(np.float32) * np.float32(255)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(from_float(image, np.dtype(np.uint8)), expected)


def test_large_from_float16_to_uint8_preserves_float32_rounding_edges() -> None:
    values = np.array(
        [-np.inf, -1, -0.0, 0, 0.5 / 17.3, 1.5 / 17.3, 127.5 / 17.3, 255 / 17.3, 1, np.inf, np.nan],
        dtype=np.float16,
    )
    image = np.resize(values, (128, 160, 3))
    result = from_float(image, np.dtype(np.uint8), max_value=17.3)
    expected = from_float(image.astype(np.float32), np.dtype(np.uint8), max_value=17.3)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize("max_value", [128.0, 255.0])
def test_from_float_uint8_saturates_nonfinite_and_rounding_edges(layout, max_value):
    source = np.array(
        [[[-np.inf], [-0.5 / 255], [0.5 / 255], [1.5 / 255], [1.2], [np.inf], [np.nan]]], dtype=np.float32
    )
    source = np.broadcast_to(source, (2, 7, 3)).copy()
    image = {
        "contiguous": source,
        "fortran": np.asfortranarray(source),
        "strided": np.repeat(source, 2, axis=1)[:, ::2],
    }[layout]
    with np.errstate(invalid="ignore"):
        expected = np.clip(np.rint(image * np.float32(max_value)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(from_float(image, np.uint8, max_value), expected)


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_uint8_roundtrip(channels, batch):
    image = generate_img(np.uint8, channels, batch)
    np.testing.assert_array_equal(from_float(to_float(image), np.uint8), image)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_to_float_roundtrip(dtype, channels, batch):
    image = generate_img(np.float32, channels, batch)
    result = to_float(from_float(image, dtype))
    np.testing.assert_array_almost_equal(image, result, decimal=2)
