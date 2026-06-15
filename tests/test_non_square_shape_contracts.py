from __future__ import annotations

import pytest
import numpy as np

cv2 = pytest.importorskip("cv2")

import albucore as ac  # noqa: E402
from tests.verification_constants import NON_SQUARE_IMAGE_HW, PROPERTY_CHANNELS, RESIZE_TARGET_WH  # noqa: E402


def _image(dtype: np.dtype, channels: int) -> np.ndarray:
    height, width = NON_SQUARE_IMAGE_HW
    shape = (height, width, channels)
    if dtype == np.uint8:
        return np.arange(np.prod(shape), dtype=np.uint32).reshape(shape).astype(np.uint8)
    return np.linspace(0.0, 1.0, num=np.prod(shape), dtype=np.float32).reshape(shape)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", PROPERTY_CHANNELS, ids=["c1", "c3", "c9"])
def test_arithmetic_preserves_non_square_shape(dtype: np.dtype, channels: int) -> None:
    img = _image(dtype, channels)
    value = 2 if dtype == np.uint8 else np.float32(0.25)

    assert ac.add(img, value).shape == img.shape
    assert ac.multiply(img, value).shape == img.shape
    assert ac.multiply_add(img, value, value).shape == img.shape


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", PROPERTY_CHANNELS, ids=["c1", "c3", "c9"])
def test_flips_preserve_non_square_shape(dtype: np.dtype, channels: int) -> None:
    img = _image(dtype, channels)

    assert ac.hflip(img).shape == img.shape
    assert ac.vflip(img).shape == img.shape
    np.testing.assert_array_equal(ac.hflip(ac.hflip(img)), img)
    np.testing.assert_array_equal(ac.vflip(ac.vflip(img)), img)


@pytest.mark.parametrize("channels", PROPERTY_CHANNELS, ids=["c1", "c3", "c9"])
def test_conversion_preserves_non_square_shape(channels: int) -> None:
    img = _image(np.uint8, channels)

    as_float = ac.to_float(img)
    assert as_float.shape == img.shape
    assert as_float.dtype == np.float32

    restored = ac.from_float(as_float, np.uint8)
    assert restored.shape == img.shape
    assert restored.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", PROPERTY_CHANNELS, ids=["c1", "c3", "c9"])
def test_resize_uses_width_height_order(dtype: np.dtype, channels: int) -> None:
    img = _image(dtype, channels)
    width, height = RESIZE_TARGET_WH

    resized = ac.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    assert resized.shape == (height, width, channels)

    if dtype == np.uint8 and channels == 3:
        expected = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(resized, expected)
