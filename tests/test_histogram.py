from typing import Literal

import cv2
import numpy as np
import pytest

from albucore import equalize_uint8
import albucore.histogram as histogram_module
from albucore.histogram import _histogram, _opencv_lut


@pytest.mark.parametrize("mode", ["cv", "pil"])
@pytest.mark.parametrize("layout", ["dense", "crop", "flip"])
def test_equalize_uint8_unmasked_known_histograms(mode: Literal["cv", "pil"], layout: str) -> None:
    width = 32
    levels = np.resize(np.array([20, 100, 180, 240], dtype=np.uint8), (32, width))
    source = np.empty((32, width + 2, 3), dtype=np.uint8)
    image = source[:, 1:-1]
    image[..., 0] = levels
    image[..., 1] = 137
    image[..., 2] = levels[::-1, ::-1]
    if layout == "dense":
        image = image.copy()
    elif layout == "flip":
        image = image[:, ::-1]
    image.flags.writeable = False

    result = equalize_uint8(image, mode=mode)

    expected = image.copy()
    for channel in (0, 2):
        if mode == "cv":
            expected[..., channel] = cv2.equalizeHist(image[..., channel])
        else:
            # 256 entries per level; Pillow's integer step is 3.
            for level, mapped in zip((20, 100, 180, 240), (0, 85, 171, 255), strict=True):
                expected[..., channel][image[..., channel] == level] = mapped
    np.testing.assert_array_equal(result, expected, strict=True)
    assert result.flags.writeable
    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("mode", ["cv", "pil"])
@pytest.mark.parametrize("mask_shape", ["hw", "shared", "per-channel"])
def test_equalize_uint8_mask_selects_histogram_and_maps_unselected_pixels(
    mode: Literal["cv", "pil"],
    mask_shape: str,
) -> None:
    plane = np.repeat(np.array([10, 60, 110, 200], dtype=np.uint8), 256).reshape(32, 32)
    image = np.stack([plane, plane], axis=-1)
    selected = (plane == 10) | (plane == 110)
    mask = selected.astype(np.uint8) * 137
    if mask_shape == "shared":
        mask = mask[..., None]
    elif mask_shape == "per-channel":
        mask = np.stack([mask, np.zeros_like(mask)], axis=-1)
    mask.flags.writeable = False
    before = image.copy()

    result = equalize_uint8(image, mode=mode, mask=mask)

    if mode == "cv":
        mapped = np.where(plane >= 110, 255, 0).astype(np.uint8)
    else:
        mapped = np.where(plane > 10, 255, 0).astype(np.uint8)
    expected = np.stack([mapped, plane if mask_shape == "per-channel" else mapped], axis=-1)
    np.testing.assert_array_equal(result, expected, strict=True)
    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(result, image)


def test_equalize_uint8_empty_image() -> None:
    image = np.empty((0, 5, 3), dtype=np.uint8)
    image.flags.writeable = False
    result = equalize_uint8(image)
    assert result is not image
    assert result.shape == (0, 5, 3)
    assert result.dtype == np.uint8
    assert result.flags.writeable


@pytest.mark.parametrize("repeats", [7, 25])
@pytest.mark.parametrize("channels", [1, 4])
def test_cv_equalization_rounding_matches_native_with_full_mask(repeats: int, channels: int) -> None:
    plane = np.repeat(np.array([20, 100, 180], dtype=np.uint8), [1, repeats, repeats])[None, :]
    image = np.repeat(plane[..., None], channels, axis=-1)
    expected = np.repeat(cv2.equalizeHist(plane)[..., None], channels, axis=-1)

    np.testing.assert_array_equal(equalize_uint8(image), expected, strict=True)
    np.testing.assert_array_equal(equalize_uint8(image, mask=np.ones_like(plane)), expected, strict=True)


def test_large_histogram_preserves_counts_beyond_float32_precision() -> None:
    plane = np.zeros((4096, 4097), dtype=np.uint8)
    plane[:, -1] = 255
    plane[0, -1] = 0

    counts = _histogram(plane)

    expected = np.zeros(256, dtype=np.int64)
    expected[[0, 255]] = [2**24 + 1, 4095]
    np.testing.assert_array_equal(counts, expected, strict=True)


@pytest.mark.parametrize("mask_shape", ["hw", "shared", "per-channel"])
@pytest.mark.parametrize(("height", "width", "packed_route"), [(64, 64, True), (63, 64, False), (96, 160, True)])
def test_masked_rgb_cv_equalization_uses_packed_route_from_4096_pixels(
    monkeypatch: pytest.MonkeyPatch,
    mask_shape: str,
    height: int,
    width: int,
    *,
    packed_route: bool,
) -> None:
    rng = np.random.default_rng(height + width)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    mask = (rng.random((height, width, 3)) > 0.4).astype(np.uint8) * 255
    if mask_shape == "hw":
        mask = mask[..., 0]
    elif mask_shape == "shared":
        mask = mask[..., :1]
    calls = 0
    original = histogram_module._equalize_cv_masked_rgb

    def spy(img: np.ndarray, selection: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(img, selection)

    monkeypatch.setattr(histogram_module, "_equalize_cv_masked_rgb", spy)
    result = equalize_uint8(image, mask=mask)

    expected = np.empty_like(image)
    for channel in range(3):
        selection = mask[..., channel] if mask.ndim == 3 and mask.shape[-1] == 3 else mask.reshape(height, width)
        lut = _opencv_lut(_histogram(image[..., channel], selection))
        expected[..., channel] = image[..., channel] if lut is None else lut[image[..., channel]]
    np.testing.assert_array_equal(result, expected, strict=True)
    assert calls == int(packed_route)
