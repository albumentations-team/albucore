from typing import Literal

import cv2
import numpy as np
import pytest

from albucore import equalize_uint8


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
    result = equalize_uint8(np.empty((0, 5, 3), dtype=np.uint8))
    assert result.shape == (0, 5, 3)
    assert result.dtype == np.uint8
    assert result.flags.writeable
