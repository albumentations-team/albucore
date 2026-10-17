from __future__ import annotations

import numpy as np
from hypothesis import given

import albucore as ac
from tests.property.strategies import hwc_images


def _rolled_per_channel_lut(channels: int) -> np.ndarray:
    lut = np.empty((256, 1, channels), dtype=np.uint8)
    base = np.arange(256, dtype=np.uint8)
    for channel in range(channels):
        lut[:, 0, channel] = np.roll(base, channel + 1)
    return lut


@given(hwc_images(dtypes=(np.dtype("uint8"),)))
def test_identity_shared_lut_preserves_values(img: np.ndarray) -> None:
    lut = np.arange(256, dtype=np.uint8)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), img)


@given(hwc_images(dtypes=(np.dtype("uint8"),)))
def test_shared_lut_matches_numpy_lookup_for_non_identity_mapping(img: np.ndarray) -> None:
    lut = np.arange(255, -1, -1, dtype=np.uint8)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), lut[img])


@given(hwc_images(dtypes=(np.dtype("uint8"),), channels=(1, 3)))
def test_identity_per_channel_lut_preserves_values(img: np.ndarray) -> None:
    lut = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], img.shape[-1], axis=2)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), img)


@given(hwc_images(dtypes=(np.dtype("uint8"),), channels=(1, 3)))
def test_per_channel_lut_matches_numpy_lookup(img: np.ndarray) -> None:
    lut = _rolled_per_channel_lut(img.shape[-1])
    expected = lut[img, 0, np.arange(img.shape[-1])]

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), expected)


@given(hwc_images(dtypes=(np.dtype("uint8"),), channels=(3,)))
def test_non_contiguous_per_channel_lut_matches_numpy_lookup(img: np.ndarray) -> None:
    non_contiguous_img = np.asfortranarray(img)
    lut = _rolled_per_channel_lut(non_contiguous_img.shape[-1])
    expected = lut[non_contiguous_img, 0, np.arange(non_contiguous_img.shape[-1])]

    assert not non_contiguous_img.flags.c_contiguous
    np.testing.assert_array_equal(ac.apply_uint8_lut(non_contiguous_img, lut), expected)
