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


def _apply_per_channel_with_sz_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            ac.sz_lut(
                np.ascontiguousarray(img[..., channel : channel + 1]),
                np.ascontiguousarray(lut[:, 0, channel]),
                inplace=False,
            )
            for channel in range(img.shape[-1])
        ],
        axis=-1,
    )


@given(hwc_images(dtypes=(np.dtype("uint8"),)))
def test_identity_shared_lut_preserves_values(img: np.ndarray) -> None:
    lut = np.arange(256, dtype=np.uint8)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), img)


@given(hwc_images(dtypes=(np.dtype("uint8"),)))
def test_shared_lut_matches_sz_lut_for_non_identity_mapping(img: np.ndarray) -> None:
    lut = np.arange(255, -1, -1, dtype=np.uint8)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), ac.sz_lut(img, lut, inplace=False))


@given(hwc_images(dtypes=(np.dtype("uint8"),), channels=(1, 3)))
def test_identity_per_channel_lut_preserves_values(img: np.ndarray) -> None:
    lut = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], img.shape[-1], axis=2)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), img)


@given(hwc_images(dtypes=(np.dtype("uint8"),), channels=(1, 3)))
def test_per_channel_lut_matches_independent_sz_lut_channels(img: np.ndarray) -> None:
    lut = _rolled_per_channel_lut(img.shape[-1])
    expected = _apply_per_channel_with_sz_lut(img, lut)

    np.testing.assert_array_equal(ac.apply_uint8_lut(img, lut), expected)


@given(hwc_images(dtypes=(np.dtype("uint8"),), channels=(3,)))
def test_non_contiguous_per_channel_lut_matches_sz_lut_fallback(img: np.ndarray) -> None:
    non_contiguous_img = np.asfortranarray(img)
    lut = _rolled_per_channel_lut(non_contiguous_img.shape[-1])
    expected = _apply_per_channel_with_sz_lut(non_contiguous_img, lut)

    assert not non_contiguous_img.flags.c_contiguous
    np.testing.assert_array_equal(ac.apply_uint8_lut(non_contiguous_img, lut, inplace=False), expected)
