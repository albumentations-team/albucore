from __future__ import annotations

import cv2
import numpy as np
from hypothesis import given

import albucore as ac
from tests.property.strategies import hwc_images


@given(hwc_images())
def test_resize_uses_width_height_order(img: np.ndarray) -> None:
    width = img.shape[-2] + 3
    height = img.shape[-3] + 5

    resized = ac.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    assert resized.shape == (height, width, img.shape[-1])


@given(hwc_images())
def test_double_flip_roundtrip(img: np.ndarray) -> None:
    np.testing.assert_array_equal(ac.hflip(ac.hflip(img)), img)
    np.testing.assert_array_equal(ac.vflip(ac.vflip(img)), img)
