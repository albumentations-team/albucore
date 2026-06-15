from __future__ import annotations

import numpy as np
from hypothesis import given

import albucore as ac
from tests.property.strategies import hwc_images


@given(hwc_images())
def test_add_zero_preserves_values(img: np.ndarray) -> None:
    value = 0 if img.dtype == np.uint8 else np.float32(0.0)
    np.testing.assert_array_equal(ac.add(img, value), img)


@given(hwc_images())
def test_multiply_one_preserves_values(img: np.ndarray) -> None:
    value = 1 if img.dtype == np.uint8 else np.float32(1.0)
    np.testing.assert_array_equal(ac.multiply(img, value), img)


@given(hwc_images(dtypes=(np.dtype("uint8"),)))
def test_uint8_arithmetic_bounds(img: np.ndarray) -> None:
    added = ac.add(img, 250)
    multiplied = ac.multiply(img, 3)

    assert added.dtype == np.uint8
    assert multiplied.dtype == np.uint8
    assert int(added.min()) >= 0
    assert int(added.max()) <= 255
    assert int(multiplied.min()) >= 0
    assert int(multiplied.max()) <= 255
