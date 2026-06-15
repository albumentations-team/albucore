from __future__ import annotations

import numpy as np
from hypothesis import given

import albucore as ac
from tests.property.strategies import hwc_images


@given(hwc_images())
def test_flips_preserve_shape_dtype(img: np.ndarray) -> None:
    assert ac.hflip(img).shape == img.shape
    assert ac.hflip(img).dtype == img.dtype
    assert ac.vflip(img).shape == img.shape
    assert ac.vflip(img).dtype == img.dtype


@given(hwc_images(dtypes=(np.dtype("uint8"),)))
def test_to_from_float_preserves_shape(img: np.ndarray) -> None:
    as_float = ac.to_float(img)
    assert as_float.shape == img.shape
    assert as_float.dtype == np.float32

    restored = ac.from_float(as_float, np.uint8)
    assert restored.shape == img.shape
    assert restored.dtype == np.uint8
