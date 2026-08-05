from __future__ import annotations

import numpy as np
from hypothesis import given

from albucore.decorators import (
    reshape_for_spatial,
    restore_from_spatial,
)
from tests.property.strategies import xhwc_images


@given(xhwc_images())
def test_xhwc_spatial_reshape_restore_roundtrip(img: np.ndarray) -> None:
    reshaped, original_shape = reshape_for_spatial(img)

    np.testing.assert_array_equal(restore_from_spatial(reshaped, original_shape), img)
