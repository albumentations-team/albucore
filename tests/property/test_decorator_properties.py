from __future__ import annotations

import numpy as np
from hypothesis import given

from albucore.decorators import (
    restore_from_channel,
    restore_from_spatial,
    reshape_for_channel,
    reshape_for_spatial,
)
from tests.property.strategies import ndhwc_images, xhwc_images


@given(xhwc_images())
def test_xhwc_spatial_reshape_restore_roundtrip(img: np.ndarray) -> None:
    reshaped, original_shape = reshape_for_spatial(img)

    np.testing.assert_array_equal(restore_from_spatial(reshaped, original_shape), img)


@given(ndhwc_images())
def test_ndhwc_channel_reshape_restore_roundtrip(img: np.ndarray) -> None:
    reshaped, original_shape = reshape_for_channel(img)

    np.testing.assert_array_equal(restore_from_channel(reshaped, original_shape), img)
