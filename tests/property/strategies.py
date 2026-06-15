from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tests.verification_constants import NON_SQUARE_HW, PROPERTY_CHANNELS


@st.composite
def hwc_images(
    draw: st.DrawFn,
    *,
    dtypes: tuple[np.dtype, ...] = (np.dtype("uint8"), np.dtype("float32")),
    channels: tuple[int, ...] = PROPERTY_CHANNELS,
) -> np.ndarray:
    height, width = draw(st.sampled_from(NON_SQUARE_HW))
    num_channels = draw(st.sampled_from(channels))
    dtype = draw(st.sampled_from(dtypes))
    shape = (height, width, num_channels)

    if dtype == np.dtype("uint8"):
        elements: Any = st.integers(min_value=0, max_value=255)
    else:
        elements = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32)

    arr = draw(hnp.arrays(dtype=dtype, shape=shape, elements=elements))
    return np.ascontiguousarray(arr)


@st.composite
def xhwc_images(draw: st.DrawFn) -> np.ndarray:
    x_dim = draw(st.sampled_from((2, 3)))
    height, width = draw(st.sampled_from(NON_SQUARE_HW))
    channels = draw(st.sampled_from((1, 3)))
    shape = (x_dim, height, width, channels)
    arr = draw(
        hnp.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
        ),
    )
    return np.ascontiguousarray(arr)


@st.composite
def ndhwc_images(draw: st.DrawFn) -> np.ndarray:
    n_dim = draw(st.sampled_from((1, 2)))
    d_dim = draw(st.sampled_from((2, 3)))
    height, width = draw(st.sampled_from(NON_SQUARE_HW))
    channels = draw(st.sampled_from((1, 3)))
    shape = (n_dim, d_dim, height, width, channels)
    arr = draw(
        hnp.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
        ),
    )
    return np.ascontiguousarray(arr)
