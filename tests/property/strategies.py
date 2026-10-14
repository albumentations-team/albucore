from __future__ import annotations

from typing import Any, cast

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tests.verification_constants import NON_SQUARE_HW, PROPERTY_CHANNELS


@st.composite
def hwc_images(
    draw: st.DrawFn,
    *,
    dtypes: tuple[np.dtype, ...] = (np.dtype("uint8"), np.dtype("float16"), np.dtype("float32")),
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
def dhwc_volumes(draw: st.DrawFn) -> np.ndarray:
    """Generate supported non-empty single volumes, including unit axes and high channels."""
    dtype = cast("np.dtype[Any]", draw(st.sampled_from((np.dtype(np.uint8), np.dtype(np.float32)))))
    shape = (
        draw(st.integers(min_value=1, max_value=4)),
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=6)),
        draw(st.sampled_from((1, 3, 5))),
    )
    elements: st.SearchStrategy[object]
    if dtype == np.dtype(np.uint8):
        elements = st.integers(min_value=0, max_value=255)
    else:
        elements = st.floats(min_value=-4.0, max_value=8.0, allow_nan=False, allow_infinity=False, width=32)
    return np.ascontiguousarray(draw(hnp.arrays(dtype=dtype, shape=shape, elements=elements)))
