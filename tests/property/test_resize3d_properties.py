# ruff: noqa: INP001, S101
"""Property tests for DHWC NumPy and CPU CDHW Torch ``resize3d`` contracts."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import albucore as ac


@st.composite
def dhwc_volumes(draw: st.DrawFn) -> np.ndarray:
    """Generate small supported DHWC volumes, including singleton spatial axes and C>4."""
    dtype = cast("np.dtype[Any]", draw(st.sampled_from((np.dtype(np.uint8), np.dtype(np.float32)))))
    shape = (
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=6)),
        draw(st.integers(min_value=1, max_value=7)),
        draw(st.sampled_from((1, 3, 5))),
    )
    elements: st.SearchStrategy[object]
    if dtype == np.dtype(np.uint8):
        elements = st.integers(min_value=0, max_value=255)
    else:
        elements = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32)
    return np.ascontiguousarray(draw(hnp.arrays(dtype=dtype, shape=shape, elements=elements)))


resize_sizes = st.tuples(
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=1, max_value=7),
    st.integers(min_value=1, max_value=8),
)


@given(dhwc_volumes(), resize_sizes)
@settings(max_examples=50, deadline=None)
def test_resize3d_numpy_property_preserves_shape_dtype_and_uint8_range(
    volume: np.ndarray,
    size: tuple[int, int, int],
) -> None:
    """Every supported NumPy volume returns the requested DHWC shape under linear antialiasing."""
    result = ac.resize3d(volume, size, interpolation=cv2.INTER_LINEAR, antialias=True)

    assert result.shape == (*size, volume.shape[-1])
    assert result.dtype == volume.dtype
    if volume.dtype == np.uint8:
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255


@given(dhwc_volumes(), resize_sizes)
@settings(max_examples=50, deadline=None)
def test_resize3d_torch_property_preserves_the_documented_native_error_bound(
    volume: np.ndarray,
    size: tuple[int, int, int],
) -> None:
    """Tensor results match native interpolation exactly or the bounded faster all-axis-upscale route."""
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)

    result = ac.resize3d(tensor, size)
    working = tensor if tensor.dtype == torch.float32 else tensor.to(torch.float32)
    expected = torch_f.interpolate(working.unsqueeze(0), size=size, mode="trilinear", align_corners=False)
    if tensor.dtype == torch.uint8:
        expected = torch.minimum(expected + 0.5, expected.new_tensor(255)).to(torch.uint8)

    expected = expected.squeeze(0)
    if tensor.dtype == torch.float32:
        torch.testing.assert_close(result, expected, rtol=2e-4, atol=3e-5)
    else:
        delta = (result.to(torch.int16) - expected.to(torch.int16)).abs()
        assert int(delta.max()) <= 1
