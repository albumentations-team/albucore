# ruff: noqa: INP001, S101
"""Property tests for the single-volume ``remap3d`` NumPy route and grid containers."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import albucore as ac


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


@st.composite
def normalized_grids(draw: st.DrawFn) -> np.ndarray:
    """Generate small non-cubic output grids with in- and out-of-bounds normalized pulls."""
    shape = (
        draw(st.integers(min_value=1, max_value=4)),
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=6)),
        3,
    )
    values = st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False, width=32)
    return np.ascontiguousarray(draw(hnp.arrays(dtype=np.float32, shape=shape, elements=values)))


@given(
    dhwc_volumes(),
    normalized_grids(),
    st.sampled_from((cv2.INTER_NEAREST, cv2.INTER_LINEAR)),
    st.sampled_from((cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE)),
)
@settings(max_examples=50, deadline=None)
def test_remap3d_property_preserves_shape_dtype_range_and_cross_container_parity(
    volume: np.ndarray,
    sampling_grid: np.ndarray,
    interpolation: int,
    border_mode: int,
) -> None:
    """Every supported volume and NumPy/Tensor grid preserves the documented public contract."""
    numpy_result = ac.remap3d(
        volume,
        sampling_grid,
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=17.0,
    )
    tensor_grid_result = ac.remap3d(
        volume,
        torch.from_numpy(sampling_grid),
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=17.0,
    )

    assert numpy_result.shape == (*sampling_grid.shape[:3], volume.shape[-1])
    assert numpy_result.dtype == volume.dtype
    assert tensor_grid_result.shape == numpy_result.shape
    assert tensor_grid_result.dtype == volume.dtype
    np.testing.assert_array_equal(numpy_result, tensor_grid_result)
    if volume.dtype == np.uint8:
        assert int(numpy_result.min()) >= 0
        assert int(numpy_result.max()) <= 255
    else:
        assert np.all(np.isfinite(numpy_result))


@given(dhwc_volumes(), normalized_grids())
@settings(max_examples=50, deadline=None)
def test_remap3d_nearest_uint8_output_uses_only_input_or_fill_values(
    volume: np.ndarray,
    sampling_grid: np.ndarray,
) -> None:
    """Nearest categorical sampling cannot synthesize a label absent from source and constant fill."""
    labels = volume.astype(np.uint8, copy=False)
    result = ac.remap3d(labels, sampling_grid, interpolation=cv2.INTER_NEAREST, border_value=251.0)

    assert set(np.unique(result)).issubset(set(np.unique(labels)) | {251})
