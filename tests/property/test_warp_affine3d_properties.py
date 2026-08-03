# ruff: noqa: INP001, S101
"""Property tests for the single-volume ``warp_affine3d`` NumPy and Torch contract."""

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


output_sizes = st.tuples(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=1, max_value=7),
)
translations = st.tuples(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=32),
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=32),
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=32),
)


def _translation(values: tuple[float, float, float]) -> np.ndarray:
    """Build one non-singular forward matrix from generated x/y/z translation values."""
    x, y, z = values
    return np.array(((1.0, 0.0, 0.0, x), (0.0, 1.0, 0.0, y), (0.0, 0.0, 1.0, z)), dtype=np.float32)


@given(dhwc_volumes(), output_sizes, translations, st.sampled_from((cv2.INTER_NEAREST, cv2.INTER_LINEAR)))
@settings(max_examples=50, deadline=None)
def test_warp_affine3d_property_preserves_shape_dtype_range_and_cross_container_parity(
    volume: np.ndarray,
    size: tuple[int, int, int],
    translation: tuple[float, float, float],
    interpolation: int,
) -> None:
    """Every supported single volume keeps its contract through both public container routes."""
    matrix = _translation(translation)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)

    numpy_result = ac.warp_affine3d(volume, matrix, size, interpolation=interpolation, border_value=17.0)
    tensor_result = ac.warp_affine3d(tensor, matrix, size, interpolation=interpolation, border_value=17.0)

    assert numpy_result.shape == (*size, volume.shape[-1])
    assert numpy_result.dtype == volume.dtype
    assert tensor_result.shape == (volume.shape[-1], *size)
    assert tensor_result.dtype == tensor.dtype
    np.testing.assert_array_equal(numpy_result, tensor_result.permute(1, 2, 3, 0).numpy())
    if volume.dtype == np.uint8:
        assert int(numpy_result.min()) >= 0
        assert int(numpy_result.max()) <= 255
    else:
        assert np.all(np.isfinite(numpy_result))


@given(dhwc_volumes(), translations)
@settings(max_examples=50, deadline=None)
def test_warp_affine3d_nearest_uint8_output_uses_only_input_or_fill_values(
    volume: np.ndarray,
    translation: tuple[float, float, float],
) -> None:
    """Nearest categorical sampling cannot synthesize a label that is absent from source and constant fill."""
    labels = volume.astype(np.uint8, copy=False)
    result = ac.warp_affine3d(
        labels,
        _translation(translation),
        labels.shape[:3],
        interpolation=cv2.INTER_NEAREST,
        border_value=251.0,
    )

    assert set(np.unique(result)).issubset(set(np.unique(labels)) | {251})
