# ruff: noqa: INP001
"""Property tests for the shared-grid 3D volume batch routers."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from albucore import remap3d, warp_affine3d


@st.composite
def volume_batches(draw: st.DrawFn) -> np.ndarray:
    """Generate small contiguous NDHWC batches with independently sampled voxels."""
    dtype = cast("np.dtype[Any]", draw(st.sampled_from((np.dtype(np.uint8), np.dtype(np.float32)))))
    shape = (
        draw(st.integers(min_value=1, max_value=4)),
        draw(st.integers(min_value=1, max_value=3)),
        draw(st.integers(min_value=1, max_value=4)),
        draw(st.integers(min_value=1, max_value=3)),
        draw(st.sampled_from((1, 3, 5))),
    )
    elements: st.SearchStrategy[object]
    if dtype == np.dtype(np.uint8):
        elements = st.integers(min_value=0, max_value=255)
    else:
        elements = st.floats(min_value=-2.0, max_value=4.0, allow_nan=False, allow_infinity=False, width=32)
    return np.ascontiguousarray(draw(hnp.arrays(dtype=dtype, shape=shape, elements=elements)))


@st.composite
def output_sizes(draw: st.DrawFn) -> tuple[int, int, int]:
    """Generate nonempty output shapes, including unit axes and changed dimensions."""
    size = st.integers(min_value=1, max_value=5)
    return draw(st.tuples(size, size, size))


@st.composite
def normalized_grids(draw: st.DrawFn) -> np.ndarray:
    """Generate one shared normalized pull grid with interior and outside coordinates."""
    size = draw(output_sizes())
    coordinates = st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False, width=32)
    return np.ascontiguousarray(draw(hnp.arrays(dtype=np.float32, shape=(*size, 3), elements=coordinates)))


@given(volume_batches(), output_sizes(), st.sampled_from((cv2.INTER_NEAREST, cv2.INTER_LINEAR)))
@settings(max_examples=40, deadline=None)
def test_warp_affine3d_batch_property_matches_numpy_and_tensor_single_volume_references(
    volumes: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
) -> None:
    """Random layouts, channels, dtypes, and sizes preserve per-volume affine results."""
    matrix = np.array(((1.0, 0.03, 0.0, 0.2), (0.0, 0.95, 0.02, -0.15), (0.01, 0.0, 1.0, 0.1)), dtype=np.float32)
    numpy_result = warp_affine3d(volumes, matrix, size, interpolation=interpolation, border_value=3.0)
    expected = np.stack(
        [warp_affine3d(item, matrix, size, interpolation=interpolation, border_value=3.0) for item in volumes],
    )
    np.testing.assert_array_equal(numpy_result, expected)

    tensor_volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
    tensor_result = warp_affine3d(
        tensor_volumes,
        matrix,
        size,
        interpolation=interpolation,
        border_value=3.0,
    )
    tensor_expected = torch.stack(
        [warp_affine3d(item, matrix, size, interpolation=interpolation, border_value=3.0) for item in tensor_volumes],
    )
    torch.testing.assert_close(tensor_result, tensor_expected, rtol=0, atol=0)


@given(volume_batches(), normalized_grids(), st.sampled_from((cv2.INTER_NEAREST, cv2.INTER_LINEAR)))
@settings(max_examples=40, deadline=None)
def test_remap3d_batch_property_matches_numpy_and_tensor_single_volume_references(
    volumes: np.ndarray,
    sampling_grid: np.ndarray,
    interpolation: int,
) -> None:
    """Random shared grids preserve each source's scalar, border, dtype, and layout behavior."""
    fill = tuple(float(2 + channel) for channel in range(volumes.shape[-1]))
    numpy_result = remap3d(volumes, sampling_grid, interpolation=interpolation, border_value=fill)
    expected = np.stack(
        [remap3d(item, sampling_grid, interpolation=interpolation, border_value=fill) for item in volumes],
    )
    np.testing.assert_array_equal(numpy_result, expected)

    tensor_volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
    tensor_result = remap3d(
        tensor_volumes,
        torch.from_numpy(sampling_grid),
        interpolation=interpolation,
        border_value=fill,
    )
    tensor_expected = torch.stack(
        [
            remap3d(
                item,
                torch.from_numpy(sampling_grid),
                interpolation=interpolation,
                border_value=fill,
            )
            for item in tensor_volumes
        ],
    )
    torch.testing.assert_close(tensor_result, tensor_expected, rtol=0, atol=0)
