# ruff: noqa: INP001, S101
"""Property tests for the single-volume ``warp_affine3d`` NumPy and Torch contract."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import albucore as ac
from albucore.affine3d import _affine_grid, _inverse_matrix, _normalize_matrix
from albucore.sampling3d import _normalize_border_value, _sample3d_torch_cpu
from tests.property.strategies import dhwc_volumes


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


@given(
    dhwc_volumes(),
    st.sampled_from((-1.5, -0.5, 0.5, 1.5)),
    st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False, width=32),
    st.sampled_from((cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE)),
)
@settings(max_examples=100, deadline=None)
def test_warp_affine3d_xy_nearest_matches_native_3d_coordinates(
    volume: np.ndarray,
    shift: float,
    shear: float,
    border_mode: int,
) -> None:
    matrix = np.array(((0.9, shear, 0.0, 64 + shift), (0.0, 1.1, 0.0, 80 - shift), (0.0, 0.0, 1.0, 0.0)))
    size = (volume.shape[0], 128, 160)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    grid = _affine_grid(_inverse_matrix(_normalize_matrix(matrix)), volume.shape[:3], volume.shape[-1], size)
    expected = _sample3d_torch_cpu(
        tensor,
        grid,
        cv2.INTER_NEAREST,
        border_mode,
        _normalize_border_value(3.2, volume.shape[-1]),
    )

    for source in (volume, tensor, tensor.contiguous()):
        result = ac.warp_affine3d(source, matrix, size, cv2.INTER_NEAREST, border_mode, 3.2)
        if isinstance(result, torch.Tensor):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        else:
            np.testing.assert_array_equal(result, expected.permute(1, 2, 3, 0).numpy())
