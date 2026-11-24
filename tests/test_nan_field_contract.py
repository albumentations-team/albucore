# ruff: noqa: S101
"""NaN contract for the operations AlbumentationsX applies to vector fields.

Vector fields are the only data that carries NaN (NaN marks an invalid vector). AlbumentationsX calls only ``remap``,
``remap3d`` and the ``tensor_bridge`` helpers on fields, so only these must keep NaN exactly. Every other operation
assumes finite input and is not tested with NaN. Nearest sampling uses integer pull coordinates: no arithmetic touches
a sampled value, so any NaN-unsafe fast route changes the result.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from albucore import TensorLayout, remap, remap3d
from albucore.tensor_bridge import (
    numpy_to_tensor_layout,
    numpy_to_tensor_storage,
    tensor_to_numpy_channel_last,
    tensor_to_numpy_storage,
)

FIELD_DTYPES = (np.float16, np.float32)
FIELD_CHANNELS = (2, 3, 5)


def _field(shape: tuple[int, ...], dtype: type[np.floating]) -> np.ndarray:
    """Build a channel-last field whose finite values are unique and whose invalid vectors are partly or fully NaN."""
    values = (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) % 251) / np.float32(4)
    field = values.astype(dtype)
    flat = field.reshape(-1, shape[-1])
    flat[::3, 0] = np.nan
    flat[1::7] = np.nan
    flat[2::11, -1] = np.nan
    return field


def _assert_same(actual: np.ndarray, expected: np.ndarray) -> None:
    """Require equal dtype, shape, NaN positions and finite values (-0.0 equals +0.0)."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual, expected)


def _remap_maps(height: int, width: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    y, x = np.mgrid[:height, :width].astype(np.float32)
    return {
        "identity": (x, y),
        "flip": (np.float32(width - 1) - x, y),
        "shift_and_swap_rows": ((x + 2) % width, np.float32(height - 1) - y),
    }


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
@pytest.mark.parametrize("channels", FIELD_CHANNELS)
@pytest.mark.parametrize("map_name", ["identity", "flip", "shift_and_swap_rows"])
def test_remap_nearest_keeps_nan_exactly(dtype: type[np.floating], channels: int, map_name: str) -> None:
    field = _field((6, 9, channels), dtype)
    map_x, map_y = _remap_maps(6, 9)[map_name]
    expected = field[map_y.astype(np.intp), map_x.astype(np.intp)]

    _assert_same(remap(field, map_x, map_y, interpolation=cv2.INTER_NEAREST), expected)


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
def test_remap_nearest_keeps_nan_in_non_contiguous_field(dtype: type[np.floating]) -> None:
    field = _field((6, 9, 3), dtype)[::-1, ::2]
    map_x, map_y = _remap_maps(6, 5)["flip"]
    expected = field[map_y.astype(np.intp), map_x.astype(np.intp)]

    _assert_same(remap(field, map_x, map_y, interpolation=cv2.INTER_NEAREST), expected)


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
def test_remap_nearest_border_keeps_nan_and_fills_outside(dtype: type[np.floating]) -> None:
    field = _field((6, 9, 2), dtype)
    map_x, map_y = _remap_maps(6, 9)["identity"]
    map_x = map_x + np.float32(3)
    expected = np.zeros_like(field)
    expected[:, :6] = field[:, 3:]

    _assert_same(remap(field, map_x, map_y, interpolation=cv2.INTER_NEAREST, border_value=0), expected)


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
def test_remap_tensor_nearest_keeps_nan_exactly(dtype: type[np.floating]) -> None:
    field = _field((6, 9, 3), dtype)
    map_x, map_y = _remap_maps(6, 9)["flip"]
    expected = field[map_y.astype(np.intp), map_x.astype(np.intp)]
    tensor = torch.from_numpy(field.transpose(2, 0, 1).copy())

    result = remap(tensor, map_x, map_y, interpolation=cv2.INTER_NEAREST, layout="CHW")

    assert isinstance(result, torch.Tensor)
    _assert_same(result.numpy().transpose(1, 2, 0), expected)


def _remap3d_grid(depth: int, height: int, width: int, *, flip: bool) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return an integer-center pull grid in normalized x/y/z order and its source indices."""
    z, y, x = (axis.astype(np.float32) for axis in np.mgrid[:depth, :height, :width])
    if flip:
        x = np.float32(width - 1) - x
        z = np.float32(depth - 1) - z
    sizes = (width, height, depth)
    grid = np.stack([(c * 2 + 1) / np.float32(n) - 1 for c, n in zip((x, y, z), sizes, strict=True)], axis=-1)
    return grid.astype(np.float32), (z.astype(np.intp), y.astype(np.intp), x.astype(np.intp))


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
@pytest.mark.parametrize("flip", [False, True])
def test_remap3d_nearest_keeps_nan_exactly(dtype: type[np.floating], *, flip: bool) -> None:
    field = _field((3, 4, 5, 3), dtype)
    grid, (z, y, x) = _remap3d_grid(3, 4, 5, flip=flip)

    _assert_same(remap3d(field, grid, interpolation=cv2.INTER_NEAREST), field[z, y, x])


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
def test_remap3d_tensor_nearest_keeps_nan_exactly(dtype: type[np.floating]) -> None:
    field = _field((3, 4, 5, 3), dtype)
    grid, (z, y, x) = _remap3d_grid(3, 4, 5, flip=True)
    tensor = torch.from_numpy(field.transpose(3, 0, 1, 2).copy())

    result = remap3d(tensor, grid, interpolation=cv2.INTER_NEAREST)

    assert isinstance(result, torch.Tensor)
    _assert_same(result.numpy().transpose(1, 2, 3, 0), field[z, y, x])


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
@pytest.mark.parametrize(
    ("layout", "shape"),
    [("CHW", (4, 5, 3)), ("CHW", (4, 5, 2)), ("NCHW", (2, 4, 5, 3)), ("CDHW", (2, 3, 4, 3)), ("NCDHW", (2, 2, 3, 4, 3))],
)
@pytest.mark.parametrize("contiguous", [False, True])
def test_numpy_to_tensor_layout_keeps_nan_exactly(
    dtype: type[np.floating],
    layout: TensorLayout,
    shape: tuple[int, ...],
    *,
    contiguous: bool,
) -> None:
    field = _field(shape, dtype)

    tensor = numpy_to_tensor_layout(field, layout, contiguous=contiguous)

    assert not contiguous or tensor.is_contiguous()
    _assert_same(tensor_to_numpy_channel_last(tensor, layout), field)


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
def test_numpy_to_tensor_layout_keeps_nan_for_flipped_and_strided_views(dtype: type[np.floating]) -> None:
    field = _field((6, 8, 3), dtype)
    for view in (field[::-1], field[:, ::2], field[:, ::-1][::2]):
        tensor = numpy_to_tensor_layout(view, "CHW", contiguous=True)

        assert tensor.is_contiguous()
        _assert_same(tensor_to_numpy_channel_last(tensor, "CHW"), view)


@pytest.mark.parametrize("dtype", FIELD_DTYPES)
def test_storage_bridges_keep_nan_exactly(dtype: type[np.floating]) -> None:
    field = _field((4, 5, 3), dtype)

    for contiguous in (False, True):
        _assert_same(tensor_to_numpy_storage(numpy_to_tensor_storage(field, contiguous=contiguous)), field)
    _assert_same(tensor_to_numpy_storage(numpy_to_tensor_storage(field[::-1])), field[::-1])
