# ruff: noqa: S101
"""Contract tests for public DHWC/CPU-CDHW volume geometry routers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

from albucore import flip_volume, rot90_volume, transpose_volume


def _numpy_volume(dtype: type[np.uint8 | np.float32], channels: int) -> np.ndarray:
    values = np.arange(2 * 3 * 5 * channels, dtype=np.int32).reshape(2, 3, 5, channels)
    return values.astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize(
    "axes",
    [0, 1, 2, (1, 2), (0, 1, 2)],
)
def test_flip_volume_numpy_preserves_dhwc_contract(
    dtype: type[np.uint8 | np.float32],
    channels: int,
    axes: int | tuple[int, ...],
) -> None:
    """NumPy volume flips preserve the requested native axes and view storage."""
    volume = _numpy_volume(dtype, channels)

    result = flip_volume(volume, axes)

    expected = np.flip(volume, axis=axes)
    assert isinstance(result, np.ndarray)
    assert result.shape == volume.shape
    assert result.dtype == volume.dtype
    assert np.shares_memory(result, volume)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize(
    "axes",
    [-3, -2, -1, (-2, -1), (-3, -2, -1)],
)
def test_flip_volume_tensor_preserves_cdhw_contract(
    dtype: torch.dtype,
    channels: int,
    axes: int | tuple[int, ...],
) -> None:
    """CPU Tensor volume flips preserve the requested native axes, layout, and dtype."""
    volume = torch.arange(channels * 2 * 3 * 5, dtype=torch.int32).reshape(channels, 2, 3, 5).to(dtype)

    result = flip_volume(volume, axes)

    expected = torch.flip(volume, dims=(axes,) if isinstance(axes, int) else axes)
    assert isinstance(result, torch.Tensor)
    assert result.shape == volume.shape
    assert result.dtype == volume.dtype
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize(("axis1", "axis2"), [(0, 1), (0, 2), (1, 2)])
def test_transpose_volume_numpy_preserves_dhwc_contract(
    dtype: type[np.uint8 | np.float32],
    channels: int,
    axis1: int,
    axis2: int,
) -> None:
    """NumPy transpose swaps the requested native axes and preserves view storage."""
    volume = _numpy_volume(dtype, channels)

    result = transpose_volume(volume, axis1, axis2)

    expected = np.swapaxes(volume, axis1, axis2)
    assert isinstance(result, np.ndarray)
    assert result.shape == expected.shape
    assert result.dtype == volume.dtype
    assert np.shares_memory(result, volume)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize(("axis1", "axis2"), [(-3, -2), (-3, -1), (-2, -1)])
def test_transpose_volume_tensor_preserves_cdhw_contract(
    dtype: torch.dtype,
    channels: int,
    axis1: int,
    axis2: int,
) -> None:
    """Tensor transpose swaps the requested native axes with native view stride semantics."""
    volume = torch.arange(channels * 2 * 3 * 5, dtype=torch.int32).reshape(channels, 2, 3, 5).to(dtype)

    result = transpose_volume(volume, axis1, axis2)

    expected = volume.transpose(axis1, axis2)
    assert isinstance(result, torch.Tensor)
    assert result.shape == expected.shape
    assert result.dtype == volume.dtype
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("axes", [(0, 1), (0, 2), (1, 2)])
def test_rot90_volume_numpy_preserves_dhwc_contract(
    dtype: type[np.uint8 | np.float32],
    channels: int,
    k: int,
    axes: tuple[int, int],
) -> None:
    """NumPy quarter-turns rotate the requested native plane and preserve view storage."""
    volume = _numpy_volume(dtype, channels)

    result = rot90_volume(volume, k, axes)

    expected = np.rot90(volume, k, axes=axes)
    assert isinstance(result, np.ndarray)
    assert result.shape == expected.shape
    assert result.dtype == volume.dtype
    assert np.shares_memory(result, volume)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("axes", [(-3, -2), (-3, -1), (-2, -1)])
def test_rot90_volume_tensor_preserves_cdhw_contract(
    dtype: torch.dtype,
    channels: int,
    k: int,
    axes: tuple[int, int],
) -> None:
    """Tensor quarter-turns rotate the requested native plane with native stride semantics."""
    volume = torch.arange(channels * 2 * 3 * 5, dtype=torch.int32).reshape(channels, 2, 3, 5).to(dtype)

    result = rot90_volume(volume, k, axes)

    expected = torch.rot90(volume, k, dims=axes)
    assert isinstance(result, torch.Tensor)
    assert result.shape == expected.shape
    assert result.dtype == volume.dtype
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("operation", ["flip", "transpose", "rot90"])
def test_volume_geometry_tensor_accepts_strided_cdhw_input(operation: Literal["flip", "transpose", "rot90"]) -> None:
    """Native Tensor geometry accepts the caller-permitted strided CDHW layout."""
    volume = torch.arange(3 * 2 * 5 * 7, dtype=torch.float32).reshape(3, 2, 5, 7)[:, :, :, ::2]

    if operation == "flip":
        result = flip_volume(volume, (-2, -1))
        expected = torch.flip(volume, dims=(-2, -1))
    elif operation == "transpose":
        result = transpose_volume(volume, -2, -1)
        expected = volume.transpose(-2, -1)
    else:
        result = rot90_volume(volume, 1, (-2, -1))
        expected = torch.rot90(volume, 1, dims=(-2, -1))

    torch.testing.assert_close(result, expected)
