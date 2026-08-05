# ruff: noqa: S101
"""Contract and differential tests for public DHWC/CPU-CDHW ``resize3d``."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f

from albucore import resize3d
from albucore.geometric import _can_resize3d_joint_hw, _resize3d_numpy_axis_packing
from albucore.utils import get_opencv_max_channels


def _numpy_volume(dtype: type[np.uint8 | np.float32], channels: int) -> np.ndarray:
    """Create a reproducible DHWC volume with the requested supported dtype."""
    rng = np.random.default_rng(20260803)
    shape = (5, 7, 9, channels)
    if dtype is np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _torch_linear_reference(volume: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Explicit public Torch contract for linear uint8 and float32 Tensor inputs."""
    working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
    with torch.inference_mode():
        result = torch_f.interpolate(working.unsqueeze(0), size=size, mode="trilinear", align_corners=False)
        if volume.dtype == torch.uint8:
            result = torch.minimum(result + 0.5, result.new_tensor(255)).to(torch.uint8)
    return cast("torch.Tensor", result.squeeze(0))


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 3, 5], ids=["c1", "c3", "c5"])
@pytest.mark.parametrize("size", [(3, 5, 12), (8, 11, 4)], ids=["mixed", "up_down"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR], ids=["nearest", "linear"])
def test_resize3d_numpy_preserves_public_contract(
    dtype: type[np.uint8 | np.float32],
    channels: int,
    size: tuple[int, int, int],
    interpolation: int,
) -> None:
    """NumPy DHWC accepts 1 and high channel counts while preserving dtype and requested shape."""
    volume = _numpy_volume(dtype, channels)

    result = resize3d(volume, size, interpolation=interpolation)

    assert isinstance(result, np.ndarray)
    assert result.shape == (*size, channels)
    assert result.dtype == volume.dtype
    if interpolation == cv2.INTER_NEAREST:
        expected = _resize3d_numpy_axis_packing(volume, size, interpolation, antialias=False)
        np.testing.assert_array_equal(result, expected)
    elif dtype is np.float32:
        expected = _resize3d_numpy_axis_packing(volume, size, interpolation, antialias=False)
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
    else:
        assert result.min() >= 0
        assert result.max() <= 255


def test_resize3d_numpy_antialias_mixed_axes_uses_three_pass_semantics() -> None:
    """Mixed H/W scale directions retain per-axis antialias behaviour through the safe fallback."""
    volume = _numpy_volume(np.float32, channels=5)
    size = (3, 4, 12)

    result = resize3d(volume, size, interpolation=cv2.INTER_LINEAR, antialias=True)
    expected = _resize3d_numpy_axis_packing(volume, size, cv2.INTER_LINEAR, antialias=True)

    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_resize3d_numpy_falls_back_when_joint_hw_packing_exceeds_opencv_limit() -> None:
    """A small high-channel DHWC volume never asks OpenCV to encode more than its supported channel count."""
    depth = 3
    channels = get_opencv_max_channels() // depth + 1
    volume = np.arange(depth * 5 * 7 * channels, dtype=np.uint8).reshape(depth, 5, 7, channels)
    size = (4, 8, 11)

    assert not _can_resize3d_joint_hw(volume, size, antialias=False)
    result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)
    expected = _resize3d_numpy_axis_packing(volume, size, cv2.INTER_NEAREST, antialias=False)

    np.testing.assert_array_equal(result, expected)


def test_resize3d_numpy_accepts_noncontiguous_dhwc() -> None:
    """Axis packing correctly handles a valid DHWC view with non-contiguous spatial strides."""
    volume = _numpy_volume(np.float32, channels=5)[:, :, ::2, :]

    result = resize3d(volume, (4, 6, 8))

    assert result.shape == (4, 6, 8, 5)
    assert result.dtype == np.float32


def test_resize3d_numpy_identity_returns_input() -> None:
    """A NumPy identity resize neither allocates nor changes the input container."""
    volume = _numpy_volume(np.uint8, channels=3)

    assert resize3d(volume, volume.shape[:3]) is volume


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("source_shape", [(1, 7, 9), (5, 1, 9), (5, 7, 1)], ids=["unit_d", "unit_h", "unit_w"])
def test_resize3d_numpy_accepts_every_unit_input_axis(
    dtype: type[np.uint8 | np.float32],
    source_shape: tuple[int, int, int],
) -> None:
    """Every valid singleton input spatial axis retains DHWC layout, dtype, and independent output storage."""
    rng = np.random.default_rng(20260803)
    shape = (*source_shape, 5)
    volume: np.ndarray
    if dtype is np.uint8:
        volume = rng.integers(0, 256, size=shape, dtype=np.uint8)
    else:
        volume = rng.random(shape, dtype=np.float32)

    result = resize3d(volume, (4, 6, 8))

    assert result.shape == (4, 6, 8, 5)
    assert result.dtype == volume.dtype
    assert not np.shares_memory(result, volume)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_resize3d_numpy_depth_unit_linear_preserves_dtype(dtype: type[np.uint8 | np.float32]) -> None:
    """The measured D-to-one NumPy route keeps float32 intermediates and the public dtype."""
    volume = _numpy_volume(dtype, channels=5)

    result = resize3d(volume, (1, 6, 8))

    assert result.shape == (1, 6, 8, 5)
    assert result.dtype == volume.dtype


def test_resize3d_numpy_uint8_depth_unit_matches_torch_full_route() -> None:
    """The measured uint8 D-to-one route may use the pre-imported CPU Torch kernel without a dtype change."""
    volume = _numpy_volume(np.uint8, channels=5)
    size = (1, 6, 8)

    result = resize3d(volume, size)
    expected = _torch_linear_reference(torch.from_numpy(volume).permute(3, 0, 1, 2), size)

    np.testing.assert_array_equal(result, expected.permute(1, 2, 3, 0).numpy())


def test_resize3d_numpy_float32_downscale_matches_torch_full_route() -> None:
    """The float32 all-downscale route shares NumPy storage with CPU Torch and keeps DHWC output layout."""
    volume = np.random.default_rng(20260803).random((8, 64, 80, 3), dtype=np.float32)
    size = (4, 48, 60)

    result = resize3d(volume, size)
    expected = _torch_linear_reference(torch.from_numpy(volume).permute(3, 0, 1, 2), size)

    np.testing.assert_allclose(result, expected.permute(1, 2, 3, 0).numpy(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("size", [(1, 4, 5), (3, 1, 5), (3, 4, 1)], ids=["unit_d", "unit_h", "unit_w"])
def test_resize3d_cross_container_unit_output_contract(
    dtype: type[np.uint8 | np.float32],
    size: tuple[int, int, int],
) -> None:
    """NumPy DHWC and Torch CDHW keep the documented shape, dtype, and bounded cross-container difference."""
    volume = _numpy_volume(dtype, channels=3)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)

    numpy_result = resize3d(volume, size)
    tensor_result = resize3d(tensor, size).permute(1, 2, 3, 0).numpy()

    assert numpy_result.shape == tensor_result.shape == (*size, 3)
    assert numpy_result.dtype == tensor_result.dtype == volume.dtype
    if dtype is np.float32:
        np.testing.assert_allclose(numpy_result, tensor_result, rtol=2e-5, atol=2e-5)
    else:
        delta = np.abs(numpy_result.astype(np.int16) - tensor_result.astype(np.int16))
        assert delta.max() <= 1


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize("size", [(3, 5, 12), (8, 11, 4)], ids=["mixed", "up_down"])
def test_resize3d_torch_linear_matches_native_cpu_interpolate_outside_all_axis_upscales(
    dtype: torch.dtype,
    channels: int,
    size: tuple[int, int, int],
) -> None:
    """Non-upscale CPU CDHW Tensor regions retain the exact native Torch result."""
    if dtype == torch.uint8:
        volume = torch.arange(channels * 5 * 7 * 9, dtype=dtype).reshape(channels, 5, 7, 9)
    else:
        volume = torch.rand((channels, 5, 7, 9), dtype=dtype)

    result = resize3d(volume, size)
    expected = _torch_linear_reference(volume, size)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (channels, *size)
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
@pytest.mark.parametrize("channel_last_strided", [False, True], ids=["contiguous", "channel_last_strided"])
def test_resize3d_torch_all_axis_upscale_uses_the_fast_numpy_route(
    dtype: torch.dtype,
    channels: int,
    channel_last_strided: bool,
) -> None:
    """At-threshold large all-axis Tensor upscales use the selected route without boundary copies."""
    rng = np.random.default_rng(20260803)
    volume: np.ndarray
    if dtype == torch.uint8:
        volume = rng.integers(0, 256, size=(5, 20, 20, channels), dtype=np.uint8)
    else:
        volume = rng.random((5, 20, 20, channels), dtype=np.float32)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).contiguous()
    if channel_last_strided:
        tensor = tensor.permute(1, 2, 3, 0).contiguous().permute(3, 0, 1, 2)
    size = (10, 25, 40)

    result = resize3d(tensor, size)
    expected = torch.from_numpy(resize3d(volume, size)).permute(3, 0, 1, 2)
    native = _torch_linear_reference(tensor, size)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if dtype == torch.float32:
        torch.testing.assert_close(result, native, rtol=2e-4, atol=3e-5)
    else:
        delta = (result.to(torch.int16) - native.to(torch.int16)).abs()
        assert int(delta.max()) <= 1


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
def test_resize3d_torch_small_all_axis_upscale_matches_native_cpu_interpolate(dtype: torch.dtype) -> None:
    """Small all-axis Tensor upscales avoid a marginal bridge win and retain the native result exactly."""
    if dtype == torch.uint8:
        volume = torch.arange(5 * 7 * 9, dtype=dtype).reshape(1, 5, 7, 9)
    else:
        volume = torch.rand((1, 5, 7, 9), dtype=dtype)
    size = (8, 11, 13)

    result = resize3d(volume, size)
    expected = _torch_linear_reference(volume, size)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32], ids=["uint8", "float32"])
def test_resize3d_torch_nearest_matches_native_cpu_interpolate(dtype: torch.dtype) -> None:
    """CPU CDHW nearest interpolation does not need a float conversion for either supported dtype."""
    if dtype == torch.uint8:
        volume = torch.arange(3 * 5 * 7 * 9, dtype=dtype).reshape(3, 5, 7, 9)
    else:
        volume = torch.rand((3, 5, 7, 9), dtype=dtype)
    size = (3, 5, 12)

    result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)
    expected = torch_f.interpolate(volume.unsqueeze(0), size=size, mode="nearest").squeeze(0)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_resize3d_torch_noncontiguous_and_identity_contract() -> None:
    """Non-contiguous CPU CDHW Tensor views work, while identity preserves the original Tensor."""
    volume = torch.rand((5, 5, 7, 9), dtype=torch.float32)[:, :, :, ::2]

    result = resize3d(volume, (4, 6, 8))

    assert result.shape == (5, 4, 6, 8)
    identity_size = (volume.shape[1], volume.shape[2], volume.shape[3])
    assert resize3d(volume, identity_size) is volume
