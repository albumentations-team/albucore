# ruff: noqa: S101
"""Tensor ``resize3d`` regions that bridge through the NumPy/OpenCV route."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f

import albucore.geometric as geometric
from albucore import resize3d

_DTYPES = [torch.uint8, torch.float16, torch.float32]
_DTYPE_IDS = ["uint8", "float16", "float32"]


def _tensor(shape: tuple[int, int, int, int], dtype: torch.dtype, *, channel_last: bool) -> torch.Tensor:
    """Create a CDHW Tensor, optionally with interleaved channel-last storage."""
    channels, depth, height, width = shape
    rng = np.random.default_rng(20261124)
    values = rng.random((depth, height, width, channels), dtype=np.float32)
    array = (values * 255).astype(np.uint8) if dtype == torch.uint8 else values.astype(np.float16 if dtype == torch.float16 else np.float32)
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2)
    return tensor if channel_last else tensor.contiguous()


def _native(volume: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    with torch.inference_mode():
        working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
        result = torch_f.interpolate(working.unsqueeze(0), size=size, mode="trilinear", align_corners=False)
        if volume.dtype == torch.uint8:
            result = torch.minimum(result + 0.5, result.new_tensor(255)).to(torch.uint8)
        elif volume.dtype == torch.float16:
            result = result.to(torch.float16)
    return result.squeeze(0)


def _assert_close_to_native(result: torch.Tensor, volume: torch.Tensor, size: tuple[int, int, int]) -> None:
    expected = _native(volume, size)
    assert result.dtype == volume.dtype
    assert tuple(result.shape) == (volume.shape[0], *size)
    if volume.dtype == torch.uint8:
        assert int((result.to(torch.int16) - expected.to(torch.int16)).abs().max()) <= 1
    elif volume.dtype == torch.float16:
        torch.testing.assert_close(result, expected, rtol=0, atol=float(np.finfo(np.float16).eps / 2))
    else:
        torch.testing.assert_close(result, expected, rtol=2e-4, atol=3e-5)


def _spy_bridge(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    calls: list[tuple[int, int, int]] = []
    original = geometric._resize3d_torch_via_numpy

    def spy(volume: torch.Tensor, size: tuple[int, int, int], interpolation: int) -> torch.Tensor:
        calls.append(size)
        return original(volume, size, interpolation)

    monkeypatch.setattr(geometric, "_resize3d_torch_via_numpy", spy)
    return calls


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("channels", [1, 3, 9], ids=["c1", "c3", "c9"])
@pytest.mark.parametrize("layout", ["contiguous", "channel_last"])
def test_xy_growth_bridges(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    channels: int,
    layout: str,
) -> None:
    volume = _tensor((channels, 4, 10, 12), dtype, channel_last=layout == "channel_last")
    size = (4, 15, 18)
    calls = _spy_bridge(monkeypatch)

    result = resize3d(volume, size)

    assert calls == [size]
    _assert_close_to_native(result, volume, size)


def test_channel_last_float_xy_growth_with_many_channels_stops_above_the_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geometric, "_RESIZE3D_TORCH_NUMPY_BRIDGE_CHANNEL_LAST_XY_UP_MAX_ELEMENTS", 100)
    size = (4, 15, 18)
    calls = _spy_bridge(monkeypatch)

    resize3d(_tensor((9, 4, 10, 12), torch.float32, channel_last=True), size)
    resize3d(_tensor((9, 4, 10, 12), torch.float32, channel_last=False), size)
    resize3d(_tensor((9, 4, 10, 12), torch.uint8, channel_last=True), size)

    assert calls == [size, size]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_single_channel_xy_shrink_bridges(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    volume = _tensor((1, 4, 10, 12), dtype, channel_last=False)
    size = (4, 5, 6)
    calls = _spy_bridge(monkeypatch)

    result = resize3d(volume, size)

    assert calls == [size]
    _assert_close_to_native(result, volume, size)


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_multichannel_xy_shrink_bridges_only_for_large_channel_last_storage(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    size = (4, 5, 6)
    contiguous = _tensor((3, 4, 10, 12), dtype, channel_last=False)
    interleaved = _tensor((3, 4, 10, 12), dtype, channel_last=True)
    calls = _spy_bridge(monkeypatch)

    resize3d(contiguous, size)
    resize3d(interleaved, size)
    assert not calls

    monkeypatch.setattr(geometric, "_RESIZE3D_TORCH_NUMPY_BRIDGE_XY_DOWN_MIN_ELEMENTS", interleaved.numel())
    resize3d(contiguous, size)
    result = resize3d(interleaved, size)

    assert calls == [size]
    _assert_close_to_native(result, interleaved, size)


@pytest.mark.parametrize("layout", ["contiguous", "channel_last"])
def test_uint8_depth_only_bridges_from_the_output_size_floor(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    volume = _tensor((3, 6, 10, 12), torch.uint8, channel_last=layout == "channel_last")
    size = (4, 10, 12)
    calls = _spy_bridge(monkeypatch)

    resize3d(volume, size)
    assert not calls

    monkeypatch.setattr(geometric, "_RESIZE3D_TORCH_NUMPY_BRIDGE_UINT8_Z_MIN_OUTPUT_ELEMENTS", 3 * 4 * 10 * 12)
    result = resize3d(volume, size)

    assert calls == [size]
    _assert_close_to_native(result, volume, size)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids=["float16", "float32"])
def test_float_depth_only_stays_native(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    monkeypatch.setattr(geometric, "_RESIZE3D_TORCH_NUMPY_BRIDGE_UINT8_Z_MIN_OUTPUT_ELEMENTS", 1)
    calls = _spy_bridge(monkeypatch)

    resize3d(_tensor((3, 6, 10, 12), dtype, channel_last=False), (4, 10, 12))

    assert not calls


@pytest.mark.parametrize(
    ("dtype", "channels", "route"),
    [
        (torch.uint8, 3, "bridge"),
        (torch.uint8, 2, "bridge"),
        (torch.uint8, 1, "native"),
        (torch.float16, 3, "bridge"),
        (torch.float16, 1, "bridge"),
        (torch.float32, 3, "bridge"),
        (torch.float32, 1, "bridge"),
    ],
    ids=lambda value: str(value),
)
def test_small_all_axis_upscale_bridges_except_single_channel_uint8(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    channels: int,
    route: str,
) -> None:
    volume = _tensor((channels, 3, 8, 10), dtype, channel_last=False)
    size = (5, 11, 13)
    assert channels * np.prod(size) < geometric._RESIZE3D_TORCH_NUMPY_BRIDGE_MIN_OUTPUT_ELEMENTS
    calls = _spy_bridge(monkeypatch)

    result = resize3d(volume, size)

    assert bool(calls) is (route == "bridge")
    _assert_close_to_native(result, volume, size)


@pytest.mark.parametrize(
    "size",
    [(3, 15, 6), (4, 15, 12), (3, 15, 18), (8, 5, 6), (3, 5, 6)],
    ids=["mixed_hw", "w_unchanged", "depth_down_hw_up", "depth_up_hw_down", "depth_down_hw_down"],
)
def test_unmeasured_regions_stay_native(monkeypatch: pytest.MonkeyPatch, size: tuple[int, int, int]) -> None:
    monkeypatch.setattr(geometric, "_RESIZE3D_TORCH_NUMPY_BRIDGE_XY_DOWN_MIN_ELEMENTS", 1)
    monkeypatch.setattr(geometric, "_RESIZE3D_TORCH_NUMPY_BRIDGE_UINT8_Z_MIN_OUTPUT_ELEMENTS", 1)
    volume = _tensor((3, 4, 10, 12), torch.uint8, channel_last=True)
    calls = _spy_bridge(monkeypatch)

    resize3d(volume, size)

    assert not calls


def test_nearest_never_bridges(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_bridge(monkeypatch)

    resize3d(_tensor((1, 4, 10, 12), torch.uint8, channel_last=False), (4, 15, 18), interpolation=cv2.INTER_NEAREST)

    assert not calls
