# ruff: noqa: S101
"""NumPy linear ``resize3d`` routes selected by per-platform benchmark thresholds."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f

import albucore.geometric as geometric
from albucore import resize3d

_LINEAR = cv2.INTER_LINEAR


def _volume(shape: tuple[int, ...], dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(20261124)
    if dtype is np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32).astype(dtype)


def _float_reference(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Float32 three-pass linear resize without the final uint8 rounding."""
    result = volume.astype(np.float32)
    for axis, output_size in enumerate(size):
        result = geometric._resize3d_numpy_linear_axis(result, axis, output_size)
    return np.asarray(result)


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[int, ...]]:
    calls: list[tuple[int, ...]] = []
    original: Callable[..., np.ndarray] = getattr(geometric, name)

    def spy(volume: np.ndarray, size: tuple[int, int, int], *args: object) -> np.ndarray:
        calls.append(size)
        return original(volume, size, *args)

    monkeypatch.setattr(geometric, name, spy)
    return calls


def _assert_uint8_close_to_float(result: np.ndarray, volume: np.ndarray, size: tuple[int, int, int], bound: int) -> None:
    expected = np.clip(np.floor(_float_reference(volume, size) + 0.5), 0, 255)
    assert result.dtype == np.uint8
    assert np.max(np.abs(result.astype(np.float64) - expected)) <= bound


_U8_SLICE_CASES = [
    pytest.param((6, 24, 40, 3), (6, 36, 60), 1, id="c3_xy_up"),
    pytest.param((6, 24, 40, 3), (6, 36, 20), 1, id="c3_xy_up_down"),
    pytest.param((6, 24, 40, 3), (9, 12, 20), 2, id="c3_depth_up_hw_down"),
    pytest.param((8, 24, 40, 3), (4, 12, 20), 2, id="c3_down_all"),
    pytest.param((8, 24, 40, 3), (6, 17, 28), 2, id="c3_odd_down"),
    pytest.param((8, 24, 40, 1), (6, 17, 28), 2, id="c1_odd_down"),
    pytest.param((6, 24, 40, 2), (6, 36, 60), 1, id="c2_xy_up"),
    pytest.param((6, 24, 40, 9), (9, 12, 20), 2, id="c9_depth_up_hw_down"),
    # One- and four-channel OpenCV planes: documented uint8 tolerance of four levels versus the float reference.
    pytest.param((6, 24, 40, 1), (6, 36, 60), 4, id="c1_xy_up"),
    pytest.param((6, 24, 40, 1), (6, 36, 20), 4, id="c1_xy_up_down"),
    pytest.param((6, 24, 40, 1), (9, 12, 20), 4, id="c1_depth_up_hw_down"),
    pytest.param((6, 24, 40, 4), (6, 36, 60), 4, id="c4_xy_up"),
    pytest.param((6, 24, 40, 4), (6, 36, 20), 4, id="c4_xy_up_down"),
    pytest.param((6, 24, 40, 4), (9, 12, 20), 4, id="c4_depth_up_hw_down"),
]


@pytest.mark.parametrize(("shape", "size", "bound"), _U8_SLICE_CASES)
def test_uint8_linear_takes_the_per_slice_route(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int, int],
    size: tuple[int, int, int],
    bound: int,
) -> None:
    for name in (
        "_RESIZE3D_UINT8_SLICE_MIN_PLANE_ELEMENTS",
        "_RESIZE3D_UINT8_SLICE_SINGLE_CHANNEL_MIN_PLANE_ELEMENTS",
        "_MACOS_RESIZE3D_UINT8_SLICE_DOWN_MIN_PLANE_ELEMENTS",
        "_LINUX_RESIZE3D_UINT8_SLICE_DOWN_MIN_PLANE_ELEMENTS",
        "_LINUX_RESIZE3D_UINT8_SLICE_SINGLE_CHANNEL_GROW_MIN_PLANE_ELEMENTS",
    ):
        monkeypatch.setattr(geometric, name, 1)
    volume = _volume(shape, np.uint8)
    calls = _spy(monkeypatch, "_resize3d_numpy_per_slice")

    result = resize3d(volume, size)

    assert calls == [size]
    _assert_uint8_close_to_float(result, volume, size, bound)


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        pytest.param((6, 24, 40, 4), (4, 12, 20), id="c4_down_all"),
        pytest.param((6, 24, 40, 5), (4, 12, 20), id="c5_down_all"),
        pytest.param((6, 24, 40, 2), (4, 12, 20), id="c2_down_all"),
        pytest.param((6, 24, 40, 3), (3, 36, 60), id="c3_depth_down_hw_up"),
        pytest.param((6, 24, 40, 3), (9, 36, 60), id="c3_up_all"),
    ],
)
def test_uint8_linear_regions_without_a_measured_win_skip_the_new_route(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int, int],
    size: tuple[int, int, int],
) -> None:
    monkeypatch.setattr(geometric, "_RESIZE3D_UINT8_SLICE_MIN_PLANE_ELEMENTS", 1)
    monkeypatch.setattr(geometric, "_RESIZE3D_UINT8_SLICE_SINGLE_CHANNEL_MIN_PLANE_ELEMENTS", 1)
    volume = _volume(shape, np.uint8)

    assert not geometric._can_resize3d_numpy_uint8_linear_per_slice(volume, size, _LINEAR, antialias=False)


@pytest.mark.parametrize("channels", [1, 3], ids=["c1", "c3"])
def test_uint8_hw_only_plane_floor_depends_on_channel_count(monkeypatch: pytest.MonkeyPatch, channels: int) -> None:
    monkeypatch.setattr(geometric, "_LINUX_X86_64", False)

    def routed(plane_height: int) -> bool:
        volume = _volume((4, plane_height, 40, channels), np.uint8)
        return geometric._can_resize3d_numpy_uint8_linear_per_slice(
            volume, (4, plane_height + 8, 60), _LINEAR, antialias=False
        )

    floor = 5_120 if channels == 1 else 3_840
    assert routed(floor // (40 * channels))
    assert not routed(floor // (40 * channels) - 4)


@pytest.mark.parametrize("channels", [1, 4], ids=["c1", "c4"])
def test_uint8_one_and_four_channel_planes_stay_within_four_levels_on_random_shapes(channels: int) -> None:
    rng = np.random.default_rng(channels)
    worst = 0
    for _ in range(60):
        height, width = int(rng.integers(40, 90)), int(rng.integers(40, 90))
        depth = int(rng.integers(2, 6))
        volume = rng.integers(0, 256, size=(depth, height, width, channels), dtype=np.uint8)
        new_height = int(rng.integers(height // 2, height * 2))
        new_width = int(rng.integers(width // 2, width * 2))
        size = (depth, new_height, new_width)
        if not geometric._can_resize3d_numpy_uint8_linear_per_slice(volume, size, _LINEAR, antialias=False):
            continue
        expected = np.clip(np.floor(_float_reference(volume, size) + 0.5), 0, 255)
        worst = max(worst, int(np.max(np.abs(resize3d(volume, size).astype(np.float64) - expected))))
    assert worst <= 4


def test_uint8_linear_antialias_and_other_dtypes_skip_the_new_route() -> None:
    volume = _volume((6, 24, 40, 3), np.uint8)

    assert not geometric._can_resize3d_numpy_uint8_linear_per_slice(volume, (6, 36, 60), _LINEAR, antialias=True)
    assert not geometric._can_resize3d_numpy_uint8_linear_per_slice(volume, (6, 36, 60), cv2.INTER_NEAREST, antialias=False)
    assert not geometric._can_resize3d_numpy_uint8_linear_per_slice(
        volume.astype(np.float32),
        (6, 36, 60),
        _LINEAR,
        antialias=False,
    )


@pytest.mark.parametrize("platform", ["linux", "macos"])
def test_uint8_all_axis_shrink_plane_floor_is_per_platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    linux = platform == "linux"
    monkeypatch.setattr(geometric, "_LINUX_X86_64", linux)
    small = _volume((4, 32, 40, 3), np.uint8)  # plane 3,840
    mid = _volume((4, 64, 80, 3), np.uint8)  # plane 15,360

    def routed(volume: np.ndarray, size: tuple[int, int, int]) -> bool:
        return geometric._can_resize3d_numpy_uint8_linear_per_slice(volume, size, _LINEAR, antialias=False)

    assert routed(small, (2, 16, 20)) is (not linux)
    assert routed(mid, (2, 32, 40))
    assert routed(small, (4, 48, 20))


@pytest.mark.parametrize("platform", ["linux", "macos"])
def test_uint8_single_channel_hw_growth_plane_floor_is_per_platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Linux x86-64 measured the C=1 per-plane growth route 0.75-0.95x of the fallback at 5,120 elements."""
    linux = platform == "linux"
    monkeypatch.setattr(geometric, "_LINUX_X86_64", linux)

    def routed(shape: tuple[int, ...], size: tuple[int, int, int]) -> bool:
        return geometric._can_resize3d_numpy_uint8_linear_per_slice(_volume(shape, np.uint8), size, _LINEAR, antialias=False)

    assert routed((4, 64, 80, 1), (4, 128, 160)) is (not linux)  # plane 5,120 grows
    assert routed((4, 64, 80, 1), (4, 128, 40)) is (not linux)  # mixed growth and shrink
    assert routed((4, 64, 160, 1), (4, 128, 320))  # plane 10,240
    assert routed((4, 64, 80, 1), (8, 32, 40))  # H/W shrink with depth growth keeps the 5,120 floor
    assert routed((4, 64, 80, 3), (4, 128, 160))  # other channel counts are untouched


def test_uint8_single_channel_down_all_has_no_element_cap() -> None:
    volume = _volume((64, 128, 160, 1), np.uint8)
    assert volume.size > geometric._RESIZE3D_NUMPY_PER_SLICE_MAX_ELEMENTS
    size = (32, 64, 80)

    result = resize3d(volume, size)

    assert geometric._can_resize3d_numpy_uint8_linear_per_slice(volume, size, _LINEAR, antialias=False)
    _assert_uint8_close_to_float(result, volume, size, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("channels", [1, 3, 9], ids=["c1", "c3", "c9"])
@pytest.mark.parametrize("depth", [4, 7, 18], ids=["down", "odd", "up"])
def test_depth_only_float_linear_uses_the_three_pass_route(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    channels: int,
    depth: int,
) -> None:
    volume = _volume((9, 10, 12, channels), dtype)
    size = (depth, 10, 12)
    calls = _spy(monkeypatch, "_resize3d_numpy_linear_three_pass")

    result = resize3d(volume, size)

    assert len(calls) == 1
    reference = geometric._resize3d_numpy_axis_packing(volume.astype(np.float32), size, _LINEAR, antialias=False)
    atol = 2e-3 if dtype is np.float16 else 2e-5
    np.testing.assert_allclose(result.astype(np.float32), reference, rtol=2e-5, atol=atol)
    assert result.dtype == volume.dtype


def test_depth_only_uint8_linear_keeps_its_route(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch, "_resize3d_numpy_linear_three_pass")

    resize3d(_volume((9, 10, 12, 3), np.uint8), (4, 10, 12))

    assert not calls


@pytest.mark.parametrize("platform", ["linux", "macos"])
@pytest.mark.parametrize("channels", [1, 3], ids=["c1", "c3"])
@pytest.mark.parametrize("hw", [(5, 6), (20, 24), (10, 6)], ids=["hw_down", "hw_up", "hw_mixed"])
def test_float32_depth_one_with_hw_change_uses_torch_at_every_size(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    channels: int,
    hw: tuple[int, int],
) -> None:
    linux = platform == "linux"
    monkeypatch.setattr(geometric, "_LINUX_X86_64", linux)
    volume = _volume((3, 10, 12, channels), np.float32)
    size = (1, *hw)
    assert volume.size < geometric._RESIZE3D_DEPTH_ONE_MIN_ELEMENTS
    calls = _spy(monkeypatch, "_resize3d_numpy_torch_cpu")

    result = resize3d(volume, size)

    all_hw_up = hw[0] > 10 and hw[1] > 12
    assert bool(calls) is not (linux and channels == 1 and all_hw_up)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).unsqueeze(0)
    expected = torch_f.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    np.testing.assert_allclose(result, expected.squeeze(0).permute(1, 2, 3, 0).numpy(), rtol=2e-4, atol=3e-5)


def test_float32_depth_one_without_hw_change_keeps_its_route(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy(monkeypatch, "_resize3d_numpy_torch_cpu")

    resize3d(_volume((3, 10, 12, 3), np.float32), (1, 10, 12))

    assert not calls
