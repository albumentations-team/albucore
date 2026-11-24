# ruff: noqa: S101
"""NumPy ``resize3d`` nearest routes that gather with one ``np.take`` per axis."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import albucore.geometric as geometric
from albucore import resize3d

_DTYPES = [np.uint8, np.float16, np.float32]
_DTYPE_IDS = ["uint8", "float16", "float32"]


def _volume(shape: tuple[int, ...], dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(20261124)
    if dtype is np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32).astype(dtype)


def _opencv_nearest(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Reference: the three OpenCV nearest axis passes the take route replaces."""
    expected = geometric._resize3d_numpy_axis_packing(volume.astype(np.float32), size, cv2.INTER_NEAREST, antialias=False)
    return expected.astype(volume.dtype)


def _spy_take(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    calls: list[tuple[int, int, int]] = []
    original = geometric._resize3d_numpy_nearest_take

    def spy(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
        calls.append(size)
        return original(volume, size)

    monkeypatch.setattr(geometric, "_resize3d_numpy_nearest_take", spy)
    return calls


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("channels", [1, 3, 9], ids=["c1", "c3", "c9"])
@pytest.mark.parametrize("size", [(4, 7, 9), (11, 7, 9), (3, 7, 9), (8, 7, 9)], ids=["z_down", "z_up", "z_odd", "z_up2"])
def test_depth_only_nearest_takes_for_every_dtype_and_channel_count(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    channels: int,
    size: tuple[int, int, int],
) -> None:
    volume = _volume((6, 7, 9, channels), dtype)
    calls = _spy_take(monkeypatch)

    result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)

    assert calls == [size]
    assert result.dtype == volume.dtype
    np.testing.assert_array_equal(result, _opencv_nearest(volume, size))


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize(
    ("channels", "size"),
    [
        *[(channels, size) for channels in (2, 3, 9) for size in ((3, 5, 6), (11, 5, 6), (4, 4, 5))],
        *[(channels, (2, 11, 13)) for channels in (3, 9)],
    ],
    ids=lambda value: str(value),
)
def test_multichannel_depth_and_hw_nearest_takes(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    channels: int,
    size: tuple[int, int, int],
) -> None:
    volume = _volume((6, 8, 9, channels), dtype)
    calls = _spy_take(monkeypatch)

    result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)

    assert calls == [size]
    np.testing.assert_array_equal(result, _opencv_nearest(volume, size))


def test_uint8_all_axis_upscale_takes_for_three_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = _volume((6, 8, 9, 3), np.uint8)
    size = (9, 12, 14)
    calls = _spy_take(monkeypatch)

    result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)

    assert calls == [size]
    np.testing.assert_array_equal(result, _opencv_nearest(volume, size))


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_single_channel_hw_down_takes_only_from_the_size_floor(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
) -> None:
    volume = _volume((6, 8, 9, 1), dtype)
    size = (3, 4, 5)
    expected = _opencv_nearest(volume, size)
    for floor, taken in ((volume.size + 1, False), (volume.size, True)):
        monkeypatch.setattr(geometric, "_RESIZE3D_NEAREST_TAKE_SINGLE_CHANNEL_MIN_ELEMENTS", floor)
        calls = _spy_take(monkeypatch)

        result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)

        assert bool(calls) is taken
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ((6, 8, 9, 1), (9, 12, 14)),
        ((6, 8, 9, 3), (6, 12, 14)),
        ((6, 8, 9, 3), (6, 4, 5)),
        ((6, 8, 9, 3), (9, 12, 5)),
    ],
    ids=["c1_up_all", "xy_up", "xy_down", "mixed_hw"],
)
def test_regions_without_a_measured_win_keep_opencv(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int, int],
    size: tuple[int, int, int],
) -> None:
    volume = _volume(shape, np.float32)
    calls = _spy_take(monkeypatch)

    result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)

    assert not calls
    np.testing.assert_array_equal(result, _opencv_nearest(volume, size))


def test_float32_all_axis_upscale_needs_four_channels_or_many_packed_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    size = (9, 12, 14)
    small = _volume((6, 8, 9, 3), np.float32)
    four = _volume((6, 8, 9, 4), np.float32)
    deep = _volume((50, 4, 5, 3), np.float32)
    calls = _spy_take(monkeypatch)

    results = [resize3d(volume, size if volume is not deep else (60, 6, 8), cv2.INTER_NEAREST) for volume in (small, four, deep)]

    assert calls == [size, (60, 6, 8)]
    np.testing.assert_array_equal(results[1], _opencv_nearest(four, size))
    np.testing.assert_array_equal(results[2], _opencv_nearest(deep, (60, 6, 8)))


def test_antialias_disables_the_nearest_take(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = _volume((6, 8, 9, 3), np.uint8)
    calls = _spy_take(monkeypatch)

    resize3d(volume, (3, 4, 5), interpolation=cv2.INTER_NEAREST, antialias=True)

    assert not calls


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_nearest_take_accepts_strided_readonly_and_fortran_inputs(dtype: type[np.generic]) -> None:
    base = _volume((12, 16, 18, 3), dtype)
    strided = base[::-1, :, ::2]
    readonly = base.copy()
    readonly.flags.writeable = False
    fortran = np.asfortranarray(base)
    for volume in (strided, readonly, fortran):
        for size in ((5, volume.shape[1], volume.shape[2]), (5, 7, 4), (17, 21, 13)):
            result = resize3d(volume, size, interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(result, _opencv_nearest(volume, size))


def test_nearest_take_preserves_nan_and_negative_zero() -> None:
    volume = _volume((6, 8, 9, 3), np.float32)
    volume[2, 3, 4] = np.nan
    volume[1, 0, 0] = -0.0

    result = resize3d(volume, (12, 8, 9), interpolation=cv2.INTER_NEAREST)

    expected = _opencv_nearest(volume, (12, 8, 9))
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_axis_beyond_float32_integer_range_keeps_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_take(monkeypatch)
    depth = (1 << 24) + 4
    volume = (np.arange(depth, dtype=np.uint32) % 251).astype(np.uint8).reshape(depth, 1, 1, 1)
    result = resize3d(volume, (depth + 1, 1, 1), interpolation=cv2.INTER_NEAREST)
    assert not calls
    assert result.shape == (depth + 1, 1, 1, 1)
    assert result[-1, 0, 0, 0] == volume[-1, 0, 0, 0]
