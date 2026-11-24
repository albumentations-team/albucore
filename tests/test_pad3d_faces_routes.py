# ruff: noqa: S101
"""``pad3d`` routes that fill only the border faces, with per-platform size floors."""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pytest
import torch

from albucore import pad3d

pad3d_module: ModuleType = importlib.import_module("albucore.pad3d")

_PADDINGS = [(1, 1, 1, 1, 1, 1), (8, 8, 8, 8, 8, 8), (1, 2, 3, 4, 5, 6), (0, 0, 4, 4, 4, 4)]
_PADDING_IDS = ["p1", "p8", "asym", "xy4"]


def _volume(shape: tuple[int, int, int, int], dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(20261124)
    if dtype is np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32).astype(dtype)


def _reference(volume: np.ndarray, padding: tuple[int, int, int, int, int, int], value: float) -> np.ndarray:
    front, back, top, bottom, left, right = padding
    return np.pad(volume, ((front, back), (top, bottom), (left, right), (0, 0)), constant_values=value)


class _Calls:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.faces = 0
        self.torch = 0
        self.full = 0
        for name, attribute in (("_pad3d_numpy_faces", "faces"), ("_pad3d_torch", "torch"), ("_pad3d_numpy", "full")):
            self._wrap(monkeypatch, name, attribute)

    def _wrap(self, monkeypatch: pytest.MonkeyPatch, name: str, attribute: str) -> None:
        original = getattr(pad3d_module, name)

        def wrapper(*args: object, **kwargs: object) -> object:
            setattr(self, attribute, getattr(self, attribute) + 1)
            return original(*args, **kwargs)

        monkeypatch.setattr(pad3d_module, name, wrapper)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("channels", [2, 3, 9], ids=["c2", "c3", "c9"])
@pytest.mark.parametrize("padding", _PADDINGS, ids=_PADDING_IDS)
def test_float_multichannel_faces_apply_to_every_pad_width_from_the_floor(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    channels: int,
    padding: tuple[int, int, int, int, int, int],
) -> None:
    volume = _volume((6, 7, 9, channels), dtype)
    monkeypatch.setattr(pad3d_module, "_FLOAT32_FACES_MIN_ELEMENTS", volume.size)
    monkeypatch.setattr(pad3d_module, "_FLOAT16_FACES_MIN_ELEMENTS", volume.size)
    calls = _Calls(monkeypatch)

    result = pad3d(volume, padding, 0.25)

    assert (calls.faces, calls.full, calls.torch) == (1, 0, 0)
    np.testing.assert_array_equal(result, _reference(volume, padding, 0.25))


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["float32", "float16"])
def test_float_multichannel_below_the_floor_and_other_dtypes_keep_the_full_fill(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
) -> None:
    volume = _volume((6, 7, 9, 3), dtype)
    monkeypatch.setattr(pad3d_module, "_FLOAT32_FACES_MIN_ELEMENTS", volume.size + 1)
    monkeypatch.setattr(pad3d_module, "_FLOAT16_FACES_MIN_ELEMENTS", volume.size + 1)
    calls = _Calls(monkeypatch)

    pad3d(volume, (8, 8, 8, 8, 8, 8), 0.25)
    pad3d(_volume((6, 7, 9, 3), np.uint8), (8, 8, 8, 8, 8, 8), 0.25)

    assert (calls.faces, calls.full) == (0, 2)


def test_float_faces_skip_non_contiguous_volumes_and_tuple_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = _volume((6, 7, 18, 3), np.float32)[:, :, ::2]
    monkeypatch.setattr(pad3d_module, "_FLOAT32_FACES_MIN_ELEMENTS", 1)
    calls = _Calls(monkeypatch)

    result = pad3d(volume, (1, 2, 3, 4, 5, 6), 0.25)
    tuple_result = pad3d(np.ascontiguousarray(volume), (1, 2, 3, 4, 5, 6), (1.0, 2.0))

    assert calls.faces == 0
    np.testing.assert_array_equal(result, _reference(volume, (1, 2, 3, 4, 5, 6), 0.25))
    np.testing.assert_array_equal(tuple_result, np.pad(volume, ((1, 2), (3, 4), (5, 6), (0, 0)), constant_values=(1.0, 2.0)))


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.uint8], ids=["float32", "float16", "uint8"])
@pytest.mark.parametrize("platform", ["linux", "macos"])
def test_single_channel_contiguous_routes_by_platform(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    platform: str,
) -> None:
    linux = platform == "linux"
    volume = _volume((6, 7, 9, 1), dtype)
    monkeypatch.setattr(pad3d_module, "_LINUX_X86_64", linux)
    monkeypatch.setattr(pad3d_module, "_TORCH_MIN_ELEMENTS", volume.size)
    monkeypatch.setattr(pad3d_module, "_LINUX_TORCH_SINGLE_CHANNEL_MIN_ELEMENTS", volume.size + 1)
    calls = _Calls(monkeypatch)

    result = pad3d(volume, (8, 8, 8, 8, 8, 8), 0.25)

    expected_faces = linux and dtype is not np.uint8
    expected_full = linux and dtype is np.uint8
    expected_torch = not linux
    assert (calls.faces, calls.full, calls.torch) == (int(expected_faces), int(expected_full), int(expected_torch))
    np.testing.assert_array_equal(result, _reference(volume, (8, 8, 8, 8, 8, 8), 0.25))


@pytest.mark.parametrize("dtype", [np.float32, np.uint8], ids=["float32", "uint8"])
def test_single_channel_linux_uses_torch_from_the_linux_floor(monkeypatch: pytest.MonkeyPatch, dtype: type[np.generic]) -> None:
    volume = _volume((6, 7, 9, 1), dtype)
    monkeypatch.setattr(pad3d_module, "_LINUX_X86_64", True)
    monkeypatch.setattr(pad3d_module, "_TORCH_MIN_ELEMENTS", volume.size)
    monkeypatch.setattr(pad3d_module, "_LINUX_TORCH_SINGLE_CHANNEL_MIN_ELEMENTS", volume.size)
    calls = _Calls(monkeypatch)

    result = pad3d(volume, (8, 8, 8, 8, 8, 8), 0.25)

    assert (calls.faces, calls.full, calls.torch) == (0, 0, 1)
    np.testing.assert_array_equal(result, _reference(volume, (8, 8, 8, 8, 8, 8), 0.25))


def test_single_channel_strided_volume_keeps_torch_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = _volume((6, 7, 18, 1), np.float32)[:, :, ::2]
    monkeypatch.setattr(pad3d_module, "_LINUX_X86_64", True)
    monkeypatch.setattr(pad3d_module, "_TORCH_MIN_ELEMENTS", 1)
    calls = _Calls(monkeypatch)

    result = pad3d(volume, (1, 2, 3, 4, 5, 6), 0.25)

    assert (calls.faces, calls.torch) == (0, 1)
    np.testing.assert_array_equal(result, _reference(volume, (1, 2, 3, 4, 5, 6), 0.25))


def _channel_last_tensor(shape: tuple[int, int, int, int], dtype: type[np.generic]) -> tuple[torch.Tensor, np.ndarray]:
    volume = _volume(shape, dtype)
    return torch.from_numpy(volume).permute(3, 0, 1, 2), volume


@pytest.mark.parametrize("padding", _PADDINGS, ids=_PADDING_IDS)
def test_channel_last_float32_tensor_bridges_to_faces_from_the_floor(
    monkeypatch: pytest.MonkeyPatch,
    padding: tuple[int, int, int, int, int, int],
) -> None:
    tensor, volume = _channel_last_tensor((6, 7, 9, 3), np.float32)
    monkeypatch.setattr(pad3d_module, "_FLOAT32_FACES_MIN_ELEMENTS", volume.size)
    calls = _Calls(monkeypatch)

    result = pad3d(tensor, padding, 0.25)

    assert (calls.faces, calls.full, calls.torch) == (1, 0, 0)
    assert result.dtype == torch.float32
    np.testing.assert_array_equal(result.permute(1, 2, 3, 0).numpy(), _reference(volume, padding, 0.25))


@pytest.mark.parametrize("dtype", [np.float16, np.uint8], ids=["float16", "uint8"])
def test_channel_last_tensor_other_dtypes_and_small_float32_keep_the_full_fill(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
) -> None:
    monkeypatch.setattr(pad3d_module, "_FLOAT32_FACES_MIN_ELEMENTS", 6 * 7 * 9 * 3 + 1)
    monkeypatch.setattr(pad3d_module, "_FLOAT16_FACES_MIN_ELEMENTS", 1)
    calls = _Calls(monkeypatch)

    other, _ = _channel_last_tensor((6, 7, 9, 3), dtype)
    small, _ = _channel_last_tensor((6, 7, 9, 3), np.float32)
    pad3d(other, (1, 2, 3, 4, 5, 6), 0.25)
    pad3d(small, (1, 2, 3, 4, 5, 6), 0.25)

    assert (calls.faces, calls.full) == (0, 2)


def test_contiguous_tensor_keeps_native_torch_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    volume = torch.from_numpy(_volume((3, 6, 7, 9), np.float32))
    monkeypatch.setattr(pad3d_module, "_FLOAT32_FACES_MIN_ELEMENTS", 1)
    calls = _Calls(monkeypatch)

    pad3d(volume, (1, 2, 3, 4, 5, 6), 0.25)

    assert (calls.faces, calls.torch) == (0, 1)
