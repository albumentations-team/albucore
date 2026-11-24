"""Tests for the Torch pad route of Tensor copy_make_border."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

import albucore.geometric as geometric
from albucore import copy_make_border

_PADS = (3, 5, 2, 4)
_BORDERS = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect101": cv2.BORDER_REFLECT_101,
}
_DTYPES = {"uint8": torch.uint8, "float16": torch.float16, "float32": torch.float32}


def _tensor(dtype: torch.dtype, layout: str, channels: int, shape: tuple[int, int] = (11, 13)) -> torch.Tensor:
    size = (2, channels, *shape) if layout == "NCHW" else (channels, *shape)
    values = torch.from_numpy(np.random.default_rng(7).random(size, dtype=np.float32))
    return (values * 255).to(torch.uint8) if dtype == torch.uint8 else (values * 1.5 - 0.25).to(dtype)


def _reference(
    tensor: torch.Tensor,
    layout: str,
    border_type: int,
    value: float | tuple[float, ...] | None,
    pads: tuple[int, ...] = _PADS,
) -> torch.Tensor:
    images = tensor.numpy() if layout == "NCHW" else tensor.numpy()[None]
    padded = [
        copy_make_border(np.ascontiguousarray(image.transpose(1, 2, 0)), *pads, border_type, value).transpose(2, 0, 1)
        for image in images
    ]
    result = np.stack(padded) if layout == "NCHW" else padded[0]
    return torch.from_numpy(np.ascontiguousarray(result))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16) if tensor.dtype == torch.float16 else tensor.contiguous()


@pytest.fixture
def pad_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = torch.nn.functional.pad

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "pad", spy)
    return calls


def _set_platform(monkeypatch: pytest.MonkeyPatch, platform_name: str) -> None:
    monkeypatch.setattr(geometric, "_LINUX_X86_64", platform_name == "linux")
    monkeypatch.setattr(geometric, "_MACOS_ARM64", platform_name == "macos")


@pytest.mark.parametrize("platform_name", ["linux", "macos"])
@pytest.mark.parametrize("border", ["constant", "replicate", "reflect101"])
@pytest.mark.parametrize("dtype_name", ["uint8", "float16", "float32"])
@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("channels", [1, 3, 9])
def test_tensor_pad_route_matches_opencv_bridge(
    monkeypatch: pytest.MonkeyPatch,
    pad_spy: list[int],
    platform_name: str,
    border: str,
    dtype_name: str,
    layout: str,
    channels: int,
) -> None:
    _set_platform(monkeypatch, platform_name)
    dtype = _DTYPES[dtype_name]
    tensor = _tensor(dtype, layout, channels)
    value = 7.0 if border == "constant" else None
    original = tensor.clone()

    actual = copy_make_border(tensor, *_PADS, _BORDERS[border], value, layout=layout)

    expected = _reference(tensor, layout, _BORDERS[border], value)
    assert actual.dtype == dtype
    assert torch.equal(_bits(actual), _bits(expected))
    assert torch.equal(_bits(tensor), _bits(original))
    expect_route = not (layout == "CHW" and channels == 1)
    if dtype is torch.float16 and border != "constant":
        expect_route = expect_route and (channels == 3 if platform_name == "linux" else channels >= 3)
    assert bool(pad_spy) is expect_route


@pytest.mark.parametrize("dtype_name", ["uint8", "float16", "float32"])
@pytest.mark.parametrize("value", [7, 7.6, 300.5, -4.0, 0.1])
def test_tensor_pad_constant_value_conversion_matches_opencv(
    monkeypatch: pytest.MonkeyPatch, pad_spy: list[int], dtype_name: str, value: float
) -> None:
    _set_platform(monkeypatch, "linux")
    tensor = _tensor(_DTYPES[dtype_name], "CHW", 3)

    actual = copy_make_border(tensor, *_PADS, cv2.BORDER_CONSTANT, value, layout="CHW")

    expected = _reference(tensor, "CHW", cv2.BORDER_CONSTANT, value)
    assert torch.equal(_bits(actual), _bits(expected))
    assert pad_spy


def test_tensor_pad_float16_linux_size_gate_keeps_large_batches_on_opencv(
    monkeypatch: pytest.MonkeyPatch, pad_spy: list[int]
) -> None:
    _set_platform(monkeypatch, "linux")
    monkeypatch.setattr(geometric, "_LINUX_FLOAT16_TENSOR_PAD_MAX_ELEMENTS", 100)
    tensor = _tensor(torch.float16, "NCHW", 3)

    actual = copy_make_border(tensor, *_PADS, cv2.BORDER_REPLICATE, layout="NCHW")

    expected = _reference(tensor, "NCHW", cv2.BORDER_REPLICATE, None)
    assert torch.equal(_bits(actual), _bits(expected))
    assert not pad_spy


@pytest.mark.parametrize("platform_name", ["linux", "macos"])
def test_tensor_pad_reflect_larger_than_image_keeps_opencv(
    monkeypatch: pytest.MonkeyPatch, pad_spy: list[int], platform_name: str
) -> None:
    _set_platform(monkeypatch, platform_name)
    tensor = _tensor(torch.uint8, "CHW", 3, (4, 5))
    pads = (6, 1, 2, 7)

    actual = copy_make_border(tensor, *pads, cv2.BORDER_REFLECT_101, layout="CHW")

    expected = _reference(tensor, "CHW", cv2.BORDER_REFLECT_101, None, pads)
    assert torch.equal(_bits(actual), _bits(expected))
    assert not pad_spy


@pytest.mark.parametrize("case", ["per_channel_value", "reflect", "wrap", "destination", "unsupported_platform"])
def test_tensor_pad_not_used_outside_measured_regions(
    monkeypatch: pytest.MonkeyPatch, pad_spy: list[int], case: str
) -> None:
    _set_platform(monkeypatch, "other" if case == "unsupported_platform" else "linux")
    tensor = _tensor(torch.uint8, "CHW", 3)
    border: int = cv2.BORDER_CONSTANT
    value: float | tuple[float, ...] = 7.0
    dst = None
    if case == "per_channel_value":
        value = (1.0, 2.0, 3.0)
    elif case == "reflect":
        border = cv2.BORDER_REFLECT
    elif case == "wrap":
        border = cv2.BORDER_WRAP
    elif case == "destination":
        dst = torch.empty((11 + 8, 13 + 6, 3), dtype=torch.uint8).permute(2, 0, 1)

    actual = copy_make_border(tensor, *_PADS, border, value, dst=dst, layout="CHW")

    expected = _reference(tensor, "CHW", border, value)
    assert torch.equal(_bits(actual), _bits(expected))
    assert not pad_spy
