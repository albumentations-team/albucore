"""Tests for the float16 NumPy and Tensor median_blur routes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from albucore import median_blur, ops_misc


def _float16_image(channels: int, pattern: str, seed: int = 0) -> np.ndarray:
    values = np.random.default_rng(7_001 + channels + seed).random((19, 23, channels), dtype=np.float32)
    if pattern == "negative":
        values = values * 2 - 1
    elif pattern == "nan":
        values[3, 4, 0] = np.nan
    elif pattern == "infinity":
        values[2:4, 5:7, :] = np.inf
        values[7, 8, :] = 0.0
    elif pattern == "negative_zero":
        values[5, 6, :] = -0.0
    return values.astype(np.float16)


def _bits(array: np.ndarray) -> np.ndarray:
    return array.view(np.uint16 if array.dtype == np.float16 else np.uint32)


class _MedianSpy:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.dtypes: list[np.dtype] = []
        self.original = cv2.medianBlur
        monkeypatch.setattr(cv2, "medianBlur", self)

    def __call__(self, src: np.ndarray, size: int, dst: np.ndarray | None = None) -> np.ndarray:
        self.shapes.append(src.shape)
        self.dtypes.append(src.dtype)
        return self.original(src, size, dst=dst)

    @property
    def used_planes(self) -> bool:
        return any(len(shape) == 2 for shape in self.shapes)


@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("pattern", ["positive", "infinity", "negative", "nan", "negative_zero"])
def test_float16_numpy_uses_u16_view_only_for_nonnegative_data(
    monkeypatch: pytest.MonkeyPatch, channels: int, kernel_size: int, pattern: str
) -> None:
    image = _float16_image(channels, pattern)
    before = image.copy()
    spy = _MedianSpy(monkeypatch)

    result = median_blur(image, kernel_size)

    expected = cv2.medianBlur(image.astype(np.float32), kernel_size).reshape(image.shape).astype(np.float16)
    np.testing.assert_array_equal(_bits(result), _bits(expected))
    np.testing.assert_array_equal(_bits(image), _bits(before))
    assert result.shape == image.shape
    assert (np.dtype(np.uint16) in spy.dtypes) is (pattern in {"positive", "infinity"})


@pytest.mark.parametrize("platform_name", ["linux", "macos"])
@pytest.mark.parametrize("pattern", ["positive", "negative", "nan", "infinity", "negative_zero"])
@pytest.mark.parametrize("kernel_size", [3, 5, 7])
@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_tensor_planes_match_channel_last_route(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    pattern: str,
    kernel_size: int,
    layout: str,
    channels: int,
    dtype: type[np.floating],
) -> None:
    monkeypatch.setattr(ops_misc, "_LINUX_X86_64", platform_name == "linux")
    batch = np.stack([_float16_image(channels, pattern, seed) for seed in (0, 1)]).astype(dtype)
    source = batch if layout == "NCHW" else batch[:1]
    channel_first = torch.from_numpy(source).permute(0, 3, 1, 2).contiguous()
    tensor = channel_first if layout == "NCHW" else channel_first[0]
    before = tensor.clone()
    expected = np.stack([median_blur(image.astype(np.float32), kernel_size).astype(dtype) for image in source])
    spy = _MedianSpy(monkeypatch)

    result = median_blur(tensor, kernel_size, layout=layout)

    actual = result.permute(0, 2, 3, 1).numpy() if layout == "NCHW" else result.permute(1, 2, 0).numpy()[None]
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert result.dtype == tensor.dtype
    assert result.data_ptr() != tensor.data_ptr()
    np.testing.assert_array_equal(_bits(tensor.numpy()), _bits(before.numpy()))
    if kernel_size in (3, 5):
        assert spy.used_planes is (channels >= 3)


@pytest.mark.parametrize("platform_name", ["linux", "macos"])
def test_float16_tensor_planes_size_gate_applies_only_on_linux(
    monkeypatch: pytest.MonkeyPatch, platform_name: str
) -> None:
    monkeypatch.setattr(ops_misc, "_LINUX_X86_64", platform_name == "linux")
    monkeypatch.setattr(ops_misc, "_LINUX_MEDIAN_BLUR_FLOAT16_TENSOR_PLANES_MAX_ELEMENTS", 100)
    tensor = torch.from_numpy(_float16_image(3, "positive")).permute(2, 0, 1).contiguous()
    spy = _MedianSpy(monkeypatch)

    median_blur(tensor, 3, layout="CHW")

    assert spy.used_planes is (platform_name == "macos")


def test_float32_non_contiguous_tensor_keeps_channel_last_route(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.random.default_rng(3).random((19, 23, 3), dtype=np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    expected = cv2.medianBlur(image, 3)
    spy = _MedianSpy(monkeypatch)

    result = median_blur(tensor, 3, layout="CHW")

    np.testing.assert_array_equal(result.permute(1, 2, 0).numpy(), expected)
    assert not spy.used_planes
