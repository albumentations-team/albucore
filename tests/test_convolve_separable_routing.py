"""Tests for float32 convolve routing between filter2D and sepFilter2D."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

import albucore.filtering as filtering
from albucore import convolve

_WIDTH = 64


def _gaussian_kernel(size: int) -> np.ndarray:
    vector = cv2.getGaussianKernel(size, 0, cv2.CV_32F).reshape(-1)
    return np.outer(vector, vector).astype(np.float32)


def _image(channels: int, elements: int, *, below: bool = False) -> np.ndarray:
    rows = math.ceil(elements / (_WIDTH * channels)) - (1 if below else 0)
    return np.random.default_rng(elements + channels).random((rows, _WIDTH, channels), dtype=np.float32)


class _SepSpy:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        self.original = cv2.sepFilter2D
        monkeypatch.setattr(cv2, "sepFilter2D", self)

    def __call__(self, *args: object, **kwargs: object) -> np.ndarray:
        self.calls += 1
        return self.original(*args, **kwargs)


def _set_platform(monkeypatch: pytest.MonkeyPatch, platform_name: str) -> None:
    monkeypatch.setattr(filtering, "_LINUX_X86_64", platform_name == "linux")
    monkeypatch.setattr(filtering, "_MACOS_ARM64", platform_name == "macos")


def _expected(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.clip(cv2.filter2D(image, -1, kernel), 0, 1).reshape(image.shape)


_LINUX_GATES = {9: 46_080, 11: 20_480, 13: 20_480, 15: 12_288, 17: 12_288, 19: 12_288, 21: 5_120, 31: 5_120}


@pytest.mark.parametrize("platform_name", ["linux", "macos"])
@pytest.mark.parametrize("ksize", [9, 11, 13, 15, 17, 19, 21, 31])
@pytest.mark.parametrize("channels", [1, 3, 9])
def test_separable_float32_blur_uses_platform_gate(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, ksize: int, channels: int
) -> None:
    _set_platform(monkeypatch, platform_name)
    gate = _LINUX_GATES[ksize] if platform_name == "linux" else 5_120
    kernel = _gaussian_kernel(ksize)

    for below in (True, False):
        image = _image(channels, gate, below=below)
        image.flags.writeable = False
        spy = _SepSpy(monkeypatch)

        actual = convolve(image, kernel)

        np.testing.assert_allclose(actual, _expected(image, kernel), rtol=1e-5, atol=1e-5)
        assert actual.dtype == np.float32
        assert actual.shape == image.shape
        assert not np.shares_memory(actual, image)
        assert (spy.calls == 1) is (not below)
        monkeypatch.setattr(cv2, "sepFilter2D", spy.original)


@pytest.mark.parametrize("platform_name", ["linux", "macos"])
@pytest.mark.parametrize("case", ["k3", "k5", "k7_single_channel", "k7_below_table", "even", "nonseparable", "negative", "unnormalized"])
def test_convolve_keeps_filter2d_outside_separable_region(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, case: str
) -> None:
    _set_platform(monkeypatch, platform_name)
    channels = 3
    elements = 2_000_000
    if case in {"k3", "k5"}:
        kernel = _gaussian_kernel(int(case[1:]))
    elif case == "k7_single_channel":
        kernel, channels = _gaussian_kernel(7), 1
    elif case == "k7_below_table":
        kernel, elements = _gaussian_kernel(7), 100_000
    elif case == "even":
        kernel = _gaussian_kernel(10)
    elif case == "nonseparable":
        kernel = np.random.default_rng(3).random((11, 11), dtype=np.float32)
        kernel /= kernel.sum()
    elif case == "negative":
        kernel = _gaussian_kernel(11)
        kernel[0, 0] = -kernel[0, 0]
    else:
        kernel = _gaussian_kernel(11) * np.float32(1.01)
    image = _image(channels, elements)
    spy = _SepSpy(monkeypatch)

    actual = convolve(image, kernel)

    np.testing.assert_array_equal(actual, _expected(image, kernel))
    assert spy.calls == 0


def test_convolve_separable_gate_is_skipped_on_unmeasured_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_platform(monkeypatch, "other")
    image = _image(3, 2_000_000)
    kernel = _gaussian_kernel(15)
    spy = _SepSpy(monkeypatch)

    actual = convolve(image, kernel)

    np.testing.assert_array_equal(actual, _expected(image, kernel))
    assert spy.calls == 0


def test_nearly_separable_kernel_with_accumulated_error_stays_on_filter2d(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_platform(monkeypatch, "linux")
    kernel = np.full((31, 31), 1 / 961, dtype=np.float32)
    kernel[0, 0] += np.float32(9e-8)
    kernel[1, 1] -= np.float32(9e-8)
    image = _image(3, 2_000_000)
    spy = _SepSpy(monkeypatch)

    actual = convolve(image, kernel)

    assert filtering._separable_float32_kernel(kernel) is None
    np.testing.assert_array_equal(actual, _expected(image, kernel))
    assert spy.calls == 0


@pytest.mark.parametrize("size", [7, 9, 15, 21, 31])
def test_exactly_separable_kernels_still_factor(size: int) -> None:
    assert filtering._separable_float32_kernel(_gaussian_kernel(size)) is not None
