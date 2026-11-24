# ruff: noqa: S101
"""Route and accuracy checks for global float16 mean_std and std."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

import albucore.stats as stats_mod
from albucore import mean_std, std

EPS = 1e-4


def _half_image(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random(shape, dtype=np.float32) * 0.5 + 0.25).astype(np.float16)


def _reference(image: np.ndarray) -> tuple[float, float]:
    return float(np.mean(image, dtype=np.float64)), float(np.std(image, dtype=np.float64)) + EPS


def _spy_widened(monkeypatch: pytest.MonkeyPatch) -> list[np.dtype]:
    dtypes: list[np.dtype] = []
    original: Callable[[np.ndarray, float], tuple[float, float] | None] = stats_mod._mean_std_global_float32_opencv

    def spy(arr: np.ndarray, eps: float) -> tuple[float, float] | None:
        dtypes.append(arr.dtype)
        return original(arr, eps)

    monkeypatch.setattr(stats_mod, "_mean_std_global_float32_opencv", spy)
    return dtypes


@pytest.mark.parametrize("shape", [(64, 64, 1), (64, 80, 3), (32, 32, 9), (2, 40, 52, 3), (4, 32, 32, 1)])
def test_numpy_global_stats_widen_to_float32_from_4096_elements(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    image = _half_image(shape)
    assert image.size >= 4_096
    dtypes = _spy_widened(monkeypatch)
    expected_mean, expected_std = _reference(image)

    result_mean, result_std = mean_std(image)
    only_std = std(image)

    assert dtypes == [np.dtype(np.float32)] * 2
    np.testing.assert_allclose(result_mean, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(result_std, expected_std, rtol=1e-12)
    np.testing.assert_allclose(only_std, expected_std, rtol=1e-12)


def test_numpy_global_stats_below_4096_elements_skip_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _half_image((63, 64, 1))
    dtypes = _spy_widened(monkeypatch)
    result_mean, result_std = mean_std(image)
    assert dtypes == []
    np.testing.assert_allclose((result_mean, result_std), _reference(image), rtol=1e-12)


def test_numpy_global_stats_keep_near_constant_and_nonfinite_inputs_exact() -> None:
    constant = np.full((64, 64, 3), 0.575, dtype=np.float16)
    constant[0, 0, 0] = np.float16(0.5752)
    np.testing.assert_allclose(mean_std(constant), _reference(constant), rtol=1e-12)
    with_nan = _half_image((64, 64, 1))
    with_nan[3, 3, 0] = np.nan
    result_mean, result_std = mean_std(with_nan)
    assert np.isnan(result_mean)
    assert np.isnan(result_std)
    subnormal = np.full((64, 64, 1), 2.0**-24, dtype=np.float16)
    subnormal[::2] *= 3
    np.testing.assert_allclose(mean_std(subnormal), _reference(subnormal), rtol=1e-12)


@pytest.mark.parametrize(("macos", "linux"), [(True, False), (False, True)])
@pytest.mark.parametrize(("shape", "layout"), [((3, 64, 80), "CHW"), ((1, 128, 160), "CHW"), ((2, 3, 32, 32), "NCHW")])
def test_tensor_global_std_widens_from_4096_elements_on_both_platforms(
    monkeypatch: pytest.MonkeyPatch, macos: bool, linux: bool, shape: tuple[int, ...], layout: str
) -> None:
    monkeypatch.setattr(stats_mod, "_MACOS_ARM64", macos)
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", linux)
    image = torch.from_numpy(_half_image(shape, seed=1))
    dtypes = _spy_widened(monkeypatch)

    result = std(image, layout=layout)

    assert dtypes == [np.dtype(np.float32)]
    assert result.dtype == torch.float64
    np.testing.assert_allclose(result.item(), _reference(image.numpy())[1], rtol=1e-12)
    kept = std(image, keepdims=True, layout=layout)
    assert kept.shape == (1,) * image.ndim


@pytest.mark.parametrize(
    ("macos", "linux", "elements", "widened"),
    [
        (True, False, 4_096, True),
        (True, False, 61_440, True),
        (False, True, 4_096, True),
        (False, True, 15_360, True),
        (False, True, 20_480, False),
        (False, True, 61_440, False),
    ],
)
def test_tensor_global_mean_std_platform_gate(
    monkeypatch: pytest.MonkeyPatch, *, macos: bool, linux: bool, elements: int, widened: bool
) -> None:
    monkeypatch.setattr(stats_mod, "_MACOS_ARM64", macos)
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", linux)
    image = torch.from_numpy(_half_image((4, elements // 4 // 16, 16), seed=2))
    dtypes = _spy_widened(monkeypatch)

    result_mean, result_std = mean_std(image, layout="CHW")

    assert bool(dtypes) is widened
    expected_mean, expected_std = _reference(image.numpy())
    # The unwidened route is CV_16F, which keeps the established looser tolerance.
    rtol = 1e-12 if widened else 1e-3
    np.testing.assert_allclose(result_mean.item(), expected_mean, rtol=rtol)
    np.testing.assert_allclose(result_std.item(), expected_std, rtol=rtol)


def test_tensor_global_std_near_constant_falls_back_to_float64() -> None:
    image = torch.full((3, 64, 80), 0.575, dtype=torch.float16)
    image[0, 0, 0] = 0.5752
    np.testing.assert_allclose(std(image, layout="CHW").item(), _reference(image.numpy())[1], rtol=1e-12)
