# ruff: noqa: S101
"""Route and accuracy checks for float32 channel-first Tensor per-channel mean and mean_std."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

import albucore.stats as stats_mod
from albucore import mean, mean_std

EPS = 1e-4


def _image(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).random(shape, dtype=np.float32) * 3 + 5)


def _spy_dispatch(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = stats_mod.dispatch_tensor_reduction

    def spy(
        tensor: torch.Tensor,
        layout: str,
        router: Callable[..., Any],
        axis: str,
        kwargs: dict[str, Any],
    ) -> Any:
        calls.append(1)
        return original(tensor, layout, router, axis, kwargs)

    monkeypatch.setattr(stats_mod, "dispatch_tensor_reduction", spy)
    return calls


def _reference(image: torch.Tensor, layout: str, *, keepdims: bool) -> tuple[np.ndarray, np.ndarray]:
    channel_axis = 0 if layout == "CHW" else 1
    axes = tuple(axis for axis in range(image.ndim) if axis != channel_axis)
    values = image.numpy().astype(np.float64)
    return values.mean(axis=axes, keepdims=keepdims), values.std(axis=axes, keepdims=keepdims) + EPS


@pytest.mark.parametrize(
    ("shape", "layout", "direct"),
    [
        ((2, 64, 80), "CHW", True),
        ((3, 64, 80), "CHW", True),
        ((4, 32, 32), "CHW", True),
        ((9, 64, 80), "CHW", True),
        ((4, 9, 32, 32), "NCHW", True),
        ((1, 64, 80), "CHW", False),
        ((4, 3, 32, 32), "NCHW", False),
        ((4, 1, 32, 32), "NCHW", False),
        ((4, 4, 32, 32), "NCHW", False),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_float32_tensor_per_channel_mean_route(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], layout: str, keepdims: bool, *, direct: bool
) -> None:
    image = _image(shape)
    calls = _spy_dispatch(monkeypatch)
    result = mean(image, "per_channel", keepdims=keepdims, layout=layout)
    assert bool(calls) is not direct
    assert result.dtype == torch.float64
    # Routes that keep the NumPy path accumulate in float32 pairs and agree to about 1e-10.
    np.testing.assert_allclose(
        result.numpy(), _reference(image, layout, keepdims=keepdims)[0], rtol=1e-12 if direct else 1e-9
    )


def test_float32_tensor_per_channel_mean_skips_non_contiguous_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _image((3, 64, 80)).permute(0, 2, 1)
    calls = _spy_dispatch(monkeypatch)
    result = mean(image, "per_channel", layout="CHW")
    assert calls
    np.testing.assert_allclose(result.numpy(), image.numpy().astype(np.float64).mean(axis=(1, 2)), rtol=1e-12)


@pytest.mark.parametrize(
    ("shape", "layout", "planes"),
    [
        ((2, 64, 80), "CHW", True),
        ((3, 64, 80), "CHW", True),
        ((4, 32, 32), "CHW", True),
        ((1, 64, 80), "CHW", False),
        ((9, 32, 32), "CHW", False),
        ((4, 3, 32, 32), "NCHW", False),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_float32_tensor_per_channel_mean_std_plane_moments_route(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], layout: str, keepdims: bool, *, planes: bool
) -> None:
    image = _image(shape, seed=1)
    calls = 0
    original = stats_mod._mean_std_per_channel_plane_moments

    def spy(arr: np.ndarray, axis: tuple[int, ...], *, keepdims: bool, eps: float) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return original(arr, axis, keepdims=keepdims, eps=eps)

    monkeypatch.setattr(stats_mod, "_mean_std_per_channel_plane_moments", spy)
    result_mean, result_std = mean_std(image, "per_channel", keepdims=keepdims, layout=layout)

    assert bool(calls) is planes
    expected_mean, expected_std = _reference(image, layout, keepdims=keepdims)
    assert result_mean.shape == expected_mean.shape
    np.testing.assert_allclose(result_mean.numpy(), expected_mean, rtol=1e-12)
    np.testing.assert_allclose(result_std.numpy(), expected_std, rtol=1e-6)


@pytest.mark.parametrize("offset", [0.0, 1000.0, 1.0e5])
def test_float32_tensor_per_channel_mean_std_near_constant_channels_stay_accurate(offset: float) -> None:
    rng = np.random.default_rng(3)
    values = offset + rng.standard_normal((3, 64, 80)).astype(np.float32) * np.float32(0.01)
    values[1] = np.float32(offset + 7.0)
    image = torch.from_numpy(values)
    result_mean, result_std = mean_std(image, "per_channel", layout="CHW")
    expected_mean, expected_std = _reference(image, "CHW", keepdims=False)
    np.testing.assert_allclose(result_mean.numpy(), expected_mean, rtol=1e-12)
    np.testing.assert_allclose(result_std.numpy(), expected_std, rtol=1e-3, atol=1e-6)
