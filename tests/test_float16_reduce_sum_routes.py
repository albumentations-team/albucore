# ruff: noqa: S101
"""Route checks for small float16 reduce_sum inputs."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

import albucore.stats as stats_mod
from albucore import reduce_sum


def _half_image(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random(shape, dtype=np.float32) * 0.5 + 0.25).astype(np.float16)


@pytest.mark.parametrize("shape", [(8, 8, 9), (32, 32, 9), (64, 64, 9), (128, 160, 9), (2, 32, 32, 9), (4, 64, 64, 9)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_c9_per_channel_sum_uses_widened_opencv_reduce_at_every_size(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], keepdims: bool
) -> None:
    image = _half_image(shape)
    dtypes: list[np.dtype] = []
    original: Callable[[np.ndarray], np.ndarray] = stats_mod._cv2_reduce_sum_per_channel

    def spy(arr: np.ndarray) -> np.ndarray:
        dtypes.append(arr.dtype)
        return original(arr)

    monkeypatch.setattr(stats_mod, "_cv2_reduce_sum_per_channel", spy)
    result = reduce_sum(image, "per_channel", keepdims=keepdims)

    assert dtypes == [np.dtype(np.float32)]
    assert result.dtype == np.float64
    expected = np.sum(image, axis=tuple(range(image.ndim - 1)), dtype=np.float64, keepdims=keepdims)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("shape", "layout", "axis", "uses_torch"),
    [
        ((3, 10, 10), "CHW", None, True),
        ((3, 16, 16), "CHW", "per_channel", True),
        ((9, 40, 40), "CHW", "per_channel", True),
        ((2, 3, 32, 32), "NCHW", "per_channel", True),
        ((4, 3, 16, 16), "NCHW", None, True),
        ((1, 16, 16), "CHW", None, True),
        ((3, 8, 10), "CHW", None, False),
        ((3, 8, 8), "CHW", "per_channel", False),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_tensor_sum_uses_torch_float64_from_256_elements(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    layout: str,
    axis: str | None,
    keepdims: bool,
    *,
    uses_torch: bool,
) -> None:
    image = torch.from_numpy(_half_image(shape, seed=1))
    dispatched = 0
    original = stats_mod.dispatch_tensor_reduction

    def spy(
        tensor: torch.Tensor,
        tensor_layout: str,
        router: Callable[..., Any],
        reduction_axis: str | None,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        nonlocal dispatched
        dispatched += 1
        return original(tensor, tensor_layout, router, reduction_axis, kwargs)

    monkeypatch.setattr(stats_mod, "dispatch_tensor_reduction", spy)
    result = reduce_sum(image, axis, keepdims=keepdims, layout=layout)

    assert dispatched == (0 if uses_torch else 1)
    channel_axis = 0 if layout == "CHW" else 1
    reduced = None if axis is None else tuple(i for i in range(image.ndim) if i != channel_axis)
    expected = np.sum(image.numpy(), axis=reduced, dtype=np.float64, keepdims=keepdims)
    assert result.dtype == torch.float64
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-12)
