# ruff: noqa: S101
"""Route and exactness checks for uint8 sum and mean reductions."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

import albucore.stats as stats_mod
from albucore import mean, reduce_sum
from albucore.torch_backend import TORCH_MIN_ELEMENTS


def _forbid(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        msg = f"{name} must not be reached."
        raise AssertionError(msg)

    monkeypatch.setattr(stats_mod, name, fail)


@pytest.mark.parametrize("shape", [(1024, 520, 1), (2, 360, 400, 3)])
def test_global_uint8_sum_skips_torch_above_old_threshold(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    assert image.size >= TORCH_MIN_ELEMENTS
    _forbid(monkeypatch, "reduce_sum_torch")
    result = reduce_sum(image)
    assert result == np.sum(image, dtype=np.uint64)
    assert type(result) is np.uint64
    kept = reduce_sum(image, keepdims=True)
    assert kept.shape == (1,) * image.ndim
    assert kept.dtype == np.uint64


@pytest.mark.parametrize("channels", [5, 9])
@pytest.mark.parametrize("shape_prefix", [(), (2,)])
def test_per_channel_uint8_up_to_nine_channels_uses_numkong(
    monkeypatch: pytest.MonkeyPatch, channels: int, shape_prefix: tuple[int, ...]
) -> None:
    image = np.random.default_rng(1).integers(0, 256, (*shape_prefix, 256, 260, channels), dtype=np.uint8)
    expected = np.sum(image, axis=tuple(range(image.ndim - 1)), dtype=np.uint64)
    _forbid(monkeypatch, "_cv2_reduce_sum_per_channel")
    _forbid(monkeypatch, "_cv2_reduce_mean_per_channel")
    np.testing.assert_array_equal(reduce_sum(image, "per_channel"), expected)
    np.testing.assert_array_equal(mean(image, "per_channel"), expected / (image.size // channels))


def test_per_channel_uint8_wide_images_keep_opencv_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.random.default_rng(2).integers(0, 256, (256, 260, 12), dtype=np.uint8)
    calls: list[str] = []
    for name in ("_cv2_reduce_sum_per_channel", "_cv2_reduce_mean_per_channel"):
        original: Callable[[np.ndarray], np.ndarray] = getattr(stats_mod, name)

        def spy(arr: np.ndarray, original: Callable[[np.ndarray], np.ndarray] = original, name: str = name) -> np.ndarray:
            calls.append(name)
            return original(arr)

        monkeypatch.setattr(stats_mod, name, spy)
    reduce_sum(image, "per_channel")
    mean(image, "per_channel")
    assert calls == ["_cv2_reduce_sum_per_channel", "_cv2_reduce_mean_per_channel"]


@pytest.mark.parametrize("shape", [(5, 7, 2), (33, 41, 3), (480, 640, 4), (1, 1, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_hwc_uint8_small_channel_mean_is_bit_equal_to_sum_over_count(shape: tuple[int, ...], keepdims: bool) -> None:
    image = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
    expected = np.sum(image, axis=(0, 1), dtype=np.uint64).astype(np.float64) / (shape[0] * shape[1])
    result = mean(image, "per_channel", keepdims=keepdims)
    assert result.dtype == np.float64
    assert result.shape == ((1, 1, shape[2]) if keepdims else (shape[2],))
    np.testing.assert_array_equal(result.reshape(-1), expected)


def test_hwc_uint8_small_channel_mean_handles_strided_input() -> None:
    image = np.random.default_rng(4).integers(0, 256, (40, 60, 3), dtype=np.uint8)[:, ::2]
    expected = np.sum(image, axis=(0, 1), dtype=np.uint64) / (image.shape[0] * image.shape[1])
    np.testing.assert_array_equal(mean(image, "per_channel"), expected)


@pytest.mark.parametrize(("shape", "layout"), [((9, 256, 260), "CHW"), ((2, 5, 256, 130), "NCHW")])
@pytest.mark.parametrize("keepdims", [False, True])
def test_tensor_uint8_wide_per_channel_uses_plane_sums(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], layout: str, keepdims: bool
) -> None:
    image = torch.from_numpy(np.random.default_rng(5).integers(0, 256, shape, dtype=np.uint8))
    channel_axis = 0 if layout == "CHW" else 1
    reduced = tuple(axis for axis in range(image.ndim) if axis != channel_axis)
    expected_sum = image.numpy().astype(np.uint64).sum(axis=reduced, keepdims=keepdims)
    expected_mean = expected_sum / (image.numel() // shape[channel_axis])
    monkeypatch.setattr(stats_mod, "dispatch_tensor_reduction", None)
    total = reduce_sum(image, "per_channel", keepdims=keepdims, layout=layout)
    averaged = mean(image, "per_channel", keepdims=keepdims, layout=layout)
    assert total.dtype == torch.uint64
    assert averaged.dtype == torch.float64
    assert total.shape == expected_sum.shape
    np.testing.assert_array_equal(total.numpy(), expected_sum)
    np.testing.assert_array_equal(averaged.numpy(), expected_mean)


@pytest.mark.parametrize(("shape", "layout"), [((3, 256, 260), "CHW"), ((9, 64, 64), "CHW"), ((2, 5, 64, 64), "NCHW")])
def test_tensor_uint8_narrow_or_small_per_channel_keeps_numpy_route(shape: tuple[int, ...], layout: str) -> None:
    image = torch.from_numpy(np.random.default_rng(6).integers(0, 256, shape, dtype=np.uint8))
    channel_axis = 0 if layout == "CHW" else 1
    reduced = tuple(axis for axis in range(image.ndim) if axis != channel_axis)
    expected = image.numpy().astype(np.uint64).sum(axis=reduced)
    np.testing.assert_array_equal(reduce_sum(image, "per_channel", layout=layout).numpy(), expected)
