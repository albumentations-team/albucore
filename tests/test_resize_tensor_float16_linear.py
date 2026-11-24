"""Tests for the float16 Tensor linear resize split between OpenCV and Torch."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from albucore import resize


@pytest.fixture
def interpolate_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = torch.nn.functional.interpolate

    def spy(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", spy)
    return calls


def _tensor(layout: str, channels: int, height: int, width: int) -> torch.Tensor:
    shape = (2, channels, height, width) if layout == "NCHW" else (channels, height, width)
    values = np.random.default_rng(height * width + channels).random(shape, dtype=np.float32)
    return torch.from_numpy(values).to(torch.float16)


def _reference(tensor: torch.Tensor, layout: str, dsize: tuple[int, int]) -> torch.Tensor:
    images = tensor.numpy() if layout == "NCHW" else tensor.numpy()[None]
    resized = [
        resize(np.ascontiguousarray(image.transpose(1, 2, 0)), dsize).transpose(2, 0, 1) for image in images
    ]
    result = np.stack(resized) if layout == "NCHW" else resized[0]
    return torch.from_numpy(np.ascontiguousarray(result))


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("channels", [3, 9])
@pytest.mark.parametrize(
    ("dsize", "uses_torch"),
    [((240, 180), False), ((200, 150), False), ((120, 90), True), ((240, 60), True), ((100, 180), True)],
)
def test_float16_linear_upscale_uses_opencv_bridge_and_other_changes_keep_torch(
    interpolate_spy: list[int],
    layout: str,
    channels: int,
    dsize: tuple[int, int],
    uses_torch: bool,
) -> None:
    tensor = _tensor(layout, channels, 120, 160)
    before = tensor.clone()

    result = resize(tensor, dsize, interpolation=cv2.INTER_LINEAR, layout=layout)

    expected = _reference(tensor, layout, dsize)
    assert result.dtype == torch.float16
    assert result.shape == expected.shape
    assert bool(interpolate_spy) is uses_torch
    if uses_torch:
        np.testing.assert_allclose(
            result.float().numpy(), expected.float().numpy(), rtol=0.0, atol=float(np.finfo(np.float16).eps / 2)
        )
    else:
        assert torch.equal(result.contiguous().view(torch.int16), expected.view(torch.int16))
    assert torch.equal(tensor.view(torch.int16), before.view(torch.int16))


def test_float16_linear_same_size_uses_opencv_bridge(interpolate_spy: list[int]) -> None:
    tensor = _tensor("CHW", 3, 120, 160)

    result = resize(tensor, (160, 120), interpolation=cv2.INTER_LINEAR, layout="CHW")

    assert torch.equal(result.contiguous().view(torch.int16), tensor.view(torch.int16))
    assert not interpolate_spy
