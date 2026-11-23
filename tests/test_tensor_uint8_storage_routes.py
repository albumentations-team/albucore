"""Tensor uint8 storage routes (scalar LUTs, blend, normalize) match the NumPy channel-last routers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from albucore import add, add_weighted, multiply, multiply_add, normalize, power
from albucore.tensor_bridge import tensor_to_numpy_channel_last

OPERATIONS: dict[str, Callable[..., object]] = {
    "add_positive": lambda image, **kwargs: add(image, 37.6, **kwargs),
    "add_negative": lambda image, **kwargs: add(image, -300.0, **kwargs),
    "multiply": lambda image, **kwargs: multiply(image, 1.7, **kwargs),
    "power": lambda image, **kwargs: power(image, 0.5, **kwargs),
    "multiply_add": lambda image, **kwargs: multiply_add(image, 1.3, -20.0, **kwargs),
}


def _tensor(layout: str, storage: str) -> torch.Tensor:
    shape = (3, 5, 7) if layout == "CHW" else (2, 3, 5, 7)
    values = torch.from_numpy(np.random.default_rng(31).integers(0, 256, shape, dtype=np.uint8))
    values.view(-1)[:2] = torch.tensor([0, 255], dtype=torch.uint8)
    if storage == "channels_last":
        values = values.movedim(-3, -1).contiguous().movedim(-1, -3)
    return values


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("storage", ["contiguous", "channels_last"])
@pytest.mark.parametrize("destination", ["none", "separate", "image"])
def test_tensor_uint8_scalar_matches_numpy(name: str, layout: str, storage: str, destination: str) -> None:
    operation = OPERATIONS[name]
    image = _tensor(layout, storage)
    expected = operation(tensor_to_numpy_channel_last(image, layout).copy())
    original = image.clone()

    if destination == "none":
        result = operation(image, layout=layout)
        torch.testing.assert_close(image, original, rtol=0, atol=0)
    elif destination == "separate":
        out = torch.empty_like(image)
        result = operation(image, layout=layout, out=out)
        assert result is out
    else:
        result = operation(image, layout=layout, out=image)
        assert result is image

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.uint8
    assert result.shape == image.shape
    np.testing.assert_array_equal(tensor_to_numpy_channel_last(result, layout), expected)


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("storage", ["contiguous", "channels_last"])
@pytest.mark.parametrize("destination", ["none", "separate", "first", "second"])
def test_tensor_uint8_add_weighted_matches_numpy(layout: str, storage: str, destination: str) -> None:
    first = _tensor(layout, storage)
    second = torch.flip(_tensor(layout, storage), dims=(-1,)).contiguous()
    if storage == "channels_last":
        second = second.movedim(-3, -1).contiguous().movedim(-1, -3)
    expected = add_weighted(
        tensor_to_numpy_channel_last(first, layout).copy(),
        0.3,
        tensor_to_numpy_channel_last(second, layout).copy(),
        0.7,
    )
    out = {"none": None, "separate": torch.empty_like(first), "first": first, "second": second}[destination]

    result = add_weighted(first, 0.3, second, 0.7, layout=layout, out=out)

    if out is not None:
        assert result is out
    np.testing.assert_array_equal(tensor_to_numpy_channel_last(result, layout), expected)


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("storage", ["contiguous", "channels_last"])
@pytest.mark.parametrize("with_out", [False, True])
def test_tensor_uint8_scalar_normalize_matches_numpy(layout: str, storage: str, with_out: bool) -> None:
    image = _tensor(layout, storage)
    expected = normalize(tensor_to_numpy_channel_last(image, layout).copy(), 127.3, 1 / 57.1)
    out = torch.empty(image.shape, dtype=torch.float32) if with_out else None

    result = normalize(image, 127.3, 1 / 57.1, layout=layout, out=out)

    if out is not None:
        assert result is out
    assert result.dtype == torch.float32
    np.testing.assert_array_equal(tensor_to_numpy_channel_last(result, layout), expected)
