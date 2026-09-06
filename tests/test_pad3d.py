# ruff: noqa: S101
"""Contract tests for constant 3D padding across NumPy and Torch."""

import gc
import warnings

import numpy as np
import pytest
import torch

from albucore import pad3d


def _pad3d_reference(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...],
) -> np.ndarray:
    front, back, top, bottom, left, right = padding
    return np.pad(volume, ((front, back), (top, bottom), (left, right), (0, 0)), constant_values=value)


def test_pad3d_identity() -> None:
    volume = np.ones((2, 3, 4, 1), dtype=np.uint8)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    for source in (volume, tensor):
        assert pad3d(source, (0, 0, 0, 0, 0, 0)) is source


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 3, 1)])
@pytest.mark.parametrize("padding", [(1, 2, 3, 0, 2, 1), (0, 0, 0, 0, 0, 4), (3, 0, 0, 1, 0, 0)])
@pytest.mark.parametrize("tensor", [False, True])
def test_pad3d_parity(
    dtype: type[np.generic],
    channels: int,
    shape: tuple[int, int, int],
    padding: tuple[int, int, int, int, int, int],
    tensor: bool,
) -> None:
    volume = np.arange(np.prod(shape) * channels).reshape(*shape, channels).astype(dtype)
    source = torch.from_numpy(volume).permute(3, 0, 1, 2).contiguous() if tensor else volume
    expected = _pad3d_reference(volume, padding, 7.5)
    result = pad3d(source, padding, 7.5)
    assert result.dtype == source.dtype
    actual = result.permute(1, 2, 3, 0).numpy() if tensor else result
    np.testing.assert_array_equal(actual, expected)
    actual[...] = 0
    np.testing.assert_array_equal(volume, np.arange(volume.size).reshape(volume.shape).astype(dtype))
    if tensor:
        np.testing.assert_array_equal(source.permute(1, 2, 3, 0).numpy(), volume)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
@pytest.mark.parametrize("value", [-1, 256, -7.5, (3,), (3, 7), (-1, 256)])
@pytest.mark.parametrize("tensor", [False, True])
def test_pad3d_fill_cast_and_tuple_corners(
    dtype: type[np.generic],
    value: float | tuple[float, ...],
    tensor: bool,
) -> None:
    volume = np.arange(72).reshape(2, 3, 4, 3).astype(dtype)
    source = torch.from_numpy(volume).permute(3, 0, 1, 2).contiguous() if tensor else volume
    padding = (1, 2, 2, 1, 3, 2)
    result = pad3d(source, padding, value)
    actual = result.permute(1, 2, 3, 0).numpy() if tensor else result
    np.testing.assert_array_equal(actual, _pad3d_reference(volume, padding, value))


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 1e39])
def test_pad3d_float_fill(value: float) -> None:
    volume = np.zeros((2, 3, 4, 1), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = _pad3d_reference(volume, (1, 1, 1, 1, 1, 1), value)
        for source in (volume, torch.from_numpy(volume).permute(3, 0, 1, 2)):
            result = pad3d(source, (1, 1, 1, 1, 1, 1), value)
            actual = result.permute(1, 2, 3, 0).numpy() if isinstance(result, torch.Tensor) else result
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
@pytest.mark.parametrize("layout", ["positive", "negative", "readonly", "fortran", "broadcast"])
def test_pad3d_numpy_strides(dtype: type[np.generic], layout: str) -> None:
    volume = np.arange(2 * 6 * 4 * 5).reshape(2, 6, 4, 5).astype(dtype)
    if layout == "positive":
        volume = volume[:, ::2]
    elif layout == "negative":
        volume = volume[::-1, :, ::-1]
    elif layout == "readonly":
        volume.flags.writeable = False
    elif layout == "fortran":
        volume = np.asfortranarray(volume)
    elif layout == "broadcast":
        volume = np.broadcast_to(volume[:1], volume.shape)
    snapshot = volume.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pad3d(volume, (1, 2, 3, 4, 5, 6), 7)
    np.testing.assert_array_equal(result, _pad3d_reference(volume, (1, 2, 3, 4, 5, 6), 7))
    np.testing.assert_array_equal(volume, snapshot)
    assert not np.shares_memory(result, volume)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.float32])
@pytest.mark.parametrize("layout", ["channel_last", "sliced", "planar_sliced", "expanded"])
def test_pad3d_tensor_strides(dtype: torch.dtype, layout: str) -> None:
    volume = torch.arange(2 * 6 * 4 * 5).reshape(2, 6, 4, 5).to(dtype).permute(3, 0, 1, 2)
    if layout == "sliced":
        volume = volume[:, :, ::2]
    elif layout == "planar_sliced":
        volume = volume.contiguous()[:, :, ::2]
    elif layout == "expanded":
        volume = volume[:1].expand(5, -1, -1, -1)
    snapshot = volume.clone()
    result = pad3d(volume, (1, 2, 3, 4, 5, 6), -1)
    np.testing.assert_array_equal(
        result.permute(1, 2, 3, 0).numpy(),
        _pad3d_reference(volume.permute(1, 2, 3, 0).numpy(), (1, 2, 3, 4, 5, 6), -1),
    )
    assert torch.equal(volume, snapshot)
    assert result.data_ptr() != volume.data_ptr()


@pytest.mark.parametrize("value", [0, (3, 7)])
@pytest.mark.parametrize("tensor", [False, True])
def test_pad3d_output_lifetime_and_training(value: float | tuple[float, ...], tensor: bool) -> None:
    volume = np.ones((2, 3, 4, 3), dtype=np.float32)
    source = torch.from_numpy(volume).permute(3, 0, 1, 2).contiguous() if tensor else volume
    result = pad3d(source, (1, 1, 1, 1, 1, 1), value)
    del volume, source
    gc.collect()
    result = result if tensor else torch.from_numpy(result).permute(3, 0, 1, 2)
    np.testing.assert_array_equal(result[:, 1:-1, 1:-1, 1:-1].numpy(), np.ones((3, 2, 3, 4)))
    assert not result.is_inference()
    model = torch.nn.Conv3d(3, 1, 1, bias=False)
    model(result.unsqueeze(0)).sum().backward()
    assert model.weight.grad is not None
    assert torch.all(model.weight.grad > 0)

@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape",
    [(31, 64, 256, 1), (32, 64, 256, 1), (31, 64, 256, 2), (32, 64, 256, 2), (128, 64, 256, 2), (129, 64, 256, 2)],
)
def test_pad3d_numpy_routing_boundaries(
    dtype: type[np.generic],
    shape: tuple[int, int, int, int],
) -> None:
    volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(dtype)
    if shape[-1] == 2:
        volume = np.repeat(volume, 2, axis=1)[:, ::2]
    padding = (1, 3, 2, 0, 0, 4)
    result = pad3d(volume, padding, -1)
    np.testing.assert_array_equal(result, _pad3d_reference(volume, padding, -1))
    assert not np.shares_memory(result, volume)


@pytest.mark.parametrize("layout", ["negative", "readonly", "unaligned_stride", "fortran"])
def test_pad3d_large_numpy_without_torch_wrapping(layout: str) -> None:
    shape = (32, 128, 160, 1)
    volume = np.ones(shape, dtype=np.float32)
    if layout == "negative":
        volume = volume[::-1]
    elif layout == "readonly":
        volume.flags.writeable = False
    elif layout == "fortran":
        volume = np.asfortranarray(volume)
    else:
        row_bytes = 160 * 4 + 1
        volume = np.ndarray(
            shape,
            dtype=np.float32,
            buffer=bytearray(32 * 128 * row_bytes),
            strides=(128 * row_bytes, row_bytes, 4, 4),
        )
        volume[...] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pad3d(volume, (1, 2, 3, 4, 5, 6), 0.25)
    np.testing.assert_array_equal(result, _pad3d_reference(volume, (1, 2, 3, 4, 5, 6), 0.25))


def test_pad3d_numpy_torch_bridge_lifetime() -> None:
    result = pad3d(np.ones((32, 128, 160, 1), dtype=np.float32), (1, 1, 1, 1, 1, 1), 0.25)
    gc.collect()
    np.testing.assert_array_equal(result[1:-1, 1:-1, 1:-1], np.ones((32, 128, 160, 1), dtype=np.float32))
    tensor = torch.from_numpy(result).permute(3, 0, 1, 2)
    assert not tensor.is_inference()
    model = torch.nn.Conv3d(1, 1, 1, bias=False)
    model(tensor.unsqueeze(0)).sum().backward()
    assert model.weight.grad is not None
    assert torch.all(model.weight.grad > 0)
