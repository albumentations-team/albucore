# ruff: noqa: S101
"""Batch contracts and differential tests for ``warp_affine3d``."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from albucore import warp_affine3d


def _volume(dtype: type[np.uint8 | np.float32], channels: int, batch: int = 3) -> np.ndarray:
    """Build distinct non-cubic DHWC volumes to expose batch and axis mixing."""
    shape = (batch, 3, 4, 5, channels)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if dtype is np.uint8:
        return values.astype(np.uint8)
    return values / np.float32(10.0)


def _matrix() -> np.ndarray:
    """Return an invertible shared forward affine transform."""
    return np.array(
        ((0.9, 0.1, 0.0, 0.4), (0.0, 1.1, 0.1, -0.3), (0.05, 0.0, 1.0, 0.2)),
        dtype=np.float32,
    )


def _tensor_batch(volumes: np.ndarray) -> torch.Tensor:
    """Convert NDHWC to an explicitly declared NCDHW Tensor view."""
    return torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("batch_size", [1, 4], ids=("n1", "n4"))
@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=("uint8", "float32"))
@pytest.mark.parametrize("channels", [1, 5], ids=("c1", "c5"))
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR], ids=("nearest", "linear"))
@pytest.mark.parametrize(
    ("border_mode", "fill"),
    [(cv2.BORDER_CONSTANT, (11.0, 23.0, 37.0, 41.0, 53.0)), (cv2.BORDER_REPLICATE, None)],
    ids=("constant", "replicate"),
)
def test_warp_affine3d_batch_matches_stacked_single_volume_calls(  # noqa: PLR0913, PLR0917
    container: str,
    batch_size: int,
    dtype: type[np.uint8 | np.float32],
    channels: int,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> None:
    """One shared matrix preserves every single-volume result across the batch axis."""
    volumes = _volume(dtype, channels, batch=batch_size)
    volumes = volumes if container == "numpy" else _tensor_batch(volumes)
    matrix = _matrix()
    size = (2, 5, 4)
    channel_fill = None if fill is None else fill[:channels]
    result = warp_affine3d(
        volumes,
        matrix,
        size,
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=channel_fill,
    )

    if container == "numpy":
        expected = np.stack(
            [
                warp_affine3d(
                    volume,
                    matrix,
                    size,
                    interpolation=interpolation,
                    border_mode=border_mode,
                    border_value=channel_fill,
                )
                for volume in volumes
            ],
        )
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
    else:
        expected = torch.stack(
            [
                warp_affine3d(
                    volume,
                    matrix,
                    size,
                    interpolation=interpolation,
                    border_mode=border_mode,
                    border_value=channel_fill,
                )
                for volume in volumes
            ],
        )
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    assert result.shape == ((batch_size, *size, channels) if container == "numpy" else (batch_size, channels, *size))
    assert result.dtype == (dtype if container == "numpy" else torch.from_numpy(np.empty((), dtype=dtype)).dtype)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_warp_affine3d_batch_identity_aliases_and_empty_batch_uses_requested_shape(container: str) -> None:
    """Exact identity preserves storage, while a real empty warp returns its requested shape."""
    volumes = _volume(np.float32, channels=3, batch=0)
    matrix = np.eye(4, dtype=np.float32)
    volumes = volumes if container == "numpy" else _tensor_batch(volumes)

    identity = warp_affine3d(volumes, matrix, (3, 4, 5))
    assert identity is volumes

    translated = _matrix()
    result = warp_affine3d(volumes, translated, (1, 2, 6))
    assert result.shape == ((0, 1, 2, 6, 3) if container == "numpy" else (0, 3, 1, 2, 6))
    assert result.dtype == volumes.dtype
    assert result is not volumes


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_warp_affine3d_batch_accepts_homogeneous_matrix_and_preserves_input(container: str) -> None:
    """The 4x4 encoding matches the 3x4 transform without mutating the batch."""
    volumes = _volume(np.float32, channels=3)
    source = volumes.copy()
    volumes = volumes if container == "numpy" else _tensor_batch(volumes).contiguous()
    original = volumes.copy() if isinstance(volumes, np.ndarray) else volumes.clone()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3] = _matrix()

    result = warp_affine3d(volumes, matrix, (2, 5, 4), border_value=(2.0, 3.0, 5.0))
    expected_numpy = np.stack(
        [warp_affine3d(item, matrix, (2, 5, 4), border_value=(2.0, 3.0, 5.0)) for item in source],
    )

    if isinstance(result, np.ndarray):
        np.testing.assert_array_equal(result, expected_numpy)
        np.testing.assert_array_equal(volumes, original)
        assert not np.shares_memory(result, volumes)
    else:
        torch.testing.assert_close(result, torch.from_numpy(expected_numpy).permute(0, 4, 1, 2, 3), rtol=0, atol=0)
        torch.testing.assert_close(volumes, original, rtol=0, atol=0)
        assert result.untyped_storage().data_ptr() != volumes.untyped_storage().data_ptr()


def test_warp_affine3d_batch_supports_strided_numpy_and_tensor_batches() -> None:
    """Positive and negative NumPy strides and non-contiguous NCDHW views match the reference."""
    source = _volume(np.float32, channels=3, batch=2)
    numpy_volumes = source[:, :, :, ::-1, :]
    expected = np.stack([warp_affine3d(item, _matrix(), (2, 3, 5)) for item in numpy_volumes])
    numpy_result = warp_affine3d(numpy_volumes, _matrix(), (2, 3, 5))
    np.testing.assert_array_equal(numpy_result, expected)

    tensor_volumes = _tensor_batch(source)[:, :, :, :, ::2]
    expected_tensor = torch.stack([warp_affine3d(item, _matrix(), (2, 4, 3)) for item in tensor_volumes])
    tensor_result = warp_affine3d(tensor_volumes, _matrix(), (2, 4, 3))
    torch.testing.assert_close(tensor_result, expected_tensor, rtol=0, atol=0)


def test_warp_affine3d_batch_accepts_read_only_numpy_input() -> None:
    volumes = _volume(np.float32, channels=1, batch=4)
    volumes.setflags(write=False)
    expected = np.stack([warp_affine3d(item, _matrix(), (2, 3, 4)) for item in volumes])

    result = warp_affine3d(volumes, _matrix(), (2, 3, 4))

    np.testing.assert_array_equal(result, expected)
    assert not volumes.flags.writeable


@pytest.mark.parametrize("shape", [(1, 3, 5), (3, 1, 5), (3, 4, 1)], ids=("unit_d", "unit_h", "unit_w"))
def test_warp_affine3d_batch_preserves_unit_input_axes(shape: tuple[int, int, int]) -> None:
    """Sampling accepts a unit depth, height, or width axis in both declared layouts."""
    source = np.arange(2 * np.prod(shape) * 3, dtype=np.float32).reshape((2, *shape, 3))
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = 0.25

    result = warp_affine3d(source, matrix, shape)
    expected = np.stack([warp_affine3d(item, matrix, shape) for item in source])
    np.testing.assert_array_equal(result, expected)

    tensor_source = _tensor_batch(source)
    tensor_result = warp_affine3d(tensor_source, matrix, shape)
    torch.testing.assert_close(tensor_result, torch.from_numpy(expected).permute(0, 4, 1, 2, 3), rtol=0, atol=0)


def test_warp_affine3d_batch_result_can_feed_a_trainable_torch_module() -> None:
    """Sampling runs without autograd while its result remains usable by downstream training layers."""
    volumes = _tensor_batch(_volume(np.float32, channels=1, batch=2))
    layer = torch.nn.Conv3d(1, 1, kernel_size=1)

    result = warp_affine3d(volumes, _matrix(), (2, 3, 4))
    layer(result).sum().backward()

    assert layer.weight.grad is not None
