"""Dense 3D resampling for NumPy and Torch volumes and batches."""

from __future__ import annotations

from typing import overload

import cv2
import numpy as np
import torch

from albucore.sampling3d import _normalize_border_value, _sample3d_torch_cpu, _sample3d_torch_cpu_batch

__all__ = ["remap3d"]


def _sampling_grid_to_tensor(sampling_grid: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Share caller-owned NumPy grid storage with Torch or retain a Tensor grid directly."""
    if isinstance(sampling_grid, np.ndarray):
        return torch.from_numpy(sampling_grid)
    return sampling_grid


def _remap3d_numpy(
    volume: np.ndarray,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> np.ndarray:
    """Bridge one ``DHWC`` NumPy volume through the shared CPU Torch sampler."""
    if not volume.flags.writeable or any(stride < 0 for stride in volume.strides):
        volume = np.array(volume, copy=True, order="C")
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    result = _sample3d_torch_cpu(
        tensor,
        _sampling_grid_to_tensor(sampling_grid),
        interpolation,
        border_mode,
        border_values,
    )
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


def _remap3d_numpy_batch(
    volumes: np.ndarray,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> np.ndarray:
    """Bridge one ``NDHWC`` NumPy batch through the shared CPU Torch sampler."""
    if not volumes.flags.writeable or any(stride < 0 for stride in volumes.strides):
        volumes = np.array(volumes, copy=True, order="C")
    tensor = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
    result = _remap3d_torch_cpu_batch(
        tensor,
        _sampling_grid_to_tensor(sampling_grid),
        interpolation,
        border_mode,
        border_values,
    )
    return np.asarray(result.permute(0, 2, 3, 4, 1).numpy())


def _remap3d_torch_cpu_batch(
    volumes: torch.Tensor,
    sampling_grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample wide prevalidated batches independently while sharing grid and fill preparation."""
    if volumes.shape[0] >= 4 and volumes.shape[1] > 1:
        # A native grid_sample batch is slower for these remap workloads.
        return torch.stack(
            [
                _sample3d_torch_cpu(volume, sampling_grid, interpolation, border_mode, border_values)
                for volume in volumes
            ],
        )
    return _sample3d_torch_cpu_batch(volumes, sampling_grid, interpolation, border_mode, border_values)


@overload
def remap3d(
    volume: np.ndarray,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def remap3d(
    volume: torch.Tensor,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> torch.Tensor: ...


def remap3d(
    volume: np.ndarray | torch.Tensor,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply one normalized dense pull grid to a NumPy or CPU Tensor volume or batch.

    A single NumPy volume uses ``(D, H, W, C)`` and a batch uses ``(N, D, H, W, C)``.
    A single CPU Tensor uses ``(C, D, H, W)`` and a batch uses ``(N, C, D, H, W)``.
    ``sampling_grid`` has ``(D_out, H_out, W_out, 3)`` layout, contains normalized
    ``align_corners=False`` coordinates in ``(x, y, z)`` order, and may be a NumPy
    array or CPU Torch tensor. Its spatial shape defines the output. One grid is shared
    by every volume. Only ``uint8`` and ``float32`` volumes are supported. Callers
    validate the volume, grid, and parameter contract before this low-level kernel.
    This primitive does not inspect the dense grid for identity.

    Args:
        volume: NumPy ``DHWC``/``NDHWC`` array or CPU Torch ``CDHW``/``NCDHW`` tensor.
        sampling_grid: Float32 normalized pull grid with shape ``(D_out, H_out, W_out, 3)``.
        interpolation: ``cv2.INTER_LINEAR`` for trilinear sampling or ``cv2.INTER_NEAREST``.
        border_mode: ``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE``.
        border_value: Constant scalar or one value per channel, shared by the batch.

    Returns:
        A newly sampled volume or batch with the input container, dtype, layout, and channels.
    """
    if isinstance(volume, np.ndarray):
        if volume.ndim == 5:
            output_size = tuple(sampling_grid.shape[:3])
            channels = volume.shape[-1]
            if volume.shape[0] == 0:
                return np.empty((0, *output_size, channels), dtype=volume.dtype)
            border_values = _normalize_border_value(border_value, channels)
            return _remap3d_numpy_batch(volume, sampling_grid, interpolation, border_mode, border_values)
        channels = volume.shape[-1]
        border_values = _normalize_border_value(border_value, channels)
        return _remap3d_numpy(volume, sampling_grid, interpolation, border_mode, border_values)

    if volume.ndim == 5:
        output_size = tuple(sampling_grid.shape[:3])
        channels = volume.shape[1]
        if volume.shape[0] == 0:
            return volume.new_empty((0, channels, *output_size))
        border_values = _normalize_border_value(border_value, channels)
        return _remap3d_torch_cpu_batch(
            volume,
            _sampling_grid_to_tensor(sampling_grid),
            interpolation,
            border_mode,
            border_values,
        )

    channels = volume.shape[0]
    border_values = _normalize_border_value(border_value, channels)
    return _sample3d_torch_cpu(
        volume,
        _sampling_grid_to_tensor(sampling_grid),
        interpolation,
        border_mode,
        border_values,
    )
