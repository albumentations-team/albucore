"""Dense 3D resampling for single NumPy volumes."""

from __future__ import annotations

from typing import overload

import cv2
import numpy as np
import torch

from albucore.sampling3d import _normalize_border_value, _sample3d_torch_cpu

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


def _remap3d_tensor(
    volume: torch.Tensor,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Bridge one ``CDHW`` Tensor through the benchmark-selected NumPy public route."""
    numpy_volume = np.asarray(volume.permute(1, 2, 3, 0).numpy())
    result = _remap3d_numpy(numpy_volume, sampling_grid, interpolation, border_mode, border_values)
    return torch.from_numpy(result).permute(3, 0, 1, 2)


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
    """Apply one normalized dense pull grid to a single NumPy or CPU Tensor volume.

    NumPy volumes use ``(D, H, W, C)`` layout and CPU Tensors use ``(C, D, H, W)``.
    ``sampling_grid`` has
    ``(output_depth, output_height, output_width, 3)`` layout, contains normalized
    ``align_corners=False`` coordinates in ``(x, y, z)`` order, and may be a NumPy
    array or CPU Torch tensor. Its spatial shape defines the output. Only ``uint8``
    and ``float32`` volumes are supported. Callers validate the volume, grid, and
    parameter contract before calling this low-level kernel. This primitive does not
    inspect the dense grid for identity.

    Args:
        volume: NumPy ``DHWC`` array or CPU Torch ``CDHW`` tensor.
        sampling_grid: Float32 normalized pull grid with shape ``(D_out, H_out, W_out, 3)``.
        interpolation: ``cv2.INTER_LINEAR`` for trilinear sampling or ``cv2.INTER_NEAREST``.
        border_mode: ``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE``.
        border_value: Constant scalar or one value per channel; used only for constant borders.

    Returns:
        A newly sampled volume with the input container, dtype, layout, and channel count.

    Notes:
        The direct CPU Tensor sampler lost the full-path route gate. Tensor inputs therefore
        use a zero-copy Tensor-to-NumPy-to-Tensor bridge, while retaining a Tensor result.

    """
    channels = volume.shape[-1] if isinstance(volume, np.ndarray) else volume.shape[0]
    border_values = _normalize_border_value(border_value, channels)
    if isinstance(volume, np.ndarray):
        return _remap3d_numpy(volume, sampling_grid, interpolation, border_mode, border_values)
    return _remap3d_tensor(volume, sampling_grid, interpolation, border_mode, border_values)
