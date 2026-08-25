"""Dense 3D resampling for single NumPy volumes."""

from __future__ import annotations

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


def remap3d(
    volume: np.ndarray,
    sampling_grid: np.ndarray | torch.Tensor,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    """Apply one normalized dense pull grid to a single NumPy volume.

    Volumes use ``(D, H, W, C)`` layout. ``sampling_grid`` has
    ``(output_depth, output_height, output_width, 3)`` layout, contains normalized
    ``align_corners=False`` coordinates in ``(x, y, z)`` order, and may be a NumPy
    array or CPU Torch tensor. Its spatial shape defines the output. Only ``uint8``
    and ``float32`` volumes are supported. Callers validate the volume, grid, and
    parameter contract before calling this low-level kernel. This primitive does not
    inspect the dense grid for identity.

    Args:
        volume: NumPy ``DHWC`` array.
        sampling_grid: Float32 normalized pull grid with shape ``(D_out, H_out, W_out, 3)``.
        interpolation: ``cv2.INTER_LINEAR`` for trilinear sampling or ``cv2.INTER_NEAREST``.
        border_mode: ``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE``.
        border_value: Constant scalar or one value per channel; used only for constant borders.

    Returns:
        A newly sampled NumPy volume with the input dtype, layout, and channel count.

    """
    border_values = _normalize_border_value(border_value, volume.shape[-1])
    return _remap3d_numpy(volume, sampling_grid, interpolation, border_mode, border_values)
