"""Shared CPU Torch sampling mechanics for single 3D volumes."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

_INTERPOLATIONS = {
    cv2.INTER_LINEAR: "bilinear",
    cv2.INTER_NEAREST: "nearest",
}
_BORDERS = {
    cv2.BORDER_CONSTANT: "zeros",
    cv2.BORDER_REPLICATE: "border",
}


def _normalize_border_value(
    border_value: float | tuple[float, ...] | np.ndarray | None,
    channels: int,
) -> np.ndarray:
    """Convert prevalidated constant-border values to contiguous float32 channel data."""
    if border_value is None:
        return np.zeros(channels, dtype=np.float32)
    values = np.asarray(border_value, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(channels, values.item(), dtype=np.float32)
    return np.ascontiguousarray(values, dtype=np.float32)


def _restore_uint8(result: torch.Tensor) -> torch.Tensor:
    """Saturate and round one freshly allocated float32 sampling result exactly once."""
    return torch.clamp(result, 0.0, 255.0).add_(0.5).to(torch.uint8)


def _sample3d_torch_cpu(
    volume: torch.Tensor,
    sampling_grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample one prevalidated CPU ``CDHW`` volume through one normalized ``DHWC3`` pull grid."""
    mode = _INTERPOLATIONS[interpolation]
    padding_mode = _BORDERS[border_mode]
    working_volume = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
    grid = sampling_grid.unsqueeze(0)

    with torch.no_grad():
        if border_mode == cv2.BORDER_CONSTANT and np.any(border_values):
            fill = torch.from_numpy(border_values).reshape(1, volume.shape[0], 1, 1, 1)
            result = torch_f.grid_sample(
                working_volume.unsqueeze(0) - fill,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False,
            ).add_(fill)
        else:
            result = torch_f.grid_sample(
                working_volume.unsqueeze(0),
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False,
            )
        if volume.dtype == torch.uint8:
            result = _restore_uint8(result)
    return result.squeeze(0)
