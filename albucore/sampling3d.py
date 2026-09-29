"""Shared CPU Torch sampling mechanics for 3D volumes and volume batches."""

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


def _sample3d_torch_cpu_batch(
    volumes: torch.Tensor,
    sampling_grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample prevalidated CPU ``NCDHW`` volumes through one shared normalized ``DHWC3`` pull grid."""
    if volumes.shape[0] >= 4 and volumes.shape[1] == 1:
        # Full-call CPU benchmarks favor treating one-channel volumes as a channel batch.
        batch_size = volumes.shape[0]
        folded = volumes.reshape(batch_size, *volumes.shape[2:])
        folded_result = _sample3d_torch_cpu(
            folded,
            sampling_grid,
            interpolation,
            border_mode,
            np.tile(border_values, batch_size),
        )
        return folded_result.reshape(batch_size, 1, *folded_result.shape[1:])

    return _sample3d_torch_cpu_batch_native(volumes, sampling_grid, interpolation, border_mode, border_values)


def _sample3d_torch_cpu_batch_native(
    volumes: torch.Tensor,
    sampling_grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample prevalidated CPU ``NCDHW`` volumes through one shared normalized ``DHW3`` pull grid."""
    mode = _INTERPOLATIONS[interpolation]
    padding_mode = _BORDERS[border_mode]
    working_volume = volumes if volumes.dtype == torch.float32 else volumes.to(torch.float32)
    grid = sampling_grid.unsqueeze(0).expand(volumes.shape[0], -1, -1, -1, -1)

    with torch.no_grad():
        if border_mode == cv2.BORDER_CONSTANT and np.any(border_values):
            fill = torch.from_numpy(border_values).reshape(1, volumes.shape[1], 1, 1, 1)
            result = torch_f.grid_sample(
                working_volume - fill,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False,
            ).add_(fill)
        else:
            result = torch_f.grid_sample(
                working_volume,
                grid,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False,
            )
        if volumes.dtype == torch.uint8:
            result = _restore_uint8(result)
    return result


def _sample3d_torch_cpu(
    volume: torch.Tensor,
    sampling_grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample one prevalidated CPU ``CDHW`` volume through one normalized ``DHW3`` pull grid."""
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
