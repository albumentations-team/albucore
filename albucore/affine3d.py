# ruff: noqa: PLR0913  # The public OpenCV-compatible signature and internal kernel contract need these parameters.
"""True 3D affine resampling for single NumPy and Torch volumes."""

from __future__ import annotations

from typing import cast, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

__all__ = ["warp_affine3d"]


_WARP_AFFINE3D_INTERPOLATIONS = {
    cv2.INTER_LINEAR: "bilinear",
    cv2.INTER_NEAREST: "nearest",
}
_WARP_AFFINE3D_BORDERS = {
    cv2.BORDER_CONSTANT: "zeros",
    cv2.BORDER_REPLICATE: "border",
}


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert prevalidated forward affine control data to homogeneous float64 form."""
    matrix_array = np.asarray(matrix, dtype=np.float64)
    if matrix_array.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3] = matrix_array
    else:
        homogeneous = matrix_array
    return homogeneous


def _inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert prevalidated homogeneous affine control data once."""
    return np.linalg.inv(matrix)


def _is_identity_matrix(matrix: np.ndarray) -> bool:
    """Identify an exact affine identity without accepting a nearby real transform."""
    return bool(np.array_equal(matrix, np.eye(4, dtype=np.float64)))


def _normalized_theta(
    inverse_matrix: np.ndarray,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
) -> np.ndarray:
    """Convert inverse voxel coordinates to the ``align_corners=False`` Torch affine-grid convention."""
    input_depth, input_height, input_width = input_size
    output_depth, output_height, output_width = output_size
    normalized_from_input_voxel = np.array(
        (
            (2.0 / input_width, 0.0, 0.0, -(input_width - 1.0) / input_width),
            (0.0, 2.0 / input_height, 0.0, -(input_height - 1.0) / input_height),
            (0.0, 0.0, 2.0 / input_depth, -(input_depth - 1.0) / input_depth),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    output_voxel_from_normalized = np.array(
        (
            (output_width / 2.0, 0.0, 0.0, (output_width - 1.0) / 2.0),
            (0.0, output_height / 2.0, 0.0, (output_height - 1.0) / 2.0),
            (0.0, 0.0, output_depth / 2.0, (output_depth - 1.0) / 2.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    theta = normalized_from_input_voxel @ inverse_matrix @ output_voxel_from_normalized
    return np.ascontiguousarray(theta[:3], dtype=np.float32)


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


def _warp_affine3d_torch_cpu(
    volume: torch.Tensor,
    inverse_matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample one prevalidated CPU ``CDHW`` tensor through the native Torch 3D kernel."""
    mode = _WARP_AFFINE3D_INTERPOLATIONS[interpolation]
    padding_mode = _WARP_AFFINE3D_BORDERS[border_mode]
    input_size = volume.shape[1], volume.shape[2], volume.shape[3]
    theta = torch.from_numpy(_normalized_theta(inverse_matrix, input_size, size)).unsqueeze(0)
    working_volume = volume if volume.dtype == torch.float32 else volume.to(torch.float32)

    with torch.no_grad():
        grid = torch_f.affine_grid(theta, [1, volume.shape[0], *size], align_corners=False)
        if border_mode == cv2.BORDER_CONSTANT and np.any(border_values):
            fill = torch.from_numpy(border_values).reshape(1, volume.shape[0], 1, 1, 1)
            result = (
                torch_f.grid_sample(
                    working_volume.unsqueeze(0) - fill,
                    grid,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=False,
                )
                + fill
            )
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


def _warp_affine3d_numpy(
    volume: np.ndarray,
    inverse_matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> np.ndarray:
    """Bridge one ``DHWC`` NumPy volume through the native CPU Torch kernel."""
    if not volume.flags.writeable or any(stride < 0 for stride in volume.strides):
        volume = np.array(volume, copy=True, order="C")
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    result = _warp_affine3d_torch_cpu(tensor, inverse_matrix, size, interpolation, border_mode, border_values)
    return cast("np.ndarray", result.permute(1, 2, 3, 0).numpy())


@overload
def warp_affine3d(
    volume: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def warp_affine3d(
    volume: torch.Tensor,
    matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> torch.Tensor: ...


def warp_affine3d(
    volume: np.ndarray | torch.Tensor,
    matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply one forward 3D affine matrix to a single NumPy or CPU Torch volume.

    NumPy input uses ``(D, H, W, C)`` layout; Torch input uses ``(C, D, H, W)``.
    The matrix maps voxel-center ``(x, y, z)`` input coordinates to output coordinates.
    ``size`` is ordered as ``(depth, height, width)``. Only ``uint8`` and ``float32`` are
    supported. Callers validate the input container, layout, dtype, device, and control
    data before calling this low-level kernel.

    Args:
        volume: NumPy ``DHWC`` array or CPU Torch ``CDHW`` tensor.
        matrix: Forward affine matrix with shape ``(3, 4)`` or homogeneous ``(4, 4)``.
        size: Output spatial ``(depth, height, width)``.
        interpolation: ``cv2.INTER_LINEAR`` for trilinear sampling or ``cv2.INTER_NEAREST``.
        border_mode: ``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE``.
        border_value: Constant scalar or one value per channel; used only for constant borders.

    Returns:
        A volume with the input container, dtype, layout, and channels. An exact identity
        matrix with unchanged spatial shape returns ``volume`` itself.

    """
    output_size = size
    homogeneous_matrix = _normalize_matrix(matrix)
    if isinstance(volume, np.ndarray):
        border_values = _normalize_border_value(border_value, volume.shape[-1])
        if volume.shape[:3] == output_size and _is_identity_matrix(homogeneous_matrix):
            return volume
        inverse_matrix = _inverse_matrix(homogeneous_matrix)
        return _warp_affine3d_numpy(
            volume,
            inverse_matrix,
            output_size,
            interpolation,
            border_mode,
            border_values,
        )

    border_values = _normalize_border_value(border_value, volume.shape[0])
    if tuple(volume.shape[1:]) == output_size and _is_identity_matrix(homogeneous_matrix):
        return volume
    inverse_matrix = _inverse_matrix(homogeneous_matrix)
    return _warp_affine3d_torch_cpu(
        volume,
        inverse_matrix,
        output_size,
        interpolation,
        border_mode,
        border_values,
    )
