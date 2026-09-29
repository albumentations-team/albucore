"""True 3D affine resampling for NumPy and Torch volumes and batches."""

from __future__ import annotations

from typing import overload

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from albucore.sampling3d import _normalize_border_value, _sample3d_torch_cpu, _sample3d_torch_cpu_batch

__all__ = ["warp_affine3d"]


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


def _warp_affine3d_torch_cpu(
    volume: torch.Tensor,
    inverse_matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample one prevalidated CPU ``CDHW`` tensor through the native Torch 3D kernel."""
    input_size = volume.shape[1], volume.shape[2], volume.shape[3]
    theta = torch.from_numpy(_normalized_theta(inverse_matrix, input_size, size)).unsqueeze(0)
    grid = torch_f.affine_grid(theta, [1, volume.shape[0], *size], align_corners=False).squeeze(0)
    return _sample3d_torch_cpu(volume, grid, interpolation, border_mode, border_values)


def _warp_affine3d_torch_cpu_batch(
    volumes: torch.Tensor,
    inverse_matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    border_mode: int,
    border_values: np.ndarray,
) -> torch.Tensor:
    """Sample prevalidated CPU ``NCDHW`` volumes with one shared affine grid."""
    input_size = volumes.shape[2], volumes.shape[3], volumes.shape[4]
    theta = torch.from_numpy(_normalized_theta(inverse_matrix, input_size, size)).unsqueeze(0)
    grid = torch_f.affine_grid(theta, [1, volumes.shape[1], *size], align_corners=False).squeeze(0)
    return _sample3d_torch_cpu_batch(volumes, grid, interpolation, border_mode, border_values)


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
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


def _warp_affine3d_batch(
    volumes: np.ndarray | torch.Tensor,
    homogeneous_matrix: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray | None,
) -> np.ndarray | torch.Tensor:
    """Sample a prevalidated ``NDHWC`` NumPy or ``NCDHW`` CPU Tensor batch."""
    if isinstance(volumes, np.ndarray):
        input_size = tuple(volumes.shape[1:4])
        if input_size == size and _is_identity_matrix(homogeneous_matrix):
            return volumes
        channels = volumes.shape[-1]
        if volumes.shape[0] == 0:
            return np.empty((0, *size, channels), dtype=volumes.dtype)
        border_values = _normalize_border_value(border_value, channels)
        inverse_matrix = _inverse_matrix(homogeneous_matrix)
        if not volumes.flags.writeable or any(stride < 0 for stride in volumes.strides):
            volumes = np.array(volumes, copy=True, order="C")
        tensor = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
        result = _warp_affine3d_torch_cpu_batch(
            tensor,
            inverse_matrix,
            size,
            interpolation,
            border_mode,
            border_values,
        )
        return np.asarray(result.permute(0, 2, 3, 4, 1).numpy())

    input_size = tuple(volumes.shape[2:5])
    if input_size == size and _is_identity_matrix(homogeneous_matrix):
        return volumes
    channels = volumes.shape[1]
    if volumes.shape[0] == 0:
        return volumes.new_empty((0, channels, *size))
    border_values = _normalize_border_value(border_value, channels)
    inverse_matrix = _inverse_matrix(homogeneous_matrix)
    return _warp_affine3d_torch_cpu_batch(
        volumes,
        inverse_matrix,
        size,
        interpolation,
        border_mode,
        border_values,
    )


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
    """Apply one forward 3D affine matrix to a NumPy or CPU Torch volume or batch.

    A single NumPy volume uses ``(D, H, W, C)`` and a batch uses ``(N, D, H, W, C)``.
    A single Torch volume uses ``(C, D, H, W)`` and a batch uses ``(N, C, D, H, W)``.
    One matrix is shared by every volume. The matrix maps voxel-center ``(x, y, z)``
    input coordinates to output coordinates. ``size`` is ordered as ``(depth, height,
    width)``. Only ``uint8`` and ``float32`` are supported. Callers validate the input
    container, layout, dtype, device, and control data before this low-level kernel.

    Args:
        volume: NumPy ``DHWC``/``NDHWC`` array or CPU Torch ``CDHW``/``NCDHW`` tensor.
        matrix: Shared forward affine matrix with shape ``(3, 4)`` or homogeneous ``(4, 4)``.
        size: Output spatial ``(depth, height, width)``.
        interpolation: ``cv2.INTER_LINEAR`` for trilinear sampling or ``cv2.INTER_NEAREST``.
        border_mode: ``cv2.BORDER_CONSTANT`` or ``cv2.BORDER_REPLICATE``.
        border_value: Constant scalar or one value per channel, shared by the batch.

    Returns:
        A volume or batch in the input container, dtype, and layout. An exact identity
        matrix with unchanged spatial shape returns ``volume`` itself.
    """
    output_size = size
    homogeneous_matrix = _normalize_matrix(matrix)
    if isinstance(volume, np.ndarray):
        if volume.ndim == 5:
            return _warp_affine3d_batch(
                volume,
                homogeneous_matrix,
                output_size,
                interpolation,
                border_mode,
                border_value,
            )
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

    if volume.ndim == 5:
        return _warp_affine3d_batch(
            volume,
            homogeneous_matrix,
            output_size,
            interpolation,
            border_mode,
            border_value,
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
