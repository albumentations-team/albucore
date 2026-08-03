# ruff: noqa: PLR0913  # The public OpenCV-compatible signature and internal kernel contract need these parameters.
"""True 3D affine resampling for single NumPy and Torch volumes."""

from __future__ import annotations

from operator import index
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


def _validate_size(size: object) -> tuple[int, int, int]:
    """Return one valid output ``(depth, height, width)`` tuple."""
    if not isinstance(size, tuple) or len(size) != 3:
        msg = f"size must contain three positive integer dimensions, got {size!r}."
        raise ValueError(msg)
    try:
        output_size = tuple(index(axis) for axis in size)
    except TypeError as error:
        msg = f"size must contain three positive integer dimensions, got {size!r}."
        raise ValueError(msg) from error
    if any(axis <= 0 for axis in output_size):
        msg = f"size must contain three positive integer dimensions, got {size!r}."
        raise ValueError(msg)
    return cast("tuple[int, int, int]", output_size)


def _validate_numpy_volume(volume: np.ndarray) -> None:
    """Validate the public NumPy ``DHWC`` contract without copying data."""
    if volume.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        msg = f"Unsupported dtype {volume.dtype}. Albucore warp_affine3d supports only uint8 and float32."
        raise ValueError(msg)
    if volume.ndim != 4 or any(axis_size <= 0 for axis_size in volume.shape):
        msg = f"warp_affine3d expects a non-empty NumPy DHWC volume, got shape {volume.shape}."
        raise ValueError(msg)


def _validate_torch_volume(volume: torch.Tensor) -> None:
    """Validate the public CPU Torch ``CDHW`` contract without moving data."""
    if volume.dtype not in (torch.uint8, torch.float32):
        msg = f"Unsupported dtype {volume.dtype}. Albucore warp_affine3d supports only torch.uint8 and torch.float32."
        raise ValueError(msg)
    if volume.ndim != 4 or any(axis_size <= 0 for axis_size in volume.shape):
        msg = f"warp_affine3d expects a non-empty Torch CDHW volume, got shape {tuple(volume.shape)}."
        raise ValueError(msg)
    if volume.device.type != "cpu":
        msg = f"warp_affine3d supports CPU Torch tensors only, got device {volume.device}."
        raise ValueError(msg)
    if volume.layout != torch.strided:
        msg = f"warp_affine3d supports strided Torch tensors only, got layout {volume.layout}."
        raise ValueError(msg)
    if volume.requires_grad:
        msg = "warp_affine3d supports eager tensors with requires_grad=False only."
        raise ValueError(msg)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate one forward voxel-space affine matrix and return homogeneous float64 control data."""
    raw_matrix = np.asarray(matrix)
    if np.iscomplexobj(raw_matrix):
        msg = "matrix must contain real numeric values."
        raise ValueError(msg)
    try:
        matrix_array = np.asarray(raw_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "matrix must contain finite numeric values."
        raise ValueError(msg) from error

    if matrix_array.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3] = matrix_array
    elif matrix_array.shape == (4, 4):
        homogeneous = matrix_array
        if not np.array_equal(homogeneous[3], np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)):
            msg = "A 4x4 warp_affine3d matrix must have homogeneous final row [0, 0, 0, 1]."
            raise ValueError(msg)
    else:
        msg = f"matrix must have shape (3, 4) or (4, 4), got {matrix_array.shape}."
        raise ValueError(msg)

    if not np.all(np.isfinite(homogeneous)):
        msg = "matrix must contain only finite values."
        raise ValueError(msg)
    return homogeneous


def _inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert small affine control data once and reject singular transforms."""
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        msg = "matrix must be invertible."
        raise ValueError(msg) from error
    if not np.all(np.isfinite(inverse)):
        msg = "matrix inverse must contain only finite values."
        raise ValueError(msg)
    return inverse


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
    """Return finite float32 constant-border values in channel order."""
    if border_value is None:
        return np.zeros(channels, dtype=np.float32)
    try:
        raw_values = np.asarray(border_value)
    except (TypeError, ValueError) as error:
        msg = "border_value must be a scalar or a one-dimensional numeric array."
        raise ValueError(msg) from error
    if np.iscomplexobj(raw_values):
        msg = "border_value must contain real numeric values."
        raise ValueError(msg)
    try:
        values = np.asarray(raw_values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        msg = "border_value must be a scalar or a one-dimensional numeric array."
        raise ValueError(msg) from error

    if values.ndim == 0:
        values = np.full(channels, values.item(), dtype=np.float32)
    elif values.ndim != 1 or values.shape[0] != channels:
        msg = f"border_value must be a scalar or have one value per channel ({channels}), got shape {values.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "border_value must contain only finite values."
        raise ValueError(msg)
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

    with torch.inference_mode():
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
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


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
    volume: object,
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
    supported. Torch inputs must be CPU strided eager tensors with ``requires_grad=False``.

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

    Raises:
        ValueError: If the public dtype, shape, matrix, device, autograd, interpolation, border,
            fill, or output-size contract is invalid.
        TypeError: If ``volume`` is neither ``np.ndarray`` nor ``torch.Tensor``.
    """
    output_size = _validate_size(size)
    if interpolation not in _WARP_AFFINE3D_INTERPOLATIONS:
        msg = f"warp_affine3d supports only cv2.INTER_LINEAR and cv2.INTER_NEAREST; got interpolation={interpolation}."
        raise ValueError(msg)
    if border_mode not in _WARP_AFFINE3D_BORDERS:
        msg = (
            f"warp_affine3d supports only cv2.BORDER_CONSTANT and cv2.BORDER_REPLICATE; got border_mode={border_mode}."
        )
        raise ValueError(msg)

    homogeneous_matrix = _normalize_matrix(matrix)
    if isinstance(volume, np.ndarray):
        _validate_numpy_volume(volume)
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

    if isinstance(volume, torch.Tensor):
        _validate_torch_volume(volume)
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

    msg = f"warp_affine3d supports np.ndarray and torch.Tensor, got {type(volume).__name__}."
    raise TypeError(msg)
