"""Constant spatial padding for single volumes and masks."""

from typing import overload

import numpy as np
import torch
import torch.nn.functional as torch_f

__all__ = ["pad3d"]

_TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, type[np.generic]] = {
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.float32: np.float32,
}
_TORCH_MIN_ELEMENTS = 524_288
_TORCH_UINT8_STRIDED_MIN_ELEMENTS = 1_048_576
_TORCH_FLOAT32_STRIDED_MAX_ELEMENTS = 4_194_304


def _cast_pad_value(value: float, dtype: np.dtype | type[np.generic]) -> float:
    # Scalar assignment matches np.pad, including uint8 wrapping and int16 overflow.
    fill = np.empty((), dtype=dtype)
    fill[()] = np.asarray(value)[()]
    return float(fill)


def _pad3d_numpy(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...],
) -> np.ndarray:
    front, back, top, bottom, left, right = padding
    if isinstance(value, tuple):
        return np.pad(volume, ((front, back), (top, bottom), (left, right), (0, 0)), constant_values=value)
    depth, height, width, channels = volume.shape
    result = np.full(
        (depth + front + back, height + top + bottom, width + left + right, channels),
        _cast_pad_value(value, volume.dtype),
        dtype=volume.dtype,
        order="F" if volume.flags.f_contiguous else "C",
    )
    result[front : front + depth, top : top + height, left : left + width] = volume
    return result


@torch.no_grad()
def _pad3d_torch(
    volume: torch.Tensor,
    padding: tuple[int, int, int, int, int, int],
    value: float,
    channel_last: bool = False,
) -> torch.Tensor:
    front, back, top, bottom, left, right = padding
    fill = _cast_pad_value(value, _TORCH_TO_NUMPY_DTYPE[volume.dtype])
    torch_padding: tuple[int, ...] = (left, right, top, bottom, front, back)
    if channel_last:
        torch_padding = (0, 0, *torch_padding)
    return torch_f.pad(volume, torch_padding, value=fill)


def _numpy_pad_use_torch(volume: np.ndarray) -> bool:
    # Thresholds come from the paired full-path measurements in benchmarks/results/benchmark_pad3d.md.
    if volume.size < _TORCH_MIN_ELEMENTS or volume.dtype == np.int16 or volume.flags.f_contiguous:
        return False
    torch_is_faster = volume.shape[-1] == 1 or (
        not volume.flags.c_contiguous
        and (
            (volume.dtype == np.uint8 and volume.size >= _TORCH_UINT8_STRIDED_MIN_ELEMENTS)
            or (volume.dtype == np.float32 and volume.size <= _TORCH_FLOAT32_STRIDED_MAX_ELEMENTS)
        )
    )
    return (
        torch_is_faster
        and volume.flags.writeable
        and all(stride >= 0 and stride % volume.itemsize == 0 for stride in volume.strides)
    )


@overload
def pad3d(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...] = 0,
) -> np.ndarray: ...


@overload
def pad3d(
    volume: torch.Tensor,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...] = 0,
) -> torch.Tensor: ...


def pad3d(
    volume: np.ndarray | torch.Tensor,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...] = 0,
) -> np.ndarray | torch.Tensor:
    """Pad the spatial axes of one prevalidated volume with constant values.

    Supports uint8 and float32 volumes, and int16 masks. Callers validate layout,
    dtype, CPU placement, ``requires_grad=False``, and nonnegative padding before
    dispatch. Channel-less DHW inputs must gain a channel axis at the caller boundary.

    Args:
        volume: NumPy ``(D, H, W, C)`` array or CPU Torch ``(C, D, H, W)`` tensor.
            NumPy strides may be positive or negative, and arrays may be read-only.
            Non-contiguous strided Tensors are supported.
        padding: ``(front, back, top, bottom, left, right)`` spatial border widths.
        value: Scalar fill, a one-element tuple, or a ``(before, after)`` tuple
            shared by the spatial axes, following ``np.pad``. At intersecting
            borders, width takes precedence over height, then depth. Tuple values
            are not per-channel fills. Casts follow NumPy scalar assignment:
            integer fractions truncate and uint8 values wrap instead of clipping.

    Returns:
        The input itself for all-zero padding; otherwise an independent volume
        with the same container, dtype, channel count, and layout. Tensor outputs
        can feed subsequent training operations. The input is never modified.

    """
    if not any(padding):
        return volume
    if isinstance(volume, np.ndarray):
        if not isinstance(value, tuple) and _numpy_pad_use_torch(volume):
            return np.asarray(_pad3d_torch(torch.from_numpy(volume), padding, value, channel_last=True).numpy())
        return _pad3d_numpy(volume, padding, value)
    # Interleaved channels make the native CDHW copy strided; retain DHWC storage.
    if isinstance(value, tuple) or (volume.shape[0] > 1 and volume.stride(0) == 1):
        result = _pad3d_numpy(np.asarray(volume.permute(1, 2, 3, 0).numpy()), padding, value)
        return torch.from_numpy(result).permute(3, 0, 1, 2)
    return _pad3d_torch(volume, padding, value)
