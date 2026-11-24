"""Constant spatial padding for single volumes and masks."""

import sys
from typing import overload

import numpy as np
import torch
import torch.nn.functional as torch_f

from albucore._types import ArrayT
from albucore.tensor_bridge import numpy_to_tensor_layout, tensor_to_numpy_channel_last
from albucore.utils import _LINUX_X86_64

__all__ = ["pad3d"]

_TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, type[np.generic]] = {
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.float16: np.float16,
    torch.float32: np.float32,
}
_TORCH_MIN_ELEMENTS = 524_288
_TORCH_UINT8_STRIDED_MIN_ELEMENTS = 1_048_576
_TORCH_FLOAT32_STRIDED_MAX_ELEMENTS = 4_194_304
_NUMPY_FACES_MAX_EXPANSION = 1.1
# Float C-contiguous volumes with C>1, including channel-last float32 Tensors: faces beat the full-array fill from
# these sizes for every pad width.
# float16 faces lose 1.1-1.4x on macOS at every size, so only Linux uses them.
_FLOAT32_FACES_MIN_ELEMENTS = 245_760 if _LINUX_X86_64 else 1_310_720
_FLOAT16_FACES_MIN_ELEMENTS = 983_040 if _LINUX_X86_64 else sys.maxsize
# C-contiguous single-channel volumes: on Linux NumPy beats Torch at every size measured (to 1.31M elements).
_LINUX_TORCH_SINGLE_CHANNEL_MIN_ELEMENTS = 2_097_152


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


def _numpy_pad_use_faces(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
) -> bool:
    if not volume.flags.c_contiguous:
        return False
    front, back, top, bottom, left, right = padding
    depth, height, width, channels = volume.shape
    if volume.dtype == np.float32 and channels > 1 and volume.size >= _FLOAT32_FACES_MIN_ELEMENTS:
        return True
    if volume.dtype == np.float16 and channels > 1 and volume.size >= _FLOAT16_FACES_MIN_ELEMENTS:
        return True
    if channels == 1:
        return bool(
            _LINUX_X86_64
            and volume.dtype in (np.float16, np.float32)
            and _TORCH_MIN_ELEMENTS <= volume.size < _LINUX_TORCH_SINGLE_CHANNEL_MIN_ELEMENTS,
        )
    padded_elements = (depth + front + back) * (height + top + bottom) * (width + left + right) * channels
    return bool(volume.size >= _TORCH_MIN_ELEMENTS and padded_elements <= _NUMPY_FACES_MAX_EXPANSION * volume.size)


def _pad3d_numpy_faces(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
    value: float,
) -> np.ndarray:
    front, back, top, bottom, left, right = padding
    depth, height, width, _ = volume.shape
    padded = np.empty(
        (depth + front + back, height + top + bottom, width + left + right, volume.shape[-1]),
        dtype=volume.dtype,
    )
    fill = _cast_pad_value(value, volume.dtype)
    padded[:front] = fill
    padded[front + depth :] = fill
    center_depth = padded[front : front + depth]
    center_depth[:, :top] = fill
    center_depth[:, top + height :] = fill
    center_height = center_depth[:, top : top + height]
    center_height[:, :, :left] = fill
    center_height[:, :, left + width :] = fill
    center_height[:, :, left : left + width] = volume
    return padded


def _tensor_pad_use_faces(volume: np.ndarray) -> bool:
    """Select channel-last float32 Tensor storage whose NumPy view is C-contiguous and large enough for faces."""
    return bool(
        volume.dtype == np.float32
        and volume.shape[-1] > 1
        and volume.flags.c_contiguous
        and volume.size >= _FLOAT32_FACES_MIN_ELEMENTS,
    )


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
    numpy_is_faster_single_channel = (
        _LINUX_X86_64 and volume.flags.c_contiguous and volume.size < _LINUX_TORCH_SINGLE_CHANNEL_MIN_ELEMENTS
    )
    torch_is_faster = (volume.shape[-1] == 1 and not numpy_is_faster_single_channel) or (
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


def _pad3d_tensor_via_numpy(
    volume: torch.Tensor,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...],
) -> torch.Tensor:
    numpy_volume = tensor_to_numpy_channel_last(volume, "CDHW")
    if not isinstance(value, tuple) and _tensor_pad_use_faces(numpy_volume):
        return numpy_to_tensor_layout(_pad3d_numpy_faces(numpy_volume, padding, value), "CDHW")
    return numpy_to_tensor_layout(_pad3d_numpy(numpy_volume, padding, value), "CDHW")


@overload
def pad3d(
    volume: ArrayT,
    padding: tuple[int, int, int, int, int, int],
    value: float | tuple[float, ...] = 0,
) -> ArrayT: ...


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

    Supports uint8, float16, and float32 volumes, and int16 masks. Callers validate layout,
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
        if not isinstance(value, tuple) and _numpy_pad_use_faces(volume, padding):
            return _pad3d_numpy_faces(volume, padding, value)
        if not isinstance(value, tuple) and _numpy_pad_use_torch(volume):
            return np.asarray(_pad3d_torch(torch.from_numpy(volume), padding, value, channel_last=True).numpy())
        return _pad3d_numpy(volume, padding, value)
    # Interleaved channels make the native CDHW copy strided; retain DHWC storage.
    if isinstance(value, tuple) or (volume.shape[0] > 1 and volume.stride(0) == 1):
        return _pad3d_tensor_via_numpy(volume, padding, value)
    return _pad3d_torch(volume, padding, value)
