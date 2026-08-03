"""CPU Torch kernels for large NumPy arrays.

Torch is an Albucore dependency. The helpers activate only for large, writable
arrays with compatible strides, so the NumPy wrapper and Torch Tensor can share
CPU storage without an input repair copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from albucore.utils import ImageFloat32, ImageType, ImageUInt8

TORCH_MIN_ELEMENTS = 512 * 1024
TORCH_CPU_BACKEND_ENABLED = True


def _is_torch_compatible_array(array: np.ndarray) -> bool:
    """Check whether a NumPy array can use a measured Torch CPU route without an input repair copy."""
    return (
        TORCH_CPU_BACKEND_ENABLED
        and array.size >= TORCH_MIN_ELEMENTS
        and array.flags.writeable
        and all(stride >= 0 for stride in array.strides)
    )


def _parameter_tensor(value: float | np.ndarray, ndim: int, channels: int) -> torch.Tensor:
    """Create a float32 scalar or broadcastable channel-last Torch parameter."""
    if not isinstance(value, np.ndarray):
        return torch.tensor(value, dtype=torch.float32)

    value_float32 = value.astype(np.float32, copy=False)
    if not value_float32.flags.writeable:
        value_float32 = value_float32.copy()
    tensor = torch.from_numpy(value_float32)
    return tensor.reshape((1,) * (ndim - 1) + (channels,)) if value_float32.shape == (channels,) else tensor


def from_float_uint8_torch(img: ImageFloat32, max_value: float) -> ImageUInt8 | None:
    """Scale, round, and saturate float32 input to uint8 with a Torch CPU kernel."""
    if img.dtype != np.float32 or not _is_torch_compatible_array(img):
        return None
    try:
        result = torch.round(torch.from_numpy(img) * max_value).clamp_(0.0, 255.0).to(torch.uint8)
    except (RuntimeError, TypeError, ValueError):
        return None
    return cast("ImageUInt8", result.numpy())


def multiply_add_float32_torch(img: ImageFloat32, factor: float, value: float) -> ImageFloat32 | None:
    """Compute ``img * factor + value`` using Torch without copying the input buffer."""
    if img.dtype != np.float32 or not _is_torch_compatible_array(img):
        return None
    try:
        result = torch.from_numpy(img) * factor + value
    except (RuntimeError, TypeError, ValueError):
        return None
    return cast("ImageFloat32", result.numpy())


def normalize_float32_torch(
    img: ImageFloat32,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
) -> ImageFloat32 | None:
    """Compute ``img * denominator - mean * denominator`` for multi-channel float32 input."""
    if img.dtype != np.float32 or img.shape[-1] <= 1 or not _is_torch_compatible_array(img):
        return None
    try:
        mean_tensor = _parameter_tensor(mean, img.ndim, img.shape[-1])
        denominator_tensor = _parameter_tensor(denominator, img.ndim, img.shape[-1])
        result = torch.addcmul(-mean_tensor * denominator_tensor, torch.from_numpy(img), denominator_tensor)
    except (RuntimeError, TypeError, ValueError):
        return None
    return cast("ImageFloat32", result.numpy())


def reduce_sum_torch(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> np.generic | np.ndarray | None:
    """Match Albucore's uint64/float64 sum accumulator for a large Torch-compatible array."""
    if not _is_torch_compatible_array(arr):
        return None
    if arr.dtype == np.uint8:
        torch_dtype = torch.int64
        output_dtype: type[np.generic] = np.uint64
    elif arr.dtype == np.float32:
        torch_dtype = torch.float64
        output_dtype = np.float64
    else:
        return None
    try:
        result = torch.sum(torch.from_numpy(arr), dim=axes, dtype=torch_dtype, keepdim=keepdims).numpy()
    except (RuntimeError, TypeError, ValueError):
        return None
    if output_dtype is np.uint64:
        result = result.view(np.uint64)
    if len(axes) == arr.ndim and not keepdims:
        return output_dtype(result)
    return np.asarray(result)
