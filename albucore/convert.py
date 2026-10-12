"""Convert image arrays among uint8, float16, and float32."""

import platform
import sys
from typing import Any, cast

import cv2
import numkong as nk
import numpy as np
import torch

from albucore._float_cast import _cast_float_numpy
from albucore.decorators import preserve_channel_dim
from albucore.torch_backend import from_float_uint8_torch
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ImageFloat16,
    ImageFloat32,
    ImageType,
    ImageUInt8,
    clip,
    get_max_value,
    get_num_channels,
)

_NUMPY_TO_TORCH_DTYPE = {
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
}
_MACOS_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"
_TENSOR_NUMPY_CAST_MIN_ELEMENTS = 61_440


def to_float_numpy(img: ImageType, max_value: float | None = None) -> ImageFloat16 | ImageFloat32:
    if img.dtype == np.float16:
        return cast("ImageFloat16", img)
    if max_value is None:
        max_value = get_max_value(img.dtype)
    return (img / max_value).astype(np.float32, copy=False)


@preserve_channel_dim
def to_float_opencv(img: ImageType, max_value: float | None = None) -> ImageFloat16 | ImageFloat32:
    if img.dtype == np.float16:
        return cast("ImageFloat16", img)
    if max_value is None:
        max_value = get_max_value(img.dtype)

    img_float = img.astype(np.float32, copy=False)

    if get_num_channels(img) > MAX_OPENCV_WORKING_CHANNELS:
        max_value_array = np.full_like(img_float, max_value)
        return cast("ImageFloat32", cv2.divide(img_float, max_value_array))

    # Keep scalar fast path (no full-image temporary); typing narrowed via Any.
    return cast("ImageFloat32", cv2.divide(img_float, cast("Any", max_value)))


@preserve_channel_dim
def to_float_lut(img: ImageUInt8, max_value: float | None = None) -> ImageFloat32:
    if max_value is None:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    lut = (np.arange(256, dtype=np.float32) / max_value).astype(np.float32)
    return cast("ImageFloat32", cv2.LUT(img, lut))


def to_float(
    img: ImageType | torch.Tensor,
    max_value: float | None = None,
) -> ImageFloat16 | ImageFloat32 | torch.Tensor:
    """Convert uint8 to normalized float32 and preserve floating inputs.

    Routing:
    - **float16/float32**: no-op (returned as-is).
    - **uint8**: LUT via ``cv2.LUT`` — 256-element float32 lookup table; fastest path.
    - **other dtypes**: NumPy division (``img / max_value``).

    Alternative: ``to_float_lut``, ``to_float_opencv``, ``to_float_numpy`` for explicit backends.

    Args:
        img: Input image of any dtype, shape ``(H, W, C)`` or batch/volume.
        max_value: Divisor. Defaults to dtype max (255 for uint8, 65535 for uint16, etc.).

    Returns:
        Float16 input remains float16; float32 and uint8 inputs return float32. Uint8 values are
        divided by ``max_value``; floating inputs are returned unchanged.
    """
    if isinstance(img, torch.Tensor):
        if img.dtype in (torch.float16, torch.float32):
            return img
        divisor = MAX_VALUES_BY_DTYPE[np.dtype(np.uint8)] if max_value is None else max_value
        return img.to(torch.float32).div_(divisor)

    if img.dtype in (np.float16, np.float32):
        return cast("ImageFloat16 | ImageFloat32", img)
    if img.dtype == np.uint8:
        return to_float_lut(cast("ImageUInt8", img), max_value)
    return to_float_numpy(img, max_value)


def from_float_numpy(img: ImageFloat32, target_dtype: np.dtype, max_value: float | None = None) -> ImageType:
    if target_dtype in (np.dtype(np.float16), np.dtype(np.float32)):
        return cast("ImageType", img.astype(target_dtype, copy=False))
    if max_value is None:
        max_value = get_max_value(target_dtype)
    return clip(np.rint(img * max_value), target_dtype, inplace=True)


def _from_float_uint8_numkong(img: ImageFloat32, max_value: float) -> ImageUInt8:
    """Scale, round, and saturate float32 input with one float32 work buffer."""
    rounded = np.empty_like(img)
    np.multiply(img, max_value, out=rounded)
    np.rint(rounded, out=rounded)
    result = np.empty(img.shape, dtype=np.uint8)
    # NumKong's runtime API exposes ``astype``; its type stubs may lag the installed release.
    cast("Any", nk).astype(rounded, "uint8", out=result)
    return result


@preserve_channel_dim
def from_float_opencv(img: ImageFloat32, target_dtype: np.dtype, max_value: float | None = None) -> ImageType:
    if target_dtype in (np.dtype(np.float16), np.dtype(np.float32)):
        return cast("ImageType", img.astype(target_dtype, copy=False))
    if max_value is None:
        max_value = get_max_value(target_dtype)

    img_float = img.astype(np.float32, copy=False)

    num_channels = get_num_channels(img)

    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        # NumPy rint beats ``cv2.multiply`` + clip on router (e.g. 128²x9 float32 → uint8).
        return clip(np.rint(img_float * max_value), target_dtype, inplace=False)

    # C<=4: NumPy multiply + rint (faster than cv2.multiply on typical sizes; cv2.multiply(float (H,W,1), scalar)
    # does not match NumPy elementwise semantics).
    return clip(np.rint(img * max_value), target_dtype, inplace=False)


def _from_float_torch(
    img: torch.Tensor,
    target_dtype: np.dtype | type[np.generic] | torch.dtype,
    max_value: float | None,
) -> torch.Tensor:
    if target_dtype in (torch.uint8, torch.float16, torch.float32):
        torch_dtype = target_dtype
    else:
        torch_dtype = _NUMPY_TO_TORCH_DTYPE[np.dtype(target_dtype)]
    if torch_dtype in (torch.float16, torch.float32):
        if (
            _MACOS_ARM64
            and img.dtype != torch_dtype
            and img.numel() >= _TENSOR_NUMPY_CAST_MIN_ELEMENTS
            and img.is_contiguous()
        ):
            numpy_dtype = np.float32 if torch_dtype == torch.float32 else np.float16
            with np.errstate(all="ignore"):
                result = _cast_float_numpy(img.numpy(), numpy_dtype)
            return torch.from_numpy(result)
        return img.to(dtype=torch_dtype)

    scale = get_max_value(np.dtype(np.uint8)) if max_value is None else max_value
    if img.dtype == torch.float16 and img.is_contiguous():
        with np.errstate(over="ignore", invalid="ignore"):
            half_scale = np.float16(scale)
        if np.isfinite(half_scale) and float(half_scale) == float(scale):
            return torch.round(img * float(half_scale)).clamp_(0.0, 255.0).to(torch.uint8)
    return torch.round(img.to(torch.float32) * scale).clamp_(0.0, 255.0).to(torch.uint8)


def from_float(
    img: ImageFloat16 | ImageFloat32 | torch.Tensor,
    target_dtype: np.dtype | type[np.generic] | torch.dtype,
    max_value: float | None = None,
) -> ImageType | torch.Tensor:
    """Convert a floating image to the requested storage dtype.

    Floating targets use a dtype cast. Integer targets multiply by ``max_value``, round with
    ``np.rint``, and clip to the target dtype range.

    Routing:
    - **target == float16/float32**: cast without scaling or clipping.
    - **NumPy float16 → uint8**: widen to float32 before scale, rounding, and saturation to preserve the exact
      float32 result.
    - **contiguous float16 Tensor → uint8**: use half arithmetic when the scale is exactly representable; the result
      differs from float32 rounding by at most one uint8 unit.
    - **large float32 → uint8**: a zero-copy CPU Torch input wrapper, then Torch scale/round/clip/cast.
    - **remaining float32 → uint8**: NumPy scale and round in one reusable float32 buffer, then NumKong's
      saturating buffer-first cast. This removes the separate full-array clip and output cast.
    - **other target dtypes**: use the existing NumPy/OpenCV scale, round, and clip route.

    Alternative: ``from_float_numpy``, ``from_float_opencv`` for explicit backends.

    Args:
        img: Float16/float32 image, shape ``(H, W, C)`` or batch/volume.
        target_dtype: Output dtype (e.g. ``np.uint8``).
        max_value: Multiplier before rounding. Defaults to dtype max (255 for uint8, etc.).

    Returns:
        Image in ``target_dtype``, same spatial shape, values clipped to dtype range.
    """
    if isinstance(img, torch.Tensor):
        return _from_float_torch(img, target_dtype, max_value)

    if target_dtype in (np.dtype(np.float16), np.dtype(np.float32), np.float16, np.float32):
        numpy_dtype = np.dtype(target_dtype)
        return _cast_float_numpy(img, numpy_dtype)

    numpy_target_dtype = np.dtype(target_dtype)
    if img.dtype in (np.float16, np.float32) and target_dtype == np.uint8:
        resolved_max = get_max_value(numpy_target_dtype) if max_value is None else max_value
        torch_result = from_float_uint8_torch(cast("ImageFloat32", img), resolved_max)
        if torch_result is not None:
            return torch_result
        if img.dtype == np.float16:
            img = cast("ImageFloat32", img.astype(np.float32))
        return _from_float_uint8_numkong(cast("ImageFloat32", img), resolved_max)

    if img.dtype == np.float16:
        img = cast("ImageFloat32", img.astype(np.float32))

    return from_float_opencv(cast("ImageFloat32", img), numpy_target_dtype, max_value)
