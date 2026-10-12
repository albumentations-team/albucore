"""Elementwise transcendental functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from albucore.utils import ImageFloat16, ImageFloat32, ImageType

from albucore._float_cast import _cast_float_numpy
from albucore.torch_backend import float16_unary_torch

__all__ = [
    "exp",
    "exp_numpy",
    "exp_opencv",
    "log",
    "log_numpy",
    "log_opencv",
    "sqrt",
    "sqrt_numpy",
    "sqrt_opencv",
]

_OPENCV_CONTIGUOUS_MIN_ELEMENTS = 4_096
_EXP_OPENCV_STRIDED_MIN_ELEMENTS = 65_536
_LOG_OPENCV_STRIDED_SINGLE_CHANNEL_MIN_ELEMENTS = 8_192
_LOG_OPENCV_STRIDED_HIGH_CHANNEL_MIN_ELEMENTS = 65_536
_FLOAT32_TINY = np.finfo(np.float32).tiny
_TORCH_FLOAT16_EXP_MIN_ELEMENTS = 12_288
_TORCH_FLOAT16_LOG_MIN_ELEMENTS = 12_288
_TORCH_FLOAT16_SQRT_MIN_ELEMENTS = 20_480


def _float16_unary_numpy(
    array: ImageFloat16,
    operation: Callable[..., np.ndarray],
    *,
    inplace: bool,
) -> ImageFloat16:
    output = array if inplace else np.empty_like(array)
    result = operation(array, dtype=np.float32, out=output, casting="unsafe")
    return cast("ImageFloat16", result)


def _float16_unary(
    array: ImageFloat16,
    operation: Literal["exp", "log", "sqrt"],
    numpy_operation: Callable[..., np.ndarray],
    *,
    inplace: bool,
) -> ImageFloat16:
    mutate = _can_mutate(array, inplace)
    if array.flags["C_CONTIGUOUS"] and array.size >= _OPENCV_CONTIGUOUS_MIN_ELEMENTS:
        if operation == "exp":
            return cast("ImageFloat16", exp_opencv(array, inplace=mutate))
        if operation == "log":
            working = _cast_float_numpy(array, np.float32)
            result = cast("ImageFloat32", log_opencv(working))
            half_result = _cast_float_numpy(result, np.float16)
            if mutate:
                np.copyto(array, half_result)
                return array
            return half_result
    minimum = {
        "exp": _TORCH_FLOAT16_EXP_MIN_ELEMENTS,
        "log": _TORCH_FLOAT16_LOG_MIN_ELEMENTS,
        "sqrt": _TORCH_FLOAT16_SQRT_MIN_ELEMENTS,
    }[operation]
    torch_result = float16_unary_torch(array, operation, minimum_elements=minimum, inplace=mutate)
    if torch_result is not None:
        return cast("ImageFloat16", torch_result)
    return _float16_unary_numpy(array, numpy_operation, inplace=mutate)


def _can_mutate(array: np.ndarray, inplace: bool) -> bool:
    return bool(inplace and array.flags["OWNDATA"] and array.flags["WRITEABLE"])


def _opencv_unary(
    array: ImageFloat16 | ImageFloat32,
    operation: Callable[..., np.ndarray],
    *,
    inplace: bool,
) -> ImageFloat16 | ImageFloat32:
    if array.dtype == np.float16:
        result = cast("ImageFloat32", _opencv_unary(_cast_float_numpy(array, np.float32), operation, inplace=False))
        if _can_mutate(array, inplace):
            np.copyto(array, result)
            return array
        return _cast_float_numpy(result, np.float16)
    if array.size == 0:
        return array if inplace else np.empty_like(array)
    result = operation(array, dst=array if inplace else None)
    if inplace:
        return array
    return cast("ImageFloat32", result.reshape(array.shape))


def exp_numpy(array: ImageFloat16 | ImageFloat32, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32:
    """Apply NumPy's elementwise exponential."""
    if array.dtype == np.float16:
        return _float16_unary_numpy(cast("ImageFloat16", array), np.exp, inplace=_can_mutate(array, inplace))
    return np.exp(array, out=array if inplace else None)


def exp_opencv(array: ImageFloat16 | ImageFloat32, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32:
    """Apply OpenCV's elementwise exponential and restore the exact input shape."""
    return _opencv_unary(array, cv2.exp, inplace=inplace)


def _torch_unary(
    array: torch.Tensor,
    operation: Callable[[torch.Tensor], torch.Tensor],
    *,
    inplace: bool,
) -> torch.Tensor:
    working = array if array.dtype != torch.uint8 else array.to(torch.float32)
    can_mutate = bool(inplace and array.dtype != torch.uint8 and array.is_leaf and array.is_contiguous())
    with torch.no_grad():
        if can_mutate:
            if operation is torch.exp:
                return working.exp_()
            if operation is torch.log:
                return working.log_()
            return working.sqrt_()
        return operation(working)


def exp(array: ImageType | torch.Tensor, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32 | torch.Tensor:
    """Compute the elementwise exponential.

    OpenCV is used for arrays with at least 4,096 elements when C-contiguous, or
    65,536 elements when strided. NumPy is faster below those conservative
    benchmark-derived thresholds.

    Args:
        array: Prevalidated float16/float32 array or uint8 image.
        inplace: NumPy views and read-only arrays produce a new array. Contiguous leaf Tensors may be mutated; aliases
            observe the same storage change.

    Returns:
        The same shape; float16 stays float16, while uint8 and float32 return float32.

    Notes:
        OpenCV finite results are float32-close to NumPy but are not guaranteed
        bit-exact. Tests use ``rtol=1e-5`` and ``atol=1e-7``.
    """
    if isinstance(array, torch.Tensor):
        return _torch_unary(array, torch.exp, inplace=inplace)
    if array.dtype == np.uint8:
        return exp_numpy(cast("ImageFloat32", array.astype(np.float32)))
    if array.dtype == np.float16:
        return _float16_unary(cast("ImageFloat16", array), "exp", np.exp, inplace=inplace)
    mutate = _can_mutate(array, inplace)
    min_elements = _OPENCV_CONTIGUOUS_MIN_ELEMENTS if array.flags["C_CONTIGUOUS"] else _EXP_OPENCV_STRIDED_MIN_ELEMENTS
    if array.size >= min_elements and (not mutate or array.flags["C_CONTIGUOUS"]):
        return exp_opencv(cast("ImageFloat32", array), inplace=mutate)
    return exp_numpy(cast("ImageFloat32", array), inplace=mutate)


def log_numpy(array: ImageFloat16 | ImageFloat32, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32:
    """Apply NumPy's elementwise natural logarithm."""
    if array.dtype == np.float16:
        return _float16_unary_numpy(cast("ImageFloat16", array), np.log, inplace=_can_mutate(array, inplace))
    return np.log(array, out=array if inplace else None)


def log_opencv(array: ImageFloat16 | ImageFloat32, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32:
    """Apply OpenCV log when it matches NumPy, otherwise fall back to NumPy.

    OpenCV's log differs for negative values, zero, subnormals, NaN, and infinity.
    The two reductions preserve NumPy semantics without paying for full-size masks.
    """
    if array.size == 0:
        return log_numpy(array, inplace=inplace)
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < _FLOAT32_TINY or not np.isfinite(minimum) or not np.isfinite(maximum):
        return log_numpy(array, inplace=inplace)
    return _opencv_unary(array, cv2.log, inplace=inplace)


def log(array: ImageType | torch.Tensor, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32 | torch.Tensor:
    """Compute the NumPy-compatible natural logarithm.

    Large C-contiguous arrays, strided single-channel arrays, and strided arrays
    with at least eight channels use OpenCV only after verifying that all values
    are finite, positive, and normal. Other inputs use NumPy, preserving its
    behavior for negative values, zero, subnormals, NaN, and infinity.

    Args:
        array: Prevalidated float16/float32 array or uint8 image.
        inplace: NumPy views and read-only arrays produce a new array. Contiguous leaf Tensors may be mutated; aliases
            observe the same storage change.

    Returns:
        The same shape; float16 stays float16, while uint8 and float32 return float32.

    Notes:
        Eligible finite results are float32-close to NumPy but are not guaranteed
        bit-exact. Tests use ``rtol=1e-5`` and ``atol=1e-7``. Special values follow
        NumPy exactly because they bypass OpenCV.
    """
    if isinstance(array, torch.Tensor):
        return _torch_unary(array, torch.log, inplace=inplace)
    if array.dtype == np.uint8:
        return log_numpy(cast("ImageFloat32", array.astype(np.float32)))
    if array.dtype == np.float16:
        return _float16_unary(cast("ImageFloat16", array), "log", np.log, inplace=inplace)
    mutate = _can_mutate(array, inplace)
    contiguous_candidate = array.flags["C_CONTIGUOUS"] and array.size >= _OPENCV_CONTIGUOUS_MIN_ELEMENTS
    single_channel = array.ndim == 2 or (array.ndim >= 3 and array.shape[-1] == 1)
    high_channel = array.ndim >= 3 and array.shape[-1] >= 8
    strided_min_elements = (
        _LOG_OPENCV_STRIDED_SINGLE_CHANNEL_MIN_ELEMENTS
        if single_channel
        else _LOG_OPENCV_STRIDED_HIGH_CHANNEL_MIN_ELEMENTS
        if high_channel
        else None
    )
    strided_candidate = (
        not array.flags["C_CONTIGUOUS"]
        and strided_min_elements is not None
        and array.size >= strided_min_elements
        and not mutate
    )
    if contiguous_candidate or strided_candidate:
        return log_opencv(cast("ImageFloat32", array), inplace=mutate)
    return log_numpy(cast("ImageFloat32", array), inplace=mutate)


def sqrt_numpy(array: ImageFloat16 | ImageFloat32, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32:
    """Apply NumPy's elementwise square root."""
    if array.dtype == np.float16:
        return _float16_unary_numpy(cast("ImageFloat16", array), np.sqrt, inplace=_can_mutate(array, inplace))
    return np.sqrt(array, out=array if inplace else None)


def sqrt_opencv(array: ImageFloat16 | ImageFloat32, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32:
    """Apply OpenCV's elementwise square root and restore the exact input shape."""
    return _opencv_unary(array, cv2.sqrt, inplace=inplace)


def sqrt(array: ImageType | torch.Tensor, *, inplace: bool = False) -> ImageFloat16 | ImageFloat32 | torch.Tensor:
    """Compute the NumPy-compatible elementwise square root.

    NumPy is used for all layouts: benchmarks found no durable OpenCV win. With
    ``inplace=True``, NumPy's ``out=`` avoids allocation for an owned writable
    buffer.

    Args:
        array: Prevalidated float16/float32 array or uint8 image.
        inplace: NumPy views and read-only arrays produce a new array. Contiguous leaf Tensors may be mutated; aliases
            observe the same storage change.

    Returns:
        The same shape; float16 stays float16, while uint8 and float32 return float32.

    """
    if isinstance(array, torch.Tensor):
        return _torch_unary(array, torch.sqrt, inplace=inplace)
    if array.dtype == np.uint8:
        return sqrt_numpy(cast("ImageFloat32", array.astype(np.float32)))
    if array.dtype == np.float16:
        return _float16_unary(cast("ImageFloat16", array), "sqrt", np.sqrt, inplace=inplace)
    return sqrt_numpy(cast("ImageFloat32", array), inplace=_can_mutate(array, inplace))
