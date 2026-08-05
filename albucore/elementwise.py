"""Elementwise transcendental functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from albucore.utils import _validate_image_rank

if TYPE_CHECKING:
    from collections.abc import Callable

    from albucore.utils import ImageFloat32

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


def _validate_float32(array: np.ndarray) -> None:
    _validate_image_rank(array)
    if array.dtype != np.float32:
        raise ValueError(f"Elementwise operation supports only float32 arrays, got {array.dtype}.")


def _can_mutate(array: np.ndarray, inplace: bool) -> bool:
    return bool(inplace and array.flags["OWNDATA"] and array.flags["WRITEABLE"])


def _opencv_unary(
    array: ImageFloat32,
    operation: Callable[..., np.ndarray],
    *,
    inplace: bool,
) -> ImageFloat32:
    if array.size == 0:
        return array if inplace else np.empty_like(array)
    result = operation(array, dst=array if inplace else None)
    if inplace:
        return array
    return cast("ImageFloat32", result.reshape(array.shape))


def exp_numpy(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Apply NumPy's elementwise exponential."""
    return np.exp(array, out=array if inplace else None)


def exp_opencv(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Apply OpenCV's elementwise exponential and restore the exact input shape."""
    return _opencv_unary(array, cv2.exp, inplace=inplace)


def exp(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Compute the elementwise exponential of a float32 array.

    OpenCV is used for arrays with at least 4,096 elements when C-contiguous, or
    65,536 elements when strided. NumPy is faster below those conservative
    benchmark-derived thresholds.

    Args:
        array: Float32 array with at most four dimensions. Image-like inputs use channel-last shapes.
        inplace: Reuse ``array`` only when it owns a writable buffer. Views and
            read-only arrays are never mutated and produce a new array instead.

    Returns:
        The elementwise exponential with the same shape and float32 dtype.

    Raises:
        ValueError: If ``array`` is not float32 or has more than four dimensions.

    Notes:
        OpenCV finite results are float32-close to NumPy but are not guaranteed
        bit-exact. Tests use ``rtol=1e-5`` and ``atol=1e-7``.
    """
    _validate_float32(array)
    mutate = _can_mutate(array, inplace)
    min_elements = _OPENCV_CONTIGUOUS_MIN_ELEMENTS if array.flags["C_CONTIGUOUS"] else _EXP_OPENCV_STRIDED_MIN_ELEMENTS
    if array.size >= min_elements and (not mutate or array.flags["C_CONTIGUOUS"]):
        return exp_opencv(array, inplace=mutate)
    return exp_numpy(array, inplace=mutate)


def log_numpy(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Apply NumPy's elementwise natural logarithm."""
    return np.log(array, out=array if inplace else None)


def log_opencv(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
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


def log(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Compute the NumPy-compatible natural logarithm of a float32 array.

    Large C-contiguous arrays, strided single-channel arrays, and strided arrays
    with at least eight channels use OpenCV only after verifying that all values
    are finite, positive, and normal. Other inputs use NumPy, preserving its
    behavior for negative values, zero, subnormals, NaN, and infinity.

    Args:
        array: Float32 array with at most four dimensions. Image-like inputs use channel-last shapes.
        inplace: Reuse ``array`` only when it owns a writable buffer. Views and
            read-only arrays are never mutated and produce a new array instead.

    Returns:
        The elementwise natural logarithm with the same shape and float32 dtype.

    Raises:
        ValueError: If ``array`` is not float32 or has more than four dimensions.

    Notes:
        Eligible finite results are float32-close to NumPy but are not guaranteed
        bit-exact. Tests use ``rtol=1e-5`` and ``atol=1e-7``. Special values follow
        NumPy exactly because they bypass OpenCV.
    """
    _validate_float32(array)
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
        return log_opencv(array, inplace=mutate)
    return log_numpy(array, inplace=mutate)


def sqrt_numpy(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Apply NumPy's elementwise square root."""
    return np.sqrt(array, out=array if inplace else None)


def sqrt_opencv(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Apply OpenCV's elementwise square root and restore the exact input shape."""
    return _opencv_unary(array, cv2.sqrt, inplace=inplace)


def sqrt(array: ImageFloat32, *, inplace: bool = False) -> ImageFloat32:
    """Compute the NumPy-compatible elementwise square root of a float32 array.

    NumPy is used for all layouts: benchmarks found no durable OpenCV win. With
    ``inplace=True``, NumPy's ``out=`` avoids allocation for an owned writable
    buffer.

    Args:
        array: Float32 array with at most four dimensions. Image-like inputs use channel-last shapes.
        inplace: Reuse ``array`` only when it owns a writable buffer. Views and
            read-only arrays are never mutated and produce a new array instead.

    Returns:
        The elementwise square root with the same shape and float32 dtype.

    Raises:
        ValueError: If ``array`` is not float32 or has more than four dimensions.
    """
    _validate_float32(array)
    return sqrt_numpy(array, inplace=_can_mutate(array, inplace))
