"""Elementwise transcendental functions with explicit output storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast, overload

import cv2
import numpy as np
import torch

from albucore._float_cast import _cast_float_numpy
from albucore.torch_backend import _float16_unary_torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray

    from albucore._types import ImageType

__all__ = ["exp", "log", "sqrt"]

_FloatDTypeT = TypeVar("_FloatDTypeT", np.float16, np.float32)
_OPENCV_CONTIGUOUS_MIN_ELEMENTS = 4_096
_EXP_OPENCV_STRIDED_MIN_ELEMENTS = 65_536
_LOG_OPENCV_STRIDED_SINGLE_CHANNEL_MIN_ELEMENTS = 8_192
_LOG_OPENCV_STRIDED_HIGH_CHANNEL_MIN_ELEMENTS = 65_536
_FLOAT32_TINY = np.finfo(np.float32).tiny
_TORCH_FLOAT16_MIN_ELEMENTS = {"exp": 12_288, "log": 12_288, "sqrt": 20_480}


def _numpy_unary(
    array: NDArray[_FloatDTypeT],
    operation: Callable[..., np.ndarray],
    *,
    out: NDArray[_FloatDTypeT] | None = None,
) -> NDArray[_FloatDTypeT]:
    if array.dtype == np.float16:
        destination = np.empty_like(array) if out is None else out
        return cast("NDArray[_FloatDTypeT]", operation(array, dtype=np.float32, out=destination, casting="unsafe"))
    return cast("NDArray[_FloatDTypeT]", operation(array, out=out))


def _opencv_unary(
    array: NDArray[np.float32],
    operation: Callable[..., np.ndarray],
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    if array.size == 0:
        return np.empty_like(array) if out is None else out
    if out is not None:
        operation(array.reshape(-1), dst=out.reshape(-1))
        return out
    result = operation(array)
    return cast("NDArray[np.float32]", result.reshape(array.shape))


def _log_opencv(array: NDArray[np.float32], *, out: NDArray[np.float32] | None = None) -> NDArray[np.float32]:
    # OpenCV log differs from NumPy for zero, negative, subnormal, and non-finite values.
    if array.size == 0:
        return _numpy_unary(array, np.log, out=out)
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < _FLOAT32_TINY or not np.isfinite(minimum) or not np.isfinite(maximum):
        return _numpy_unary(array, np.log, out=out)
    return _opencv_unary(array, cv2.log, out=out)


def _float16_unary(
    array: NDArray[np.float16],
    operation: Literal["exp", "log", "sqrt"],
    numpy_operation: Callable[..., np.ndarray],
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16]:
    if array.flags.c_contiguous and array.size >= _OPENCV_CONTIGUOUS_MIN_ELEMENTS and operation != "sqrt":
        if operation == "log":
            working = _cast_float_numpy(array, np.float32)
            result = _log_opencv(working)
        else:
            result = _opencv_unary(_cast_float_numpy(array, np.float32), cv2.exp)
        return _cast_float_numpy(result, np.float16, out=out)
    torch_result = _float16_unary_torch(
        array,
        operation,
        minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS[operation],
        out=out,
    )
    if torch_result is not None:
        return torch_result
    return _numpy_unary(array, numpy_operation, out=out)


def _torch_unary(
    array: torch.Tensor,
    operation: Callable[..., torch.Tensor],
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    working = array if array.dtype != torch.uint8 else array.to(torch.float32)
    with torch.no_grad():
        if out is array:
            if operation is torch.exp:
                return working.exp_()
            if operation is torch.log:
                return working.log_()
            return working.sqrt_()
        return operation(working) if out is None else operation(working, out=out)


@overload
def exp(array: NDArray[np.float16], *, out: NDArray[np.float16] | None = None) -> NDArray[np.float16]: ...


@overload
def exp(
    array: NDArray[np.uint8] | NDArray[np.float32],
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def exp(array: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor: ...


def exp(
    array: ImageType | torch.Tensor,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Compute the elementwise exponential without writing input unless supplied as output.

    Args:
        array: Prevalidated uint8, float16, or float32 array.
        out: Optional caller-owned output of the result dtype and shape. May be the floating input.
            The caller validates strides, writeability, and supported overlap.

    Returns:
        The same shape; float16 stays float16, while uint8 and float32 return float32.
        A supplied output is returned by identity.

    Notes:
        OpenCV finite results match NumPy within rtol=1e-5 and atol=1e-7.
    """
    if isinstance(array, torch.Tensor):
        return _torch_unary(array, torch.exp, out=cast("torch.Tensor | None", out))
    output = cast("NDArray[np.float16] | NDArray[np.float32] | None", out)
    if array.dtype == np.uint8:
        return _numpy_unary(
            cast("NDArray[np.float32]", array.astype(np.float32)),
            np.exp,
            out=cast("NDArray[np.float32] | None", output),
        )
    if array.dtype == np.float16:
        return _float16_unary(
            cast("NDArray[np.float16]", array),
            "exp",
            np.exp,
            out=cast("NDArray[np.float16] | None", output),
        )
    destination = cast("NDArray[np.float32] | None", output)
    minimum = _OPENCV_CONTIGUOUS_MIN_ELEMENTS if array.flags.c_contiguous else _EXP_OPENCV_STRIDED_MIN_ELEMENTS
    if array.size >= minimum and (destination is None or (array.flags.c_contiguous and destination.flags.c_contiguous)):
        return _opencv_unary(cast("NDArray[np.float32]", array), cv2.exp, out=destination)
    return _numpy_unary(cast("NDArray[np.float32]", array), np.exp, out=destination)


@overload
def log(array: NDArray[np.float16], *, out: NDArray[np.float16] | None = None) -> NDArray[np.float16]: ...


@overload
def log(
    array: NDArray[np.uint8] | NDArray[np.float32],
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def log(array: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor: ...


def log(
    array: ImageType | torch.Tensor,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Compute the NumPy-compatible natural logarithm with explicit output storage.

    Args:
        array: Prevalidated uint8, float16, or float32 array.
        out: Optional caller-owned output of the result dtype and shape. May be the floating input.
            The caller validates strides, writeability, and supported overlap.

    Returns:
        The same shape; float16 stays float16, while uint8 and float32 return float32.
        A supplied output is returned by identity.

    Notes:
        Eligible OpenCV results match NumPy within rtol=1e-5 and atol=1e-7.
        Zero, negative, subnormal, and non-finite values use NumPy semantics.
    """
    if isinstance(array, torch.Tensor):
        return _torch_unary(array, torch.log, out=cast("torch.Tensor | None", out))
    output = cast("NDArray[np.float16] | NDArray[np.float32] | None", out)
    if array.dtype == np.uint8:
        return _numpy_unary(
            cast("NDArray[np.float32]", array.astype(np.float32)),
            np.log,
            out=cast("NDArray[np.float32] | None", output),
        )
    if array.dtype == np.float16:
        return _float16_unary(
            cast("NDArray[np.float16]", array),
            "log",
            np.log,
            out=cast("NDArray[np.float16] | None", output),
        )
    destination = cast("NDArray[np.float32] | None", output)
    contiguous = array.flags.c_contiguous and array.size >= _OPENCV_CONTIGUOUS_MIN_ELEMENTS
    single_channel = array.ndim == 2 or (array.ndim >= 3 and array.shape[-1] == 1)
    high_channel = array.ndim >= 3 and array.shape[-1] >= 8
    minimum = (
        _LOG_OPENCV_STRIDED_SINGLE_CHANNEL_MIN_ELEMENTS
        if single_channel
        else _LOG_OPENCV_STRIDED_HIGH_CHANNEL_MIN_ELEMENTS
        if high_channel
        else None
    )
    strided = not array.flags.c_contiguous and minimum is not None and array.size >= minimum
    if (contiguous or strided) and (
        destination is None or (array.flags.c_contiguous and destination.flags.c_contiguous)
    ):
        return _log_opencv(cast("NDArray[np.float32]", array), out=destination)
    return _numpy_unary(cast("NDArray[np.float32]", array), np.log, out=destination)


@overload
def sqrt(array: NDArray[np.float16], *, out: NDArray[np.float16] | None = None) -> NDArray[np.float16]: ...


@overload
def sqrt(
    array: NDArray[np.uint8] | NDArray[np.float32],
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def sqrt(array: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor: ...


def sqrt(
    array: ImageType | torch.Tensor,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Compute the NumPy-compatible square root with explicit output storage.

    Args:
        array: Prevalidated uint8, float16, or float32 array.
        out: Optional caller-owned output of the result dtype and shape. May be the floating input.
            The caller validates strides, writeability, and supported overlap.

    Returns:
        The same shape; float16 stays float16, while uint8 and float32 return float32.
        A supplied output is returned by identity.
    """
    if isinstance(array, torch.Tensor):
        return _torch_unary(array, torch.sqrt, out=cast("torch.Tensor | None", out))
    output = cast("NDArray[np.float16] | NDArray[np.float32] | None", out)
    if array.dtype == np.uint8:
        return _numpy_unary(
            cast("NDArray[np.float32]", array.astype(np.float32)),
            np.sqrt,
            out=cast("NDArray[np.float32] | None", output),
        )
    if array.dtype == np.float16:
        return _float16_unary(
            cast("NDArray[np.float16]", array),
            "sqrt",
            np.sqrt,
            out=cast("NDArray[np.float16] | None", output),
        )
    return _numpy_unary(cast("NDArray[np.float32]", array), np.sqrt, out=cast("NDArray[np.float32] | None", output))
