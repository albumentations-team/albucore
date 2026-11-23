"""Elementwise transcendental functions with explicit output storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast, overload

import cv2
import numpy as np
import torch

from albucore._float_cast import _cast_float_numpy
from albucore.lut import _apply_float_lut
from albucore.torch_backend import _float16_unary_torch
from albucore.utils import _LINUX_X86_64, _MACOS_ARM64

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray


__all__ = ["exp", "sqrt"]

_FloatDTypeT = TypeVar("_FloatDTypeT", np.float16, np.float32)
_OPENCV_CONTIGUOUS_MIN_ELEMENTS = 4_096
_EXP_OPENCV_STRIDED_MIN_ELEMENTS = 65_536
_TORCH_FLOAT16_MIN_ELEMENTS = {"exp": 12_288, "sqrt": 20_480}
_UINT8_SQRT_LUT_MIN_ELEMENTS = 20_480
_LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS = 921_600
_UINT8_SQRT_LUT = np.sqrt(np.arange(256, dtype=np.float32))
with np.errstate(over="ignore"):
    _UINT8_EXP_LUT = np.exp(np.arange(256, dtype=np.float32))
_NATIVE_HALF_EXP = _LINUX_X86_64


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
    array: NDArray[_FloatDTypeT],
    operation: Callable[..., np.ndarray],
    *,
    out: NDArray[_FloatDTypeT] | None = None,
) -> NDArray[_FloatDTypeT]:
    if array.size == 0:
        return np.empty_like(array) if out is None else out
    if out is not None:
        operation(array.reshape(-1), dst=out.reshape(-1))
        return out
    result = operation(array)
    return cast("NDArray[_FloatDTypeT]", result.reshape(array.shape))


def _float16_unary(
    array: NDArray[np.float16],
    operation: Literal["exp", "sqrt"],
    numpy_operation: Callable[..., np.ndarray],
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16]:
    if array.flags.c_contiguous and array.size >= _OPENCV_CONTIGUOUS_MIN_ELEMENTS and operation == "exp":
        if not _NATIVE_HALF_EXP:
            working = _cast_float_numpy(array, np.float32)
            return _cast_float_numpy(_opencv_unary(working, cv2.exp), np.float16, out=out)
        if out is None or out.flags.c_contiguous:
            return _opencv_unary(array, cv2.exp, out=out)
        np.copyto(out, _opencv_unary(array, cv2.exp))
        return out
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
    array: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
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
        return _apply_float_lut(
            cast("NDArray[np.uint8]", array),
            _UINT8_EXP_LUT,
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
    array: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
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
        channels = array.shape[-1] if array.ndim == 3 else 0
        macos_lut_region = _MACOS_ARM64 and (
            (channels in (1, 3) and array.size >= _UINT8_SQRT_LUT_MIN_ELEMENTS)
            or (channels == 9 and array.size >= 7_000_000 and output is None)
        )
        linux_lut_region = (
            _LINUX_X86_64
            and array.size >= _LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS
            and (channels == 9 or (channels == 3 and output is None))
        )
        if (
            (macos_lut_region or linux_lut_region)
            and array.ndim == 3
            and array.flags.c_contiguous
            and (output is None or output.flags.c_contiguous)
        ):
            return _apply_float_lut(
                cast("NDArray[np.uint8]", array),
                _UINT8_SQRT_LUT,
                out=cast("NDArray[np.float32] | None", output),
            )
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
