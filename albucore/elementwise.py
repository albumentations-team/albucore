"""Elementwise transcendental functions with explicit output storage."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeVar, cast, overload

import cv2
import numpy as np
import torch

from albucore._float_cast import _cast_float_numpy
from albucore.lut import _apply_float_lut
from albucore.torch_backend import _float16_unary_torch, sqrt_uint8_torch
from albucore.utils import _LINUX_X86_64, _MACOS_ARM64

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from numpy.typing import NDArray


__all__ = ["exp", "sqrt"]

_FloatDTypeT = TypeVar("_FloatDTypeT", np.float16, np.float32)
_OPENCV_CONTIGUOUS_MIN_ELEMENTS = 512 if _MACOS_ARM64 else 4_096
_LINUX_OPENCV_CONTIGUOUS_MIN_ELEMENTS = 64
_EXP_OPENCV_STRIDED_MIN_ELEMENTS = 65_536
_TORCH_FLOAT16_MIN_ELEMENTS = {
    "exp": 12_288,
    "sqrt": 128 if _LINUX_X86_64 else 12_288 if _MACOS_ARM64 else 20_480,
}
_UINT8_SQRT_LUT_MIN_ELEMENTS = 20_480
_LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS = 230_400
_MACOS_UINT8_SQRT_TORCH_MIN_ELEMENTS = 12_288
_MACOS_UINT8_SQRT_TORCH_AFTER_LUT_MIN_ELEMENTS = 36_864
_UINT8_SQRT_LUT = np.sqrt(np.arange(256, dtype=np.float32))
with np.errstate(over="ignore"):
    _UINT8_EXP_LUT = np.exp(np.arange(256, dtype=np.float32))
_NATIVE_HALF_EXP = _LINUX_X86_64
# Contiguous Tensor exp reaches OpenCV through shared storage above these sizes; Linux keeps Torch for float16.
_TENSOR_EXP_NUMPY_MIN_ELEMENTS: dict[torch.dtype, int] = (
    {torch.uint8: 1_024, torch.float32: 12_288}
    if _LINUX_X86_64
    else {torch.uint8: 2_048, torch.float32: 4_096, torch.float16: 12_288}
    if _MACOS_ARM64
    else {}
)


def _opencv_contiguous_min_elements() -> int:
    return _LINUX_OPENCV_CONTIGUOUS_MIN_ELEMENTS if _LINUX_X86_64 else _OPENCV_CONTIGUOUS_MIN_ELEMENTS


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


def _sqrt_float16_via_float32(array: NDArray[np.float16], out: NDArray[np.float16] | None) -> NDArray[np.float16]:
    """Take the root in one float32 buffer, avoiding the buffered-cast ufunc loop that dominates tiny arrays."""
    working = array.astype(np.float32)
    np.sqrt(working, out=working)
    if out is None:
        return working.astype(np.float16)
    np.copyto(out, working, casting="unsafe")
    return out


def _float16_unary(
    array: NDArray[np.float16],
    operation: Literal["exp", "sqrt"],
    numpy_operation: Callable[..., np.ndarray],
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16]:
    if array.flags.c_contiguous and array.size >= _opencv_contiguous_min_elements() and operation == "exp":
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
    if operation == "sqrt" and _LINUX_X86_64 and array.size < _TORCH_FLOAT16_MIN_ELEMENTS["sqrt"]:
        return _sqrt_float16_via_float32(array, out)
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


def _exp_tensor_numpy(array: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Compute a contiguous CPU Tensor exponential through the shared-storage NumPy/OpenCV kernels.

    The flat view keeps OpenCV's channel limit from applying to arbitrary Tensor layouts.
    """
    source = array.numpy().reshape(-1)
    destination = None if out is None else out.numpy().reshape(-1)
    result: np.ndarray
    if array.dtype == torch.uint8:
        result = _apply_float_lut(source, _UINT8_EXP_LUT, out=destination)
    elif array.dtype == torch.float32:
        result = _opencv_unary(source, cv2.exp, out=destination)
    else:
        # Narrowing overflow must stay silent, as in Torch.
        with np.errstate(all="ignore"):
            working = _cast_float_numpy(source, np.float32)
            result = _cast_float_numpy(_opencv_unary(working, cv2.exp), np.float16, out=destination)
    if out is None:
        return torch.from_numpy(result).reshape(array.shape)
    torch.autograd.graph.increment_version(out)
    return out


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
        tensor_out = cast("torch.Tensor | None", out)
        if (
            array.numel() >= _TENSOR_EXP_NUMPY_MIN_ELEMENTS.get(array.dtype, sys.maxsize)
            and array.is_contiguous()
            and (tensor_out is None or tensor_out.is_contiguous())
        ):
            return _exp_tensor_numpy(array, tensor_out)
        return _torch_unary(array, torch.exp, out=tensor_out)
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
    minimum = _opencv_contiguous_min_elements() if array.flags.c_contiguous else _EXP_OPENCV_STRIDED_MIN_ELEMENTS
    if array.size >= minimum and (destination is None or (array.flags.c_contiguous and destination.flags.c_contiguous)):
        return _opencv_unary(cast("NDArray[np.float32]", array), cv2.exp, out=destination)
    return _numpy_unary(cast("NDArray[np.float32]", array), np.exp, out=destination)


def _sqrt_uint8(array: NDArray[np.uint8], out: NDArray[np.float32] | None) -> NDArray[np.float32]:
    """Route uint8 square roots among Torch, the shared float LUT, and NumPy by measured platform regions."""
    lut_contiguous = array.flags.c_contiguous and (out is None or out.flags.c_contiguous)
    if _MACOS_ARM64:
        channels = array.shape[-1] if array.ndim == 3 else 0
        macos_lut_region = (channels in (1, 3) and array.size >= _UINT8_SQRT_LUT_MIN_ELEMENTS) or (
            channels == 9 and array.size >= 7_000_000 and out is None
        )
        # The LUT ties Torch from its own minimum up to about 37K elements for C=1 and C=3.
        torch_minimum = (
            _MACOS_UINT8_SQRT_TORCH_AFTER_LUT_MIN_ELEMENTS
            if channels in (1, 3) and array.size >= _UINT8_SQRT_LUT_MIN_ELEMENTS
            else _MACOS_UINT8_SQRT_TORCH_MIN_ELEMENTS
        )
        torch_result = sqrt_uint8_torch(array, minimum_elements=torch_minimum, out=out)
        if torch_result is not None:
            return torch_result
        use_lut = macos_lut_region and lut_contiguous
    else:
        use_lut = _LINUX_X86_64 and array.size >= _LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS and lut_contiguous
    if use_lut:
        return _apply_float_lut(array, _UINT8_SQRT_LUT, out=out)
    return _numpy_unary(cast("NDArray[np.float32]", array.astype(np.float32)), np.sqrt, out=out)


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
        return _sqrt_uint8(cast("NDArray[np.uint8]", array), cast("NDArray[np.float32] | None", output))
    if array.dtype == np.float16:
        return _float16_unary(
            cast("NDArray[np.float16]", array),
            "sqrt",
            np.sqrt,
            out=cast("NDArray[np.float16] | None", output),
        )
    return _numpy_unary(cast("NDArray[np.float32]", array), np.sqrt, out=cast("NDArray[np.float32] | None", output))
