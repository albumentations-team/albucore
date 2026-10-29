"""Convert image arrays among uint8, float16, and float32."""

import platform
import sys
from typing import TypeVar, cast, overload

import numkong as nk
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import ImageType, SupportedDType
from albucore.lut import _cv2_lut_channel_last
from albucore.torch_backend import from_float_uint8_torch
from albucore.utils import MAX_VALUES_BY_DTYPE, get_max_value

__all__ = ["from_float", "to_float"]

_OutputDTypeT = TypeVar("_OutputDTypeT", bound=np.uint8 | np.float16 | np.float32)

_NUMPY_TO_TORCH_DTYPE = {
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
}
_MACOS_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"
_TENSOR_NUMPY_CAST_MIN_ELEMENTS = 61_440


@overload
def to_float(
    img: NDArray[np.float16],
    max_value: float | None = None,
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16]: ...


@overload
def to_float(
    img: NDArray[np.uint8] | NDArray[np.float32],
    max_value: float | None = None,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def to_float(
    img: torch.Tensor,
    max_value: float | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...


def to_float(
    img: ImageType | torch.Tensor,
    max_value: float | None = None,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Convert uint8 to normalized float32 and preserve floating inputs.

    Uint8 input is divided by ``max_value`` (default 255); floating inputs ignore
    the divisor and retain dtype and values. NumPy is channel-last; pointwise
    Tensor conversion preserves any declared layout without an axis argument.

    Without ``out``, floating inputs return by identity and uint8 inputs allocate.
    A supplied output has the same shape/container and the result dtype and is
    returned by identity. Writable strided outputs are supported. Exact input
    aliasing is allowed for floating inputs; other overlap is outside the contract.
    The caller validates dtype, CPU placement, output metadata and overlap.
    """
    if isinstance(img, torch.Tensor):
        if img.dtype != torch.uint8:
            return img if out is None or out is img else cast("torch.Tensor", out).copy_(img)
        tensor_out = cast("torch.Tensor | None", out)
        divisor = MAX_VALUES_BY_DTYPE[np.dtype(np.uint8)] if max_value is None else max_value
        result = img.to(torch.float32) if tensor_out is None else tensor_out.copy_(img)
        return result.div_(divisor)

    if img.dtype.kind == "f":
        if out is None or out is img:
            return cast("NDArray[np.float16] | NDArray[np.float32]", img)
        numpy_out = cast("NDArray[np.float16] | NDArray[np.float32]", out)
        np.copyto(numpy_out, img)
        return numpy_out
    divisor = MAX_VALUES_BY_DTYPE[img.dtype] if max_value is None else max_value
    lut = (np.arange(256, dtype=np.float32) / divisor).astype(np.float32)
    return cast(
        "NDArray[np.float32]",
        _cv2_lut_channel_last(cast("NDArray[np.uint8]", img), lut, out=cast("NDArray[np.float32] | None", out)),
    )


def _from_float_uint8_numkong(
    img: NDArray[np.float32],
    max_value: float,
    *,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Scale, round, and saturate float32 input with one float32 work buffer."""
    rounded = np.empty_like(img)
    np.multiply(img, max_value, out=rounded)
    np.rint(rounded, out=rounded)
    result = np.empty(img.shape, dtype=np.uint8) if out is None or not out.flags.c_contiguous else out
    nk.astype(rounded, "uint8", out=result)
    if out is not None and result is not out:
        np.copyto(out, result)
        return out
    return result


def _from_float_torch(
    img: torch.Tensor,
    target_dtype: np.dtype | type[np.generic] | torch.dtype,
    max_value: float | None,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if target_dtype in (torch.uint8, torch.float16, torch.float32):
        torch_dtype = cast("torch.dtype", target_dtype)
    else:
        torch_dtype = _NUMPY_TO_TORCH_DTYPE[np.dtype(cast("SupportedDType", target_dtype))]
    if torch_dtype in (torch.float16, torch.float32):
        if img.dtype == torch_dtype and (out is None or out is img):
            return img
        if (
            _MACOS_ARM64
            and img.dtype != torch_dtype
            and img.numel() >= _TENSOR_NUMPY_CAST_MIN_ELEMENTS
            and img.is_contiguous()
        ):
            numpy_dtype = np.float32 if torch_dtype == torch.float32 else np.float16
            with np.errstate(all="ignore"):
                result = _cast_float_numpy(img.numpy(), numpy_dtype, out=None if out is None else out.numpy())
            return torch.from_numpy(result) if out is None else out
        return img.to(dtype=torch_dtype) if out is None else out.copy_(img)

    scale = get_max_value(np.dtype(np.uint8)) if max_value is None else max_value
    if img.dtype == torch.float16 and img.is_contiguous():
        with np.errstate(over="ignore", invalid="ignore"):
            half_scale = np.float16(scale)
        if np.isfinite(half_scale) and float(half_scale) == float(scale):
            rounded = torch.round(img * float(half_scale)).clamp_(0.0, 255.0)
            return rounded.to(torch.uint8) if out is None else out.copy_(rounded)
    rounded = torch.round(img.to(torch.float32) * scale).clamp_(0.0, 255.0)
    return rounded.to(torch.uint8) if out is None else out.copy_(rounded)


@overload
def from_float(
    img: NDArray[np.float16] | NDArray[np.float32],
    target_dtype: np.dtype[np.uint8] | type[np.uint8],
    max_value: float | None = None,
    *,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]: ...


@overload
def from_float(
    img: NDArray[np.float16] | NDArray[np.float32],
    target_dtype: np.dtype[np.float16] | type[np.float16],
    max_value: float | None = None,
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16]: ...


@overload
def from_float(
    img: NDArray[np.float16] | NDArray[np.float32],
    target_dtype: np.dtype[np.float32] | type[np.float32],
    max_value: float | None = None,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def from_float(
    img: NDArray[np.float16] | NDArray[np.float32],
    target_dtype: np.dtype[np.uint8] | np.dtype[np.float32] | type[np.uint8] | type[np.float32],
    max_value: float | None = None,
    *,
    out: None = None,
) -> NDArray[np.uint8] | NDArray[np.float32]: ...


@overload
def from_float(
    img: NDArray[np.float16] | NDArray[np.float32],
    target_dtype: SupportedDType,
    max_value: float | None = None,
    *,
    out: None = None,
) -> ImageType: ...


@overload
def from_float(
    img: NDArray[np.float16] | NDArray[np.float32],
    target_dtype: np.dtype[_OutputDTypeT] | type[_OutputDTypeT],
    max_value: float | None = None,
    *,
    out: None = None,
) -> NDArray[_OutputDTypeT]: ...


@overload
def from_float(
    img: torch.Tensor,
    target_dtype: np.dtype | type[np.generic] | torch.dtype,
    max_value: float | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...


def from_float(
    img: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    target_dtype: np.dtype | type[np.generic] | torch.dtype,
    max_value: float | None = None,
    *,
    out: ImageType | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Convert a floating image to the requested storage dtype.

    Float16/float32 targets cast without intensity scaling or clipping. Uint8
    targets scale by ``max_value`` (default 255), round ties to even, saturate to
    [0, 255], and map NaN to zero. NumPy half input uses float32 arithmetic.
    Contiguous half Tensors use half arithmetic for exactly half-representable
    scales, differing from the wide result by at most one uint8 unit; other
    Tensor quantization uses float32 arithmetic.

    Without ``out``, input storage is not written; same-dtype casts may return
    by identity. A supplied output matches the result dtype, input shape and
    container and is returned by identity. Writable strides are supported and
    may need a temporary. Exact input aliasing is allowed only for same-dtype
    casts. The caller validates image/output metadata, CPU placement and overlap.
    """
    if isinstance(img, torch.Tensor):
        return _from_float_torch(img, target_dtype, max_value, out=cast("torch.Tensor | None", out))

    numpy_dtype = np.dtype(cast("SupportedDType", target_dtype))
    if numpy_dtype.type in (np.float16, np.float32):
        if img.dtype == numpy_dtype and (out is None or out is img):
            return img
        floating_dtype = cast("np.dtype[np.float16 | np.float32]", numpy_dtype)
        if out is None:
            return _cast_float_numpy(img, floating_dtype)
        return _cast_float_numpy(img, floating_dtype, out=cast("NDArray[np.float16] | NDArray[np.float32] | None", out))

    resolved_max = MAX_VALUES_BY_DTYPE[np.dtype(np.uint8)] if max_value is None else max_value
    numpy_out = cast("NDArray[np.uint8] | None", out)
    torch_result = from_float_uint8_torch(img, resolved_max, out=numpy_out)
    if torch_result is not None:
        return torch_result
    if img.dtype == np.float16:
        img = cast("NDArray[np.float32]", img.astype(np.float32))
    return _from_float_uint8_numkong(cast("NDArray[np.float32]", img), resolved_max, out=numpy_out)
