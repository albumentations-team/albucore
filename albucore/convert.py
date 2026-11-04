"""Convert image arrays among uint8, float16, and float32."""

import platform
import sys
from typing import TypeVar, cast, overload

import numkong as nk
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import SupportedDType
from albucore.lut import _cv2_lut_channel_last
from albucore.torch_backend import quantize_uint8_torch
from albucore.utils import MAX_VALUES_BY_DTYPE, get_max_value

__all__ = ["convert_dtype"]

_OutputDTypeT = TypeVar("_OutputDTypeT", bound=np.uint8 | np.float16 | np.float32)

_TORCH_DTYPES: dict[SupportedDType | torch.dtype, torch.dtype] = {
    np.uint8: torch.uint8,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    torch.uint8: torch.uint8,
    torch.float16: torch.float16,
    torch.float32: torch.float32,
}
_MACOS_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"
_LINUX_X86_64 = sys.platform == "linux" and platform.machine() == "x86_64"
_TENSOR_NUMPY_CAST_MIN_ELEMENTS = 61_440


def _quantize_uint8_numkong(
    img: NDArray[np.float32],
    max_value: float,
    *,
    out: NDArray[np.uint8] | None = None,
    input_is_working_buffer: bool = False,
) -> NDArray[np.uint8]:
    """Scale, round, and saturate float32 input with one float32 work buffer."""
    rounded = img if input_is_working_buffer else np.empty_like(img)
    np.multiply(img, max_value, out=rounded)
    np.rint(rounded, out=rounded)
    result = np.empty(img.shape, dtype=np.uint8) if out is None or not out.flags.c_contiguous else out
    nk.astype(rounded, "uint8", out=result)
    if out is not None and result is not out:
        np.copyto(out, result)
        return out
    return result


def _convert_dtype_torch(
    img: torch.Tensor,
    dtype: np.dtype | type[np.generic] | torch.dtype,
    max_value: float | None,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    torch_dtype = _TORCH_DTYPES[cast("SupportedDType | torch.dtype", dtype)]
    if img.dtype == torch_dtype:
        return img if out is None or out is img else out.copy_(img)
    if img.dtype == torch.uint8:
        divisor = MAX_VALUES_BY_DTYPE[np.dtype(np.uint8)] if max_value is None else max_value
        working_dtype = torch.float32 if _LINUX_X86_64 and torch_dtype == torch.float16 else torch_dtype
        normalized = (
            img.to(working_dtype)
            if out is None or working_dtype != torch_dtype or (torch_dtype == torch.float16 and not out.is_contiguous())
            else out.copy_(img)
        )
        normalized.div_(divisor)
        if out is None and working_dtype != torch_dtype:
            normalized = normalized.to(torch_dtype)
        return normalized if out is None or normalized is out else out.copy_(normalized)
    if torch_dtype in (torch.float16, torch.float32):
        if _MACOS_ARM64 and img.numel() >= _TENSOR_NUMPY_CAST_MIN_ELEMENTS and img.is_contiguous():
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


def _convert_uint8_numpy(
    image: NDArray[np.uint8],
    dtype: np.dtype[np.float16] | np.dtype[np.float32],
    divisor: float,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | None,
) -> NDArray[np.float16] | NDArray[np.float32]:
    """Select raw casting or normalized lookup without changing output storage rules."""
    if divisor == 1 and dtype == np.float32 and image.flags.c_contiguous:
        return _cast_float_numpy(image, np.float32, out=cast("NDArray[np.float32] | None", out))
    lut = (np.arange(256, dtype=np.float32) / divisor).astype(dtype)
    return _cv2_lut_channel_last(image, lut, out=out)


@overload
def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dtype: np.dtype[np.uint8] | type[np.uint8],
    *,
    max_value: float | None = None,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]: ...


@overload
def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dtype: np.dtype[np.float16] | type[np.float16],
    *,
    max_value: float | None = None,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16]: ...


@overload
def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dtype: np.dtype[np.float32] | type[np.float32],
    *,
    max_value: float | None = None,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dtype: np.dtype[np.uint8] | np.dtype[np.float32] | type[np.uint8] | type[np.float32],
    *,
    max_value: float | None = None,
    out: None = None,
) -> NDArray[np.uint8] | NDArray[np.float32]: ...


@overload
def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dtype: SupportedDType,
    *,
    max_value: float | None = None,
    out: None = None,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: ...


@overload
def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    dtype: np.dtype[_OutputDTypeT] | type[_OutputDTypeT],
    *,
    max_value: float | None = None,
    out: None = None,
) -> NDArray[_OutputDTypeT]: ...


@overload
def convert_dtype(
    img: torch.Tensor,
    dtype: np.dtype | type[np.generic] | torch.dtype,
    *,
    max_value: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...


def convert_dtype(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    dtype: np.dtype | type[np.generic] | torch.dtype,
    *,
    max_value: float | None = None,
    out: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Convert an image among uint8, float16, and float32 with intensity scaling.

    Uint8-to-float conversion divides by ``max_value`` (default 255). Floating
    sources converted to uint8 multiply by the same scale, round ties to even,
    saturate to [0, 255], and map NaN to zero. Float16/float32 casts and same-dtype
    conversions ignore the scale and do not clip. The image's actual min/max
    values are never used to determine the scale. NumPy half quantization uses
    float32 arithmetic.
    Contiguous half Tensors use half arithmetic for exactly half-representable
    scales, differing from the wide result by at most one uint8 unit; other
    Tensor quantization uses float32 arithmetic.

    Without ``out``, input storage is not written; same-dtype conversions return
    the input object. A supplied output matches the result dtype, input shape and
    container and is returned by identity. Writable strides are supported and
    may need a temporary. Exact input aliasing is allowed only for same-dtype
    conversions. NumPy is channel-last; pointwise Tensor conversion preserves
    its declared layout without an axis argument. The caller validates
    image/output metadata, CPU placement and overlap.

    Examples:
        >>> image = np.array([[[0], [255]]], dtype=np.uint8)
        >>> convert_dtype(image, np.float16).tolist()
        [[[0.0], [1.0]]]
    """
    if isinstance(img, torch.Tensor):
        return _convert_dtype_torch(img, dtype, max_value, out=cast("torch.Tensor | None", out))

    numpy_dtype = np.dtype(cast("SupportedDType", dtype))
    if img.dtype == numpy_dtype:
        if out is not None and out is not img:
            np.copyto(cast("NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]", out), img)
        return img if out is None else out
    if img.dtype == np.uint8:
        divisor = MAX_VALUES_BY_DTYPE[img.dtype] if max_value is None else max_value
        return _convert_uint8_numpy(
            cast("NDArray[np.uint8]", img),
            cast("np.dtype[np.float16] | np.dtype[np.float32]", numpy_dtype),
            divisor,
            out=cast("NDArray[np.float16] | NDArray[np.float32] | None", out),
        )
    if numpy_dtype.type in (np.float16, np.float32):
        floating_dtype = cast("np.dtype[np.float16 | np.float32]", numpy_dtype)
        return _cast_float_numpy(img, floating_dtype, out=cast("NDArray[np.float16] | NDArray[np.float32] | None", out))

    resolved_max = MAX_VALUES_BY_DTYPE[np.dtype(np.uint8)] if max_value is None else max_value
    numpy_out = cast("NDArray[np.uint8] | None", out)
    torch_result = quantize_uint8_torch(
        cast("NDArray[np.float16] | NDArray[np.float32]", img),
        resolved_max,
        out=numpy_out,
    )
    if torch_result is not None:
        return torch_result
    input_is_working_buffer = img.dtype == np.float16
    if input_is_working_buffer:
        with np.errstate(invalid="ignore"):
            img = cast("NDArray[np.float32]", img.astype(np.float32))
    return _quantize_uint8_numkong(
        cast("NDArray[np.float32]", img),
        resolved_max,
        out=numpy_out,
        input_is_working_buffer=input_is_working_buffer,
    )
