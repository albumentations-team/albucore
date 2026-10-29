"""Measured FP16/FP32 storage casts without intensity scaling."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import numkong as nk
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from albucore._types import ImageType

_FloatType = TypeVar("_FloatType", bound=np.float16 | np.float32)
_NUMKONG_X86_CAST = bool(nk.get_capabilities().get("skylake"))
_NUMKONG_CAST_MIN_ELEMENTS = 16_384
_FLOAT_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))


def _numkong_cast_preserves_numpy_errors(image: ImageType) -> bool:
    if image.dtype == np.float16:
        return bool(np.max(image.view(np.uint16) & 0x7C00) != 0x7C00)
    # NumKong does not report NumPy's narrowing overflow, underflow, or invalid errors.
    return np.geterr()["under"] == "ignore" and bool(image.min() > -65_520 and image.max() < 65_520)


def _cast_float_numpy(
    image: ImageType,
    dtype: type[_FloatType] | np.dtype[_FloatType],
    *,
    out: NDArray[_FloatType] | None = None,
) -> NDArray[_FloatType]:
    """Preserve NumPy cast semantics and ownership in the measured native contiguous x86 region."""
    if (
        _NUMKONG_X86_CAST
        and image.size >= _NUMKONG_CAST_MIN_ELEMENTS
        and image.flags.c_contiguous
        and (out is None or out.flags.c_contiguous)
        and image.dtype != dtype
        and image.dtype in _FLOAT_DTYPES
        and dtype in _FLOAT_DTYPES
        and _numkong_cast_preserves_numpy_errors(image)
    ):
        result = np.empty(image.shape, dtype=dtype) if out is None else out
        nk.astype(image, "float32" if dtype == np.float32 else "float16", out=result)
        return cast("NDArray[_FloatType]", result)
    if out is not None:
        np.copyto(out, image, casting="unsafe")
        return out
    return cast("NDArray[_FloatType]", image.astype(dtype, copy=False))
