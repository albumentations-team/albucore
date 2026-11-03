"""Checked conversion input, result dtype, and destination relationships."""

from typing import TypeVar, assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import SupportedDType, convert_dtype

_DTypeT = TypeVar("_DTypeT", bound=np.uint8 | np.float16 | np.float32)


def conversion_outputs(
    image: NDArray[np.uint8], half: NDArray[np.float16], floats: NDArray[np.float32], tensor: torch.Tensor,
) -> None:
    assert_type(convert_dtype(img=image, dtype=np.float32, max_value=255, out=floats), NDArray[np.float32])
    assert_type(convert_dtype(half, np.float16, out=half), NDArray[np.float16])
    assert_type(convert_dtype(floats, np.float32, out=floats), NDArray[np.float32])
    assert_type(convert_dtype(tensor, torch.float32, out=tensor), torch.Tensor)
    assert_type(convert_dtype(image, np.float16, out=half), NDArray[np.float16])
    assert_type(convert_dtype(image, np.uint8, out=image), NDArray[np.uint8])
    assert_type(convert_dtype(half, np.uint8, out=image), NDArray[np.uint8])
    assert_type(convert_dtype(floats, np.dtype(np.float16), out=half), NDArray[np.float16])
    assert_type(convert_dtype(half, np.dtype(np.float32), out=floats), NDArray[np.float32])
    assert_type(convert_dtype(tensor, torch.uint8, out=tensor), torch.Tensor)
    convert_dtype(half)  # type: ignore[call-overload]
    convert_dtype(image, np.float32, 255)  # type: ignore[call-overload]
    convert_dtype(floats, np.float16, out=floats)  # type: ignore[arg-type]
    convert_dtype(floats, np.uint8, out=half)  # type: ignore[arg-type]


def union_conversion(
    image: NDArray[np.uint8] | torch.Tensor,
    floats: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    target: np.dtype[np.uint8] | np.dtype[np.float32],
) -> None:
    assert_type(convert_dtype(image, np.float32), NDArray[np.float32] | torch.Tensor)
    assert_type(convert_dtype(floats, np.uint8), NDArray[np.uint8] | torch.Tensor)
    assert_type(convert_dtype(floats, target), NDArray[np.uint8] | NDArray[np.float32] | torch.Tensor)


def all_storage_dtypes(
    floats: NDArray[np.float16] | NDArray[np.float32],
    image: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> None:
    assert_type(convert_dtype(floats, image.dtype), NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32])


def generic_target(floats: NDArray[np.float32], dtype: np.dtype[_DTypeT]) -> NDArray[_DTypeT]:
    return convert_dtype(floats, dtype)


def selected_target(
    floats: NDArray[np.float32], dtype: SupportedDType,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    return convert_dtype(floats, dtype)
