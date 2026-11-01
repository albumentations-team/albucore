"""Checked conversion input, result dtype, and destination relationships."""

from typing import TypeVar, assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import SupportedDType, from_float, to_float

_DTypeT = TypeVar("_DTypeT", bound=np.uint8 | np.float16 | np.float32)


def conversion_outputs(
    image: NDArray[np.uint8], half: NDArray[np.float16], floats: NDArray[np.float32], tensor: torch.Tensor
) -> None:
    assert_type(to_float(img=image, max_value=255, out=floats), NDArray[np.float32])
    assert_type(to_float(half, out=half), NDArray[np.float16])
    assert_type(to_float(floats, out=floats), NDArray[np.float32])
    assert_type(to_float(tensor, out=tensor), torch.Tensor)
    assert_type(from_float(half, np.uint8, out=image), NDArray[np.uint8])
    assert_type(from_float(floats, np.dtype(np.float16), out=half), NDArray[np.float16])
    assert_type(from_float(half, np.dtype(np.float32), out=floats), NDArray[np.float32])
    assert_type(from_float(tensor, torch.uint8, out=tensor), torch.Tensor)
    from_float(half)  # type: ignore[call-overload]
    to_float(image, 255, floats)  # type: ignore[call-overload]
    from_float(floats, np.float16, out=floats)  # type: ignore[arg-type]
    from_float(floats, np.uint8, out=half)  # type: ignore[arg-type]


def union_conversion(
    image: NDArray[np.uint8] | torch.Tensor,
    floats: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    target: np.dtype[np.uint8] | np.dtype[np.float32],
) -> None:
    assert_type(to_float(image), NDArray[np.float32] | torch.Tensor)
    assert_type(from_float(floats, np.uint8), NDArray[np.uint8] | torch.Tensor)
    assert_type(from_float(floats, target), NDArray[np.uint8] | NDArray[np.float32] | torch.Tensor)


def all_storage_dtypes(
    floats: NDArray[np.float16] | NDArray[np.float32],
    image: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> None:
    assert_type(from_float(floats, image.dtype), NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32])


def generic_target(floats: NDArray[np.float32], dtype: np.dtype[_DTypeT]) -> NDArray[_DTypeT]:
    return from_float(floats, dtype)


def selected_target(
    floats: NDArray[np.float32], dtype: SupportedDType
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    return from_float(floats, dtype)
