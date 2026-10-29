"""Checked dtype and output relationships for clipping."""

from typing import TypeVar, assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import clip

_DTypeT = TypeVar("_DTypeT", bound=np.uint8 | np.float16 | np.float32)


def check_clip_outputs(
    image: NDArray[np.float32],
    bytes_output: NDArray[np.uint8],
    half_output: NDArray[np.float16],
    tensor: torch.Tensor,
) -> None:
    assert_type(clip(image, np.uint8, out=bytes_output), NDArray[np.uint8])
    assert_type(clip(image, np.float16, out=half_output), NDArray[np.float16])
    assert_type(clip(image, np.float32, out=image), NDArray[np.float32])
    assert_type(clip(tensor, np.float32, out=tensor), torch.Tensor)
    assert_type(clip(image), NDArray[np.float32])
    assert_type(clip(tensor), torch.Tensor)
    assert_type(clip(image, out=half_output), NDArray[np.float16])
    # warn_unused_ignores rejects a reintroduced positional or inplace API.
    clip(image, np.float32, True)  # type: ignore[call-overload]
    clip(image, np.float32, inplace=True)  # type: ignore[call-overload]
    clip(image, np.float32, out=half_output)  # type: ignore[arg-type]
    clip(image, np.uint8, out=image)  # type: ignore[arg-type]


def all_storage_dtypes(image: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]) -> None:
    assert_type(clip(image, image.dtype), NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32])
    assert_type(clip(image), NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32])
    assert_type(clip(image, out=image), NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32])


def generic_target(image: np.ndarray, dtype: np.dtype[_DTypeT]) -> NDArray[_DTypeT]:
    return clip(image, dtype)


def generic_input(image: NDArray[_DTypeT]) -> NDArray[_DTypeT]:
    return clip(image)
