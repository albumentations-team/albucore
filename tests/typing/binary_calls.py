"""Binary arithmetic preserves the input dtype and requires a matching output."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import add, multiply


def check_outputs(
    bytes_image: NDArray[np.uint8],
    half: NDArray[np.float16],
    floats: NDArray[np.float32],
    tensor: torch.Tensor,
) -> None:
    assert_type(add(bytes_image, 1.5), NDArray[np.uint8])
    assert_type(add(bytes_image, 1.5, out=bytes_image), NDArray[np.uint8])
    assert_type(add(half, 1.5, out=half), NDArray[np.float16])
    assert_type(add(floats, 1.5, out=floats), NDArray[np.float32])
    assert_type(add(tensor, 1.5, layout="CHW", out=tensor), torch.Tensor)
    assert_type(multiply(bytes_image, 1.5), NDArray[np.uint8])
    assert_type(multiply(bytes_image, 1.5, out=bytes_image), NDArray[np.uint8])
    assert_type(multiply(half, 1.5, out=half), NDArray[np.float16])
    assert_type(multiply(floats, 1.5, out=floats), NDArray[np.float32])
    assert_type(multiply(tensor, 1.5, layout="CHW", out=tensor), torch.Tensor)
    add(bytes_image, 1.5, out=floats)  # type: ignore[arg-type]
    add(half, 1.5, out=bytes_image)  # type: ignore[arg-type]
    add(floats, 1.5, out=half)  # type: ignore[arg-type]
    add(half, 1.5, inplace=True)  # type: ignore[call-overload]
    multiply(bytes_image, 1.5, out=floats)  # type: ignore[arg-type]
    multiply(half, 1.5, out=bytes_image)  # type: ignore[arg-type]
    multiply(floats, 1.5, out=half)  # type: ignore[arg-type]
    multiply(half, 1.5, inplace=True)  # type: ignore[call-overload]


def check_union(image: NDArray[np.float32] | torch.Tensor) -> None:
    assert_type(add(img=image, value=1.5, layout="CHW"), NDArray[np.float32] | torch.Tensor)
    assert_type(multiply(img=image, value=1.5, layout="CHW"), NDArray[np.float32] | torch.Tensor)
