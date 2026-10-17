"""Concrete unary arguments and dtype-dependent explicit outputs."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import exp, log, sqrt


def check_unary_outputs(
    bytes_image: NDArray[np.uint8], half: NDArray[np.float16], floats: NDArray[np.float32], tensor: torch.Tensor
) -> None:
    assert_type(exp(bytes_image, out=floats), NDArray[np.float32])
    assert_type(log(bytes_image, out=floats), NDArray[np.float32])
    assert_type(sqrt(bytes_image, out=floats), NDArray[np.float32])
    assert_type(exp(half, out=half), NDArray[np.float16])
    assert_type(log(half, out=half), NDArray[np.float16])
    assert_type(sqrt(half, out=half), NDArray[np.float16])
    assert_type(exp(floats, out=floats), NDArray[np.float32])
    assert_type(log(floats, out=floats), NDArray[np.float32])
    assert_type(sqrt(floats, out=floats), NDArray[np.float32])
    assert_type(exp(tensor, out=tensor), torch.Tensor)
    assert_type(log(tensor, out=tensor), torch.Tensor)
    assert_type(sqrt(tensor, out=tensor), torch.Tensor)
    exp(half, inplace=True)  # type: ignore[call-overload]
    log(half, inplace=True)  # type: ignore[call-overload]
    sqrt(half, inplace=True)  # type: ignore[call-overload]
