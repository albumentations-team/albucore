"""Multiply-add preserves image dtype/container and checks its explicit output."""
from typing import assert_type
import numpy as np
import torch
from numpy.typing import NDArray
from albucore import multiply_add


def check_outputs(byte: NDArray[np.uint8], half: NDArray[np.float16], floats: NDArray[np.float32], tensor: torch.Tensor) -> None:
    assert_type(multiply_add(byte, 1.5, 0.25, out=byte), NDArray[np.uint8])
    assert_type(multiply_add(half, 1.5, 0.25, out=half), NDArray[np.float16])
    assert_type(multiply_add(floats, 1.5, 0.25, out=floats), NDArray[np.float32])
    assert_type(multiply_add(tensor, 1.5, 0.25, out=tensor, layout="CHW"), torch.Tensor)
    multiply_add(byte, 1.5, 0.25, out=floats)  # type: ignore[arg-type]
    multiply_add(half, 1.5, 0.25, out=byte)  # type: ignore[arg-type]
    multiply_add(floats, 1.5, 0.25, out=half)  # type: ignore[arg-type]
    multiply_add(half, 1.5, 0.25, inplace=True)  # type: ignore[call-overload]


def check_union(image: NDArray[np.float32] | torch.Tensor) -> None:
    assert_type(multiply_add(img=image, factor=1.5, value=0.25, layout="CHW"), NDArray[np.float32] | torch.Tensor)
