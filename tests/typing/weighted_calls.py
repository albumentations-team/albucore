"""Input, output and required-argument relationships for weighted sums."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import add_weighted


def check_numpy(
    first_u8: NDArray[np.uint8],
    second_u8: NDArray[np.uint8],
    first_half: NDArray[np.float16],
    second_half: NDArray[np.float16],
    first_float: NDArray[np.float32],
    second_float: NDArray[np.float32],
) -> None:
    assert_type(add_weighted(first_u8, 0.25, second_u8, 0.75), NDArray[np.uint8])
    assert_type(add_weighted(first_half, 0.25, second_half, 0.75), NDArray[np.float16])
    assert_type(add_weighted(first_float, 0.25, second_float, 0.75), NDArray[np.float32])
    assert_type(add_weighted(first_u8, 0.25, second_u8, 0.75, out=second_u8), NDArray[np.uint8])
    assert_type(add_weighted(first_half, 0.25, second_half, 0.75, out=second_half), NDArray[np.float16])
    assert_type(
        add_weighted(img1=first_float, weight1=0.25, img2=second_float, weight2=0.75, out=first_float),
        NDArray[np.float32],
    )


def check_tensor(first: torch.Tensor, second: torch.Tensor) -> None:
    assert_type(add_weighted(first, 0.25, second, 0.75, layout="CHW"), torch.Tensor)
    assert_type(add_weighted(first, 0.25, second, 0.75, out=second, layout="NCDHW"), torch.Tensor)


def check_numpy_union(image: NDArray[np.float16] | NDArray[np.float32]) -> None:
    assert_type(add_weighted(image, 0.25, image, 0.75), NDArray[np.float16] | NDArray[np.float32])


def check_union(image: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor) -> None:
    if isinstance(image, torch.Tensor):
        assert_type(add_weighted(image, 0.25, image, 0.75, layout="CHW"), torch.Tensor)
    else:
        check_numpy_union(image)
