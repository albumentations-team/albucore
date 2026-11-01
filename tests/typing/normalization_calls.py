"""Normalization result dtype and explicit-output relationships."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import normalize


def check_numpy(image: NDArray[np.uint8], half: NDArray[np.float16], single: NDArray[np.float32]) -> None:
    assert_type(normalize(image, 0.5, 2.0), NDArray[np.float32])
    assert_type(normalize(half, 0.5, 2.0, out=half), NDArray[np.float16])
    assert_type(normalize(image, 0.5, 2.0, out=single), NDArray[np.float32])
    assert_type(normalize(single, 0.5, 2.0, out=single), NDArray[np.float32])
    assert_type(normalize(img=half, mean=0.5, denominator=2.0), NDArray[np.float16])


def check_tensor(image: torch.Tensor) -> None:
    assert_type(normalize(image, 0.5, 2.0, layout="CHW"), torch.Tensor)
    assert_type(normalize(image, 0.5, 2.0, out=image, layout="NCDHW"), torch.Tensor)


def check_union(image: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor) -> None:
    assert_type(normalize(image, 0.5, 2.0, layout="CHW"), NDArray[np.float16] | NDArray[np.float32] | torch.Tensor)
