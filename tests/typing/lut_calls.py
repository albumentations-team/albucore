"""Consumer checks for uint8 lookup values and output containers."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import apply_uint8_lut


def lookup_calls(image: NDArray[np.uint8], table: NDArray[np.uint8], tensor: torch.Tensor) -> None:
    assert_type(apply_uint8_lut(img=image, lut=table), NDArray[np.uint8])
    assert_type(apply_uint8_lut(image, table, out=image), NDArray[np.uint8])
    assert_type(apply_uint8_lut(tensor, table), torch.Tensor)
    assert_type(apply_uint8_lut(tensor, table, out=tensor, layout="CHW"), torch.Tensor)
    apply_uint8_lut(image, table, inplace=True)  # type: ignore[call-overload]


def union_lookup(image: NDArray[np.uint8] | torch.Tensor, table: NDArray[np.uint8]) -> None:
    assert_type(apply_uint8_lut(image, table), NDArray[np.uint8] | torch.Tensor)
