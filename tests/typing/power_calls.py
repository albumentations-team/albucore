"""Power preserves the concrete input dtype and accepts a matching explicit output."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import power


def check_outputs(
    bytes_image: NDArray[np.uint8], half: NDArray[np.float16], floats: NDArray[np.float32], tensor: torch.Tensor
) -> None:
    assert_type(power(bytes_image, 1.5), NDArray[np.uint8])
    assert_type(power(bytes_image, 1.5, out=bytes_image), NDArray[np.uint8])
    assert_type(power(half, 1.5, out=half), NDArray[np.float16])
    assert_type(power(floats, 1.5, out=floats), NDArray[np.float32])
    assert_type(power(tensor, 1.5, layout="CHW", out=tensor), torch.Tensor)
    power(bytes_image, 1.5, out=floats)  # type: ignore[arg-type]
    power(half, 1.5, out=bytes_image)  # type: ignore[arg-type]
    power(floats, 1.5, out=half)  # type: ignore[arg-type]
    power(half, 1.5, inplace=True)  # type: ignore[call-overload]


def check_union(image: NDArray[np.float32] | torch.Tensor) -> None:
    assert_type(power(img=image, exponent=1.5, layout="CHW"), NDArray[np.float32] | torch.Tensor)
