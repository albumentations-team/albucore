"""Consumer checks for histogram equalization's uint8 contract."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from albucore import equalize_uint8


def equalize_calls(image: NDArray[np.uint8], mask: NDArray[np.uint8], floating: NDArray[np.float32]) -> None:
    assert_type(equalize_uint8(image), NDArray[np.uint8])
    assert_type(equalize_uint8(image, mode="pil", mask=mask), NDArray[np.uint8])
    equalize_uint8(floating)  # type: ignore[arg-type]
    equalize_uint8(image, mask=floating)  # type: ignore[arg-type]
    equalize_uint8(image, mode="unsupported")  # type: ignore[arg-type]
