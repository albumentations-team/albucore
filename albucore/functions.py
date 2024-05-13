from typing import cast

import cv2
import numpy as np

from albucore.utils import MAX_VALUES_BY_DTYPE, FloatOrUIntArray, clipped

__all__ = ["multiply_by_constant"]


@clipped
def multiply_by_constant(img: FloatOrUIntArray, value: float) -> FloatOrUIntArray:
    if img.dtype == np.uint8:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        lut = np.arange(0, max_value + 1) * value
        lut = np.clip(lut, 0, max_value).astype(img.dtype)
        return cv2.LUT(img, lut)

    return cast(FloatOrUIntArray, img * value)
