from functools import wraps
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Concatenate, ParamSpec

FloatOrUIntArray = TypeVar("FloatOrUIntArray", NDArray[np.float32], NDArray[np.uint8])

P = ParamSpec("P")

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.float32: 1.0,
}

NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}


def clip(img: FloatOrUIntArray, dtype: Any) -> FloatOrUIntArray:
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    return np.clip(img, 0, max_value).astype(dtype)


def clipped(
    func: Callable[Concatenate[FloatOrUIntArray, P], FloatOrUIntArray],
) -> Callable[Concatenate[FloatOrUIntArray, P], FloatOrUIntArray]:
    @wraps(func)
    def wrapped_function(img: FloatOrUIntArray, *args: P.args, **kwargs: P.kwargs) -> FloatOrUIntArray:
        dtype = img.dtype
        return clip(func(img, *args, **kwargs), dtype)

    return wrapped_function
