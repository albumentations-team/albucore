from functools import wraps
from typing import Any, Callable, TypeVar

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


def clip(img: FloatOrUIntArray, dtype: Any, maxval: float) -> FloatOrUIntArray:
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(
    func: Callable[Concatenate[FloatOrUIntArray, P], FloatOrUIntArray],
) -> Callable[Concatenate[FloatOrUIntArray, P], FloatOrUIntArray]:
    @wraps(func)
    def wrapped_function(img: FloatOrUIntArray, *args: P.args, **kwargs: P.kwargs) -> FloatOrUIntArray:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE[dtype]
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function
