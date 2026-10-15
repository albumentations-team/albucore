from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any, TypeVar, cast

import numpy as np
from torch import Tensor

Array = np.ndarray[Any, Any]
_FunctionT = TypeVar("_FunctionT", bound=Callable[..., Array | Tensor])


def preserve_channel_dim(func: _FunctionT) -> _FunctionT:
    """Preserve NumPy's single channel dimension when OpenCV drops it."""
    try:
        image_parameter = next(iter(signature(func).parameters))
    except (TypeError, ValueError):
        image_parameter = "img"

    @wraps(func)
    def wrapped_function(*args: Any, **kwargs: Any) -> Array | Tensor:
        if args:
            img = args[0]
        else:
            img = kwargs.pop(image_parameter if image_parameter in kwargs else "img")
            args = (img,)
        shape = img.shape
        result = func(*args, **kwargs)
        if (
            len(shape) == 3
            and shape[-1] == 1
            and result.ndim == 2
            and isinstance(img, np.ndarray)
            and isinstance(result, np.ndarray)
        ):
            return np.expand_dims(result, axis=-1)
        return result

    return cast("_FunctionT", wrapped_function)


__all__ = [
    "preserve_channel_dim",
]
