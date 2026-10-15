from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
from torch import Tensor

Array = np.ndarray[Any, Any]
_FunctionT = TypeVar("_FunctionT", bound=Callable[..., Array | Tensor])


def preserve_channel_dim(func: _FunctionT) -> _FunctionT:
    """Preserve single channel dimension when OpenCV drops it."""

    @wraps(func)
    def wrapped_function(img: Array | Tensor, *args: Any, **kwargs: Any) -> Array | Tensor:
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and result.ndim == 2:
            return np.expand_dims(result, axis=-1)
        return result

    return cast("_FunctionT", wrapped_function)


__all__ = [
    "preserve_channel_dim",
]
