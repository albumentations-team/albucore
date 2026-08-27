from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate

import numpy as np

from albucore.utils import P

Array = np.ndarray[Any, Any]


def preserve_channel_dim(
    func: Callable[Concatenate[Array, P], Array],
) -> Callable[Concatenate[Array, P], Array]:
    """Preserve single channel dimension when OpenCV drops it."""

    @wraps(func)
    def wrapped_function(img: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        shape = img.shape
        result = func(img, *args, **kwargs)
        # If input had 3 dims with last dim = 1, and OpenCV dropped it to 2 dims
        if len(shape) == 3 and shape[-1] == 1 and result.ndim == 2:
            return np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


__all__ = [
    "preserve_channel_dim",
]
