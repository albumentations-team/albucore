import sys
from functools import wraps
from typing import Callable

import numpy as np

from albucore.utils import MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS, P

if sys.version_info >= (3, 10):
    from typing import Concatenate
else:
    from typing_extensions import Concatenate


def contiguous(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Ensure that input img is contiguous and the output array is also contiguous."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        # Ensure the input array is contiguous
        img = np.require(img, requirements=["C_CONTIGUOUS"])
        # Call the original function with the contiguous input
        result = func(img, *args, **kwargs)
        # Ensure the output array is contiguous
        if not result.flags["C_CONTIGUOUS"]:
            return np.require(result, requirements=["C_CONTIGUOUS"])

        return result

    return wrapped_function


def preserve_channel_dim(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Preserve dummy channel dim."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS and shape[-1] == 1 and result.ndim == MONO_CHANNEL_DIMENSIONS:
            return np.expand_dims(result, axis=-1)

        if len(shape) == MONO_CHANNEL_DIMENSIONS and result.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            return result[:, :, 0]
        return result

    return wrapped_function
