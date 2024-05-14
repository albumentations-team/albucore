from functools import wraps
from typing import Any, Callable

import cv2
import numpy as np
from typing_extensions import Concatenate, ParamSpec

MONO_CHANNEL_DIMENSIONS = 2
NUM_MULTI_CHANNEL_DIMENSIONS = 3
FOUR = 4
TWO = 2

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


def maybe_process_in_chunks(
    process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray], **kwargs: Any
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        np.ndarray: Transformed image.

    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = get_num_channels(img)
        if num_channels > FOUR:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == TWO:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            return np.dstack(chunks)

        return process_fn(img, **kwargs)

    return __process_fn


def clip(img: np.ndarray, dtype: Any) -> np.ndarray:
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    return np.clip(img, 0, max_value).astype(dtype)


def clipped(func: Callable[Concatenate[np.ndarray, P], np.ndarray]) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        dtype = img.dtype
        return clip(func(img, *args, **kwargs), dtype)

    return wrapped_function


def get_num_channels(image: np.ndarray) -> int:
    return image.shape[2] if image.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else 1


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


def is_grayscale_image(image: np.ndarray) -> bool:
    return get_num_channels(image) == 1
