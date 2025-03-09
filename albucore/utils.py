from __future__ import annotations

import sys
from functools import wraps
from typing import Any, Callable, Literal, Union

if sys.version_info >= (3, 10):
    from typing import Concatenate, ParamSpec
else:
    from typing_extensions import Concatenate, ParamSpec

import cv2
import numpy as np

NUM_RGB_CHANNELS = 3
MONO_CHANNEL_DIMENSIONS = 2
NUM_MULTI_CHANNEL_DIMENSIONS = 3
FOUR = 4
TWO = 2

MAX_OPENCV_WORKING_CHANNELS = 4

NormalizationType = Literal["image", "image_per_channel", "min_max", "min_max_per_channel"]

P = ParamSpec("P")

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float16"): 1.0,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.float16: 1.0,
    np.float32: 1.0,
    np.float64: 1.0,
    np.int32: 2147483647,
}

NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.int32: cv2.CV_32S,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
    np.dtype("int32"): cv2.CV_32S,
}


def maybe_process_in_chunks(
    process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        args: Additional positional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray, *process_args: P.args, **process_kwargs: P.kwargs) -> np.ndarray:
        # Merge args and kwargs
        all_args = (*args, *process_args)
        all_kwargs: dict[str, Any] = kwargs | process_kwargs

        num_channels = get_num_channels(img)
        if num_channels > MAX_OPENCV_WORKING_CHANNELS:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == TWO:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, *all_args, **all_kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, *all_args, **all_kwargs)
                    chunks.append(chunk)
            return np.dstack(chunks)

        return process_fn(img, *all_args, **all_kwargs)

    return __process_fn


def clip(img: np.ndarray, dtype: Any, inplace: bool = False) -> np.ndarray:
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if inplace:
        return np.clip(img, 0, max_value, out=img)
    return np.clip(img, 0, max_value).astype(dtype, copy=False)


def clipped(func: Callable[Concatenate[np.ndarray, P], np.ndarray]) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        dtype = img.dtype
        result = func(img, *args, **kwargs)

        if result.dtype == np.uint8:
            return result

        return clip(result, dtype)

    return wrapped_function


def get_num_channels(image: np.ndarray) -> int:
    return image.shape[2] if image.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else 1


def is_grayscale_image(image: np.ndarray) -> bool:
    return get_num_channels(image) == 1


def get_opencv_dtype_from_numpy(value: np.ndarray | int | np.dtype | object) -> int:
    if isinstance(value, np.ndarray):
        value = value.dtype
    return NPDTYPE_TO_OPENCV_DTYPE[value]


def is_rgb_image(image: np.ndarray) -> bool:
    return get_num_channels(image) == NUM_RGB_CHANNELS


def is_multispectral_image(image: np.ndarray) -> bool:
    num_channels = get_num_channels(image)
    return num_channels not in {1, 3}


def convert_value(value: np.ndarray | float, num_channels: int) -> float | np.ndarray:
    """Convert a value to a float or numpy array based on its shape and number of channels.

    Args:
        value: Input value to convert (numpy array, float, or int)
        num_channels: Number of channels in the target image

    Returns:
        float: If value is a scalar or 1D array that should be converted to scalar
        np.ndarray: If value is a multi-dimensional array or channel vector

    Raises:
        TypeError: If value is of unsupported type
    """
    # Handle scalar types
    if isinstance(value, (float, int, np.float32, np.float64)):
        return float(value) if isinstance(value, (float, int)) else value.item()

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        # Return scalars and 0-dim arrays as float
        if value.ndim == 0:
            return value.item()

        # Return multi-dimensional arrays as-is
        if value.ndim > 1:
            return value

        # Handle 1D arrays
        if len(value) == 1 or num_channels == 1 or len(value) < num_channels:
            return float(value[0])

        return value[:num_channels]

    raise TypeError(f"Unsupported value type: {type(value)}")


ValueType = Union[np.ndarray, float, int]


def get_max_value(dtype: np.dtype) -> float:
    if dtype not in MAX_VALUES_BY_DTYPE:
        msg = (
            f"Can't infer the maximum value for dtype {dtype}. "
            "You need to specify the maximum value manually by passing the max_value argument."
        )
        raise RuntimeError(msg)
    return MAX_VALUES_BY_DTYPE[dtype]
