from typing import Sequence, Union

import cv2
import numpy as np

from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    clip,
    clipped,
    contiguous,
    get_num_channels,
    preserve_channel_dim,
)


@preserve_channel_dim
def multiply_with_lut(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if isinstance(multiplier, float):
        lut = clip(np.arange(0, max_value + 1, dtype=np.float32) * multiplier, dtype)
        return cv2.LUT(img, lut)

    num_channels = img.shape[-1]

    luts = [clip(np.arange(0, max_value + 1, dtype=np.float32) * multiplier[i], dtype) for i in range(num_channels)]

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


@preserve_channel_dim
@clipped
def multiply_with_opencv(img: np.ndarray, multiplier: Union[np.ndarray, float]) -> np.ndarray:
    return cv2.multiply(img.astype(np.float32), multiplier, dtype=cv2.CV_64F)


@clipped
def multiply_with_numpy(img: np.ndarray, multiplier: Union[float, np.ndarray]) -> np.ndarray:
    return np.multiply(img, multiplier)


def multiply_by_constant(img: np.ndarray, multiplier: float) -> np.ndarray:
    if img.dtype == np.uint8:
        return multiply_with_lut(img, multiplier)
    if img.dtype == np.float32:
        return multiply_with_numpy(img, multiplier)
    return multiply_with_opencv(img, multiplier)


def multiply_by_vector(img: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    num_channels = get_num_channels(img)
    # Handle uint8 images separately to use a lookup table for performance
    if img.dtype == np.uint8:
        return multiply_with_lut(img, multiplier)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return multiply_with_numpy(img, multiplier)
    return multiply_with_opencv(img, multiplier)


def multiply_by_array(img: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    return multiply_with_numpy(img, multiplier)


def convert_multiplier(multiplier: Union[Sequence[float], np.ndarray], num_channels: int) -> Union[float, np.ndarray]:
    if isinstance(multiplier, float):
        return multiplier
    if (
        # Case 1: num_channels is 1 and multiplier is a list or tuple
        (
            num_channels == 1
            and (isinstance(multiplier, Sequence) or (isinstance(multiplier, np.ndarray) and multiplier.ndim == 1))
        )
        or
        # Case 2: multiplier length is 1, regardless of num_channels
        (isinstance(multiplier, (Sequence, np.ndarray)) and len(multiplier) == 1)
    ):
        # Convert to a float
        return float(multiplier[0])

    if isinstance(multiplier, Sequence):
        return np.array(multiplier, dtype=np.float64)

    return multiplier


@contiguous
@clipped
def multiply(img: np.ndarray, multiplier: Union[Sequence[Union[int, float]], np.ndarray, float]) -> np.ndarray:
    num_channels = get_num_channels(img)
    multiplier = convert_multiplier(multiplier, num_channels)

    if isinstance(multiplier, float):
        return multiply_by_constant(img, multiplier)

    if isinstance(multiplier, np.ndarray) and multiplier.ndim == 1:
        return multiply_by_vector(img, multiplier)

    return multiply_by_array(img, multiplier)
