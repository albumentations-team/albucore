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

    luts = []
    for i in range(num_channels):
        lut = np.arange(0, max_value + 1, dtype=np.float32) * multiplier[i]
        lut = clip(lut, dtype)
        luts.append(lut)

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


@preserve_channel_dim
@clipped
def multiply_with_opencv(img: np.ndarray, multiplier: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(multiplier, np.ndarray):
        multiplier = multiplier.astype(np.float32)
    return cv2.multiply(img.astype(np.float32), multiplier)


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
    if img.dtype == np.uint8:
        return multiply_with_lut(img, multiplier)
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return multiply_with_numpy(img, multiplier)
    return multiply_with_opencv(img, multiplier)


def multiply_by_array(img: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    return multiply_with_opencv(img, multiplier)


@contiguous
@clipped
def multiply(img: np.ndarray, multiplier: Union[Sequence[float], np.ndarray, float]) -> np.ndarray:
    num_channels = get_num_channels(img)

    if num_channels == 1 and (
        isinstance(multiplier, Sequence) or isinstance(multiplier, np.ndarray) and multiplier.ndim == 1
    ):
        multiplier = multiplier[0]

    if isinstance(multiplier, float):
        return multiply_by_constant(img, multiplier)

    if isinstance(multiplier, Sequence):
        multiplier = np.array(multiplier, np.float32)

    if isinstance(multiplier, np.ndarray) and multiplier.ndim == 1:
        return multiply_by_vector(img, multiplier)

    return multiply_by_array(img, multiplier)
