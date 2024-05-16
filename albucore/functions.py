from typing import Sequence, Union

import cv2
import numpy as np

from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    NPDTYPE_TO_OPENCV_DTYPE,
    clip,
    clipped,
    get_num_channels,
    preserve_channel_dim,
)


@clipped
def _multiply_non_uint_optimized(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    if isinstance(multiplier, float) or get_num_channels(img) > MAX_OPENCV_WORKING_CHANNELS or img.dtype == np.uint32:
        return np.multiply(img, multiplier)
    return cv2.multiply(img, multiplier, dtype=NPDTYPE_TO_OPENCV_DTYPE[img.dtype])


@preserve_channel_dim
def _multiply_uint_optimized(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    if num_channels == 1:
        lut = np.arange(0, max_value + 1, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, dtype)
        return cv2.LUT(img, lut)

    lut = [np.arange(0, max_value + 1, dtype=np.float32)] * num_channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, dtype)

    images = [cv2.LUT(img[:, :, i], lut[:, i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


def multiply(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    num_channels = get_num_channels(img)
    if num_channels == 1 and isinstance(multiplier, Sequence):
        multiplier = multiplier[0]

    if img.dtype == np.uint8 and num_channels <= MAX_OPENCV_WORKING_CHANNELS:
        return _multiply_uint_optimized(img, multiplier)

    return _multiply_non_uint_optimized(img, multiplier)
