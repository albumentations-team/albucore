from typing import Sequence, Union

import cv2
import numpy as np

from albucore.utils import (
    MAX_VALUES_BY_DTYPE,
    clip,
    clipped,
    is_grayscale_image,
    maybe_process_in_chunks,
    preserve_channel_dim,
)


@clipped
def _multiply_non_uint8_optimized(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    img = img.astype(np.float32)
    return np.multiply(img, multiplier)


@preserve_channel_dim
def _multiply_uint8_optimized(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]
    if isinstance(multiplier, float):
        lut = np.arange(0, max_value + 1, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8)
        func = maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img)

    if is_grayscale_image(img):
        multiplier = multiplier[0]
        lut = np.arange(0, max_value + 1, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8)
        func = maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img)

    num_channels = img.shape[-1]
    lut = [np.arange(0, max_value + 1, dtype=np.float32)] * num_channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, np.uint8)

    images = []
    for i in range(num_channels):
        func = maybe_process_in_chunks(cv2.LUT, lut=lut[:, i])
        images.append(func(img[:, :, i]))
    return np.stack(images, axis=-1)


def multiply(img: np.ndarray, multiplier: Union[Sequence[float], float]) -> np.ndarray:
    if img.dtype == np.uint8:
        return _multiply_uint8_optimized(img, multiplier)

    return _multiply_non_uint8_optimized(img, multiplier)
