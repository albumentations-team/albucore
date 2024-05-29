from typing import Sequence, Union

import cv2
import numpy as np

from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ValueType,
    clip,
    clipped,
    convert_value,
    get_num_channels,
    preserve_channel_dim,
)


@preserve_channel_dim
def multiply_with_lut(img: np.ndarray, value: Union[Sequence[float], float]) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if isinstance(value, (int, float)):
        lut = clip(np.arange(0, max_value + 1, dtype=np.float32) * value, dtype)
        return cv2.LUT(img, lut)

    num_channels = img.shape[-1]

    value = np.array(value, dtype=np.float32).reshape(-1, 1)
    luts = clip(np.arange(0, max_value + 1, dtype=np.float32) * value, dtype)

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


@preserve_channel_dim
@clipped
def multiply_with_opencv(img: np.ndarray, value: Union[np.ndarray, float]) -> np.ndarray:
    return cv2.multiply(img.astype(np.float32), value, dtype=cv2.CV_64F)


@clipped
def multiply_with_numpy(img: np.ndarray, value: Union[float, np.ndarray]) -> np.ndarray:
    return np.multiply(img, value)


def multiply_by_constant(img: np.ndarray, value: float) -> np.ndarray:
    if img.dtype == np.uint8:
        return multiply_with_lut(img, value)
    if img.dtype == np.float32:
        return multiply_with_numpy(img, value)
    return multiply_with_opencv(img, value)


def multiply_by_vector(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    num_channels = get_num_channels(img)
    # Handle uint8 images separately to use a lookup table for performance
    if img.dtype == np.uint8:
        return multiply_with_lut(img, value)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return multiply_with_numpy(img, value)
    return multiply_with_opencv(img, value)


def multiply_by_array(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    return multiply_with_numpy(img, value)


@clipped
def multiply(img: np.ndarray, value: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        return multiply_by_constant(img, value)

    if isinstance(value, np.ndarray) and value.ndim == 1:
        return multiply_by_vector(img, value)

    return multiply_by_array(img, value)


@preserve_channel_dim
@clipped
def add_with_opencv(img: np.ndarray, value: Union[np.ndarray, float]) -> np.ndarray:
    return cv2.add(img.astype(np.float32), value, dtype=cv2.CV_64F)


@clipped
def add_with_numpy(img: np.ndarray, value: Union[float, np.ndarray]) -> np.ndarray:
    return np.add(img.astype(np.float32), value)


@preserve_channel_dim
def add_with_lut(img: np.ndarray, value: Union[Sequence[float], float]) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if isinstance(value, (float, int)):
        lut = clip(np.arange(0, max_value + 1, dtype=np.float32) + value, dtype)
        return cv2.LUT(img, lut)

    num_channels = img.shape[-1]

    value = np.array(value, dtype=np.float32).reshape(-1, 1)

    luts = clip(np.arange(0, max_value + 1, dtype=np.float32) + value, dtype)

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


def add_constant(img: np.ndarray, value: float) -> np.ndarray:
    if img.dtype == np.uint8:
        return add_with_lut(img, value)
    if img.dtype == np.float32:
        return add_with_numpy(img, value)
    return add_with_opencv(img, value)


def add_vector(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    num_channels = get_num_channels(img)
    # Handle uint8 images separately to use a lookup table for performance
    if img.dtype == np.uint8:
        return add_with_lut(img, value)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return add_with_numpy(img, value)
    return add_with_opencv(img, value)


def add_array(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    return add_with_numpy(img, value)


def add(img: np.ndarray, value: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        if value == 0:
            return img
        return add_constant(img, value)

    if isinstance(value, np.ndarray) and value.ndim == 1:
        return add_vector(img, value)

    return add_array(img, value)


def normalize_numpy(
    img: np.ndarray, mean: Union[float, np.ndarray], denominator: Union[float, np.ndarray]
) -> np.ndarray:
    img = img.astype(np.float32)
    img -= mean
    return img * denominator


@preserve_channel_dim
def normalize_opencv(
    img: np.ndarray, mean: Union[float, np.ndarray], denominator: Union[float, np.ndarray]
) -> np.ndarray:
    img = img.astype(np.float32)
    mean_img = np.zeros_like(img, dtype=np.float32)
    denominator_img = np.zeros_like(img, dtype=np.float32)

    # If mean or denominator are scalar, convert them to arrays
    if isinstance(mean, (float, int)):
        mean = np.full(img.shape, mean, dtype=np.float32)
    if isinstance(denominator, (float, int)):
        denominator = np.full(img.shape, denominator, dtype=np.float32)

    # Ensure the shapes match for broadcasting
    mean_img = (mean_img + mean).astype(np.float32)
    denominator_img = denominator_img + denominator

    result = cv2.subtract(img, mean_img)
    return cv2.multiply(result, denominator_img, dtype=cv2.CV_32F)


@preserve_channel_dim
def normalize_lut(img: np.ndarray, mean: Union[float, np.ndarray], denominator: Union[float, np.ndarray]) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    if isinstance(denominator, (float, int)) and isinstance(mean, (float, int)):
        lut = (np.arange(0, max_value + 1, dtype=np.float32) - mean) * denominator
        return cv2.LUT(img, lut)

    if isinstance(denominator, np.ndarray) and denominator.shape != ():
        denominator = denominator.reshape(-1, 1)

    if isinstance(mean, np.ndarray):
        mean = mean.reshape(-1, 1)

    luts = (np.arange(0, max_value + 1, dtype=np.float32) - mean) * denominator

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


def normalize(img: np.ndarray, mean: ValueType, denominator: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    denominator = convert_value(denominator, num_channels)
    mean = convert_value(mean, num_channels)
    if img.dtype == np.uint8:
        return normalize_lut(img, mean, denominator)

    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return normalize_numpy(img, mean, denominator)

    return normalize_opencv(img, mean, denominator)
