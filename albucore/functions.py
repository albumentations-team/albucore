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
def multiply_lut(img: np.ndarray, value: Union[Sequence[float], float]) -> np.ndarray:
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
def multiply_opencv(img: np.ndarray, value: Union[np.ndarray, float]) -> np.ndarray:
    return cv2.multiply(img.astype(np.float32), value, dtype=cv2.CV_64F)


def multiply_numpy(img: np.ndarray, value: Union[float, np.ndarray]) -> np.ndarray:
    return np.multiply(img, value)


def multiply_by_constant(img: np.ndarray, value: float) -> np.ndarray:
    if img.dtype == np.uint8:
        return multiply_lut(img, value)
    if img.dtype == np.float32:
        return multiply_numpy(img, value)
    return multiply_opencv(img, value)


def multiply_by_vector(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    num_channels = get_num_channels(img)
    # Handle uint8 images separately to use a lookup table for performance
    if img.dtype == np.uint8:
        return multiply_lut(img, value)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return multiply_numpy(img, value)
    return multiply_opencv(img, value)


def multiply_by_array(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    return multiply_numpy(img, value)


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
def add_opencv(img: np.ndarray, value: Union[np.ndarray, float]) -> np.ndarray:
    return cv2.add(img.astype(np.float32), value, dtype=cv2.CV_64F)


def add_numpy(img: np.ndarray, value: Union[float, np.ndarray]) -> np.ndarray:
    return np.add(img.astype(np.float32), value)


@preserve_channel_dim
def add_lut(img: np.ndarray, value: Union[Sequence[float], float]) -> np.ndarray:
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
        return add_lut(img, value)
    if img.dtype == np.float32:
        return add_numpy(img, value)
    return add_opencv(img, value)


def add_vector(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    num_channels = get_num_channels(img)
    # Handle uint8 images separately to use a lookup table for performance
    if img.dtype == np.uint8:
        return add_lut(img, value)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return add_numpy(img, value)
    return add_opencv(img, value)


def add_array(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    return add_numpy(img, value)


@clipped
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

    return normalize_opencv(img, mean, denominator)


def power_numpy(img: np.ndarray, exponent: Union[float, np.ndarray]) -> np.ndarray:
    return np.power(img, exponent)


@preserve_channel_dim
def power_opencv(img: np.ndarray, exponent: Union[float, np.ndarray]) -> np.ndarray:
    return cv2.pow(img.astype(np.float32), exponent)


@preserve_channel_dim
def power_lut(img: np.ndarray, exponent: Union[float, np.ndarray]) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    if isinstance(exponent, (float, int)):
        lut = clip(np.power(np.arange(0, max_value + 1, dtype=np.float32), exponent), dtype)
        return cv2.LUT(img, lut)

    if isinstance(exponent, np.ndarray) and exponent.shape != ():
        exponent = exponent.reshape(-1, 1)

    luts = clip(np.power(np.arange(0, max_value + 1, dtype=np.float32), exponent), dtype)

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


@clipped
def power(img: np.ndarray, exponent: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    exponent = convert_value(exponent, num_channels)
    if img.dtype == np.uint8:
        return power_lut(img, exponent)

    if isinstance(exponent, (float, int)):
        return power_opencv(img, exponent)

    return power_numpy(img, exponent)


def add_weighted_numpy(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    return img1 * weight1 + img2 * weight2


@preserve_channel_dim
def add_weighted_opencv(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    return cv2.addWeighted(img1.astype(np.float32), weight1, img2.astype(np.float32), weight2, 0)


@preserve_channel_dim
def add_weighted_lut(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    dtype = img1.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if weight1 == 1 and weight2 == 0:
        return img1

    if weight1 == 0 and weight2 == 1:
        return img2

    if weight1 == 0 and weight2 == 0:
        return np.zeros_like(img1)

    if weight1 == 1 and weight2 == 1:
        return add_array(img1, img2)

    lut1 = np.arange(0, max_value + 1, dtype=np.float32) * weight1
    result1 = cv2.LUT(img1, lut1)

    lut2 = np.arange(0, max_value + 1, dtype=np.float32) * weight2
    result2 = cv2.LUT(img2, lut2)

    return add_array(result1, result2)


@clipped
def add_weighted(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError(f"The input images must have the same shape. Got {img1.shape} and {img2.shape}.")

    return add_weighted_opencv(img1, weight1, img2, weight2)


def multiply_add_numpy(img: np.ndarray, value: ValueType, factor: ValueType) -> np.ndarray:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img)
    result = img
    result = np.multiply(result, factor) if factor != 0 else np.zeros_like(result)
    if value != 0:
        result = np.add(result, value)
    return result


@preserve_channel_dim
def multiply_add_opencv(img: np.ndarray, value: ValueType, factor: ValueType) -> np.ndarray:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img)

    result = img.astype(np.float32)
    result = (
        cv2.multiply(result, np.ones_like(result) * factor, dtype=cv2.CV_64F)
        if factor != 0
        else np.zeros_like(result, dtype=img.dtype)
    )
    if value != 0:
        result = cv2.add(result, np.ones_like(result) * value, dtype=cv2.CV_64F)
    return result


@preserve_channel_dim
def multiply_add_lut(img: np.ndarray, value: ValueType, factor: ValueType) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    if isinstance(factor, (float, int)) and isinstance(value, (float, int)):
        lut = clip(np.arange(0, max_value + 1, dtype=np.float32) * factor + value, dtype)
        return cv2.LUT(img, lut)

    if isinstance(factor, np.ndarray) and factor.shape != ():
        factor = factor.reshape(-1, 1)

    if isinstance(value, np.ndarray) and value.shape != ():
        value = value.reshape(-1, 1)

    luts = clip(np.arange(0, max_value + 1, dtype=np.float32) * factor + value, dtype)

    images = [cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)]
    return np.stack(images, axis=-1)


@clipped
def multiply_add(img: np.ndarray, value: ValueType, factor: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    factor = convert_value(factor, num_channels)
    value = convert_value(value, num_channels)

    if img.dtype == np.uint8:
        return multiply_add_lut(img, value, factor)

    return multiply_add_opencv(img, value, factor)
