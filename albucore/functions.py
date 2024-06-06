from typing import Literal, Sequence, Union

import cv2
import numpy as np

from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    MONO_CHANNEL_DIMENSIONS,
    NormalizationType,
    ValueType,
    clip,
    clipped,
    convert_value,
    get_num_channels,
    preserve_channel_dim,
)

np_operations = {"multiply": np.multiply, "add": np.add, "power": np.power}

cv2_operations = {"multiply": cv2.multiply, "add": cv2.add, "power": cv2.pow}


def create_lut_array(
    max_value: float, value: Union[float, np.ndarray], operation: Literal["add", "multiply", "power"]
) -> np.ndarray:
    value = np.array(value, dtype=np.float32).reshape(-1, 1)
    lut = np.arange(0, max_value + 1, dtype=np.float32)

    if operation in np_operations:
        return np_operations[operation](lut, value)

    raise ValueError(f"Unsupported operation: {operation}")


def apply_lut(
    img: np.ndarray, value: Union[float, np.ndarray], operation: Literal["add", "multiply", "power"]
) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if isinstance(value, (int, float)):
        lut = create_lut_array(max_value, value, operation)
        return cv2.LUT(img, clip(lut, dtype))

    num_channels = img.shape[-1]
    luts = create_lut_array(max_value, value, operation)
    return cv2.merge([cv2.LUT(img[:, :, i], clip(luts[i], dtype)) for i in range(num_channels)])


@preserve_channel_dim
def apply_opencv(
    img: np.ndarray, value: Union[np.ndarray, float], operation: Literal["add", "multiply", "power"]
) -> np.ndarray:
    img_float = img.astype(np.float32)
    if operation in cv2_operations:
        return cv2_operations[operation](img_float, value)

    raise ValueError(f"Unsupported operation: {operation}")


def apply_numpy(
    img: np.ndarray, value: Union[float, np.ndarray], operation: Literal["add", "multiply", "power"]
) -> np.ndarray:
    img_float = img.astype(np.float32)

    if operation in np_operations:
        return np_operations[operation](img_float, value)

    raise ValueError(f"Unsupported operation: {operation}")


@preserve_channel_dim
def multiply_lut(img: np.ndarray, value: Union[Sequence[float], float]) -> np.ndarray:
    return apply_lut(img, value, "multiply")


@preserve_channel_dim
def multiply_opencv(img: np.ndarray, value: Union[np.ndarray, float]) -> np.ndarray:
    return apply_opencv(img, value, "multiply")


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
    return apply_opencv(img, value, "add")


def add_numpy(img: np.ndarray, value: Union[float, np.ndarray]) -> np.ndarray:
    return apply_numpy(img, value, "add")


@preserve_channel_dim
def add_lut(img: np.ndarray, value: Union[Sequence[float], float]) -> np.ndarray:
    return apply_lut(img, value, "add")


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

    return cv2.merge([cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)])


def normalize(img: np.ndarray, mean: ValueType, denominator: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    denominator = convert_value(denominator, num_channels)
    mean = convert_value(mean, num_channels)
    if img.dtype == np.uint8:
        return normalize_lut(img, mean, denominator)

    return normalize_opencv(img, mean, denominator)


def power_numpy(img: np.ndarray, exponent: Union[float, np.ndarray]) -> np.ndarray:
    return apply_numpy(img, exponent, "power")


@preserve_channel_dim
def power_opencv(img: np.ndarray, exponent: Union[float, np.ndarray]) -> np.ndarray:
    return apply_opencv(img, exponent, "power")


@preserve_channel_dim
def power_lut(img: np.ndarray, exponent: Union[float, np.ndarray]) -> np.ndarray:
    return apply_lut(img, exponent, "power")


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


def multiply_add_numpy(img: np.ndarray, factor: ValueType, value: ValueType) -> np.ndarray:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img)

    result = np.multiply(img, factor) if factor != 0 else np.zeros_like(img)
    if value != 0:
        result = np.add(result, value)
    return result


@preserve_channel_dim
def multiply_add_opencv(img: np.ndarray, factor: ValueType, value: ValueType) -> np.ndarray:
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
def multiply_add_lut(img: np.ndarray, factor: ValueType, value: ValueType) -> np.ndarray:
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

    return cv2.merge([cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)])


@clipped
def multiply_add(img: np.ndarray, factor: ValueType, value: ValueType) -> np.ndarray:
    num_channels = get_num_channels(img)
    factor = convert_value(factor, num_channels)
    value = convert_value(value, num_channels)

    if img.dtype == np.uint8:
        return multiply_add_lut(img, factor, value)

    return multiply_add_opencv(img, factor, value)


@preserve_channel_dim
def normalize_per_image_opencv(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    img = img.astype(np.float32)
    eps = 1e-4

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        img = np.expand_dims(img, axis=-1)

    if normalization == "image":
        mean = img.mean().item()
        std = img.std() + eps
        normalized_img = cv2.divide(cv2.subtract(img, mean), std)
        return normalized_img.clip(-20, 20)

    if normalization == "image_per_channel":
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1)) + eps
        normalized_img = cv2.divide(cv2.subtract(img, mean), std, dtype=cv2.CV_32F)
        return normalized_img.clip(-20, 20)

    if normalization == "min_max":
        img_min = img.min()
        img_max = img.max()
        return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if normalization == "min_max_per_channel":
        img_min = img.min(axis=(0, 1))
        img_max = img.max(axis=(0, 1))

        return cv2.divide(cv2.subtract(img, img_min), (img_max - img_min + eps), dtype=cv2.CV_32F).clip(-20, 20)

    raise ValueError(f"Unknown normalization method: {normalization}")


@preserve_channel_dim
def normalize_per_image_numpy(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    img = img.astype(np.float32)
    eps = 1e-4

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        img = np.expand_dims(img, axis=-1)

    if normalization == "image":
        mean = img.mean()
        std = img.std() + eps
        normalized_img = (img - mean) / std
        return normalized_img.clip(-20, 20)

    if normalization == "image_per_channel":
        pixel_mean = img.mean(axis=(0, 1))
        pixel_std = img.std(axis=(0, 1)) + eps
        normalized_img = (img - pixel_mean) / pixel_std
        return normalized_img.clip(-20, 20)

    if normalization == "min_max":
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + eps)

    if normalization == "min_max_per_channel":
        img_min = img.min(axis=(0, 1))
        img_max = img.max(axis=(0, 1))
        return (img - img_min) / (img_max - img_min + eps)

    raise ValueError(f"Unknown normalization method: {normalization}")


@preserve_channel_dim
def normalize_per_image_lut(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    eps = 1e-4
    num_channels = get_num_channels(img)

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        img = np.expand_dims(img, axis=-1)

    if normalization == "image":
        mean = img.mean()
        std = img.std() + eps
        lut = (np.arange(0, max_value + 1, dtype=np.float32) - mean) / std
        return cv2.LUT(img, lut).clip(-20, 20)

    if normalization == "image_per_channel":
        pixel_mean = img.mean(axis=(0, 1))
        pixel_std = img.std(axis=(0, 1)) + eps
        luts = [
            (np.arange(0, max_value + 1, dtype=np.float32) - pixel_mean[c]) / pixel_std[c] for c in range(num_channels)
        ]
        return cv2.merge([cv2.LUT(img[:, :, i], luts[i]).clip(-20, 20) for i in range(num_channels)])

    if normalization == "min_max":
        img_min = img.min()
        img_max = img.max()
        lut = (np.arange(0, max_value + 1, dtype=np.float32) - img_min) / (img_max - img_min + eps)
        return cv2.LUT(img, lut)

    if normalization == "min_max_per_channel":
        img_min = img.min(axis=(0, 1))
        img_max = img.max(axis=(0, 1))
        luts = [
            (np.arange(0, max_value + 1, dtype=np.float32) - img_min[c]) / (img_max[c] - img_min[c] + eps)
            for c in range(num_channels)
        ]

        return cv2.merge([cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)])

    raise ValueError(f"Unknown normalization method: {normalization}")


def normalize_per_image(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    if img.dtype == np.uint8:
        return normalize_per_image_lut(img, normalization)

    num_channels = get_num_channels(img)

    if num_channels <= MAX_OPENCV_WORKING_CHANNELS:
        return normalize_per_image_opencv(img, normalization)

    return normalize_per_image_numpy(img, normalization)
