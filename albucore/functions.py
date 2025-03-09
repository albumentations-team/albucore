from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Literal

import cv2
import numpy as np
import simsimd as ss
import stringzilla as sz

from albucore.decorators import contiguous, preserve_channel_dim
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    MONO_CHANNEL_DIMENSIONS,
    NormalizationType,
    ValueType,
    clip,
    clipped,
    convert_value,
    get_max_value,
    get_num_channels,
)

np_operations = {"multiply": np.multiply, "add": np.add, "power": np.power}

cv2_operations = {"multiply": cv2.multiply, "add": cv2.add, "power": cv2.pow}


def add_weighted_simsimd(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    original_shape = img1.shape
    original_dtype = img1.dtype

    if img2.dtype != original_dtype:
        img2 = clip(img2.astype(original_dtype, copy=False), original_dtype, inplace=True)

    return np.frombuffer(
        ss.wsum(img1.reshape(-1), img2.astype(original_dtype, copy=False).reshape(-1), alpha=weight1, beta=weight2),
        dtype=original_dtype,
    ).reshape(
        original_shape,
    )


def add_array_simsimd(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    return add_weighted_simsimd(img, 1, value, 1)


def multiply_by_constant_simsimd(img: np.ndarray, value: float) -> np.ndarray:
    return add_weighted_simsimd(img, value, np.zeros_like(img), 0)


def add_constant_simsimd(img: np.ndarray, value: float) -> np.ndarray:
    return add_weighted_simsimd(img, 1, (np.ones_like(img) * value).astype(img.dtype, copy=False), 1)


def create_lut_array(
    dtype: type[np.number],
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
) -> np.ndarray:
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.uint8 and operation == "add":
        value = np.trunc(value)

    value = np.array(value, dtype=np.float32).reshape(-1, 1)
    lut = np.arange(0, max_value + 1, dtype=np.float32)

    if operation in np_operations:
        return np_operations[operation](lut, value)

    raise ValueError(f"Unsupported operation: {operation}")


@contiguous
def sz_lut(img: np.ndarray, lut: np.ndarray, inplace: bool = True) -> np.ndarray:
    if not inplace:
        img = img.copy()

    sz.translate(memoryview(img), memoryview(lut), inplace=True)
    return img


def apply_lut(
    img: np.ndarray,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    inplace: bool,
) -> np.ndarray:
    dtype = img.dtype

    if isinstance(value, (int, float)):
        lut = create_lut_array(dtype, value, operation)
        return sz_lut(img, clip(lut, dtype, inplace=False), False)

    num_channels = img.shape[-1]

    luts = clip(create_lut_array(dtype, value, operation), dtype, inplace=False)
    return cv2.merge([sz_lut(img[:, :, i], luts[i], inplace) for i in range(num_channels)])


def prepare_value_opencv(
    img: np.ndarray,
    value: np.ndarray | float,
    operation: Literal["add", "multiply"],
) -> np.ndarray:
    return (
        _prepare_scalar_value(img, value, operation)
        if isinstance(value, (int, float))
        else _prepare_array_value(img, value, operation)
    )


def _prepare_scalar_value(
    img: np.ndarray,
    value: float,
    operation: Literal["add", "multiply"],
) -> np.ndarray | float:
    if operation == "add" and img.dtype == np.uint8:
        value = int(value)
    num_channels = get_num_channels(img)
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        if operation == "add":
            # Cast to float32 if value is negative to handle potential underflow issues
            cast_type = np.float32 if value < 0 else img.dtype
            return np.full(img.shape, value, dtype=cast_type)
        if operation == "multiply":
            return np.full(img.shape, value, dtype=np.float32)
    return value


def _prepare_array_value(
    img: np.ndarray,
    value: np.ndarray,
    operation: Literal["add", "multiply"],
) -> np.ndarray:
    if value.dtype == np.float64:
        value = value.astype(np.float32, copy=False)
    if value.ndim == 1:
        value = value.reshape(1, 1, -1)
    value = np.broadcast_to(value, img.shape)
    if operation == "add" and img.dtype == np.uint8:
        if np.all(value >= 0):
            return clip(value, np.uint8, inplace=False)
        return np.trunc(value).astype(np.float32, copy=False)
    return value


def apply_numpy(
    img: np.ndarray,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
) -> np.ndarray:
    if operation == "add" and img.dtype == np.uint8:
        value = np.int16(value)

    return np_operations[operation](img.astype(np.float32, copy=False), value)


def multiply_lut(img: np.ndarray, value: np.ndarray | float, inplace: bool) -> np.ndarray:
    return apply_lut(img, value, "multiply", inplace)


@preserve_channel_dim
def multiply_opencv(img: np.ndarray, value: np.ndarray | float) -> np.ndarray:
    value = prepare_value_opencv(img, value, "multiply")
    if img.dtype == np.uint8:
        return cv2.multiply(img.astype(np.float32, copy=False), value)
    return cv2.multiply(img, value)


def multiply_numpy(img: np.ndarray, value: float | np.ndarray) -> np.ndarray:
    return apply_numpy(img, value, "multiply")


@clipped
def multiply_by_constant(img: np.ndarray, value: float, inplace: bool) -> np.ndarray:
    if img.dtype == np.uint8:
        return multiply_lut(img, value, inplace)
    if img.dtype == np.float32:
        return multiply_numpy(img, value)
    return multiply_opencv(img, value)


@clipped
def multiply_by_vector(img: np.ndarray, value: np.ndarray, num_channels: int, inplace: bool) -> np.ndarray:
    # Handle uint8 images separately to use 1a lookup table for performance
    if img.dtype == np.uint8:
        return multiply_lut(img, value, inplace)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return multiply_numpy(img, value)
    return multiply_opencv(img, value)


@clipped
def multiply_by_array(img: np.ndarray, value: np.ndarray) -> np.ndarray:
    return multiply_opencv(img, value)


def multiply(img: np.ndarray, value: ValueType, inplace: bool = False) -> np.ndarray:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        return multiply_by_constant(img, value, inplace)

    if isinstance(value, np.ndarray) and value.ndim == 1:
        return multiply_by_vector(img, value, num_channels, inplace)

    return multiply_by_array(img, value)


@preserve_channel_dim
def add_opencv(img: np.ndarray, value: np.ndarray | float, inplace: bool = False) -> np.ndarray:
    value = prepare_value_opencv(img, value, "add")

    # Convert to float32 if:
    # 1. uint8 image with negative scalar value
    # 2. uint8 image with non-uint8 array value
    needs_float = img.dtype == np.uint8 and (
        (isinstance(value, (int, float)) and value < 0) or (isinstance(value, np.ndarray) and value.dtype != np.uint8)
    )

    if needs_float:
        return cv2.add(
            img.astype(np.float32, copy=False),
            value if isinstance(value, (int, float)) else value.astype(np.float32, copy=False),
        )

    # Use img as the destination array if inplace=True
    dst = img if inplace else None
    return cv2.add(img, value, dst=dst)


def add_numpy(img: np.ndarray, value: float | np.ndarray) -> np.ndarray:
    return apply_numpy(img, value, "add")


def add_lut(img: np.ndarray, value: np.ndarray | float, inplace: bool) -> np.ndarray:
    return apply_lut(img, value, "add", inplace)


@clipped
def add_constant(img: np.ndarray, value: float, inplace: bool = False) -> np.ndarray:
    return add_opencv(img, value, inplace)


@clipped
def add_vector(img: np.ndarray, value: np.ndarray, inplace: bool) -> np.ndarray:
    if img.dtype == np.uint8:
        return add_lut(img, value, inplace)
    return add_opencv(img, value, inplace)


@clipped
def add_array(img: np.ndarray, value: np.ndarray, inplace: bool = False) -> np.ndarray:
    return add_opencv(img, value, inplace)


def add(img: np.ndarray, value: ValueType, inplace: bool = False) -> np.ndarray:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        if value == 0:
            return img

        if img.dtype == np.uint8:
            value = int(value)

        return add_constant(img, value, inplace)

    return add_vector(img, value, inplace) if value.ndim == 1 else add_array(img, value, inplace)


def normalize_numpy(img: np.ndarray, mean: float | np.ndarray, denominator: float | np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    img -= mean
    return img * denominator


@preserve_channel_dim
def normalize_opencv(img: np.ndarray, mean: float | np.ndarray, denominator: float | np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mean_img = np.zeros_like(img, dtype=np.float32)
    denominator_img = np.zeros_like(img, dtype=np.float32)

    # If mean or denominator are scalar, convert them to arrays
    if isinstance(mean, (float, int)):
        mean = np.full(img.shape, mean, dtype=np.float32)
    if isinstance(denominator, (float, int)):
        denominator = np.full(img.shape, denominator, dtype=np.float32)

    # Ensure the shapes match for broadcasting
    mean_img = (mean_img + mean).astype(np.float32, copy=False)
    denominator_img = denominator_img + denominator

    result = cv2.subtract(img, mean_img)
    return cv2.multiply(result, denominator_img, dtype=cv2.CV_32F)


@preserve_channel_dim
def normalize_lut(img: np.ndarray, mean: float | np.ndarray, denominator: float | np.ndarray) -> np.ndarray:
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


def power_numpy(img: np.ndarray, exponent: float | np.ndarray) -> np.ndarray:
    return apply_numpy(img, exponent, "power")


@preserve_channel_dim
def power_opencv(img: np.ndarray, value: float) -> np.ndarray:
    """Handle the 'power' operation for OpenCV."""
    if img.dtype == np.float32:
        # For float32 images, cv2.pow works directly
        return cv2.pow(img, value)
    if img.dtype == np.uint8 and int(value) == value:
        # For uint8 images, cv2.pow works directly if value is actual integer, even if it's type is float
        return cv2.pow(img, value)
    if img.dtype == np.uint8 and isinstance(value, float):
        # For uint8 images, convert to float32, apply power, then convert back to uint8
        img_float = img.astype(np.float32, copy=False)
        return cv2.pow(img_float, value)

    raise ValueError(f"Unsupported image type {img.dtype} for power operation with value {value}")


# @preserve_channel_dim
def power_lut(img: np.ndarray, exponent: float | np.ndarray, inplace: bool = False) -> np.ndarray:
    return apply_lut(img, exponent, "power", inplace)


@clipped
def power(img: np.ndarray, exponent: ValueType, inplace: bool = False) -> np.ndarray:
    num_channels = get_num_channels(img)
    exponent = convert_value(exponent, num_channels)
    if img.dtype == np.uint8:
        return power_lut(img, exponent, inplace)

    if isinstance(exponent, (float, int)):
        return power_opencv(img, exponent)

    return power_numpy(img, exponent)


def add_weighted_numpy(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    return img1.astype(np.float32, copy=False) * weight1 + img2.astype(np.float32, copy=False) * weight2


@preserve_channel_dim
def add_weighted_opencv(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    return cv2.addWeighted(img1, weight1, img2, weight2, 0)


@preserve_channel_dim
def add_weighted_lut(
    img1: np.ndarray,
    weight1: float,
    img2: np.ndarray,
    weight2: float,
    inplace: bool = False,
) -> np.ndarray:
    dtype = img1.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if weight1 == 1 and weight2 == 0:
        return img1

    if weight1 == 0 and weight2 == 1:
        return img2

    if weight1 == 0 and weight2 == 0:
        return np.zeros_like(img1)

    if weight1 == 1 and weight2 == 1:
        return add_array(img1, img2, inplace)

    lut1 = np.arange(0, max_value + 1, dtype=np.float32) * weight1
    result1 = cv2.LUT(img1, lut1)

    lut2 = np.arange(0, max_value + 1, dtype=np.float32) * weight2
    result2 = cv2.LUT(img2, lut2)

    return add_opencv(result1, result2, inplace)


@clipped
def add_weighted(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError(f"The input images must have the same shape. Got {img1.shape} and {img2.shape}.")

    return add_weighted_simsimd(img1, weight1, img2, weight2)


def multiply_add_numpy(img: np.ndarray, factor: ValueType, value: ValueType) -> np.ndarray:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img, dtype=img.dtype)

    result = np.multiply(img, factor) if factor != 0 else np.zeros_like(img)

    return result if value == 0 else np.add(result, value)


@preserve_channel_dim
def multiply_add_opencv(img: np.ndarray, factor: ValueType, value: ValueType) -> np.ndarray:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img)

    result = img.astype(np.float32, copy=False)
    result = (
        cv2.multiply(result, np.ones_like(result) * factor, dtype=cv2.CV_64F)
        if factor != 0
        else np.zeros_like(result, dtype=img.dtype)
    )
    return result if value == 0 else cv2.add(result, np.ones_like(result) * value, dtype=cv2.CV_64F)


def multiply_add_lut(img: np.ndarray, factor: ValueType, value: ValueType, inplace: bool) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    if isinstance(factor, (float, int)) and isinstance(value, (float, int)):
        lut = clip(np.arange(0, max_value + 1, dtype=np.float32) * factor + value, dtype, inplace=False)
        return sz_lut(img, lut, inplace)

    if isinstance(factor, np.ndarray) and factor.shape != ():
        factor = factor.reshape(-1, 1)

    if isinstance(value, np.ndarray) and value.shape != ():
        value = value.reshape(-1, 1)

    luts = clip(np.arange(0, max_value + 1, dtype=np.float32) * factor + value, dtype, inplace=True)

    return cv2.merge([sz_lut(img[:, :, i], luts[i], inplace) for i in range(num_channels)])


@clipped
def multiply_add(img: np.ndarray, factor: ValueType, value: ValueType, inplace: bool = False) -> np.ndarray:
    num_channels = get_num_channels(img)
    factor = convert_value(factor, num_channels)
    value = convert_value(value, num_channels)

    if img.dtype == np.uint8:
        return multiply_add_lut(img, factor, value, inplace)

    return multiply_add_opencv(img, factor, value)


@preserve_channel_dim
def normalize_per_image_opencv(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    eps = 1e-4

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        img = np.expand_dims(img, axis=-1)

    if normalization == "image" or (img.shape[-1] == 1 and normalization == "image_per_channel"):
        mean = img.mean().item()
        std = img.std().item() + eps
        if img.shape[-1] > MAX_OPENCV_WORKING_CHANNELS:
            mean = np.full_like(img, mean)
            std = np.full_like(img, std)
        normalized_img = cv2.divide(cv2.subtract(img, mean), std)
        return np.clip(normalized_img, -20, 20, out=normalized_img)

    if normalization == "image_per_channel":
        mean, std = cv2.meanStdDev(img)
        mean = mean[:, 0]
        std = std[:, 0]

        if img.shape[-1] > MAX_OPENCV_WORKING_CHANNELS:
            mean = np.full_like(img, mean)
            std = np.full_like(img, std)

        normalized_img = cv2.divide(cv2.subtract(img, mean), std, dtype=cv2.CV_32F)
        return np.clip(normalized_img, -20, 20, out=normalized_img)

    if normalization == "min_max" or (img.shape[-1] == 1 and normalization == "min_max_per_channel"):
        img_min = img.min()
        img_max = img.max()
        return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if normalization == "min_max_per_channel":
        img_min = img.min(axis=(0, 1))
        img_max = img.max(axis=(0, 1))

        if img.shape[-1] > MAX_OPENCV_WORKING_CHANNELS:
            img_min = np.full_like(img, img_min)
            img_max = np.full_like(img, img_max)

        return np.clip(
            cv2.divide(cv2.subtract(img, img_min), (img_max - img_min + eps), dtype=cv2.CV_32F),
            -20,
            20,
            out=img,
        )

    raise ValueError(f"Unknown normalization method: {normalization}")


@preserve_channel_dim
def normalize_per_image_numpy(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    eps = 1e-4

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        img = np.expand_dims(img, axis=-1)

    if normalization == "image":
        mean = img.mean()
        std = img.std() + eps
        normalized_img = (img - mean) / std
        return np.clip(normalized_img, -20, 20, out=normalized_img)

    if normalization == "image_per_channel":
        pixel_mean = img.mean(axis=(0, 1))
        pixel_std = img.std(axis=(0, 1)) + eps
        normalized_img = (img - pixel_mean) / pixel_std
        return np.clip(normalized_img, -20, 20, out=normalized_img)

    if normalization == "min_max":
        img_min = img.min()
        img_max = img.max()
        return np.clip((img - img_min) / (img_max - img_min + eps), -20, 20, out=img)

    if normalization == "min_max_per_channel":
        img_min = img.min(axis=(0, 1))
        img_max = img.max(axis=(0, 1))
        return np.clip((img - img_min) / (img_max - img_min + eps), -20, 20, out=img)

    raise ValueError(f"Unknown normalization method: {normalization}")


@preserve_channel_dim
def normalize_per_image_lut(img: np.ndarray, normalization: NormalizationType) -> np.ndarray:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    eps = 1e-4
    num_channels = get_num_channels(img)

    if img.ndim == MONO_CHANNEL_DIMENSIONS:
        img = np.expand_dims(img, axis=-1)

    if normalization == "image" or (img.shape[-1] == 1 and normalization == "image_per_channel"):
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

    if normalization == "min_max" or (img.shape[-1] == 1 and normalization == "min_max_per_channel"):
        img_min = img.min()
        img_max = img.max()
        lut = (np.arange(0, max_value + 1, dtype=np.float32) - img_min) / (img_max - img_min + eps)
        return cv2.LUT(img, lut).clip(-20, 20)

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
    if img.dtype == np.uint8 and normalization != "per_image_per_channel":
        return normalize_per_image_lut(img, normalization)

    return normalize_per_image_opencv(img, normalization)


def to_float_numpy(img: np.ndarray, max_value: float | None = None) -> np.ndarray:
    if max_value is None:
        max_value = get_max_value(img.dtype)
    return (img / max_value).astype(np.float32, copy=False)


@preserve_channel_dim
def to_float_opencv(img: np.ndarray, max_value: float | None = None) -> np.ndarray:
    if max_value is None:
        max_value = get_max_value(img.dtype)

    img_float = img.astype(np.float32, copy=False)

    num_channels = get_num_channels(img)

    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        # For images with more than 4 channels, create a full-sized divisor
        max_value_array = np.full_like(img_float, max_value)
        return cv2.divide(img_float, max_value_array)

    # For images with 4 or fewer channels, use scalar division
    return cv2.divide(img_float, max_value)


@preserve_channel_dim
def to_float_lut(img: np.ndarray, max_value: float | None = None) -> np.ndarray:
    if img.dtype != np.uint8:
        raise ValueError("LUT method is only applicable for uint8 images")

    if max_value is None:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    lut = np.arange(256, dtype=np.float32) / max_value
    return cv2.LUT(img, lut)


def to_float(img: np.ndarray, max_value: float | None = None) -> np.ndarray:
    if img.dtype == np.float64:
        return img.astype(np.float32, copy=False)
    if img.dtype == np.float32:
        return img
    if img.dtype == np.uint8:
        return to_float_lut(img, max_value)
    return to_float_numpy(img, max_value)


def from_float_numpy(img: np.ndarray, target_dtype: np.dtype, max_value: float | None = None) -> np.ndarray:
    if max_value is None:
        max_value = get_max_value(target_dtype)
    return clip(np.rint(img * max_value), target_dtype, inplace=True)


@preserve_channel_dim
def from_float_opencv(img: np.ndarray, target_dtype: np.dtype, max_value: float | None = None) -> np.ndarray:
    if max_value is None:
        max_value = get_max_value(target_dtype)

    img_float = img.astype(np.float32, copy=False)

    num_channels = get_num_channels(img)

    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        # For images with more than 4 channels, create a full-sized multiplier
        max_value_array = np.full_like(img_float, max_value)
        return clip(np.rint(cv2.multiply(img_float, max_value_array)), target_dtype, inplace=False)

    # For images with 4 or fewer channels, use scalar multiplication
    return clip(np.rint(img * max_value), target_dtype, inplace=False)


def from_float(img: np.ndarray, target_dtype: np.dtype, max_value: float | None = None) -> np.ndarray:
    """Convert a floating-point image to the specified target data type.

    This function converts an input floating-point image to the specified target data type,
    scaling the values appropriately based on the max_value parameter or the maximum value
    of the target data type.

    Args:
        img (np.ndarray): Input floating-point image array.
        target_dtype (np.dtype): Target numpy data type for the output image.
        max_value (float | None, optional): Maximum value to use for scaling. If None,
            the maximum value of the target data type will be used. Defaults to None.

    Returns:
        np.ndarray: Image converted to the target data type.

    Notes:
        - If the input image is of type float32, the function uses OpenCV for faster processing.
        - For other input types, it falls back to a numpy-based implementation.
        - The function clips values to ensure they fit within the range of the target data type.
    """
    if target_dtype == np.float32:
        return img

    if target_dtype == np.float64:
        return img.astype(np.float32, copy=False)

    if img.dtype == np.float32:
        return from_float_opencv(img, target_dtype, max_value)

    return from_float_numpy(img, target_dtype, max_value)


@contiguous
def hflip_numpy(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, ...]


@preserve_channel_dim
def hflip_cv2(img: np.ndarray) -> np.ndarray:
    # OpenCV's flip function has a limitation of 512 channels
    if img.ndim > 2 and img.shape[2] > 512:
        return _flip_multichannel(img, flip_code=1)
    return cv2.flip(img, 1)


def hflip(img: np.ndarray) -> np.ndarray:
    return hflip_cv2(img)


@preserve_channel_dim
def vflip_cv2(img: np.ndarray) -> np.ndarray:
    # OpenCV's flip function has a limitation of 512 channels
    if img.ndim > 2 and img.shape[2] > 512:
        return _flip_multichannel(img, flip_code=0)
    return cv2.flip(img, 0)


@contiguous
def vflip_numpy(img: np.ndarray) -> np.ndarray:
    return img[::-1, ...]


def vflip(img: np.ndarray) -> np.ndarray:
    return vflip_cv2(img)


def _flip_multichannel(img: np.ndarray, flip_code: int) -> np.ndarray:
    """Process images with more than 512 channels by splitting into chunks.

    OpenCV's flip function has a limitation where it can only handle images with up to 512 channels.
    This function works around that limitation by splitting the image into chunks of 512 channels,
    flipping each chunk separately, and then concatenating the results.

    Args:
        img: Input image with many channels
        flip_code: OpenCV flip code (0 for vertical, 1 for horizontal, -1 for both)

    Returns:
        Flipped image with all channels preserved
    """
    # Get image dimensions
    height, width = img.shape[:2]
    num_channels = 1 if img.ndim == 2 else img.shape[2]

    # If the image has 2 dimensions or fewer than 512 channels, use cv2.flip directly
    if img.ndim == 2 or num_channels <= 512:
        return cv2.flip(img, flip_code)

    # Process in chunks of 512 channels
    chunk_size = 512
    result_chunks = []

    for i in range(0, num_channels, chunk_size):
        end_idx = min(i + chunk_size, num_channels)
        chunk = img[:, :, i:end_idx]
        flipped_chunk = cv2.flip(chunk, flip_code)

        # Ensure the chunk maintains its dimensionality
        # This is needed when the last chunk has only one channel and cv2.flip reduces the dimensions
        if flipped_chunk.ndim == 2 and img.ndim == 3:
            flipped_chunk = np.expand_dims(flipped_chunk, axis=2)

        result_chunks.append(flipped_chunk)

    # Concatenate the chunks along the channel dimension
    return np.concatenate(result_chunks, axis=2)


def float32_io(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Decorator to ensure float32 input/output for image processing functions.

    This decorator converts the input image to float32 before passing it to the wrapped function,
    and then converts the result back to the original dtype if it wasn't float32.

    Args:
        func (Callable[..., np.ndarray]): The image processing function to be wrapped.

    Returns:
        Callable[..., np.ndarray]: A wrapped function that handles float32 conversion.

    Example:
        @float32_io
        def some_image_function(img: np.ndarray) -> np.ndarray:
            # Function implementation
            return processed_img
    """

    @wraps(func)
    def float32_wrapper(img: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        input_dtype = img.dtype
        if input_dtype != np.float32:
            img = to_float(img)
        result = func(img, *args, **kwargs)

        return from_float(result, target_dtype=input_dtype) if input_dtype != np.float32 else result

    return float32_wrapper


def uint8_io(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Decorator to ensure uint8 input/output for image processing functions.

    This decorator converts the input image to uint8 before passing it to the wrapped function,
    and then converts the result back to the original dtype if it wasn't uint8.

    Args:
        func (Callable[..., np.ndarray]): The image processing function to be wrapped.

    Returns:
        Callable[..., np.ndarray]: A wrapped function that handles uint8 conversion.

    Example:
        @uint8_io
        def some_image_function(img: np.ndarray) -> np.ndarray:
            # Function implementation
            return processed_img
    """

    @wraps(func)
    def uint8_wrapper(img: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        input_dtype = img.dtype

        if input_dtype != np.uint8:
            img = from_float(img, target_dtype=np.uint8)

        result = func(img, *args, **kwargs)

        return to_float(result) if input_dtype != np.uint8 else result

    return uint8_wrapper
