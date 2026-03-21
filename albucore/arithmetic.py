"""Arithmetic: add, multiply, power, value-normalize, weighted sum, multiply-add."""

from typing import Literal

import cv2
import numkong as nk
import numpy as np
import stringzilla as sz

from albucore.decorators import contiguous, preserve_channel_dim
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ImageFloat32,
    ImageType,
    ImageUInt8,
    SupportedDType,
    ValueType,
    clip,
    clipped,
    convert_value,
    get_num_channels,
)
from albucore.weighted import (
    add_array_numkong,
    add_weighted_numkong,
    multiply_by_constant_numkong,
)

np_operations = {"multiply": np.multiply, "add": np.add, "power": np.power}

cv2_operations = {"multiply": cv2.multiply, "add": cv2.add, "power": cv2.pow}


def create_lut_array(
    dtype: SupportedDType,
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
def sz_lut(img: ImageUInt8, lut: ImageUInt8, inplace: bool = True) -> ImageUInt8:
    """Apply lookup table using stringzilla. Only works with uint8 images and uint8 LUTs."""
    if not inplace:
        img = img.copy()

    sz.translate(memoryview(img), memoryview(lut), inplace=True)
    return img


def apply_lut(
    img: ImageUInt8,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    inplace: bool,
) -> ImageUInt8:
    """Apply lookup table operation. Only works with uint8 images."""
    dtype = img.dtype

    if isinstance(value, (int, float)):
        lut = create_lut_array(dtype, value, operation)
        return sz_lut(img, clip(lut, dtype, inplace=False), False)

    num_channels = img.shape[-1]

    luts = clip(create_lut_array(dtype, value, operation), dtype, inplace=False)

    result = np.empty_like(img, dtype=dtype)

    for i in range(num_channels):
        result[..., i] = sz_lut(img[..., i], luts[i], inplace)

    return result


def prepare_value_opencv(
    img: ImageType,
    value: np.ndarray | float,
    operation: Literal["add", "multiply"],
) -> np.ndarray:
    return (
        _prepare_scalar_value(img, value, operation)
        if isinstance(value, (int, float))
        else _prepare_array_value(img, value, operation)
    )


def _prepare_scalar_value(
    img: ImageType,
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
    img: ImageType,
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
    img: ImageType,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
) -> ImageFloat32:
    if operation == "add" and img.dtype == np.uint8:
        value = np.int16(value)

    return np_operations[operation](img.astype(np.float32, copy=False), value)


def multiply_lut(img: ImageUInt8, value: np.ndarray | float, inplace: bool) -> ImageUInt8:
    return apply_lut(img, value, "multiply", inplace)


@preserve_channel_dim
def multiply_opencv(img: ImageType, value: np.ndarray | float) -> ImageFloat32:
    value = prepare_value_opencv(img, value, "multiply")
    if img.dtype == np.uint8:
        return cv2.multiply(img.astype(np.float32, copy=False), value)
    return cv2.multiply(img, value)


def multiply_numpy(img: ImageType, value: float | np.ndarray) -> ImageFloat32:
    return apply_numpy(img, value, "multiply")


@clipped
def multiply_by_constant(img: ImageType, value: float, inplace: bool) -> ImageType:
    if img.dtype == np.uint8:
        return multiply_lut(img, value, inplace)
    if img.dtype == np.float32:
        return multiply_by_constant_numkong(img, value)
    return multiply_opencv(img, value)


@clipped
def multiply_by_vector(img: ImageType, value: np.ndarray, num_channels: int, inplace: bool) -> ImageType:
    # Handle uint8 images separately to use 1a lookup table for performance
    if img.dtype == np.uint8:
        return multiply_lut(img, value, inplace)
    # Check if the number of channels exceeds the maximum that OpenCV can handle
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return multiply_numpy(img, value)
    return multiply_opencv(img, value)


@clipped
def multiply_by_array(img: ImageType, value: np.ndarray) -> ImageType:
    if img.dtype == np.float32:
        return multiply_numpy(img, value)
    return multiply_opencv(img, value)


def multiply(img: ImageType, value: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        return multiply_by_constant(img, value, inplace)

    if isinstance(value, np.ndarray) and value.ndim == 1:
        return multiply_by_vector(img, value, num_channels, inplace)

    return multiply_by_array(img, value)


@preserve_channel_dim
def add_opencv(img: ImageType, value: np.ndarray | float, inplace: bool = False) -> ImageType:
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


def add_numpy(img: ImageType, value: float | np.ndarray) -> ImageFloat32:
    return apply_numpy(img, value, "add")


def add_lut(img: ImageUInt8, value: np.ndarray | float, inplace: bool) -> ImageUInt8:
    return apply_lut(img, value, "add", inplace)


@clipped
def add_constant(img: ImageType, value: float, inplace: bool = False) -> ImageType:
    return add_opencv(img, value, inplace)


@clipped
def add_vector(img: ImageType, value: np.ndarray, inplace: bool) -> ImageType:
    if img.dtype == np.uint8:
        return add_lut(img, value, inplace)
    return add_opencv(img, value, inplace)


@clipped
def add_array(img: ImageType, value: np.ndarray, inplace: bool = False) -> ImageType:
    if not inplace and value.shape == img.shape and value.dtype == img.dtype and img.dtype in (np.uint8, np.float32):
        return add_array_numkong(img, value)
    return add_opencv(img, value, inplace)


def add(img: ImageType, value: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        if value == 0:
            return img

        if img.dtype == np.uint8:
            value = int(value)

        return add_constant(img, value, inplace)

    return add_vector(img, value, inplace) if value.ndim == 1 else add_array(img, value, inplace)


def normalize_numpy(img: ImageType, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    img = img.astype(np.float32, copy=True)
    # Ensure mean and denominator are float32 to avoid dtype promotion
    mean = mean.astype(np.float32, copy=False) if isinstance(mean, np.ndarray) else np.float32(mean)
    denominator = (
        denominator.astype(np.float32, copy=False) if isinstance(denominator, np.ndarray) else np.float32(denominator)
    )
    img -= mean
    return (img * denominator).astype(np.float32, copy=True)


@preserve_channel_dim
def normalize_opencv(img: ImageType, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    img = img.astype(np.float32, copy=False)
    mean_img = np.zeros_like(img, dtype=np.float32)
    denominator_img = np.zeros_like(img, dtype=np.float32)

    # Ensure the shapes match for broadcasting
    mean_img = (mean_img + mean).astype(np.float32, copy=False)
    denominator_img = denominator_img + denominator

    result = cv2.subtract(img, mean_img)
    return cv2.multiply(result, denominator_img, dtype=cv2.CV_32F)


@preserve_channel_dim
def normalize_lut(img: ImageUInt8, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    if isinstance(denominator, (float, int)) and isinstance(mean, (float, int)):
        lut = ((np.arange(0, max_value + 1, dtype=np.float32) - mean) * denominator).astype(np.float32)
        return cv2.LUT(img, lut)

    # Vectorized LUT creation - shape: (256, num_channels)
    arange_vals = np.arange(0, max_value + 1, dtype=np.float32)
    luts = ((arange_vals[:, np.newaxis] - mean) * denominator).astype(np.float32)

    # Pre-allocate result array
    result = np.empty_like(img, dtype=np.float32)
    for i in range(num_channels):
        result[..., i] = cv2.LUT(img[..., i], luts[:, i])

    return result


def normalize(img: ImageType, mean: ValueType, denominator: ValueType) -> ImageFloat32:
    num_channels = get_num_channels(img)
    denominator = convert_value(denominator, num_channels)
    mean = convert_value(mean, num_channels)

    if img.dtype == np.uint8:
        return normalize_lut(img, mean, denominator)

    if img.dtype == np.float32:
        return normalize_numpy(img, mean, denominator)

    # Fallback to OpenCV for other dtypes
    return normalize_opencv(img, mean, denominator)


def power_numpy(img: ImageType, exponent: float | np.ndarray) -> ImageFloat32:
    return apply_numpy(img, exponent, "power")


@preserve_channel_dim
def power_opencv(img: ImageType, value: float) -> ImageFloat32:
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


def power_lut(img: ImageUInt8, exponent: float | np.ndarray, inplace: bool = False) -> ImageUInt8:
    return apply_lut(img, exponent, "power", inplace)


@clipped
def power(img: ImageType, exponent: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    exponent = convert_value(exponent, num_channels)
    if img.dtype == np.uint8:
        return power_lut(img, exponent, inplace)

    if isinstance(exponent, (float, int)):
        return power_opencv(img, exponent)

    return power_numpy(img, exponent)


def add_weighted_numpy(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageFloat32:
    return img1.astype(np.float32, copy=False) * weight1 + img2.astype(np.float32, copy=False) * weight2


@preserve_channel_dim
def add_weighted_opencv(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    return cv2.addWeighted(img1, weight1, img2, weight2, 0)


@preserve_channel_dim
def add_weighted_lut(
    img1: ImageUInt8,
    weight1: float,
    img2: ImageUInt8,
    weight2: float,
    inplace: bool = False,
) -> ImageFloat32:
    """Add weighted using LUT. Only works with uint8 images."""
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
def add_weighted(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    if img1.shape != img2.shape:
        raise ValueError(f"The input images must have the same shape. Got {img1.shape} and {img2.shape}.")

    return add_weighted_numkong(img1, weight1, img2, weight2)


def _multiply_add_numkong_scalar(img: ImageFloat32, factor: float, value: float) -> ImageFloat32:
    if factor == 0 and value == 0:
        return np.zeros_like(img, dtype=np.float32)
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    scaled = nk.scale(nk.Tensor(flat), alpha=float(factor), beta=float(value))
    return np.frombuffer(scaled, dtype=np.float32).reshape(img.shape)


def multiply_add_numpy(img: ImageType, factor: ValueType, value: ValueType) -> ImageType:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img, dtype=img.dtype)

    result = np.multiply(img, factor) if factor != 0 else np.zeros_like(img)

    return result if value == 0 else np.add(result, value)


@preserve_channel_dim
def multiply_add_opencv(img: ImageType, factor: ValueType, value: ValueType) -> ImageFloat32:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img, dtype=np.float32)

    result = img.astype(np.float32, copy=False)
    result = (
        cv2.multiply(result, np.ones_like(result) * factor, dtype=cv2.CV_32F)
        if factor != 0
        else np.zeros_like(result, dtype=img.dtype)
    )
    return result if value == 0 else cv2.add(result, np.ones_like(result) * value, dtype=cv2.CV_32F)


def multiply_add_lut(img: ImageUInt8, factor: ValueType, value: ValueType, inplace: bool) -> ImageUInt8:
    """Apply multiply-add operation using LUT. Only works with uint8 images."""
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

    luts = clip(np.arange(0, max_value + 1, dtype=np.float32) * factor + value, dtype, inplace=False)

    result = np.empty_like(img, dtype=dtype)
    for i in range(num_channels):
        result[..., i] = sz_lut(img[..., i], luts[i], inplace)

    return result


@clipped
def multiply_add(img: ImageType, factor: ValueType, value: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    factor = convert_value(factor, num_channels)
    value = convert_value(value, num_channels)

    if img.dtype == np.uint8:
        return multiply_add_lut(img, factor, value, inplace)

    if isinstance(factor, (int, float)) and isinstance(value, (int, float)):
        return _multiply_add_numkong_scalar(img.astype(np.float32, copy=False), float(factor), float(value))

    return multiply_add_opencv(img, factor, value)
