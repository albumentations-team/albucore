"""dtype conversion: uint8 <-> float32 in [0,1]."""

import cv2
import numpy as np

from albucore.decorators import preserve_channel_dim
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ImageFloat32,
    ImageType,
    ImageUInt8,
    clip,
    get_max_value,
    get_num_channels,
)


def to_float_numpy(img: ImageType, max_value: float | None = None) -> ImageFloat32:
    if max_value is None:
        max_value = get_max_value(img.dtype)
    return (img / max_value).astype(np.float32, copy=False)


@preserve_channel_dim
def to_float_opencv(img: ImageType, max_value: float | None = None) -> ImageFloat32:
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
def to_float_lut(img: ImageUInt8, max_value: float | None = None) -> ImageFloat32:
    if img.dtype != np.uint8:
        raise ValueError("LUT method is only applicable for uint8 images")

    if max_value is None:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    lut = (np.arange(256, dtype=np.float32) / max_value).astype(np.float32)
    return cv2.LUT(img, lut)


def to_float(img: ImageType, max_value: float | None = None) -> ImageFloat32:
    if img.dtype == np.float32:
        return img
    if img.dtype == np.uint8:
        return to_float_lut(img, max_value)
    return to_float_numpy(img, max_value)


def from_float_numpy(img: ImageFloat32, target_dtype: np.dtype, max_value: float | None = None) -> ImageType:
    if max_value is None:
        max_value = get_max_value(target_dtype)
    return clip(np.rint(img * max_value), target_dtype, inplace=True)


@preserve_channel_dim
def from_float_opencv(img: ImageFloat32, target_dtype: np.dtype, max_value: float | None = None) -> ImageType:
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


def from_float(img: ImageFloat32, target_dtype: np.dtype, max_value: float | None = None) -> ImageType:
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

    if img.dtype == np.float32:
        return from_float_opencv(img, target_dtype, max_value)

    return from_float_numpy(img, target_dtype, max_value)
