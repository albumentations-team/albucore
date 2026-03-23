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
    """Convert image to float32 in [0, 1] by dividing by ``max_value``.

    Routing:
    - **float32**: no-op (returned as-is).
    - **uint8**: LUT via ``cv2.LUT`` — 256-element float32 lookup table; fastest path.
    - **other dtypes**: NumPy division (``img / max_value``).

    Alternative: ``to_float_lut``, ``to_float_opencv``, ``to_float_numpy`` for explicit backends.

    Args:
        img: Input image of any dtype, shape ``(H, W, C)`` or batch/volume.
        max_value: Divisor. Defaults to dtype max (255 for uint8, 65535 for uint16, etc.).

    Returns:
        float32 image with same spatial shape, values in [0, 1] for standard integer dtypes.
    """
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
        # NumPy rint beats ``cv2.multiply`` + clip on router (e.g. 128²x9 float32 → uint8).
        return clip(np.rint(img_float * max_value), target_dtype, inplace=False)

    # C<=4: NumPy multiply + rint (faster than cv2.multiply on typical sizes; cv2.multiply(float (H,W,1), scalar)
    # does not match NumPy elementwise semantics).
    return clip(np.rint(img * max_value), target_dtype, inplace=False)


def from_float(img: ImageFloat32, target_dtype: np.dtype, max_value: float | None = None) -> ImageType:
    """Convert a float32 image back to an integer dtype by scaling and rounding.

    Inverse of ``to_float``. Values are multiplied by ``max_value``, rounded with ``np.rint``,
    and clipped to the target dtype range.

    Routing:
    - **target == float32**: no-op (returned as-is).
    - **float32 input, C ≤ 4**: NumPy ``rint(img * max_value)`` then clip — fastest on benchmarks
      (``benchmarks/benchmark_grayscale_paths.py``).
    - **float32 input, C > 4**: same NumPy path (``cv2.multiply`` broadcast is slower here).
    - **non-float32 input**: ``from_float_numpy`` (generic path).

    Alternative: ``from_float_numpy``, ``from_float_opencv`` for explicit backends.

    Args:
        img: Float32 image, shape ``(H, W, C)`` or batch/volume.
        target_dtype: Output dtype (e.g. ``np.uint8``).
        max_value: Multiplier before rounding. Defaults to dtype max (255 for uint8, etc.).

    Returns:
        Image in ``target_dtype``, same spatial shape, values clipped to dtype range.
    """
    if target_dtype == np.float32:
        return img

    if img.dtype == np.float32:
        return from_float_opencv(img, target_dtype, max_value)

    return from_float_numpy(img, target_dtype, max_value)
