"""Arithmetic: add, multiply, power, value-normalize, weighted sum, multiply-add."""

from typing import Literal

import cv2
import numpy as np

from albucore.decorators import preserve_channel_dim
from albucore.lut import apply_uint8_lut as _apply_uint8_lut
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
from albucore.weighted import add_array_numkong, add_weighted_numkong

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

    v = np.asarray(value, dtype=np.float32).reshape(-1, 1)
    lut = np.arange(max_value + 1, dtype=np.float32)
    if operation not in np_operations:
        raise ValueError(f"Unsupported operation: {operation}")
    return np_operations[operation](lut, v)


def apply_lut(
    img: ImageUInt8,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    inplace: bool,
) -> ImageUInt8:
    """Apply lookup table operation. Only works with uint8 images.

    ``inplace=True`` avoids an extra buffer when the whole image uses one LUT (scalar path) or when
    callers write channel-wise into a preallocated ``result`` (see per-channel loop).
    """
    dtype = img.dtype

    if isinstance(value, (int, float)):
        lut = clip(create_lut_array(dtype, value, operation), dtype, inplace=True)
        return _apply_uint8_lut(img, lut, inplace=inplace)

    luts = clip(create_lut_array(dtype, value, operation), dtype, inplace=True)
    return _apply_uint8_lut(img, luts, inplace=inplace)


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


def multiply_lut(img: ImageUInt8, value: np.ndarray | float, inplace: bool = False) -> ImageUInt8:
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
def multiply_by_constant(img: ImageType, value: float, inplace: bool = False) -> ImageType:
    if img.dtype == np.uint8:
        return multiply_lut(img, value, inplace)
    # float32: match 0.0.40 (`multiply_numpy`); OpenCV / NumKong scalar paths regressed on router grid
    return multiply_numpy(img, value)


@clipped
def multiply_by_vector(img: ImageType, value: np.ndarray, inplace: bool = False) -> ImageType:
    if img.dtype == np.uint8:
        # LUT beats OpenCV float-multiply + clip for per-channel uint8 (see benchmarks/benchmark_grayscale_paths.py).
        return multiply_lut(img, value, inplace)
    return multiply_numpy(img, value)


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
        return multiply_by_vector(img, value, inplace)

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

    dst = img if inplace else None
    return cv2.add(img, value, dst=dst)


def add_numpy(img: ImageType, value: float | np.ndarray) -> ImageFloat32:
    return apply_numpy(img, value, "add")


def add_lut(img: ImageUInt8, value: np.ndarray | float, inplace: bool = False) -> ImageUInt8:
    return apply_lut(img, value, "add", inplace)


@clipped
def add_constant(img: ImageType, value: float, inplace: bool = False) -> ImageType:
    if img.dtype == np.float32:
        return add_numpy(img, value)
    # uint8 (all C): OpenCV + ``prepare_value_opencv`` broadcast. NumKong ``add_constant_numkong``
    # regressed vs 0.0.40 on large (H,W,C>4) router cells; helper remains in ``weighted`` for benches.
    return add_opencv(img, value, inplace)


@clipped
def add_vector(img: ImageType, value: np.ndarray, inplace: bool = False) -> ImageType:
    if img.dtype == np.uint8:
        return add_lut(img, value, inplace)
    return add_numpy(img, value)


@clipped
def add_array(img: ImageType, value: np.ndarray) -> ImageType:
    """Elementwise ``img + value`` for a full array ``value``.

    float32 → NumPy; uint8 with same shape and dtype → NumKong ``blend``; otherwise OpenCV.
    There is no ``inplace`` flag: an in-place OpenCV path was slower than the NumKong out-of-place
    path for same-shape uint8 in benchmarks; use ``numpy.copyto`` yourself if you must reuse a buffer.
    """
    if img.dtype == np.float32:
        # Benchmarks: NumPy broadcast add beats NumKong and OpenCV scalar/tensor prep on float32.
        return add_numpy(img, value)
    if value.shape == img.shape and value.dtype == img.dtype:
        return add_array_numkong(img, value)
    return add_opencv(img, value)


def add(img: ImageType, value: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        if value == 0:
            return img

        if img.dtype == np.uint8:
            value = int(value)

        return add_constant(img, value, inplace)

    return add_vector(img, value, inplace) if value.ndim == 1 else add_array(img, value)


def normalize_numpy(img: ImageType, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    img_f = img.astype(np.float32, copy=False)
    mean_f = mean.astype(np.float32, copy=False) if isinstance(mean, np.ndarray) else np.float32(mean)
    denom_f = (
        denominator.astype(np.float32, copy=False) if isinstance(denominator, np.ndarray) else np.float32(denominator)
    )
    # Fused: (img - mean) * denom = img * denom - mean * denom (no full-image copy vs subtract-then-multiply).
    if isinstance(mean_f, np.ndarray) or isinstance(denom_f, np.ndarray):
        n_dim = img_f.ndim
        c = int(img_f.shape[-1])

        def _broadcast_last(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
            if isinstance(x, np.ndarray) and x.shape == (c,):
                return x.reshape((1,) * (n_dim - 1) + (c,))
            return x

        mean_b = _broadcast_last(mean_f)
        denom_b = _broadcast_last(denom_f)
        offset = mean_b * denom_b
        return img_f * denom_b - offset

    return img_f * denom_f - mean_f * denom_f


@preserve_channel_dim
def normalize_opencv(img: ImageType, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    img_f = img.astype(np.float32, copy=False)
    if isinstance(mean, (int, float)):
        mean_img = np.full_like(img_f, mean)
    else:
        mean_img = np.broadcast_to(mean.astype(np.float32, copy=False), img_f.shape)
    if isinstance(denominator, (int, float)):
        denom_img = np.full_like(img_f, denominator)
    else:
        denom_img = np.broadcast_to(denominator.astype(np.float32, copy=False), img_f.shape)
    result = cv2.subtract(img_f, mean_img)
    return cv2.multiply(result, denom_img, dtype=cv2.CV_32F)


@preserve_channel_dim
def normalize_lut(img: ImageUInt8, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    num_channels = get_num_channels(img)

    x = np.arange(max_value + 1, dtype=np.float32)
    if isinstance(denominator, (float, int)) and isinstance(mean, (float, int)):
        return cv2.LUT(img, (x - mean) * denominator)

    luts = (x[:, np.newaxis] - mean) * denominator

    # Pre-allocate result array
    result = np.empty_like(img, dtype=np.float32)
    for i in range(num_channels):
        result[..., i] = cv2.LUT(img[..., i], luts[:, i])

    return result


def _normalize_is_identity(mean: float | np.ndarray, denominator: float | np.ndarray, *, eps: float = 1e-5) -> bool:
    if isinstance(mean, np.ndarray):
        if not bool(np.all(np.abs(mean.astype(np.float64, copy=False)) <= eps)):
            return False
    elif abs(float(mean)) > eps:
        return False
    if isinstance(denominator, np.ndarray):
        d = denominator.astype(np.float64, copy=False)
        if not bool(np.all(np.abs(d - 1.0) <= eps)):
            return False
    elif abs(float(denominator) - 1.0) > eps:
        return False
    return True


def normalize(img: ImageType, mean: ValueType, denominator: ValueType) -> ImageFloat32:
    """Affine normalize with **caller-fixed** ``mean`` and ``denominator``: ``(img - mean) * denominator``.

    This is **not** ``normalize_per_image`` (which estimates stats from ``img``). Pass scalars or
    length-``C`` vectors (after ``convert_value``) for per-channel constants, e.g. ImageNet on uint8
    in 0-255: ``mean = mean_01 * 255``, ``denominator = 1 / (std_01 * 255)``.
    """
    num_channels = get_num_channels(img)
    denominator = convert_value(denominator, num_channels)
    mean = convert_value(mean, num_channels)

    if img.dtype == np.uint8:
        return normalize_lut(img, mean, denominator)

    if _normalize_is_identity(mean, denominator):
        return img.astype(np.float32, copy=False)

    return normalize_numpy(img, mean, denominator)


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
        return add_array(img1, img2)

    base = np.arange(max_value + 1, dtype=np.float32)
    result1 = cv2.LUT(img1, base * weight1)
    result2 = cv2.LUT(img2, base * weight2)

    return add_opencv(result1, result2, inplace)


@preserve_channel_dim
def _add_weighted_opencv(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    return cv2.addWeighted(img1, weight1, img2, weight2, 0)


@clipped
def add_weighted(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    if img1.shape != img2.shape:
        raise ValueError(f"The input images must have the same shape. Got {img1.shape} and {img2.shape}.")

    # NK blend degrades on large float32 tensors (memory pressure at >~4M elems on ARM);
    # OpenCV addWeighted is ~4x faster at 1024x1024x9 float32.
    # Threshold calibrated from benchmarks/reliable_benchmark_numkong_vs_albucore_backends.md.
    if img1.dtype == np.float32 and img1.size > 4_000_000:
        return _add_weighted_opencv(img1, weight1, img2, weight2)

    return add_weighted_numkong(img1, weight1, img2, weight2)


def _broadcast_channel_vector(x: ValueType, n_dim: int, c: int) -> float | np.ndarray:
    if isinstance(x, np.ndarray) and x.shape == (c,):
        return x.reshape((1,) * (n_dim - 1) + (c,))
    return x


def _is_all_zero_param(x: float | np.ndarray) -> bool:
    if isinstance(x, np.ndarray):
        return not np.any(x)
    return float(x) == 0.0


def multiply_add_numpy(img: ImageType, factor: ValueType, value: ValueType) -> ImageFloat32:
    img_f = img.astype(np.float32, copy=False)

    def _scalar_float(x: ValueType) -> float | None:
        if isinstance(x, np.ndarray):
            return None
        return float(x)

    sf, sv = _scalar_float(factor), _scalar_float(value)
    if sf is not None and sv is not None and sf == 0.0 and sv == 0.0:
        return np.zeros_like(img_f, dtype=np.float32)

    if sf is not None and sv is not None and sf == 1.0 and sv == 0.0:
        return img_f

    n_dim = img_f.ndim
    c = int(img_f.shape[-1])
    f_b = _broadcast_channel_vector(factor, n_dim, c)
    v_b = _broadcast_channel_vector(value, n_dim, c)

    result = np.zeros_like(img_f) if _is_all_zero_param(f_b) else np.multiply(img_f, f_b)
    return result if _is_all_zero_param(v_b) else np.add(result, v_b)


@preserve_channel_dim
def multiply_add_opencv(img: ImageType, factor: ValueType, value: ValueType) -> ImageFloat32:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img, dtype=np.float32)

    result = img.astype(np.float32, copy=False)
    num_channels = result.shape[-1]
    if factor != 0:
        if isinstance(factor, (int, float)) and num_channels <= MAX_OPENCV_WORKING_CHANNELS:
            result = cv2.multiply(result, factor, dtype=cv2.CV_32F)
        else:
            factor_img = (
                np.full_like(result, factor)
                if isinstance(factor, (int, float))
                else np.broadcast_to(
                    factor.astype(np.float32, copy=False),
                    result.shape,
                )
            )
            result = cv2.multiply(result, factor_img, dtype=cv2.CV_32F)
    else:
        result = np.zeros_like(result)
    if value != 0:
        if isinstance(value, (int, float)) and num_channels <= MAX_OPENCV_WORKING_CHANNELS:
            result = cv2.add(result, value, dtype=cv2.CV_32F)
        else:
            value_img = (
                np.full_like(result, value)
                if isinstance(value, (int, float))
                else np.broadcast_to(
                    value.astype(np.float32, copy=False),
                    result.shape,
                )
            )
            result = cv2.add(result, value_img, dtype=cv2.CV_32F)
    return result


def multiply_add_lut(img: ImageUInt8, factor: ValueType, value: ValueType, inplace: bool) -> ImageUInt8:
    """Apply multiply-add operation using LUT. Only works with uint8 images."""
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    domain = np.arange(max_value + 1, dtype=np.float32)

    if isinstance(factor, (float, int)) and isinstance(value, (float, int)):
        lut = clip(domain * factor + value, dtype, inplace=False)
        return _apply_uint8_lut(img, lut, inplace=inplace)

    if isinstance(factor, np.ndarray) and factor.shape != ():
        factor = factor.reshape(-1, 1)

    if isinstance(value, np.ndarray) and value.shape != ():
        value = value.reshape(-1, 1)

    luts = clip(domain * factor + value, dtype, inplace=False)
    return _apply_uint8_lut(img, luts, inplace=inplace)


@clipped
def multiply_add(img: ImageType, factor: ValueType, value: ValueType, inplace: bool = False) -> ImageType:
    num_channels = get_num_channels(img)
    factor = convert_value(factor, num_channels)
    value = convert_value(value, num_channels)

    if img.dtype == np.uint8:
        return multiply_add_lut(img, factor, value, inplace)

    return multiply_add_numpy(img, factor, value)
