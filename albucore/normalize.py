"""Per-image normalization (mean/std, min/max) and helpers."""

from typing import Any, cast, overload

import cv2
import numpy as np
import torch

from albucore._float_cast import _cast_float_numpy
from albucore.decorators import preserve_channel_dim
from albucore.lut import _apply_float_lut
from albucore.stats import DEFAULT_EPS, mean_std
from albucore.tensor_bridge import TensorLayout, dispatch_tensor_image
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ImageFloat16,
    ImageFloat32,
    ImageType,
    ImageUInt8,
    NormalizationType,
    get_num_channels,
)


def _restore_float_image_dtype(image: ImageType, result: ImageFloat32) -> ImageFloat16 | ImageFloat32:
    return _cast_float_numpy(result, np.float16) if image.dtype == np.float16 else result


def _compute_image_stats_opencv(img: ImageType) -> tuple[float, float]:
    """Compute global mean and std for an image."""
    eps = DEFAULT_EPS
    if img.ndim > 3:
        mean, std = cv2.meanStdDev(img)
        return float(mean[0, 0]), float(std[0, 0]) + eps
    # 3D (H,W,C): same as pre-stats-router 0.0.40 — two NumPy passes, avoids mean_std dispatch overhead
    return float(img.mean()), float(img.std()) + eps


def _compute_per_channel_stats_opencv(img: ImageType) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std."""
    m, s = mean_std(img, "per_channel", eps=DEFAULT_EPS)
    return np.asarray(m, dtype=np.float64), np.asarray(s, dtype=np.float64)


def _normalize_mean_std_opencv(img: ImageType, mean: float | np.ndarray, std: float | np.ndarray) -> ImageFloat32:
    """Apply mean-std normalization using OpenCV or NumPy based on dimensionality."""
    img_f = _cast_float_numpy(img, np.float32)
    if img_f.ndim > 3 or (img_f.ndim == 3 and img_f.shape[-1] > MAX_OPENCV_WORKING_CHANNELS):
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        normalized_img = cast("ImageFloat32", (img_f - mean_arr) / std_arr)
    else:
        mean_cv2: Any = mean if np.isscalar(mean) else np.asarray(mean, dtype=np.float32)
        std_cv2: Any = std if np.isscalar(std) else np.asarray(std, dtype=np.float32)
        normalized_img = cast(
            "ImageFloat32",
            cv2.divide(cv2.subtract(img_f, mean_cv2, dtype=cv2.CV_32F), std_cv2, dtype=cv2.CV_32F),
        )
    return np.clip(normalized_img, -20, 20, out=normalized_img)


def _min_max_per_channel(img: ImageType) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(img.ndim - 1))
    if img.shape[-1] == 1:
        return img.min(axis=axes), img.max(axis=axes)

    flat = np.ascontiguousarray(img).reshape(-1, img.shape[-1])
    flat_cv2 = cast("Any", flat)
    dtype = cv2.CV_32F if img.dtype == np.float32 else -1
    img_min = cv2.reduce(flat_cv2, 0, cv2.REDUCE_MIN, dtype=dtype).reshape(-1)
    img_max = cv2.reduce(flat_cv2, 0, cv2.REDUCE_MAX, dtype=dtype).reshape(-1)
    return img_min, img_max


def _normalize_min_max_per_channel_opencv(img: ImageType) -> ImageFloat32:
    """Apply per-channel min-max normalization."""
    eps = 1e-4
    img_min, img_max = _min_max_per_channel(img)

    if img.shape[-1] > MAX_OPENCV_WORKING_CHANNELS:
        img_min = np.full_like(img, img_min)
        img_max = np.full_like(img, img_max)

    # Use NumPy operations for 4D arrays (faster), OpenCV for 3D
    if img.ndim > 3:
        normalized_img = cast("ImageFloat32", (img - img_min) / (img_max - img_min + eps))
    else:
        normalized_img = cast(
            "ImageFloat32",
            cv2.divide(cv2.subtract(img, img_min), (img_max - img_min + eps), dtype=cv2.CV_32F),
        )

    return np.clip(normalized_img, -20, 20, out=normalized_img)


@overload
def normalize_per_image_opencv(img: ImageFloat16, normalization: NormalizationType) -> ImageFloat16: ...


@overload
def normalize_per_image_opencv(img: ImageUInt8 | ImageFloat32, normalization: NormalizationType) -> ImageFloat32: ...


@preserve_channel_dim
def normalize_per_image_opencv(
    img: ImageType,
    normalization: NormalizationType,
) -> ImageFloat16 | ImageFloat32:
    """Normalize with OpenCV and NumPy kernels for :func:`normalize_per_image` modes.

    Mean/std modes clip to [-20, 20]; min/max modes scale to [0, 1].
    Float16 inputs compute in float32 and return float16; uint8 and float32
    inputs return float32. Single-channel per-channel modes use global statistics.

    Raises:
        ValueError: If the normalization mode is unknown.
    """
    if img.dtype == np.float16:
        result = normalize_per_image_opencv(_cast_float_numpy(img, np.float32), normalization)
        return _cast_float_numpy(result, np.float16)

    if img.shape[-1] == 1 and normalization == "image_per_channel":
        normalization = "image"
    if img.shape[-1] == 1 and normalization == "min_max_per_channel":
        normalization = "min_max"

    if normalization == "image":
        mean_scalar, std_scalar = _compute_image_stats_opencv(img)
        return _normalize_mean_std_opencv(img, mean_scalar, std_scalar)

    if normalization == "image_per_channel":
        mean_arr, std_arr = _compute_per_channel_stats_opencv(img)
        return _normalize_mean_std_opencv(img, mean_arr, std_arr)

    if normalization == "min_max":
        dst = np.empty_like(img, dtype=np.float32)
        return cast(
            "ImageFloat32",
            cv2.normalize(img, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F),
        )

    if normalization == "min_max_per_channel":
        return _normalize_min_max_per_channel_opencv(img)

    raise ValueError(f"Unknown normalization method: {normalization}")


@overload
def normalize_per_image_numpy(img: ImageFloat16, normalization: NormalizationType) -> ImageFloat16: ...


@overload
def normalize_per_image_numpy(img: ImageUInt8 | ImageFloat32, normalization: NormalizationType) -> ImageFloat32: ...


@preserve_channel_dim
def normalize_per_image_numpy(
    img: ImageType,
    normalization: NormalizationType,
) -> ImageFloat16 | ImageFloat32:
    """Normalize with NumPy kernels for :func:`normalize_per_image` modes.

    Mean/std modes clip to [-20, 20]; min/max modes scale to [0, 1].
    Computation uses float32 with epsilon 1e-4 in the denominator.
    Float16 input returns float16; uint8 and float32 inputs return float32.

    Raises:
        ValueError: If the normalization mode is unknown.
    """
    img_f = _cast_float_numpy(img, np.float32)
    eps = DEFAULT_EPS

    if normalization == "image":
        # Match 0.0.40: raw ndarray mean/std (router regressed when this used stats.mean_std global float32)
        mean_f = img_f.mean()
        std_f = img_f.std() + eps
        normalized_img = cast("ImageFloat32", (img_f - mean_f) / std_f)
        return _restore_float_image_dtype(img, np.clip(normalized_img, -20, 20, out=normalized_img))

    if normalization == "image_per_channel":
        pixel_mean, pixel_std = mean_std(img_f, "per_channel", eps=eps)
        pm = np.asarray(pixel_mean, dtype=np.float32)
        ps = np.asarray(pixel_std, dtype=np.float32)
        normalized_img = cast("ImageFloat32", (img_f - pm) / ps)
        return _restore_float_image_dtype(img, np.clip(normalized_img, -20, 20, out=normalized_img))

    if normalization == "min_max":
        img_min = img_f.min()
        img_max = img_f.max()
        normalized_img = cast("ImageFloat32", (img_f - img_min) / (img_max - img_min + eps))
        return _restore_float_image_dtype(img, cast("ImageFloat32", np.clip(normalized_img, 0, 1)))

    if normalization == "min_max_per_channel":
        axes = tuple(range(img_f.ndim - 1))  # All axes except channel
        img_min = img_f.min(axis=axes)
        img_max = img_f.max(axis=axes)
        normalized_img = cast("ImageFloat32", (img_f - img_min) / (img_max - img_min + eps))
        return _restore_float_image_dtype(img, cast("ImageFloat32", np.clip(normalized_img, 0, 1)))

    raise ValueError(f"Unknown normalization method: {normalization}")


def _create_mean_std_lut(mean: float, std: float, max_value: float, clip_range: tuple[float, float]) -> np.ndarray:
    """Create a mean-std normalization LUT."""
    lut = (np.arange(0, max_value + 1, dtype=np.float32) - mean) / std
    return lut.clip(*clip_range).astype(np.float32)


def _create_min_max_lut(img_min: float, img_max: float, max_value: float, eps: float) -> np.ndarray:
    """Create a min-max normalization LUT."""
    lut = (np.arange(0, max_value + 1, dtype=np.float32) - img_min) / (img_max - img_min + eps)
    return lut.clip(0, 1).astype(np.float32)


def _normalize_image_lut(img: ImageUInt8, max_value: float, eps: float) -> ImageFloat32:
    """Normalize using global mean and std with LUT."""
    m, s = mean_std(img, "global", eps=eps)
    lut = _create_mean_std_lut(float(m), float(s), max_value, (-20, 20))
    return cast("ImageFloat32", cv2.LUT(img, lut))


def _normalize_image_per_channel_lut(img: ImageUInt8, max_value: float, eps: float, num_channels: int) -> ImageFloat32:
    """Normalize per-channel using mean and std with LUT."""
    pixel_mean, pixel_std = mean_std(img, "per_channel", eps=eps)

    arange_vals = np.arange(0, max_value + 1, dtype=np.float32).reshape(-1, 1, 1)
    mean_lut = np.asarray(pixel_mean, dtype=np.float32).reshape(1, 1, num_channels)
    std_lut = np.asarray(pixel_std, dtype=np.float32).reshape(1, 1, num_channels)
    luts = ((arange_vals - mean_lut) / std_lut).clip(-20, 20).astype(np.float32)

    return _apply_float_lut(img, luts)


def _normalize_min_max_lut(img: ImageUInt8, max_value: float, eps: float) -> ImageFloat32:
    """Normalize using global min-max with LUT."""
    img_min, img_max = img.min(), img.max()
    lut = _create_min_max_lut(img_min, img_max, max_value, eps)
    return cast("ImageFloat32", cv2.LUT(img, lut))


def _normalize_min_max_per_channel_lut(
    img: ImageUInt8,
    max_value: float,
    eps: float,
    num_channels: int,
) -> ImageFloat32:
    """Normalize per-channel using min-max with LUT."""
    axes = tuple(range(img.ndim - 1))
    img_min, img_max = img.min(axis=axes), img.max(axis=axes)

    arange_vals = np.arange(0, max_value + 1, dtype=np.float32).reshape(-1, 1, 1)
    img_min_lut = np.asarray(img_min, dtype=np.float32).reshape(1, 1, num_channels)
    img_max_lut = np.asarray(img_max, dtype=np.float32).reshape(1, 1, num_channels)
    luts = ((arange_vals - img_min_lut) / (img_max_lut - img_min_lut + eps)).clip(0, 1).astype(np.float32)

    return _apply_float_lut(img, luts)


@preserve_channel_dim
def normalize_per_image_lut(
    img: ImageUInt8,
    normalization: NormalizationType,
) -> ImageFloat32:
    """Normalize uint8 inputs through shared or per-channel lookup tables.

    Uses the modes from :func:`normalize_per_image`. Mean/std modes clip to
    [-20, 20]; min/max modes scale to [0, 1]. Returns float32 with the input shape.
    Single-channel per-channel modes use global statistics.

    Raises:
        ValueError: If the normalization mode is unknown.
    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    eps = 1e-4
    num_channels = get_num_channels(img)
    is_single_channel = img.shape[-1] == 1

    if is_single_channel and normalization in ("image_per_channel", "min_max_per_channel"):
        normalization = "image" if normalization == "image_per_channel" else "min_max"

    if normalization == "image":
        return _normalize_image_lut(img, max_value, eps)

    if normalization == "image_per_channel":
        return _normalize_image_per_channel_lut(img, max_value, eps, num_channels)

    if normalization == "min_max":
        return _normalize_min_max_lut(img, max_value, eps)

    if normalization == "min_max_per_channel":
        return _normalize_min_max_per_channel_lut(img, max_value, eps, num_channels)

    raise ValueError(f"Unknown normalization method: {normalization}")


@overload
def normalize_per_image(
    img: ImageFloat16,
    normalization: NormalizationType,
    *,
    layout: TensorLayout | None = None,
) -> ImageFloat16: ...


@overload
def normalize_per_image(
    img: ImageUInt8 | ImageFloat32,
    normalization: NormalizationType,
    *,
    layout: TensorLayout | None = None,
) -> ImageFloat32: ...


@overload
def normalize_per_image(
    img: torch.Tensor,
    normalization: NormalizationType,
    *,
    layout: TensorLayout | None = None,
) -> torch.Tensor: ...


def normalize_per_image(
    img: ImageType | torch.Tensor,
    normalization: NormalizationType,
    *,
    layout: TensorLayout | None = None,
) -> ImageFloat16 | ImageFloat32 | torch.Tensor:
    """Normalize an image using statistics computed from the image itself.

    Unlike ``normalize`` (which takes caller-supplied ``mean``/``denominator``), this function
    estimates the statistics from ``img`` at call time.

    ``normalization`` options:

    - ``"image"``: global mean/std across all pixels and channels; output clipped to [-20, 20].
    - ``"image_per_channel"``: per-channel mean/std; output clipped to [-20, 20].
    - ``"min_max"``: global min-max scale to [0, 1].
    - ``"min_max_per_channel"``: per-channel min-max scale to [0, 1].

    Routing:
    - **uint8, ``"image"`` / ``"image_per_channel"`` / ``"min_max_per_channel"``**: LUT
      (fastest — 256-entry float32 table per channel, applied via ``cv2.LUT``).
    - **uint8, ``"min_max"``**: ``cv2.normalize`` (~3x faster than LUT on router benchmarks).
    - **float16/float32, ``"image"``**: float32 statistics and arithmetic, cast to the input float dtype.
    - **float16/float32, other modes**: float32 OpenCV/NumPy work, cast once to the input float dtype.
    - **ndim > 3 (batch/volume)**: NumPy path.

    Alternative: ``normalize`` for fixed per-channel ImageNet-style constants.

    Args:
        img: uint8, float16, or float32 image, shape ``(H, W, C)`` or ``(X, H, W, C)``.
        normalization: One of ``"image"``, ``"image_per_channel"``, ``"min_max"``,
            ``"min_max_per_channel"``.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Floating image with the same dtype and spatial shape as ``img``. Uint8 input returns float32.
    """
    result: ImageFloat16 | ImageFloat32
    if isinstance(img, torch.Tensor):
        return dispatch_tensor_image(img, layout, normalize_per_image, normalization)

    if img.dtype == np.uint8:
        if normalization == "min_max":
            result = normalize_per_image_opencv(cast("ImageUInt8", img), normalization)
        else:
            result = normalize_per_image_lut(cast("ImageUInt8", img), normalization)
    elif img.dtype == np.float32:
        if normalization == "image":
            result = normalize_per_image_numpy(cast("ImageFloat32", img), normalization)
        else:
            result = normalize_per_image_opencv(cast("ImageFloat32", img), normalization)
    elif img.dtype == np.float16:
        float32_img = _cast_float_numpy(img, np.float32)
        if normalization == "image":
            float32_result = normalize_per_image_numpy(float32_img, normalization)
        else:
            float32_result = normalize_per_image_opencv(float32_img, normalization)
        result = _cast_float_numpy(float32_result, np.float16)
    elif img.ndim > 3:
        result = normalize_per_image_numpy(img, normalization)
    else:
        result = normalize_per_image_opencv(img, normalization)
    return result
