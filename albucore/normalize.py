"""Per-image normalization (mean/std, min/max) and helpers."""

from typing import Any, cast

import cv2
import numpy as np

from albucore.decorators import preserve_channel_dim
from albucore.lut import _apply_float_lut
from albucore.stats import DEFAULT_EPS, mean_std
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ImageFloat32,
    ImageType,
    ImageUInt8,
    NormalizationType,
    get_num_channels,
)


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
    img_f = img.astype(np.float32, copy=False)
    if img_f.ndim > 3 or (img_f.ndim == 3 and img_f.shape[-1] > MAX_OPENCV_WORKING_CHANNELS):
        # Use NumPy operations for 4D/5D and 3D images with >4 channels.
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        normalized_img = cast("ImageFloat32", (img_f - mean_arr) / std_arr)
    else:
        # Use OpenCV for 3D images up to 4 channels.
        mean_cv2: Any = mean if np.isscalar(mean) else np.asarray(mean, dtype=np.float32)
        std_cv2: Any = std if np.isscalar(std) else np.asarray(std, dtype=np.float32)
        normalized_img = cast(
            "ImageFloat32",
            cv2.divide(cv2.subtract(img_f, mean_cv2, dtype=cv2.CV_32F), std_cv2, dtype=cv2.CV_32F),
        )
    return np.clip(normalized_img, -20, 20, out=normalized_img)


def _normalize_min_max_per_channel_opencv(img: ImageType) -> ImageFloat32:
    """Apply per-channel min-max normalization."""
    eps = 1e-4
    axes = tuple(range(img.ndim - 1))  # All axes except channel

    img_min = img.min(axis=axes)
    img_max = img.max(axis=axes)

    if img.shape[-1] > MAX_OPENCV_WORKING_CHANNELS:
        img_min = np.full_like(img, img_min)
        img_max = np.full_like(img, img_max)

    # Use NumPy operations for 4D/5D (faster), OpenCV for 3D
    if img.ndim > 3:
        normalized_img = cast("ImageFloat32", (img - img_min) / (img_max - img_min + eps))
    else:
        normalized_img = cast(
            "ImageFloat32",
            cv2.divide(cv2.subtract(img, img_min), (img_max - img_min + eps), dtype=cv2.CV_32F),
        )

    return np.clip(normalized_img, -20, 20, out=normalized_img)


@preserve_channel_dim
def normalize_per_image_opencv(
    img: ImageType,
    normalization: NormalizationType,
) -> ImageFloat32:
    """Normalize an image using OpenCV operations based on the specified normalization type.

    This function normalizes an image using various strategies, optimized with OpenCV operations
    for better performance on standard image types.

    Args:
        img: Input image as a numpy array with shape (H, W, C).
        normalization: Type of normalization to apply. Options are:
            - "image": Normalize using global mean and std across all pixels
            - "image_per_channel": Normalize each channel separately using its own mean and std
            - "min_max": Scale to [0, 1] using global min and max values
            - "min_max_per_channel": Scale each channel to [0, 1] using per-channel min and max

    Returns:
        Normalized image as float32 array with values clipped to [-20, 20] range to prevent
        extreme values that could cause training instability.

    Raises:
        ValueError: If an unknown normalization method is specified.

    Notes:
        - The function automatically converts input to float32
        - Adds epsilon (1e-4) to std deviation to prevent division by zero
        - For images with >4 channels, falls back to array operations as OpenCV has limitations
        - Single channel images treated as "image" normalization when "image_per_channel" is specified
    """
    # Handle single-channel edge case
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


@preserve_channel_dim
def normalize_per_image_numpy(
    img: ImageType,
    normalization: NormalizationType,
) -> ImageFloat32:
    """Normalize an image using pure NumPy operations based on the specified normalization type.

    This function provides the same normalization strategies as normalize_per_image_opencv but uses
    pure NumPy operations. This can be useful for compatibility or when OpenCV is not available.

    Args:
        img: Input image as a numpy array with shape (H, W, C).
        normalization: Type of normalization to apply. Options are:
            - "image": Normalize using global mean and std across all pixels
            - "image_per_channel": Normalize each channel separately using its own mean and std
            - "min_max": Scale to [0, 1] using global min and max values
            - "min_max_per_channel": Scale each channel to [0, 1] using per-channel min and max

    Returns:
        Normalized image as float32 array with values clipped to [-20, 20] range to prevent
        extreme values that could cause training instability.

    Raises:
        ValueError: If an unknown normalization method is specified.

    Notes:
        - The function automatically converts input to float32
        - Adds epsilon (1e-4) to std deviation to prevent division by zero
        - Uses in-place operations where possible for memory efficiency
        - Generally slower than the OpenCV version but more portable
    """
    img_f = cast("ImageFloat32", img.astype(np.float32, copy=False))
    eps = DEFAULT_EPS

    if normalization == "image":
        # Match 0.0.40: raw ndarray mean/std (router regressed when this used stats.mean_std global float32)
        mean_f = img_f.mean()
        std_f = img_f.std() + eps
        normalized_img = cast("ImageFloat32", (img_f - mean_f) / std_f)
        return np.clip(normalized_img, -20, 20, out=normalized_img)

    if normalization == "image_per_channel":
        pixel_mean, pixel_std = mean_std(img_f, "per_channel", eps=eps)
        pm = np.asarray(pixel_mean, dtype=np.float32)
        ps = np.asarray(pixel_std, dtype=np.float32)
        normalized_img = cast("ImageFloat32", (img_f - pm) / ps)
        return np.clip(normalized_img, -20, 20, out=normalized_img)

    if normalization == "min_max":
        img_min = img_f.min()
        img_max = img_f.max()
        normalized_img = cast("ImageFloat32", (img_f - img_min) / (img_max - img_min + eps))
        return cast("ImageFloat32", np.clip(normalized_img, 0, 1))

    if normalization == "min_max_per_channel":
        axes = tuple(range(img_f.ndim - 1))  # All axes except channel
        img_min = img_f.min(axis=axes)
        img_max = img_f.max(axis=axes)
        normalized_img = cast("ImageFloat32", (img_f - img_min) / (img_max - img_min + eps))
        return cast("ImageFloat32", np.clip(normalized_img, 0, 1))

    raise ValueError(f"Unknown normalization method: {normalization}")


def _create_mean_std_lut(mean: float, std: float, max_value: float, clip_range: tuple[float, float]) -> np.ndarray:
    """Create a mean-std normalization LUT."""
    lut = (np.arange(0, max_value + 1, dtype=np.float32) - mean) / std
    return lut.clip(*clip_range).astype(np.float32)


def _create_min_max_lut(img_min: float, img_max: float, max_value: float, eps: float) -> np.ndarray:
    """Create a min-max normalization LUT."""
    lut = (np.arange(0, max_value + 1, dtype=np.float32) - img_min) / (img_max - img_min + eps)
    return lut.clip(0, 1).astype(np.float32)


def _apply_per_channel_lut(img: ImageUInt8, luts: np.ndarray, num_channels: int) -> ImageFloat32:
    """Apply per-channel LUTs to an image."""
    if luts.shape != (256, num_channels):
        msg = f"Expected per-channel LUTs shaped (256, {num_channels}), got {luts.shape}"
        raise ValueError(msg)
    return _apply_float_lut(img, luts)


def _normalize_image_lut(img: ImageUInt8, max_value: float, eps: float) -> ImageFloat32:
    """Normalize using global mean and std with LUT."""
    m, s = mean_std(img, "global", eps=eps)
    lut = _create_mean_std_lut(float(m), float(s), max_value, (-20, 20))
    return cast("ImageFloat32", cv2.LUT(img, lut))


def _normalize_image_per_channel_lut(img: ImageUInt8, max_value: float, eps: float, num_channels: int) -> ImageFloat32:
    """Normalize per-channel using mean and std with LUT."""
    pixel_mean, pixel_std = mean_std(img, "per_channel", eps=eps)

    arange_vals = np.arange(0, max_value + 1, dtype=np.float32)
    luts = ((arange_vals[:, np.newaxis] - pixel_mean) / pixel_std).clip(-20, 20).astype(np.float32)

    return _apply_per_channel_lut(img, luts, num_channels)


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

    arange_vals = np.arange(0, max_value + 1, dtype=np.float32)
    luts = ((arange_vals[:, np.newaxis] - img_min) / (img_max - img_min + eps)).clip(0, 1).astype(np.float32)

    return _apply_per_channel_lut(img, luts, num_channels)


@preserve_channel_dim
def normalize_per_image_lut(
    img: ImageUInt8,
    normalization: NormalizationType,
) -> ImageFloat32:
    """Normalize an image using lookup tables (LUT) for optimized performance on uint8 images.

    This function implements the same normalization strategies but uses pre-computed lookup tables
    for extremely fast normalization of uint8 images. This is the fastest method for uint8 data.

    Args:
        img: Input image as a numpy array with uint8 dtype and shape (H, W, C).
        normalization: Type of normalization to apply. Options are:
            - "image": Normalize using global mean and std across all pixels
            - "image_per_channel": Normalize each channel separately using its own mean and std
            - "min_max": Scale to [0, 1] using global min and max values
            - "min_max_per_channel": Scale each channel to [0, 1] using per-channel min and max

    Returns:
        Normalized image as float32 array with values clipped to [-20, 20] range to prevent
        extreme values that could cause training instability.

    Raises:
        ValueError: If an unknown normalization method is specified.

    Notes:
        - Designed specifically for uint8 images for maximum performance
        - Creates a 256-element lookup table mapping each possible uint8 value to its normalized value
        - Uses OpenCV's LUT function for fast application of the transformation
        - For per-channel normalization, creates separate LUTs for each channel
        - Single channel images treated as "image" normalization when "image_per_channel" is specified
    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    eps = 1e-4
    num_channels = get_num_channels(img)
    is_single_channel = img.shape[-1] == 1

    # Handle single-channel edge cases
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


def normalize_per_image(img: ImageType, normalization: NormalizationType) -> ImageFloat32:
    """Normalize an image using statistics computed from the image itself → float32.

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
    - **float32, ``"image"``**: raw ``img.mean()`` / ``img.std()`` NumPy pass (matches 0.0.40).
    - **float32, others**: ``cv2`` subtract + divide.
    - **ndim > 3 (batch/volume)**: NumPy path.

    Alternative: ``normalize`` for fixed per-channel ImageNet-style constants.

    Args:
        img: uint8 or float32 image, shape ``(H, W, C)``, ``(N, H, W, C)``, or ``(N, D, H, W, C)``.
        normalization: One of ``"image"``, ``"image_per_channel"``, ``"min_max"``,
            ``"min_max_per_channel"``.

    Returns:
        float32 image, same spatial shape as ``img``.
    """
    # Route uint8 images
    if img.dtype == np.uint8:
        # Use LUT for everything except min_max (where OpenCV is 3x faster)
        if normalization == "min_max":
            return normalize_per_image_opencv(cast("ImageUInt8", img), normalization)
        # LUT is fastest for "image", "image_per_channel", and "min_max_per_channel"
        return normalize_per_image_lut(cast("ImageUInt8", img), normalization)

    # Route float32 images
    if img.dtype == np.float32:
        if normalization == "image":
            # ``normalize_per_image_numpy`` uses raw ``img.mean``/``img.std`` (matches 0.0.40)
            return normalize_per_image_numpy(cast("ImageFloat32", img), normalization)
        return normalize_per_image_opencv(cast("ImageFloat32", img), normalization)

    # Default fallback: OpenCV for single images, NumPy for videos/volumes
    if img.ndim > 3:
        return normalize_per_image_numpy(img, normalization)
    return normalize_per_image_opencv(img, normalization)
