"""Histogram equalization for channel-last uint8 images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from albucore.lut import apply_uint8_lut
from albucore.utils import NUM_RGB_CHANNELS, get_opencv_max_channels

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = ["equalize_uint8", "match_histograms_uint8"]

_OPENCV_MAX_CHANNELS = get_opencv_max_channels()
_MAX_EXACT_HISTOGRAM_PIXELS = 2**24
_MASKED_CV_MIN_PIXELS = 4_096


def _histogram(img: NDArray[np.uint8], mask: NDArray[np.uint8] | None = None) -> NDArray[np.int64]:
    if img.size <= _MAX_EXACT_HISTOGRAM_PIXELS:
        return np.asarray(cv2.calcHist([img], [0], mask, [256], (0, 256)), dtype=np.int64).ravel()

    # Sum integer histograms from tiles whose counts fit exactly in float32.
    rows_per_tile = max(1, _MAX_EXACT_HISTOGRAM_PIXELS // img.shape[1])
    cols_per_tile = min(img.shape[1], _MAX_EXACT_HISTOGRAM_PIXELS)
    counts = np.zeros(256, dtype=np.int64)
    for row in range(0, img.shape[0], rows_per_tile):
        for col in range(0, img.shape[1], cols_per_tile):
            tile = np.s_[row : row + rows_per_tile, col : col + cols_per_tile]
            counts += _histogram(img[tile], None if mask is None else mask[tile])
    return counts


def _pillow_lut(histogram: NDArray[np.int64]) -> NDArray[np.uint8] | None:
    nonzero = np.flatnonzero(histogram)
    if nonzero.size <= 1:
        return None
    # The final occupied bin does not contribute to Pillow's step.
    step = int(histogram.sum() - histogram[nonzero[-1]]) // 255
    if step == 0:
        return None

    before_bin = np.cumsum(histogram) - histogram
    return ((before_bin + step // 2) // step).clip(0, 255).astype(np.uint8)


def _opencv_lut(histogram: NDArray[np.int64]) -> NDArray[np.uint8] | None:
    nonzero = np.flatnonzero(histogram)
    if nonzero.size <= 1:
        return None

    first = nonzero[0]
    scale = np.float32(255) / np.float32(histogram.sum() - histogram[first])
    counts = np.cumsum(histogram) - histogram[first]
    scaled = counts.astype(np.float32) * scale
    return np.asarray(np.clip(scaled.round(), 0, 255), dtype=np.uint8)


def _equalize_cv_unmasked(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    channels = img.shape[-1]
    if channels == 1:
        return np.asarray(cv2.equalizeHist(img[..., 0]), dtype=np.uint8)[..., None]
    if channels <= _OPENCV_MAX_CHANNELS:
        planes = [cv2.equalizeHist(plane) for plane in cv2.split(img)]
        return np.asarray(cv2.merge(planes), dtype=np.uint8)

    result = np.empty_like(img)
    for channel in range(channels):
        result[..., channel] = cv2.equalizeHist(img[..., channel])
    return result


def _equalize_pillow_rgb(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    result = np.empty_like(img, order="C")
    plane = np.empty(img.shape[:2], dtype=np.uint8)
    for channel in range(NUM_RGB_CHANNELS):
        cv2.extractChannel(img, channel, dst=plane)
        lut = _pillow_lut(_histogram(plane))
        if lut is not None:
            apply_uint8_lut(plane, lut, out=plane)
        cv2.insertChannel(plane, result, channel)
    return result


def _equalize_cv_masked_rgb(img: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask
    identity = np.arange(256, dtype=np.uint8)
    luts = np.empty((256, 1, NUM_RGB_CHANNELS), dtype=np.uint8)
    for channel in range(NUM_RGB_CHANNELS):
        channel_mask = mask[..., channel] if mask.ndim == 3 else mask
        histogram = np.asarray(
            cv2.calcHist([img], [channel], channel_mask, [256], (0, 256)),
            dtype=np.int64,
        ).ravel()
        lut = _opencv_lut(histogram)
        luts[:, 0, channel] = identity if lut is None else lut
    return apply_uint8_lut(img, luts)


def equalize_uint8(
    img: NDArray[np.uint8],
    *,
    mode: Literal["cv", "pil"] = "cv",
    mask: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Equalize each channel of a prevalidated HWC uint8 image independently.

    Args:
        img: Input image with an explicit channel axis.
        mode: OpenCV or Pillow histogram mapping.
        mask: Optional uint8 histogram selection of shape HW, HWC with one
            shared channel, or HWC with one mask per image channel. Nonzero
            values select pixels for the histogram. The mapping applies to
            every pixel in that image channel.

    Returns:
        An independent writable uint8 array with the input shape.

    Notes:
        A mask with no selected pixels leaves the channel values unchanged.
        Pillow mode reuses a single channel buffer for packed RGB images and
        row-strided RGB crops.

    Examples:
        >>> image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        >>> equalize_uint8(image, mode="cv").shape
        (16, 16, 1)
    """
    if not img.size:
        return img.copy()
    if mode == "cv" and mask is None:
        return _equalize_cv_unmasked(img)

    channels = img.shape[-1]
    if (
        mode == "cv"
        and mask is not None
        and channels == NUM_RGB_CHANNELS
        and _MASKED_CV_MIN_PIXELS <= img.shape[0] * img.shape[1] <= _MAX_EXACT_HISTOGRAM_PIXELS
        and img.strides[1:] == (NUM_RGB_CHANNELS, 1)
        and img.strides[0] >= img.shape[1] * NUM_RGB_CHANNELS
        and all(stride >= 0 for stride in img.strides)
    ):
        return _equalize_cv_masked_rgb(img, mask)
    if (
        mode == "pil"
        and mask is None
        and channels == NUM_RGB_CHANNELS
        and img.strides[1:] == (NUM_RGB_CHANNELS, 1)
        and img.strides[0] >= img.shape[1] * NUM_RGB_CHANNELS
    ):
        return _equalize_pillow_rgb(img)

    lut_for = _pillow_lut if mode == "pil" else _opencv_lut
    if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    result = np.empty_like(img)
    for channel in range(channels):
        plane = img[..., channel]
        selection = mask[..., channel] if mask is not None and mask.ndim == 3 else mask
        lut = lut_for(_histogram(plane, selection))
        result[..., channel] = plane if lut is None else apply_uint8_lut(plane, lut)
    return result


def _channel_histograms(img: NDArray[np.uint8]) -> list[NDArray[np.int64]]:
    channels = img.shape[-1]
    pixels = img if img.ndim == 3 else img.reshape(1, -1, channels)
    if pixels.size // channels > _MAX_EXACT_HISTOGRAM_PIXELS:
        return [_histogram(pixels[..., channel]) for channel in range(channels)]
    # One OpenCV call per channel reads the packed or row-strided image in place; slicing a plane first copies it.
    return [
        np.asarray(cv2.calcHist([pixels], [channel], None, [256], (0, 256)), dtype=np.int64).ravel()
        for channel in range(channels)
    ]


def _matched_lut(source_counts: NDArray[np.int64], reference_counts: NDArray[np.int64]) -> NDArray[np.uint8]:
    # Levels absent from the reference carry no mass; skip them as the CDF-matching rule requires.
    reference_levels = np.flatnonzero(reference_counts)
    source_quantiles = np.cumsum(source_counts) / source_counts.sum()
    reference_quantiles = np.cumsum(reference_counts[reference_levels]) / reference_counts.sum()
    return np.asarray(np.interp(source_quantiles, reference_quantiles, reference_levels), dtype=np.uint8)


def match_histograms_uint8(img: NDArray[np.uint8], reference: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Match the histogram of each channel of ``img`` to the same channel of ``reference``.

    Each source level maps to the reference level at the same cumulative quantile, as in
    ``skimage.exposure.match_histograms``. Histograms are exact, and one ``(256, 1, C)`` table
    is applied with :func:`albucore.lut.apply_uint8_lut`.

    Args:
        img: Channel-last uint8 array with shape ``(..., C)``.
        reference: Channel-last uint8 array with the same ``C``. Its spatial size may differ.

    Returns:
        An independent writable uint8 array with the shape of ``img``. Neither input is written.
        An empty ``img`` returns an empty copy.

    Examples:
        >>> image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        >>> reference = np.full((4, 4, 1), 7, dtype=np.uint8)
        >>> int(match_histograms_uint8(image, reference).max())
        7
    """
    if not img.size:
        return img.copy()
    luts = np.empty((256, 1, img.shape[-1]), dtype=np.uint8)
    for channel, (source_counts, reference_counts) in enumerate(
        zip(_channel_histograms(img), _channel_histograms(reference), strict=True),
    ):
        luts[:, 0, channel] = _matched_lut(source_counts, reference_counts)
    return apply_uint8_lut(img, luts)
