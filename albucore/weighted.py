"""NumKong ``blend``-based weighted ops (used by arithmetic routers)."""

import numkong as nk
import numpy as np

from albucore.utils import ImageType, clip


def add_weighted_numkong(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    """Fused weighted sum via NumKong ``blend`` (alpha*img1 + beta*img2), same as OpenCV addWeighted."""
    original_shape = img1.shape
    original_dtype = img1.dtype

    if img2.dtype != original_dtype:
        img2 = clip(img2.astype(original_dtype, copy=False), original_dtype, inplace=True)

    a = np.ascontiguousarray(img1).reshape(-1)
    b = np.ascontiguousarray(img2).reshape(-1)
    blended = nk.blend(a, b, alpha=weight1, beta=weight2)
    return np.frombuffer(blended, dtype=original_dtype).reshape(original_shape)


def add_array_numkong(img: ImageType, value: np.ndarray) -> ImageType:
    return add_weighted_numkong(img, 1, value, 1)


def multiply_by_constant_numkong(img: ImageType, value: float) -> ImageType:
    """Scalar multiply via ``nk.scale`` (alpha*x + 0; no full-size zero buffer vs ``blend``)."""
    original_shape = img.shape
    original_dtype = img.dtype
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=float(value), beta=0.0)
    return np.frombuffer(out, dtype=original_dtype).reshape(original_shape)


def add_constant_numkong(img: ImageType, value: float) -> ImageType:
    """Constant add via ``nk.scale`` (1*x + beta; no full-size ones buffer vs ``blend``)."""
    original_shape = img.shape
    original_dtype = img.dtype
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=1.0, beta=float(value))
    return np.frombuffer(out, dtype=original_dtype).reshape(original_shape)
