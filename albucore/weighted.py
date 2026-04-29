"""NumKong ``blend``-based weighted ops (used by arithmetic routers)."""

from typing import cast

import numkong as nk
import numpy as np

from albucore.utils import ImageType, clip


def add_weighted_numkong(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    """Fused weighted sum via NumKong ``blend`` (alpha*img1 + beta*img2), same as OpenCV addWeighted."""
    original_shape = img1.shape
    original_dtype = img1.dtype

    if img2.dtype != original_dtype:
        img2 = clip(cast("ImageType", img2.astype(original_dtype, copy=False)), original_dtype, inplace=True)

    a = np.ascontiguousarray(img1).reshape(-1)
    b = np.ascontiguousarray(img2).reshape(-1)
    blended = nk.blend(a, b, alpha=weight1, beta=weight2)
    if blended is None:
        raise RuntimeError("nk.blend returned None")
    return cast("ImageType", np.asarray(blended, dtype=original_dtype).reshape(original_shape))


def add_array_numkong(img: ImageType, value: np.ndarray) -> ImageType:
    return add_weighted_numkong(img, 1, value, 1)


def multiply_by_constant_numkong(img: ImageType, value: float) -> ImageType:
    """Scalar multiply via ``nk.scale`` (alpha*x + 0; no full-size zero buffer vs ``blend``)."""
    original_shape = img.shape
    original_dtype = img.dtype
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=float(value), beta=0.0)
    if out is None:
        raise RuntimeError("nk.scale returned None")
    return cast("ImageType", np.asarray(out, dtype=original_dtype).reshape(original_shape))


def add_constant_numkong(img: ImageType, value: float) -> ImageType:
    """Constant add via ``nk.scale`` (1*x + beta; no full-size ones buffer vs ``blend``)."""
    original_shape = img.shape
    original_dtype = img.dtype
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=1.0, beta=float(value))
    if out is None:
        raise RuntimeError("nk.scale returned None")
    return cast("ImageType", np.asarray(out, dtype=original_dtype).reshape(original_shape))


def multiply_add_numkong(img: ImageType, factor: float, value: float) -> ImageType:
    """Scalar affine transform via ``nk.scale`` (alpha*x + beta)."""
    original_shape = img.shape
    original_dtype = img.dtype
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=float(factor), beta=float(value))
    if out is None:
        raise RuntimeError("nk.scale returned None")
    return cast("ImageType", np.asarray(out, dtype=original_dtype).reshape(original_shape))
