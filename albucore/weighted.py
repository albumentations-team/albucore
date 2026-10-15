"""NumKong ``blend``-based weighted ops (used by arithmetic routers)."""

from typing import cast

import numkong as nk
import numpy as np

from albucore.utils import ImageArrayT, ImageType

# Skip the NumKong half route when runtime dispatch would use its unmeasured scalar fallback.
_NUMKONG_NEON_HALF = bool(nk.get_capabilities().get("neonhalf", False))


def add_weighted_numkong(img1: ImageArrayT, weight1: float, img2: ImageType, weight2: float) -> ImageArrayT:
    """Fused weighted sum for already validated, same-shape and same-dtype inputs."""
    original_shape = img1.shape
    original_dtype = img1.dtype

    a = np.ascontiguousarray(img1).reshape(-1)
    b = np.ascontiguousarray(img2).reshape(-1)
    blended = nk.blend(a, b, alpha=weight1, beta=weight2)
    if blended is None:
        raise RuntimeError("nk.blend returned None")
    return cast("ImageArrayT", np.asarray(blended, dtype=original_dtype).reshape(original_shape))


def add_array_numkong(img: ImageArrayT, value: np.ndarray) -> ImageArrayT:
    return add_weighted_numkong(img, 1, value, 1)


def multiply_by_constant_numkong(img: ImageArrayT, value: float) -> ImageArrayT:
    """Scalar multiply via ``nk.scale`` (alpha*x + 0; no full-size zero buffer vs ``blend``)."""
    return multiply_add_numkong(img, value, 0.0)


def add_constant_numkong(img: ImageArrayT, value: float) -> ImageArrayT:
    """Constant add via ``nk.scale`` (1*x + beta; no full-size ones buffer vs ``blend``)."""
    return multiply_add_numkong(img, 1.0, value)


def multiply_add_numkong(img: ImageArrayT, factor: float, value: float) -> ImageArrayT:
    """Scalar affine transform via ``nk.scale`` (alpha*x + beta)."""
    original_shape = img.shape
    original_dtype = img.dtype
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=float(factor), beta=float(value))
    if out is None:
        raise RuntimeError("nk.scale returned None")
    return cast("ImageArrayT", np.asarray(out, dtype=original_dtype).reshape(original_shape))
