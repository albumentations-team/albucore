"""Uint8 lookup tables: StringZilla ``translate`` and OpenCV ``LUT`` with size-based routing."""

from __future__ import annotations

import cv2
import numpy as np
import stringzilla as sz

from albucore.decorators import contiguous, preserve_channel_dim
from albucore.utils import ImageUInt8  # noqa: TC001


@preserve_channel_dim
def _cv2_lut_uint8(img: ImageUInt8, lut: ImageUInt8) -> ImageUInt8:
    """``cv2.LUT`` on single-channel HWC returns ``(H, W)``; restore ``(H, W, 1)`` like other OpenCV paths."""
    return cv2.LUT(img, lut)


@contiguous
def sz_lut(img: ImageUInt8, lut: ImageUInt8, inplace: bool = True) -> ImageUInt8:
    """Apply a shared 256-entry uint8→uint8 LUT using StringZilla ``translate``.

    StringZilla operates on raw bytes without channel awareness, making it ideal for shared
    (same table for every channel) transformations.  Faster than ``cv2.LUT`` for small images
    and single-channel; ``apply_uint8_lut`` routes to ``cv2.LUT`` for large multi-channel images.

    Alternative: ``apply_uint8_lut`` (auto-routes between this and OpenCV based on shape/size);
    ``cv2.LUT`` directly for per-channel (256, 1, C) tables.

    Args:
        img: uint8 image, any shape — must be C-contiguous (enforced by ``@contiguous``).
        lut: 1-D uint8 array of length 256.
        inplace: If True, mutates ``img`` in-place and returns it.
                 If False, operates on a copy.

    Returns:
        uint8 image with each pixel value replaced by ``lut[pixel]``.
    """
    if inplace:
        sz.translate(memoryview(img), memoryview(lut), inplace=True)
        return img

    # sz.translate(inplace=False) allocates + writes in one pass — faster than copy + inplace.
    raw = sz.translate(memoryview(img), memoryview(lut), inplace=False)
    return np.frombuffer(raw, dtype=np.uint8).reshape(img.shape)


def opencv_shared_uint8_lut_faster_hwc(shape: tuple[int, ...]) -> bool:
    """Return True if OpenCV ``LUT`` is expected to beat StringZilla on shared ``(256,)`` HWC LUTs.

    Only ``ndim == 3`` ``(H, W, C)`` is considered; other layouts are handled elsewhere.

    Routing uses ``n = np.prod(shape)`` (total elements ``H*W*C``) and ``hw = n // C``.
    **Rule:** ``C >= 2`` and (``hw >= 409600`` i.e. about ``640*640``, or
    (``hw >= 262_144`` i.e. ``512*512`` and ``n >= 1_310_000``)).
    Single-channel stays on StringZilla.

    Tuned against ``benchmarks/benchmark_lut_shared_routing.py``. Re-benchmark after OpenCV/SZ upgrades.
    """
    if len(shape) != 3:
        return False
    c = int(shape[-1])
    if c < 2:
        return False
    n = int(np.prod(shape, dtype=np.int64))
    hw = n // c
    if hw >= 409600:  # ~640x640
        return True
    return hw >= 262_144 and n >= 1_310_000


def _apply_shared_uint8_lut(img: ImageUInt8, lut: ImageUInt8, inplace: bool) -> ImageUInt8:
    """One ``(256,)`` table for every byte; routes by layout and buffer size.

    HWC contiguous: OpenCV for large multi-channel (``opencv_shared_uint8_lut_faster_hwc``),
    StringZilla for the rest.
    Non-HWC (DHWC, NDHWC): ``sz_lut(inplace=False)`` (which uses ``sz.translate(inplace=False)``,
    a single allocate+write pass) is faster than inplace for large buffers ≳500 KB.
    Source: ``benchmarks/benchmark_scale_vs_lut.py``.
    """
    if img.ndim == 3 and img.flags["C_CONTIGUOUS"] and opencv_shared_uint8_lut_faster_hwc(img.shape):
        out = _cv2_lut_uint8(img, lut)
        if inplace:
            np.copyto(img, out)
            return img
        return out
    # Non-HWC large buffers: sz_lut(inplace=False) uses sz.translate(inplace=False) — one allocate+write
    # pass vs copy+inplace. Wins at ≳2 MB; 1.5 MB shapes are anomalously faster with inplace (cache effect).
    # Threshold from benchmarks/benchmark_scale_vs_lut.py (arm64); re-run after sz upgrades.
    if img.ndim != 3 and img.nbytes >= 2_000_000:
        return sz_lut(img, lut, inplace=False)
    return sz_lut(img, lut, inplace)


def _apply_per_channel_uint8_luts(img: ImageUInt8, luts: ImageUInt8, inplace: bool) -> ImageUInt8:
    """``luts`` shaped ``(C, 256)`` uint8. Vector ``apply_lut`` ignores ``inplace`` (always new array)."""
    del inplace  # API symmetry with ``apply_uint8_lut``; matches historical ``apply_lut`` behaviour.
    num_channels = img.shape[-1]
    if luts.shape != (num_channels, 256):
        msg = f"Expected luts shaped (C, 256) with C={num_channels}, got {luts.shape}"
        raise ValueError(msg)

    if num_channels == 1:
        return _apply_shared_uint8_lut(img, luts[0], inplace=False)

    if img.ndim == 3 and img.flags["C_CONTIGUOUS"]:
        lut_cv2 = np.empty((256, 1, num_channels), dtype=np.uint8)
        lut_cv2[:, 0, :] = luts.T
        return cv2.LUT(img, lut_cv2)

    result = np.empty_like(img, dtype=img.dtype)
    for i in range(num_channels):
        result[..., i] = sz_lut(img[..., i], luts[i], inplace=False)
    return result


def apply_uint8_lut(
    img: ImageUInt8,
    lut: np.ndarray,
    *,
    inplace: bool = False,
) -> ImageUInt8:
    """Apply a uint8→uint8 LUT to an image.  Supports shared and per-channel tables.

    ``lut`` shapes:

    - ``(256,)`` — one shared table applied to every byte regardless of channel.
      Routes to ``cv2.LUT`` or ``sz_lut`` (StringZilla) based on image size and channel count
      (see ``opencv_shared_uint8_lut_faster_hwc``).  ``inplace=True`` mutates ``img`` when
      StringZilla wins, or copies the OpenCV result back when that path wins.

    - ``(C, 256)`` — one row per channel.  On contiguous ``(H, W, C)`` images uses a single
      ``cv2.LUT`` call with a ``(256, 1, C)`` table; volumes and batches fall back to
      per-channel StringZilla.  ``inplace`` is ignored for the per-channel path.

    Degenerate shapes ``(1, 256)`` and ``(256, 1)`` are reshaped to ``(256,)`` automatically.

    Alternative: ``sz_lut`` for the StringZilla path directly; ``cv2.LUT`` with a
    ``(256, 1, C)``-shaped table for explicit per-channel OpenCV dispatch.

    Args:
        img: uint8 image, shape ``(H, W, C)`` or batch/volume.
        lut: uint8 array, shape ``(256,)`` or ``(C, 256)``.
        inplace: Reuse ``img`` buffer when the shared-LUT StringZilla path is chosen.

    Returns:
        uint8 image with pixel values remapped through the LUT.

    Raises:
        TypeError: If ``img`` or ``lut`` is not uint8.
        ValueError: If ``lut`` has an unsupported shape.
    """
    if img.dtype != np.uint8 or lut.dtype != np.uint8:
        msg = "apply_uint8_lut expects uint8 image and uint8 LUT"
        raise TypeError(msg)
    # ``create_lut_array`` + NumPy 2 broadcast: scalar ``value`` yields ``(1, 256)``, not ``(256,)``.
    if lut.ndim == 2 and lut.shape == (1, 256):
        lut = lut.reshape(256)
    if lut.ndim == 2 and lut.shape == (256, 1):
        lut = lut.reshape(256)
    if lut.ndim == 1:
        if lut.shape != (256,):
            msg = f"1D LUT must have length 256, got {lut.shape}"
            raise ValueError(msg)
        return _apply_shared_uint8_lut(img, lut, inplace)
    if lut.ndim == 2:
        return _apply_per_channel_uint8_luts(img, lut, inplace)
    msg = f"LUT must be (256,) or (C, 256), got shape {lut.shape}"
    raise ValueError(msg)
