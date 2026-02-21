"""Geometric operations with multi-channel support.

Drop-in for cv2.medianBlur, cv2.warpAffine, cv2.warpPerspective,
cv2.copyMakeBorder, cv2.remap. Chunking when OpenCV limits apply.
blur, GaussianBlur, resize, filter2D work out of the box for >4ch — use cv2 directly.
Run: python tools/verify_opencv_channel_limits.py
"""
# ruff: noqa: N803, PLR0911, PLR0913  # cv2 API uses M, borderMode, borderValue; chunked fns need many args

import cv2
import numpy as np

from albucore.decorators import preserve_channel_dim
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageType, get_num_channels, maybe_process_in_chunks

# Interpolations that require chunking for >4ch (CI: _src.channels() <= 4)
_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT}


def _border_value_for_cv2(
    value: float | tuple[float, ...] | np.ndarray,
    num_channels: int,
) -> float | tuple[float, ...] | None:
    """Convert border/value to cv2-compatible format (max 4 elements).

    OpenCV's warpAffine, warpPerspective, and copyMakeBorder accept at most 4 values
    for borderValue (one per channel, BGR + alpha). This helper normalizes user input:

    - Scalar or int: broadcast to (v, v, v, v)
    - len <= 4: pass through (possibly as scalar if len 1)
    - len > 4 uniform (all same): (v,)*4
    - len > 4 per-channel (different values): return None — caller must use chunked path

    Returns:
        cv2-compatible value, or None if per-channel len>4 (needs chunking).
    """
    if isinstance(value, (int, float)):
        return (value,) * 4
    if isinstance(value, np.ndarray):
        val = value.flatten()
        if len(val) <= 4:
            return tuple(val.tolist()) if len(val) > 1 else float(val[0])
        if len(val) > 4 and np.all(val == val[0]):
            return (float(val[0]),) * 4
        return None  # per-channel len>4, needs chunking
    if isinstance(value, (tuple, list)):
        if len(value) <= 4:
            return tuple(value) if len(value) > 1 else value[0]
        if len(value) > 4 and all(v == value[0] for v in value):
            return (value[0],) * 4
        return None
    return (value,) * 4


__all__ = [
    "copy_make_border",
    "median_blur",
    "remap",
    "warp_affine",
    "warp_perspective",
]


@preserve_channel_dim
def median_blur(img: ImageType, ksize: int, inplace: bool = False) -> ImageType:
    """Median blur. Works for any number of channels.

    Drop-in for cv2.medianBlur. OpenCV natively supports up to 4 channels; some CI
    builds enforce this limit. For >4 channels we chunk into groups of 4 (or 1 when
    remainder is 2, since cv2.medianBlur fails on 2-channel images), process each,
    and reassemble into a pre-allocated output array.

    Args:
        img: Image with shape (H, W, C). Supported dtypes: uint8, float32.
        ksize: Kernel size (must be odd, >= 3). Same as cv2.medianBlur.
        inplace: If True and C <= 4, write result into img via dst=img. Ignored for >4ch.

    Returns:
        Median-filtered image, same shape and dtype as input.

    Notes:
        - For C > 4: uses maybe_process_in_chunks (pre-alloc + copy per chunk).
        - Verify limits: python tools/verify_opencv_channel_limits.py
    """
    num_channels = get_num_channels(img)
    if num_channels <= MAX_OPENCV_WORKING_CHANNELS:
        if inplace:
            cv2.medianBlur(img, ksize, dst=img)
            return img
        return cv2.medianBlur(img, ksize)
    return maybe_process_in_chunks(cv2.medianBlur, ksize=ksize)(img)


def _warp_affine_chunked(
    img: ImageType,
    M: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    borderMode: int,
    borderValue: tuple[float, ...] | np.ndarray | None,
) -> ImageType:
    """Chunk warpAffine when per-channel borderValue has len > 4.

    OpenCV warpAffine accepts at most 4 border values. When the user passes
    per-channel values (e.g. (r,g,b,a,b2,b3,...) for 8ch), we must process in
    chunks of 4 and apply the correct slice to each chunk.

    Chunking strategy: split channels into groups of 4. For remainder 2, process
    as 2 single-channel chunks (cv2 often fails on 2ch). Pre-allocate output array
    after first chunk, then copy each chunk into its slice. Avoids np.concatenate.

    Args:
        img: (H, W, C) image.
        M: 2x3 affine matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags.
        borderMode: cv2.BORDER_* constant.
        borderValue: Per-channel values, len == C. Flattened to 1D.

    Returns:
        Warped image (dsize[1], dsize[0], C).
    """
    num_channels = img.shape[-1]
    result: np.ndarray | None = None
    offset = 0
    val = np.array(borderValue, dtype=np.float64).flatten() if borderValue is not None else None
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                ch = img[:, :, i + j : i + j + 1]
                bv = (float(val[i + j]),) * 4 if val is not None else 0
                out = cv2.warpAffine(ch, M, dsize, flags=flags, borderMode=borderMode, borderValue=bv)
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            ch = img[:, :, i : min(i + 4, num_channels)]
            bv = tuple(val[i : i + 4].tolist()) if val is not None and len(val) >= i + 4 else 0
            out = cv2.warpAffine(ch, M, dsize, flags=flags, borderMode=borderMode, borderValue=bv or 0)
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            n = out.shape[-1]
            result[:, :, offset : offset + n] = out
            offset += n
    return result


@preserve_channel_dim
def warp_affine(
    img: ImageType,
    M: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: float | tuple[float, ...] | np.ndarray | None = None,
) -> ImageType:
    """Affine warp. Drop-in for cv2.warpAffine with multi-channel support.

    OpenCV warpAffine accepts >4 channels when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - borderValue is scalar or len <= 4

    We chunk when:
    - C > 4 AND (flags in {INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR_EXACT} OR bv is None)
    - bv is None: per-channel borderValue len>4 → _warp_affine_chunked
    - bv is not None: uniform borderValue → maybe_process_in_chunks

    Args:
        img: (H, W, C) image. uint8 or float32.
        M: 2x3 affine transformation matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags (cv2.INTER_*).
        borderMode: Border mode (cv2.BORDER_*).
        borderValue: Scalar, tuple, or array. Per-channel len>4 triggers chunking.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    num_channels = get_num_channels(img)
    bv = _border_value_for_cv2(borderValue, num_channels) if borderValue is not None else 0

    needs_chunk = num_channels > MAX_OPENCV_WORKING_CHANNELS and (flags in _INTERP_NEEDS_CHUNK or bv is None)
    if needs_chunk:
        if bv is None:
            return _warp_affine_chunked(img, M, dsize, flags, borderMode, borderValue)
        return maybe_process_in_chunks(
            cv2.warpAffine,
            M=M,
            dsize=dsize,
            flags=flags,
            borderMode=borderMode,
            borderValue=bv,
        )(img)

    return cv2.warpAffine(img, M, dsize, flags=flags, borderMode=borderMode, borderValue=bv or 0)


def _warp_perspective_chunked(
    img: ImageType,
    M: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    borderMode: int,
    borderValue: tuple[float, ...] | np.ndarray,
) -> ImageType:
    """Chunk warpPerspective when per-channel borderValue has len > 4.

    Same logic as _warp_affine_chunked: split into chunks of 4 (or 1 when
    remainder is 2), apply per-channel borderValue slice to each, pre-allocate
    output, copy chunks into place. borderValue is required (not None).
    """
    num_channels = img.shape[-1]
    val = np.array(borderValue, dtype=np.float64).flatten()
    result: np.ndarray | None = None
    offset = 0
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                ch = img[:, :, i + j : i + j + 1]
                bv = (float(val[i + j]),) * 4
                out = cv2.warpPerspective(ch, M, dsize, flags=flags, borderMode=borderMode, borderValue=bv)
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            ch = img[:, :, i : min(i + 4, num_channels)]
            bv = tuple(val[i : i + 4].tolist())
            out = cv2.warpPerspective(ch, M, dsize, flags=flags, borderMode=borderMode, borderValue=bv)
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            n = out.shape[-1]
            result[:, :, offset : offset + n] = out
            offset += n
    return result


@preserve_channel_dim
def warp_perspective(
    img: ImageType,
    M: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: float | tuple[float, ...] | np.ndarray | None = None,
) -> ImageType:
    """Perspective warp. Drop-in for cv2.warpPerspective with multi-channel support.

    OpenCV warpPerspective has a 4-channel limit on some builds (CI). For C > 4,
    we chunk into groups of 4 and reassemble. If borderValue is per-channel (len>4),
    use _warp_perspective_chunked; otherwise maybe_process_in_chunks with uniform bv.

    Args:
        img: (H, W, C) image.
        M: 3x3 perspective matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags.
        borderMode: Border mode.
        borderValue: Scalar, tuple, or array. Per-channel len>4 → chunked path.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    num_channels = get_num_channels(img)
    bv = _border_value_for_cv2(borderValue, num_channels) if borderValue is not None else 0
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        if bv is None:
            return _warp_perspective_chunked(img, M, dsize, flags, borderMode, borderValue)
        return maybe_process_in_chunks(
            cv2.warpPerspective,
            M=M,
            dsize=dsize,
            flags=flags,
            borderMode=borderMode,
            borderValue=bv,
        )(img)
    return cv2.warpPerspective(img, M, dsize, flags=flags, borderMode=borderMode, borderValue=bv or 0)


def _copy_make_border_chunked(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    borderType: int,
    value: tuple[float, ...] | np.ndarray,
) -> ImageType:
    """Chunk copyMakeBorder when per-channel value has len > 4.

    cv2.copyMakeBorder accepts at most 4 values. For per-channel padding (e.g.
    different fill per channel), split into chunks of 4, pad each with its slice,
    pre-allocate output, copy chunks. Same pattern as warp chunked functions.
    """
    num_channels = img.shape[-1]
    val = np.array(value, dtype=np.float64).flatten()
    result: np.ndarray | None = None
    offset = 0
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                ch = img[:, :, i + j : i + j + 1]
                v = (float(val[i + j]),) * 4
                out = cv2.copyMakeBorder(ch, top, bottom, left, right, borderType=borderType, value=v)
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            ch = img[:, :, i : min(i + 4, num_channels)]
            v = tuple(val[i : i + 4].tolist())
            out = cv2.copyMakeBorder(ch, top, bottom, left, right, borderType=borderType, value=v)
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            n = out.shape[-1]
            result[:, :, offset : offset + n] = out
            offset += n
    return result


@preserve_channel_dim
def copy_make_border(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    borderType: int = cv2.BORDER_CONSTANT,
    value: float | tuple[float, ...] | np.ndarray | None = None,
) -> ImageType:
    """Pad image with border. Drop-in for cv2.copyMakeBorder with multi-channel support.

    Chunks only when C > 4 AND value is per-channel (len>4, non-uniform). Otherwise
    uses cv2.copyMakeBorder directly. For BORDER_CONSTANT with scalar or len<=4,
    no chunking needed.

    Args:
        img: (H, W, C) image.
        top: Padding in pixels on top.
        bottom: Padding in pixels on bottom.
        left: Padding in pixels on left.
        right: Padding in pixels on right.
        borderType: cv2.BORDER_CONSTANT, BORDER_REPLICATE, etc.
        value: Fill value for BORDER_CONSTANT. Scalar or per-channel array.

    Returns:
        Padded image, shape (H+top+bottom, W+left+right, C).
    """
    num_channels = get_num_channels(img)
    bv = _border_value_for_cv2(value, num_channels) if value is not None else 0

    if num_channels > MAX_OPENCV_WORKING_CHANNELS and bv is None:
        return _copy_make_border_chunked(img, top, bottom, left, right, borderType, value)
    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=borderType,
        value=bv if value is not None else 0,
    )


@preserve_channel_dim
def remap(
    img: ImageType,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_CONSTANT,
) -> ImageType:
    """Remap image. Drop-in for cv2.remap with multi-channel support.

    cv2.remap has a 4-channel limit on some builds. For C > 4 we use
    maybe_process_in_chunks: split into chunks of 4, remap each, pre-allocate
    output, copy chunks. Same map_x/map_y applied to all channels.

    Args:
        img: (H, W, C) image.
        map_x: X-coordinate map, shape (H, W), float32.
        map_y: Y-coordinate map, shape (H, W), float32.
        interpolation: cv2.INTER_*.
        borderMode: cv2.BORDER_*.

    Returns:
        Remapped image, same shape as input.
    """
    num_channels = get_num_channels(img)
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return maybe_process_in_chunks(
            cv2.remap,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=borderMode,
        )(img)
    return cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=borderMode)
