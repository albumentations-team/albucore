"""Geometric operations with multi-channel support.

Drop-in for cv2.warpAffine, cv2.warpPerspective, cv2.copyMakeBorder, cv2.remap.
Chunking when OpenCV limits apply. blur, GaussianBlur, medianBlur, resize, filter2D
work out of the box for >4ch — use cv2 directly.
Run: python tools/verify_opencv_channel_limits.py
"""
# ruff: noqa: PLR0911 PLR0913  # chunked fns need many args

import cv2
import numpy as np

from albucore.decorators import preserve_channel_dim
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageType, get_num_channels, maybe_process_in_chunks

# Interpolations that require chunking for >4ch (CI: _src.channels() <= 4)
_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT}
# remap does not support INTER_LINEAR_EXACT; only CUBIC/LANCZOS4 need chunking
_REMAP_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}


def _border_value_for_cv2(
    value: float | tuple[float, ...] | np.ndarray,
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
        values_flat = value.flatten()
        if len(values_flat) <= 4:
            return tuple(values_flat.tolist()) if len(values_flat) > 1 else float(values_flat[0])
        if len(values_flat) > 4 and np.all(values_flat == values_flat[0]):
            return (float(values_flat[0]),) * 4
        return None  # per-channel len>4, needs chunking
    if isinstance(value, (tuple, list)):
        if len(value) <= 4:
            return tuple(value) if len(value) > 1 else value[0]
        if len(value) > 4 and all(elem == value[0] for elem in value):
            return (value[0],) * 4
        return None
    return (value,) * 4


__all__ = [
    "copy_make_border",
    "remap",
    "warp_affine",
    "warp_perspective",
]


def _warp_affine_chunked(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: tuple[float, ...] | np.ndarray | None,
) -> ImageType:
    """Chunk warpAffine when per-channel border_value has len > 4.

    OpenCV warpAffine accepts at most 4 border values. When the user passes
    per-channel values (e.g. (r,g,b,a,b2,b3,...) for 8ch), we must process in
    chunks of 4 and apply the correct slice to each chunk.

    Chunking strategy: split channels into groups of 4. For remainder 2, process
    as 2 single-channel chunks (cv2 often fails on 2ch). Pre-allocate output array
    after first chunk, then copy each chunk into its slice. Avoids np.concatenate.

    Args:
        img: (H, W, C) image.
        m: 2x3 affine matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags.
        border_mode: cv2.BORDER_* constant.
        border_value: Per-channel values, len == C. Flattened to 1D.

    Returns:
        Warped image (dsize[1], dsize[0], C).
    """
    num_channels = img.shape[-1]
    result: np.ndarray | None = None
    offset = 0
    channel_values = np.array(border_value, dtype=np.float64).flatten() if border_value is not None else None
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                channel_chunk = img[:, :, i + j : i + j + 1]
                border_value_cv2 = (float(channel_values[i + j]),) * 4 if channel_values is not None else 0
                out = cv2.warpAffine(
                    channel_chunk,
                    m,
                    dsize,
                    flags=flags,
                    borderMode=border_mode,
                    borderValue=border_value_cv2,
                )
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            channel_chunk = img[:, :, i : min(i + 4, num_channels)]
            border_value_cv2 = (
                tuple(channel_values[i : i + 4].tolist())
                if channel_values is not None and len(channel_values) >= i + 4
                else 0
            )
            out = cv2.warpAffine(
                channel_chunk,
                m,
                dsize,
                flags=flags,
                borderMode=border_mode,
                borderValue=border_value_cv2 or 0,
            )
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            chunk_size = out.shape[-1]
            result[:, :, offset : offset + chunk_size] = out
            offset += chunk_size
    return result


@preserve_channel_dim
def warp_affine(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> ImageType:
    """Affine warp. Drop-in for cv2.warpAffine with multi-channel support.

    OpenCV warpAffine accepts >4 channels when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - border_value is scalar or len <= 4

    We chunk when:
    - C > 4 AND (flags in {INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR_EXACT} OR border_value_cv2 is None)
    - border_value_cv2 is None: per-channel border_value len>4 → _warp_affine_chunked
    - border_value_cv2 is not None: uniform border_value → maybe_process_in_chunks

    Args:
        img: (H, W, C) image. uint8 or float32.
        m: 2x3 affine transformation matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags (cv2.INTER_*).
        border_mode: Border mode (cv2.BORDER_*).
        border_value: Scalar, tuple, or array. Per-channel len>4 triggers chunking.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    num_channels = get_num_channels(img)
    border_value_cv2 = _border_value_for_cv2(border_value) if border_value is not None else 0

    needs_chunk = num_channels > MAX_OPENCV_WORKING_CHANNELS and (
        flags in _INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            return _warp_affine_chunked(img, m, dsize, flags, border_mode, border_value)
        return maybe_process_in_chunks(
            cv2.warpAffine,
            M=m,
            dsize=dsize,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value_cv2,
        )(img)

    return cv2.warpAffine(img, m, dsize, flags=flags, borderMode=border_mode, borderValue=border_value_cv2 or 0)


def _warp_perspective_chunked(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: tuple[float, ...] | np.ndarray,
) -> ImageType:
    """Chunk warpPerspective when per-channel border_value has len > 4.

    Same logic as _warp_affine_chunked: split into chunks of 4 (or 1 when
    remainder is 2), apply per-channel border_value slice to each, pre-allocate
    output, copy chunks into place. border_value is required (not None).
    """
    num_channels = img.shape[-1]
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    result: np.ndarray | None = None
    offset = 0
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                channel_chunk = img[:, :, i + j : i + j + 1]
                border_value_cv2 = (float(channel_values[i + j]),) * 4
                out = cv2.warpPerspective(
                    channel_chunk,
                    m,
                    dsize,
                    flags=flags,
                    borderMode=border_mode,
                    borderValue=border_value_cv2,
                )
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            channel_chunk = img[:, :, i : min(i + 4, num_channels)]
            border_value_cv2 = tuple(channel_values[i : i + 4].tolist())
            out = cv2.warpPerspective(
                channel_chunk,
                m,
                dsize,
                flags=flags,
                borderMode=border_mode,
                borderValue=border_value_cv2,
            )
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            chunk_size = out.shape[-1]
            result[:, :, offset : offset + chunk_size] = out
            offset += chunk_size
    return result


@preserve_channel_dim
def warp_perspective(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> ImageType:
    """Perspective warp. Drop-in for cv2.warpPerspective with multi-channel support.

    OpenCV warpPerspective accepts >4 channels when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - border_value is scalar or len <= 4

    We chunk when:
    - C > 4 AND (flags in {INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR_EXACT} OR border_value_cv2 is None)
    - border_value_cv2 is None: per-channel border_value len>4 → _warp_perspective_chunked
    - border_value_cv2 is not None: uniform border_value → maybe_process_in_chunks

    Args:
        img: (H, W, C) image.
        m: 3x3 perspective matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags.
        border_mode: Border mode.
        border_value: Scalar, tuple, or array. Per-channel len>4 → chunked path.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    num_channels = get_num_channels(img)
    border_value_cv2 = _border_value_for_cv2(border_value) if border_value is not None else 0

    needs_chunk = num_channels > MAX_OPENCV_WORKING_CHANNELS and (
        flags in _INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            return _warp_perspective_chunked(img, m, dsize, flags, border_mode, border_value)
        return maybe_process_in_chunks(
            cv2.warpPerspective,
            M=m,
            dsize=dsize,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value_cv2,
        )(img)

    return cv2.warpPerspective(img, m, dsize, flags=flags, borderMode=border_mode, borderValue=border_value_cv2 or 0)


def _copy_make_border_chunked(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int,
    value: tuple[float, ...] | np.ndarray,
) -> ImageType:
    """Chunk copyMakeBorder when per-channel value has len > 4.

    cv2.copyMakeBorder accepts at most 4 values. For per-channel padding (e.g.
    different fill per channel), split into chunks of 4, pad each with its slice,
    pre-allocate output, copy chunks. Same pattern as warp chunked functions.
    """
    num_channels = img.shape[-1]
    channel_values = np.array(value, dtype=np.float64).flatten()
    result: np.ndarray | None = None
    offset = 0
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                channel_chunk = img[:, :, i + j : i + j + 1]
                chunk_border_value = (float(channel_values[i + j]),) * 4
                out = cv2.copyMakeBorder(
                    channel_chunk,
                    top,
                    bottom,
                    left,
                    right,
                    borderType=border_type,
                    value=chunk_border_value,
                )
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            channel_chunk = img[:, :, i : min(i + 4, num_channels)]
            chunk_border_value = tuple(channel_values[i : i + 4].tolist())
            out = cv2.copyMakeBorder(
                channel_chunk,
                top,
                bottom,
                left,
                right,
                borderType=border_type,
                value=chunk_border_value,
            )
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            chunk_size = out.shape[-1]
            result[:, :, offset : offset + chunk_size] = out
            offset += chunk_size
    return result


@preserve_channel_dim
def copy_make_border(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int = cv2.BORDER_CONSTANT,
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
        border_type: cv2.BORDER_CONSTANT, BORDER_REPLICATE, etc.
        value: Fill value for BORDER_CONSTANT. Scalar or per-channel array.

    Returns:
        Padded image, shape (H+top+bottom, W+left+right, C).
    """
    num_channels = get_num_channels(img)
    border_value_cv2 = _border_value_for_cv2(value) if value is not None else 0

    if num_channels > MAX_OPENCV_WORKING_CHANNELS and border_value_cv2 is None:
        return _copy_make_border_chunked(img, top, bottom, left, right, border_type, value)
    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=border_type,
        value=border_value_cv2 if value is not None else 0,
    )


def _remap_chunked(
    img: ImageType,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: tuple[float, ...] | np.ndarray,
) -> ImageType:
    """Chunk remap when per-channel border_value has len > 4.

    OpenCV remap accepts at most 4 border values. When the user passes
    per-channel values (e.g. (1,2,3,4,5,6,7,8) for 8ch), we must process in
    chunks of 4 and apply the correct slice to each chunk.
    """
    num_channels = img.shape[-1]
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    result: np.ndarray | None = None
    offset = 0
    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                channel_chunk = img[:, :, i + j : i + j + 1]
                border_value_cv2 = (float(channel_values[i + j]),) * 4
                out = cv2.remap(
                    channel_chunk,
                    map_x,
                    map_y,
                    interpolation,
                    borderMode=border_mode,
                    borderValue=border_value_cv2,
                )
                out = np.expand_dims(out, -1)
                if result is None:
                    result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
                result[:, :, offset : offset + 1] = out
                offset += 1
        else:
            channel_chunk = img[:, :, i : min(i + 4, num_channels)]
            border_value_cv2 = tuple(channel_values[i : i + 4].tolist())
            out = cv2.remap(
                channel_chunk,
                map_x,
                map_y,
                interpolation,
                borderMode=border_mode,
                borderValue=border_value_cv2,
            )
            if out.ndim == 2:
                out = np.expand_dims(out, -1)
            if result is None:
                result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
            chunk_size = out.shape[-1]
            result[:, :, offset : offset + chunk_size] = out
            offset += chunk_size
    return result


@preserve_channel_dim
def remap(
    img: ImageType,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
) -> ImageType:
    """Remap image. Drop-in for cv2.remap with multi-channel support.

    cv2.remap works for >4 channels when interpolation is NEAREST, LINEAR, or AREA,
    and border_value is scalar or len<=4. We chunk when:
    - C > 4 AND (interpolation in {CUBIC, LANCZOS4} OR per-channel border_value len>4)

    Args:
        img: (H, W, C) image.
        map_x: X-coordinate map, shape (H, W), float32.
        map_y: Y-coordinate map, shape (H, W), float32.
        interpolation: cv2.INTER_*.
        border_mode: cv2.BORDER_*.
        border_value: Scalar, tuple, or array. Per-channel len>4 triggers chunking.

    Returns:
        Remapped image, same shape as input.
    """
    num_channels = get_num_channels(img)
    border_value_cv2 = _border_value_for_cv2(border_value) if border_value is not None else 0

    needs_chunk = num_channels > MAX_OPENCV_WORKING_CHANNELS and (
        interpolation in _REMAP_INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            return _remap_chunked(img, map_x, map_y, interpolation, border_mode, border_value)
        return maybe_process_in_chunks(
            cv2.remap,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value_cv2,
        )(img)

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_value_cv2 or 0,
    )
