"""Geometric operations with multi-channel support.

Drop-in for cv2.warpAffine, cv2.warpPerspective, cv2.copyMakeBorder, cv2.remap.
Chunking when OpenCV limits apply. blur, GaussianBlur, medianBlur, resize, filter2D
work out of the box for >4ch — use cv2 directly.
Run: python tools/verify_opencv_channel_limits.py
"""
# ruff: noqa: PLR0911 PLR0913  # chunked fns need many args

from collections.abc import Callable

import cv2
import numpy as np

from albucore.decorators import preserve_channel_dim
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageType, get_num_channels, maybe_process_in_chunks

# Interpolations that require chunking for >4ch (CI: _src.channels() <= 4)
_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT}
# remap does not support INTER_LINEAR_EXACT; only CUBIC/LANCZOS4 need chunking
_REMAP_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}

__all__ = [
    "copy_make_border",
    "remap",
    "resize",
    "warp_affine",
    "warp_perspective",
]


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
        if np.all(values_flat == values_flat[0]):
            return (float(values_flat[0]),) * 4
        return None  # per-channel len>4, needs chunking
    if isinstance(value, (tuple, list)):
        if len(value) <= 4:
            return tuple(value) if len(value) > 1 else value[0]
        if all(elem == value[0] for elem in value):
            return (value[0],) * 4
        return None
    return (value,) * 4


def _apply_in_chunks(
    img: ImageType,
    channel_values: np.ndarray,
    fn: Callable[[ImageType, tuple[float, ...]], ImageType],
    dst: np.ndarray | None = None,
) -> np.ndarray:
    """Apply fn(chunk, border_value) over groups of ≤4 channels.

    The 2-channel remainder is processed as individual 1-channel slices because
    cv2 can fail on 2-channel inputs for some operations.

    Args:
        img: (H, W, C) image.
        channel_values: Per-channel border values, shape (C,).
        fn: Function taking (chunk, border_value_tuple) → warped chunk.
        dst: Optional pre-allocated output array. Allocated on first chunk if None.

    Returns:
        Output image (H_out, W_out, C).
    """
    num_channels = img.shape[-1]
    result: np.ndarray | None = dst
    offset = 0

    def apply(chunk: ImageType, bv: tuple[float, ...]) -> None:
        nonlocal result, offset
        out = np.atleast_3d(fn(chunk, bv))
        if result is None:
            result = np.empty((*out.shape[:2], num_channels), dtype=img.dtype)
        chunk_size = out.shape[-1]
        result[:, :, offset : offset + chunk_size] = out
        offset += chunk_size

    for i in range(0, num_channels, 4):
        if num_channels - i == 2:
            for j in range(2):
                apply(img[:, :, i + j : i + j + 1], (float(channel_values[i + j]),) * 4)
        else:
            apply(img[:, :, i : min(i + 4, num_channels)], tuple(channel_values[i : i + 4].tolist()))

    return result


def _warp_affine_chunked(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk warpAffine when per-channel border_value has len > 4."""
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cv2.warpAffine(chunk, m, dsize, flags=flags, borderMode=border_mode, borderValue=bv),
        dst=dst,
    )


@preserve_channel_dim
def warp_affine(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | None = None,
) -> ImageType:
    """Affine warp. Drop-in for cv2.warpAffine with multi-channel support.

    Accepts 2x3 or 3x3 affine matrix (3x3 uses first two rows).

    OpenCV warpAffine accepts >4 channels when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - border_value is scalar or len <= 4

    We chunk when:
    - C > 4 AND (flags in {INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR_EXACT} OR border_value_cv2 is None)
    - border_value_cv2 is None: per-channel border_value len>4 → _warp_affine_chunked
    - border_value_cv2 is not None: uniform border_value → maybe_process_in_chunks

    Args:
        img: (H, W, C) image. uint8 or float32.
        m: 2x3 or 3x3 affine matrix (3x3 uses first two rows).
        dsize: (width, height) output size.
        flags: Interpolation flags (cv2.INTER_*).
        border_mode: Border mode (cv2.BORDER_*).
        border_value: Scalar, tuple, or array. Per-channel len>4 triggers chunking.
        dst: Optional pre-allocated output array.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    m = np.asarray(m[:2, :], dtype=np.float32)
    num_channels = get_num_channels(img)
    border_value_cv2 = _border_value_for_cv2(border_value) if border_value is not None else 0

    needs_chunk = num_channels > MAX_OPENCV_WORKING_CHANNELS and (
        flags in _INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            return _warp_affine_chunked(img, m, dsize, flags, border_mode, border_value, dst=dst)
        return maybe_process_in_chunks(
            cv2.warpAffine,
            M=m,
            dsize=dsize,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value_cv2,
            dst=dst,
        )(img)

    return cv2.warpAffine(
        img,
        m,
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value_cv2 or 0,
        dst=dst,
    )


def _warp_perspective_chunked(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk warpPerspective when per-channel border_value has len > 4."""
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cv2.warpPerspective(chunk, m, dsize, flags=flags, borderMode=border_mode, borderValue=bv),
        dst=dst,
    )


@preserve_channel_dim
def warp_perspective(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | None = None,
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
        dst: Optional pre-allocated output array.

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
            return _warp_perspective_chunked(img, m, dsize, flags, border_mode, border_value, dst=dst)
        return maybe_process_in_chunks(
            cv2.warpPerspective,
            M=m,
            dsize=dsize,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value_cv2,
            dst=dst,
        )(img)

    return cv2.warpPerspective(
        img,
        m,
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value_cv2 or 0,
        dst=dst,
    )


def _copy_make_border_chunked(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int,
    value: tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk copyMakeBorder when per-channel value has len > 4."""
    channel_values = np.array(value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cv2.copyMakeBorder(chunk, top, bottom, left, right, borderType=border_type, value=bv),
        dst=dst,
    )


@preserve_channel_dim
def copy_make_border(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int = cv2.BORDER_CONSTANT,
    value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | None = None,
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
        dst: Optional pre-allocated output array.

    Returns:
        Padded image, shape (H+top+bottom, W+left+right, C).
    """
    num_channels = get_num_channels(img)
    border_value_cv2 = _border_value_for_cv2(value) if value is not None else 0

    if num_channels > MAX_OPENCV_WORKING_CHANNELS and border_value_cv2 is None:
        return _copy_make_border_chunked(img, top, bottom, left, right, border_type, value, dst=dst)

    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=border_type,
        value=border_value_cv2 if value is not None else 0,
        dst=dst,
    )


def _remap_chunked(
    img: ImageType,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk remap when per-channel border_value has len > 4."""
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cv2.remap(chunk, map_x, map_y, interpolation, borderMode=border_mode, borderValue=bv),
        dst=dst,
    )


@preserve_channel_dim
def remap(
    img: ImageType,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | None = None,
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
        dst: Optional pre-allocated output array.

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
            return _remap_chunked(img, map_x, map_y, interpolation, border_mode, border_value, dst=dst)
        return maybe_process_in_chunks(
            cv2.remap,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value_cv2,
            dst=dst,
        )(img)

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_value_cv2 or 0,
        dst=dst,
    )


@preserve_channel_dim
def resize(
    img: ImageType,
    dsize: tuple[int, int],
    fx: float = 0.0,
    fy: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> ImageType:
    """Resize image. Drop-in for cv2.resize with multi-channel support and optimized performance.

    For images with 5 or more channels, cv2.warpAffine is often faster than cv2.resize.
    This function dynamically selects the fastest OpenCV implementation.

    Args:
        img: (H, W, C) image. uint8 or float32.
        dsize: (width, height) output size.
        fx: Scale factor along the horizontal axis.
        fy: Scale factor along the vertical axis.
        interpolation: Interpolation flag (e.g., cv2.INTER_LINEAR).

    Returns:
        Resized image.
    """
    num_channels = get_num_channels(img)

    # Calculate actual output size only if dsize is (0, 0), matching cv2.resize semantics
    if dsize[0] == 0 and dsize[1] == 0:
        if fx <= 0 or fy <= 0:
            msg = "When dsize is (0, 0), fx and fy must be positive to compute the output size."
            raise ValueError(msg)
        width = round(img.shape[1] * fx)
        height = round(img.shape[0] * fy)
        if width <= 0 or height <= 0:
            msg = f"Computed dsize from fx and fy is invalid: ({width}, {height})."
            raise ValueError(msg)
        dsize = (width, height)

    # Use warpAffine for 5+ channels if it's a simple interpolation
    if num_channels >= 5 and interpolation in {cv2.INTER_LINEAR, cv2.INTER_NEAREST}:
        height, width = img.shape[:2]
        scale_x = dsize[0] / width
        scale_y = dsize[1] / height

        # Shift to match cv2.resize pixel center mapping
        m = np.float32(
            [
                [scale_x, 0.0, scale_x * 0.5 - 0.5],
                [0.0, scale_y, scale_y * 0.5 - 0.5],
            ],
        )

        return cv2.warpAffine(
            img,
            m,
            dsize,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return cv2.resize(img, dsize, fx=fx, fy=fy, interpolation=interpolation)
