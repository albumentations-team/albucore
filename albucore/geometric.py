"""Geometric operations with multi-channel support.

Drop-in for cv2.warpAffine, cv2.warpPerspective, cv2.copyMakeBorder, cv2.remap.
Chunking when OpenCV limits apply. blur, GaussianBlur, medianBlur, resize, filter2D
work out of the box for >4ch — use cv2 directly.
OpenCV channel limits: see ``benchmarks/README.md``.
"""
# ruff: noqa: PLR0911 PLR0913  # chunked fns need many args

from collections.abc import Callable
from typing import cast, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from albucore.affine3d import warp_affine3d
from albucore.decorators import preserve_channel_dim
from albucore.filter3d import gaussian_blur3d, separable_filter3d
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    ImageType,
    get_num_channels,
    get_opencv_max_channels,
    maybe_process_in_chunks,
)

# Interpolations that require chunking for >4ch (CI: _src.channels() <= 4)
_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT}
# remap does not support INTER_LINEAR_EXACT; only CUBIC/LANCZOS4 need chunking
_REMAP_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}
_MAX_OPENCV_CHANNELS = get_opencv_max_channels()

__all__ = [
    "copy_make_border",
    "gaussian_blur3d",
    "remap",
    "resize",
    "resize3d",
    "separable_filter3d",
    "warp_affine",
    "warp_affine3d",
    "warp_perspective",
]


def _border_value_for_cv2(
    value: object,
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
    return None


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

    if result is None:
        msg = "Chunked geometric operation produced no output."
        raise RuntimeError(msg)
    return result


def _warp_affine_chunked(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk warpAffine when per-channel border_value has len > 4."""
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cast(
            "ImageType",
            cv2.warpAffine(chunk, m, dsize, flags=flags, borderMode=border_mode, borderValue=bv),
        ),
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

    OpenCV warpAffine accepts >4 channels up to its encoded channel limit when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - border_value is scalar or len <= 4

    We chunk when:
    - C exceeds OpenCV's encoded channel limit
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
        num_channels > _MAX_OPENCV_CHANNELS or flags in _INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            if border_value is None:
                msg = "border_value is required for chunked affine warp."
                raise ValueError(msg)
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

    return cast(
        "ImageType",
        cv2.warpAffine(
            img,
            m,
            dsize,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value_cv2 or 0,
            dst=dst,
        ),
    )


def _warp_perspective_chunked(
    img: ImageType,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk warpPerspective when per-channel border_value has len > 4."""
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cast(
            "ImageType",
            cv2.warpPerspective(chunk, m, dsize, flags=flags, borderMode=border_mode, borderValue=bv),
        ),
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

    OpenCV warpPerspective accepts >4 channels up to its encoded channel limit when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - border_value is scalar or len <= 4

    We chunk when:
    - C exceeds OpenCV's encoded channel limit
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
        num_channels > _MAX_OPENCV_CHANNELS or flags in _INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            if border_value is None:
                msg = "border_value is required for chunked perspective warp."
                raise ValueError(msg)
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

    return cast(
        "ImageType",
        cv2.warpPerspective(
            img,
            m,
            dsize,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value_cv2 or 0,
            dst=dst,
        ),
    )


def _copy_make_border_chunked(
    img: ImageType,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int,
    value: float | tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk copyMakeBorder when per-channel value has len > 4."""
    channel_values = np.array(value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cast(
            "ImageType",
            cv2.copyMakeBorder(chunk, top, bottom, left, right, borderType=border_type, value=bv),
        ),
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
        if value is None:
            msg = "value is required for chunked copy_make_border."
            raise ValueError(msg)
        return _copy_make_border_chunked(img, top, bottom, left, right, border_type, value, dst=dst)

    border_value_arg: float | tuple[float, ...]
    border_value_arg = 0.0 if value is None else cast("float | tuple[float, ...]", border_value_cv2)

    return cast(
        "ImageType",
        cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            borderType=border_type,
            value=border_value_arg,
            dst=dst,
        ),
    )


def _remap_chunked(
    img: ImageType,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray,
    dst: np.ndarray | None = None,
) -> ImageType:
    """Chunk remap when per-channel border_value has len > 4."""
    channel_values = np.array(border_value, dtype=np.float64).flatten()
    return _apply_in_chunks(
        img,
        channel_values,
        lambda chunk, bv: cast(
            "ImageType",
            cv2.remap(chunk, map_x, map_y, interpolation, borderMode=border_mode, borderValue=bv),
        ),
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

    cv2.remap works for >4 channels up to its encoded channel limit when
    interpolation is NEAREST, LINEAR, or AREA and border_value is scalar or
    len<=4. We chunk when:
    - C exceeds OpenCV's encoded channel limit
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
        num_channels > _MAX_OPENCV_CHANNELS or interpolation in _REMAP_INTERP_NEEDS_CHUNK or border_value_cv2 is None
    )
    if needs_chunk:
        if border_value_cv2 is None:
            if border_value is None:
                msg = "border_value is required for chunked remap."
                raise ValueError(msg)
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

    return cast(
        "ImageType",
        cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value_cv2 or 0,
            dst=dst,
        ),
    )


@preserve_channel_dim
def resize(
    img: ImageType,
    dsize: tuple[int, int],
    fx: float = 0.0,
    fy: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> ImageType:
    """Resize image. Drop-in for cv2.resize with full multi-channel support.

    cv2.resize with INTER_AREA asserts cn <= 4 internally on downscale, so 5+ channel images
    being downscaled with INTER_AREA are processed in chunks of up to 4 channels.
    All other cases are passed directly to cv2.resize.

    Args:
        img: (H, W, C) image. uint8 or float32.
        dsize: (width, height) output size. Pass (0, 0) and use fx/fy for scale factors.
        fx: Scale factor along the horizontal axis. Used only when dsize is (0, 0).
        fy: Scale factor along the vertical axis. Used only when dsize is (0, 0).
        interpolation: Interpolation flag (cv2.INTER_LINEAR, INTER_NEAREST, INTER_CUBIC,
            INTER_AREA, INTER_LANCZOS4, etc.).

    Returns:
        Resized image with same dtype and channel count as input.
    """
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

    num_channels = get_num_channels(img)
    is_downscale = dsize[0] < img.shape[1] or dsize[1] < img.shape[0]
    if num_channels > MAX_OPENCV_WORKING_CHANNELS and interpolation == cv2.INTER_AREA and is_downscale:
        return maybe_process_in_chunks(cv2.resize, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)(img)

    return cast("ImageType", cv2.resize(img, dsize, fx=fx, fy=fy, interpolation=interpolation))


_RESIZE3D_INTERPOLATIONS = frozenset((cv2.INTER_LINEAR, cv2.INTER_NEAREST))
_RESIZE3D_NUMPY_PER_SLICE_MAX_ELEMENTS = 1_000_000
_RESIZE3D_TORCH_NUMPY_BRIDGE_MIN_OUTPUT_ELEMENTS = 10_000


def _validate_resize3d_interpolation(interpolation: int, antialias: bool) -> None:
    """Reject modes that do not have one shared NumPy/Torch 3D contract."""
    if interpolation not in _RESIZE3D_INTERPOLATIONS:
        msg = f"resize3d supports only cv2.INTER_LINEAR and cv2.INTER_NEAREST; got interpolation={interpolation}."
        raise ValueError(msg)
    if antialias and interpolation == cv2.INTER_NEAREST:
        msg = "antialias=True requires cv2.INTER_LINEAR."
        raise ValueError(msg)


def _validate_resize3d_numpy(volume: np.ndarray) -> None:
    """Validate the NumPy dtype required by the resize kernels."""
    if volume.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        msg = f"Unsupported dtype {volume.dtype}. Albucore resize3d supports only uint8 and float32."
        raise ValueError(msg)


def _validate_resize3d_torch(volume: torch.Tensor) -> None:
    """Validate the Torch dtype required by the resize kernels."""
    if volume.dtype not in (torch.uint8, torch.float32):
        msg = f"Unsupported dtype {volume.dtype}. Albucore resize3d supports only torch.uint8 and torch.float32."
        raise ValueError(msg)


def _resize3d_axis_packing(
    volume: np.ndarray,
    axis: int,
    output_size: int,
    interpolation: int,
) -> np.ndarray:
    """Resize one DHWC spatial axis through a channel-safe packed 2D call."""
    depth, height, width, channels = volume.shape
    if volume.shape[axis] == output_size:
        return volume

    if axis == 0:
        flattened = volume.transpose(1, 2, 0, 3).reshape(height * width, depth, channels)
        resized = resize(flattened, (output_size, height * width), interpolation=interpolation)
        return cast("np.ndarray", resized.reshape(height, width, output_size, channels).transpose(2, 0, 1, 3))
    if axis == 1:
        flattened = volume.transpose(0, 2, 1, 3).reshape(depth * width, height, channels)
        resized = resize(flattened, (output_size, depth * width), interpolation=interpolation)
        return cast("np.ndarray", resized.reshape(depth, width, output_size, channels).transpose(0, 2, 1, 3))

    flattened = volume.reshape(depth * height, width, channels)
    resized = resize(flattened, (output_size, depth * height), interpolation=interpolation)
    return resized.reshape(depth, height, output_size, channels)


def _axis_interpolation(input_size: int, output_size: int, interpolation: int, antialias: bool) -> int:
    """Select area resampling only for a NumPy axis that shrinks."""
    return cv2.INTER_AREA if antialias and output_size < input_size else interpolation


def _resize3d_numpy_axis_packing(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> np.ndarray:
    """Three independent axis passes; fallback for high packed channel counts and mixed antialiasing."""
    result = volume
    for axis, output_size in enumerate(size):
        axis_interpolation = _axis_interpolation(result.shape[axis], output_size, interpolation, antialias)
        result = _resize3d_axis_packing(result, axis, output_size, axis_interpolation)
    return result


def _resize3d_numpy_linear_axis(volume: np.ndarray, axis: int, output_size: int) -> np.ndarray:
    """Apply one half-pixel linear axis pass in float32 without OpenCV channel packing."""
    input_size = volume.shape[axis]
    if input_size == output_size:
        return volume

    output_coordinates = np.arange(output_size, dtype=np.float32)
    scale = np.float32(input_size) / np.float32(output_size)
    source_coordinates = (output_coordinates + np.float32(0.5)) * scale - np.float32(0.5)
    source_floor = np.floor(source_coordinates).astype(np.intp)
    left = np.clip(source_floor, 0, input_size - 1)
    right = np.clip(source_floor + 1, 0, input_size - 1)
    weight_shape = [1] * volume.ndim
    weight_shape[axis] = output_size
    right_weight = (source_coordinates - source_floor.astype(np.float32)).reshape(weight_shape)
    left_values = np.take(volume, left, axis=axis)
    right_values = np.take(volume, right, axis=axis)
    return cast("np.ndarray", left_values * (np.float32(1.0) - right_weight) + right_values * right_weight)


def _resize3d_numpy_linear_three_pass(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Resize DHWC linearly with full-float intermediates and one final uint8 rounding step."""
    result = volume.astype(np.float32, copy=False)
    for axis, output_size in enumerate(size):
        result = _resize3d_numpy_linear_axis(result, axis, output_size)
    if volume.dtype == np.uint8:
        return np.asarray(np.minimum(result + np.float32(0.5), np.float32(255)).astype(np.uint8))
    return result


def _resize3d_numpy_per_slice(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> np.ndarray:
    """Resize each HWC slice in H/W before one packed depth pass."""
    depth, height, width, channels = volume.shape
    result = volume
    if (height, width) != size[1:]:
        hw_interpolation = _axis_interpolation(height, size[1], interpolation, antialias)
        result = np.empty((depth, size[1], size[2], channels), dtype=volume.dtype)
        for index in range(depth):
            result[index] = resize(volume[index], (size[2], size[1]), interpolation=hw_interpolation)
    if result.shape[0] != size[0]:
        depth_interpolation = _axis_interpolation(result.shape[0], size[0], interpolation, antialias)
        result = _resize3d_axis_packing(result, 0, size[0], depth_interpolation)
    return result


def _can_resize3d_numpy_torch(volume: np.ndarray, interpolation: int, antialias: bool) -> bool:
    """Check that a NumPy volume can share CPU storage with Torch without an input repair copy."""
    return (
        interpolation == cv2.INTER_LINEAR
        and not antialias
        and volume.flags.writeable
        and all(stride >= 0 for stride in volume.strides)
    )


def _resize3d_numpy_torch_cpu(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Use the measured full NumPy DHWC → Torch CDHW → NumPy route for selected CPU regions."""
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    result = _resize3d_torch_cpu(tensor, size, cv2.INTER_LINEAR)
    return cast("np.ndarray", result.permute(1, 2, 3, 0).numpy())


def _can_resize3d_joint_hw(volume: np.ndarray, size: tuple[int, int, int], antialias: bool) -> bool:
    """Check whether packing D*C as OpenCV channels preserves the requested antialias semantics."""
    if volume.shape[0] * volume.shape[-1] > _MAX_OPENCV_CHANNELS:
        return False
    if not antialias:
        return True

    source_height, source_width = volume.shape[1:3]
    target_height, target_width = size[1:]
    directions = {
        target_size > source_size
        for source_size, target_size in ((source_height, target_height), (source_width, target_width))
        if target_size != source_size
    }
    return len(directions) <= 1


def _resize3d_numpy_joint_hw(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> np.ndarray:
    """Resize H/W together, then depth; reduces one interpolation pass when D*C fits OpenCV."""
    depth, height, width, channels = volume.shape
    result = volume

    if (height, width) != size[1:]:
        hw_interpolation = _axis_interpolation(height, size[1], interpolation, antialias)
        packed = result.transpose(1, 2, 0, 3).reshape(height, width, depth * channels)
        resized_hw = resize(packed, (size[2], size[1]), interpolation=hw_interpolation)
        result = resized_hw.reshape(size[1], size[2], depth, channels).transpose(2, 0, 1, 3)

    if result.shape[0] != size[0]:
        depth_interpolation = _axis_interpolation(result.shape[0], size[0], interpolation, antialias)
        result = _resize3d_axis_packing(result, 0, size[0], depth_interpolation)
    return result


def _resize3d_numpy(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> np.ndarray:
    """Choose the measured NumPy, OpenCV, or CPU Torch full route for one DHWC volume."""
    source_size = volume.shape[:3]
    all_down = all(target_axis < source_axis for source_axis, target_axis in zip(source_size, size, strict=True))
    all_up = all(target_axis > source_axis for source_axis, target_axis in zip(source_size, size, strict=True))
    torch_compatible = _can_resize3d_numpy_torch(volume, interpolation, antialias)

    if interpolation == cv2.INTER_LINEAR and not antialias and size[0] == 1 and volume.shape[0] != 1:
        if volume.dtype == np.uint8 and torch_compatible:
            return _resize3d_numpy_torch_cpu(volume, size)
        return _resize3d_numpy_linear_three_pass(volume, size)

    if all_down:
        if torch_compatible and (volume.dtype == np.float32 or volume.shape[-1] > 1):
            return _resize3d_numpy_torch_cpu(volume, size)
        if volume.shape[-1] == 1 and volume.size <= _RESIZE3D_NUMPY_PER_SLICE_MAX_ELEMENTS:
            return _resize3d_numpy_per_slice(volume, size, interpolation, antialias)

    if (
        torch_compatible
        and volume.dtype == np.float32
        and (volume.shape[-1] > 1 or volume.size > _RESIZE3D_NUMPY_PER_SLICE_MAX_ELEMENTS)
        and (volume.shape[1] == size[1] or volume.shape[2] == size[2])
    ):
        return _resize3d_numpy_torch_cpu(volume, size)

    if volume.shape[1] == size[1] or volume.shape[2] == size[2]:
        return _resize3d_numpy_axis_packing(volume, size, interpolation, antialias)
    if _can_resize3d_joint_hw(volume, size, antialias):
        return _resize3d_numpy_joint_hw(volume, size, interpolation, antialias)
    if all_up:
        return _resize3d_numpy_per_slice(volume, size, interpolation, antialias)
    return _resize3d_numpy_axis_packing(volume, size, interpolation, antialias)


def _should_resize3d_torch_use_numpy_route(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int,
) -> bool:
    """Select the measured zero-copy path for sufficiently large all-axis linear Tensor upscales."""
    return (
        interpolation == cv2.INTER_LINEAR
        and volume.shape[0] * size[0] * size[1] * size[2] >= _RESIZE3D_TORCH_NUMPY_BRIDGE_MIN_OUTPUT_ELEMENTS
        and all(output_size > input_size for input_size, output_size in zip(volume.shape[1:], size, strict=True))
    )


def _resize3d_torch_via_numpy(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int,
) -> torch.Tensor:
    """Bridge a prevalidated CPU CDHW Tensor through the faster benchmark-routed DHWC CPU path."""
    numpy_volume = volume.permute(1, 2, 3, 0).numpy()
    resized = _resize3d_numpy(numpy_volume, size, interpolation, antialias=False)
    return torch.from_numpy(resized).permute(3, 0, 1, 2)


def _resize3d_torch_cpu(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int,
) -> torch.Tensor:
    """Resize one CPU CDHW tensor with native Torch interpolation and no autograd graph."""
    with torch.inference_mode():
        if interpolation == cv2.INTER_NEAREST:
            result = cast("torch.Tensor", torch_f.interpolate(volume.unsqueeze(0), size=size, mode="nearest"))
            return result.squeeze(0)

        working_volume = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
        result = cast(
            "torch.Tensor",
            torch_f.interpolate(working_volume.unsqueeze(0), size=size, mode="trilinear", align_corners=False),
        )
        if volume.dtype == torch.uint8:
            result = torch.minimum(result + 0.5, result.new_tensor(255)).to(torch.uint8)
        return result.squeeze(0)


@overload
def resize3d(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> np.ndarray: ...


@overload
def resize3d(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> torch.Tensor: ...


def resize3d(
    volume: object,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> np.ndarray | torch.Tensor:
    """Resize one volume along its depth, height, and width axes.

    NumPy input uses channel-last ``(D, H, W, C)`` layout. Torch input uses channel-first
    ``(C, D, H, W)`` layout. AlbumentationsX validates the input layout, spatial dimensions, output size,
    device, and autograd state before calling this primitive. The container, dtype, and channel count are preserved.
    Linear Tensor resize that enlarges all three spatial axes and produces at least 10,000 output elements uses the
    benchmark-selected zero-copy NumPy/OpenCV CPU route. Its float32 output is within ``rtol=2e-4``, ``atol=3e-5``
    of native Torch interpolation; uint8 differs by at most one value. Other Tensor regions use native Torch
    interpolation.

    Args:
        volume: Prevalidated NumPy ``DHWC`` array or Torch ``CDHW`` tensor. Only uint8 and float32 are supported.
        size: Prevalidated output ``(depth, height, width)``.
        interpolation: ``cv2.INTER_LINEAR`` or ``cv2.INTER_NEAREST``.
        antialias: For NumPy linear interpolation, use ``INTER_AREA`` on shrinking axes. Torch does not support
            antialiased 5D trilinear interpolation and raises ``NotImplementedError`` when this is true.

    Returns:
        Resized volume in the same container and layout as ``volume``. An identity resize returns ``volume`` itself.

    Raises:
        ValueError: If the dtype or interpolation contract is invalid.
        NotImplementedError: If a Torch input requests antialiasing.
        TypeError: If ``volume`` is neither ``np.ndarray`` nor ``torch.Tensor``.
    """
    _validate_resize3d_interpolation(interpolation, antialias)

    if isinstance(volume, np.ndarray):
        _validate_resize3d_numpy(volume)
        if volume.shape[:3] == size:
            return volume
        return _resize3d_numpy(volume, size, interpolation, antialias)

    if isinstance(volume, torch.Tensor):
        _validate_resize3d_torch(volume)
        if antialias:
            msg = (
                "Torch does not support antialias=True for 5D trilinear interpolation. "
                "See https://github.com/pytorch/pytorch/issues/191896."
            )
            raise NotImplementedError(msg)
        if tuple(volume.shape[1:]) == size:
            return volume
        if _should_resize3d_torch_use_numpy_route(volume, size, interpolation):
            return _resize3d_torch_via_numpy(volume, size, interpolation)
        return _resize3d_torch_cpu(volume, size, interpolation)

    msg = f"resize3d supports np.ndarray and torch.Tensor, got {type(volume).__name__}."
    raise TypeError(msg)
