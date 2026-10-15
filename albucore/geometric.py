"""Geometric operations with multi-channel support.

Drop-in for cv2.warpAffine, cv2.warpPerspective, cv2.copyMakeBorder, cv2.remap.
Chunking when OpenCV limits apply. blur, GaussianBlur, medianBlur, resize, filter2D
work out of the box for >4ch — use cv2 directly.
OpenCV channel limits: see ``benchmarks/README.md``.
"""

from collections.abc import Callable
from typing import Any, cast, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from albucore._float_cast import _cast_float_numpy
from albucore.affine3d import warp_affine3d
from albucore.decorators import preserve_channel_dim
from albucore.filter3d import gaussian_blur3d, separable_filter3d
from albucore.pad3d import pad3d
from albucore.remap3d import remap3d
from albucore.tensor_bridge import dispatch_tensor_image, tensor_operand_to_numpy, tensor_to_numpy_channel_last
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    ArrayT,
    ImageFloat16,
    ImageType,
    TensorImageLayout,
    get_num_channels,
    get_opencv_max_channels,
    maybe_process_in_chunks,
)

# Interpolations that require chunking for >4ch (CI: _src.channels() <= 4)
_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT}
# remap does not support INTER_LINEAR_EXACT; only CUBIC/LANCZOS4 need chunking
_REMAP_INTERP_NEEDS_CHUNK = {cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}
_MAX_OPENCV_CHANNELS = get_opencv_max_channels()
_RESIZE_TENSOR_FLOAT16_LINEAR_MIN_ELEMENTS = 16_384
_RESIZE_TENSOR_FLOAT16_NEAREST_MIN_ELEMENTS = 61_440

__all__ = [
    "copy_make_border",
    "flip_volume",
    "gaussian_blur3d",
    "pad3d",
    "remap",
    "remap3d",
    "resize",
    "resize3d",
    "rot90_volume",
    "separable_filter3d",
    "transpose_volume",
    "warp_affine",
    "warp_affine3d",
    "warp_perspective",
]


@overload
def flip_volume(
    volume: ArrayT,
    axes: int | tuple[int, ...],
) -> ArrayT: ...


@overload
def flip_volume(
    volume: torch.Tensor,
    axes: int | tuple[int, ...],
) -> torch.Tensor: ...


def flip_volume(
    volume: np.ndarray | torch.Tensor,
    axes: int | tuple[int, ...],
) -> np.ndarray | torch.Tensor:
    """Flip one or more axes in a prevalidated volume.

    NumPy volumes use ``(D, H, W, C)`` layout and CPU Tensors use ``(C, D, H, W)``.

    Args:
        volume: Prevalidated uint8, float16, or float32 NumPy ``DHWC`` volume or CPU Torch ``CDHW`` tensor.
        axes: Axis or axes in the supplied container's native layout.

    Returns:
        A flipped volume with the same container, layout, shape, and dtype, preserving the selected backend's native
        stride semantics.

    The caller validates container, layout, dtype, device, strides, and autograd state before entering Albucore.
    """
    if isinstance(volume, np.ndarray):
        return np.flip(volume, axis=axes)
    if volume.dtype == torch.float16 and volume.is_contiguous():
        dims = (axes,) if isinstance(axes, int) else axes
        return torch.flip(volume.view(torch.int16), dims=dims).view(torch.float16)
    return torch.flip(volume, dims=(axes,) if isinstance(axes, int) else axes)


@overload
def transpose_volume(volume: ArrayT, axis1: int, axis2: int) -> ArrayT: ...


@overload
def transpose_volume(volume: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor: ...


def transpose_volume(volume: np.ndarray | torch.Tensor, axis1: int, axis2: int) -> np.ndarray | torch.Tensor:
    """Transpose two axes in a prevalidated volume.

    NumPy volumes use ``(D, H, W, C)`` layout and CPU Tensors use ``(C, D, H, W)``.

    Args:
        volume: Prevalidated uint8, float16, or float32 NumPy ``DHWC`` volume or CPU Torch ``CDHW`` tensor.
        axis1: First axis in the supplied container's native layout.
        axis2: Second axis in the supplied container's native layout.

    Returns:
        A transposed volume with the same container and dtype, preserving the selected backend's native stride
        semantics.

    The caller validates container, layout, dtype, device, strides, and autograd state before entering Albucore.
    """
    if isinstance(volume, np.ndarray):
        return np.swapaxes(volume, axis1, axis2)
    return volume.transpose(axis1, axis2)


@overload
def rot90_volume(
    volume: ArrayT,
    k: int,
    axes: tuple[int, int],
) -> ArrayT: ...


@overload
def rot90_volume(
    volume: torch.Tensor,
    k: int,
    axes: tuple[int, int],
) -> torch.Tensor: ...


def rot90_volume(
    volume: np.ndarray | torch.Tensor,
    k: int,
    axes: tuple[int, int],
) -> np.ndarray | torch.Tensor:
    """Rotate a prevalidated volume counterclockwise in one ordered axis plane.

    NumPy volumes use ``(D, H, W, C)`` layout and CPU Tensors use ``(C, D, H, W)``.

    Args:
        volume: Prevalidated uint8, float16, or float32 NumPy ``DHWC`` volume or CPU Torch ``CDHW`` tensor.
        k: Number of counterclockwise 90-degree turns.
        axes: Ordered axis pair in the supplied container's native layout.

    Returns:
        A rotated volume with the same container and dtype, preserving the selected backend's native stride semantics.

    The caller validates container, layout, dtype, device, strides, autograd state, k, and axes before entering
    Albucore.
    """
    if isinstance(volume, np.ndarray):
        return np.rot90(volume, k, axes=axes)
    if volume.dtype == torch.float16 and volume.is_contiguous():
        return torch.rot90(volume.view(torch.int16), k, dims=axes).view(torch.float16)
    return torch.rot90(volume, k, dims=axes)


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
        img: (H, W, C) image. uint8, float16, or float32.
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


def _float16_nearest_integer_source(img: ImageFloat16) -> np.ndarray | None:
    channels = img.shape[-1]
    if channels == 9 and channels * 2 <= _MAX_OPENCV_CHANNELS:
        return img.view(np.uint8).reshape(*img.shape[:2], channels * 2)
    if channels in (1, 3) and channels <= _MAX_OPENCV_CHANNELS:
        return img.view(np.uint16)
    return None


def _warp_affine_float16_nearest(
    img: ImageFloat16,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray | None,
    dst: ImageType | None,
) -> ImageType | None:
    if (
        not img.flags.c_contiguous
        or flags != cv2.INTER_NEAREST
        or border_mode != cv2.BORDER_CONSTANT
        or border_value is not None
        or dst is not None
        or dsize[0] <= 0
        or dsize[1] <= 0
    ):
        return None

    source = _float16_nearest_integer_source(img)
    if source is None:
        return None
    channels = img.shape[-1]
    matrix = np.asarray(m[:2, :], dtype=np.float32)
    result = cv2.warpAffine(
        source,
        matrix,
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=0,
    )
    return cast("ImageType", result.view(np.float16).reshape(dsize[1], dsize[0], channels))


def _warp_affine_float16(
    img: ImageFloat16,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray | None,
    dst: ImageType | None,
) -> ImageType:
    result = _warp_affine_float16_nearest(img, m, dsize, flags, border_mode, border_value, dst)
    if result is None:
        wide_result = warp_affine(
            _cast_float_numpy(img, np.float32),
            m,
            dsize,
            flags,
            border_mode,
            border_value,
        )
        result = _cast_float_numpy(wide_result, np.float16)
    if dst is None:
        return result
    np.copyto(dst, result)
    return dst


@overload
def warp_affine(
    img: ArrayT,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ArrayT | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ArrayT: ...


@overload
def warp_affine(
    img: torch.Tensor,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> torch.Tensor: ...


@preserve_channel_dim
def warp_affine(
    img: ImageType | torch.Tensor,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ImageType | torch.Tensor:
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

    Contiguous float16 nearest calls with the default zero constant border use bit-preserving integer views: CV_16U
    for C=1/3 and a CV_8U byte view for C=9. Other float16 controls and interpolation modes use the float32 path.

    Args:
        img: (H, W, C) image. uint8, float16, or float32.
        m: 2x3 or 3x3 affine matrix (3x3 uses first two rows).
        dsize: (width, height) output size.
        flags: Interpolation flags (cv2.INTER_*).
        border_mode: Border mode (cv2.BORDER_*).
        border_value: Scalar, tuple, or array. Per-channel len>4 triggers chunking.
        dst: Optional pre-allocated output array.
        layout: Required for Tensor input; selects CHW or NCHW.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    if isinstance(img, torch.Tensor):
        matrix = tensor_operand_to_numpy(m, layout)
        dst_view = tensor_to_numpy_channel_last(dst, layout) if isinstance(dst, torch.Tensor) else dst
        result = dispatch_tensor_image(
            img,
            layout,
            warp_affine,
            matrix,
            dsize,
            flags,
            border_mode,
            border_value,
            dst=dst_view,
        )
        return dst if isinstance(dst, torch.Tensor) else result

    dst = cast("ImageType | None", dst)

    if img.dtype == np.float16:
        return _warp_affine_float16(cast("ImageFloat16", img), m, dsize, flags, border_mode, border_value, dst)

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


def _warp_perspective_float16_nearest(
    img: ImageFloat16,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray | None,
    dst: ImageType | None,
) -> ImageType | None:
    if (
        not img.flags.c_contiguous
        or m.dtype != np.float32
        or flags != cv2.INTER_NEAREST
        or border_mode != cv2.BORDER_CONSTANT
        or border_value is not None
        or dst is not None
        or dsize[0] <= 0
        or dsize[1] <= 0
    ):
        return None

    source = _float16_nearest_integer_source(img)
    if source is None:
        return None

    channels = img.shape[-1]
    result = cv2.warpPerspective(
        source,
        m,
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=0,
    )
    return cast("ImageType", result.view(np.float16).reshape(dsize[1], dsize[0], channels))


@overload
def warp_perspective(
    img: ArrayT,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ArrayT | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ArrayT: ...


@overload
def warp_perspective(
    img: torch.Tensor,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> torch.Tensor: ...


@preserve_channel_dim
def warp_perspective(
    img: ImageType | torch.Tensor,
    m: np.ndarray,
    dsize: tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ImageType | torch.Tensor:
    """Perspective warp. Drop-in for cv2.warpPerspective with multi-channel support.

    OpenCV warpPerspective accepts >4 channels up to its encoded channel limit when:
    - Interpolation is INTER_NEAREST, INTER_LINEAR, or INTER_AREA
    - border_value is scalar or len <= 4

    We chunk when:
    - C exceeds OpenCV's encoded channel limit
    - C > 4 AND (flags in {INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR_EXACT} OR border_value_cv2 is None)
    - border_value_cv2 is None: per-channel border_value len>4 → _warp_perspective_chunked
    - border_value_cv2 is not None: uniform border_value → maybe_process_in_chunks

    Contiguous NumPy float16 nearest calls with a float32 matrix and the default zero constant border use a
    bit-preserving integer view for C=1/3/9. Other float16 controls and interpolation modes use the float32 path.

    Args:
        img: (H, W, C) image. uint8, float16, or float32.
        m: 3x3 perspective matrix.
        dsize: (width, height) output size.
        flags: Interpolation flags.
        border_mode: Border mode.
        border_value: Scalar, tuple, or array. Per-channel len>4 → chunked path.
        dst: Optional pre-allocated output array.
        layout: Required for Tensor input; selects CHW or NCHW.

    Returns:
        Warped image, shape (dsize[1], dsize[0], C).
    """
    result: np.ndarray | torch.Tensor
    if img.dtype == np.float16:
        dst = cast("ImageType | None", dst)
        nearest_result = _warp_perspective_float16_nearest(
            cast("ImageFloat16", img),
            m,
            dsize,
            flags,
            border_mode,
            border_value,
            dst,
        )
        if nearest_result is not None:
            return nearest_result
        result = warp_perspective(
            _cast_float_numpy(img, np.float32),
            m,
            dsize,
            flags,
            border_mode,
            border_value,
        )
        restored = _cast_float_numpy(result, np.float16)
        if dst is not None:
            np.copyto(dst, restored)
            return dst
        return restored

    if isinstance(img, torch.Tensor):
        matrix = tensor_operand_to_numpy(m, layout)
        dst_view = tensor_to_numpy_channel_last(dst, layout) if isinstance(dst, torch.Tensor) else dst
        result = dispatch_tensor_image(
            img,
            layout,
            warp_perspective,
            matrix,
            dsize,
            flags,
            border_mode,
            border_value,
            dst=dst_view,
        )
        return dst if isinstance(dst, torch.Tensor) else result

    dst = cast("ImageType | None", dst)

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


@overload
def copy_make_border(
    img: ArrayT,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int = cv2.BORDER_CONSTANT,
    value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ArrayT | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ArrayT: ...


@overload
def copy_make_border(
    img: torch.Tensor,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int = cv2.BORDER_CONSTANT,
    value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> torch.Tensor: ...


@preserve_channel_dim
def copy_make_border(
    img: ImageType | torch.Tensor,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int = cv2.BORDER_CONSTANT,
    value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ImageType | torch.Tensor:
    """Pad image with border. Drop-in for cv2.copyMakeBorder with multi-channel support.

    Chunks only when C > 4 AND value is per-channel (len>4, non-uniform). Otherwise
    uses cv2.copyMakeBorder directly. For BORDER_CONSTANT with scalar or len<=4,
    no chunking needed.

    Args:
        img: (H, W, C) uint8, float16, or float32 image.
        top: Padding in pixels on top.
        bottom: Padding in pixels on bottom.
        left: Padding in pixels on left.
        right: Padding in pixels on right.
        border_type: cv2.BORDER_CONSTANT, BORDER_REPLICATE, etc.
        value: Fill value for BORDER_CONSTANT. Scalar or per-channel array.
        layout: Required for Tensor input; selects CHW or NCHW.
        dst: Optional pre-allocated output array.

    Returns:
        Padded image, shape (H+top+bottom, W+left+right, C).
    """
    if isinstance(img, torch.Tensor):
        dst_view = tensor_to_numpy_channel_last(dst, layout) if isinstance(dst, torch.Tensor) else dst
        result = dispatch_tensor_image(
            img,
            layout,
            copy_make_border,
            top,
            bottom,
            left,
            right,
            border_type,
            value,
            dst=dst_view,
        )
        return dst if isinstance(dst, torch.Tensor) else result

    dst = cast("ImageType | None", dst)

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


def _remap_float16_nearest(
    img: ImageFloat16,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | np.ndarray | None,
    dst: ImageType | None,
) -> ImageType | None:
    if (
        not img.flags.c_contiguous
        or not map_x.flags.c_contiguous
        or not map_y.flags.c_contiguous
        or interpolation != cv2.INTER_NEAREST
        or border_mode != cv2.BORDER_CONSTANT
        or border_value is not None
        or dst is not None
    ):
        return None

    source = _float16_nearest_integer_source(img)
    if source is None:
        return None

    channels = img.shape[-1]
    output_shape = (*map_x.shape[:2], channels)
    remap_map_x, _ = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2, nninterpolation=True)
    # OpenCV stubs: https://github.com/opencv/opencv/issues/30296 (review 2026-11-15).
    result = cv2.remap(
        source,
        remap_map_x,
        cast("Any", None),
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=0,
    )
    return cast("ImageType", result.view(np.float16).reshape(output_shape))


@overload
def remap(
    img: ArrayT,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ArrayT | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ArrayT: ...


@overload
def remap(
    img: torch.Tensor,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> torch.Tensor: ...


@preserve_channel_dim
def remap(
    img: ImageType | torch.Tensor,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] | np.ndarray | None = None,
    dst: ImageType | torch.Tensor | None = None,
    *,
    layout: TensorImageLayout | None = None,
) -> ImageType | torch.Tensor:
    """Remap image. Drop-in for cv2.remap with multi-channel support.

    cv2.remap works for >4 channels up to its encoded channel limit when
    interpolation is NEAREST, LINEAR, or AREA and border_value is scalar or
    len<=4. We chunk when:
    - C exceeds OpenCV's encoded channel limit
    - C > 4 AND (interpolation in {CUBIC, LANCZOS4} OR per-channel border_value len>4)

    Contiguous NumPy float16 nearest remaps with the default zero border convert maps to fixed-point before using
    integer pixel views for C=1/3/9. This preserves float32 tie rounding; other float16 paths use float32 working data.

    Args:
        img: (H, W, C) uint8, float16, or float32 image.
        map_x: X-coordinate map, shape (H, W), float32.
        map_y: Y-coordinate map, shape (H, W), float32.
        interpolation: cv2.INTER_*.
        border_mode: cv2.BORDER_*.
        border_value: Scalar, tuple, or array. Per-channel len>4 triggers chunking.
        dst: Optional pre-allocated output array.
        layout: Required for Tensor input; selects CHW or NCHW.

    Returns:
        Remapped image, same shape as input.
    """
    result: np.ndarray | torch.Tensor
    if isinstance(img, torch.Tensor):
        map_x_numpy = tensor_operand_to_numpy(map_x, layout)
        map_y_numpy = tensor_operand_to_numpy(map_y, layout)
        dst_view = tensor_to_numpy_channel_last(dst, layout) if isinstance(dst, torch.Tensor) else dst
        result = dispatch_tensor_image(
            img,
            layout,
            remap,
            map_x_numpy,
            map_y_numpy,
            interpolation,
            border_mode,
            border_value,
            dst=dst_view,
        )
        return dst if isinstance(dst, torch.Tensor) else result

    dst = cast("ImageType | None", dst)

    if img.dtype == np.float16:
        nearest_result = _remap_float16_nearest(
            cast("ImageFloat16", img),
            map_x,
            map_y,
            interpolation,
            border_mode,
            border_value,
            dst,
        )
        if nearest_result is not None:
            return nearest_result
        result = remap(
            _cast_float_numpy(img, np.float32),
            map_x,
            map_y,
            interpolation,
            border_mode,
            border_value,
        )
        restored = _cast_float_numpy(result, np.float16)
        if dst is not None:
            np.copyto(dst, restored)
            return dst
        return restored

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


def _resize_float16_nearest(
    img: ImageType,
    dsize: tuple[int, int],
    fx: float,
    fy: float,
    interpolation: int,
) -> ImageType | None:
    if interpolation != cv2.INTER_NEAREST or dsize[0] <= 0 or dsize[1] <= 0:
        return None

    if not img.flags.c_contiguous:
        height, width, channels = img.shape[-3], img.shape[-2], img.shape[-1]
        has_dense_pixels = img.strides[-1] == img.itemsize and img.strides[-2] == channels * img.itemsize
        is_half_size = height % 2 == 0 and width % 2 == 0 and dsize == (width // 2, height // 2)
        if is_half_size and (channels == 1 or not has_dense_pixels):
            return img[::2, ::2].copy(order="C")
        if not has_dense_pixels:
            return None
    elif img.shape[-1] == 1:
        height, width = img.shape[-3], img.shape[-2]
        if height % 2 == 0 and width % 2 == 0 and dsize == (width // 2, height // 2):
            return img[::2, ::2].copy(order="C")

    # Nearest resize copies samples, so CV_16U can move each float16 bit unchanged.
    source_u16 = img.view(np.uint16)
    result_u16 = cv2.resize(source_u16, dsize, fx=fx, fy=fy, interpolation=interpolation)
    return cast("ImageType", result_u16.view(np.float16))


def _resize_tensor_float16(
    img: torch.Tensor,
    layout: TensorImageLayout | None,
    dsize: tuple[int, int],
    interpolation: int,
) -> torch.Tensor | None:
    if layout is None or not img.is_contiguous() or dsize[0] <= 0 or dsize[1] <= 0:
        return None

    channels = img.shape[0] if layout == "CHW" else img.shape[1]
    batched = img.unsqueeze(0) if layout == "CHW" else img
    output_size = (dsize[1], dsize[0])
    if (
        interpolation == cv2.INTER_NEAREST
        and img.numel() >= _RESIZE_TENSOR_FLOAT16_NEAREST_MIN_ELEMENTS
        and (layout == "NCHW" or channels >= 3)
    ):
        result = torch_f.interpolate(batched, size=output_size, mode="nearest")
        return result.squeeze(0) if layout == "CHW" else result

    if (
        interpolation == cv2.INTER_LINEAR
        and img.numel() >= _RESIZE_TENSOR_FLOAT16_LINEAR_MIN_ELEMENTS
        and channels >= 3
    ):
        result = torch_f.interpolate(
            batched.to(torch.float32),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).to(torch.float16)
        return result.squeeze(0) if layout == "CHW" else result
    return None


@overload
def resize(
    img: ArrayT,
    dsize: tuple[int, int],
    fx: float = 0.0,
    fy: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
    *,
    layout: TensorImageLayout | None = None,
) -> ArrayT: ...


@overload
def resize(
    img: torch.Tensor,
    dsize: tuple[int, int],
    fx: float = 0.0,
    fy: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
    *,
    layout: TensorImageLayout | None = None,
) -> torch.Tensor: ...


@preserve_channel_dim
def resize(
    img: ImageType | torch.Tensor,
    dsize: tuple[int, int],
    fx: float = 0.0,
    fy: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
    *,
    layout: TensorImageLayout | None = None,
) -> ImageType | torch.Tensor:
    """Resize image. Drop-in for cv2.resize with full multi-channel support.

    cv2.resize with INTER_AREA asserts cn <= 4 internally on downscale, so 5+ channel images
    being downscaled with INTER_AREA are processed in chunks of up to 4 channels.
    All other cases are passed directly to cv2.resize.
    Contiguous float16 NCHW and multi-channel CHW Tensors use native CPU nearest interpolation from 61,440 elements;
    single-channel CHW keeps the existing route. For linear interpolation, multi-channel Tensors use float32 working
    interpolation and cast back to float16. NumPy inputs use the float32 OpenCV path except for bit-preserving nearest
    routes.

    Args:
        img: (H, W, C) image. uint8, float16, or float32.
        dsize: (width, height) output size. Pass (0, 0) and use fx/fy for scale factors.
        fx: Scale factor along the horizontal axis. Used only when dsize is (0, 0).
        fy: Scale factor along the vertical axis. Used only when dsize is (0, 0).
        interpolation: Interpolation flag (cv2.INTER_LINEAR, INTER_NEAREST, INTER_CUBIC,
            INTER_AREA, INTER_LANCZOS4, etc.).
        layout: Required for Tensor input; selects CHW or NCHW.

    Returns:
        Resized image with same dtype and channel count as input.
    """
    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float16:
            half_result = _resize_tensor_float16(img, layout, dsize, interpolation)
            if half_result is not None:
                return half_result
        return dispatch_tensor_image(img, layout, resize, dsize, fx, fy, interpolation)

    if img.dtype == np.float16:
        nearest_result = _resize_float16_nearest(img, dsize, fx, fy, interpolation)
        if nearest_result is not None:
            return nearest_result
        result = resize(_cast_float_numpy(img, np.float32), dsize, fx, fy, interpolation)
        return _cast_float_numpy(result, np.float16)

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


_RESIZE3D_NUMPY_PER_SLICE_MAX_ELEMENTS = 1_000_000
_RESIZE3D_TORCH_NUMPY_BRIDGE_MIN_OUTPUT_ELEMENTS = 10_000
_RESIZE3D_XY_MIN_PLANE_ELEMENTS = 16_384


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

    resized: np.ndarray
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
    result = _cast_float_numpy(volume, np.float32)
    for axis, output_size in enumerate(size):
        result = _resize3d_numpy_linear_axis(result, axis, output_size)
    if volume.dtype == np.uint8:
        return np.asarray(np.minimum(result + np.float32(0.5), np.float32(255)).astype(np.uint8))
    if volume.dtype == np.float16:
        return _cast_float_numpy(result, np.float16)
    return result


def _resize3d_numpy_per_slice(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> np.ndarray:
    """Resize H/W per slice, then resize depth only when it changes."""
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
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


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


def _can_resize3d_xy_per_slice(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> bool:
    """Avoid tiny-plane loop overhead and fractional uint8 linear backend differences."""
    depth, height, width, channels = volume.shape
    return (
        depth > 1
        and size[0] == depth
        and size[1] < height
        and size[2] < width
        and height * width * channels >= _RESIZE3D_XY_MIN_PLANE_ELEMENTS
        and (
            volume.dtype == np.float32
            or interpolation == cv2.INTER_NEAREST
            or antialias
            or (height % size[1] == 0 and width % size[2] == 0)
        )
    )


def _resize3d_numpy(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int,
    antialias: bool,
) -> np.ndarray:
    """Choose the measured NumPy, OpenCV, or CPU Torch full route for one DHWC volume."""
    source_size = volume.shape[:3]
    if _can_resize3d_xy_per_slice(volume, size, interpolation, antialias):
        return _resize3d_numpy_per_slice(volume, size, interpolation, antialias)

    all_down = all(target_axis < source_axis for source_axis, target_axis in zip(source_size, size, strict=True))
    all_up = all(target_axis > source_axis for source_axis, target_axis in zip(source_size, size, strict=True))
    torch_compatible = _can_resize3d_numpy_torch(volume, interpolation, antialias)

    if interpolation == cv2.INTER_LINEAR and not antialias and size[0] == 1 and volume.shape[0] != 1:
        if volume.dtype == np.uint8 and torch_compatible:
            return _resize3d_numpy_torch_cpu(volume, size)
        return _resize3d_numpy_linear_three_pass(volume, size)

    if all_down and torch_compatible and (volume.dtype == np.float32 or volume.shape[-1] > 1):
        return _resize3d_numpy_torch_cpu(volume, size)
    if all_down and volume.shape[-1] == 1 and volume.size <= _RESIZE3D_NUMPY_PER_SLICE_MAX_ELEMENTS:
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
    restore_half = volume.dtype == torch.float16
    if restore_half:
        numpy_volume = _cast_float_numpy(numpy_volume, np.float32)
    resized = _resize3d_numpy(numpy_volume, size, interpolation, antialias=False)
    if restore_half:
        resized = _cast_float_numpy(resized, np.float16)
    return torch.from_numpy(resized).permute(3, 0, 1, 2)


def _resize3d_torch_cpu(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int,
) -> torch.Tensor:
    """Resize one CPU CDHW tensor with native Torch interpolation and no autograd graph."""
    with torch.no_grad():
        if interpolation == cv2.INTER_NEAREST:
            result = torch_f.interpolate(volume.unsqueeze(0), size=size, mode="nearest")
            return result.squeeze(0)

        working_volume = volume if volume.dtype in (torch.float16, torch.float32) else volume.to(torch.float32)
        result = torch_f.interpolate(working_volume.unsqueeze(0), size=size, mode="trilinear", align_corners=False)
        if volume.dtype == torch.uint8:
            result = torch.minimum(result + 0.5, result.new_tensor(255)).to(torch.uint8)
        return result.squeeze(0)


@overload
def resize3d(
    volume: ArrayT,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> ArrayT: ...


@overload
def resize3d(
    volume: torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> torch.Tensor: ...


def resize3d(
    volume: np.ndarray | torch.Tensor,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    antialias: bool = False,
) -> np.ndarray | torch.Tensor:
    """Resize one volume along its depth, height, and width axes.

    NumPy input uses channel-last ``(D, H, W, C)`` layout. Torch input uses channel-first
    ``(C, D, H, W)`` layout. AlbumentationsX validates the input layout, spatial dimensions, output size,
    device, and autograd state before calling this primitive. The container, dtype, and channel count are preserved
    for uint8, float16, and float32 inputs.
    Linear Tensor resize that enlarges all three spatial axes and produces at least 10,000 output elements uses the
    benchmark-selected NumPy/OpenCV CPU route. Float16 bridge inputs widen to float32 and round back once. Its float32
    output is within ``rtol=2e-4``, ``atol=3e-5`` of native Torch interpolation; uint8 differs by at most one value.
    Other Tensor regions use native Torch interpolation, including the CPU float16 kernel.

    Args:
        volume: Prevalidated NumPy ``DHWC`` array or Torch ``CDHW`` tensor with uint8, float16, or float32 data.
        size: Prevalidated output ``(depth, height, width)``.
        interpolation: ``cv2.INTER_LINEAR`` or ``cv2.INTER_NEAREST``.
        antialias: For NumPy linear interpolation, use ``INTER_AREA`` on shrinking axes. The caller must pass
            ``False`` for the Torch contract because Torch does not support antialiased trilinear interpolation.

    Returns:
        Resized volume in the same container and layout as ``volume``. An identity resize returns ``volume`` itself.

    The caller validates the container, layout, dtype, output size, interpolation,
    device, autograd state, and antialias contract before entering Albucore.
    """
    if isinstance(volume, np.ndarray):
        if volume.shape[:3] == size:
            return volume
        if volume.dtype == np.float16:
            float32_volume = _cast_float_numpy(volume, np.float32)
            result = _resize3d_numpy(float32_volume, size, interpolation, antialias)
            return _cast_float_numpy(result, np.float16)
        return _resize3d_numpy(volume, size, interpolation, antialias)

    if tuple(volume.shape[1:]) == size:
        return volume
    if _should_resize3d_torch_use_numpy_route(volume, size, interpolation):
        return _resize3d_torch_via_numpy(volume, size, interpolation)
    return _resize3d_torch_cpu(volume, size, interpolation)
