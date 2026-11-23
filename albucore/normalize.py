"""Fixed-statistics and per-image normalization."""

import struct
from functools import lru_cache
from typing import Any, Literal, cast, overload

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import NormalizationType, ValueType
from albucore.lut import _apply_float_lut
from albucore.stats import DEFAULT_EPS, _mean_std_global_float32_opencv, mean_std
from albucore.tensor_bridge import (
    TensorLayout,
    numpy_to_tensor_layout,
    tensor_operand_to_numpy,
    tensor_to_numpy_channel_last,
)
from albucore.torch_backend import normalize_float16_torch
from albucore.utils import (
    _FLOAT32_MAX,
    _MACOS_ARM64,
    _TENSOR_CHANNEL_AXES,
    MAX_OPENCV_WORKING_CHANNELS,
    _batch_operand,
    _broadcast_channel_vector,
    convert_value,
    get_num_channels,
)

_TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS = 16_384
_FLOAT16_MAX = float(np.finfo(np.float16).max)
_FLOAT16_STRUCT = struct.Struct("e")
_FLOAT32_STRUCT = struct.Struct("f")
_TENSOR_BATCH_FLOAT32_IMAGE_MAX_ELEMENTS = 2_000_000
_TENSOR_BATCH_FLOAT32_MIN_ELEMENTS = 20_480


def _single_channel_stats(img: NDArray[np.float32]) -> tuple[float, float]:
    if img.ndim > 3:
        center, std = cv2.meanStdDev(img)
        return float(center[0, 0]), float(std[0, 0]) + DEFAULT_EPS
    return float(img.mean()), float(img.std()) + DEFAULT_EPS


def _channel_stats(img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]) -> tuple[np.ndarray, np.ndarray]:
    center, std = mean_std(img, "per_channel", eps=DEFAULT_EPS)
    return np.asarray(center, dtype=np.float64), np.asarray(std, dtype=np.float64)


def _center_scale_numpy(
    img: NDArray[np.float32],
    center: float | np.float32 | np.ndarray,
    divisor: float | np.float32 | np.ndarray,
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    result = np.empty_like(img) if out is None else out
    np.subtract(img, center, out=result)
    np.divide(result, divisor, out=result)
    return np.clip(result, -20, 20, out=result)


def _center_scale_cv(
    img: NDArray[np.float32],
    center: float | np.ndarray,
    divisor: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    if img.ndim > 3 or img.shape[-1] > MAX_OPENCV_WORKING_CHANNELS:
        return _center_scale_numpy(
            img,
            np.asarray(center, dtype=np.float32),
            np.asarray(divisor, dtype=np.float32),
            out=out,
        )
    if out is not None and not out.flags.c_contiguous:
        np.copyto(out, _center_scale_cv(img, center, divisor, out=None))
        return out
    result = np.empty(img.shape, dtype=np.float32) if out is None else out
    target = result if out is None else result.reshape(-1).reshape(img.shape)
    center_cv: Any = center if np.isscalar(center) else np.asarray(center, dtype=np.float32)
    divisor_cv: Any = divisor if np.isscalar(divisor) else np.asarray(divisor, dtype=np.float32)
    cv2.subtract(img, center_cv, dst=target, dtype=cv2.CV_32F)
    cv2.divide(target, divisor_cv, dst=target, dtype=cv2.CV_32F)
    return np.clip(result, -20, 20, out=result)


def _min_max_per_channel(img: NDArray[np.float32]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.ascontiguousarray(img).reshape(-1, img.shape[-1])
    flat_cv = cast("Any", flat)
    lower = cv2.reduce(flat_cv, 0, cv2.REDUCE_MIN, dtype=cv2.CV_32F).reshape(-1)
    upper = cv2.reduce(flat_cv, 0, cv2.REDUCE_MAX, dtype=cv2.CV_32F).reshape(-1)
    return lower, upper


def _global_min_max(
    img: NDArray[np.uint8] | NDArray[np.float32],
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    if out is not None and not out.flags.c_contiguous:
        np.copyto(out, _global_min_max(img, out=None))
        return out
    result = np.empty(img.shape, dtype=np.float32) if out is None else out
    target = result if out is None else result.reshape(-1).reshape(img.shape)
    cv2.normalize(img, target, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return result


def _adaptive_float32(
    img: NDArray[np.float32],
    normalization: NormalizationType,
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    if normalization == "image":
        cv_stats = _mean_std_global_float32_opencv(img, DEFAULT_EPS)
        if cv_stats is None:
            center = img.mean()
            divisor = img.std() + DEFAULT_EPS
        else:
            center = np.float32(cv_stats[0])
            divisor = np.float32(cv_stats[1])
        return _center_scale_numpy(img, center, divisor, out=out)
    if normalization == "image_per_channel":
        if img.shape[-1] == 1:
            center_value, std_value = _single_channel_stats(img)
            return _center_scale_cv(img, center_value, std_value, out=out)
        centers, stds = _channel_stats(img)
        return _center_scale_cv(img, centers, stds, out=out)
    if normalization == "min_max" or img.shape[-1] == 1:
        return _global_min_max(img, out=out)
    lower, upper = _min_max_per_channel(img)
    return _center_scale_cv(img, lower, upper - lower + DEFAULT_EPS, out=out)


def _adaptive_half(
    img: NDArray[np.float16],
    normalization: NormalizationType,
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16]:
    working = _cast_float_numpy(img, np.float32)
    reuse_working = normalization != "image" and working.flags.c_contiguous
    result = _adaptive_float32(working, normalization, out=working if reuse_working else None)
    del working
    return _cast_float_numpy(result, img.dtype if out is None else out.dtype, out=out)


def _adaptive_uint8(
    img: NDArray[np.uint8],
    normalization: NormalizationType,
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    if normalization == "min_max":
        return _global_min_max(img, out=out)
    channels = img.shape[-1]
    values = np.arange(256, dtype=np.float32)
    if normalization == "image" or (normalization == "image_per_channel" and channels == 1):
        center, std = mean_std(img, "global", eps=DEFAULT_EPS)
        table = (values - float(center)) / float(std)
    elif normalization == "image_per_channel":
        centers, stds = _channel_stats(img)
        table = (values.reshape(-1, 1, 1) - centers.astype(np.float32).reshape(1, 1, channels)) / stds.astype(
            np.float32,
        ).reshape(1, 1, channels)
    elif channels == 1:
        lower_value, upper_value = np.min(img), np.max(img)
        table = (values - lower_value) / (upper_value - lower_value + DEFAULT_EPS)
    else:
        axes = tuple(range(img.ndim - 1))
        lower = np.min(img, axis=axes).astype(np.float32).reshape(1, 1, channels)
        upper = np.max(img, axis=axes).astype(np.float32).reshape(1, 1, channels)
        table = (values.reshape(-1, 1, 1) - lower) / (upper - lower + DEFAULT_EPS)
    bounds = (-20, 20) if normalization in ("image", "image_per_channel") else (0, 1)
    return _apply_float_lut(img, np.clip(table, *bounds).astype(np.float32, copy=False), out=out)


def _adaptive_tensor_batch_float32(
    img: torch.Tensor,
    normalization: NormalizationType,
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    source = img.numpy()
    per_channel = normalization == "image_per_channel" or (
        normalization == "min_max_per_channel" and source.shape[1] > 1
    )
    axes = tuple(range(2 if per_channel else 1, source.ndim))
    if normalization in ("image", "image_per_channel"):
        center = source.mean(axis=axes, keepdims=True)
        std = source.std(axis=axes, keepdims=True)
        if (
            not np.isfinite(center).all()
            or not np.isfinite(std).all()
            or np.any(np.square(std.astype(np.float64)) <= np.float64(1e-4) * np.square(center.astype(np.float64)))
        ):
            return None
        divisor = std + DEFAULT_EPS
        lower, upper = -20, 20
    else:
        center = source.min(axis=axes, keepdims=True)
        divisor = source.max(axis=axes, keepdims=True) - center
        if per_channel:
            divisor += DEFAULT_EPS
        if not np.isfinite(center).all() or not np.isfinite(divisor).all():
            return None
        lower, upper = 0, 1
    result = np.empty_like(source) if out is None else out.numpy()
    np.subtract(source, center, out=result)
    if normalization.startswith("min_max"):
        np.divide(result, divisor, out=result, where=divisor != 0)
    else:
        np.divide(result, divisor, out=result)
    np.clip(result, lower, upper, out=result)
    if out is None:
        return torch.from_numpy(result)
    torch.autograd.graph.increment_version(out)
    return out


def _use_tensor_batch_float32_route(
    img: torch.Tensor,
    normalization: NormalizationType,
    layout: Literal["NCHW", "NCDHW"],
    out: torch.Tensor | None,
) -> bool:
    if (
        img.dtype != torch.float32
        or not img.is_contiguous()
        or img.shape[0] < 4
        or (out is not None and not out.is_contiguous())
    ):
        return False

    channels = img.shape[1]
    elements = img.numel()
    if normalization == "image":
        return (
            channels in (3, 9)
            and _TENSOR_BATCH_FLOAT32_MIN_ELEMENTS <= elements <= _TENSOR_BATCH_FLOAT32_IMAGE_MAX_ELEMENTS
        )
    if elements < _TENSOR_BATCH_FLOAT32_MIN_ELEMENTS:
        return False
    if normalization == "image_per_channel":
        return (layout == "NCHW" and channels in (1, 3)) or (_MACOS_ARM64 and layout == "NCDHW" and channels == 3)
    return (normalization == "min_max" and channels == 3) or (
        normalization == "min_max_per_channel" and channels in (3, 9)
    )


def _adaptive_tensor_batch(
    img: torch.Tensor,
    normalization: NormalizationType,
    layout: Literal["NCHW", "NCDHW"],
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if _use_tensor_batch_float32_route(img, normalization, layout, out):
        batch_result = _adaptive_tensor_batch_float32(img, normalization, out)
        if batch_result is not None:
            return batch_result

    item_layout: TensorLayout = "CHW" if layout == "NCHW" else "CDHW"
    if out is not None:
        for index, item in enumerate(img):
            normalize_per_image(item, normalization, out=out[index], layout=item_layout)
        return out
    image = tensor_to_numpy_channel_last(img, layout)
    first = normalize_per_image(image[0], normalization)
    storage = np.empty(image.shape[0] * first.size, dtype=first.dtype)
    result: NDArray[np.float16] | NDArray[np.float32] = np.ndarray(
        image.shape,
        dtype=first.dtype,
        buffer=storage,
        strides=(first.nbytes, *first.strides),
    )
    np.copyto(result[0], first)
    del first
    for index in range(1, image.shape[0]):
        normalize_per_image(image[index], normalization, out=result[index])
    return numpy_to_tensor_layout(result, layout)


def _adaptive_tensor(
    img: torch.Tensor,
    normalization: NormalizationType,
    layout: TensorLayout,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if img.numel() == 0:
        return img.to(torch.float32) if out is None and img.dtype == torch.uint8 else img if out is None else out
    if layout in ("NCHW", "NCDHW"):
        return _adaptive_tensor_batch(img, normalization, layout, out=out)
    image = tensor_to_numpy_channel_last(img, layout)
    destination = None if out is None else image if out is img else tensor_to_numpy_channel_last(out, layout)
    direct_out = destination if destination is not None and destination.flags.c_contiguous else None
    result = normalize_per_image(image, normalization, out=direct_out)
    if out is not None and result is destination:
        return out
    tensor = numpy_to_tensor_layout(result, layout)
    if out is not None:
        out.copy_(tensor)
        return out
    return tensor


@overload
def normalize_per_image(
    img: NDArray[np.float16],
    normalization: NormalizationType,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def normalize_per_image(
    img: NDArray[np.uint8] | NDArray[np.float32],
    normalization: NormalizationType,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def normalize_per_image(
    img: torch.Tensor,
    normalization: NormalizationType,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def normalize_per_image(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    normalization: NormalizationType,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Normalize from global or per-channel statistics with explicit output storage.

    Modes are image/image_per_channel (mean/std, clipped to [-20,20]) and
    min_max/min_max_per_channel. Float16 returns float16; uint8/float32 return float32.
    Floating NumPy inputs retain their byte order.
    NumPy global modes span the whole array. Tensor NCHW/NCDHW normalize each batch
    item independently; CHW/CDHW span one image or volume. Tensor layout is required.

    Without out, input storage is unchanged. A supplied result-dtype output is returned
    by identity and may alias a floating input. AX validates metadata and overlap.
    Empty inputs return empty results without estimating statistics. Half uses float32
    working values. Global min/max has no epsilon; per-channel min/max adds epsilon
    except for floating C=1. Use normalize for fixed caller constants.
    """
    if isinstance(img, torch.Tensor):
        return _adaptive_tensor(img, normalization, cast("TensorLayout", layout), out=cast("torch.Tensor | None", out))
    if img.size == 0:
        return (
            cast(
                "NDArray[np.float16] | NDArray[np.float32]",
                img.astype(np.float32 if img.dtype == np.uint8 else img.dtype, copy=False),
            )
            if out is None
            else out
        )
    if img.dtype == np.uint8:
        return _adaptive_uint8(
            cast("NDArray[np.uint8]", img),
            normalization,
            out=cast("NDArray[np.float32] | None", out),
        )
    if img.dtype.type is np.float16:
        return _adaptive_half(
            cast("NDArray[np.float16]", img),
            normalization,
            out=cast("NDArray[np.float16] | None", out),
        )
    if not img.dtype.isnative:
        image = cast("NDArray[np.float32]", img)
        working = _cast_float_numpy(image, np.float32)
        if normalization == "image":
            result = _center_scale_numpy(working, working.mean(), working.std() + DEFAULT_EPS, out=None)
        else:
            result = _adaptive_float32(working, normalization, out=None)
        del working
        return _cast_float_numpy(result, image.dtype, out=cast("NDArray[np.float32] | None", out))
    return _adaptive_float32(
        cast("NDArray[np.float32]", img),
        normalization,
        out=cast("NDArray[np.float32] | None", out),
    )


def _centered_numpy(
    img: NDArray[np.float16] | NDArray[np.float32],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Compute ``(img - mean) * denominator`` in float32; subtracting first keeps a pixel at the mean exactly zero."""
    mean_f = mean.astype(np.float32, copy=False) if isinstance(mean, np.ndarray) else np.float32(mean)
    denom_f = (
        denominator.astype(np.float32, copy=False) if isinstance(denominator, np.ndarray) else np.float32(denominator)
    )
    if isinstance(mean_f, np.ndarray) or isinstance(denom_f, np.ndarray):
        mean_operand = _broadcast_channel_vector(mean_f, img.ndim, img.shape[-1])
        denominator_operand = _broadcast_channel_vector(denom_f, img.ndim, img.shape[-1])
    else:
        mean_operand, denominator_operand = mean_f, denom_f
    result = cast("NDArray[np.float32]", np.subtract(img, mean_operand, dtype=np.float32, out=out))
    np.multiply(result, denominator_operand, out=result)
    return result


def _channel_vector_float32(value: float | np.ndarray, channels: int) -> NDArray[np.float32] | None:
    """Return a ``(channels,)`` float32 control, or ``None`` for full-array controls."""
    if not isinstance(value, np.ndarray):
        return np.full(channels, value, dtype=np.float32)
    return value.astype(np.float32, copy=False) if value.shape == (channels,) else None


def _normalize_float32(
    img: NDArray[np.float32],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Route centered float32 normalization by measured speed; every route subtracts before scaling."""
    channels = img.shape[-1]
    contiguous = img.flags.c_contiguous and (out is None or out.flags.c_contiguous)
    # Scalar controls are fastest as two NumPy passes; channel vectors broadcast slowly over a short last axis.
    if contiguous and channels > 1 and (isinstance(mean, np.ndarray) or isinstance(denominator, np.ndarray)):
        mean_vector = _channel_vector_float32(mean, channels)
        denominator_vector = _channel_vector_float32(denominator, channels)
        if mean_vector is not None and denominator_vector is not None:
            if channels <= MAX_OPENCV_WORKING_CHANNELS:
                return _centered_opencv(img, mean_vector, denominator_vector, out=out)
            return _centered_numpy_rows(img, mean_vector, denominator_vector, out=out)
    return _centered_numpy(img, mean, denominator, out=out)


def _centered_opencv(
    img: NDArray[np.float32],
    mean: NDArray[np.float32],
    denominator: NDArray[np.float32],
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Apply per-channel OpenCV scalar subtract and multiply to a contiguous 2-4 channel image.

    Flattening first gives size-1 axes canonical strides, which OpenCV requires for ``dst``.
    """
    padding = (0.0,) * (4 - img.shape[-1])
    shape = (-1, img.shape[-2], img.shape[-1])
    rows = img.reshape(-1).reshape(shape)
    result = cv2.subtract(rows, (*mean.tolist(), *padding), dst=None if out is None else out.reshape(-1).reshape(shape))
    cv2.multiply(result, (*denominator.tolist(), *padding), dst=result)
    return out if out is not None else cast("NDArray[np.float32]", result.reshape(img.shape))


def _centered_numpy_rows(
    img: NDArray[np.float32],
    mean: NDArray[np.float32],
    denominator: NDArray[np.float32],
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Subtract and scale contiguous rows by controls repeated across the width, a long contiguous inner axis."""
    width = img.shape[-2]
    rows = img.reshape(-1, width * img.shape[-1])
    destination = None if out is None else out.reshape(rows.shape)
    result = np.subtract(rows, np.tile(mean, width), out=destination)
    np.multiply(result, np.tile(denominator, width), out=result)
    return out if out is not None else cast("NDArray[np.float32]", result.reshape(img.shape))


def _round_to(value: float, layout: struct.Struct) -> float:
    """Round a Python float to the struct's float16 or float32 format without NumPy scalar overhead."""
    return cast("float", layout.unpack(layout.pack(value))[0])


def _float16_scalar_transform_kernel(
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
) -> tuple[NDArray[np.float32], bool] | None:
    """Return the ``[[denominator, bias]]`` kernel for scalar controls and whether its mean output is zero."""
    if isinstance(mean, np.ndarray):
        if mean.size != 1:
            return None
        mean = float(mean.item())
    if isinstance(denominator, np.ndarray):
        if denominator.size != 1:
            return None
        denominator = float(denominator.item())
    return _scalar_transform_kernel(float(mean), float(denominator))


@lru_cache(maxsize=64)
def _scalar_transform_kernel(mean: float, denominator: float) -> tuple[NDArray[np.float32], bool] | None:
    """Build the read-only kernel; caching removes about 1 µs per call that dominates small images.

    A fused multiply-add returns the float32 bias rounding error, not zero, for a pixel equal to the mean.
    """
    # Written as ``not <=`` so NaN is rejected together with infinity and float32 overflow.
    if not (abs(mean) <= _FLOAT32_MAX and abs(denominator) <= _FLOAT32_MAX):
        return None
    mean32, denominator32 = _round_to(mean, _FLOAT32_STRUCT), _round_to(denominator, _FLOAT32_STRUCT)
    # Exact in float64: two 24-bit significands; matches the other routes' float32 control rounding.
    product = mean32 * denominator32
    if abs(product) > _FLOAT32_MAX:
        return None
    bias = -_round_to(product, _FLOAT32_STRUCT)
    center_exact = bias == -product or abs(mean32) > _FLOAT16_MAX or _round_to(mean32, _FLOAT16_STRUCT) != mean32
    kernel = np.array([[denominator32, bias]], dtype=np.float32)
    kernel.flags.writeable = False
    return kernel, center_exact


def _normalize_float16_opencv_transform(
    img: NDArray[np.float16],
    kernel: NDArray[np.float32],
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16]:
    """Apply a scalar affine kernel with one CV16F ``transform`` over a one-channel 2D view.

    The 2D view covers any rank and channel count and drops size-1 axes whose strides OpenCV rejects.
    ``transform`` never multiplies a second source by zero, so infinite pixels stay infinite.
    """
    rows = img.reshape(img.shape[0], -1)
    if out is not None and out.flags.c_contiguous:
        cv2.transform(rows, kernel, dst=out.reshape(rows.shape))
        result = out
    else:
        result = cast("NDArray[np.float16]", cv2.transform(rows, kernel).reshape(img.shape))
        if out is not None:
            np.copyto(out, result)
            result = out
    return result


def _normalize_float16_channel_values(value: float | np.ndarray, channels: int) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != channels:
            return None
        values = value.astype(np.float32, copy=False)
    else:
        values = np.full(channels, np.float32(value), dtype=np.float32)
    return values if np.isfinite(values).all() else None


def _normalize_float16_tensor_controls(
    mean: object,
    denominator: object,
    channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mean_values = _normalize_float16_channel_values(convert_value(mean, channels), channels)
    denominator_values = _normalize_float16_channel_values(convert_value(denominator, channels), channels)
    if mean_values is None or denominator_values is None:
        return None
    biases = -mean_values * denominator_values
    if not np.isfinite(biases).all():
        return None
    return mean_values, denominator_values, biases


def _normalize_float16_tensor_opencv(
    img: torch.Tensor,
    mean: object,
    denominator: object,
    layout: TensorLayout | None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Normalize measured channel-first half Tensors through one CV_16F call per plane."""
    if layout not in ("CHW", "NCHW") or not img.is_contiguous() or img.numel() < _TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS:
        return None

    channel_axis = _TENSOR_CHANNEL_AXES[layout]
    channels = img.shape[channel_axis]
    if channels not in (3, 9):
        return None
    controls = _normalize_float16_tensor_controls(mean, denominator, channels)
    if controls is None:
        return None
    mean_values, denominator_values, biases = controls
    if _normalize_is_identity(mean_values, denominator_values):
        if out is not None and out is not img:
            out.copy_(img)
        return img if out is None else out

    source = img.numpy()
    result = out if out is not None and out.is_contiguous() else torch.empty_like(img)
    _normalize_float16_opencv_planes(source, result.numpy(), denominator_values, biases, layout)
    if out is not None:
        if result is not out:
            out.copy_(result)
        return out
    return result


def _normalize_float16_opencv_planes(
    source: np.ndarray,
    destination: np.ndarray,
    denominator: np.ndarray,
    biases: np.ndarray,
    layout: Literal["CHW", "NCHW"],
) -> None:
    zero_plane = np.zeros(source.shape[-2:], dtype=np.float16)
    source_batches = source[None, ...] if layout == "CHW" else source
    destination_batches = destination[None, ...] if layout == "CHW" else destination
    for batch in range(source_batches.shape[0]):
        for channel in range(source_batches.shape[1]):
            cv2.addWeighted(
                source_batches[batch, channel],
                float(denominator[channel]),
                zero_plane,
                0.0,
                float(biases[channel]),
                dst=destination_batches[batch, channel],
            )


def _affine_lut(
    img: NDArray[np.uint8],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    if img.size == 0:
        return np.empty_like(img, dtype=np.float32) if out is None else out
    x = np.arange(256, dtype=np.float32)
    if isinstance(denominator, (float, int)) and isinstance(mean, (float, int)):
        return _apply_float_lut(img, (x - mean) * denominator, out=out)

    x_lut = x.reshape(-1, 1, 1)
    mean_lut = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    denominator_lut = np.asarray(denominator, dtype=np.float32).reshape(1, 1, -1)
    return _apply_float_lut(img, (x_lut - mean_lut) * denominator_lut, out=out)


def _within_identity_tolerance(value: np.ndarray, target: float, eps: float) -> bool:
    if value.size == 0:
        return True
    return abs(float(value.flat[0]) - target) <= eps and (
        value.size == 1 or bool(np.all(np.abs(value.astype(np.float64, copy=False) - target) <= eps))
    )


def _normalize_is_identity(mean: float | np.ndarray, denominator: float | np.ndarray, *, eps: float = 1e-5) -> bool:
    if isinstance(mean, np.ndarray):
        if not _within_identity_tolerance(mean, 0.0, eps):
            return False
    elif abs(float(mean)) > eps:
        return False
    if isinstance(denominator, np.ndarray):
        if not _within_identity_tolerance(denominator, 1.0, eps):
            return False
    elif abs(float(denominator) - 1.0) > eps:
        return False
    return True


def _normalize_half(
    img: NDArray[np.float16],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16]:
    scalar = _float16_scalar_transform_kernel(mean, denominator)
    if scalar is not None:
        kernel, center_exact = scalar
        if not center_exact:
            # NumPy subtracts the mean before scaling; the fused OpenCV and Torch kernels do not.
            return _cast_float_numpy(_centered_numpy(img, mean, denominator), np.float16, out=out)
        if img.flags.c_contiguous:
            return _normalize_float16_opencv_transform(img, kernel, out=out)
    half_result = normalize_float16_torch(
        img,
        mean,
        denominator,
        minimum_elements=_TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS,
        out=out,
    )
    return (
        half_result
        if half_result is not None
        else _cast_float_numpy(_centered_numpy(img, mean, denominator), np.float16, out=out)
    )


def _normalize_batch(
    image: np.ndarray,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    mean = convert_value(mean, image.shape[-1])
    denominator = convert_value(denominator, image.shape[-1])
    if image.dtype == np.float32 and _normalize_is_identity(mean, denominator):
        if out is not None and out is not image:
            np.copyto(out, image)
        return cast("NDArray[np.float32]", image) if out is None else out
    preserve_order = image.dtype == np.float32 or (
        (isinstance(mean, np.ndarray) or isinstance(denominator, np.ndarray))
        and image.shape[-1] <= MAX_OPENCV_WORKING_CHANNELS
    )
    result = np.empty_like(image, dtype=np.float32, order="K" if preserve_order else "C") if out is None else out
    for index, item in enumerate(image):
        normalize(
            item,
            _batch_operand(mean, image, index),
            _batch_operand(denominator, image, index),
            out=result[index],
        )
    return result


def _normalize_uint8_tensor_storage(
    img: torch.Tensor,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor | None:
    """Translate contiguous uint8 storage through one shared table; channel layout does not matter."""
    if not (
        img.dtype == torch.uint8
        and np.ndim(mean) == 0
        and np.ndim(denominator) == 0
        and img.numel()
        and img.is_contiguous()
        and (out is None or out.is_contiguous())
    ):
        return None
    table = (np.arange(256, dtype=np.float32) - float(mean)) * float(denominator)
    rows = img.numpy().reshape(-1, img.shape[-1])
    if out is None:
        return torch.from_numpy(cv2.LUT(rows, table).reshape(img.shape))
    cv2.LUT(rows, table, dst=out.numpy().reshape(rows.shape))
    torch.autograd.graph.increment_version(out)
    return out


def _normalize_tensor(
    img: torch.Tensor,
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    layout: TensorLayout,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    mean_value = tensor_operand_to_numpy(mean, layout)
    denominator_value = tensor_operand_to_numpy(denominator, layout)
    if img.dtype != torch.uint8 and _normalize_is_identity(mean_value, denominator_value):
        if out is not None and out is not img:
            out.copy_(img)
        return img if out is None else out
    if img.dtype == torch.float16:
        direct_result = _normalize_float16_tensor_opencv(img, mean_value, denominator_value, layout, out=out)
        if direct_result is not None:
            return direct_result
    storage_result = _normalize_uint8_tensor_storage(img, mean_value, denominator_value, out=out)
    if storage_result is not None:
        return storage_result
    image = tensor_to_numpy_channel_last(img, layout)
    destination = None if out is None else image if out is img else tensor_to_numpy_channel_last(out, layout)
    kernel_out = destination
    if img.dtype == torch.uint8 and destination is not None and not destination.flags.c_contiguous:
        kernel_out = None
    if img.dtype != torch.float16 and layout in ("NCHW", "NCDHW") and img.numel() != 0:
        result = _normalize_batch(image, mean_value, denominator_value, out=kernel_out)
    else:
        result = normalize(image, mean_value, denominator_value, out=kernel_out)
    if out is not None and result is destination:
        return out
    tensor = img if result is image else numpy_to_tensor_layout(result, layout)
    if out is not None:
        out.copy_(tensor)
        return out
    return tensor


@overload
def normalize(
    img: NDArray[np.float16],
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def normalize(
    img: NDArray[np.uint8] | NDArray[np.float32],
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def normalize(
    img: torch.Tensor,
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def normalize(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Normalize with fixed caller constants as ``(img - mean) * denominator``.

    Float16 input returns float16; uint8 and float32 return float32. NumPy arrays are
    channel-last. Tensor calls require CHW, NCHW, CDHW or NCDHW layout. Float32 results
    subtract the mean before scaling, so a pixel equal to the mean returns exact zero.

    Without ``out``, input storage is unchanged; approximate identity controls may
    return it directly. A supplied output matches the result dtype, shape and container
    and is returned by identity. Exact floating source aliases are supported; AX
    validates other overlap and destination metadata. Strided output may need a temporary.

    Use ``normalize_per_image`` to compute adaptive statistics from the input instead.
    """
    if (
        out is img
        and isinstance(mean, (float, int))
        and isinstance(denominator, (float, int))
        and _normalize_is_identity(mean, denominator)
    ):
        return cast("NDArray[np.float16] | NDArray[np.float32] | torch.Tensor", img)
    if isinstance(img, torch.Tensor):
        return _normalize_tensor(
            img,
            mean,
            denominator,
            cast("TensorLayout", layout),
            out=cast("torch.Tensor | None", out),
        )
    num_channels = get_num_channels(img)
    denominator = convert_value(denominator, num_channels)
    mean = convert_value(mean, num_channels)
    if img.dtype == np.uint8:
        return _affine_lut(
            cast("NDArray[np.uint8]", img),
            mean,
            denominator,
            out=cast("NDArray[np.float32] | None", out),
        )
    if img.size == 0 or _normalize_is_identity(mean, denominator):
        if out is not None and out is not img:
            np.copyto(cast("np.ndarray", out), img)
        return cast("NDArray[np.float16] | NDArray[np.float32]", img) if out is None else out
    if img.dtype == np.float16:
        return _normalize_half(
            cast("NDArray[np.float16]", img),
            mean,
            denominator,
            out=cast("NDArray[np.float16] | None", out),
        )
    return _normalize_float32(
        cast("NDArray[np.float32]", img),
        mean,
        denominator,
        out=cast("NDArray[np.float32] | None", out),
    )
