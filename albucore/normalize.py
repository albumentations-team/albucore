"""Fixed-statistics and per-image normalization."""

import math
from typing import Any, Literal, cast, overload

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import ImageType, NormalizationType, ValueType
from albucore.lut import _apply_float_lut
from albucore.stats import DEFAULT_EPS, mean_std
from albucore.tensor_bridge import (
    _CHANNEL_LAST_TO_TENSOR,
    TensorLayout,
    tensor_operand_to_numpy,
    tensor_to_numpy_channel_last,
)
from albucore.torch_backend import normalize_float16_torch, normalize_float32_torch
from albucore.utils import (
    _FLOAT32_MAX,
    MAX_OPENCV_WORKING_CHANNELS,
    _batch_operand,
    _broadcast_channel_vector,
    convert_value,
    get_num_channels,
)

_TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS = 16_384


def _single_channel_stats(img: NDArray[np.float32]) -> tuple[float, float]:
    if img.ndim > 3:
        center, std = cv2.meanStdDev(img)
        return float(center[0, 0]), float(std[0, 0]) + DEFAULT_EPS
    return float(img.mean()), float(img.std()) + DEFAULT_EPS


def _channel_stats(img: ImageType) -> tuple[np.ndarray, np.ndarray]:
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
        center = img.mean()
        divisor = img.std() + DEFAULT_EPS
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


def _adaptive_tensor_batch(
    img: torch.Tensor,
    normalization: NormalizationType,
    layout: Literal["NCHW", "NCDHW"],
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
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
    return torch.from_numpy(result).permute(_CHANNEL_LAST_TO_TENSOR[layout])


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
    tensor = torch.from_numpy(result).permute(_CHANNEL_LAST_TO_TENSOR[layout])
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
    img: ImageType | torch.Tensor,
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
        result = _adaptive_float32(working, normalization, out=None)
        del working
        return _cast_float_numpy(result, image.dtype, out=cast("NDArray[np.float32] | None", out))
    return _adaptive_float32(
        cast("NDArray[np.float32]", img),
        normalization,
        out=cast("NDArray[np.float32] | None", out),
    )


def _affine_numpy(
    img: ImageType,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    mean_f = mean.astype(np.float32, copy=False) if isinstance(mean, np.ndarray) else np.float32(mean)
    denom_f = (
        denominator.astype(np.float32, copy=False) if isinstance(denominator, np.ndarray) else np.float32(denominator)
    )
    if isinstance(mean_f, np.ndarray) or isinstance(denom_f, np.ndarray):
        mean_b = _broadcast_channel_vector(mean_f, img.ndim, img.shape[-1])
        denom_b = _broadcast_channel_vector(denom_f, img.ndim, img.shape[-1])
        offset = mean_b * denom_b
        result = cast("NDArray[np.float32]", np.multiply(img, denom_b, dtype=np.float32, out=out))
        np.subtract(result, offset, out=result)
        return result

    result = cast("NDArray[np.float32]", np.multiply(img, denom_f, dtype=np.float32, out=out))
    np.subtract(result, mean_f * denom_f, out=result)
    return result


def _normalize_float16_single_channel_opencv(
    img: NDArray[np.float16],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    if isinstance(mean, np.ndarray):
        if mean.size != 1:
            return None
        mean_value = float(mean.item())
    else:
        mean_value = float(mean)
    if isinstance(denominator, np.ndarray):
        if denominator.size != 1:
            return None
        denominator_value = float(denominator.item())
    else:
        denominator_value = float(denominator)
    if (
        not math.isfinite(mean_value)
        or abs(mean_value) > _FLOAT32_MAX
        or not math.isfinite(denominator_value)
        or abs(denominator_value) > _FLOAT32_MAX
    ):
        return None
    bias = -mean_value * denominator_value
    if abs(bias) > _FLOAT32_MAX:
        return None
    if out is not None and out.flags.c_contiguous:
        cv2.addWeighted(img, denominator_value, img, 0.0, bias, dst=out.reshape(-1).reshape(img.shape))
        result = out
    else:
        result = cast("NDArray[np.float16]", cv2.addWeighted(img, denominator_value, img, 0.0, bias).reshape(img.shape))
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
    if (
        img.dtype != torch.float16
        or layout not in ("CHW", "NCHW")
        or not img.is_contiguous()
        or img.numel() < _TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS
    ):
        return None

    channel_axis = 0 if layout == "CHW" else 1
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
    half_result = normalize_float16_torch(
        img,
        mean,
        denominator,
        minimum_elements=_TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS,
        out=out,
    )
    if half_result is None and img.ndim == 3 and img.shape[-1] == 1 and img.flags.c_contiguous:
        half_result = _normalize_float16_single_channel_opencv(img, mean, denominator, out=out)
    return (
        half_result
        if half_result is not None
        else _cast_float_numpy(_affine_numpy(img, mean, denominator), np.float16, out=out)
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
    tensor = img if result is image else torch.from_numpy(result).permute(_CHANNEL_LAST_TO_TENSOR[layout])
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
    img: ImageType | torch.Tensor,
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Normalize with fixed caller constants as ``(img - mean) * denominator``.

    Float16 input returns float16; uint8 and float32 return float32. NumPy arrays are
    channel-last. Tensor calls require CHW, NCHW, CDHW or NCDHW layout. Scalar and
    per-channel controls use the existing backend's arithmetic order and precision.

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
    float_out = cast("NDArray[np.float32] | None", out)
    float32_result = normalize_float32_torch(cast("NDArray[np.float32]", img), mean, denominator, out=float_out)
    return float32_result if float32_result is not None else _affine_numpy(img, mean, denominator, out=float_out)
