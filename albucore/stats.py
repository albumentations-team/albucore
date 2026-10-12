"""Benchmark-driven mean / std / mean_std for Albucore channel-last array layouts."""

from typing import Literal, TypeGuard, cast

import cv2
import numkong as nk
import numpy as np
import torch
from numpy.typing import DTypeLike

from albucore.tensor_bridge import TensorLayout, dispatch_tensor_reduction, tensor_to_numpy_channel_last
from albucore.torch_backend import reduce_sum_torch
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageFloat32, ImageType, ImageUInt8, get_opencv_max_channels
from albucore.weighted import _NUMKONG_NEON_HALF

DEFAULT_EPS = 1e-4
# Tiny relative variance makes E[x²] - E[x]² unreliable for half moments.
_FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR = 1e-4
# Host NEON-half NumKong moments reported a 1.68e-4 relative variance for constant 0.575 inputs.
_NUMKONG_FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR = 1e-2
_NUMKONG_FLOAT16_HWC_STATS_MIN_ELEMENTS = 20_480
_FLOAT16_PER_CHANNEL_TORCH_MIN_ELEMENTS = 20_480
_FLOAT16_TENSOR_MEAN_MIN_ELEMENTS = 40_960
_FLOAT16_TENSOR_REDUCE_SUM_MIN_ELEMENTS = 16_384
_FLOAT16_GLOBAL_NUMPY_REDUCE_SUM_MIN_ELEMENTS = 20_480
_FLOAT16_GLOBAL_TENSOR_STD_MIN_ELEMENTS = 40_960
_FLOAT16_GLOBAL_TENSOR_STD_F32_MIN_STD = 1e-3
# OpenCV #25019 tracks CV_16F meanStdDev accuracy; revisit this cap when that issue is resolved.
_FLOAT16_OPENCV_STATS_MAX_CHANNELS = min(get_opencv_max_channels(), MAX_OPENCV_WORKING_CHANNELS)
_FLOAT16_OPENCV_MEAN_MIN_ELEMENTS = 512_000
_FLOAT16_GLOBAL_MEAN_OPENCV_MIN_ELEMENTS = 4_096
_FLOAT16_GLOBAL_TENSOR_MOMENTS_MIN_ELEMENTS = 20_480
# The rank-4 C=3 Torch reduction wins for the measured NHWC/DHWC region.
_FLOAT16_RANK4_PER_CHANNEL_TORCH_MAX_ELEMENTS = 245_760
_FLOAT16_RANK4_C1_PER_CHANNEL_TORCH_MAX_ELEMENTS = 307_200
_FLOAT16_RANK4_C1_MEAN_TORCH_MAX_ELEMENTS = 8_000_000
_FLOAT16_GLOBAL_STATS_OPENCV_MIN_ELEMENTS = 20_480
# Single-channel CV16F stats use the measured contiguous HWC/NHWC size regions.
_FLOAT16_GLOBAL_STATS_C1_HWC_MIN_ELEMENTS = 20_480
_FLOAT16_GLOBAL_STATS_C1_NHWC_MIN_ELEMENTS = 81_920
_FLOAT16_GLOBAL_STATS_C1_NHWC_MAX_ELEMENTS = 1_228_800
_FLOAT16_GLOBAL_STD_C1_HWC_MIN_ELEMENTS = 307_200
_FLOAT16_GLOBAL_STD_C1_NHWC_MIN_ELEMENTS = 204_800
_FLOAT16_GLOBAL_STD_C1_NHWC_MAX_ELEMENTS = 614_400
# Paired timings show an early Torch win only within these contiguous HWC cells.
_FLOAT16_PER_CHANNEL_TORCH_MAX_ELEMENTS_BY_CHANNELS = {
    1: 307_200,
    3: 307_200,
    4: 307_200,
    5: 184_320,
    9: 184_320,
}

AxisSpec = None | int | tuple[int, ...] | Literal["global", "per_channel"]

__all__ = ["DEFAULT_EPS", "mean", "mean_std", "reduce_sum", "std"]


def _is_uint8_image(arr: ImageType) -> TypeGuard[ImageUInt8]:
    return arr.dtype == np.uint8


def _is_float32_image(arr: ImageType) -> TypeGuard[ImageFloat32]:
    return arr.dtype == np.float32


def _resolve_axes(arr: np.ndarray, axis: AxisSpec) -> tuple[int, ...] | None:
    """None → reduce all axes; 'per_channel' → all but last."""
    if axis is None or axis == "global":
        return None
    if axis == "per_channel":
        return tuple(range(arr.ndim - 1))
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


def _per_channel_spatial_axes(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(range(arr.ndim - 1))


def _numkong_float16_hwc_stats_eligible(arr: ImageType, axes: tuple[int, ...], keepdims: bool) -> bool:
    return (
        _NUMKONG_NEON_HALF
        and arr.dtype == np.float16
        and arr.ndim == 3
        and not keepdims
        and axes == _per_channel_spatial_axes(arr)
        and arr.shape[-1] <= _FLOAT16_OPENCV_STATS_MAX_CHANNELS
        and arr.flags.c_contiguous
        and arr.size >= _NUMKONG_FLOAT16_HWC_STATS_MIN_ELEMENTS
    )


def _float16_per_channel_torch_minimum(
    arr: ImageType,
    axes: tuple[int, ...] | None,
    keepdims: bool,
) -> int | None:
    if arr.dtype != np.float16 or axes != _per_channel_spatial_axes(arr) or not arr.flags.c_contiguous:
        return None
    if arr.ndim == 3:
        if keepdims:
            return None
        max_elements = _FLOAT16_PER_CHANNEL_TORCH_MAX_ELEMENTS_BY_CHANNELS.get(arr.shape[-1])
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        max_elements = _FLOAT16_RANK4_PER_CHANNEL_TORCH_MAX_ELEMENTS
    else:
        return None
    if max_elements is None or not _FLOAT16_PER_CHANNEL_TORCH_MIN_ELEMENTS <= arr.size <= max_elements:
        return None
    return _FLOAT16_PER_CHANNEL_TORCH_MIN_ELEMENTS


def _cv2_reduce_sum_per_channel(arr: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    result = np.asarray(cv2.reduce(flat, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).reshape(-1))
    if _is_uint8_image(arr):
        return np.asarray(result, dtype=np.uint64)
    return np.asarray(result, dtype=np.float64)


def _cv2_reduce_mean_per_channel(arr: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    return np.asarray(cv2.reduce(flat, 0, cv2.REDUCE_AVG, dtype=cv2.CV_64F).reshape(-1), dtype=np.float64)


def _reduce_sum_global_uint8(arr: ImageUInt8, *, keepdims: bool) -> np.uint64 | np.ndarray:
    torch_result = reduce_sum_torch(arr, tuple(range(arr.ndim)), keepdims=keepdims)
    if torch_result is not None:
        return cast("np.uint64 | np.ndarray", torch_result)
    out = np.uint64(int(nk.sum(arr)))
    if keepdims:
        return np.array(out, dtype=np.uint64).reshape((1,) * arr.ndim)
    return out


def _reduce_sum_global_float32(arr: ImageFloat32, *, keepdims: bool) -> np.float64 | np.ndarray:
    torch_result = reduce_sum_torch(arr, tuple(range(arr.ndim)), keepdims=keepdims)
    if torch_result is not None:
        return cast("np.float64 | np.ndarray", torch_result)
    out = np.sum(arr, dtype=np.float64)
    if keepdims:
        return np.asarray(out, dtype=np.float64).reshape((1,) * arr.ndim)
    return out


def _reduce_sum_per_channel_uint8(arr: ImageUInt8, *, keepdims: bool) -> np.ndarray:
    axes = _per_channel_spatial_axes(arr)
    c = arr.shape[-1]
    n_spatial = arr.size // c
    if c > MAX_OPENCV_WORKING_CHANNELS and n_spatial >= 65_536:
        result = _cv2_reduce_sum_per_channel(arr)
    else:
        result = np.asarray(nk.sum(arr, axis=axes), dtype=np.uint64)
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (c,))
    return result


def _reduce_sum_per_channel_float32(arr: ImageFloat32, axes: tuple[int, ...], *, keepdims: bool) -> np.ndarray:
    """Per-channel sum for float32: nk.sum for ndim ≤ 4 and 1 < C ≤ 4 (4x faster than numpy), numpy otherwise."""
    c = arr.shape[-1]
    if c <= MAX_OPENCV_WORKING_CHANNELS:
        torch_result = reduce_sum_torch(arr, axes, keepdims=keepdims)
        if torch_result is not None:
            return cast("np.ndarray", torch_result)
    n_spatial = arr.size // c
    if c > MAX_OPENCV_WORKING_CHANNELS and n_spatial >= 16_384:
        result = _cv2_reduce_sum_per_channel(arr)
        if keepdims:
            return result.reshape((1,) * (arr.ndim - 1) + (c,))
        return result
    if arr.ndim <= 4 and 1 < c <= MAX_OPENCV_WORKING_CHANNELS:
        result = np.asarray(nk.sum(arr, axis=axes), dtype=np.float64)
        if keepdims:
            return result.reshape((1,) * (arr.ndim - 1) + (c,))
        return result
    return np.asarray(np.sum(arr, axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)


def _reduce_sum_per_channel_float16_fastpath(
    arr: ImageType,
    axes: tuple[int, ...] | None,
    *,
    keepdims: bool,
) -> np.ndarray | None:
    if arr.dtype != np.float16 or axes != _per_channel_spatial_axes(arr) or not arr.flags.c_contiguous:
        return None

    if (
        arr.ndim == 4
        and arr.shape[-1] == 1
        and not keepdims
        and _FLOAT16_PER_CHANNEL_TORCH_MIN_ELEMENTS <= arr.size <= _FLOAT16_RANK4_C1_PER_CHANNEL_TORCH_MAX_ELEMENTS
        and arr.flags.writeable
        and all(stride >= 0 for stride in arr.strides)
    ):
        return cast("np.ndarray", torch.sum(torch.from_numpy(arr), dim=axes, dtype=torch.float64).numpy())

    if (
        arr.ndim not in (3, 4)
        or arr.shape[-1] != 9
        or arr.size <= _FLOAT16_PER_CHANNEL_TORCH_MAX_ELEMENTS_BY_CHANNELS[9]
    ):
        return None
    result = _cv2_reduce_sum_per_channel(arr.astype(np.float32))
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return result


def _reduce_sum_float16_torch(
    arr: ImageType,
    axes: tuple[int, ...] | None,
    *,
    keepdims: bool,
) -> np.generic | np.ndarray | None:
    if arr.dtype != np.float16:
        return None
    if axes is None:
        result = _reduce_sum_global_float16_numpy(arr, keepdims=keepdims)
        if result is not None:
            return result
    torch_axes = tuple(range(arr.ndim)) if axes is None else axes
    minimum_elements = _float16_per_channel_torch_minimum(arr, axes, keepdims)
    return reduce_sum_torch(arr, torch_axes, keepdims=keepdims, minimum_elements=minimum_elements)


def _reduce_sum_global_float16_numpy(arr: ImageType, *, keepdims: bool) -> np.float64 | None:
    if (
        arr.dtype != np.float16
        or keepdims
        or arr.size < _FLOAT16_GLOBAL_NUMPY_REDUCE_SUM_MIN_ELEMENTS
        or not arr.flags.c_contiguous
        or not arr.flags.writeable
    ):
        return None
    image = torch.from_numpy(arr)
    return cast("np.float64", torch.sum(image, dtype=torch.float64).numpy()[()])


def _reduce_sum_float16_tensor(
    arr: torch.Tensor,
    axis: AxisSpec,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
) -> torch.Tensor | None:
    if (
        arr.dtype != torch.float16
        or arr.numel() < _FLOAT16_TENSOR_REDUCE_SUM_MIN_ELEMENTS
        or not arr.is_contiguous()
        or layout is None
        or axis == ()
    ):
        return None
    axes: int | tuple[int, ...]
    if axis is None or axis == "global":
        axes = tuple(range(arr.ndim))
    elif axis == "per_channel":
        channel_axis = 0 if layout in ("CHW", "CDHW") else 1
        axes = tuple(index for index in range(arr.ndim) if index != channel_axis)
    elif isinstance(axis, (int, tuple)):
        axes = axis
    return torch.sum(arr, dim=axes, dtype=torch.float64, keepdim=keepdims)


def _reduce_sum_tensor(
    arr: torch.Tensor,
    axis: AxisSpec,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
) -> torch.Tensor:
    tensor_result = _reduce_sum_float16_tensor(arr, axis, layout, keepdims=keepdims)
    if tensor_result is not None:
        return tensor_result
    return cast("torch.Tensor", dispatch_tensor_reduction(arr, layout, reduce_sum, axis, {"keepdims": keepdims}))


def _reduce_sum_numpy(
    arr: np.ndarray,
    axes: tuple[int, ...] | None,
    *,
    keepdims: bool,
) -> np.generic | np.ndarray:
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        acc: type = np.uint64
    elif np.issubdtype(arr.dtype, np.floating):
        acc = np.float64
    else:
        # signed integers and bool
        acc = np.int64
    return np.sum(arr, axis=axes, dtype=acc, keepdims=keepdims)


def reduce_sum(
    arr: ImageType | torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> np.generic | np.ndarray | torch.Tensor:
    r"""Sum over image tensor axes with benchmark-driven routing.

    Routing:
    - **large uint8 global**: Torch ``int64`` sum reinterpreted as the
      documented uint64 accumulator; other global uint8 input uses NumKong ``nk.sum``.
    - **uint8 per-channel**: NumKong ``nk.sum(arr, axis=spatial_axes)`` — single call over
      all spatial dimensions for all ranks (HWC, DHWC, NHWC, …).
    - **large float32 global and per-channel C ≤ 4**: Torch float64
      accumulation through a zero-copy NumPy wrapper.
    - **contiguous float16 Tensor, 16,384+ elements**: direct Torch ``sum(dtype=float64)`` for global,
      per-channel, and explicit axes.
    - **contiguous NumPy float16 global, 20,480+ elements**: Torch ``sum(dtype=float64)`` through a zero-copy
      wrapper; tested ``keepdims=False`` calls matched NumPy float64 bitwise and ran at 0.879x-0.991x float32 time.
    - **other float16 global layouts and small arrays**: the default Torch threshold or NumPy float64 fallback.
    - **float16 per-channel HWC**: measured Torch windows for contiguous arrays. Rank-4 channel-last C=1 arrays up
      to 307,200 elements use Torch's float64 accumulator when ``keepdims=False``; C=3 arrays up to 245,760 elements
      use it with either ``keepdims`` value. Other ranks, channel counts and sizes keep the standard Torch threshold
      or NumPy wide-accumulator path. Contiguous C=9 arrays beyond the measured Torch window widen once and use
      OpenCV ``CV_64F reduce`` because CV_16F reduce is unsupported.
    - **other float32 global**: ``numpy.sum(dtype=float64)``.
    - **other float32 per-channel, ndim ≤ 4, 1 < C ≤ 4**: NumKong ``nk.sum`` — 4x faster than
      NumPy for RGB/RGBA images and 4-D batch/volume layouts.
    - **float32 per-channel, C=1 or C>4, or explicit-axis cases**: ``numpy.sum(dtype=float64)``.

    ``axis="per_channel"`` reduces all spatial axes (everything except the last / channel dim),
    matching the convention used by :func:`mean` and :func:`std`.

    Alternative: ``mean`` / ``std`` / ``mean_std`` for normalised statistics.

    Args:
        arr: Array with explicit channel dimension and ``uint8``, ``float16``, or ``float32`` data;
            other dtypes use NumPy's wide-accumulator fallback.
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → shape ``(C,)``;
            or explicit ``int`` / ``tuple[int, ...]`` axes in the input layout.
        keepdims: Same semantics as :func:`numpy.sum`.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Scalar (``uint64`` / ``int64`` / ``float64``) for a full reduction, else an array.
        The accumulator dtype follows the input: unsigned → uint64, float → float64,
        signed int / bool → int64.
    """
    if isinstance(arr, torch.Tensor):
        return _reduce_sum_tensor(arr, axis, layout, keepdims=keepdims)

    axes = _resolve_axes(arr, axis)
    bridge_result = _reduce_sum_per_channel_float16_fastpath(arr, axes, keepdims=keepdims)
    if bridge_result is not None:
        return bridge_result
    torch_result = _reduce_sum_float16_torch(arr, axes, keepdims=keepdims)
    if torch_result is not None:
        return torch_result
    result: np.generic | np.ndarray
    if axes is None:
        if _is_uint8_image(arr):
            result = _reduce_sum_global_uint8(arr, keepdims=keepdims)
        elif _is_float32_image(arr):
            result = _reduce_sum_global_float32(arr, keepdims=keepdims)
        else:
            result = _reduce_sum_numpy(arr, axes, keepdims=keepdims)
    elif axes == _per_channel_spatial_axes(arr):
        if _is_uint8_image(arr):
            result = _reduce_sum_per_channel_uint8(arr, keepdims=keepdims)
        elif _is_float32_image(arr):
            result = _reduce_sum_per_channel_float32(arr, axes, keepdims=keepdims)
        else:
            result = _reduce_sum_numpy(arr, axes, keepdims=keepdims)
    else:
        result = _reduce_sum_numpy(arr, axes, keepdims=keepdims)
    return result


def _global_mean_std_uint8(arr: ImageUInt8, eps: float) -> tuple[float, float]:
    s_sum, s_sq = nk.moments(arr)
    n = arr.size
    mean = float(s_sum) / n
    var = max(float(s_sq) / n - mean * mean, 0.0)
    return mean, float(np.sqrt(var)) + eps


def _global_mean_uint8_only(arr: ImageUInt8) -> float:
    """Global mean only: ``nk.sum`` / N (single reduction, no moments/sqrt work)."""
    return float(nk.sum(arr)) / arr.size


def _mean_std_per_channel_float64(arr: ImageType, eps: float) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(arr.ndim - 1))
    mean_values = np.asarray(np.mean(arr, axis=axes, dtype=np.float64), dtype=np.float64)
    std_values = np.asarray(np.std(arr, axis=axes, dtype=np.float64) + eps, dtype=np.float64)
    return mean_values, std_values


def _mean_std_per_channel_float32(arr: ImageFloat32, eps: float) -> tuple[np.ndarray, np.ndarray]:
    c = arr.shape[-1]
    if arr.ndim == 3 and 1 < c <= MAX_OPENCV_WORKING_CHANNELS:
        means, stds = cv2.meanStdDev(arr)
        mean_values = np.asarray(means[:, 0], dtype=np.float64)
        std_values = np.asarray(stds[:, 0], dtype=np.float64)
        if np.any(np.square(std_values) <= _FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR * np.square(mean_values)):
            return _mean_std_per_channel_float64(arr, eps)
        return mean_values, std_values + eps

    n = arr.size // c
    means = np.empty(c, dtype=np.float64)
    stds = np.empty(c, dtype=np.float64)
    for channel in range(c):
        total, squared_total = nk.moments(arr[..., channel])
        mean_value = float(total) / n
        variance = max(float(squared_total) / n - mean_value * mean_value, 0.0)
        if variance <= _FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR * mean_value * mean_value:
            return _mean_std_per_channel_float64(arr, eps)
        means[channel] = mean_value
        stds[channel] = np.sqrt(variance) + eps
    return means, stds


def _mean_std_per_channel_numkong(arr: ImageType, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel population mean/std via one NumKong moments call per channel."""
    if arr.dtype == np.float16:
        magnitude_bits = np.bitwise_and(arr.view(np.uint16), 0x7FFF)
        # Subtracting one maps zero to 0xffff and leaves subnormal magnitudes below 0x03ff.
        np.subtract(magnitude_bits, np.uint16(1), out=magnitude_bits)
        has_subnormals = np.min(magnitude_bits) < 0x03FF
        if has_subnormals:
            return _mean_std_per_channel_float32(cast("ImageFloat32", arr.astype(np.float32)), eps)

    c = arr.shape[-1]
    n = arr.size // c
    means = np.empty(c, dtype=np.float64)
    stds = np.empty(c, dtype=np.float64)
    for channel in range(c):
        channel_values = arr[..., channel]
        s_sum, s_sq = nk.moments(channel_values)
        mean_value = float(s_sum) / n
        var = max(float(s_sq) / n - mean_value * mean_value, 0.0)
        near_constant = var <= _NUMKONG_FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR * mean_value * mean_value
        if arr.dtype == np.float16 and near_constant:
            mean_value = float(np.mean(channel_values, dtype=np.float64))
            var = float(np.var(channel_values, dtype=np.float64))
        means[channel] = mean_value
        stds[channel] = np.sqrt(var) + eps
    return means, stds


def _mean_std_per_channel_opencv(arr: ImageType, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Widen half inputs because CV_16F statistics lose subnormal precision."""
    is_half = arr.dtype == np.float16
    work = arr.astype(np.float32) if is_half else arr
    mean, std = cv2.meanStdDev(work)
    if is_half and np.any(np.square(std[:, 0]) <= _FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR * np.square(mean[:, 0])):
        return _mean_std_per_channel_numkong(arr, eps)
    mean_values = np.asarray(mean[:, 0], dtype=np.float64)
    std_values = np.asarray(std[:, 0] + eps, dtype=np.float64)
    return mean_values, std_values


def _mean_std_global_float16_opencv(arr: ImageType, eps: float) -> tuple[float, float] | None:
    channels = arr.shape[-1]
    if (
        arr.dtype != np.float16
        or not arr.flags.c_contiguous
        or arr.size < _FLOAT16_GLOBAL_STATS_OPENCV_MIN_ELEMENTS
        or arr.ndim not in (3, 4)
        or channels > MAX_OPENCV_WORKING_CHANNELS
    ):
        return None
    if channels == 1:
        if arr.ndim == 3 and arr.size < _FLOAT16_GLOBAL_STATS_C1_HWC_MIN_ELEMENTS:
            return None
        if arr.ndim == 4 and not (
            _FLOAT16_GLOBAL_STATS_C1_NHWC_MIN_ELEMENTS <= arr.size <= _FLOAT16_GLOBAL_STATS_C1_NHWC_MAX_ELEMENTS
        ):
            return None

    image = arr if arr.ndim == 3 else arr.reshape(-1, 1, channels)
    channel_mean, channel_std = cv2.meanStdDev(image)
    means = np.asarray(channel_mean[:, 0], dtype=np.float64)
    stds = np.asarray(channel_std[:, 0], dtype=np.float64)
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        return None
    if np.any(np.square(stds) <= _FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR * np.square(means)):
        return None

    mean_value = float(np.mean(means, dtype=np.float64))
    variance = np.mean(np.square(stds) + np.square(means - mean_value), dtype=np.float64)
    return mean_value, float(np.sqrt(max(variance, 0.0))) + eps


def _mean_std_global_float16_fallback(arr: ImageType, eps: float) -> tuple[float, float]:
    mean_value = float(np.mean(arr, dtype=np.float64))
    std_value = float(np.std(arr, dtype=np.float64)) + eps
    return mean_value, std_value


def _mean_std_global_float16_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        arr.dtype != torch.float16
        or not arr.is_contiguous()
        or arr.numel() < _FLOAT16_GLOBAL_TENSOR_MOMENTS_MIN_ELEMENTS
    ):
        return None
    if layout is None:
        return None

    image = arr.numpy().reshape(-1, 1)
    mean_array, std_array = cv2.meanStdDev(image)
    mean_value = float(mean_array[0, 0])
    std_value = float(std_array[0, 0])
    if (
        not np.isfinite(mean_value)
        or not np.isfinite(std_value)
        or std_value * std_value <= _FLOAT16_MOMENTS_RELATIVE_VARIANCE_FLOOR * mean_value * mean_value
    ):
        mean_value, std_value = _mean_std_global_float16_fallback(arr.numpy(), eps)
    else:
        std_value += eps
    return (
        torch.tensor(mean_value, dtype=torch.float64),
        torch.tensor(std_value, dtype=torch.float64),
    )


def _mean_std_global(
    arr: ImageType,
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.floating | float | np.ndarray, np.floating | float | np.ndarray]:
    mean_value: np.floating | float
    std_value: np.floating | float
    if _is_uint8_image(arr):
        mean_value, std_value = _global_mean_std_uint8(arr, eps)
    elif arr.dtype == np.float16:
        cv_stats = _mean_std_global_float16_opencv(arr, eps)
        mean_value, std_value = _mean_std_global_float16_fallback(arr, eps) if cv_stats is None else cv_stats
    else:
        mean_value = np.mean(arr, dtype=np.float64)
        std_value = np.std(arr, dtype=np.float64) + eps
    if keepdims:
        shape = (1,) * arr.ndim
        return (
            np.asarray(mean_value, dtype=np.float64).reshape(shape),
            np.asarray(std_value, dtype=np.float64).reshape(shape),
        )
    return mean_value, std_value


def _mean_global(arr: ImageType, *, keepdims: bool) -> np.floating | float | np.ndarray:
    if _is_uint8_image(arr):
        m = _global_mean_uint8_only(arr)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(m, dtype=np.float64).reshape(kd)
        return m
    if arr.dtype == np.float16 and arr.flags.c_contiguous and arr.size >= _FLOAT16_GLOBAL_MEAN_OPENCV_MIN_ELEMENTS:
        wide = arr.astype(np.float32)
        mean_value = np.float64(cv2.mean(wide.reshape(-1, 1))[0])
        if keepdims:
            return mean_value.reshape((1,) * arr.ndim)
        return mean_value
    if _is_float32_image(arr):
        return np.mean(arr, dtype=np.float64, keepdims=keepdims)
    return np.mean(arr, dtype=np.float64, keepdims=keepdims)


def _mean_global_float16_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
) -> torch.Tensor | None:
    if arr.dtype != torch.float16 or arr.numel() < _FLOAT16_TENSOR_MEAN_MIN_ELEMENTS or layout is None:
        return None
    if tensor_to_numpy_channel_last(arr, layout).shape[-1] == 1:
        return None
    result = torch.mean(arr, dtype=torch.float32).to(torch.float64)
    return result.reshape((1,) * arr.ndim) if keepdims else result


def _mean_per_channel_float16_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
) -> torch.Tensor | None:
    if (
        arr.dtype != torch.float16
        or arr.numel() < _FLOAT16_TENSOR_MEAN_MIN_ELEMENTS
        or not arr.is_contiguous()
        or layout is None
        or (layout == "CHW" and arr.shape[0] == 1)
    ):
        return None
    channel_axis = 0 if layout in ("CHW", "CDHW") else 1
    axes = tuple(axis for axis in range(arr.ndim) if axis != channel_axis)
    result = torch.mean(arr, dim=axes, dtype=torch.float64)
    if not keepdims:
        return result
    output_shape = [1] * arr.ndim
    output_shape[channel_axis] = arr.shape[channel_axis]
    return result.reshape(output_shape)


def _std_global_float16_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
    eps: float,
) -> torch.Tensor | None:
    if (
        arr.dtype != torch.float16
        or arr.numel() < _FLOAT16_GLOBAL_TENSOR_STD_MIN_ELEMENTS
        or not arr.is_contiguous()
        or layout is None
    ):
        return None
    low, high = torch.aminmax(arr)
    if float(low) >= -1.0 and float(high) <= 1.0:
        image_view = tensor_to_numpy_channel_last(arr, layout)
        std_value = np.std(image_view, dtype=np.float32)
        if std_value > _FLOAT16_GLOBAL_TENSOR_STD_F32_MIN_STD:
            result = torch.tensor(np.float64(std_value) + eps, dtype=torch.float64)
            return result.reshape((1,) * arr.ndim) if keepdims else result
    mean_value = torch.mean(arr, dtype=torch.float64)
    centered = arr.to(torch.float64)
    centered.sub_(mean_value).square_()
    result = torch.mean(centered, dtype=torch.float64).sqrt() + eps
    return result.reshape((1,) * arr.ndim) if keepdims else result


def _std_global(arr: ImageType, *, keepdims: bool, eps: float) -> np.floating | float | np.ndarray:
    std_value: np.floating | float
    if _is_uint8_image(arr):
        std_value = _global_mean_std_uint8(arr, eps)[1]
    elif arr.dtype == np.float16:
        std_opencv_eligible = arr.shape[-1] != 1 or (
            (arr.ndim == 3 and arr.size >= _FLOAT16_GLOBAL_STD_C1_HWC_MIN_ELEMENTS)
            or (
                arr.ndim == 4
                and _FLOAT16_GLOBAL_STD_C1_NHWC_MIN_ELEMENTS <= arr.size <= _FLOAT16_GLOBAL_STD_C1_NHWC_MAX_ELEMENTS
            )
        )
        cv_stats = _mean_std_global_float16_opencv(arr, eps) if std_opencv_eligible else None
        std_value = float(np.std(arr, dtype=np.float64)) + eps if cv_stats is None else cv_stats[1]
    else:
        std_value = np.std(arr, dtype=np.float64) + eps
    if keepdims:
        return np.asarray(std_value, dtype=np.float64).reshape((1,) * arr.ndim)
    return float(std_value)


def _mean_std_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_axes = _per_channel_spatial_axes(arr)
    if _numkong_float16_hwc_stats_eligible(arr, axes, keepdims):
        return _mean_std_per_channel_numkong(arr, eps)
    if (
        arr.dtype == np.float16
        and arr.ndim == 3
        and not keepdims
        and axes == spatial_axes
        and arr.shape[-1] <= _FLOAT16_OPENCV_STATS_MAX_CHANNELS
    ):
        return _mean_std_per_channel_opencv(arr, eps)
    if axes == spatial_axes and not keepdims:
        c = arr.shape[-1]
        if _is_uint8_image(arr) or arr.dtype == np.float16:
            return _mean_std_per_channel_numkong(arr, eps)
        if _is_float32_image(arr) and (arr.ndim != 3 or c == 1 or c > MAX_OPENCV_WORKING_CHANNELS):
            return _mean_std_per_channel_numkong(arr, eps)

    if arr.ndim == 3 and not keepdims and axes == spatial_axes and arr.shape[-1] <= MAX_OPENCV_WORKING_CHANNELS:
        mean, std = cv2.meanStdDev(arr)
        m = mean[:, 0].astype(np.float64, copy=False)
        st = (std[:, 0] + eps).astype(np.float64, copy=False)
        return m, st

    m = arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims)
    st = arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps
    return m, st


def _mean_per_channel_uint8(arr: ImageUInt8, axes: tuple[int, ...], *, keepdims: bool) -> np.ndarray:
    """Per-channel mean for uint8 via nk.sum — fastest for all shapes and channel counts."""
    n_spatial = arr.size // arr.shape[-1]
    c = arr.shape[-1]
    if c > MAX_OPENCV_WORKING_CHANNELS and n_spatial >= 65_536:
        result = _cv2_reduce_mean_per_channel(arr)
    else:
        result = np.asarray(nk.sum(arr, axis=axes), dtype=np.float64) / n_spatial
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (c,))
    return result


def _mean_per_channel_rank4(arr: ImageType, axes: tuple[int, ...], *, keepdims: bool) -> np.ndarray:
    c = arr.shape[-1]
    if arr.dtype == np.float16:
        if c == 1 and arr.size <= _FLOAT16_RANK4_C1_MEAN_TORCH_MAX_ELEMENTS:
            total = _reduce_sum_global_float16_numpy(arr, keepdims=False)
            if total is not None:
                shape = (1,) * arr.ndim if keepdims else (1,)
                return np.asarray(total / arr.size, dtype=np.float64).reshape(shape)
        if (
            c == 3
            and arr.size >= _FLOAT16_PER_CHANNEL_TORCH_MIN_ELEMENTS
            and arr.flags.c_contiguous
            and arr.flags.writeable
        ):
            return np.asarray(
                torch.mean(torch.from_numpy(arr), dim=axes, dtype=torch.float64, keepdim=keepdims).numpy(),
                dtype=np.float64,
            )
        return np.asarray(arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)
    if c == 1 or keepdims:
        return np.asarray(arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)
    n_spatial = arr.size // c
    result = np.asarray(nk.sum(arr, axis=axes), dtype=np.float64)
    return cast("np.ndarray", result / n_spatial)


def _mean_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> np.ndarray:
    spatial_axes = _per_channel_spatial_axes(arr)
    if _is_uint8_image(arr) and axes == spatial_axes:
        return _mean_per_channel_uint8(arr, axes, keepdims=keepdims)
    c = arr.shape[-1]
    # HWC, C ≤ 4: cv2.mean is fastest; half inputs widen for accuracy.
    if arr.ndim == 3 and not keepdims and axes == spatial_axes and c <= MAX_OPENCV_WORKING_CHANNELS:
        # CV_16F mean can lose precision on skewed and subnormal inputs.
        mu = cv2.mean(arr.astype(np.float32)) if arr.dtype == np.float16 else cv2.mean(arr)
        return np.asarray(mu[:c], dtype=np.float64)
    if arr.ndim == 4 and axes == spatial_axes and 1 <= c <= MAX_OPENCV_WORKING_CHANNELS:
        return _mean_per_channel_rank4(arr, axes, keepdims=keepdims)
    if c > MAX_OPENCV_WORKING_CHANNELS and axes == spatial_axes and arr.size // c >= 16_384:
        if (
            arr.dtype == np.float16
            and arr.ndim == 3
            and not keepdims
            and c == 9
            and c <= _FLOAT16_OPENCV_STATS_MAX_CHANNELS
            and arr.size >= _FLOAT16_OPENCV_MEAN_MIN_ELEMENTS
        ):
            result, _ = _mean_std_per_channel_opencv(arr, 0.0)
        else:
            work = arr.astype(np.float32) if arr.dtype == np.float16 else arr
            result = _cv2_reduce_mean_per_channel(work)
        if keepdims:
            return result.reshape((1,) * (arr.ndim - 1) + (c,))
        return result
    return np.asarray(arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)


def _std_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> np.ndarray:
    spatial_axes = _per_channel_spatial_axes(arr)
    c = arr.shape[-1]
    if _numkong_float16_hwc_stats_eligible(arr, axes, keepdims):
        return _mean_std_per_channel_numkong(arr, eps)[1]
    if (
        arr.dtype == np.float16
        and arr.ndim == 3
        and not keepdims
        and axes == spatial_axes
        and c <= _FLOAT16_OPENCV_STATS_MAX_CHANNELS
    ):
        _, std_values = _mean_std_per_channel_opencv(arr, eps)
        return std_values
    if (
        _is_float32_image(arr)
        and arr.ndim == 3
        and not keepdims
        and axes == spatial_axes
        and 1 < c <= MAX_OPENCV_WORKING_CHANNELS
    ):
        _, std = cv2.meanStdDev(arr)
        return np.asarray(std[:, 0] + eps, dtype=np.float64)
    if (
        axes == spatial_axes
        and not keepdims
        and (_is_uint8_image(arr) or _is_float32_image(arr) or arr.dtype == np.float16)
    ):
        _, std = _mean_std_per_channel_numkong(arr, eps)
        return std
    return np.asarray(arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps, dtype=np.float64)


def mean_std(
    arr: ImageType | torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[np.floating | float | np.ndarray | torch.Tensor, np.floating | float | np.ndarray | torch.Tensor]:
    """Compute population mean and standard deviation (+ eps) jointly.

    More efficient than calling ``mean`` and ``std`` separately when both are needed,
    because the uint8 global path uses a single ``nk.moments`` pass.

    Routing:
    - **uint8 global**: NumKong ``nk.moments`` (single pass, wide accumulator).
    - **float32 global**: ``np.mean`` + ``np.std`` (``dtype=float64`` for accuracy).
    - **float16 global**: fused CV16F channel statistics in measured contiguous HWC/NHWC C≤4 regions; use a wide
      NumPy fallback when cancellation dominates or the layout is unmeasured.
    - **contiguous float16 Tensor global, 20,480 elements and larger**: CV16F ``meanStdDev`` over a zero-copy scalar
      view; near-constant and non-finite inputs widen to NumPy float64.
    - **per-channel float16 HWC, C ≤ 4**: OpenCV ``meanStdDev`` after widening to float32; direct CV16F statistics
      can lose low-order precision and drop subnormals.
    - **per-channel uint8 and float16 C > 4**: NumKong moments per channel, with a stable two-pass fallback for
      near-constant half data.
    - **per-channel float32, ndim == 3, 2 ≤ C ≤ 4**: ``cv2.meanStdDev``.
    - **per-channel float32, C=1, C>4, or ndim>3**: one NumKong ``moments`` call per channel.
    - **explicit axes / keepdims**: NumPy-compatible ``mean``/``std``.

    Alternative: ``mean`` or ``std`` if only one statistic is needed.

    Args:
        arr: uint8, float16, or float32 image/batch/volume with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar pair; ``"per_channel"`` → ``(C,)`` pair;
            explicit ``int`` / ``tuple[int, ...]`` for NumPy-style axes.
        keepdims: Preserve reduced dimensions (same semantics as NumPy).
        eps: Added to std to prevent division-by-zero (default ``1e-4``).
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        ``(mean, std + eps)`` — scalars for global reduction, arrays for per-channel.
    """
    if isinstance(arr, torch.Tensor):
        if axis is None or axis == "global":
            tensor_result = _mean_std_global_float16_tensor(arr, layout, eps)
            if tensor_result is not None:
                if keepdims:
                    shape = (1,) * arr.ndim
                    return tensor_result[0].reshape(shape), tensor_result[1].reshape(shape)
                return tensor_result
        return cast(
            "tuple[np.floating | float | np.ndarray | torch.Tensor, np.floating | float | np.ndarray | torch.Tensor]",
            dispatch_tensor_reduction(arr, layout, mean_std, axis, {"keepdims": keepdims, "eps": eps}),
        )

    axes = _resolve_axes(arr, axis)
    if axes is None:
        return _mean_std_global(arr, keepdims=keepdims, eps=eps)
    return _mean_std_per_channel(arr, axes, keepdims=keepdims, eps=eps)


def mean(
    arr: ImageType | torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = None,
    layout: TensorLayout | None = None,
) -> np.floating | float | np.ndarray | torch.Tensor:
    """Compute population mean over image tensor axes.

    Routing:
    - **uint8 global**: NumKong ``nk.sum(arr) / N`` (single reduction; avoids moments/sqrt work).
    - **uint8 per-channel**: NumKong ``nk.sum(spatial_axes) / n`` — wins for all shapes
      and channel counts (2x over cv2.mean for HWC C=3, 40x for NHWC/DHWC).
    - **float32 global**: ``np.mean(dtype=float64)``.
    - **float32 per-channel, ndim==3, C ≤ 4**: ``cv2.mean`` (fastest for HWC images).
    - **float32 per-channel, ndim==4, 1 < C ≤ 4**: NumKong ``nk.sum / n`` — 4x over arr.mean
      for NHWC/DHWC batch/volume layouts.
    - **contiguous float16 global, 4,096 elements and larger**: direct ``CV_16F cv2.mean`` on a zero-copy scalar view.
    - **global float16 Tensor, 40,960 elements and larger, C>1**: one CPU Torch reduction with float32 accumulation;
      the scalar output remains float64.
    - **per-channel float16 Tensor, 40,960 elements and larger**: contiguous NCHW/NCDHW, CDHW, and multi-channel CHW
      inputs use a CPU Torch reduction with float64 accumulation; single-channel CHW keeps its measured OpenCV path.
    - **float16 per-channel HWC, C ≤ 4**: widen to CV_32F before ``cv2.mean`` because direct CV_16F reduction can
      exceed the float64-reference tolerance. Higher channel counts use a float32 reduction path.
    - **NumPy float16 per-channel, rank-4 C=3**: zero-copy Torch ``mean(dtype=float64)`` from 20,480 elements;
      rank-4 C=1 uses the global Torch float64 sum from 20,480 through 8,000,000 elements.
    - **everything else**: ``arr.mean(dtype=float64)``.

    Alternative: ``mean_std`` if std is also needed (avoids a second pass for uint8).

    Args:
        arr: uint8, float16, or float32 image/batch/volume with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → ``(C,)`` array;
            explicit ``int`` / ``tuple[int, ...]`` for NumPy-style axes.
        keepdims: Preserve reduced dimensions.
        dtype: Cast the output to this dtype (e.g. ``np.float32``).
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Scalar or array of float64 means (cast to ``dtype`` if provided).
    """
    if isinstance(arr, torch.Tensor):
        if dtype is None and (axis is None or axis == "global"):
            result = _mean_global_float16_tensor(arr, layout, keepdims=keepdims)
            if result is not None:
                return result
        if dtype is None and axis == "per_channel":
            result = _mean_per_channel_float16_tensor(arr, layout, keepdims=keepdims)
            if result is not None:
                return result
        return dispatch_tensor_reduction(arr, layout, mean, axis, {"keepdims": keepdims, "dtype": dtype})

    axes = _resolve_axes(arr, axis)
    m = _mean_global(arr, keepdims=keepdims) if axes is None else _mean_per_channel(arr, axes, keepdims=keepdims)
    if dtype is not None:
        return np.asarray(m, dtype=dtype)
    return m


def std(
    arr: ImageType | torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: DTypeLike | None = None,
    layout: TensorLayout | None = None,
) -> np.floating | float | np.ndarray | torch.Tensor:
    """Compute population standard deviation (+ eps) over image tensor axes.

    Routing:
    - **uint8 global**: NumKong ``nk.moments`` (single pass, wide accumulator).
    - **float16 global NumPy**: fused CV16F channel statistics in measured contiguous HWC/NHWC C≤4 regions; other
      layouts use wide NumPy accumulation.
    - **contiguous float16 Tensor global, 40,960 elements and larger**: a float32 accumulator when all values are in
      [-1, 1] and the resulting std exceeds 1e-3; otherwise a two-pass float64 Torch reduction. Non-contiguous inputs
      retain the wide NumPy route.
    - **float32 global**: ``np.std(dtype=float64)``.
    - **per-channel uint8**: one NumKong ``moments`` call per channel.
    - **per-channel float32, ndim == 3, 2 ≤ C ≤ 4**: ``cv2.meanStdDev``.
    - **per-channel float32, C=1, C>4, or ndim>3**: one NumKong ``moments`` call per channel.
    - **per-channel float16 HWC, C ≤ 4**: widen to float32 before ``cv2.meanStdDev``.
    - **explicit axes / keepdims**: ``arr.std(dtype=float64)``.

    Alternative: ``mean_std`` if mean is also needed (avoids a redundant pass for uint8).

    Args:
        arr: uint8, float16, or float32 image/batch/volume with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → ``(C,)`` array;
            explicit ``int`` / ``tuple[int, ...]`` for NumPy-style axes.
        keepdims: Preserve reduced dimensions.
        eps: Added to std to prevent division-by-zero (default ``1e-4``).
        dtype: Cast the output to this dtype (e.g. ``np.float32``).
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Scalar or array of float64 stds + eps (cast to ``dtype`` if provided).
    """
    if isinstance(arr, torch.Tensor):
        if dtype is None and (axis is None or axis == "global"):
            result = _std_global_float16_tensor(arr, layout, keepdims=keepdims, eps=eps)
            if result is not None:
                return result
        return dispatch_tensor_reduction(
            arr,
            layout,
            std,
            axis,
            {"keepdims": keepdims, "eps": eps, "dtype": dtype},
        )

    axes = _resolve_axes(arr, axis)
    s = (
        _std_global(arr, keepdims=keepdims, eps=eps)
        if axes is None
        else _std_per_channel(
            arr,
            axes,
            keepdims=keepdims,
            eps=eps,
        )
    )
    if dtype is not None:
        return np.asarray(s, dtype=dtype)
    return s
