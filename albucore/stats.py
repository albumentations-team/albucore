"""Benchmark-driven mean / std / mean_std for Albucore channel-last array layouts."""

import math
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast, overload

import cv2
import numkong as nk
import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray

from albucore._float_cast import _cast_float_numpy
from albucore.tensor_bridge import (
    TensorLayout,
    dispatch_tensor_reduction,
    numpy_to_tensor_storage,
    tensor_to_numpy_channel_last,
)
from albucore.torch_backend import reduce_sum_torch
from albucore.utils import (
    _LINUX_X86_64,
    _MACOS_ARM64,
    _TENSOR_CHANNEL_AXES,
    MAX_OPENCV_WORKING_CHANNELS,
    _is_float32_image,
    _is_uint8_image,
    get_opencv_max_channels,
)
from albucore.weighted import _NUMKONG_NEON_HALF

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_EPS = 1e-4
_NUMPY_STD_MEAN_SUPPORTED = np.lib.NumpyVersion(np.__version__) >= "2.0.0"
_FLOAT32_GLOBAL_STATS_OPENCV_MIN_ELEMENTS = 2
_FLOAT32_GLOBAL_STATS_RELATIVE_VARIANCE_FLOOR = 1e-4
# Tiny relative variance makes E[x²] - E[x]² unreliable for float16 and float32 moments.
_MOMENTS_RELATIVE_VARIANCE_FLOOR = 1e-4
# Float64-accumulated float32 moments cancelled to about 5e-12 relative variance on 262,144 constant elements;
# this floor keeps a 1000x margin and leaves images whose std is at least 0.01% of the mean on the one-pass path.
_FLOAT32_MOMENTS_CANCELLATION_FLOOR = 1e-8
# Host NEON-half NumKong moments reported a 1.68e-4 relative variance for constant 0.575 inputs.
_NUMKONG_MOMENTS_RELATIVE_VARIANCE_FLOOR = 1e-2
_NUMKONG_FLOAT16_HWC_STATS_MIN_ELEMENTS = 20_480
_FLOAT16_PER_CHANNEL_TORCH_MIN_ELEMENTS = 20_480
_FLOAT16_TENSOR_MEAN_MIN_ELEMENTS = 40_960
_FLOAT16_TENSOR_REDUCE_SUM_MIN_ELEMENTS = 256
_FLOAT16_GLOBAL_NUMPY_REDUCE_SUM_MIN_ELEMENTS = 20_480
# OpenCV #25019 tracks CV_16F meanStdDev accuracy; revisit this cap when that issue is resolved.
_FLOAT16_OPENCV_STATS_MAX_CHANNELS = min(get_opencv_max_channels(), MAX_OPENCV_WORKING_CHANNELS)
_FLOAT16_GLOBAL_MEAN_OPENCV_MIN_ELEMENTS = 4_096
_FLOAT16_GLOBAL_TENSOR_MOMENTS_MIN_ELEMENTS = 20_480
# The rank-4 C=3 Torch reduction wins for the measured NHWC/DHWC region.
_FLOAT16_RANK4_PER_CHANNEL_TORCH_MAX_ELEMENTS = 245_760
_FLOAT16_RANK4_C1_PER_CHANNEL_TORCH_MAX_ELEMENTS = 307_200
_FLOAT16_RANK4_C1_MEAN_TORCH_MAX_ELEMENTS = 8_000_000
# Widening to float32 for OpenCV meanStdDev beats the NumPy float64 reductions from this size on all channel counts.
_FLOAT16_GLOBAL_STATS_OPENCV_MIN_ELEMENTS = 4_096
# Paired timings show an early Torch win only within these contiguous HWC cells.
_FLOAT16_PER_CHANNEL_TORCH_MAX_ELEMENTS_BY_CHANNELS = {
    1: 307_200,
    3: 307_200,
    4: 307_200,
    5: 184_320,
}

# Float32 CHW Tensors with 2..4 channels reduce each contiguous plane; wider and single-channel inputs do not win.
_FLOAT32_TENSOR_PLANE_MOMENTS_MAX_CHANNELS = 4
_FLOAT32_TENSOR_NCHW_MEAN_CHANNELS = 9

# Per-channel uint8 sums use nk.sum up to this many channels; cv2.reduce wins only for wider images.
_UINT8_NUMKONG_PER_CHANNEL_MAX_CHANNELS = 9
_UINT8_PER_CHANNEL_OPENCV_MIN_SPATIAL = 65_536
# Per-plane nk.sum on a contiguous channel-first Tensor beats the transposed-view reduction for C > 4.
_UINT8_TENSOR_PLANE_SUM_MIN_SPATIAL = 65_536

AxisSpec = None | int | tuple[int, ...] | Literal["global", "per_channel"]
_GlobalAxis: TypeAlias = Literal["global"] | None
_StatsScalar: TypeAlias = float | np.float64
_StatsResult: TypeAlias = _StatsScalar | NDArray[np.float64]
_OutputDTypeT = TypeVar("_OutputDTypeT", bound=np.generic)

__all__ = ["DEFAULT_EPS", "mean", "mean_std", "reduce_sum", "std"]


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


def _numkong_float16_hwc_stats_eligible(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    axes: tuple[int, ...],
    keepdims: bool,
) -> bool:
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
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
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


def _reduce_sum_global_uint8(arr: NDArray[np.uint8], *, keepdims: bool) -> np.uint64 | np.ndarray:
    out = np.uint64(int(nk.sum(arr)))
    if keepdims:
        return np.array(out, dtype=np.uint64).reshape((1,) * arr.ndim)
    return out


def _float32_sum_elems(arr: NDArray[np.float32]) -> np.float64 | None:
    """Return the ``cv2.sumElems`` total of a contiguous float32 array, or None when it is not finite.

    OpenCV accumulates float32 blocks, so the total differs from the float64 sum by rounding only (rtol 1e-6 documented
    at the public API). A non-finite total (overflowing blocks, inf or NaN input) is left to the NumPy float64 route.
    """
    total = cv2.sumElems(arr.reshape(-1, 1))[0]
    return np.float64(total) if math.isfinite(total) else None


def _reduce_sum_global_float32(arr: NDArray[np.float32], *, keepdims: bool) -> np.float64 | np.ndarray:
    out = None
    if _LINUX_X86_64:
        # NumKong accumulates float32 in float64 and is 5-25x faster than Torch and NumPy on x86-64.
        out = np.float64(nk.sum(arr))
    elif _MACOS_ARM64 and arr.flags.c_contiguous:
        out = _float32_sum_elems(arr)
    if out is None:
        torch_result = reduce_sum_torch(arr, tuple(range(arr.ndim)), keepdims=keepdims)
        if torch_result is not None:
            return cast("np.float64 | np.ndarray", torch_result)
        out = np.sum(arr, dtype=np.float64)
    if keepdims:
        return np.asarray(out, dtype=np.float64).reshape((1,) * arr.ndim)
    return out


def _reduce_sum_per_channel_uint8(arr: NDArray[np.uint8], *, keepdims: bool) -> np.ndarray:
    axes = _per_channel_spatial_axes(arr)
    c = arr.shape[-1]
    n_spatial = arr.size // c
    if c > _UINT8_NUMKONG_PER_CHANNEL_MAX_CHANNELS and n_spatial >= _UINT8_PER_CHANNEL_OPENCV_MIN_SPATIAL:
        result = _cv2_reduce_sum_per_channel(arr)
    else:
        result = np.asarray(nk.sum(arr, axis=axes), dtype=np.uint64)
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (c,))
    return result


def _reduce_sum_per_channel_float32(arr: NDArray[np.float32], axes: tuple[int, ...], *, keepdims: bool) -> np.ndarray:
    """Per-channel float32 sum: Torch for C ≤ 4, OpenCV for large C > 4, then nk.sum for ndim ≤ 4, else NumPy."""
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
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
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
        return np.asarray(torch.sum(torch.from_numpy(arr), dim=axes, dtype=torch.float64).numpy())

    if arr.ndim not in (3, 4) or arr.shape[-1] != 9:
        return None
    result = _cv2_reduce_sum_per_channel(_cast_float_numpy(arr, np.float32))
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return result


def _reduce_sum_float16_torch(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
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


def _reduce_sum_global_float16_numpy(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    *,
    keepdims: bool,
) -> np.float64 | None:
    if (
        keepdims
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
        channel_axis = _TENSOR_CHANNEL_AXES[layout]
        axes = tuple(index for index in range(arr.ndim) if index != channel_axis)
    elif isinstance(axis, (int, tuple)):
        axes = axis
    return torch.sum(arr, dim=axes, dtype=torch.float64, keepdim=keepdims)


def _uint8_tensor_channel_sums(
    arr: torch.Tensor,
    axis: AxisSpec,
    layout: TensorLayout | None,
) -> tuple[np.ndarray, int] | None:
    """Per-channel uint8 sums of a contiguous CHW/NCHW Tensor as one ``nk.sum`` per channel plane."""
    if arr.dtype != torch.uint8 or axis != "per_channel" or layout not in ("CHW", "NCHW") or not arr.is_contiguous():
        return None
    channels = arr.shape[_TENSOR_CHANNEL_AXES[layout]]
    n_spatial = arr.numel() // channels
    if channels <= MAX_OPENCV_WORKING_CHANNELS or n_spatial < _UINT8_TENSOR_PLANE_SUM_MIN_SPATIAL:
        return None
    planes = arr.numpy()
    if layout == "CHW":
        planes = planes[None]
    sums = np.array([nk.sum(planes[:, channel]) for channel in range(channels)], dtype=np.uint64)
    return sums, n_spatial


def _channel_vector_to_tensor(
    values: np.ndarray,
    arr: torch.Tensor,
    layout: TensorLayout,
    *,
    keepdims: bool,
) -> torch.Tensor:
    result = numpy_to_tensor_storage(values)
    if not keepdims:
        return result
    shape = [1] * arr.ndim
    shape[_TENSOR_CHANNEL_AXES[layout]] = values.size
    return result.reshape(shape)


def _reduce_sum_tensor(
    arr: torch.Tensor,
    axis: AxisSpec,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
) -> torch.Tensor:
    plane_sums = _uint8_tensor_channel_sums(arr, axis, layout)
    if plane_sums is not None:
        return _channel_vector_to_tensor(plane_sums[0], arr, cast("TensorLayout", layout), keepdims=keepdims)
    tensor_result = _reduce_sum_float16_tensor(arr, axis, layout, keepdims=keepdims)
    if tensor_result is not None:
        return tensor_result
    return dispatch_tensor_reduction(arr, layout, reduce_sum, axis, {"keepdims": keepdims})


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


@overload
def reduce_sum(
    arr: NDArray[np.uint8],
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    layout: TensorLayout | None = None,
) -> np.uint64: ...


@overload
def reduce_sum(
    arr: NDArray[np.uint8],
    axis: AxisSpec = None,
    *,
    keepdims: Literal[True],
    layout: TensorLayout | None = None,
) -> NDArray[np.uint64]: ...


@overload
def reduce_sum(
    arr: NDArray[np.uint8],
    axis: Literal["per_channel"],
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint64]: ...


@overload
def reduce_sum(
    arr: NDArray[np.uint8],
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> np.uint64 | NDArray[np.uint64]: ...


@overload
def reduce_sum(
    arr: NDArray[np.float16] | NDArray[np.float32],
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    layout: TensorLayout | None = None,
) -> np.float64: ...


@overload
def reduce_sum(
    arr: NDArray[np.float16] | NDArray[np.float32],
    axis: AxisSpec = None,
    *,
    keepdims: Literal[True],
    layout: TensorLayout | None = None,
) -> NDArray[np.float64]: ...


@overload
def reduce_sum(
    arr: NDArray[np.float16] | NDArray[np.float32],
    axis: Literal["per_channel"],
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> NDArray[np.float64]: ...


@overload
def reduce_sum(
    arr: NDArray[np.float16] | NDArray[np.float32],
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> np.float64 | NDArray[np.float64]: ...


@overload
def reduce_sum(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> np.generic | np.ndarray: ...


@overload
def reduce_sum(
    arr: torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> torch.Tensor: ...


def reduce_sum(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    layout: TensorLayout | None = None,
) -> np.generic | np.ndarray | torch.Tensor:
    r"""Sum over image tensor axes with benchmark-driven routing.

    Routing:
    - **uint8 global**: NumKong ``nk.sum`` at every size.
    - **uint8 per-channel**: NumKong ``nk.sum(arr, axis=spatial_axes)`` — single call over
      all spatial dimensions for all ranks (HWC, DHWC, NHWC, …); C > 9 with 65,536+ spatial elements uses
      OpenCV ``reduce``.
    - **contiguous uint8 CHW/NCHW Tensor, C > 4, 65,536+ spatial elements**: one ``nk.sum`` per channel plane.
    - **large float32 global (except Linux x86-64 and contiguous macOS ARM64) and per-channel C ≤ 4**: Torch float64
      accumulation through a zero-copy NumPy wrapper.
    - **contiguous float16 Tensor, 256+ elements**: direct Torch ``sum(dtype=float64)`` for global,
      per-channel, and explicit axes.
    - **contiguous NumPy float16 global, 20,480+ elements**: Torch ``sum(dtype=float64)`` through a zero-copy
      wrapper; tested ``keepdims=False`` calls matched NumPy float64 bitwise and ran at 0.879x-0.991x float32 time.
    - **other float16 global layouts and small arrays**: the default Torch threshold or NumPy float64 fallback.
    - **float16 per-channel HWC**: measured Torch windows for contiguous arrays. Rank-4 channel-last C=1 arrays up
      to 307,200 elements use Torch's float64 accumulator when ``keepdims=False``; C=3 arrays up to 245,760 elements
      use it with either ``keepdims`` value. Other ranks, channel counts and sizes keep the standard Torch threshold
      or NumPy wide-accumulator path. Contiguous C=9 arrays of any size widen once and use
      OpenCV ``CV_64F reduce`` because CV_16F reduce is unsupported.
    - **float32 global on Linux x86-64**: NumKong ``nk.sum`` (float64 accumulation) at every size.
    - **contiguous float32 global on macOS ARM64**: ``cv2.sumElems``, 0.52-0.69x of the Torch/NumPy time. It accumulates
      float32 blocks, so it can differ from ``np.sum(dtype=float64)`` by rounding: documented tolerance
      ``rtol=1e-6`` of the sum of absolute values (measured at most 1.7e-8). A non-finite total falls back to NumPy.
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


def _global_mean_std_uint8(arr: NDArray[np.uint8], eps: float) -> tuple[float, float]:
    s_sum, s_sq = nk.moments(arr)
    n = arr.size
    mean = float(s_sum) / n
    var = max(float(s_sq) / n - mean * mean, 0.0)
    return mean, float(np.sqrt(var)) + eps


def _global_mean_uint8_only(arr: NDArray[np.uint8]) -> float:
    """Global mean only: ``nk.sum`` / N (single reduction, no moments/sqrt work)."""
    return float(nk.sum(arr)) / arr.size


def _mean_std_per_channel_float64(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    axes = _per_channel_spatial_axes(arr)
    mean_values = np.asarray(np.mean(arr, axis=axes, dtype=np.float64), dtype=np.float64)
    std_values = np.asarray(np.std(arr, axis=axes, dtype=np.float64) + eps, dtype=np.float64)
    return mean_values, std_values


def _mean_std_per_channel_float32(arr: NDArray[np.float32], eps: float) -> tuple[np.ndarray, np.ndarray]:
    c = arr.shape[-1]
    if arr.ndim == 3 and 1 < c <= MAX_OPENCV_WORKING_CHANNELS:
        means, stds = cv2.meanStdDev(arr)
        mean_values, std_values = means[:, 0], stds[:, 0]
        # All-zero channels have exact moments; a nonzero mean with tiny variance may be cancellation.
        cancelled = [
            mean != 0 and std * std <= _FLOAT32_MOMENTS_CANCELLATION_FLOOR * mean * mean
            for mean, std in zip(mean_values.tolist(), std_values.tolist(), strict=True)
        ]
        if any(cancelled):
            # Variance ignores the shift, and near-constant pixels minus a nearby float32 mean subtract exactly.
            # Other channels shift by zero and keep their first-pass statistics.
            shift = np.where(cancelled, mean_values.astype(np.float32), np.float32(0)).astype(np.float64)
            offsets, centered_stds = cv2.meanStdDev(cv2.subtract(arr, (*shift.tolist(), *(0.0,) * (4 - c))))
            mean_values = np.where(cancelled, shift + offsets[:, 0], mean_values)
            std_values = np.where(cancelled, centered_stds[:, 0], std_values)
        return mean_values, std_values + eps

    n = arr.size // c
    means = np.empty(c, dtype=np.float64)
    stds = np.empty(c, dtype=np.float64)
    for channel in range(c):
        total, squared_total = nk.moments(arr[..., channel])
        mean_value = float(total) / n
        variance = max(float(squared_total) / n - mean_value * mean_value, 0.0)
        if variance <= _MOMENTS_RELATIVE_VARIANCE_FLOOR * mean_value * mean_value:
            return _mean_std_per_channel_float64(arr, eps)
        means[channel] = mean_value
        stds[channel] = np.sqrt(variance) + eps
    return means, stds


def _mean_std_per_channel_numkong(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel population mean/std via one NumKong moments call per channel."""
    if arr.dtype == np.float16:
        magnitude_bits = np.bitwise_and(arr.view(np.uint16), 0x7FFF)
        # Subtracting one maps zero to 0xffff and leaves subnormal magnitudes below 0x03ff.
        np.subtract(magnitude_bits, np.uint16(1), out=magnitude_bits)
        has_subnormals = np.min(magnitude_bits) < 0x03FF
        if has_subnormals:
            return _mean_std_per_channel_float32(_cast_float_numpy(arr, np.float32), eps)

    c = arr.shape[-1]
    n = arr.size // c
    means = np.empty(c, dtype=np.float64)
    stds = np.empty(c, dtype=np.float64)
    # Integer moments are exact; floating E[x²] - E[x]² cancels near constant input.
    if arr.dtype == np.float16:
        floor: float | None = _NUMKONG_MOMENTS_RELATIVE_VARIANCE_FLOOR
    elif arr.dtype == np.float32:
        floor = _FLOAT32_MOMENTS_CANCELLATION_FLOOR
    else:
        floor = None
    for channel in range(c):
        channel_values = arr[..., channel]
        s_sum, s_sq = nk.moments(channel_values)
        mean_value = float(s_sum) / n
        var = max(float(s_sq) / n - mean_value * mean_value, 0.0)
        # A zero sum of squares means an all-zero channel, whose moments are exact.
        if floor is not None and float(s_sq) != 0 and var <= floor * mean_value * mean_value:
            if arr.dtype == np.float32:
                # Variance ignores the shift, and near-constant pixels minus a nearby float32 mean subtract exactly.
                shift = np.float32(mean_value)
                offset_sum, offset_sq = nk.moments(np.subtract(channel_values, shift))
                offset = float(offset_sum) / n
                mean_value = float(shift) + offset
                var = max(float(offset_sq) / n - offset * offset, 0.0)
            else:
                mean_value = float(np.mean(channel_values, dtype=np.float64))
                var = float(np.var(channel_values, dtype=np.float64))
        means[channel] = mean_value
        stds[channel] = np.sqrt(var) + eps
    return means, stds


def _mean_std_per_channel_opencv(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen half inputs because CV_16F statistics lose subnormal precision."""
    is_half = arr.dtype == np.float16
    work = _cast_float_numpy(arr, np.float32) if is_half else arr
    mean, std = cv2.meanStdDev(work)
    if is_half and np.any(np.square(std[:, 0]) <= _MOMENTS_RELATIVE_VARIANCE_FLOOR * np.square(mean[:, 0])):
        return _mean_std_per_channel_numkong(arr, eps)
    mean_values = np.asarray(mean[:, 0], dtype=np.float64)
    std_values = np.asarray(std[:, 0] + eps, dtype=np.float64)
    return mean_values, std_values


def _mean_std_global_float16_opencv(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    eps: float,
) -> tuple[float, float] | None:
    """Widen contiguous float16 once so OpenCV accumulates exact float32 values in float64."""
    if arr.size < _FLOAT16_GLOBAL_STATS_OPENCV_MIN_ELEMENTS or not arr.flags.c_contiguous:
        return None
    return _mean_std_global_float32_opencv(_cast_float_numpy(arr, np.float32), eps)


def _mean_std_global_float16_fallback(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    eps: float,
) -> tuple[float, float]:
    mean_value = float(np.mean(arr, dtype=np.float64))
    std_value = float(np.std(arr, dtype=np.float64)) + eps
    return mean_value, std_value


def _mean_std_global_float32_opencv(
    arr: NDArray[np.float32],
    eps: float,
) -> tuple[float, float] | None:
    if (
        arr.size < _FLOAT32_GLOBAL_STATS_OPENCV_MIN_ELEMENTS
        or not arr.flags.c_contiguous
        or not arr.dtype.isnative
        or not (_LINUX_X86_64 or _MACOS_ARM64)
    ):
        return None

    mean_array, std_array = cv2.meanStdDev(arr.reshape(-1, 1))
    mean_value = float(mean_array[0, 0])
    std_value = float(std_array[0, 0])
    if (
        not np.isfinite(mean_value)
        or not np.isfinite(std_value)
        or std_value * std_value <= _FLOAT32_GLOBAL_STATS_RELATIVE_VARIANCE_FLOOR * mean_value * mean_value
    ):
        if std_value == 0 and np.ptp(arr) == 0:
            return mean_value, std_value + eps
        return None
    return mean_value, std_value + eps


def _mean_std_global_float16_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    elements = arr.numel()
    if (
        arr.dtype != torch.float16
        or not arr.is_contiguous()
        or layout is None
        or elements < _FLOAT16_GLOBAL_STATS_OPENCV_MIN_ELEMENTS
    ):
        return None
    image = arr.numpy()
    if _MACOS_ARM64 or (_LINUX_X86_64 and elements < _FLOAT16_GLOBAL_TENSOR_MOMENTS_MIN_ELEMENTS):
        # CV_16F statistics are slow on ARM and below the x86 threshold; widening wins there.
        stats = _mean_std_global_float16_opencv(image, eps)
        mean_value, std_value = _mean_std_global_float16_fallback(image, eps) if stats is None else stats
    elif elements >= _FLOAT16_GLOBAL_TENSOR_MOMENTS_MIN_ELEMENTS:
        mean_array, std_array = cv2.meanStdDev(image.reshape(-1, 1))
        mean_value = float(mean_array[0, 0])
        std_value = float(std_array[0, 0])
        if (
            not np.isfinite(mean_value)
            or not np.isfinite(std_value)
            or std_value * std_value <= _MOMENTS_RELATIVE_VARIANCE_FLOOR * mean_value * mean_value
        ):
            mean_value, std_value = _mean_std_global_float16_fallback(image, eps)
        else:
            std_value += eps
    else:
        return None
    return (
        torch.tensor(mean_value, dtype=torch.float64),
        torch.tensor(std_value, dtype=torch.float64),
    )


def _mean_std_global(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
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
        cv_stats = _mean_std_global_float32_opencv(cast("NDArray[np.float32]", arr), eps)
        if cv_stats is not None:
            mean_value, std_value = cv_stats
        elif _NUMPY_STD_MEAN_SUPPORTED:
            mean_keepdims = np.asarray(np.mean(arr, dtype=np.float64, keepdims=True), dtype=np.float64)
            mean_value = np.float64(mean_keepdims.reshape(())[()])
            std_value = np.std(arr, dtype=np.float64, mean=mean_keepdims) + eps
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


def _mean_global(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    *,
    keepdims: bool,
) -> np.floating | float | np.ndarray:
    if _is_uint8_image(arr):
        m = _global_mean_uint8_only(arr)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(m, dtype=np.float64).reshape(kd)
        return m
    if arr.dtype == np.float16 and arr.flags.c_contiguous and arr.size >= _FLOAT16_GLOBAL_MEAN_OPENCV_MIN_ELEMENTS:
        wide = _cast_float_numpy(arr, np.float32)
        mean_value = np.float64(cv2.mean(wide.reshape(-1, 1))[0])
        if keepdims:
            return mean_value.reshape((1,) * arr.ndim)
        return mean_value
    if _is_float32_image(arr):
        total = None
        if _LINUX_X86_64:
            total = np.float64(nk.sum(arr))
        elif _MACOS_ARM64 and arr.flags.c_contiguous:
            total = _float32_sum_elems(arr)
        if total is not None:
            mean_value = total / arr.size
            return mean_value.reshape((1,) * arr.ndim) if keepdims else mean_value
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
    channel_axis = _TENSOR_CHANNEL_AXES[layout]
    axes = tuple(axis for axis in range(arr.ndim) if axis != channel_axis)
    result = torch.mean(arr, dim=axes, dtype=torch.float64)
    if not keepdims:
        return result
    output_shape = [1] * arr.ndim
    output_shape[channel_axis] = arr.shape[channel_axis]
    return result.reshape(output_shape)


def _mean_per_channel_float32_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
) -> torch.Tensor | None:
    """Per-channel float32 mean as one float64-accumulating Torch reduction over channel-first storage."""
    if arr.dtype != torch.float32 or not arr.is_contiguous():
        return None
    # NCHW C=3 keeps the NumPy route, which is 5x faster than Torch on x86-64.
    if not (
        (layout == "CHW" and arr.shape[0] > 1)
        or (layout == "NCHW" and arr.shape[1] == _FLOAT32_TENSOR_NCHW_MEAN_CHANNELS)
    ):
        return None
    channel_axis = _TENSOR_CHANNEL_AXES[layout]
    axes = tuple(axis for axis in range(arr.ndim) if axis != channel_axis)
    return torch.mean(arr, dim=axes, dtype=torch.float64, keepdim=keepdims)


def _mean_std_per_channel_plane_moments(
    arr: NDArray[np.float32],
    axis: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Channel-last view of CHW storage: each ``arr[..., channel]`` is one contiguous plane."""
    means, stds = _mean_std_per_channel_numkong(arr, eps)
    if keepdims:
        shape = (1,) * (arr.ndim - 1) + (arr.shape[-1],)
        return means.reshape(shape), stds.reshape(shape)
    return means, stds


def _std_global_float16_tensor(
    arr: torch.Tensor,
    layout: TensorLayout | None,
    *,
    keepdims: bool,
    eps: float,
) -> torch.Tensor | None:
    if arr.dtype != torch.float16 or not arr.is_contiguous() or layout is None:
        return None
    stats = _mean_std_global_float16_opencv(arr.numpy(), eps)
    if stats is None:
        return None
    result = torch.tensor(stats[1], dtype=torch.float64)
    return result.reshape((1,) * arr.ndim) if keepdims else result


def _std_global(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    *,
    keepdims: bool,
    eps: float,
) -> np.floating | float | np.ndarray:
    std_value: np.floating | float
    if _is_uint8_image(arr):
        std_value = _global_mean_std_uint8(arr, eps)[1]
    elif arr.dtype == np.float16:
        cv_stats = _mean_std_global_float16_opencv(arr, eps)
        std_value = float(np.std(arr, dtype=np.float64)) + eps if cv_stats is None else cv_stats[1]
    else:
        cv_stats = _mean_std_global_float32_opencv(cast("NDArray[np.float32]", arr), eps)
        std_value = np.std(arr, dtype=np.float64) + eps if cv_stats is None else cv_stats[1]
    if keepdims:
        return np.asarray(std_value, dtype=np.float64).reshape((1,) * arr.ndim)
    return float(std_value)


def _mean_std_per_channel(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_axes = _per_channel_spatial_axes(arr)
    c = arr.shape[-1]
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
    if axes == spatial_axes and (
        # Uint8 keepdims shares the integer moments instead of a float64 two-pass reduction.
        _is_uint8_image(arr)
        or (
            not keepdims
            and (
                arr.dtype == np.float16
                or (_is_float32_image(arr) and (arr.ndim != 3 or c == 1 or c > MAX_OPENCV_WORKING_CHANNELS))
            )
        )
    ):
        mean_values, std_values = _mean_std_per_channel_numkong(arr, eps)
        if keepdims:
            shape = (1,) * (arr.ndim - 1) + (c,)
            return mean_values.reshape(shape), std_values.reshape(shape)
        return mean_values, std_values

    if arr.ndim == 3 and not keepdims and axes == spatial_axes and arr.shape[-1] <= MAX_OPENCV_WORKING_CHANNELS:
        return _mean_std_per_channel_float32(cast("NDArray[np.float32]", arr), eps)

    m = arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims)
    st = arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps
    return m, st


def _mean_per_channel_uint8(arr: NDArray[np.uint8], axes: tuple[int, ...], *, keepdims: bool) -> np.ndarray:
    """Per-channel uint8 mean: sumElems / n for contiguous HWC C=2..4, cv2.reduce for wide images, else nk.sum."""
    n_spatial = arr.size // arr.shape[-1]
    c = arr.shape[-1]
    result: np.ndarray
    if arr.ndim == 3 and 1 < c <= MAX_OPENCV_WORKING_CHANNELS and arr.flags.c_contiguous:
        # sumElems is an exact integer sum; cv2.mean multiplies by 1/n and differs from sum / n by one ulp.
        result = np.asarray(cv2.sumElems(arr)[:c], dtype=np.float64) / n_spatial
    elif c > _UINT8_NUMKONG_PER_CHANNEL_MAX_CHANNELS and n_spatial >= _UINT8_PER_CHANNEL_OPENCV_MIN_SPATIAL:
        result = _cv2_reduce_mean_per_channel(arr)
    else:
        result = np.asarray(nk.sum(arr, axis=axes), dtype=np.float64) / n_spatial
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (c,))
    return result


def _mean_per_channel_rank4(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> np.ndarray:
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
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> np.ndarray:
    spatial_axes = _per_channel_spatial_axes(arr)
    if _is_uint8_image(arr) and axes == spatial_axes:
        return _mean_per_channel_uint8(arr, axes, keepdims=keepdims)
    c = arr.shape[-1]
    # HWC, C ≤ 4: cv2.mean is fastest.
    if arr.ndim == 3 and not keepdims and axes == spatial_axes and c <= MAX_OPENCV_WORKING_CHANNELS:
        # CV_16F mean can lose precision on skewed and subnormal inputs.
        mu = cv2.mean(_cast_float_numpy(arr, np.float32)) if arr.dtype == np.float16 else cv2.mean(arr)
        return np.asarray(mu[:c], dtype=np.float64)
    if arr.ndim == 4 and axes == spatial_axes and 1 <= c <= MAX_OPENCV_WORKING_CHANNELS:
        return _mean_per_channel_rank4(arr, axes, keepdims=keepdims)
    if c > MAX_OPENCV_WORKING_CHANNELS and axes == spatial_axes and arr.size // c >= 16_384:
        work = _cast_float_numpy(arr, np.float32) if arr.dtype == np.float16 else arr
        result = _cv2_reduce_mean_per_channel(work)
        if keepdims:
            return result.reshape((1,) * (arr.ndim - 1) + (c,))
        return result
    return np.asarray(arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)


def _std_per_channel(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> np.ndarray:
    spatial_axes = _per_channel_spatial_axes(arr)
    c = arr.shape[-1]
    if keepdims and axes == spatial_axes and _is_uint8_image(arr):
        return _mean_std_per_channel_numkong(arr, eps)[1].reshape((1,) * (arr.ndim - 1) + (c,))
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
        return _mean_std_per_channel_float32(arr, eps)[1]
    if axes == spatial_axes and not keepdims:
        _, std = _mean_std_per_channel_numkong(arr, eps)
        return std
    return np.asarray(arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps, dtype=np.float64)


@overload
def mean_std(
    arr: NDArray[np.float32],
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[np.float64, np.float64]: ...


@overload
def mean_std(
    arr: NDArray[np.uint8] | NDArray[np.float16],
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[float, float]: ...


@overload
def mean_std(
    arr: np.ndarray,
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[_StatsScalar, _StatsScalar]: ...


@overload
def mean_std(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: Literal[True],
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def mean_std(
    arr: np.ndarray,
    axis: Literal["per_channel"],
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def mean_std(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[_StatsResult, _StatsResult]: ...


@overload
def mean_std(
    arr: torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    layout: TensorLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def mean_std(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
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
    - **float32 global**: OpenCV ``meanStdDev`` on contiguous inputs from two elements on Linux x86-64 and macOS ARM64;
      low-variance and non-finite inputs use NumPy float64 reductions.
    - **contiguous float16 global, 4,096 elements and larger**: widen to float32 once and call OpenCV
      ``meanStdDev`` (float64 accumulation); near-constant and non-finite inputs use NumPy float64.
    - **contiguous float16 Tensor global, 4,096 elements and larger**: the same widened route on macOS ARM64 and,
      on Linux x86-64, below 20,480 elements; Linux keeps CV16F ``meanStdDev`` over a zero-copy scalar view from
      20,480 elements. Near-constant and non-finite inputs use NumPy float64.
    - **per-channel float16 HWC, C ≤ 4**: OpenCV ``meanStdDev`` after widening to float32; direct CV16F statistics
      can lose low-order precision and drop subnormals.
    - **per-channel uint8 and float16 C > 4**: NumKong moments per channel, with a stable two-pass fallback for
      near-constant half data.
    - **contiguous float32 CHW Tensor per-channel, 2 ≤ C ≤ 4**: one NumKong ``moments`` call per channel plane with
      the near-constant shift fallback.
    - **per-channel float32, ndim == 3, 2 ≤ C ≤ 4**: ``cv2.meanStdDev``; near-constant channels repeat it on
      mean-centered pixels.
    - **per-channel float32, C=1, C>4, or ndim>3**: one NumKong ``moments`` call per channel; near-constant
      channels repeat it on mean-centered pixels.
    - **uint8 per-channel with keepdims**: the same NumKong integer moments as ``keepdims=False``.
    - **other explicit axes / keepdims**: NumPy-compatible ``mean``/``std``.

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
        router: Callable[..., Any] = mean_std
        if (
            axis == "per_channel"
            and layout == "CHW"
            and arr.dtype == torch.float32
            and arr.is_contiguous()
            and 1 < arr.shape[0] <= _FLOAT32_TENSOR_PLANE_MOMENTS_MAX_CHANNELS
        ):
            router = _mean_std_per_channel_plane_moments
        return cast(
            "tuple[np.floating | float | np.ndarray | torch.Tensor, np.floating | float | np.ndarray | torch.Tensor]",
            dispatch_tensor_reduction(arr, layout, router, axis, {"keepdims": keepdims, "eps": eps}),
        )

    axes = _resolve_axes(arr, axis)
    if axes is None:
        return _mean_std_global(arr, keepdims=keepdims, eps=eps)
    return _mean_std_per_channel(arr, axes, keepdims=keepdims, eps=eps)


@overload
def mean(
    arr: NDArray[np.uint8],
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> float: ...


@overload
def mean(
    arr: NDArray[np.float16] | NDArray[np.float32],
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> np.float64: ...


@overload
def mean(
    arr: np.ndarray,
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> _StatsScalar: ...


@overload
def mean(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: Literal[True],
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float64]: ...


@overload
def mean(
    arr: np.ndarray,
    axis: Literal["per_channel"],
    *,
    keepdims: bool = False,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float64]: ...


@overload
def mean(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: np.dtype[_OutputDTypeT] | type[_OutputDTypeT],
    layout: TensorLayout | None = None,
) -> NDArray[_OutputDTypeT]: ...


@overload
def mean(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: DTypeLike,
    layout: TensorLayout | None = None,
) -> _StatsScalar | np.ndarray: ...


@overload
def mean(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> _StatsResult: ...


@overload
def mean(
    arr: torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = None,
    layout: TensorLayout | None = None,
) -> torch.Tensor: ...


def mean(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = None,
    layout: TensorLayout | None = None,
) -> np.floating | float | np.ndarray | torch.Tensor:
    """Compute population mean over image tensor axes.

    Routing:
    - **uint8 global**: NumKong ``nk.sum(arr) / N`` (single reduction; avoids moments/sqrt work).
    - **uint8 per-channel**: NumKong ``nk.sum(spatial_axes) / n``; contiguous HWC with C=2..4 uses
      ``cv2.sumElems / n`` (bit-equal, 0.9x time on M4 Max) and C > 9 with 65,536+ spatial elements uses OpenCV
      ``reduce``. Contiguous uint8 CHW/NCHW Tensors with C > 4 and 65,536+ spatial elements sum per channel plane.
    - **float32 global**: ``nk.sum / N`` on Linux x86-64, ``cv2.sumElems / N`` for contiguous arrays on macOS ARM64
      (same ``rtol=1e-6`` tolerance as ``reduce_sum``), ``np.mean(dtype=float64)`` elsewhere.
    - **float32 per-channel, ndim==3, C ≤ 4**: ``cv2.mean`` (fastest for HWC images).
    - **float32 per-channel, ndim==4, 1 < C ≤ 4**: NumKong ``nk.sum / n`` — 4x over arr.mean
      for NHWC/DHWC batch/volume layouts.
    - **contiguous float16 global, 4,096 elements and larger**: widen to float32 once, then call ``cv2.mean`` on a
      single-column view.
    - **global float16 Tensor, 40,960 elements and larger, C>1**: one CPU Torch reduction with float32 accumulation;
      the scalar output remains float64.
    - **per-channel float16 Tensor, 40,960 elements and larger**: contiguous NCHW/NCDHW, CDHW, and multi-channel CHW
      inputs use a CPU Torch reduction with float64 accumulation; single-channel CHW keeps its measured OpenCV path.
    - **float16 per-channel HWC, C ≤ 4**: widen to CV_32F before ``cv2.mean`` because direct CV_16F reduction can
      exceed the float64-reference tolerance. Higher channel counts use a float32 reduction path.
    - **NumPy float16 per-channel, rank-4 C=3**: zero-copy Torch ``mean(dtype=float64)`` from 20,480 elements;
      rank-4 C=1 uses the global Torch float64 sum from 20,480 through 8,000,000 elements.
    - **contiguous float32 Tensor per-channel, CHW with C>1 or NCHW with C=9**: Torch ``mean(dtype=float64)``.
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
            if result is None:
                result = _mean_per_channel_float32_tensor(arr, layout, keepdims=keepdims)
            if result is not None:
                return result
            plane_sums = _uint8_tensor_channel_sums(arr, axis, layout)
            if plane_sums is not None:
                means = plane_sums[0].astype(np.float64) / plane_sums[1]
                return _channel_vector_to_tensor(means, arr, cast("TensorLayout", layout), keepdims=keepdims)
        return dispatch_tensor_reduction(arr, layout, mean, axis, {"keepdims": keepdims, "dtype": dtype})

    axes = _resolve_axes(arr, axis)
    m = _mean_global(arr, keepdims=keepdims) if axes is None else _mean_per_channel(arr, axes, keepdims=keepdims)
    if dtype is not None:
        return np.asarray(m, dtype=dtype)
    return m


@overload
def std(
    arr: np.ndarray,
    axis: _GlobalAxis = None,
    *,
    keepdims: Literal[False] = False,
    eps: float = DEFAULT_EPS,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> float: ...


@overload
def std(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: Literal[True],
    eps: float = DEFAULT_EPS,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float64]: ...


@overload
def std(
    arr: np.ndarray,
    axis: Literal["per_channel"],
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float64]: ...


@overload
def std(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: np.dtype[_OutputDTypeT] | type[_OutputDTypeT],
    layout: TensorLayout | None = None,
) -> NDArray[_OutputDTypeT]: ...


@overload
def std(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: DTypeLike,
    layout: TensorLayout | None = None,
) -> _StatsScalar | np.ndarray: ...


@overload
def std(
    arr: np.ndarray,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: None = None,
    layout: TensorLayout | None = None,
) -> _StatsResult: ...


@overload
def std(
    arr: torch.Tensor,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: DTypeLike | None = None,
    layout: TensorLayout | None = None,
) -> torch.Tensor: ...


def std(
    arr: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
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
    - **float32 global**: OpenCV ``meanStdDev`` on contiguous inputs from two elements on Linux x86-64 and macOS ARM64;
      low-variance and non-finite inputs use NumPy float64 reductions.
    - **per-channel uint8**: one NumKong ``moments`` call per channel.
    - **per-channel float32, ndim == 3, 2 ≤ C ≤ 4**: ``cv2.meanStdDev``; near-constant channels repeat it on
      mean-centered pixels.
    - **per-channel float32, C=1, C>4, or ndim>3**: one NumKong ``moments`` call per channel; near-constant
      channels repeat it on mean-centered pixels.
    - **per-channel float16 HWC, C ≤ 4**: widen to float32 before ``cv2.meanStdDev``.
    - **uint8 per-channel with keepdims**: the same NumKong integer moments as ``keepdims=False``.
    - **other explicit axes / keepdims**: ``arr.std(dtype=float64)``.

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
