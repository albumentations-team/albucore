"""Benchmark-driven mean / std / mean_std for albucore array layouts (HWC, NHWC, NDHWC, …)."""

from typing import Literal, TypeGuard

import cv2
import numkong as nk
import numpy as np
from numpy.typing import DTypeLike

from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageFloat32, ImageType, ImageUInt8

DEFAULT_EPS = 1e-4

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


def _reduce_sum_global_uint8(arr: ImageUInt8, *, keepdims: bool) -> np.uint64 | np.ndarray:
    out = np.uint64(int(nk.sum(arr)))
    if keepdims:
        return np.array(out, dtype=np.uint64).reshape((1,) * arr.ndim)
    return out


def _reduce_sum_global_float32(arr: ImageFloat32, *, keepdims: bool) -> np.float64 | np.ndarray:
    out = np.sum(arr, dtype=np.float64)
    if keepdims:
        return np.asarray(out, dtype=np.float64).reshape((1,) * arr.ndim)
    return out


def _reduce_sum_per_channel_uint8(arr: ImageUInt8, *, keepdims: bool) -> np.ndarray:
    axes = _per_channel_spatial_axes(arr)
    result = np.asarray(nk.sum(arr, axis=axes), dtype=np.uint64)
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return result


def _reduce_sum_per_channel_float32(arr: ImageFloat32, axes: tuple[int, ...], *, keepdims: bool) -> np.ndarray:
    """Per-channel sum for float32: nk.sum for ndim ≤ 4 and 1 < C ≤ 4 (4x faster than numpy), numpy otherwise."""
    c = arr.shape[-1]
    if arr.ndim <= 4 and 1 < c <= MAX_OPENCV_WORKING_CHANNELS:
        result = np.asarray(nk.sum(arr, axis=axes), dtype=np.float64)
        if keepdims:
            return result.reshape((1,) * (arr.ndim - 1) + (c,))
        return result
    return np.asarray(np.sum(arr, axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)


def _reduce_sum_numpy(
    arr: np.ndarray,
    axes: tuple[int, ...] | None,
    *,
    keepdims: bool,
) -> np.uint64 | np.float64 | np.ndarray:
    if arr.dtype == np.uint8:
        acc: type = np.uint64
    elif np.issubdtype(arr.dtype, np.floating):
        acc = np.float64
    else:
        acc = np.int64
    return np.sum(arr, axis=axes, dtype=acc, keepdims=keepdims)


def reduce_sum(
    arr: ImageType,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
) -> np.uint64 | np.float64 | np.ndarray:
    r"""Sum over image tensor axes with benchmark-driven routing.

    Routing:
    - **uint8 global**: NumKong ``nk.sum`` (wide uint64 accumulator; avoids uint8 overflow).
    - **uint8 per-channel**: NumKong ``nk.sum(arr, axis=spatial_axes)`` — single call over
      all spatial dimensions for all ranks (HWC, DHWC, NHWC, …).
    - **float32 global**: ``numpy.sum(dtype=float64)``.
    - **float32 per-channel, ndim ≤ 4, 1 < C ≤ 4**: NumKong ``nk.sum`` — 4x faster than
      numpy for RGB/RGBA images and 4-D batch/volume layouts.
    - **float32 per-channel, C=1 or C>4, or explicit-axis cases**: ``numpy.sum(dtype=float64)``.

    ``axis="per_channel"`` reduces all spatial axes (everything except the last / channel dim),
    matching the convention used by :func:`mean` and :func:`std`.

    Alternative: ``mean`` / ``std`` / ``mean_std`` for normalised statistics.

    Args:
        arr: Array with explicit channel dimension. Optimised paths for ``uint8`` and ``float32``;
            other dtypes fall back to NumPy (float → float64 accumulator, integer/bool → int64).
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → shape ``(C,)``;
            or explicit ``int`` / ``tuple[int, ...]`` (NumPy path).
        keepdims: Same semantics as :func:`numpy.sum`.

    Returns:
        ``numpy.uint64`` or ``numpy.float64`` scalar for a full reduction, else an array.
    """
    axes = _resolve_axes(arr, axis)
    if axes is None:
        if _is_uint8_image(arr):
            return _reduce_sum_global_uint8(arr, keepdims=keepdims)
        if _is_float32_image(arr):
            return _reduce_sum_global_float32(arr, keepdims=keepdims)
    elif axes == _per_channel_spatial_axes(arr):
        if _is_uint8_image(arr):
            return _reduce_sum_per_channel_uint8(arr, keepdims=keepdims)
        if _is_float32_image(arr):
            return _reduce_sum_per_channel_float32(arr, axes, keepdims=keepdims)
    return _reduce_sum_numpy(arr, axes, keepdims=keepdims)


def _global_mean_std_uint8(arr: ImageUInt8, eps: float) -> tuple[float, float]:
    s_sum, s_sq = nk.moments(arr)
    n = arr.size
    mean = float(s_sum) / n
    var = max(float(s_sq) / n - mean * mean, 0.0)
    return mean, float(np.sqrt(var)) + eps


def _global_mean_uint8_only(arr: ImageUInt8) -> float:
    """Global mean only: ``nk.sum`` / N (single reduction, no moments/sqrt work)."""
    return float(nk.sum(arr)) / arr.size


def _mean_std_global(
    arr: ImageType,
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.floating | float | np.ndarray, np.floating | float | np.ndarray]:
    if _is_uint8_image(arr):
        m, s = _global_mean_std_uint8(arr, eps)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(m, dtype=np.float64).reshape(kd), np.array(s, dtype=np.float64).reshape(kd)
        return m, s
    if _is_float32_image(arr):
        mean_value = np.mean(arr, dtype=np.float64, keepdims=keepdims)
        std_value = np.std(arr, dtype=np.float64, keepdims=keepdims) + eps
        return mean_value, std_value
    raise ValueError(f"Unsupported dtype {arr.dtype} for mean_std; use uint8 or float32.")


def _mean_global(arr: ImageType, *, keepdims: bool) -> np.floating | float | np.ndarray:
    if _is_uint8_image(arr):
        m = _global_mean_uint8_only(arr)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(m, dtype=np.float64).reshape(kd)
        return m
    if _is_float32_image(arr):
        return np.mean(arr, dtype=np.float64, keepdims=keepdims)
    raise ValueError(f"Unsupported dtype {arr.dtype} for mean; use uint8 or float32.")


def _std_global(arr: ImageType, *, keepdims: bool, eps: float) -> np.floating | float | np.ndarray:
    if _is_uint8_image(arr):
        _, s = _global_mean_std_uint8(arr, eps)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(s, dtype=np.float64).reshape(kd)
        return s
    if _is_float32_image(arr):
        std_value = np.std(arr, dtype=np.float64, keepdims=keepdims)
        if keepdims:
            return np.asarray(std_value + eps, dtype=np.float64)
        return float(std_value) + eps
    raise ValueError(f"Unsupported dtype {arr.dtype} for std; use uint8 or float32.")


def _mean_std_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        arr.ndim == 3
        and not keepdims
        and axes == _per_channel_spatial_axes(arr)
        and arr.shape[-1] <= MAX_OPENCV_WORKING_CHANNELS
    ):
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
    result = np.asarray(nk.sum(arr, axis=axes), dtype=np.float64) / n_spatial
    if keepdims:
        return result.reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return result


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
    # float32 HWC, C ≤ 4: cv2.mean is fastest
    if arr.ndim == 3 and not keepdims and axes == spatial_axes and c <= MAX_OPENCV_WORKING_CHANNELS:
        mu = cv2.mean(arr)
        return np.asarray(mu[:c], dtype=np.float64)
    # float32 NHWC/DHWC (ndim == 4), 1 < C ≤ 4: nk.sum wins 4x vs arr.mean
    if arr.ndim == 4 and not keepdims and axes == spatial_axes and 1 < c <= MAX_OPENCV_WORKING_CHANNELS:
        n_spatial = arr.size // c
        return np.asarray(nk.sum(arr, axis=axes), dtype=np.float64) / n_spatial
    return np.asarray(arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims), dtype=np.float64)


def _std_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> np.ndarray:
    if (
        arr.ndim == 3
        and not keepdims
        and axes == _per_channel_spatial_axes(arr)
        and arr.shape[-1] <= MAX_OPENCV_WORKING_CHANNELS
    ):
        _, std = cv2.meanStdDev(arr)
        return np.asarray(std[:, 0] + eps, dtype=np.float64)
    return np.asarray(arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps, dtype=np.float64)


def mean_std(
    arr: ImageType,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
) -> tuple[np.floating | float | np.ndarray, np.floating | float | np.ndarray]:
    """Compute population mean and standard deviation (+ eps) jointly.

    More efficient than calling ``mean`` and ``std`` separately when both are needed,
    because the uint8 global path uses a single ``nk.moments`` pass.

    Routing:
    - **uint8 global**: NumKong ``nk.moments`` (single pass, wide accumulator).
    - **float32 global**: ``np.mean`` + ``np.std`` (``dtype=float64`` for accuracy).
    - **per-channel, ndim == 3, C ≤ 4**: ``cv2.meanStdDev`` (any dtype).
    - **everything else**: NumPy ``mean``/``std``.

    Alternative: ``mean`` or ``std`` if only one statistic is needed.

    Args:
        arr: uint8 or float32 image/batch/volume with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar pair; ``"per_channel"`` → ``(C,)`` pair;
            explicit ``int`` / ``tuple[int, ...]`` for NumPy-style axes.
        keepdims: Preserve reduced dimensions (same semantics as NumPy).
        eps: Added to std to prevent division-by-zero (default ``1e-4``).

    Returns:
        ``(mean, std + eps)`` — scalars for global reduction, arrays for per-channel.
    """
    axes = _resolve_axes(arr, axis)
    if axes is None:
        return _mean_std_global(arr, keepdims=keepdims, eps=eps)
    return _mean_std_per_channel(arr, axes, keepdims=keepdims, eps=eps)


def mean(
    arr: ImageType,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    dtype: DTypeLike | None = None,
) -> np.floating | float | np.ndarray:
    """Compute population mean over image tensor axes.

    Routing:
    - **uint8 global**: NumKong ``nk.sum(arr) / N`` (single reduction; avoids moments/sqrt work).
    - **uint8 per-channel**: NumKong ``nk.sum(spatial_axes) / n`` — wins for all shapes
      and channel counts (2x over cv2.mean for HWC C=3, 40x for NHWC/DHWC).
    - **float32 global**: ``np.mean(dtype=float64)``.
    - **float32 per-channel, ndim==3, C ≤ 4**: ``cv2.mean`` (fastest for HWC images).
    - **float32 per-channel, ndim==4, 1 < C ≤ 4**: NumKong ``nk.sum / n`` — 4x over arr.mean
      for NHWC/DHWC batch/volume layouts.
    - **everything else**: ``arr.mean(dtype=float64)``.

    Alternative: ``mean_std`` if std is also needed (avoids a second pass for uint8).

    Args:
        arr: uint8 or float32 image/batch/volume with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → ``(C,)`` array;
            explicit ``int`` / ``tuple[int, ...]`` for NumPy-style axes.
        keepdims: Preserve reduced dimensions.
        dtype: Cast the output to this dtype (e.g. ``np.float32``).

    Returns:
        Scalar or array of float64 means (cast to ``dtype`` if provided).
    """
    axes = _resolve_axes(arr, axis)
    m = _mean_global(arr, keepdims=keepdims) if axes is None else _mean_per_channel(arr, axes, keepdims=keepdims)
    if dtype is not None:
        return np.asarray(m, dtype=dtype)
    return m


def std(
    arr: ImageType,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
    eps: float = DEFAULT_EPS,
    dtype: DTypeLike | None = None,
) -> np.floating | float | np.ndarray:
    """Compute population standard deviation (+ eps) over image tensor axes.

    Routing:
    - **uint8 global**: NumKong ``nk.moments`` (single pass, wide accumulator).
    - **float32 global**: ``np.std(dtype=float64)``.
    - **per-channel, ndim == 3, C ≤ 4**: ``cv2.meanStdDev`` (fastest for HWC images).
    - **everything else**: ``arr.std(dtype=float64)``.

    Alternative: ``mean_std`` if mean is also needed (avoids a redundant pass for uint8).

    Args:
        arr: uint8 or float32 image/batch/volume with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → ``(C,)`` array;
            explicit ``int`` / ``tuple[int, ...]`` for NumPy-style axes.
        keepdims: Preserve reduced dimensions.
        eps: Added to std to prevent division-by-zero (default ``1e-4``).
        dtype: Cast the output to this dtype (e.g. ``np.float32``).

    Returns:
        Scalar or array of float64 stds + eps (cast to ``dtype`` if provided).
    """
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
