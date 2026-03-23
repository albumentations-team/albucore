"""Benchmark-driven mean / std / mean_std for albucore array layouts (HWC, NHWC, NDHWC, …)."""

from typing import Literal, cast

import cv2
import numkong as nk
import numpy as np
from numpy.typing import DTypeLike

from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageFloat32, ImageType, ImageUInt8

DEFAULT_EPS = 1e-4

AxisSpec = None | int | tuple[int, ...] | Literal["global", "per_channel"]

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


def _reduce_sum_global_uint8(arr: ImageUInt8, *, keepdims: bool) -> np.uint64 | np.ndarray:
    flat = np.ascontiguousarray(arr, dtype=np.uint8).reshape(-1)
    total = int(nk.sum(nk.Tensor(flat)))
    out = np.uint64(total)
    if keepdims:
        return np.array(out, dtype=np.uint64).reshape((1,) * arr.ndim)
    return out


def _reduce_sum_global_float32(arr: ImageFloat32, *, keepdims: bool) -> np.float64 | np.ndarray:
    out = np.sum(arr, dtype=np.float64)
    if keepdims:
        return np.asarray(out, dtype=np.float64).reshape((1,) * arr.ndim)
    return cast("np.float64", out)


def _reduce_sum_per_channel_uint8(arr: ImageUInt8, *, keepdims: bool) -> np.ndarray:
    # NumKong chain-reduce only pays off for 3D (HWC); for higher ranks the overhead of
    # multiple Tensor.sum() calls cancels the benefit — fall back to NumPy.
    if arr.ndim != 3:
        return _reduce_sum_numpy(arr, _per_channel_spatial_axes(arr), keepdims=keepdims)
    t = nk.Tensor(np.ascontiguousarray(arr, dtype=np.uint8))
    t = t.sum(axis=0, keepdims=True).sum(axis=1, keepdims=True) if keepdims else t.sum(axis=0).sum(axis=0)
    return np.asarray(t, dtype=np.uint64)


def _reduce_sum_numpy(
    arr: ImageType,
    axes: tuple[int, ...] | None,
    *,
    keepdims: bool,
) -> np.ndarray:
    acc = np.uint64 if arr.dtype == np.uint8 else np.float64
    return np.sum(arr, axis=axes, dtype=acc, keepdims=keepdims)


def reduce_sum(
    arr: ImageType,
    axis: AxisSpec = None,
    *,
    keepdims: bool = False,
) -> np.uint64 | np.float64 | np.ndarray:
    r"""Sum over image tensor axes with benchmark-driven routing.

    Routing:
    - **uint8 global**: NumKong ``Tensor.sum`` on flat bytes (wide uint64 accumulator; avoids
      uint8 overflow).
    - **uint8 per-channel, ndim == 3**: NumKong chained ``Tensor.sum`` over spatial axes
      (H, then W); falls back to NumPy for higher-rank tensors where the overhead outweighs
      the benefit.
    - **float32** or any other axis layout: ``numpy.sum(dtype=float64)``.

    ``axis="per_channel"`` reduces all spatial axes (everything except the last / channel dim),
    matching the convention used by :func:`mean` and :func:`std`.

    Alternative: ``mean`` / ``std`` / ``mean_std`` for normalised statistics.

    Args:
        arr: ``uint8`` or ``float32`` array with explicit channel dimension.
        axis: ``None`` / ``"global"`` → one scalar; ``"per_channel"`` → shape ``(C,)``;
            or explicit ``int`` / ``tuple[int, ...]`` (NumPy path).
        keepdims: Same semantics as :func:`numpy.sum`.

    Returns:
        ``numpy.uint64`` or ``numpy.float64`` scalar for a full reduction, else an array.
    """
    axes = _resolve_axes(arr, axis)
    if axes is None:
        if arr.dtype == np.uint8:
            return _reduce_sum_global_uint8(arr, keepdims=keepdims)
        return _reduce_sum_global_float32(cast("ImageFloat32", arr), keepdims=keepdims)
    if axes == _per_channel_spatial_axes(arr):
        if arr.dtype == np.uint8:
            return _reduce_sum_per_channel_uint8(arr, keepdims=keepdims)
        return _reduce_sum_numpy(cast("ImageFloat32", arr), axes, keepdims=keepdims)
    return _reduce_sum_numpy(arr, axes, keepdims=keepdims)


def _global_mean_std_uint8(arr: ImageUInt8, eps: float) -> tuple[float, float]:
    flat = np.ascontiguousarray(arr, dtype=np.uint8).reshape(-1)
    s_sum, s_sq = nk.moments(nk.Tensor(flat))
    n = flat.size
    mean = float(s_sum) / n
    var = max(float(s_sq) / n - mean * mean, 0.0)
    return mean, float(np.sqrt(var)) + eps


def _global_mean_uint8_only(arr: ImageUInt8) -> float:
    """Global mean only: one ``moments`` call, no sqrt."""
    flat = np.ascontiguousarray(arr, dtype=np.uint8).reshape(-1)
    s_sum, _s_sq = nk.moments(nk.Tensor(flat))
    num = float(cast("float | int", s_sum))
    return num / float(flat.size)


def _mean_std_global(
    arr: ImageType,
    *,
    keepdims: bool,
    eps: float,
) -> tuple[np.floating | float | np.ndarray, np.floating | float | np.ndarray]:
    if arr.dtype == np.uint8:
        m, s = _global_mean_std_uint8(arr, eps)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(m, dtype=np.float64).reshape(kd), np.array(s, dtype=np.float64).reshape(kd)
        return m, s
    if arr.dtype == np.float32:
        m = np.mean(arr, dtype=np.float64, keepdims=keepdims)
        st = np.std(arr, dtype=np.float64, keepdims=keepdims) + eps
        return cast("np.ndarray | np.floating", m), cast("np.ndarray | np.floating", st)
    raise ValueError(f"Unsupported dtype {arr.dtype} for mean_std; use uint8 or float32.")


def _mean_global(arr: ImageType, *, keepdims: bool) -> np.floating | float | np.ndarray:
    if arr.dtype == np.uint8:
        m = _global_mean_uint8_only(arr)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(m, dtype=np.float64).reshape(kd)
        return m
    if arr.dtype == np.float32:
        return cast("np.ndarray | np.floating", np.mean(arr, dtype=np.float64, keepdims=keepdims))
    raise ValueError(f"Unsupported dtype {arr.dtype} for mean; use uint8 or float32.")


def _std_global(arr: ImageType, *, keepdims: bool, eps: float) -> np.floating | float | np.ndarray:
    if arr.dtype == np.uint8:
        _, s = _global_mean_std_uint8(arr, eps)
        if keepdims:
            kd = (1,) * arr.ndim
            return np.array(s, dtype=np.float64).reshape(kd)
        return s
    if arr.dtype == np.float32:
        return cast(
            "np.ndarray | np.floating",
            np.std(arr, dtype=np.float64, keepdims=keepdims) + eps,
        )
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
        return cast("np.ndarray", m), cast("np.ndarray", st)

    m = arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims)
    st = arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps
    return cast("np.ndarray", m), cast("np.ndarray", st)


def _mean_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> np.ndarray:
    if (
        arr.ndim == 3
        and not keepdims
        and axes == _per_channel_spatial_axes(arr)
        and arr.shape[-1] <= MAX_OPENCV_WORKING_CHANNELS
    ):
        c = arr.shape[-1]
        mu = cv2.mean(arr)
        return cast("np.ndarray", np.asarray(mu[:c], dtype=np.float64))
    return cast("np.ndarray", arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims))


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
        return cast("np.ndarray", (std[:, 0] + eps).astype(np.float64, copy=False))
    return cast(
        "np.ndarray",
        arr.std(axis=axes, dtype=np.float64, keepdims=keepdims) + eps,
    )


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
    - **uint8 global**: NumKong ``moments`` (single pass over flat bytes; wide accumulator).
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
    - **uint8 global**: NumKong ``moments`` (single pass, wide accumulator; avoids sqrt).
    - **float32 global**: ``np.mean(dtype=float64)``.
    - **per-channel, ndim == 3, C ≤ 4**: ``cv2.mean`` (fastest for HWC images).
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
    - **uint8 global**: NumKong ``moments`` (single pass, wide accumulator).
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
