"""Benchmark-driven mean / std / mean_std for albucore array layouts (HWC, NHWC, NDHWC, …)."""

from typing import Literal, cast

import cv2
import numkong as nk
import numpy as np
from numpy.typing import DTypeLike

from albucore.utils import ImageType, ImageUInt8

DEFAULT_EPS = 1e-4

AxisSpec = None | int | tuple[int, ...] | Literal["global", "per_channel"]

__all__ = ["DEFAULT_EPS", "mean", "mean_std", "std"]


def _resolve_axes(arr: np.ndarray, axis: AxisSpec) -> tuple[int, ...] | None:
    """None → reduce all axes; 'per_channel' → all but last."""
    if axis is None or axis == "global":
        return None
    if axis == "per_channel":
        return tuple(range(arr.ndim - 1))
    if isinstance(axis, int):
        return (axis,)
    return tuple(axis)


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
    return float(s_sum) / flat.size


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
    if arr.ndim == 3 and not keepdims:
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
    if arr.ndim == 3 and not keepdims:
        c = arr.shape[-1]
        mu = cv2.mean(arr)
        return cast("np.ndarray", np.array(mu[:c], dtype=np.float64, copy=False))
    return cast("np.ndarray", arr.mean(axis=axes, dtype=np.float64, keepdims=keepdims))


def _std_per_channel(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    eps: float,
) -> np.ndarray:
    if arr.ndim == 3 and not keepdims:
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
    """Population mean and (std + eps). ``axis='per_channel'`` reduces over all but channel."""
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
    """Population mean. Float32 uses a single ``np.mean`` pass (not ``mean_std``)."""
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
    """Population std (+ eps). Float32 uses a single ``np.std`` pass (not ``mean_std``)."""
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
