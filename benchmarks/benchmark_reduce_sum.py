# ruff: noqa: T201, B023, INP001
"""sum / mean / std routing: prod (albucore.stats) vs nk candidates vs NumPy.

Five operations, uint8 + float32, HWC / DHWC / NHWC shapes:
  - sum global
  - sum per-channel  (all spatial axes → shape (C,))
  - mean global      — three nk paths: nk.sum/n, nk.moments/n, prod (moments for uint8)
  - mean per-channel — nk.sum(spatial)/n vs prod vs numpy
  - std per-channel  — nk two-pass (sum, sumsq) vs prod vs numpy

No reshape or Tensor() wrapping — nk.sum and nk.moments accept N-D arrays directly
(numkong >= 7.4, fixes for N-D moments and tuple-axis sum merged in 7.4.x).

Run from repo root::

    uv run python benchmarks/benchmark_reduce_sum.py
"""

from __future__ import annotations

import argparse

import cv2
import numkong as nk
import numpy as np
from timing import median_ms

from albucore.stats import DEFAULT_EPS, mean, reduce_sum, std


# ── nk helpers — N-D arrays accepted directly, no reshape/Tensor needed ──────


def _nk_sum_global(arr: np.ndarray) -> float:
    return float(nk.sum(arr))


def _nk_sum_per_channel(arr: np.ndarray) -> np.ndarray:
    return np.asarray(nk.sum(arr, axis=tuple(range(arr.ndim - 1))))


def _cv2_reduce_sum_per_channel(arr: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    return cv2.reduce(flat, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).reshape(-1)


def _cv2_reduce_mean_per_channel(arr: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(arr).reshape(-1, arr.shape[-1])
    return cv2.reduce(flat, 0, cv2.REDUCE_AVG, dtype=cv2.CV_64F).reshape(-1)


def _nk_mean_via_sum(arr: np.ndarray) -> float:
    """Global mean: nk.sum / n."""
    return float(nk.sum(arr)) / arr.size


def _nk_mean_via_moments(arr: np.ndarray) -> float:
    """Global mean: nk.moments (single pass, also computes sum-of-squares)."""
    s, _ = nk.moments(arr)
    return float(s) / arr.size


def _nk_mean_per_channel(arr: np.ndarray) -> np.ndarray:
    """Per-channel mean: nk.sum over spatial axes / num_spatial_elements."""
    spatial = tuple(range(arr.ndim - 1))
    n = arr.size // arr.shape[-1]
    return np.asarray(nk.sum(arr, axis=spatial), dtype=np.float64) / n


def _nk_std_per_channel(arr: np.ndarray) -> np.ndarray:
    """Per-channel std (+eps) via nk.sum on x and x^2 over spatial axes."""
    spatial = tuple(range(arr.ndim - 1))
    n = arr.size // arr.shape[-1]
    s = np.asarray(nk.sum(arr, axis=spatial), dtype=np.float64)
    arr_f = arr.astype(np.float32, copy=False)
    s2 = np.asarray(nk.sum(arr_f * arr_f, axis=spatial), dtype=np.float64)
    var = np.maximum(s2 / n - (s / n) ** 2, 0.0)
    return np.sqrt(var) + DEFAULT_EPS


# ── table helpers ─────────────────────────────────────────────────────────────

HDR3 = "| dtype | shape | prod ms | nk_new ms | numpy ms | fastest |"
SEP3 = "|-------|-------|--------:|----------:|---------:|---------|"

HDR_CAND = "| dtype | shape | prod ms | cv2.reduce ms | nk_new ms | numpy ms | fastest |"
SEP_CAND = "|-------|-------|--------:|--------------:|----------:|---------:|---------|"

HDR4 = "| dtype | shape | prod ms | nk_sum/n ms | nk_moments/n ms | numpy ms | fastest |"
SEP4 = "|-------|-------|--------:|------------:|----------------:|---------:|---------|"


def _row3(
    dtype_name: str,
    shape: tuple[int, ...],
    t_prod: float,
    t_nk: float,
    t_np: float,
) -> str:
    times = {"prod": t_prod, "nk_new": t_nk, "numpy": t_np}
    fastest = min(times, key=times.__getitem__)
    shape_str = "×".join(str(x) for x in shape)
    return f"| {dtype_name} | {shape_str} | {t_prod:.4f} | {t_nk:.4f} | {t_np:.4f} | **{fastest}** |"


def _row_candidates(
    dtype_name: str,
    shape: tuple[int, ...],
    t_prod: float,
    t_cv2: float,
    t_nk: float,
    t_np: float,
) -> str:
    times = {"prod": t_prod, "cv2.reduce": t_cv2, "nk_new": t_nk, "numpy": t_np}
    fastest = min(times, key=times.__getitem__)
    shape_str = "×".join(str(x) for x in shape)
    return (
        f"| {dtype_name} | {shape_str} | {t_prod:.4f} | {t_cv2:.4f} | "
        f"{t_nk:.4f} | {t_np:.4f} | **{fastest}** |"
    )


def _row4(
    dtype_name: str,
    shape: tuple[int, ...],
    t_prod: float,
    t_sum: float,
    t_moments: float,
    t_np: float,
) -> str:
    times = {"prod": t_prod, "nk_sum/n": t_sum, "nk_moments/n": t_moments, "numpy": t_np}
    fastest = min(times, key=times.__getitem__)
    shape_str = "×".join(str(x) for x in shape)
    return f"| {dtype_name} | {shape_str} | {t_prod:.4f} | {t_sum:.4f} | {t_moments:.4f} | {t_np:.4f} | **{fastest}** |"


def _section3(title: str, rows: list[str]) -> None:
    print(f"\n### {title}\n")
    print(HDR3)
    print(SEP3)
    for r in rows:
        print(r)


def _section4(title: str, rows: list[str]) -> None:
    print(f"\n### {title}\n")
    print(HDR4)
    print(SEP4)
    for r in rows:
        print(r)


def _section_candidates(title: str, rows: list[str]) -> None:
    print(f"\n### {title}\n")
    print(HDR_CAND)
    print(SEP_CAND)
    for r in rows:
        print(r)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    reps, wu = args.repeats, args.warmup
    rng = np.random.default_rng(0)

    shapes = [
        (256, 256, 1),
        (256, 256, 3),
        (256, 256, 9),
        (512, 512, 1),
        (512, 512, 3),
        (512, 512, 9),
        (1024, 1024, 3),
        # DHWC
        (16, 128, 128, 3),
        (32, 128, 128, 3),
        # NHWC
        (4, 256, 256, 3),
    ]

    print(f"# reduce_sum / mean: prod vs nk candidates vs NumPy  (numkong {nk.__version__})\n")
    print("Median ms, lower is better. `prod` = current `albucore.stats` router.")

    for dtype, dname in [(np.uint8, "uint8"), (np.float32, "float32")]:
        acc = np.uint64 if dtype == np.uint8 else np.float64

        sum_global_rows, sum_pc_rows, mean_global_rows, mean_pc_rows, std_pc_rows = [], [], [], [], []

        for shape in shapes:
            arr = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.uint8
                else rng.random(shape, dtype=np.float32)
            )
            axes = tuple(range(arr.ndim - 1))

            # sum global
            t_prod = median_ms(lambda a=arr: reduce_sum(a, "global"), reps, wu)
            t_nk   = median_ms(lambda a=arr: _nk_sum_global(a), reps, wu)
            t_np   = median_ms(lambda a=arr, ac=acc: np.sum(a, dtype=ac), reps, wu)
            sum_global_rows.append(_row3(dname, shape, t_prod, t_nk, t_np))

            # sum per-channel
            t_prod = median_ms(lambda a=arr: reduce_sum(a, "per_channel"), reps, wu)
            t_cv2 = median_ms(lambda a=arr: _cv2_reduce_sum_per_channel(a), reps, wu)
            t_nk   = median_ms(lambda a=arr: _nk_sum_per_channel(a), reps, wu)
            t_np   = median_ms(lambda a=arr, ax=axes, ac=acc: np.sum(a, axis=ax, dtype=ac), reps, wu)
            sum_pc_rows.append(_row_candidates(dname, shape, t_prod, t_cv2, t_nk, t_np))

            # mean global — prod vs nk.sum/n vs nk.moments/n vs numpy
            t_prod    = median_ms(lambda a=arr: mean(a, "global"), reps, wu)
            t_sum     = median_ms(lambda a=arr: _nk_mean_via_sum(a), reps, wu)
            t_moments = median_ms(lambda a=arr: _nk_mean_via_moments(a), reps, wu)
            t_np      = median_ms(lambda a=arr: float(np.mean(a, dtype=np.float64)), reps, wu)
            mean_global_rows.append(_row4(dname, shape, t_prod, t_sum, t_moments, t_np))

            # mean per-channel — prod vs nk.sum(spatial)/n vs numpy
            t_prod = median_ms(lambda a=arr: mean(a, "per_channel"), reps, wu)
            t_cv2 = median_ms(lambda a=arr: _cv2_reduce_mean_per_channel(a), reps, wu)
            t_nk   = median_ms(lambda a=arr: _nk_mean_per_channel(a), reps, wu)
            t_np   = median_ms(lambda a=arr, ax=axes: np.mean(a, axis=ax, dtype=np.float64), reps, wu)
            mean_pc_rows.append(_row_candidates(dname, shape, t_prod, t_cv2, t_nk, t_np))

            # std per-channel — prod vs nk two-pass vs numpy
            t_prod = median_ms(lambda a=arr: std(a, "per_channel"), reps, wu)
            t_nk   = median_ms(lambda a=arr: _nk_std_per_channel(a), reps, wu)
            t_np   = median_ms(lambda a=arr, ax=axes: np.std(a, axis=ax, dtype=np.float64) + DEFAULT_EPS, reps, wu)
            std_pc_rows.append(_row3(dname, shape, t_prod, t_nk, t_np))

        print(f"\n## {dname}")
        _section3("sum global", sum_global_rows)
        _section_candidates("sum per-channel", sum_pc_rows)
        _section4("mean global  (prod=moments/n · nk_sum=sum/n · nk_moments=moments/n)", mean_global_rows)
        _section_candidates("mean per-channel  (nk_new = nk.sum(spatial)/n)", mean_pc_rows)
        _section3("std per-channel  (nk_new = sqrt(sum2/n - (sum/n)^2) + eps)", std_pc_rows)


if __name__ == "__main__":
    main()
