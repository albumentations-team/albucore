#!/usr/bin/env python3
"""Time routers listed in ``albucore.functions.__all__`` on synthetic arrays.

Layouts:

- **HWC** ``(H,W,C)`` for spatial, arithmetic, and statistics routers.
- **NHWC** ``(N,H,W,C)`` for an additional statistics batch grid.

The HWC grid intentionally uses non-square sizes so height/width swaps are visible.

Optional ``--with-geometric`` also times ``copy_make_border``, ``gaussian_blur3d``, ``pad3d``, ``remap``, ``remap3d``,
``remap_nearest``, ``resize``, ``resize3d``, ``separable_filter3d``, ``warp_affine``, ``warp_affine3d``,
``warp_perspective``, and ``warp_perspective_nearest`` (they live in ``albucore.geometric``, not
``functions.__all__``).

Run::

    uv run python benchmarks/benchmark_router_synthetic.py --threads 1 --output-json benchmarks/results/run.json

``--skip-ops`` excludes routers from timing (no rows), e.g. to match an older release that lacked
those exports. Compare JSONs with ``compare_router_json.py`` (intersection = shared ``ok/ok`` cells).

Reliability: default ``--repeats`` / ``--warmup`` are tuned for stable medians; JSON includes
``ms_std`` / ``ms_mad`` (spread of timed runs). ``sz_lut`` runs on ``img.copy()`` so the shared
bench image is not corrupted across iterations.

Workload realism: thunks use non-identity parameters where it matters (e.g. ``multiply_add`` with
per-channel factor/bias, ``normalize`` with ImageNet-scale mean/denom, ``power`` with exponents
away from 1.0) so timings reflect real affine / LUT work, not no-ops.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# isort: split
# This import sets BLAS/OpenMP environment variables before cv2, NumPy, and Torch load.
from benchmark_threads import THREAD_COUNT, configure_libraries  # noqa: I001

import cv2
import numkong as nk
import numpy as np
import torch
from shape_grids import (
    ROUTER_BATCH_STATS_FULL_HW,
    ROUTER_BATCH_STATS_QUICK_HW,
    ROUTER_HWC_FULL_HW,
    ROUTER_HWC_QUICK_HW,
)
from timing import WallTimingMs, bench_wall_ms

# When ``albucore.functions`` has no ``__all__`` in a previous release, use this
# so comparison runs time the same router names. Keep in sync with
# ``albucore/functions.py`` ``__all__``.
_FUNCTIONS_PUBLIC_ROUTERS_FALLBACK: tuple[str, ...] = (
    "add",
    "add_array",
    "add_constant",
    "add_vector",
    "add_weighted",
    "apply_uint8_lut",
    "exp",
    "float32_io",
    "from_float",
    "hflip",
    "log",
    "matmul",
    "mean",
    "mean_std",
    "median_blur",
    "multiply",
    "multiply_add",
    "multiply_by_array",
    "multiply_by_constant",
    "multiply_by_vector",
    "normalize",
    "normalize_per_image",
    "pairwise_distances_squared",
    "power",
    "reduce_sum",
    "std",
    "sqrt",
    "sz_lut",
    "to_float",
    "uint8_io",
    "vflip",
)


@dataclass
class BenchRow:
    op: str
    layout: str
    shape: tuple[int, ...]
    dtype: str
    ms_median: float | None
    status: str
    detail: str = ""
    ms_mean: float | None = None
    ms_std: float | None = None
    ms_mad: float | None = None
    timing_n: int | None = None


def _parse_skip_ops(raw: str) -> frozenset[str]:
    if not raw.strip():
        return frozenset()
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


def _git_head_short(repo_root: Path) -> str:
    try:
        cp = subprocess.run(  # noqa: S607
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except OSError:
        pass
    return "unknown"


def _make_img(rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if dtype == np.float16:
        return rng.random(shape, dtype=np.float32).astype(np.float16)
    return rng.random(shape, dtype=np.float32)


def _iter_hwc(quick: bool) -> Iterator[tuple[str, tuple[int, ...], np.dtype]]:
    """3D images with non-square H/W to expose height-width swaps."""
    channels = (1, 3) if quick else (1, 3, 9)
    sizes = ROUTER_HWC_QUICK_HW if quick else ROUTER_HWC_FULL_HW
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float16, np.float32):
                yield "HWC", (h, w, c), dtype


def _iter_batch_stats(quick: bool) -> Iterator[tuple[str, tuple[int, ...], np.dtype]]:
    """4D batch for stats only. Non-square H/W catches axis-order mistakes."""
    channels = (1, 3) if quick else (1, 3, 9)
    n = 4
    sizes = ROUTER_BATCH_STATS_QUICK_HW if quick else ROUTER_BATCH_STATS_FULL_HW
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float16, np.float32):
                yield "NHWC", (n, h, w, c), dtype


def _bench(
    alb: Any,
    img: np.ndarray,
    build: Callable[[Any, np.ndarray], Callable[[], object]],
    repeats: int,
    warmup: int,
) -> tuple[WallTimingMs | None, str, str]:
    try:
        fn = build(alb, img)
        return bench_wall_ms(fn, repeats=repeats, warmup=warmup), "ok", ""
    except Exception as e:  # noqa: BLE001 — benchmark harness
        return None, "error", f"{type(e).__name__}: {e}"


def _bench_interleaved_pair(
    first: Callable[[], object],
    second: Callable[[], object],
    repeats: int,
    warmup: int,
) -> tuple[WallTimingMs, WallTimingMs]:
    first_times: list[float] = []
    second_times: list[float] = []
    for iteration in range(warmup):
        calls = ((second, first) if iteration % 2 else (first, second))
        for fn in calls:
            fn()
    for iteration in range(repeats):
        calls = (
            ((first, first_times), (second, second_times))
            if iteration % 2 == 0
            else ((second, second_times), (first, first_times))
        )
        for fn, timings in calls:
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1000.0)

    def summarize(samples: list[float]) -> WallTimingMs:
        values = np.asarray(samples, dtype=np.float64)
        median = float(np.median(values))
        return WallTimingMs(
            raw=tuple(samples),
            median=median,
            mean=float(values.mean()),
            std=float(values.std(ddof=1)) if repeats > 1 else 0.0,
            mad=float(np.median(np.abs(values - median))),
            n=repeats,
        )

    return summarize(first_times), summarize(second_times)


def _bench_pair_rows(
    alb: Any,
    op_name: str,
    layout: str,
    shape: tuple[int, ...],
    build: Callable[[Any, np.ndarray], Callable[[], object]],
    half_img: np.ndarray,
    float32_img: np.ndarray,
    repeats: int,
    warmup: int,
) -> tuple[BenchRow, BenchRow]:
    try:
        half_fn = build(alb, half_img)
        float32_fn = build(alb, float32_img)
        half_timing, float32_timing = _bench_interleaved_pair(half_fn, float32_fn, repeats, warmup)
        return (
            _bench_row(op_name, layout, shape, "float16", half_timing, "ok", ""),
            _bench_row(op_name, layout, shape, "float32", float32_timing, "ok", ""),
        )
    except Exception as e:  # noqa: BLE001 — benchmark harness
        detail = f"{type(e).__name__}: {e}"
        return (
            BenchRow(op_name, layout, shape, "float16", None, "error", detail),
            BenchRow(op_name, layout, shape, "float32", None, "error", detail),
        )


def _bench_row(
    op_name: str,
    layout: str,
    shape: tuple[int, ...],
    dname: str,
    timing: WallTimingMs | None,
    st: str,
    det: str,
) -> BenchRow:
    if timing is not None and st == "ok":
        return BenchRow(
            op_name,
            layout,
            shape,
            dname,
            timing.median,
            st,
            det,
            timing.mean,
            timing.std,
            timing.mad,
            timing.n,
        )
    return BenchRow(op_name, layout, shape, dname, None, st, det)


def _channel_vector(img: np.ndarray) -> np.ndarray:
    c = img.shape[-1]
    if img.dtype == np.uint8:
        return np.full(c, 2, dtype=np.uint8)
    return np.linspace(0.875, 1.125, num=c, dtype=np.float32)


def _registry_geometric() -> list[tuple[str, Callable[[Any, np.ndarray], Callable[[], object]]]]:
    def cmb(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.copy_make_border(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)

        return thunk

    def rsz(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        nw, nh = max(w // 2, 1), max(h // 2, 1)

        def thunk() -> None:
            alb.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        return thunk

    def rsz3(alb: Any, img: np.ndarray) -> Callable[[], object]:
        depth = 5
        h, w = img.shape[-3], img.shape[-2]
        volume = np.repeat(img[np.newaxis, ...], depth, axis=0)
        target = (max(depth // 2, 1), max(h // 2, 1), max(w // 2, 1))

        def thunk() -> None:
            alb.resize3d(volume, target, interpolation=cv2.INTER_LINEAR)

        return thunk

    def pad3(alb: Any, img: np.ndarray) -> Callable[[], object]:
        volume = np.repeat(img[np.newaxis, ...], 5, axis=0)

        def thunk() -> None:
            alb.pad3d(volume, (1, 2, 2, 3, 3, 4), value=7)

        return thunk

    def gblur3(alb: Any, img: np.ndarray) -> Callable[[], object]:
        volume = np.repeat(img[np.newaxis, ...], 5, axis=0)

        def thunk() -> None:
            alb.gaussian_blur3d(volume, sigma=(0.75, 1.25, 1.75))

        return thunk

    def sep3(alb: Any, img: np.ndarray) -> Callable[[], object]:
        volume = np.repeat(img[np.newaxis, ...], 5, axis=0)
        kernels = tuple(np.array((1.0, 2.0, 1.0), dtype=np.float32) / 4.0 for _ in range(3))

        def thunk() -> None:
            alb.separable_filter3d(volume, kernels)

        return thunk

    def waff(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def thunk() -> None:
            alb.warp_affine(img, m, (w, h), flags=cv2.INTER_LINEAR)

        return thunk

    def waff3(alb: Any, img: np.ndarray) -> Callable[[], object]:
        depth = 5
        height, width = img.shape[-3], img.shape[-2]
        volume = np.repeat(img[np.newaxis, ...], depth, axis=0)
        matrix = np.array(
            ((0.95, 0.1, 0.0, 0.25), (0.0, 1.05, 0.1, -0.25), (0.05, 0.0, 1.0, 0.125)),
            dtype=np.float32,
        )
        target = max(depth // 2, 1), max(height // 2, 1), max(width // 2, 1)

        def thunk() -> None:
            alb.warp_affine3d(volume, matrix, target, interpolation=cv2.INTER_LINEAR)

        return thunk

    def rmp3(alb: Any, img: np.ndarray) -> Callable[[], object]:
        volume = np.repeat(img[np.newaxis, ...], 5, axis=0)
        depth, height, width, _ = volume.shape
        z, y, x = np.meshgrid(
            (np.arange(depth, dtype=np.float32) * 2 + 1) / depth - 1,
            (np.arange(height, dtype=np.float32) * 2 + 1) / height - 1,
            (np.arange(width, dtype=np.float32) * 2 + 1) / width - 1,
            indexing="ij",
        )
        sampling_grid = np.stack((x, y, z), axis=-1)

        def thunk() -> None:
            alb.remap3d(volume, sampling_grid, interpolation=cv2.INTER_LINEAR)

        return thunk

    def wper(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        m = np.eye(3, dtype=np.float32)

        def thunk() -> None:
            alb.warp_perspective(img, m, (w, h), flags=cv2.INTER_LINEAR)

        return thunk

    def wper_nearest(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        m = np.array([[0.97, 0.02, 1.25], [-0.03, 0.93, -1.75], [1e-4, -2e-4, 1.0]], dtype=np.float32)

        def thunk() -> None:
            alb.warp_perspective(img, m, (w, h), flags=cv2.INTER_NEAREST)

        return thunk

    def rmp(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        def thunk() -> None:
            alb.remap(img, mx, my, interpolation=cv2.INTER_LINEAR)

        return thunk

    def rmp_nearest(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        mx = np.ascontiguousarray(xx * np.float32(0.97) + np.float32(1.25))
        my = np.ascontiguousarray(yy * np.float32(0.93) - np.float32(1.75))

        def thunk() -> None:
            alb.remap(img, mx, my, interpolation=cv2.INTER_NEAREST)

        return thunk

    return [
        ("copy_make_border", cmb),
        ("gaussian_blur3d", gblur3),
        ("pad3d", pad3),
        ("resize", rsz),
        ("resize3d", rsz3),
        ("separable_filter3d", sep3),
        ("warp_affine", waff),
        ("warp_affine3d", waff3),
        ("remap3d", rmp3),
        ("warp_perspective", wper),
        ("warp_perspective_nearest", wper_nearest),
        ("remap", rmp),
        ("remap_nearest", rmp_nearest),
    ]


def _registry_volume_batches() -> list[tuple[str, Callable[[Any, np.ndarray], Callable[[], object]]]]:
    def warp_batch(alb: Any, img: np.ndarray) -> Callable[[], object]:
        depth = 5
        batch_size = 4
        image = img[:32, :40]
        height, width = image.shape[-3], image.shape[-2]
        volume = np.repeat(image[np.newaxis, ...], depth, axis=0)
        volumes = np.repeat(volume[np.newaxis, ...], batch_size, axis=0)
        matrix = np.array(
            ((0.95, 0.1, 0.0, 0.25), (0.0, 1.05, 0.1, -0.25), (0.05, 0.0, 1.0, 0.125)),
            dtype=np.float32,
        )
        target = max(depth // 2, 1), max(height // 2, 1), max(width // 2, 1)

        def thunk() -> None:
            alb.warp_affine3d(volumes, matrix, target, interpolation=cv2.INTER_LINEAR)

        return thunk

    def remap_batch(alb: Any, img: np.ndarray) -> Callable[[], object]:
        image = img[:32, :40]
        volume = np.repeat(image[np.newaxis, ...], 5, axis=0)
        volumes = np.repeat(volume[np.newaxis, ...], 4, axis=0)
        depth, height, width, _ = volume.shape
        z, y, x = np.meshgrid(
            (np.arange(depth, dtype=np.float32) * 2 + 1) / depth - 1,
            (np.arange(height, dtype=np.float32) * 2 + 1) / height - 1,
            (np.arange(width, dtype=np.float32) * 2 + 1) / width - 1,
            indexing="ij",
        )
        sampling_grid = np.stack((x, y, z), axis=-1)

        def thunk() -> None:
            alb.remap3d(volumes, sampling_grid, interpolation=cv2.INTER_LINEAR)

        return thunk

    return [("warp_affine3d_batch", warp_batch), ("remap3d_batch", remap_batch)]


def _registry_functions() -> list[tuple[str, Callable[[Any, np.ndarray], Callable[[], object]]]]:
    """``albucore.functions.__all__`` routers (except decorators handled separately)."""

    def add_c(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 3 if img.dtype == np.uint8 else 1.5

        def thunk() -> None:
            alb.add(img, v)

        return thunk

    def add_const(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 3 if img.dtype == np.uint8 else 1.5

        def thunk() -> None:
            alb.add_constant(img, v)

        return thunk

    def add_vec(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = _channel_vector(img)

        def thunk() -> None:
            alb.add_vector(img, v)

        return thunk

    def add_arr(alb: Any, img: np.ndarray) -> Callable[[], object]:
        noise = np.full_like(img, 2 if img.dtype == np.uint8 else 0.015625)

        def thunk() -> None:
            alb.add_array(img, noise)

        return thunk

    def apply_ul(alb: Any, img: np.ndarray) -> Callable[[], object]:
        lut_u8 = np.arange(256, dtype=np.uint8)

        def thunk() -> None:
            alb.apply_uint8_lut(img, lut_u8)

        return thunk

    def mul_c(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 1.0625 if img.dtype in (np.float16, np.float32) else 2

        def thunk() -> None:
            alb.multiply(img, v)

        return thunk

    def mul_const(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 1.0625 if img.dtype in (np.float16, np.float32) else 2

        def thunk() -> None:
            alb.multiply_by_constant(img, v)

        return thunk

    def mul_vec(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = _channel_vector(img)

        def thunk() -> None:
            alb.multiply_by_vector(img, v)

        return thunk

    def mul_arr(alb: Any, img: np.ndarray) -> Callable[[], object]:
        # ``multiply_opencv`` promotes uint8 image to float32; value must match OpenCV expectations.
        if img.dtype in (np.float16, np.float32):
            factor = np.full(img.shape, np.float32(1.015625), dtype=np.float32)
        else:
            factor = np.full(img.shape, np.float32(1.02), dtype=np.float32)

        def thunk() -> None:
            alb.multiply_by_array(img, factor)

        return thunk

    def aw(alb: Any, img: np.ndarray) -> Callable[[], object]:
        img2 = (img // 2) if img.dtype == np.uint8 else (img * 0.5)

        def thunk() -> None:
            alb.add_weighted(img, 0.5, img2.astype(img.dtype, copy=False), 0.5)

        return thunk

    def pw(alb: Any, img: np.ndarray) -> Callable[[], object]:
        # Non-trivial exponent (avoid near-identity 1.01 on float32).
        c = int(img.shape[-1])
        if img.dtype in (np.float16, np.float32):
            exp = np.float32(0.875)
        else:
            exp = np.array(
                [1.06 + 0.18 * (i / max(c - 1, 1)) for i in range(c)],
                dtype=np.float32,
            )

        def thunk() -> None:
            alb.power(img, exp)

        return thunk

    def ma(alb: Any, img: np.ndarray) -> Callable[[], object]:
        c = int(img.shape[-1])
        if img.dtype in (np.float16, np.float32):
            factor = np.array([1.125, 1.0, 0.875] * ((c + 2) // 3), dtype=np.float32)[:c]
            bias = np.array([0.25, -0.125, 0.125] * ((c + 2) // 3), dtype=np.float32)[:c]
        else:
            # Per-channel affine in float space; LUT path clips to uint8.
            factor = np.linspace(1.08, 1.22, num=c, dtype=np.float32)
            bias = np.linspace(3.0, 11.0, num=c, dtype=np.float32)

        def thunk() -> None:
            alb.multiply_add(img, factor, bias)

        return thunk

    def hf(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.hflip(img)

        return thunk

    def vf(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.vflip(img)

        return thunk

    def npi(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.normalize_per_image(img, "image")

        return thunk

    def norm(alb: Any, img: np.ndarray) -> Callable[[], object]:
        # Non-trivial mean / scale (ImageNet-style order of magnitude; per-channel).
        c = int(img.shape[-1])
        if img.dtype in (np.float16, np.float32):
            mean = np.linspace(0.375, 0.5, num=c, dtype=np.float32)
            denom = np.linspace(1.5, 1.75, num=c, dtype=np.float32)
        else:
            mean = np.linspace(95.0, 135.0, num=c, dtype=np.float32)
            denom = np.linspace(1.0 / 60.0, 1.0 / 45.0, num=c, dtype=np.float32)

        def thunk() -> None:
            alb.normalize(img, mean, denom)

        return thunk

    def tf(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.to_float(img)

        return thunk

    def ff(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.from_float(img, np.uint8)

        return thunk

    def exponential(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.exp(img)

        return thunk

    def logarithm(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.log(img)

        return thunk

    def square_root(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.sqrt(img)

        return thunk

    def mb(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.median_blur(img, 3)

        return thunk

    def lut(alb: Any, img: np.ndarray) -> Callable[[], object]:
        lut_u8 = np.arange(256, dtype=np.uint8)

        def thunk() -> None:
            # Copy: ``sz_lut`` defaults to in-place on its buffer; shared ``img`` must stay untouched.
            alb.sz_lut(img.copy(), lut_u8)

        return thunk

    def mn(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.mean(img, "global")

        return thunk

    def mn_pc(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.mean(img, "per_channel")

        return thunk

    def st(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.std(img, "global")

        return thunk

    def st_pc(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.std(img, "per_channel")

        return thunk

    def ms(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.mean_std(img, "global")

        return thunk

    def ms_pc(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.mean_std(img, "per_channel")

        return thunk

    def rs(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.reduce_sum(img, "global")

        return thunk

    def rs_pc(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.reduce_sum(img, "per_channel")

        return thunk

    # Order follows functions.__all__ (geometric optional elsewhere)
    return [
        ("add", add_c),
        ("add_array", add_arr),
        ("apply_uint8_lut", apply_ul),
        ("add_constant", add_const),
        ("add_vector", add_vec),
        ("add_weighted", aw),
        ("exp", exponential),
        ("multiply", mul_c),
        ("multiply_add", ma),
        ("multiply_by_array", mul_arr),
        ("multiply_by_constant", mul_const),
        ("multiply_by_vector", mul_vec),
        ("normalize", norm),
        ("normalize_per_image", npi),
        ("hflip", hf),
        ("log", logarithm),
        ("vflip", vf),
        ("median_blur", mb),
        ("matmul", add_c),  # placeholder — replaced by special-case
        ("pairwise_distances_squared", add_c),
        ("power", pw),
        ("mean", mn),
        ("mean_per_channel", mn_pc),
        ("mean_std", ms),
        ("mean_std_per_channel", ms_pc),
        ("reduce_sum", rs),
        ("reduce_sum_per_channel", rs_pc),
        ("std", st),
        ("std_per_channel", st_pc),
        ("sqrt", square_root),
        ("sz_lut", lut),
        ("to_float", tf),
        ("from_float", ff),
    ]


def _matmul_bench(alb: Any, dtype: np.dtype, repeats: int, warmup: int) -> BenchRow:
    rng = np.random.default_rng(0)
    if dtype == np.uint8:
        a = rng.integers(0, 16, size=(128, 64), dtype=np.uint8)
        b = rng.integers(0, 16, size=(64, 32), dtype=np.uint8)
    else:
        a = rng.random((128, 64), dtype=np.float32).astype(dtype)
        b = rng.random((64, 32), dtype=np.float32).astype(dtype)

    def thunk() -> None:
        alb.matmul(a, b)

    try:
        t = bench_wall_ms(thunk, repeats=repeats, warmup=warmup)
        return BenchRow(
            "matmul",
            "2D",
            (128, 64, 64, 32),
            dtype.name,
            t.median,
            "ok",
            "",
            t.mean,
            t.std,
            t.mad,
            t.n,
        )
    except Exception as e:  # noqa: BLE001
        return BenchRow("matmul", "2D", (128, 64, 64, 32), dtype.name, None, "error", f"{type(e).__name__}: {e}")


def _pairwise_bench(alb: Any, dtype: np.dtype, repeats: int, warmup: int) -> BenchRow:
    rng = np.random.default_rng(1)
    if dtype == np.uint8:
        x1 = rng.integers(0, 256, size=(24, 3), dtype=np.uint8)
        x2 = rng.integers(0, 256, size=(16, 3), dtype=np.uint8)
    else:
        x1 = rng.random((24, 3), dtype=np.float32).astype(dtype)
        x2 = rng.random((16, 3), dtype=np.float32).astype(dtype)

    def thunk() -> None:
        alb.pairwise_distances_squared(x1, x2)

    try:
        t = bench_wall_ms(thunk, repeats=repeats, warmup=warmup)
        return BenchRow(
            "pairwise_distances_squared",
            "points",
            (x1.shape[0], x2.shape[0], 3),
            dtype.name,
            t.median,
            "ok",
            "",
            t.mean,
            t.std,
            t.mad,
            t.n,
        )
    except Exception as e:  # noqa: BLE001
        return BenchRow(
            "pairwise_distances_squared",
            "points",
            (x1.shape[0], x2.shape[0], 3),
            dtype.name,
            None,
            "error",
            f"{type(e).__name__}: {e}",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-json", type=str, default="", help="Write full results JSON")
    p.add_argument("--quick", action="store_true", help="Smaller shape/channel grid (no 1024 HWC)")
    p.add_argument(
        "--with-geometric",
        action="store_true",
        help="Also benchmark geometric routers (not in functions.__all__)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=21,
        help="Timed iterations per cell (median + spread in JSON)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed iterations before measurement (JIT, caches, alloc)",
    )
    p.add_argument(
        "--skip-ops",
        type=str,
        default="",
        help="Comma-separated op names to omit entirely (no benchmark rows), e.g. mean,std,apply_uint8_lut",
    )
    p.add_argument(
        "--benchmark-label",
        type=str,
        default="",
        help="Stored in JSON metadata (for example, current tree vs previous release)",
    )
    p.add_argument(
        "--interleave-float16",
        action="store_true",
        help="Compare float16 and float32 public calls on the same half-quantized inputs in alternating order.",
    )
    p.add_argument("--threads", type=int, default=THREAD_COUNT, help="Torch/OpenCV/BLAS CPU thread count.")
    args = p.parse_args()

    thread_settings = configure_libraries(torch, cv2, args.threads)
    skip_ops = _parse_skip_ops(args.skip_ops)

    import albucore as alb  # noqa: PLC0415
    import albucore.functions as alb_fn  # noqa: PLC0415

    # Repo root of the albucore tree actually imported (main vs worktree).
    repo_root = Path(alb.__file__).resolve().parent.parent

    ver = getattr(alb, "__version__", "unknown")
    try:
        dist_ver = importlib.metadata.version("albucore")
    except importlib.metadata.PackageNotFoundError:
        dist_ver = ver

    rows: list[BenchRow] = []
    rng = np.random.default_rng(42)
    fn_reg = _registry_functions()
    # Drop placeholder entries for matmul / pairwise (special benches)
    fn_reg = [(n, b) for n, b in fn_reg if n not in ("matmul", "pairwise_distances_squared")]
    stats_ops = {
        "mean",
        "std",
        "mean_std",
        "reduce_sum",
        "mean_per_channel",
        "std_per_channel",
        "mean_std_per_channel",
        "reduce_sum_per_channel",
    }
    skip_uint8_only = {"apply_uint8_lut", "sz_lut"}
    skip_float_only = {"from_float"}

    declared = getattr(alb_fn, "__all__", None)
    if declared is not None:
        export_names = set(declared)
        functions_all_meta: list[str] = list(declared)
    else:
        export_names = set(_FUNCTIONS_PUBLIC_ROUTERS_FALLBACK)
        functions_all_meta = list(_FUNCTIONS_PUBLIC_ROUTERS_FALLBACK)

    for name in sorted(export_names):
        if name in ("float32_io", "uint8_io"):
            rows.append(
                BenchRow(name, "meta", (-1,), "n/a", None, "skip", "decorator factory, not an image router"),
            )

    geo_reg = _registry_geometric() + _registry_volume_batches() if args.with_geometric else []
    geometric_api_aliases = {
        "remap_nearest": "remap",
        "warp_perspective_nearest": "warp_perspective",
    }
    paired_half_images: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}

    for layout, shape, dtype in _iter_hwc(args.quick):
        pair_key = (layout, shape)
        if dtype == np.float32 and pair_key in paired_half_images:
            half_img = paired_half_images[pair_key]
            img = half_img.astype(np.float32)
        else:
            img = _make_img(rng, shape, dtype)
        if dtype == np.float16:
            paired_half_images[pair_key] = img
            if args.interleave_float16:
                continue
        dname = np.dtype(dtype).name
        half_img = paired_half_images.get(pair_key) if dtype == np.float32 else None
        if args.interleave_float16 and half_img is not None:
            for op_name, build in fn_reg:
                if (op_name not in export_names and op_name not in stats_ops) or op_name in skip_ops:
                    continue
                if op_name in skip_uint8_only:
                    for float_dtype in ("float16", "float32"):
                        rows.append(BenchRow(op_name, layout, shape, float_dtype, None, "skip", "needs uint8"))
                    continue
                if op_name not in stats_ops and not hasattr(alb, op_name):
                    for float_dtype in ("float16", "float32"):
                        rows.append(BenchRow(op_name, layout, shape, float_dtype, None, "skip", "missing API"))
                    continue
                rows.extend(
                    _bench_pair_rows(alb, op_name, layout, shape, build, half_img, img, args.repeats, args.warmup),
                )
            for op_name, build in geo_reg:
                if op_name in skip_ops or op_name in skip_uint8_only or op_name in skip_float_only:
                    continue
                api_name = geometric_api_aliases.get(op_name, op_name.removesuffix("_batch"))
                if not hasattr(alb, api_name):
                    for float_dtype in ("float16", "float32"):
                        rows.append(BenchRow(op_name, layout, shape, float_dtype, None, "skip", "missing API"))
                    continue
                rows.extend(
                    _bench_pair_rows(alb, op_name, layout, shape, build, half_img, img, args.repeats, args.warmup),
                )
            continue
        for op_name, build in fn_reg:
            if op_name not in export_names and op_name not in stats_ops:
                continue
            if op_name in skip_ops:
                continue
            if op_name in skip_uint8_only and dtype != np.uint8:
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs uint8"))
                continue
            if op_name in skip_float_only and dtype not in (np.float16, np.float32):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs floating input"))
                continue
            if op_name not in stats_ops and not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            t, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(_bench_row(op_name, layout, shape, dname, t, st, det))

        for op_name, build in geo_reg:
            if op_name in skip_ops:
                continue
            if op_name in skip_uint8_only and dtype != np.uint8:
                continue
            if op_name in skip_float_only and dtype not in (np.float16, np.float32):
                continue
            api_name = geometric_api_aliases.get(op_name, op_name.removesuffix("_batch"))
            if not hasattr(alb, api_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            t, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(_bench_row(op_name, layout, shape, dname, t, st, det))

    paired_half_batches: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    for layout, shape, dtype in _iter_batch_stats(args.quick):
        pair_key = (layout, shape)
        if dtype == np.float32 and pair_key in paired_half_batches:
            half_img = paired_half_batches[pair_key]
            img = half_img.astype(np.float32)
        else:
            img = _make_img(rng, shape, dtype)
        if dtype == np.float16:
            paired_half_batches[pair_key] = img
            if args.interleave_float16:
                continue
        half_img = paired_half_batches.get(pair_key) if dtype == np.float32 else None
        dname = np.dtype(dtype).name
        if args.interleave_float16 and half_img is not None:
            for op_name, build in fn_reg:
                if op_name not in stats_ops or op_name in skip_ops:
                    continue
                if op_name not in export_names and not op_name.endswith("_per_channel"):
                    continue
                if not op_name.endswith("_per_channel") and not hasattr(alb, op_name):
                    for float_dtype in ("float16", "float32"):
                        rows.append(BenchRow(op_name, layout, shape, float_dtype, None, "skip", "missing API"))
                    continue
                rows.extend(
                    _bench_pair_rows(alb, op_name, layout, shape, build, half_img, img, args.repeats, args.warmup),
                )
            continue
        for op_name, build in fn_reg:
            if op_name not in stats_ops:
                continue
            if op_name not in export_names and not op_name.endswith("_per_channel"):
                continue
            if op_name in skip_ops:
                continue
            if not op_name.endswith("_per_channel") and not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            t, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(_bench_row(op_name, layout, shape, dname, t, st, det))

    if "matmul" in export_names and "matmul" not in skip_ops:
        if args.interleave_float16:
            half_a = rng.random((128, 64), dtype=np.float32).astype(np.float16)
            half_b = rng.random((64, 32), dtype=np.float32).astype(np.float16)
            float32_a = half_a.astype(np.float32)
            float32_b = half_b.astype(np.float32)

            def matmul_half() -> object:
                return alb.matmul(half_a, half_b)

            def matmul_float32() -> object:
                return alb.matmul(float32_a, float32_b)

            try:
                timing_half, timing_float32 = _bench_interleaved_pair(
                    matmul_half,
                    matmul_float32,
                    args.repeats,
                    args.warmup,
                )
                rows.extend(
                    (
                        _bench_row("matmul", "2D", (128, 64, 64, 32), "float16", timing_half, "ok", ""),
                        _bench_row("matmul", "2D", (128, 64, 64, 32), "float32", timing_float32, "ok", ""),
                    ),
                )
            except Exception as e:  # noqa: BLE001 — benchmark harness
                detail = f"{type(e).__name__}: {e}"
                rows.extend(
                    (
                        BenchRow("matmul", "2D", (128, 64, 64, 32), "float16", None, "error", detail),
                        BenchRow("matmul", "2D", (128, 64, 64, 32), "float32", None, "error", detail),
                    ),
                )
        else:
            for dtype in (np.uint8, np.float16, np.float32):
                rows.append(_matmul_bench(alb, np.dtype(dtype), args.repeats, args.warmup))
    if "pairwise_distances_squared" in export_names and "pairwise_distances_squared" not in skip_ops:
        if args.interleave_float16:
            half_x1 = rng.random((24, 3), dtype=np.float32).astype(np.float16)
            half_x2 = rng.random((16, 3), dtype=np.float32).astype(np.float16)
            float32_x1 = half_x1.astype(np.float32)
            float32_x2 = half_x2.astype(np.float32)

            def pairwise_half() -> object:
                return alb.pairwise_distances_squared(half_x1, half_x2)

            def pairwise_float32() -> object:
                return alb.pairwise_distances_squared(float32_x1, float32_x2)

            try:
                timing_half, timing_float32 = _bench_interleaved_pair(
                    pairwise_half,
                    pairwise_float32,
                    args.repeats,
                    args.warmup,
                )
                rows.extend(
                    (
                        _bench_row("pairwise_distances_squared", "points", (24, 16, 3), "float16", timing_half, "ok", ""),
                        _bench_row("pairwise_distances_squared", "points", (24, 16, 3), "float32", timing_float32, "ok", ""),
                    ),
                )
            except Exception as e:  # noqa: BLE001 — benchmark harness
                detail = f"{type(e).__name__}: {e}"
                rows.extend(
                    (
                        BenchRow("pairwise_distances_squared", "points", (24, 16, 3), "float16", None, "error", detail),
                        BenchRow("pairwise_distances_squared", "points", (24, 16, 3), "float32", None, "error", detail),
                    ),
                )
        else:
            for dtype in (np.uint8, np.float16, np.float32):
                rows.append(_pairwise_bench(alb, np.dtype(dtype), args.repeats, args.warmup))

    meta = {
        "albucore_version": ver,
        "distribution_version": dist_ver,
        "benchmark_label": args.benchmark_label or None,
        "git_head_short": _git_head_short(repo_root),
        "skip_ops": sorted(skip_ops) if skip_ops else [],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "quick": args.quick,
        "with_geometric": args.with_geometric,
        "interleave_float16": args.interleave_float16,
        "functions_all": functions_all_meta,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "thread_settings": thread_settings,
        "timing_fields": {
            "ms_median": "median wall time (ms) over timed runs",
            "ms_mean": "mean wall time (ms)",
            "ms_std": "sample std (ms) of timed runs; use as error bar with median",
            "ms_mad": "median absolute deviation from median (robust spread)",
            "timing_n": "number of timed runs (equals repeats)",
            "paired_order": "float16/float32 calls alternate order for each timed sample when interleave_float16 is true",
        },
        "numpy": np.__version__,
        "numkong": importlib.metadata.version("numkong"),
        "numkong_capabilities": nk.get_capabilities(),
        "stringzilla": importlib.metadata.version("stringzilla"),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "container": "numpy",
    }
    payload = {"meta": meta, "rows": [asdict(r) for r in rows]}

    txt = json.dumps(payload, indent=2)
    if args.output_json:
        path = args.output_json
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"Wrote {path}", file=sys.stderr)
    else:
        print(txt)


if __name__ == "__main__":
    main()
