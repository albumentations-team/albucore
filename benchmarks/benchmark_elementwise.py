#!/usr/bin/env python3
"""Benchmark the public ``exp``/``log``/``sqrt`` routers and candidate backends.

The standard matrix uses the canonical non-square HWC and DHWC shapes.
Additional sweeps cover tiny dispatch, Python ``math`` loops, strided inputs,
safe in-place calls, and NumKong ``minmax`` as an OpenCV-log correctness guard.

Run from the repository root::

    uv run python benchmarks/benchmark_elementwise.py \
      --output benchmarks/results/benchmark_elementwise.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import platform
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numkong as nk
import numpy as np
from timing import WallTimingMs, bench_wall_ms

import albucore as ac
from albucore.elementwise import (
    exp_numpy,
    exp_opencv,
    log_numpy,
    log_opencv,
    sqrt_numpy,
    sqrt_opencv,
)

Unary = Callable[..., np.ndarray]
HWC_SHAPES: tuple[tuple[int, int, int], ...] = tuple(
    (height, width, channels)
    for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024))
    for channels in (1, 3, 9)
)
DHWC_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
    (48, 240, 320, 3),
)
STANDARD_SHAPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    *(("HWC", shape) for shape in HWC_SHAPES),
    *(("DHWC", shape) for shape in DHWC_SHAPES),
)
TINY_SIZES: tuple[int, ...] = (1, 4, 16, 64, 256, 1_024, 4_096, 16_384, 65_536)
STRIDED_LOG_CHANNELS: tuple[int, ...] = tuple(range(1, 13))
STRIDED_LOG_SIZE_HW: tuple[tuple[int, int], ...] = (
    (8, 10),
    (16, 20),
    (32, 40),
    (64, 80),
    (96, 128),
    (128, 160),
    (192, 256),
    (240, 320),
)
FLOAT32_TINY = np.finfo(np.float32).tiny
PYTHON_MATH: dict[str, Callable[[float], float]] = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}


def _samples_to_timing(samples_ms: list[float]) -> WallTimingMs:
    samples = np.asarray(samples_ms, dtype=np.float64)
    median = float(np.median(samples))
    return WallTimingMs(
        median=median,
        mean=float(samples.mean()),
        std=float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        mad=float(np.median(np.abs(samples - median))),
        n=int(samples.size),
    )


def _bench_inplace(fn: Unary, source: np.ndarray, repeats: int, warmup: int) -> WallTimingMs:
    """Time only the in-place call; reset the owned work buffer outside the timer."""
    work = source.copy()
    for _ in range(warmup):
        np.copyto(work, source)
        fn(work, inplace=True)

    samples_ms: list[float] = []
    for _ in range(repeats):
        np.copyto(work, source)
        started = time.perf_counter()
        result = fn(work, inplace=True)
        samples_ms.append((time.perf_counter() - started) * 1_000.0)
        if result is not work:
            msg = f"{fn.__name__} did not preserve aliasing for an owned writable array"
            raise RuntimeError(msg)
    return _samples_to_timing(samples_ms)


def _input(operation: str, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    if operation == "exp":
        return np.linspace(-2.0, 2.0, size, dtype=np.float32).reshape(shape).copy()
    if operation == "log":
        return np.linspace(0.1, 4.0, size, dtype=np.float32).reshape(shape).copy()
    return np.linspace(0.0, 4.0, size, dtype=np.float32).reshape(shape).copy()


def _operations() -> dict[str, tuple[Unary, Unary, Unary]]:
    return {
        "exp": (ac.exp, exp_numpy, exp_opencv),
        "log": (ac.log, log_numpy, log_opencv),
        "sqrt": (ac.sqrt, sqrt_numpy, sqrt_opencv),
    }


def _format_timing(timing: WallTimingMs) -> str:
    return f"{timing.median:.4f} ± {timing.mad:.4f}"


def _benchmark_standard(repeats: int, warmup: int, *, inplace: bool) -> list[str]:
    rows = [
        "| operation | layout | shape | public ms | NumPy ms | OpenCV-safe ms | public / fastest |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, (public, numpy_fn, opencv_fn) in _operations().items():
        for layout, shape in STANDARD_SHAPES:
            array = _input(name, shape)
            if inplace:
                public_t = _bench_inplace(public, array, repeats, warmup)
                numpy_t = _bench_inplace(numpy_fn, array, repeats, warmup)
                opencv_t = _bench_inplace(opencv_fn, array, repeats, warmup)
            else:
                public_t = bench_wall_ms(lambda: public(array), repeats, warmup)
                numpy_t = bench_wall_ms(lambda: numpy_fn(array), repeats, warmup)
                opencv_t = bench_wall_ms(lambda: opencv_fn(array), repeats, warmup)
            fastest = min(numpy_t.median, opencv_t.median)
            rows.append(
                f"| {name} | {layout} | {'×'.join(map(str, shape))} | {_format_timing(public_t)} | "
                f"{_format_timing(numpy_t)} | {_format_timing(opencv_t)} | {public_t.median / fastest:.2f}× |",
            )
    return rows


def _benchmark_strided(repeats: int, warmup: int) -> list[str]:
    rows = [
        "| operation | logical shape | public ms | NumPy ms | OpenCV-safe ms | fastest |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, (public, numpy_fn, opencv_fn) in _operations().items():
        for shape in HWC_SHAPES:
            h, w, channels = shape
            array = _input(name, (h, w * 2, channels))[:, ::2, :]
            public_t = bench_wall_ms(lambda: public(array), repeats, warmup)
            numpy_t = bench_wall_ms(lambda: numpy_fn(array), repeats, warmup)
            opencv_t = bench_wall_ms(lambda: opencv_fn(array), repeats, warmup)
            fastest = "NumPy" if numpy_t.median <= opencv_t.median else "OpenCV-safe"
            rows.append(
                f"| {name} | {'×'.join(map(str, shape))} | {_format_timing(public_t)} | "
                f"{_format_timing(numpy_t)} | {_format_timing(opencv_t)} | {fastest} |",
            )
    return rows


def _benchmark_tiny(repeats: int, warmup: int) -> list[str]:
    rows = [
        "| operation | elements | public µs | NumPy µs | OpenCV-safe µs | Python math µs | fastest |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, (public, numpy_fn, opencv_fn) in _operations().items():
        for size in TINY_SIZES:
            array = _input(name, (size,))
            public_t = bench_wall_ms(lambda: public(array), repeats, warmup)
            numpy_t = bench_wall_ms(lambda: numpy_fn(array), repeats, warmup)
            opencv_t = bench_wall_ms(lambda: opencv_fn(array), repeats, warmup)
            math_fn = PYTHON_MATH[name]
            python_t = bench_wall_ms(
                lambda: np.asarray([math_fn(float(value)) for value in array], dtype=np.float32),
                repeats,
                warmup,
            )
            candidates = {"NumPy": numpy_t.median, "OpenCV-safe": opencv_t.median, "Python math": python_t.median}
            fastest = min(candidates, key=candidates.__getitem__)
            rows.append(
                f"| {name} | {size} | {public_t.median * 1_000:.3f} | {numpy_t.median * 1_000:.3f} | "
                f"{opencv_t.median * 1_000:.3f} | {python_t.median * 1_000:.3f} | {fastest} |",
            )
    return rows


def _benchmark_log_strided_channels(repeats: int, warmup: int) -> list[str]:
    rows = [
        "| logical HxW | channels | elements | NumPy ms | OpenCV-safe ms | OpenCV / NumPy | fastest |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)):
        for channels in STRIDED_LOG_CHANNELS:
            array = _input("log", (height, width * 2, channels))[:, ::2, :]
            numpy_t = bench_wall_ms(lambda: log_numpy(array), repeats, warmup)
            opencv_t = bench_wall_ms(lambda: log_opencv(array), repeats, warmup)
            fastest = "OpenCV-safe" if opencv_t.median < numpy_t.median else "NumPy"
            rows.append(
                f"| {height}x{width} | {channels} | {array.size} | {_format_timing(numpy_t)} | "
                f"{_format_timing(opencv_t)} | {opencv_t.median / numpy_t.median:.2f}x | {fastest} |",
            )
    return rows


def _benchmark_log_strided_sizes(repeats: int, warmup: int) -> list[str]:
    rows = [
        "| logical shape | elements | NumPy µs | OpenCV-safe µs | OpenCV / NumPy | fastest |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for channels in (1, 8):
        for height, width in STRIDED_LOG_SIZE_HW:
            array = _input("log", (height, width * 2, channels))[:, ::2, :]
            numpy_t = bench_wall_ms(lambda: log_numpy(array), repeats, warmup)
            opencv_t = bench_wall_ms(lambda: log_opencv(array), repeats, warmup)
            fastest = "OpenCV-safe" if opencv_t.median < numpy_t.median else "NumPy"
            rows.append(
                f"| {height}x{width}x{channels} | {array.size} | {numpy_t.median * 1_000:.3f} | "
                f"{opencv_t.median * 1_000:.3f} | {opencv_t.median / numpy_t.median:.2f}x | {fastest} |",
            )
    return rows


def _log_numkong_guard(array: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    minimum, _, maximum, _ = nk.minmax(array)
    safe = minimum >= FLOAT32_TINY and np.isfinite(minimum) and np.isfinite(maximum) and not np.isnan(array).any()
    if not safe:
        return np.log(array, out=array if inplace else None)
    result = cv2.log(array, dst=array if inplace else None)
    return array if inplace else result.reshape(array.shape)


def _benchmark_log_guards(repeats: int, warmup: int) -> list[str]:
    rows = [
        "| shape | NumPy log ms | NumPy min/max + OpenCV ms | NumKong minmax + NaN scan + OpenCV ms |",
        "|---:|---:|---:|---:|",
    ]
    for shape in HWC_SHAPES:
        array = _input("log", shape)
        numpy_t = bench_wall_ms(lambda: log_numpy(array), repeats, warmup)
        production_t = bench_wall_ms(lambda: log_opencv(array), repeats, warmup)
        numkong_t = bench_wall_ms(lambda: _log_numkong_guard(array), repeats, warmup)
        rows.append(
            f"| {'×'.join(map(str, shape))} | {_format_timing(numpy_t)} | {_format_timing(production_t)} | "
            f"{_format_timing(numkong_t)} |",
        )
    return rows


def _report(repeats: int, warmup: int) -> str:
    numkong_elementwise = [name for name in ("exp", "log", "sqrt") if hasattr(nk, name)]
    lines = [
        "# Elementwise exp/log/sqrt benchmark",
        "",
        "The public routers use OpenCV where the complete path wins without changing NumPy's special-value semantics.",
        "",
        "## Decision",
        "",
        "- `exp`: OpenCV at 4,096 elements for C-contiguous arrays and 65,536 elements for strided arrays; NumPy below those thresholds.",
        "- `log`: OpenCV at 4,096 elements for C-contiguous arrays, at 8,192 elements for strided single-channel arrays, and at 65,536 elements for strided arrays with C>=8. Inputs must be positive, finite, and normal float32 values. Two NumPy reductions guard the call; other inputs use NumPy.",
        "- `sqrt`: NumPy for every layout. `np.sqrt(..., out=array)` avoids allocation for safe in-place calls, and OpenCV showed no durable win.",
        "- NumKong: the installed Python API exposes none of `exp`, `log`, or `sqrt`. Its `minmax` primitive was tested as the `log` guard; the required NaN scan made it slower than NumPy `min`/`max`.",
        "- Python `math`: a per-element loop was tested on the tiny sweep and lost even at one element once array conversion and output allocation were included.",
        "",
        "## Optimization audit",
        "",
        "- Deleted or avoided work: empty inputs return immediately; `exp` and `sqrt` use one backend call; `log` uses two allocation-free reductions only on OpenCV candidates and avoids full-size masks.",
        "- Memory and vectorization: NumPy ufuncs and OpenCV kernels operate on whole arrays. No Python/channel loop or full-size dtype conversion is present in the public path.",
        "- Grouped reductions and randomness: `bincount` and random generators do not apply to deterministic pointwise float32 transforms.",
        "- LUT and StringZilla: neither applies to a continuous float32 domain. A finite uint8 table would change the requested dtype and semantics.",
        "- Backends: complete public, NumPy, and OpenCV paths are timed. NumKong has no direct primitives; its relevant `minmax` guard loses. Python `math` loses on tiny arrays.",
        "- Safe in-place: only owned writable arrays may alias the result. Views and read-only buffers allocate; strided owned buffers stay on NumPy where OpenCV cannot safely reuse them.",
        "- Reuse: the new atoms can replace local OpenCV/NumPy calls in AlbumentationsX histology and illumination helpers in a downstream follow-up.",
        "",
        "## Environment and method",
        "",
        f"Run date: {dt.date.today().isoformat()}. Platform: `{platform.platform()}` (`{platform.machine()}`). Python benchmark repeats: {repeats}; warmup: {warmup}; fixed deterministic linspace inputs.",
        "",
        f"Versions: NumPy `{np.__version__}`, OpenCV `{cv2.__version__}`, NumKong `{getattr(nk, '__version__', 'unknown')}`. NumKong direct elementwise matches found: `{numkong_elementwise or 'none'}`.",
        "",
        "Each table reports median ± median absolute deviation. The standard matrix uses the canonical non-square HWC grid with C=1/3/9 plus DHWC layouts. Allocating tables time the complete callable. In-place tables reset one owned writable buffer outside the timer, then time only the public or backend call.",
        "",
        "## Standard allocating matrix",
        "",
        *_benchmark_standard(repeats, warmup, inplace=False),
        "",
        "## Standard safe in-place matrix",
        "",
        *_benchmark_standard(repeats, warmup, inplace=True),
        "",
        "## Strided allocating matrix",
        "",
        *_benchmark_strided(repeats, warmup),
        "",
        "## Tiny dispatch sweep",
        "",
        *_benchmark_tiny(max(repeats, 41), max(warmup, 10)),
        "",
        "## Strided log channel sweep",
        "",
        "This sweep varies the channel count that controls whether OpenCV's required strided-input copy earns its cost.",
        "",
        *_benchmark_log_strided_channels(repeats, warmup),
        "",
        "### Strided log size calibration",
        "",
        "Single-channel and C>=8 are the winning regions above; this sweep locates conservative total-element thresholds.",
        "",
        *_benchmark_log_strided_sizes(max(repeats, 41), max(warmup, 10)),
        "",
        "## NumKong as a log correctness guard",
        "",
        "`cv2.log` changes negative, zero, subnormal, NaN, and infinity behavior. A guard must detect every such value before dispatch. NumKong `minmax` ignores NaN, so a separate NaN scan is required.",
        "",
        *_benchmark_log_guards(repeats, warmup),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = _report(args.repeats, args.warmup)
    if args.output is None:
        print(report)
        return
    args.output.write_text(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
