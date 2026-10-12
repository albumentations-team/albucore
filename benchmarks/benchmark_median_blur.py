#!/usr/bin/env python3
"""Benchmark public median-blur routes; pair float16 with float32 on identical values."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# isort: split
# This import sets BLAS/OpenMP environment variables before cv2, NumPy, and Torch load.
from benchmark_threads import THREAD_COUNT, configure_libraries  # noqa: I001

import cv2
import numpy as np
import torch
from shape_grids import ROUTER_HWC_FULL_HW, ROUTER_HWC_QUICK_HW
from timing import WallTimingMs, bench_wall_ms

import albucore

if TYPE_CHECKING:
    from collections.abc import Callable

CHANNELS = (1, 3, 9)
KERNEL_SIZES = (3, 5, 7)


@dataclass(frozen=True)
class MedianBlurBenchRow:
    """One public-router timing cell."""

    shape: tuple[int, int, int]
    dtype: str
    kernel_size: int
    ms_median: float
    ms_mean: float
    ms_std: float
    ms_mad: float
    timing_n: int


@dataclass(frozen=True)
class TensorMedianBlurBenchRow:
    """One channel-first public Tensor timing cell."""

    shape: tuple[int, ...]
    layout: str
    dtype: str
    kernel_size: int
    ms_median: float
    ms_mean: float
    ms_std: float
    ms_mad: float
    timing_n: int


@dataclass(frozen=True)
class MedianBlurDataPatternBenchRow:
    """One value-distribution timing cell for the single-channel half route."""

    shape: tuple[int, int, int]
    pattern: str
    dtype: str
    kernel_size: int
    ms_median: float
    ms_mean: float
    ms_std: float
    ms_mad: float
    timing_n: int


def _make_image(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    dtype: type[np.uint8 | np.float16 | np.float32],
) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32).astype(dtype)


def _bench_paired_wall_ms(
    half_call: Callable[[], object],
    float_call: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[WallTimingMs, WallTimingMs]:
    for _ in range(warmup):
        half_call()
        float_call()

    samples = ([], [])
    candidates = (half_call, float_call)
    for iteration in range(repeats):
        order = (0, 1) if iteration % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter()
            candidates[index]()
            samples[index].append((time.perf_counter() - start) * 1000.0)

    results = []
    for values in samples:
        times = np.asarray(values, dtype=np.float64)
        median = float(np.median(times))
        results.append(
            WallTimingMs(
                raw=tuple(values),
                median=median,
                mean=float(times.mean()),
                std=float(times.std(ddof=1)) if repeats > 1 else 0.0,
                mad=float(np.median(np.abs(times - median))),
                n=repeats,
            ),
        )
    return results[0], results[1]


def _tensor_image(array: np.ndarray, layout: str) -> torch.Tensor:
    if layout == "CHW":
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def _tensor_rows(
    sizes: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    repeats: int,
    warmup: int,
) -> list[TensorMedianBlurBenchRow]:
    rows = []
    for height, width in sizes:
        for layout, batch_size in (("CHW", 1), ("NCHW", 4)):
            half_images = rng.random((batch_size, height, width, 1), dtype=np.float32).astype(np.float16)
            float32_images = half_images.astype(np.float32)
            if layout == "CHW":
                half_tensor = _tensor_image(half_images[0], layout)
                float32_tensor = _tensor_image(float32_images[0], layout)
                tensor_shape = tuple(half_tensor.shape)
            else:
                half_tensor = _tensor_image(half_images, layout)
                float32_tensor = _tensor_image(float32_images, layout)
                tensor_shape = tuple(half_tensor.shape)

            half_call = lambda: albucore.median_blur(half_tensor, 5, layout=layout)
            float32_call = lambda: albucore.median_blur(float32_tensor, 5, layout=layout)
            half_result = half_call()
            float32_result = float32_call()
            if layout == "CHW":
                half_output = half_result.permute(1, 2, 0).numpy()
                float32_output = float32_result.permute(1, 2, 0).numpy()
            else:
                half_output = half_result.permute(0, 2, 3, 1).numpy()
                float32_output = float32_result.permute(0, 2, 3, 1).numpy()
            np.testing.assert_array_equal(
                half_output.view(np.uint16),
                float32_output.astype(np.float16).view(np.uint16),
            )

            half_timing, float32_timing = _bench_paired_wall_ms(
                half_call,
                float32_call,
                repeats=repeats,
                warmup=warmup,
            )
            for dtype, timing in ((np.float16, half_timing), (np.float32, float32_timing)):
                rows.append(
                    TensorMedianBlurBenchRow(
                        shape=tensor_shape,
                        layout=layout,
                        dtype=np.dtype(dtype).name,
                        kernel_size=5,
                        ms_median=timing.median,
                        ms_mean=timing.mean,
                        ms_std=timing.std,
                        ms_mad=timing.mad,
                        timing_n=timing.n,
                    ),
                )
    return rows


def _data_pattern_images(rng: np.random.Generator, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    negative_zero = np.full(shape, 0x8000, dtype=np.uint16).view(np.float16)
    mixed_zero_signs = np.resize(np.array([0x0000, 0x8000], dtype=np.uint16), shape).view(np.float16)
    sparse_positive_zeros = rng.random(shape, dtype=np.float32).astype(np.float16)
    sparse_positive_zeros[::2, ::2, :] = np.float16(0.0)
    return {
        "positive_zero": np.zeros(shape, dtype=np.float16),
        "negative_zero": negative_zero,
        "mixed_zero_signs": mixed_zero_signs,
        "sparse_positive_zeros": sparse_positive_zeros,
        "negative_random": -rng.random(shape, dtype=np.float32).astype(np.float16),
        "mixed_sign_random": rng.uniform(np.float32(-1.0), np.float32(1.0), size=shape).astype(np.float16),
    }


def _data_pattern_rows(
    sizes: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    repeats: int,
    warmup: int,
) -> list[MedianBlurDataPatternBenchRow]:
    rows = []
    for height, width in sizes:
        shape = (height, width, 1)
        for pattern, half_image in _data_pattern_images(rng, shape).items():
            float32_image = half_image.astype(np.float32)
            half_call = lambda image=half_image: albucore.median_blur(image, 5)
            float32_call = lambda image=float32_image: albucore.median_blur(image, 5)
            half_result = half_call()
            float32_result = float32_call()
            np.testing.assert_array_equal(
                half_result.view(np.uint16),
                float32_result.astype(np.float16).view(np.uint16),
            )
            half_timing, float32_timing = _bench_paired_wall_ms(
                half_call,
                float32_call,
                repeats=repeats,
                warmup=warmup,
            )
            for dtype, timing in ((np.float16, half_timing), (np.float32, float32_timing)):
                rows.append(
                    MedianBlurDataPatternBenchRow(
                        shape=shape,
                        pattern=pattern,
                        dtype=np.dtype(dtype).name,
                        kernel_size=5,
                        ms_median=timing.median,
                        ms_mean=timing.mean,
                        ms_std=timing.std,
                        ms_mad=timing.mad,
                        timing_n=timing.n,
                    ),
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--with-tensor", action="store_true", help="Also benchmark CHW/NCHW C=1 float16 ksize=5.")
    parser.add_argument("--with-data-patterns", action="store_true", help="Benchmark C=1 ksize=5 across zero/sign patterns.")
    args = parser.parse_args()

    thread_settings = configure_libraries(torch, cv2, args.threads)
    rng = np.random.default_rng(137)
    sizes = ROUTER_HWC_QUICK_HW if args.quick else ROUTER_HWC_FULL_HW
    rows: list[MedianBlurBenchRow] = []
    for height, width in sizes:
        for channels in CHANNELS:
            shape = (height, width, channels)
            float_source = rng.random(shape, dtype=np.float32)
            half_source = float_source.astype(np.float16)
            uint8_image = _make_image(rng, shape, np.uint8)
            float32_image = half_source.astype(np.float32)
            for kernel_size in KERNEL_SIZES:
                uint8_timing = bench_wall_ms(
                    lambda image=uint8_image, kernel_size=kernel_size: albucore.median_blur(image, kernel_size),
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                rows.append(
                    MedianBlurBenchRow(
                        shape=shape,
                        dtype=np.dtype(np.uint8).name,
                        kernel_size=kernel_size,
                        ms_median=uint8_timing.median,
                        ms_mean=uint8_timing.mean,
                        ms_std=uint8_timing.std,
                        ms_mad=uint8_timing.mad,
                        timing_n=uint8_timing.n,
                    ),
                )
                half_timing, float32_timing = _bench_paired_wall_ms(
                    lambda image=half_source, kernel_size=kernel_size: albucore.median_blur(image, kernel_size),
                    lambda image=float32_image, kernel_size=kernel_size: albucore.median_blur(image, kernel_size),
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                for dtype, timing in ((np.float16, half_timing), (np.float32, float32_timing)):
                    rows.append(
                        MedianBlurBenchRow(
                            shape=shape,
                            dtype=np.dtype(dtype).name,
                            kernel_size=kernel_size,
                            ms_median=timing.median,
                            ms_mean=timing.mean,
                            ms_std=timing.std,
                            ms_mad=timing.mad,
                            timing_n=timing.n,
                        ),
                    )

    tensor_rows = _tensor_rows(sizes, rng, args.repeats, args.warmup) if args.with_tensor else []
    data_pattern_rows = _data_pattern_rows(sizes, rng, args.repeats, args.warmup) if args.with_data_patterns else []

    payload = {
        "meta": {
            "albucore_version": albucore.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "torch_threads": torch.get_num_threads(),
            "thread_settings": thread_settings,
            "float32_uses_widened_float16_values": True,
            "float_pairs_are_interleaved": True,
            "quick": args.quick,
            "with_tensor": args.with_tensor,
            "with_data_patterns": args.with_data_patterns,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "rows": [asdict(row) for row in rows],
        "tensor_rows": [asdict(row) for row in tensor_rows],
        "data_pattern_rows": [asdict(row) for row in data_pattern_rows],
    }
    output = json.dumps(payload, indent=2)
    if args.output_json is None:
        print(output)  # noqa: T201
    else:
        args.output_json.write_text(output, encoding="utf-8")
        print(f"Wrote {args.output_json}")  # noqa: T201


if __name__ == "__main__":
    main()
