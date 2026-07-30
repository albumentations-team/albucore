#!/usr/bin/env python3
"""Benchmark the public ``median_blur`` router across its dtype and kernel routes."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from shape_grids import ROUTER_HWC_FULL_HW, ROUTER_HWC_QUICK_HW
from timing import bench_wall_ms

import albucore

CHANNELS = (1, 3, 9)
DTYPES = (np.uint8, np.float32)
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


def _make_image(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
    dtype: type[np.uint8 | np.float32],
) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    cv2.setNumThreads(1)
    rng = np.random.default_rng(137)
    sizes = ROUTER_HWC_QUICK_HW if args.quick else ROUTER_HWC_FULL_HW
    rows: list[MedianBlurBenchRow] = []
    for height, width in sizes:
        for channels in CHANNELS:
            shape = (height, width, channels)
            for dtype in DTYPES:
                image = _make_image(rng, shape, dtype)
                for kernel_size in KERNEL_SIZES:
                    timing = bench_wall_ms(
                        lambda image=image, kernel_size=kernel_size: albucore.median_blur(image, kernel_size),
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
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

    payload = {
        "meta": {
            "albucore_version": albucore.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "quick": args.quick,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "rows": [asdict(row) for row in rows],
    }
    output = json.dumps(payload, indent=2)
    if args.output_json is None:
        print(output)  # noqa: T201
    else:
        args.output_json.write_text(output, encoding="utf-8")
        print(f"Wrote {args.output_json}")  # noqa: T201


if __name__ == "__main__":
    main()
