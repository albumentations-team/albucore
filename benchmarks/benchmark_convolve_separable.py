# ruff: noqa: E402, INP001
"""Compare public 2D convolution with OpenCV separable filtering for rank-1 kernels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_threads import configure_libraries
from timing import bench_paired_wall_ms

import albucore
from albucore.filtering import _separable_float32_min_elements  # noqa: SLF001
from albucore.utils import clip

SHAPES = ((128, 160, 1), (128, 160, 3), (240, 320, 3), (480, 640, 3), (480, 640, 9))
KERNEL_SIZES = (3, 7, 9, 11, 15, 21, 31)


def _gaussian_kernel(size: int) -> np.ndarray:
    vector = cv2.getGaussianKernel(size, 0, cv2.CV_32F).reshape(-1)
    return np.outer(vector, vector)


def _filter2d_baseline(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    result = np.empty_like(image)
    cv2.filter2D(image, -1, kernel, dst=result, borderType=cv2.BORDER_DEFAULT)
    return clip(result, out=result)


def _measure_case(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    expected = _filter2d_baseline(image, kernel)

    def candidate() -> np.ndarray:
        return albucore.convolve(image, kernel)

    actual = candidate()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def baseline() -> np.ndarray:
        return _filter2d_baseline(image, kernel)

    baseline_timing, candidate_timing = bench_paired_wall_ms(
        baseline,
        candidate,
        repeats=repeats,
        warmup=warmup,
    )
    baseline_ms, candidate_ms = baseline_timing.median, candidate_timing.median
    return {
        "shape": list(image.shape),
        "kernel": list(kernel.shape),
        "dtype": image.dtype.name,
        "route_selected": bool(
            (
                (sys.platform == "linux" and platform.machine() == "x86_64")
                or (sys.platform == "darwin" and platform.machine() == "arm64")
            )
            and image.size >= (_separable_float32_min_elements(image.shape[-1], kernel.shape[0]) or float("inf"))
            and cv2.checkRange(image)[0]
        ),
        "baseline_ms": baseline_ms,
        "baseline_samples_ms": baseline_timing.raw,
        "separable_ms": candidate_ms,
        "separable_samples_ms": candidate_timing.raw,
        "speedup": baseline_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    thread_settings = configure_libraries(torch, cv2, 1)
    rng = np.random.default_rng(251121)
    rows = []
    for shape in SHAPES:
        image = rng.random(shape, dtype=np.float32)
        image.flags.writeable = False
        rows.extend(
            [
                _measure_case(
                    image,
                    _gaussian_kernel(kernel_size),
                    repeats=args.repeats,
                    warmup=args.warmup,
                )
                for kernel_size in KERNEL_SIZES
            ],
        )

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "threads": thread_settings,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "paired_order": "baseline starts first; call order alternates every sample",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
