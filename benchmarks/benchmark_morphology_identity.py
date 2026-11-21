# ruff: noqa: E402, INP001
"""Compare identity morphology routes with the prior OpenCV kernels."""

from __future__ import annotations

import argparse
import importlib
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

filtering_mod = importlib.import_module("albucore.filtering")

SHAPES = ((128, 160, 1), (128, 160, 3), (480, 640, 3), (480, 640, 9))


def _kernel(kind: str) -> np.ndarray:
    if kind == "one_pixel":
        return np.ones((1, 1), dtype=np.uint8)
    kernel = np.zeros((3, 3), dtype=np.uint8)
    kernel[1, 1] = 1
    return kernel


def _legacy_morphology(image: np.ndarray, operation: str, kernel: np.ndarray) -> np.ndarray:
    if image.dtype.type is np.float16:
        working = albucore.convert_dtype(image, np.float32)
        getattr(cv2, operation)(working, kernel, dst=working, iterations=1)
        return albucore.convert_dtype(working, np.float16)
    if image.shape[-1] == 1:
        result = np.empty_like(image, order="C")
        return getattr(cv2, operation)(image, kernel, dst=result, iterations=1)
    return getattr(cv2, operation)(image, kernel, iterations=1)


def _measure_case(
    image: np.ndarray,
    operation: str,
    kernel: np.ndarray,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    public = getattr(albucore, operation)
    expected = _legacy_morphology(image, operation, kernel)
    candidate_result = public(image, kernel)
    np.testing.assert_array_equal(candidate_result, expected)
    if not candidate_result.flags.writeable or np.shares_memory(candidate_result, image):
        raise AssertionError("identity morphology route must return independent writable storage")
    route_probe = filtering_mod._identity_morphology_copy(image, kernel)  # noqa: SLF001
    route_selected = route_probe is not None
    del route_probe
    del expected, candidate_result

    def baseline() -> np.ndarray:
        return _legacy_morphology(image, operation, kernel)

    def candidate() -> np.ndarray:
        return public(image, kernel)

    baseline_timing, candidate_timing = bench_paired_wall_ms(
        baseline,
        candidate,
        repeats=repeats,
        warmup=warmup,
    )
    baseline_ms, candidate_ms = baseline_timing.median, candidate_timing.median
    return {
        "shape": list(image.shape),
        "dtype": image.dtype.name,
        "operation": operation,
        "kernel": list(kernel.shape),
        "route_selected": route_selected,
        "layout": "read-only C-contiguous",
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
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
        for dtype in (np.dtype(np.uint8), np.dtype(np.float16), np.dtype(np.float32)):
            image = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.dtype(np.uint8)
                else rng.random(shape, dtype=np.float32).astype(dtype)
            )
            image.setflags(write=False)
            for kind in ("one_pixel", "center_only"):
                kernel = _kernel(kind)
                rows.extend(
                    [
                        _measure_case(image, operation, kernel, repeats=args.repeats, warmup=args.warmup)
                        for operation in ("erode", "dilate")
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
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
