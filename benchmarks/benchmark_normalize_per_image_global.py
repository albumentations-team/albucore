"""Compare float32 global normalization with OpenCV-backed shared statistics."""

# ruff: noqa: E402, INP001
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_output import output_path, write_json_atomic
from timing import bench_paired_wall_ms

from albucore.normalize import _center_scale_numpy, normalize_per_image
from albucore.stats import DEFAULT_EPS, _mean_std_global_float32_opencv

SHAPES = ((128, 160, 1), (128, 160, 3), (480, 640, 3), (768, 1024, 9), (4, 128, 160, 3))


def numpy_image_route(image: np.ndarray, out: np.ndarray | None) -> np.ndarray:
    center = image.mean()
    divisor = image.std() + DEFAULT_EPS
    return _center_scale_numpy(image, center, divisor, out=out)


def cv2_stats_route(image: np.ndarray, out: np.ndarray | None) -> tuple[np.ndarray, bool]:
    stats = _mean_std_global_float32_opencv(image, DEFAULT_EPS)
    if stats is None:
        return numpy_image_route(image, out), False
    center, divisor = stats
    result = _center_scale_numpy(image, np.float32(center), np.float32(divisor), out=out)
    return result, True


def candidate_normalize(image: np.ndarray, out: np.ndarray | None) -> np.ndarray:
    return cv2_stats_route(image, out)[0]


def _check_case(
    image: np.ndarray,
    *,
    with_output: bool,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    source = image.copy()
    output = np.empty_like(image) if with_output else None
    expected = numpy_image_route(image, None if output is None else output.copy())
    cv2_result, cv2_used = cv2_stats_route(image, None if output is None else output.copy())
    public_result = normalize_per_image(image, "image")
    np.testing.assert_allclose(cv2_result, expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(public_result, expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_array_equal(image, source)

    baseline_out = None if output is None else output.copy()
    candidate_out = None if output is None else output.copy()
    baseline = partial(numpy_image_route, image, baseline_out)
    candidate = partial(candidate_normalize, image, candidate_out)
    baseline_timing, candidate_timing = bench_paired_wall_ms(
        baseline,
        candidate,
        repeats=repeats,
        warmup=warmup,
    )
    baseline_ms, candidate_ms = baseline_timing.median, candidate_timing.median
    if output is not None:
        candidate_result = candidate_normalize(image, output)
        if candidate_result is not output:
            raise AssertionError("normalization candidate did not return the supplied output")
    return {
        "shape": list(image.shape),
        "elements": image.size,
        "output": "supplied" if output is not None else "allocated",
        "cv2_stats_used": cv2_used,
        "baseline_ms": baseline_ms,
        "baseline_samples_ms": baseline_timing.raw,
        "candidate_ms": candidate_ms,
        "candidate_samples_ms": candidate_timing.raw,
        "speedup": baseline_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)

    rng = np.random.default_rng(9917)
    rows = []
    for shape in SHAPES:
        image = rng.random(shape, dtype=np.float32)
        rows.extend(
            _check_case(image, with_output=with_output, repeats=args.repeats, warmup=args.warmup)
            for with_output in (False, True)
        )

    near_constant = np.full((128, 160, 3), np.float32(1e6), dtype=np.float32)
    near_constant[::3] = np.nextafter(np.float32(1e6), np.float32(np.inf))
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "threads": 1,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "paired_order": "baseline starts first; call order alternates every sample",
        "near_constant_fallback": _mean_std_global_float32_opencv(near_constant, DEFAULT_EPS) is None,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
