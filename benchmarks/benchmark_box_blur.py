# ruff: noqa: INP001
"""Compare public FP32 box blur with the previous source-copy path.

Run from the repository root with:

    uv run python benchmarks/benchmark_box_blur.py --threads 1 \
      --output-json benchmarks/results/box-blur.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from functools import partial
from pathlib import Path

import cv2
import numkong as nk
import numpy as np
import stringzilla as sz
import torch
from benchmark_threads import THREAD_COUNT, configure_libraries
from timing import bench_paired_wall_ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_output import output_path

import albucore as core
from albucore import box_blur

SHAPES = ((128, 160, 1), (240, 320, 3), (480, 640, 3), (768, 1024, 9))


def previous_route(image: np.ndarray, kernel_size: int) -> np.ndarray:
    working = image.copy(order="C")
    return cv2.blur(working, (kernel_size, kernel_size), dst=working)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    thread_settings = configure_libraries(torch, cv2, args.threads)
    rng = np.random.default_rng(731)
    rows = []
    for shape in SHAPES:
        image = rng.uniform(0, 1, shape).astype(np.float32)
        before = image.copy()
        old_call = partial(previous_route, image, 5)
        new_call = partial(box_blur, image, 5)
        expected = old_call()
        result = new_call()
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(image, before)
        if np.shares_memory(result, image) or not result.flags.writeable:
            raise AssertionError("box_blur must return independent writable storage")
        old_timing, new_timing = bench_paired_wall_ms(old_call, new_call, repeats=31, warmup=5)
        rows.append(
            {
                "shape": shape,
                "baseline_ms": old_timing.median,
                "baseline_mad_ms": old_timing.mad,
                "baseline_samples_ms": old_timing.raw,
                "current_ms": new_timing.median,
                "current_mad_ms": new_timing.mad,
                "current_samples_ms": new_timing.raw,
                "speedup": old_timing.median / new_timing.median,
                "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
            },
        )

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv": cv2.__version__,
            "numkong": importlib.metadata.version("numkong"),
            "stringzilla": importlib.metadata.version("stringzilla"),
            "torch": importlib.metadata.version("torch"),
        },
        "paired_order": "baseline starts first; call order alternates every sample",
        "source_paths": {
            module.__name__: str(Path(module.__file__).resolve()) for module in (core, cv2, np, nk, sz, torch)
        },
        "numkong_capabilities": nk.get_capabilities(),
        "threads": thread_settings,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    for item in rows:
        print(  # noqa: T201
            f"{item['shape']}: {item['baseline_ms']:.3f} → {item['current_ms']:.3f} ms ({item['speedup']:.3f}x)",
        )


if __name__ == "__main__":
    main()
