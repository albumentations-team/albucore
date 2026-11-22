# ruff: noqa: INP001
"""Compare the public 3D nearest-resize route with the prior OpenCV packing path.

Run from the repository root with:

    uv run python benchmarks/benchmark_resize3d_nearest.py --threads 1 \
      --output-json benchmarks/results/resize3d-nearest.json
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
from albucore import convert_dtype, resize3d
from albucore.geometric import _can_resize3d_joint_hw, _resize3d_numpy_axis_packing, _resize3d_numpy_joint_hw

CASES = (
    ((16, 128, 160, 3), (8, 65, 81)),
    ((32, 128, 160, 3), (17, 65, 81)),
    ((64, 128, 160, 3), (33, 65, 81)),
    ((48, 240, 320, 3), (25, 120, 160)),
)


def previous_route(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    def resize_backend(image: np.ndarray) -> np.ndarray:
        if _can_resize3d_joint_hw(image, size, antialias=False):
            return _resize3d_numpy_joint_hw(image, size, cv2.INTER_NEAREST, antialias=False)
        return _resize3d_numpy_axis_packing(image, size, cv2.INTER_NEAREST, antialias=False)

    if volume.dtype == np.float16:
        widened = convert_dtype(volume, np.float32)
        return convert_dtype(resize_backend(widened), np.float16)
    return resize_backend(volume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    thread_settings = configure_libraries(torch, cv2, args.threads)
    if platform.system() == "Darwin" and platform.machine() == "arm64" and not nk.get_capabilities().get("neon"):
        parser.error("Native macOS ARM64 measurements require NumKong NEON.")

    rng = np.random.default_rng(6041)
    rows = []
    for shape, size in CASES:
        float32 = rng.random(shape, dtype=np.float32)
        for dtype in (np.uint8, np.float16, np.float32):
            volume = rng.integers(0, 256, shape, dtype=np.uint8) if dtype is np.uint8 else float32.astype(dtype)
            source = volume.copy()
            old_call = partial(previous_route, volume, size)
            new_call = partial(resize3d, volume, size, interpolation=cv2.INTER_NEAREST)
            expected = old_call()
            actual = new_call()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(volume, source)
            old_timing, new_timing = bench_paired_wall_ms(old_call, new_call, repeats=31, warmup=5)
            rows.append(
                {
                    "shape": shape,
                    "size": size,
                    "dtype": np.dtype(dtype).name,
                    "baseline_ms": old_timing.median,
                    "baseline_mad_ms": old_timing.mad,
                    "baseline_samples_ms": old_timing.raw,
                    "current_ms": new_timing.median,
                    "current_mad_ms": new_timing.mad,
                    "current_samples_ms": new_timing.raw,
                    "speedup": old_timing.median / new_timing.median,
                    "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
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
        "source_paths": {
            module.__name__: str(Path(module.__file__).resolve()) for module in (core, cv2, np, nk, sz, torch)
        },
        "numkong_capabilities": nk.get_capabilities(),
        "threads": thread_settings,
        "paired_order": "baseline starts first; call order alternates every sample",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    for item in rows:
        print(  # noqa: T201
            f"{item['shape']} → {item['size']} {item['dtype']}: "
            f"{item['baseline_ms']:.3f} → {item['current_ms']:.3f} ms "
            f"({item['speedup']:.3f}x)",
        )


if __name__ == "__main__":
    main()
