"""Compare previous and candidate uint8 routes for field ``multiply`` and ``box_blur``.

Both routes are timed directly, so the comparison is valid on platforms where the
public router has not adopted the candidate. Run from the repository root with:

    uv run python benchmarks/benchmark_uint8_field_multiply_box_blur.py --threads 1 \
      --output-json benchmarks/results/uint8-field-multiply-box-blur.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable
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
from albucore.utils import clip

SHAPES = ((128, 160), (240, 320), (480, 640), (768, 1024))
CHANNELS = (1, 3, 9)
BOX_KERNELS = (3, 5, 15)


def previous_box_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    working = image.copy(order="C")
    return cv2.blur(working, (ksize, ksize), dst=working)


def candidate_box_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.blur(image, (ksize, ksize), dst=np.empty_like(image, order="C"))


def previous_multiply(image: np.ndarray, field: np.ndarray) -> np.ndarray:
    product = cv2.multiply(image.astype(np.float32), field.astype(np.float32, copy=False))
    return clip(product.reshape(image.shape), np.uint8)


def candidate_multiply(image: np.ndarray, field: np.ndarray) -> np.ndarray:
    if field.dtype == np.uint8:
        return cv2.multiply(image, field).reshape(image.shape)
    product = cv2.multiply(image, field.astype(np.float32, copy=False), dtype=cv2.CV_32F)
    return clip(product.reshape(image.shape), np.uint8)


def _cases(rng: np.random.Generator) -> list[tuple[dict[str, object], Callable[[], object], Callable[[], object]]]:
    cases = []
    for height, width in SHAPES:
        for channels in CHANNELS:
            image = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
            padded = rng.integers(0, 256, (height + 8, width + 16, channels), dtype=np.uint8)
            crop = padded[4 : 4 + height, 8 : 8 + width]
            for ksize in BOX_KERNELS:
                for layout, source in (("contiguous", image), ("row_padded_crop", crop)):
                    cell = {"op": "box_blur", "shape": [height, width, channels], "ksize": ksize, "layout": layout}
                    cases.append(
                        (cell, partial(previous_box_blur, source, ksize), partial(candidate_box_blur, source, ksize)),
                    )
            fields = {
                "float32": rng.uniform(0.5, 1.5, image.shape).astype(np.float32),
                "uint8": rng.integers(0, 4, image.shape, dtype=np.uint8),
            }
            for field_dtype, field in fields.items():
                cell = {"op": "multiply_field", "shape": [height, width, channels], "field_dtype": field_dtype}
                cases.append(
                    (cell, partial(previous_multiply, image, field), partial(candidate_multiply, image, field)),
                )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    thread_settings = configure_libraries(torch, cv2, args.threads)
    rows = []
    for cell, previous_call, candidate_call in _cases(np.random.default_rng(907)):
        expected = previous_call()
        actual = candidate_call()
        np.testing.assert_array_equal(actual, expected)
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"{cell}: shape or dtype differs")
        previous, candidate = bench_paired_wall_ms(previous_call, candidate_call, repeats=args.repeats, warmup=5)
        row = {
            **cell,
            "previous_ms": previous.median,
            "previous_mad_ms": previous.mad,
            "candidate_ms": candidate.median,
            "candidate_mad_ms": candidate.mad,
            "speedup": previous.median / candidate.median,
        }
        rows.append(row)
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()

    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "versions": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv": cv2.__version__,
            "numkong": importlib.metadata.version("numkong"),
            "stringzilla": importlib.metadata.version("stringzilla"),
            "torch": importlib.metadata.version("torch"),
        },
        "paired_order": "previous starts first; call order alternates every sample",
        "source_paths": {
            module.__name__: str(Path(module.__file__).resolve()) for module in (core, cv2, np, nk, sz, torch)
        },
        "numkong_capabilities": nk.get_capabilities(),
        "threads": thread_settings,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
