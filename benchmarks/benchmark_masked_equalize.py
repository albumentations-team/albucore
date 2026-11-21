# ruff: noqa: INP001
"""Compare masked RGB equalization with the original per-channel route.

Run from the repository root with:

    uv run python benchmarks/benchmark_masked_equalize.py --threads 1 \
      --output-json benchmarks/results/masked-equalize.json
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
from timing import bench_wall_ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_output import output_path

import albucore as core
from albucore import equalize_uint8
from albucore.histogram import _opencv_lut
from albucore.lut import apply_uint8_lut

CASES = ((128, 160), (240, 320), (480, 640))


def previous_masked_rgb_equalize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.empty_like(image)
    for channel in range(3):
        channel_mask = mask[..., channel] if mask.ndim == 3 else mask
        histogram = np.asarray(
            cv2.calcHist([image[..., channel]], [0], channel_mask, [256], (0, 256)),
            dtype=np.int64,
        ).ravel()
        lut = _opencv_lut(histogram)
        result[..., channel] = image[..., channel] if lut is None else apply_uint8_lut(image[..., channel], lut)
    return result


def row(image: np.ndarray, mask: np.ndarray, *, storage: str, repeats: int) -> dict[str, object]:
    source = image.copy()
    mask_before = mask.copy()
    previous = partial(previous_masked_rgb_equalize, image, mask)
    public = partial(equalize_uint8, image, mode="cv", mask=mask)
    expected = previous()
    result = public()
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, source)
    np.testing.assert_array_equal(mask, mask_before)
    if not result.flags.writeable or np.shares_memory(result, image):
        raise AssertionError("Masked equalization must return independent writable storage")
    old_timing = bench_wall_ms(previous, repeats=repeats, warmup=5)
    new_timing = bench_wall_ms(public, repeats=repeats, warmup=5)
    return {
        "shape": image.shape,
        "storage": storage,
        "mask_shape": mask.shape,
        "baseline_ms": old_timing.median,
        "baseline_mad_ms": old_timing.mad,
        "current_ms": new_timing.median,
        "current_mad_ms": new_timing.mad,
        "speedup": old_timing.median / new_timing.median,
        "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    thread_settings = configure_libraries(torch, cv2, args.threads)
    rng = np.random.default_rng(4417)
    rows = []
    for height, width in CASES:
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        shared = (rng.random((height, width)) < 0.5).astype(np.uint8) * 137
        per_channel = np.stack((shared, np.zeros_like(shared), np.full_like(shared, 137)), axis=-1)
        rows.append(row(image, shared, storage="contiguous", repeats=args.repeats))
        rows.append(row(image, per_channel, storage="contiguous", repeats=args.repeats))
        padded = np.empty((height, width + 2, 3), dtype=np.uint8)
        crop = padded[:, 1:-1]
        crop[...] = image
        rows.append(row(crop, shared, storage="row-crop", repeats=args.repeats))

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python-headless", "numkong", "stringzilla", "torch")
        },
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
            f"{item['shape']} {item['storage']} mask={item['mask_shape']}: "
            f"{item['baseline_ms']:.3f} → {item['current_ms']:.3f} ms "
            f"({item['speedup']:.3f}x)",
        )


if __name__ == "__main__":
    main()
