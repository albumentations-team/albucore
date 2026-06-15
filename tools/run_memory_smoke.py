"""Run advisory tracemalloc memory smoke checks for Albucore hot paths."""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

import albucore as ac
from benchmarks.shape_grids import MEMORY_SMOKE_C9_SHAPE, MEMORY_SMOKE_RGB_SHAPE

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class MemoryCase:
    """One memory smoke case."""

    name: str
    run: Callable[[], object]


def _uint8_image(shape: tuple[int, ...]) -> np.ndarray:
    data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    return np.ascontiguousarray((data * 37 + 13) % 256, dtype=np.uint8)


def _float32_image(shape: tuple[int, ...]) -> np.ndarray:
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return np.ascontiguousarray(data / np.float32(max(data.size - 1, 1)), dtype=np.float32)


def _shared_lut() -> np.ndarray:
    return (255 - np.arange(256, dtype=np.uint16)).astype(np.uint8)


def _per_channel_lut(channels: int) -> np.ndarray:
    base = np.arange(256, dtype=np.uint16)
    lut = np.empty((256, 1, channels), dtype=np.uint8)
    for channel in range(channels):
        lut[:, 0, channel] = (base + channel * 11) % 256
    return lut


def _cases() -> tuple[MemoryCase, ...]:
    uint8_rgb = _uint8_image(MEMORY_SMOKE_RGB_SHAPE)
    uint8_c9 = _uint8_image(MEMORY_SMOKE_C9_SHAPE)
    float_rgb = _float32_image(MEMORY_SMOKE_RGB_SHAPE)
    float_c9 = _float32_image(MEMORY_SMOKE_C9_SHAPE)
    mean_rgb = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    denom_rgb = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    factor = np.full(float_rgb.shape, np.float32(1.02), dtype=np.float32)

    return (
        MemoryCase("apply_uint8_lut_shared", lambda: ac.apply_uint8_lut(uint8_rgb, _shared_lut())),
        MemoryCase("apply_uint8_lut_per_channel_c9", lambda: ac.apply_uint8_lut(uint8_c9, _per_channel_lut(9))),
        MemoryCase("normalize_float32_rgb", lambda: ac.normalize(float_rgb, mean_rgb, denom_rgb)),
        MemoryCase("normalize_per_image_float32_c9", lambda: ac.normalize_per_image(float_c9, "image")),
        MemoryCase("add_array_float32_rgb", lambda: ac.add_array(float_rgb, np.full_like(float_rgb, 0.01))),
        MemoryCase("multiply_by_array_float32_rgb", lambda: ac.multiply_by_array(float_rgb, factor)),
        MemoryCase("mean_std_float32_c9", lambda: ac.mean_std(float_c9, "per_channel")),
        MemoryCase("resize_uint8_c9", lambda: ac.resize(uint8_c9, (211, 137), interpolation=cv2.INTER_NEAREST)),
    )


def _measure(case: MemoryCase) -> dict[str, Any]:
    tracemalloc.start()
    try:
        result = case.run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if isinstance(result, tuple):
        result_summary = [getattr(item, "shape", None) for item in result]
    else:
        result_summary = getattr(result, "shape", None)

    return {
        "name": case.name,
        "peak_bytes": peak,
        "peak_mib": peak / (1024 * 1024),
        "result_shape": result_summary,
    }


def run_memory_smoke() -> dict[str, Any]:
    """Run all memory smoke cases."""
    return {"checks": [_measure(case) for case in _cases()]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = json.dumps(run_memory_smoke(), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
