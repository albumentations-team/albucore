"""Golden-vector case definitions shared by generator, verifier, and tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

import albucore as ac

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
OUTPUTS_PATH = GOLDEN_DIR / "v1" / "outputs.npz"

ArrayOutputs = dict[str, np.ndarray]


@dataclass(frozen=True)
class GoldenCase:
    """One deterministic golden-vector case."""

    name: str
    compute: Callable[[], ArrayOutputs]
    rtol: float = 0.0
    atol: float = 0.0


def uint8_image(shape: tuple[int, ...]) -> np.ndarray:
    """Create deterministic uint8 data with explicit channel dimension."""
    data = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
    return np.ascontiguousarray((data * 37 + 13) % 256, dtype=np.uint8)


def float32_image(shape: tuple[int, ...]) -> np.ndarray:
    """Create deterministic float32 data in [0, 1]."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    scale = np.float32(max(data.size - 1, 1))
    return np.ascontiguousarray(data / scale, dtype=np.float32)


def _lut_shared() -> np.ndarray:
    return np.ascontiguousarray((255 - np.arange(256, dtype=np.uint16)).astype(np.uint8))


def _lut_per_channel(channels: int) -> np.ndarray:
    lut = np.empty((256, 1, channels), dtype=np.uint8)
    base = np.arange(256, dtype=np.uint16)
    for channel in range(channels):
        lut[:, 0, channel] = ((base * (channel + 1)) + 17) % 256
    return np.ascontiguousarray(lut)


def _case_add_uint8_vector() -> ArrayOutputs:
    img = uint8_image((16, 17, 3))
    value = np.array([3, 17, 31], dtype=np.uint8)
    return {"result": ac.add(img, value)}


def _case_multiply_float32_scalar() -> ArrayOutputs:
    img = float32_image((8, 9, 1))
    return {"result": ac.multiply(img, np.float32(0.75))}


def _case_multiply_add_uint8_vector() -> ArrayOutputs:
    img = uint8_image((19, 23, 9))
    factor = np.linspace(0.75, 1.25, num=9, dtype=np.float32)
    bias = np.linspace(3.0, 11.0, num=9, dtype=np.float32)
    return {"result": ac.multiply_add(img, factor, bias)}


def _case_normalize_uint8_vector() -> ArrayOutputs:
    img = uint8_image((16, 17, 3))
    mean = np.array([90.0, 110.0, 130.0], dtype=np.float32)
    denominator = np.array([1.0 / 40.0, 1.0 / 50.0, 1.0 / 60.0], dtype=np.float32)
    return {"result": ac.normalize(img, mean, denominator)}


def _case_to_from_float_roundtrip() -> ArrayOutputs:
    img = uint8_image((8, 9, 1))
    as_float = ac.to_float(img)
    return {"float": as_float, "roundtrip": ac.from_float(as_float, np.uint8)}


def _case_apply_uint8_lut_shared() -> ArrayOutputs:
    img = uint8_image((16, 17, 3))
    return {"result": ac.apply_uint8_lut(img, _lut_shared())}


def _case_apply_uint8_lut_per_channel() -> ArrayOutputs:
    img = uint8_image((19, 23, 9))
    return {"result": ac.apply_uint8_lut(img, _lut_per_channel(9))}


def _case_sz_lut_shared() -> ArrayOutputs:
    img = uint8_image((8, 9, 1))
    return {"result": ac.sz_lut(img.copy(), _lut_shared(), inplace=False)}


def _case_flips_float32() -> ArrayOutputs:
    img = float32_image((16, 17, 3))
    return {"hflip": ac.hflip(img), "vflip": ac.vflip(img)}


def _case_resize_nearest_uint8() -> ArrayOutputs:
    img = uint8_image((8, 9, 1))
    return {"result": ac.resize(img, (13, 11), interpolation=cv2.INTER_NEAREST)}


def _case_stats_uint8_per_channel() -> ArrayOutputs:
    img = uint8_image((16, 17, 3))
    mean, std = ac.mean_std(img, "per_channel")
    return {
        "mean": np.asarray(mean),
        "std": np.asarray(std),
        "sum": np.asarray(ac.reduce_sum(img, "per_channel")),
    }


def _case_batch_normalize_float32() -> ArrayOutputs:
    img = float32_image((2, 3, 8, 9, 1))
    return {"result": ac.normalize_per_image(img, "min_max")}


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase("add_uint8_vector_hwc", _case_add_uint8_vector),
    GoldenCase("multiply_float32_scalar_hwc", _case_multiply_float32_scalar),
    GoldenCase("multiply_add_uint8_vector_hwc9", _case_multiply_add_uint8_vector),
    GoldenCase("normalize_uint8_vector_hwc", _case_normalize_uint8_vector, rtol=1e-6, atol=1e-6),
    GoldenCase("to_from_float_uint8_hwc1", _case_to_from_float_roundtrip, rtol=1e-6, atol=1e-6),
    GoldenCase("apply_uint8_lut_shared_hwc", _case_apply_uint8_lut_shared),
    GoldenCase("apply_uint8_lut_per_channel_hwc9", _case_apply_uint8_lut_per_channel),
    GoldenCase("sz_lut_shared_hwc1", _case_sz_lut_shared),
    GoldenCase("flips_float32_hwc", _case_flips_float32),
    GoldenCase("resize_nearest_uint8_hwc1", _case_resize_nearest_uint8),
    GoldenCase("stats_uint8_per_channel_hwc", _case_stats_uint8_per_channel, rtol=1e-12, atol=1e-12),
    GoldenCase("batch_normalize_float32_ndhwc", _case_batch_normalize_float32, rtol=1e-6, atol=1e-6),
)


def output_key(case_name: str, output_name: str) -> str:
    """Return stable NPZ key for one case output."""
    return f"{case_name}__{output_name}"


def array_metadata(array: np.ndarray) -> dict[str, Any]:
    """Return deterministic metadata for one array."""
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.view(np.uint8)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "sum": float(np.sum(array, dtype=np.float64)) if array.size else 0.0,
        "mean": float(np.mean(array, dtype=np.float64)) if array.size else None,
        "c_contiguous": bool(array.flags.c_contiguous),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def compute_all_cases() -> dict[str, ArrayOutputs]:
    """Compute all configured golden cases."""
    return {case.name: case.compute() for case in GOLDEN_CASES}
