# ruff: noqa: INP001
"""Compare the public float16-to-uint8 route with its pre-LUT CPU path."""

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
from albucore import convert_dtype
from albucore.torch_backend import quantize_uint8_torch

SHAPES = (
    (1, 8_191, 1),
    (1, 8_192, 1),
    (1, 12_287, 1),
    (1, 12_288, 1),
    (96, 127, 1),
    (96, 128, 1),
    (128, 160, 1),
    (128, 160, 3),
    (128, 160, 9),
    (2, 128, 160, 3),
    (240, 320, 3),
    (480, 640, 3),
)


def previous_route(
    image: np.ndarray,
    *,
    out: np.ndarray | None,
) -> np.ndarray:
    torch_result = quantize_uint8_torch(image, 255, out=out)
    if torch_result is not None:
        return torch_result

    with np.errstate(invalid="ignore"):
        working = image.astype(np.float32)
    np.multiply(working, 255, out=working)
    np.rint(working, out=working)
    result = np.empty(image.shape, dtype=np.uint8) if out is None else out
    nk.astype(working, "uint8", out=result)
    return result


def _assert_equal(image: np.ndarray, *, out: bool) -> dict[str, object]:
    source = image.copy()
    baseline_output = np.empty(image.shape, dtype=np.uint8) if out else None
    candidate_output = np.empty(image.shape, dtype=np.uint8) if out else None
    old_call = partial(previous_route, image, out=baseline_output)
    new_call = partial(convert_dtype, image, np.uint8, out=candidate_output)
    expected = old_call()
    actual = new_call()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, source)
    if out and (expected is not baseline_output or actual is not candidate_output):
        raise AssertionError("convert_dtype did not return the supplied output")

    old_timing, new_timing = bench_paired_wall_ms(old_call, new_call, repeats=31, warmup=5)
    return {
        "shape": image.shape,
        "storage": "out" if out else "allocating",
        "source_writeable": image.flags.writeable,
        "baseline_ms": old_timing.median,
        "baseline_mad_ms": old_timing.mad,
        "baseline_samples_ms": old_timing.raw,
        "current_ms": new_timing.median,
        "current_mad_ms": new_timing.mad,
        "current_samples_ms": new_timing.raw,
        "speedup": old_timing.median / new_timing.median,
        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()

    thread_settings = configure_libraries(torch, cv2, args.threads)
    if platform.system() == "Darwin" and platform.machine() == "arm64" and not nk.get_capabilities().get("neon"):
        parser.error("Native macOS ARM64 measurements require NumKong NEON.")
    rng = np.random.default_rng(8043)
    rows = []
    for shape in SHAPES:
        values = rng.uniform(-0.25, 1.25, shape).astype(np.float32)
        image = values.astype(np.float16)
        rows.extend(_assert_equal(image, out=with_output) for with_output in (False, True))

    encodings = np.arange(65_536, dtype=np.uint16).view(np.float16).reshape(256, 256, 1)
    for writeable in (True, False):
        encodings.flags.writeable = writeable
        rows.append(_assert_equal(encodings, out=False))

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "numkong": nk.__version__,
            "stringzilla": importlib.metadata.version("stringzilla"),
            "torch": torch.__version__,
        },
        "capabilities": nk.get_capabilities(),
        "source_paths": {
            module.__name__: str(Path(module.__file__).resolve()) for module in (core, cv2, np, nk, sz, torch)
        },
        "threads": thread_settings,
        "paired_order": "baseline starts first; call order alternates every sample",
        "source_sha256": hashlib.sha256(Path("albucore/convert.py").read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(  # noqa: T201
            f"{row['shape']} {row['storage']} readonly={not row['source_writeable']}: "
            f"{row['baseline_ms']:.3f} → {row['current_ms']:.3f} ms "
            f"({row['speedup']:.3f}x)",
        )


if __name__ == "__main__":
    main()
