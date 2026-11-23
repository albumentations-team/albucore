"""Time public uint8 calls whose routes changed in the 2026-11 uint8 audit.

The script uses only public Albucore calls, so the same file can time two source trees. Each row records the
median, the median absolute deviation and a SHA-256 of the output bytes; equal hashes between trees show that the
routes produce identical results. Run from the repository root:

    uv run python benchmarks/benchmark_uint8_public_routes.py --threads 1 \
      --output-json benchmarks/results/uint8-public-routes.json

Compare two JSON files with ``--compare BASELINE.json CANDIDATE.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numkong as nk
import numpy as np
import torch
from benchmark_threads import THREAD_COUNT, configure_libraries
from timing import bench_wall_ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_output import output_path

import albucore as alb

SHAPES = ((128, 160), (240, 320), (480, 640), (768, 1024))
Case = tuple[str, Callable[[], object]]


def _digest(value: object) -> str:
    if isinstance(value, tuple):
        return "+".join(_digest(item) for item in value)
    array = value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _numpy_cases(rng: np.random.Generator) -> Iterator[Case]:
    def u8(*shape: int) -> np.ndarray:
        return rng.integers(0, 256, shape, dtype=np.uint8)

    for height, width in SHAPES:
        size = f"{height}x{width}"
        for channels in (1, 3, 9):
            image = u8(height, width, channels)
            yield f"add_scalar_negative {size} C{channels}", lambda image=image: alb.add(image, -20.0)
            yield f"add_scalar_positive {size} C{channels}", lambda image=image: alb.add(image, 20.0)
            yield f"exp {size} C{channels}", lambda image=image: alb.exp(image)
            uint8_field = rng.integers(0, 4, image.shape, dtype=np.uint8)
            float_field = rng.uniform(0.5, 1.5, image.shape).astype(np.float32)
            yield f"multiply_uint8_field {size} C{channels}", lambda i=image, f=uint8_field: alb.multiply(i, f)
            yield f"multiply_float32_field {size} C{channels}", lambda i=image, f=float_field: alb.multiply(i, f)
            yield f"box_blur_k5 {size} C{channels}", lambda image=image: alb.box_blur(image, 5)
            yield f"median_blur_k7 {size} C{channels}", lambda image=image: alb.median_blur(image, 7)
            yield (
                f"mean_std_per_channel_keepdims {size} C{channels}",
                lambda image=image: alb.mean_std(image, "per_channel", keepdims=True, eps=1e-4),
            )
        padded = u8(height + 8, width + 16, 3)
        crop = padded[4 : 4 + height, 8 : 8 + width]
        yield f"median_blur_k3_row_crop {size} C3", lambda crop=crop: alb.median_blur(crop, 3)
        wide = u8(height, width, 9)
        values = tuple(float(value) for value in range(10, 100, 10))
        yield (
            f"copy_make_border_constant {size} C9",
            lambda wide=wide: alb.copy_make_border(wide, 16, 16, 32, 32, cv2.BORDER_CONSTANT, values),
        )
        yield (
            f"copy_make_border_reflect {size} C9",
            lambda wide=wide: alb.copy_make_border(wide, 16, 16, 32, 32, cv2.BORDER_REFLECT_101, values),
        )
        left = u8(height, 64)
        right = u8(64, width // 4)
        yield f"matmul_uint8 {height}x64x{width // 4}", lambda left=left, right=right: alb.matmul(left, right)


def _tensor_cases(rng: np.random.Generator) -> Iterator[Case]:
    def u8(*shape: int) -> torch.Tensor:
        return torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8))

    for height, width in SHAPES:
        size = f"{height}x{width}"
        chw = u8(3, height, width)
        other = u8(3, height, width)
        nchw = u8(4, 3, height, width)
        yield f"tensor_multiply_scalar {size} CHW", lambda chw=chw: alb.multiply(chw, 1.5, layout="CHW")
        yield f"tensor_add_scalar {size} NCHW4", lambda nchw=nchw: alb.add(nchw, -20.0, layout="NCHW")
        yield f"tensor_power_scalar {size} CHW", lambda chw=chw: alb.power(chw, 0.8, layout="CHW")
        yield f"tensor_multiply_add_scalar {size} NCHW4", lambda nchw=nchw: alb.multiply_add(nchw, 1.2, 10.0, layout="NCHW")
        yield f"tensor_add_weighted {size} CHW", lambda chw=chw, other=other: alb.add_weighted(chw, 0.3, other, 0.7, layout="CHW")
        yield f"tensor_normalize_scalar {size} CHW", lambda chw=chw: alb.normalize(chw, 127.5, 1 / 63.75, layout="CHW")
        yield f"tensor_median_blur_k3 {size} NCHW4", lambda nchw=nchw: alb.median_blur(nchw, 3, layout="NCHW")


def _measure(args: argparse.Namespace) -> None:
    thread_settings = configure_libraries(torch, cv2, args.threads)
    rng = np.random.default_rng(1103)
    rows = []
    for name, call in (*_numpy_cases(rng), *_tensor_cases(rng)):
        with np.errstate(over="ignore"):
            digest = _digest(call())
            timing = bench_wall_ms(call, args.repeats, 3)
        rows.append({"case": name, "median_ms": timing.median, "mad_ms": timing.mad, "output": digest})
        sys.stdout.write(f"{name:52s} {timing.median * 1e3:10.1f} us {digest}\n")
        sys.stdout.flush()
    report = {
        "albucore_source": str(Path(alb.__file__).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "numkong", "stringzilla", "torch")
        }
        | {"opencv": cv2.__version__},
        "numkong_capabilities": nk.get_capabilities(),
        "threads": thread_settings,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")


def _compare(baseline_path: Path, candidate_path: Path) -> None:
    baseline = {row["case"]: row for row in json.loads(baseline_path.read_text())["rows"]}
    candidate = {row["case"]: row for row in json.loads(candidate_path.read_text())["rows"]}
    for name, row in candidate.items():
        before = baseline.get(name)
        if before is None:
            continue
        same = "same" if before["output"] == row["output"] else "DIFFERENT"
        speedup = before["median_ms"] / row["median_ms"]
        sys.stdout.write(f"{name:52s} {speedup:7.2f}x {same}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--output-json", type=output_path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        _compare(*args.compare)
        return
    if args.output_json is None:
        parser.error("--output-json is required unless --compare is used")
    _measure(args)


if __name__ == "__main__":
    main()
