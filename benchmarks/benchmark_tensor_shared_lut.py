# ruff: noqa: INP001
"""Compare the public Tensor shared-LUT bridge with direct NumPy output translation.

Run from the repository root with:

    uv run python benchmarks/benchmark_tensor_shared_lut.py --threads 1 \
      --output-json benchmarks/results/tensor-shared-lut.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numkong as nk
import numpy as np
import stringzilla as sz
import torch
from benchmark_threads import THREAD_COUNT, configure_libraries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_output import output_path
from timing import bench_paired_wall_ms

import albucore as core
from albucore import apply_uint8_lut

if TYPE_CHECKING:
    from collections.abc import Callable

SHAPES = (
    (1, 32, 32),
    (3, 32, 32),
    (9, 32, 32),
    (1, 128, 160),
    (3, 128, 160),
    (9, 128, 160),
    (1, 240, 320),
    (3, 240, 320),
    (9, 240, 320),
    (3, 480, 640),
    (1, 480, 640),
    (9, 480, 640),
    (5, 240, 320),
    (3, 768, 1024),
    (4, 3, 240, 320),
    (3, 16, 128, 160),
)


def previous_route(
    image: torch.Tensor,
    table: np.ndarray,
    output: torch.Tensor | None,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or table.ndim != 1:
        raise TypeError("The benchmark baseline only covers shared CPU Tensor LUT calls.")
    return previous_tensor_route(image, table, output)


def previous_tensor_route(
    image: torch.Tensor,
    table: np.ndarray,
    output: torch.Tensor | None,
) -> torch.Tensor:
    if output is None:
        result = np.array(image.numpy(), copy=True, order="C")
        apply_uint8_lut(result, table, out=result)
        return torch.from_numpy(result)
    if output is image:
        view = image.numpy()
        apply_uint8_lut(view, table, out=view)
        return image
    output.copy_(image)
    view = output.numpy()
    apply_uint8_lut(view, table, out=view)
    return output


def direct_route(
    image: torch.Tensor,
    table: np.ndarray,
    output: torch.Tensor | None,
) -> torch.Tensor:
    return apply_uint8_lut(image, table, out=output)


def numpy_index_route(
    image: torch.Tensor,
    table: np.ndarray,
    output: torch.Tensor | None,
) -> torch.Tensor:
    source = image.numpy()
    if output is None:
        return torch.from_numpy(table[source])
    np.take(table, source, out=output.numpy())
    return output


@dataclass(frozen=True, slots=True)
class Case:
    """One input, expected result, output mode, and timing configuration."""

    image: torch.Tensor
    table: np.ndarray
    source: torch.Tensor
    expected: torch.Tensor
    output: torch.Tensor | None
    repeats: int
    warmup: int


def _record_alternative(
    row: dict[str, object],
    name: str,
    case: Case,
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
) -> None:
    result = candidate()
    torch.testing.assert_close(result, case.expected, rtol=0, atol=0)
    torch.testing.assert_close(case.image, case.source, rtol=0, atol=0)
    if case.output is not None and result is not case.output:
        raise AssertionError(f"{name} LUT route did not return the supplied output")
    _, timing = bench_paired_wall_ms(baseline, candidate, repeats=case.repeats, warmup=case.warmup)
    row[f"{name}_ms"] = timing.median
    row[f"{name}_mad_ms"] = timing.mad
    row[f"{name}_samples_ms"] = timing.raw
    row[f"{name}_speedup"] = float(row["baseline_ms"]) / timing.median


def _measure_case(case: Case, *, numpy_candidate: bool) -> dict[str, object]:
    baseline_output = torch.empty_like(case.image) if case.output is not None else None
    baseline = partial(previous_route, case.image, case.table, baseline_output)
    direct = partial(direct_route, case.image, case.table, case.output)
    old_result = baseline()
    direct_result = direct()
    torch.testing.assert_close(old_result, case.expected, rtol=0, atol=0)
    torch.testing.assert_close(direct_result, case.expected, rtol=0, atol=0)
    torch.testing.assert_close(case.image, case.source, rtol=0, atol=0)
    if case.output is not None and direct_result is not case.output:
        raise AssertionError("Tensor LUT did not return the supplied output")
    baseline_timing, candidate_timing = bench_paired_wall_ms(
        baseline,
        direct,
        repeats=case.repeats,
        warmup=case.warmup,
    )
    row: dict[str, object] = {
        "shape": tuple(case.image.shape),
        "elements": case.image.numel(),
        "storage": "out" if case.output is not None else "allocating",
        "baseline_ms": baseline_timing.median,
        "baseline_mad_ms": baseline_timing.mad,
        "baseline_samples_ms": baseline_timing.raw,
        "candidate_ms": candidate_timing.median,
        "candidate_mad_ms": candidate_timing.mad,
        "candidate_samples_ms": candidate_timing.raw,
        "speedup": baseline_timing.median / candidate_timing.median,
        "output_sha256": hashlib.sha256(direct_result.numpy().tobytes()).hexdigest(),
    }
    if numpy_candidate:
        numpy = partial(numpy_index_route, case.image, case.table, case.output)
        _record_alternative(row, "numpy", case, baseline, numpy)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-elements", type=int)
    parser.add_argument("--numpy-candidate", action="store_true")
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    thread_settings = configure_libraries(torch, cv2, args.threads)
    if args.max_elements is not None and args.max_elements < 1:
        parser.error("--max-elements must be >= 1")
    capabilities = nk.get_capabilities()
    if platform.system() == "Darwin" and platform.machine() == "arm64" and not capabilities.get("neon"):
        parser.error("Native macOS ARM64 measurements require NumKong NEON.")

    rng = np.random.default_rng(42)
    table = rng.permutation(256).astype(np.uint8)
    rows = []
    shapes = tuple(shape for shape in SHAPES if args.max_elements is None or math.prod(shape) <= args.max_elements)
    for shape in shapes:
        image = torch.from_numpy(rng.integers(0, 256, shape, dtype=np.uint8))
        source = image.clone()
        expected = torch.from_numpy(table[image.numpy()])
        for with_output in (False, True):
            output = torch.empty_like(image) if with_output else None
            case = Case(image, table, source, expected, output, args.repeats, args.warmup)
            rows.append(_measure_case(case, numpy_candidate=args.numpy_candidate))

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "benchmark": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "max_elements": args.max_elements,
            "numpy_candidate": args.numpy_candidate,
        },
        "versions": {
            "numpy": importlib.metadata.version("numpy"),
            "opencv": cv2.__version__,
            "numkong": importlib.metadata.version("numkong"),
            "stringzilla": importlib.metadata.version("stringzilla"),
            "torch": torch.__version__,
        },
        "numkong_capabilities": capabilities,
        "source_paths": {
            module.__name__: str(Path(module.__file__).resolve()) for module in (core, cv2, nk, np, sz, torch)
        },
        "threads": thread_settings,
        "paired_order": "baseline starts first; call order alternates every sample",
        "table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        line = (
            f"{row['shape']} {row['storage']}: {row['baseline_ms']:.4f} → "
            f"{row['candidate_ms']:.4f} ms ({row['speedup']:.3f}x)"
        )
        if "numpy_ms" in row:
            line += f"; NumPy {row['numpy_ms']:.4f} ms ({row['numpy_speedup']:.3f}x)"
        print(line)  # noqa: T201


if __name__ == "__main__":
    main()
