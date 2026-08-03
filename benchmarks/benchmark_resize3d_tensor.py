# ruff: noqa: INP001, S101
"""Benchmark CPU ``resize3d`` Tensor-to-Tensor routes.

The timed functions receive an already-imported CPU ``CDHW`` Tensor. They include
dispatch, view adapters, interpolation, and output allocation, but exclude Tensor
creation and Torch import:

    uv run python benchmarks/benchmark_resize3d_tensor.py --quick --threads 1
    uv run python benchmarks/benchmark_resize3d_tensor.py --full --threads 1 \
      --output benchmarks/results/resize3d-tensor.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from timing import WallTimingMs, bench_wall_ms

import albucore
from albucore.geometric import _resize3d_torch_cpu, _resize3d_torch_via_numpy, resize3d

Shape = tuple[int, int, int, int]
Size = tuple[int, int, int]
TensorFunction = Callable[[], torch.Tensor]

QUICK_SHAPES: tuple[Shape, ...] = (
    (5, 11, 13, 1),
    (5, 11, 13, 5),
    (16, 128, 160, 1),
    (16, 128, 160, 5),
)
FULL_SHAPES: tuple[Shape, ...] = (
    *QUICK_SHAPES,
    (16, 128, 160, 9),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (32, 128, 160, 5),
    (32, 128, 160, 9),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
    (64, 64, 80, 9),
    (48, 240, 320, 1),
)
LAYOUTS = ("contiguous", "channel_last_strided")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One complete Tensor-to-Tensor candidate."""

    name: str
    prepare: Callable[[torch.Tensor, Size], TensorFunction]


@dataclass(frozen=True, slots=True)
class Row:
    """One timed Tensor resize cell."""

    shape: Shape
    size: Size
    dtype: str
    layout: str
    candidate: str
    timing: WallTimingMs


def _native_torch(volume: torch.Tensor, size: Size) -> torch.Tensor:
    """Call the previous direct native Tensor kernel without public routing."""
    return _resize3d_torch_cpu(volume, size, cv2.INTER_LINEAR)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("native_torch", lambda volume, size: lambda: _native_torch(volume, size)),
    Candidate(
        "tensor_numpy_bridge",
        lambda volume, size: lambda: _resize3d_torch_via_numpy(volume, size, cv2.INTER_LINEAR),
    ),
    Candidate("albucore_public", lambda volume, size: lambda: resize3d(volume, size)),
)


def _target_shape(shape: Shape, scenario: str) -> Size:
    depth, height, width, _ = shape
    if scenario == "down":
        return max(1, depth // 2), max(1, height * 3 // 4), max(1, width * 3 // 4)
    if scenario == "up":
        return depth * 2, height * 3 // 2, width * 3 // 2
    if scenario == "mixed":
        return max(1, depth // 2), height, width * 3 // 2
    if scenario == "unit":
        return 1, max(1, height * 3 // 4), width
    msg = f"Unknown scenario {scenario!r}."
    raise ValueError(msg)


def _make_tensor(shape: Shape, dtype: torch.dtype, layout: str) -> torch.Tensor:
    """Create a CPU CDHW Tensor in one of the two measured stride layouts."""
    depth, height, width, channels = shape
    source = torch.rand((depth, height, width, channels), dtype=torch.float32)
    if dtype == torch.uint8:
        source = (source * 255).to(torch.uint8)
    tensor = source.permute(3, 0, 1, 2)
    return tensor.contiguous() if layout == "contiguous" else tensor


def _validate_candidates(volume: torch.Tensor, size: Size) -> None:
    """Check the bounded Tensor contract before timing a cell."""
    native = _native_torch(volume, size)
    for candidate in CANDIDATES[1:]:
        result = candidate.prepare(volume, size)()
        assert result.shape == native.shape
        assert result.dtype == native.dtype
        if volume.dtype == torch.uint8:
            assert int(result.min()) >= 0
            assert int(result.max()) <= 255
        if candidate.name == "albucore_public":
            if volume.dtype == torch.float32:
                torch.testing.assert_close(result, native, rtol=2e-4, atol=3e-5)
            else:
                delta = (result.to(torch.int16) - native.to(torch.int16)).abs()
                assert int(delta.max()) <= 1


def _format_rows(rows: list[Row]) -> list[str]:
    """Render rows as a compact Markdown table."""
    lines = [
        "| Shape CDHW | Target DHW | Dtype | Input strides | Candidate | Median ms | MAD ms |",
        "|---|---|---|---|---|---:|---:|",
    ]
    lines.extend(
        "| "
        f"`{row.shape[3]}x{row.shape[0]}x{row.shape[1]}x{row.shape[2]}` | "
        f"`{'x'.join(map(str, row.size))}` | {row.dtype} | {row.layout} | {row.candidate} | "
        f"{row.timing.median:.3f} | {row.timing.mad:.3f} |"
        for row in rows
    )
    return lines


def _parse_shape(value: str) -> Shape:
    """Parse one ``D,H,W,C`` shape for an isolated Tensor benchmark cell."""
    try:
        shape = tuple(int(axis_size) for axis_size in value.split(","))
    except ValueError as error:
        msg = f"--shape must be four comma-separated positive integers, got {value!r}."
        raise argparse.ArgumentTypeError(msg) from error
    if len(shape) != 4 or any(axis_size <= 0 for axis_size in shape):
        msg = f"--shape must be four comma-separated positive integers, got {value!r}."
        raise argparse.ArgumentTypeError(msg)
    return shape


def parse_args() -> argparse.Namespace:
    """Read one benchmark matrix configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument("--quick", action="store_true", help="Use small plus thin-slab shapes.")
    shape_group.add_argument("--full", action="store_true", help="Use the canonical DHWC matrix.")
    parser.add_argument("--shape", action="append", type=_parse_shape, help="Benchmark one explicit DHWC shape.")
    parser.add_argument("--threads", type=int, default=1, help="Torch and OpenCV CPU thread count.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> None:
    """Measure direct Tensor, bridge, and public router performance."""
    args = parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    rows: list[Row] = []

    for shape in shapes:
        for dtype in (torch.uint8, torch.float32):
            for layout in LAYOUTS:
                volume = _make_tensor(shape, dtype, layout)
                for scenario in ("down", "up", "mixed", "unit"):
                    size = _target_shape(shape, scenario)
                    _validate_candidates(volume, size)
                    for candidate in CANDIDATES:
                        timing = bench_wall_ms(
                            candidate.prepare(volume, size),
                            repeats=args.repeats,
                            warmup=args.warmup,
                        )
                        rows.append(Row(shape, size, str(dtype).removeprefix("torch."), layout, candidate.name, timing))
                del volume
                gc.collect()

    lines = [
        "# resize3d Tensor CPU benchmark",
        "",
        f"Run date: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}. Platform: `{platform.platform()}` "
        f"(`{platform.machine()}`).",
        "",
        f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
        f"OpenCV `{cv2.__version__}`. Torch/OpenCV CPU threads: `{args.threads}`. "
        f"Repeats: `{args.repeats}`; warmup: `{args.warmup}`.",
        "",
        "Each row receives an already-imported CPU Tensor and includes dispatch, zero-copy view adapters, kernel "
        "execution, and output allocation. `albucore_public` is checked against native Tensor interpolation with "
        "float32 `rtol=2e-4`, `atol=3e-5` or uint8 delta at most one. The bridge candidate is also timed outside "
        "its selected large all-axis-upscale region, where OpenCV intermediate rounding may exceed that uint8 bound.",
        "",
        *_format_rows(rows),
        "",
    ]
    report = "\n".join(lines)
    if args.output is None:
        print(report)  # noqa: T201 - command-line benchmark output
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote {args.output}")  # noqa: T201 - command-line benchmark output


if __name__ == "__main__":
    main()
