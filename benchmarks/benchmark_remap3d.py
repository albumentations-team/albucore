# ruff: noqa: INP001
"""Benchmark the accepted CPU NumPy ``remap3d`` routes.

Run a fast development matrix with:

    uv run python benchmarks/benchmark_remap3d.py --quick --threads 1

Use ``--full`` for the required shape, dtype, grid-container, interpolation, and
border matrix. The rejected Tensor-volume prototype and its raw route-gate results
are recorded in ``benchmarks/results/benchmark_remap3d-route-decision.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import platform
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from timing import WallTimingMs, bench_wall_ms

import albucore

Shape = tuple[int, int, int, int]

QUICK_SHAPES: tuple[Shape, ...] = ((32, 32, 32, 1),)
FULL_SHAPES: tuple[Shape, ...] = (
    (32, 32, 32, 1),
    (32, 32, 32, 3),
    (64, 128, 128, 1),
    (64, 128, 128, 3),
    (128, 128, 128, 1),
    (128, 128, 128, 3),
)


@dataclass(frozen=True, slots=True)
class Row:
    """One complete public-route timing measurement."""

    shape: Shape
    dtype: str
    grid_container: str
    interpolation: str
    border: str
    timing: WallTimingMs


def _make_volume(
    rng: np.random.Generator,
    shape: Shape,
    dtype: np.dtype[np.uint8] | np.dtype[np.float32],
) -> np.ndarray:
    """Create one writable contiguous caller-owned DHWC benchmark volume."""
    if dtype == np.dtype(np.uint8):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _make_grid(rng: np.random.Generator, shape: Shape) -> np.ndarray:
    """Create one caller-owned normalized grid with interior and outside pull coordinates."""
    return rng.uniform(-1.15, 1.15, size=(*shape[:3], 3)).astype(np.float32)


def _border_name(border_mode: int, border_value: float | None) -> str:
    """Describe the two supported border modes and constant-fill variants in the report."""
    if border_mode == cv2.BORDER_REPLICATE:
        return "replicate"
    return "constant_nonzero" if border_value else "constant_zero"


def _format_rows(rows: list[Row]) -> list[str]:
    """Render summary statistics and every raw timing sample as Markdown."""
    header = "| Shape DHWC | Dtype | Grid | Interpolation | Border | Median ms | MAD ms | Raw ms |"
    lines = [header, "|---|---|---|---|---|---:|---:|---|"]
    lines.extend(
        "| "
        f"`{'x'.join(map(str, row.shape))}` | {row.dtype} | {row.grid_container} | {row.interpolation} | "
        f"{row.border} | {row.timing.median:.3f} | {row.timing.mad:.3f} | "
        f"`{', '.join(f'{sample:.3f}' for sample in row.timing.raw)}` |"
        for row in rows
    )
    return lines


def _parse_shape(value: str) -> Shape:
    """Parse one explicit ``D,H,W,C`` benchmark shape."""
    try:
        shape = tuple(int(axis_size) for axis_size in value.split(","))
    except ValueError as error:
        msg = f"--shape must be four comma-separated positive integers, got {value!r}."
        raise argparse.ArgumentTypeError(msg) from error
    if len(shape) != 4 or any(axis_size <= 0 for axis_size in shape):
        msg = f"--shape must be four comma-separated positive integers, got {value!r}."
        raise argparse.ArgumentTypeError(msg)
    return shape


def _parse_args() -> argparse.Namespace:
    """Read benchmark scope and timing controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--quick", action="store_true", help="Use the small 32-cube development matrix.")
    scope.add_argument("--full", action="store_true", help="Use every issue-154 NumPy route cell.")
    parser.add_argument("--shape", action="append", type=_parse_shape, help="Benchmark an explicit DHWC shape.")
    parser.add_argument("--threads", type=int, default=1, help="Torch and OpenCV CPU thread count.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> None:
    """Run every accepted NumPy route with pre-created caller-owned volumes and grids."""
    args = _parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    rng = np.random.default_rng(20260825)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    rows: list[Row] = []

    for shape in shapes:
        for dtype in (np.dtype(np.uint8), np.dtype(np.float32)):
            volume = _make_volume(rng, shape, dtype)
            grid_numpy = _make_grid(rng, shape)
            for grid_container, sampling_grid in (("numpy", grid_numpy), ("tensor", torch.from_numpy(grid_numpy))):
                for interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR):
                    for border_mode, border_value in (
                        (cv2.BORDER_CONSTANT, None),
                        (cv2.BORDER_CONSTANT, 13.0),
                        (cv2.BORDER_REPLICATE, None),
                    ):
                        timing = bench_wall_ms(
                            lambda volume=volume,
                            sampling_grid=sampling_grid,
                            interpolation=interpolation,
                            border_mode=border_mode,
                            border_value=border_value: albucore.remap3d(
                                volume,
                                sampling_grid,
                                interpolation=interpolation,
                                border_mode=border_mode,
                                border_value=border_value,
                            ),
                            repeats=args.repeats,
                            warmup=args.warmup,
                        )
                        rows.append(
                            Row(
                                shape,
                                dtype.name,
                                grid_container,
                                "nearest" if interpolation == cv2.INTER_NEAREST else "linear",
                                _border_name(border_mode, border_value),
                                timing,
                            ),
                        )
            del volume, grid_numpy
            gc.collect()

    peak_rss_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss = peak_rss_units / (1024 * 1024) if sys.platform == "darwin" else peak_rss_units / 1024
    lines = [
        "# remap3d NumPy CPU benchmark",
        "",
        (
            f"Run date: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}. "
            f"Platform: `{platform.platform()}` (`{platform.machine()}`)."
        ),
        "",
        (
            f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
            f"OpenCV `{cv2.__version__}`. Torch/OpenCV CPU threads: `{args.threads}`. "
            f"Repeats: `{args.repeats}`; warmup: `{args.warmup}`. Process peak RSS: `{peak_rss:.1f} MiB`."
        ),
        "",
        (
            "Each row includes public dispatch, NumPy/Torch views, uint8 float32 work and restoration, sampling, "
            "and returned NumPy-layout conversion. Inputs and grids are created before timing. The sampler receives "
            "one grid view; it allocates the output and, for uint8, one float32 working volume."
        ),
        "",
        *_format_rows(rows),
        "",
        (
            "The CPU Tensor-volume prototype was rejected after its route gate; see "
            "[`benchmark_remap3d-route-decision.md`](benchmark_remap3d-route-decision.md)."
        ),
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
