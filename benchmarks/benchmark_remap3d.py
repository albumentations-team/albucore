# ruff: noqa: INP001
"""Benchmark public ``remap3d`` CPU routes and the Tensor bridge baseline.

Run a development matrix with:

    uv run python benchmarks/benchmark_remap3d.py --quick --threads 1

Use ``--full`` for the complete NumPy and Tensor matrix. Tensor rows compare the
public direct CPU ``grid_sample`` route with the ``Tensor → NumPy → Tensor`` bridge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from timing import WallTimingMs, bench_wall_ms

import albucore

if TYPE_CHECKING:
    from collections.abc import Callable

Shape = tuple[int, int, int, int]
Grid = np.ndarray | torch.Tensor

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
    """One public-path or bridge-baseline timing measurement."""

    shape: Shape
    dtype: str
    volume: str
    grid: str
    strides: str
    interpolation: str
    border: str
    candidate: str
    timing: WallTimingMs


def _make_numpy_volume(
    rng: np.random.Generator,
    shape: Shape,
    dtype: np.dtype[np.uint8] | np.dtype[np.float32],
) -> np.ndarray:
    """Create one writable contiguous caller-owned ``DHWC`` benchmark volume."""
    if dtype == np.dtype(np.uint8):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _make_grid(rng: np.random.Generator, shape: Shape) -> np.ndarray:
    """Create one normalized grid that includes interior and outside pull coordinates."""
    return rng.uniform(-1.15, 1.15, size=(*shape[:3], 3)).astype(np.float32)


def _make_tensor_volume(volume: np.ndarray, layout: str) -> torch.Tensor:
    """Expose one shared-storage ``CDHW`` Tensor with the selected stride layout."""
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    return tensor.contiguous() if layout == "contiguous" else tensor


def _border_name(border_mode: int, border_value: float | None) -> str:
    """Name the initial supported border configurations."""
    if border_mode == cv2.BORDER_REPLICATE:
        return "replicate"
    return "constant_nonzero" if border_value is not None else "constant_zero"


def _interpolation_name(interpolation: int) -> str:
    """Map the two public interpolation constants to report labels."""
    return "nearest" if interpolation == cv2.INTER_NEAREST else "linear"


def _format_rows(rows: list[Row]) -> list[str]:
    """Render every raw timing sample in a searchable Markdown table."""
    lines = [
        (
            "| Shape DHWC | Dtype | Volume | Grid | Tensor strides | Interpolation | Border | Candidate | "
            "Median ms | MAD ms | Raw ms |"
        ),
        "|---|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    lines.extend(
        "| "
        f"`{'x'.join(map(str, row.shape))}` | {row.dtype} | {row.volume} | {row.grid} | {row.strides} | "
        f"{row.interpolation} | {row.border} | {row.candidate} | {row.timing.median:.3f} | "
        f"{row.timing.mad:.3f} | `{', '.join(f'{sample:.3f}' for sample in row.timing.raw)}` |"
        for row in rows
    )
    return lines


def _format_memory_ledger(shapes: tuple[Shape, ...]) -> list[str]:
    """State per-call allocations separately from caller-owned input and grid storage."""
    lines = [
        (
            "| Shape DHWC | Caller grid MiB | New dense grid | Output uint8 MiB | Output float32 MiB | "
            "uint8 float32 work MiB | Defensive input clone |"
        ),
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for shape in shapes:
        elements = int(np.prod(shape[:3]))
        channels = shape[-1]
        grid_mib = elements * 3 * np.dtype(np.float32).itemsize / (1024 * 1024)
        uint8_mib = elements * channels / (1024 * 1024)
        float32_mib = uint8_mib * np.dtype(np.float32).itemsize
        lines.append(
            f"| `{'x'.join(map(str, shape))}` | {grid_mib:.1f} | 0 (view) | {uint8_mib:.1f} | "
            f"{float32_mib:.1f} | {float32_mib:.1f} | 0 for supported inputs |",
        )
    return lines


def _assert_tensor_candidate_parity(
    volume: torch.Tensor,
    sampling_grid: Grid,
    interpolation: int,
    border_mode: int,
    border_value: float | None,
) -> None:
    """Verify that the benchmark-only bridge matches the public direct Tensor route."""
    bridge = _tensor_numpy_bridge(
        volume,
        sampling_grid,
        interpolation,
        border_mode,
        border_value,
    )
    public = albucore.remap3d(
        volume,
        sampling_grid,
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=border_value,
    )
    if not isinstance(public, torch.Tensor):
        msg = "The Tensor route must return a Tensor."
        raise TypeError(msg)
    torch.testing.assert_close(bridge, public, rtol=0, atol=0)


def _public_call(
    volume: np.ndarray | torch.Tensor,
    sampling_grid: Grid,
    interpolation: int,
    border_mode: int,
    border_value: float | None,
) -> Callable[[], np.ndarray | torch.Tensor]:
    """Bind one complete public call for repeated timing without recreating caller-owned inputs."""
    return lambda: albucore.remap3d(
        volume,
        sampling_grid,
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=border_value,
    )


def _tensor_numpy_bridge_call(
    volume: torch.Tensor,
    sampling_grid: Grid,
    interpolation: int,
    border_mode: int,
    border_value: float | None,
) -> Callable[[], torch.Tensor]:
    """Bind the complete Tensor-to-NumPy-to-Tensor comparison baseline."""
    return lambda: _tensor_numpy_bridge(
        volume,
        sampling_grid,
        interpolation,
        border_mode,
        border_value,
    )


def _tensor_numpy_bridge(
    volume: torch.Tensor,
    sampling_grid: Grid,
    interpolation: int,
    border_mode: int,
    border_value: float | None,
) -> torch.Tensor:
    """Run the comparison baseline through the public NumPy route."""
    numpy_volume = np.asarray(volume.permute(1, 2, 3, 0).numpy())
    result = albucore.remap3d(
        numpy_volume,
        sampling_grid,
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=border_value,
    )
    return torch.from_numpy(result).permute(3, 0, 1, 2)


def _wall_timing(samples: list[float]) -> WallTimingMs:
    """Summarize pre-collected millisecond samples like ``bench_wall_ms``."""
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return WallTimingMs(
        raw=tuple(samples),
        median=median,
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(samples) > 1 else 0.0,
        mad=float(np.median(np.abs(values - median))),
        n=len(samples),
    )


def _paired_wall_timings(
    first: Callable[[], object],
    second: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
    first_starts: bool,
) -> tuple[WallTimingMs, WallTimingMs]:
    """Time two routes in alternating order to remove fixed sequential-route bias."""
    if repeats < 1:
        msg = f"repeats must be >= 1, got {repeats}"
        raise ValueError(msg)
    if warmup < 0:
        msg = f"warmup must be >= 0, got {warmup}"
        raise ValueError(msg)

    first_samples: list[float] = []
    second_samples: list[float] = []
    for index in range(warmup + repeats):
        first_before_second = (index % 2 == 0) == first_starts
        ordered = ((first, first_samples), (second, second_samples))
        if not first_before_second:
            ordered = tuple(reversed(ordered))
        for function, samples in ordered:
            started = time.perf_counter()
            function()
            if index >= warmup:
                samples.append((time.perf_counter() - started) * 1000.0)
    return _wall_timing(first_samples), _wall_timing(second_samples)


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
    scope.add_argument("--full", action="store_true", help="Use every required NumPy and Tensor route cell.")
    parser.add_argument("--shape", action="append", type=_parse_shape, help="Benchmark an explicit DHWC shape.")
    parser.add_argument("--threads", type=int, default=1, help="Torch and OpenCV CPU thread count.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> None:
    """Measure public NumPy/Tensor routes and the complete Tensor bridge baseline."""
    args = _parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    rng = np.random.default_rng(20260825)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    rows: list[Row] = []
    tensor_cell_index = 0

    for shape in shapes:
        for dtype in (np.dtype(np.uint8), np.dtype(np.float32)):
            numpy_volume = _make_numpy_volume(rng, shape, dtype)
            grid_numpy = _make_grid(rng, shape)
            for grid_name, sampling_grid in (("numpy", grid_numpy), ("tensor", torch.from_numpy(grid_numpy))):
                for interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR):
                    for border_mode, border_value in (
                        (cv2.BORDER_CONSTANT, None),
                        (cv2.BORDER_CONSTANT, 13.0),
                        (cv2.BORDER_REPLICATE, None),
                    ):
                        numpy_timing = bench_wall_ms(
                            _public_call(numpy_volume, sampling_grid, interpolation, border_mode, border_value),
                            repeats=args.repeats,
                            warmup=args.warmup,
                        )
                        rows.append(
                            Row(
                                shape,
                                dtype.name,
                                "numpy",
                                grid_name,
                                "n/a",
                                _interpolation_name(interpolation),
                                _border_name(border_mode, border_value),
                                "public_numpy",
                                numpy_timing,
                            ),
                        )
                        for layout in ("contiguous", "channel_last_strided"):
                            tensor_volume = _make_tensor_volume(numpy_volume, layout)
                            _assert_tensor_candidate_parity(
                                tensor_volume,
                                sampling_grid,
                                interpolation,
                                border_mode,
                                border_value,
                            )
                            direct_timing, bridge_timing = _paired_wall_timings(
                                _public_call(tensor_volume, sampling_grid, interpolation, border_mode, border_value),
                                _tensor_numpy_bridge_call(
                                    tensor_volume,
                                    sampling_grid,
                                    interpolation,
                                    border_mode,
                                    border_value,
                                ),
                                repeats=args.repeats,
                                warmup=args.warmup,
                                first_starts=tensor_cell_index % 2 == 0,
                            )
                            tensor_cell_index += 1
                            row_prefix = (
                                shape,
                                dtype.name,
                                "tensor",
                                grid_name,
                                layout,
                                _interpolation_name(interpolation),
                                _border_name(border_mode, border_value),
                            )
                            rows.extend(
                                (
                                    Row(*row_prefix, "public_tensor_direct", direct_timing),
                                    Row(*row_prefix, "tensor_numpy_bridge_candidate", bridge_timing),
                                ),
                            )
                            del tensor_volume
            del numpy_volume, grid_numpy
            gc.collect()

    peak_rss_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss = peak_rss_units / (1024 * 1024) if sys.platform == "darwin" else peak_rss_units / 1024
    lines = [
        "# remap3d CPU route benchmark",
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
            "Every row times pre-created, logically identical inputs. NumPy rows include public dispatch, views, "
            "uint8 float32 work and restoration, sampling, and returned DHWC conversion. Tensor rows are paired "
            "and alternate execution order for public direct CPU `grid_sample` and the complete Tensor-to-NumPy-to-"
            "Tensor bridge baseline; both include border-value normalization and public/bridge dispatch work."
        ),
        "",
        "## Raw latency matrix",
        "",
        *_format_rows(rows),
        "",
        "## Per-call allocation ledger",
        "",
        (
            "The caller-owned normalized grid enters Torch through `torch.from_numpy` or `unsqueeze`, both views. "
            "Neither public route materializes a second `(D, H, W, 3)` grid or clones supported input storage. "
            "The only per-call full-volume allocations are the result and, for uint8, one float32 working volume."
        ),
        "",
        *_format_memory_ledger(shapes),
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
