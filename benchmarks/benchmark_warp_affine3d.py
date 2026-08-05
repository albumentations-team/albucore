# ruff: noqa: INP001, S101
"""Benchmark NumPy ``DHWC`` CPU routes for public ``warp_affine3d``.

The timed functions include NumPy/Torch wrapping, permutations, matrix conversion,
grid construction, sampling, dtype repair, and the returned NumPy view:

    uv run python benchmarks/benchmark_warp_affine3d.py --quick --threads 1
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
from albucore.affine3d import _inverse_matrix, _normalize_border_value, _normalize_matrix, _warp_affine3d_torch_cpu

Shape = tuple[int, int, int, int]
Size = tuple[int, int, int]
ArrayFunction = Callable[[], np.ndarray]

QUICK_SHAPES: tuple[Shape, ...] = (
    (5, 11, 13, 1),
    (5, 11, 13, 5),
    (16, 128, 160, 1),
    (16, 128, 160, 5),
)
FULL_SHAPES: tuple[Shape, ...] = (
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (16, 128, 160, 5),
    (16, 128, 160, 9),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
    (48, 240, 320, 3),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One complete NumPy-to-NumPy candidate."""

    name: str
    prepare: Callable[[np.ndarray, np.ndarray, Size, int, float], ArrayFunction]


@dataclass(frozen=True, slots=True)
class Row:
    """One public-path timing measurement."""

    shape: Shape
    size: Size
    dtype: str
    scenario: str
    candidate: str
    timing: WallTimingMs


def _native_torch_bridge(
    volume: np.ndarray,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    fill: float,
) -> np.ndarray:
    """Time the direct zero-copy NumPy ``DHWC`` to native Torch sampler bridge."""
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    result = _warp_affine3d_torch_cpu(
        tensor,
        _inverse_matrix(_normalize_matrix(matrix)),
        size,
        interpolation,
        cv2.BORDER_CONSTANT,
        _normalize_border_value(fill, volume.shape[-1]),
    )
    return result.permute(1, 2, 3, 0).numpy()


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "torch_affine_grid_bridge",
        lambda volume, matrix, size, interpolation, fill: (
            lambda: _native_torch_bridge(volume, matrix, size, interpolation, fill)
        ),
    ),
    Candidate(
        "albucore_public",
        lambda volume, matrix, size, interpolation, fill: (
            lambda: albucore.warp_affine3d(
                volume,
                matrix,
                size,
                interpolation=interpolation,
                border_value=fill,
            )
        ),
    ),
)


def _target_shape(shape: Shape, scenario: str) -> Size:
    """Create one nontrivial output shape for the requested scenario."""
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


def _matrix(scenario: str) -> np.ndarray:
    """Produce a fixed forward matrix that exercises the selected sampling region."""
    if scenario == "down":
        return np.array(((1.2, 0.0, 0.0, 0.5), (0.0, 1.2, 0.0, -0.5), (0.0, 0.0, 1.2, 0.25)), dtype=np.float32)
    if scenario == "up":
        return np.array(((0.8, 0.1, 0.0, 0.5), (0.0, 0.9, 0.1, 0.25), (0.0, 0.0, 0.85, -0.25)), dtype=np.float32)
    if scenario == "mixed":
        return np.array(((0.9, 0.15, 0.0, 1.0), (0.0, 1.1, 0.1, -0.75), (0.1, 0.0, 1.0, 0.5)), dtype=np.float32)
    if scenario == "unit":
        return np.array(((1.0, 0.1, 0.0, 0.5), (0.0, 0.95, 0.1, -0.25), (0.0, 0.0, 1.0, 0.0)), dtype=np.float32)
    msg = f"Unknown scenario {scenario!r}."
    raise ValueError(msg)


def _make_volume(
    rng: np.random.Generator,
    shape: Shape,
    dtype: np.dtype[np.uint8] | np.dtype[np.float32],
) -> np.ndarray:
    """Make one writable contiguous volume for zero-copy bridge measurement."""
    if dtype == np.dtype(np.uint8):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _validate_candidates(volume: np.ndarray, matrix: np.ndarray, size: Size, interpolation: int, fill: float) -> None:
    """Reject a candidate before timing when its public result differs from the direct baseline."""
    expected = _native_torch_bridge(volume, matrix, size, interpolation, fill)
    for candidate in CANDIDATES:
        result = candidate.prepare(volume, matrix, size, interpolation, fill)()
        assert result.shape == (*size, volume.shape[-1])
        assert result.dtype == volume.dtype
        np.testing.assert_array_equal(result, expected)


def _format_rows(rows: list[Row]) -> list[str]:
    """Render rows as a Markdown table."""
    lines = [
        "| Shape DHWC | Target DHW | Dtype | Scenario | Candidate | Median ms | MAD ms |",
        "|---|---|---|---|---|---:|---:|",
    ]
    lines.extend(
        "| "
        f"`{'x'.join(map(str, row.shape))}` | `{'x'.join(map(str, row.size))}` | {row.dtype} | {row.scenario} | "
        f"{row.candidate} | {row.timing.median:.3f} | {row.timing.mad:.3f} |"
        for row in rows
    )
    return lines


def _parse_shape(value: str) -> Shape:
    """Parse one explicit non-batched ``D,H,W,C`` benchmark shape."""
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
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument("--quick", action="store_true", help="Use small and thin-slab shapes.")
    shape_group.add_argument("--full", action="store_true", help="Use the canonical single-volume DHWC matrix.")
    parser.add_argument("--shape", action="append", type=_parse_shape, help="Benchmark an explicit DHWC shape.")
    parser.add_argument("--threads", type=int, default=1, help="Torch and OpenCV CPU thread count.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark matrix and print or write a Markdown report."""
    args = _parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    rng = np.random.default_rng(20260803)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    rows: list[Row] = []

    for shape in shapes:
        for dtype in (np.dtype(np.uint8), np.dtype(np.float32)):
            volume = _make_volume(rng, shape, dtype)
            for scenario in ("down", "up", "mixed", "unit"):
                size = _target_shape(shape, scenario)
                matrix = _matrix(scenario)
                interpolation = cv2.INTER_NEAREST if scenario == "unit" else cv2.INTER_LINEAR
                fill = 13.0 if scenario == "mixed" else 0.0
                _validate_candidates(volume, matrix, size, interpolation, fill)
                for candidate in CANDIDATES:
                    timing = bench_wall_ms(
                        candidate.prepare(volume, matrix, size, interpolation, fill),
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
                    rows.append(Row(shape, size, dtype.name, scenario, candidate.name, timing))
            del volume
            gc.collect()

    lines = [
        "# warp_affine3d NumPy CPU benchmark",
        "",
        f"Run date: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}. Platform: `{platform.platform()}` "
        f"(`{platform.machine()}`).",
        "",
        f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
        f"OpenCV `{cv2.__version__}`. Torch/OpenCV CPU threads: `{args.threads}`. "
        f"Repeats: `{args.repeats}`; warmup: `{args.warmup}`.",
        "",
        "Each row is one non-batched DHWC volume and includes public dispatch, matrix normalization/inversion, "
        "NumPy-to-Torch views, permutations, grid construction, sampling, dtype restoration, "
        "and the NumPy output view. "
        "The direct bridge and public router must be bitwise equal before timing.",
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
