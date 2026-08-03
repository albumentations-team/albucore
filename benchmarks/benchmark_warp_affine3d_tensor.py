# ruff: noqa: INP001, S101
"""Benchmark CPU ``CDHW`` Tensor routes for public ``warp_affine3d``.

The manual-grid candidate is an optimization probe. It receives the same forward
matrix and prevalidated Tensor as the production ``affine_grid`` path:

    uv run python benchmarks/benchmark_warp_affine3d_tensor.py --quick --threads 1
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
import torch.nn.functional as torch_f
from timing import WallTimingMs, bench_wall_ms

import albucore
from albucore.affine3d import (
    _inverse_matrix,
    _normalize_border_value,
    _normalize_matrix,
    _normalized_theta,
    _restore_uint8,
    _warp_affine3d_torch_cpu,
)

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
    """One complete Tensor-to-Tensor sampling candidate."""

    name: str
    prepare: Callable[[torch.Tensor, np.ndarray, Size, int, float], TensorFunction]


@dataclass(frozen=True, slots=True)
class Row:
    """One Tensor timing measurement."""

    shape: Shape
    size: Size
    dtype: str
    layout: str
    scenario: str
    candidate: str
    timing: WallTimingMs


def _native_affine_grid(
    volume: torch.Tensor,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    fill: float,
) -> torch.Tensor:
    """Call the internal production kernel without public validation and dispatch overhead."""
    return _warp_affine3d_torch_cpu(
        volume,
        _inverse_matrix(_normalize_matrix(matrix)),
        size,
        interpolation,
        cv2.BORDER_CONSTANT,
        _normalize_border_value(fill, volume.shape[0]),
    )


def _manual_grid(
    volume: torch.Tensor,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    fill_value: float,
) -> torch.Tensor:
    """Build the same normalized grid from 1D vectors and broadcasted affine combinations."""
    inverse_matrix = _inverse_matrix(_normalize_matrix(matrix))
    input_size = volume.shape[1], volume.shape[2], volume.shape[3]
    theta = torch.from_numpy(_normalized_theta(inverse_matrix, input_size, size))
    depth, height, width = size
    z = (torch.arange(depth, dtype=torch.float32) + 0.5) * (2.0 / depth) - 1.0
    y = (torch.arange(height, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    z, y, x = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack(
        (
            theta[0, 0] * x + theta[0, 1] * y + theta[0, 2] * z + theta[0, 3],
            theta[1, 0] * x + theta[1, 1] * y + theta[1, 2] * z + theta[1, 3],
            theta[2, 0] * x + theta[2, 1] * y + theta[2, 2] * z + theta[2, 3],
        ),
        dim=-1,
    ).unsqueeze(0)
    mode = "nearest" if interpolation == cv2.INTER_NEAREST else "bilinear"
    working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)

    with torch.no_grad():
        if fill_value != 0.0:
            fill = torch.full((1, volume.shape[0], 1, 1, 1), fill_value, dtype=torch.float32)
            result = (
                torch_f.grid_sample(
                    working.unsqueeze(0) - fill,
                    grid,
                    mode=mode,
                    padding_mode="zeros",
                    align_corners=False,
                )
                + fill
            )
        else:
            result = torch_f.grid_sample(
                working.unsqueeze(0),
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=False,
            )
        if volume.dtype == torch.uint8:
            result = _restore_uint8(result)
    return result.squeeze(0)


def _coverage_fill(
    volume: torch.Tensor,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    fill_value: float,
) -> torch.Tensor:
    """Evaluate the one-channel coverage-sampler alternative for constant fill."""
    inverse_matrix = _inverse_matrix(_normalize_matrix(matrix))
    input_size = volume.shape[1], volume.shape[2], volume.shape[3]
    theta = torch.from_numpy(_normalized_theta(inverse_matrix, input_size, size)).unsqueeze(0)
    mode = "nearest" if interpolation == cv2.INTER_NEAREST else "bilinear"
    working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)

    with torch.no_grad():
        grid = torch_f.affine_grid(theta, [1, volume.shape[0], *size], align_corners=False)
        result = torch_f.grid_sample(
            working.unsqueeze(0),
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )
        if fill_value != 0.0:
            coverage = torch_f.grid_sample(
                torch.ones((1, 1, *input_size), dtype=torch.float32),
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=False,
            )
            fill = torch.full((1, volume.shape[0], 1, 1, 1), fill_value, dtype=torch.float32)
            result = result + fill * (1.0 - coverage)
        if volume.dtype == torch.uint8:
            result = _restore_uint8(result)
    return result.squeeze(0)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "torch_affine_grid",
        lambda volume, matrix, size, interpolation, fill: (
            lambda: _native_affine_grid(volume, matrix, size, interpolation, fill)
        ),
    ),
    Candidate(
        "manual_grid",
        lambda volume, matrix, size, interpolation, fill: (
            lambda: _manual_grid(volume, matrix, size, interpolation, fill)
        ),
    ),
    Candidate(
        "coverage_fill",
        lambda volume, matrix, size, interpolation, fill: (
            lambda: _coverage_fill(volume, matrix, size, interpolation, fill)
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
    """Create one nontrivial output shape for a single-volume scenario."""
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


def _make_tensor(shape: Shape, dtype: torch.dtype, layout: str) -> torch.Tensor:
    """Make a contiguous or channel-last-strided CPU ``CDHW`` Tensor."""
    depth, height, width, channels = shape
    source = torch.rand((depth, height, width, channels), dtype=torch.float32)
    if dtype == torch.uint8:
        source = (source * 255.0).to(torch.uint8)
    tensor = source.permute(3, 0, 1, 2)
    return tensor.contiguous() if layout == "contiguous" else tensor


def _validate_candidates(volume: torch.Tensor, matrix: np.ndarray, size: Size, interpolation: int, fill: float) -> None:
    """Check production parity and bound diagnostic candidates before their timing is reported."""
    expected = _native_affine_grid(volume, matrix, size, interpolation, fill)
    for candidate in CANDIDATES[1:]:
        result = candidate.prepare(volume, matrix, size, interpolation, fill)()
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        if candidate.name == "albucore_public":
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        elif volume.dtype == torch.uint8:
            delta = (result.to(torch.int16) - expected.to(torch.int16)).abs()
            assert int(delta.max()) <= 2
        else:
            # Diagnostic grid construction changes float32 operation order. Keep
            # this deliberately loose enough for a benchmark-only candidate,
            # while production itself is required to match the native route
            # bit-for-bit above.
            torch.testing.assert_close(result, expected, rtol=1e-3, atol=2e-4)


def _format_rows(rows: list[Row]) -> list[str]:
    """Render rows as a compact Markdown table."""
    lines = [
        "| Shape CDHW | Target DHW | Dtype | Strides | Scenario | Candidate | Median ms | MAD ms |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    lines.extend(
        "| "
        f"`{row.shape[3]}x{row.shape[0]}x{row.shape[1]}x{row.shape[2]}` | "
        f"`{'x'.join(map(str, row.size))}` | {row.dtype} | {row.layout} | {row.scenario} | "
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
    """Run the Tensor benchmark matrix and print or write a Markdown report."""
    args = _parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    rows: list[Row] = []

    for shape in shapes:
        for dtype in (torch.uint8, torch.float32):
            for layout in ("contiguous", "channel_last_strided"):
                volume = _make_tensor(shape, dtype, layout)
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
                        rows.append(
                            Row(
                                shape,
                                size,
                                str(dtype).removeprefix("torch."),
                                layout,
                                scenario,
                                candidate.name,
                                timing,
                            ),
                        )
                del volume
                gc.collect()

    lines = [
        "# warp_affine3d Tensor CPU benchmark",
        "",
        f"Run date: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}. Platform: `{platform.platform()}` "
        f"(`{platform.machine()}`).",
        "",
        f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
        f"OpenCV `{cv2.__version__}`. Torch/OpenCV CPU threads: `{args.threads}`. "
        f"Repeats: `{args.repeats}`; warmup: `{args.warmup}`.",
        "",
        "Each row is one non-batched CPU CDHW Tensor and includes public dispatch, control-data validation, "
        "matrix inversion, grid construction, sampling, dtype restoration, and output allocation. Public output "
        "must be bitwise equal to native affine-grid. `manual_grid` and `coverage_fill` are diagnostic candidates: "
        "they need exact parity and a stable win before they can become a production route.",
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
