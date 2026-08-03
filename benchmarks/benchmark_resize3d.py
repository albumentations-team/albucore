# ruff: noqa: INP001
"""Benchmark CPU candidates for DHWC ``resize3d``.

The timed functions include packing, axis permutations, dtype conversion, and
output materialization. Run from the repository root:

    uv run python benchmarks/benchmark_resize3d.py --quick
    uv run python benchmarks/benchmark_resize3d.py --full --threads 1 \
      --output benchmarks/results/resize3d.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from timing import WallTimingMs, bench_wall_ms

import albucore
from albucore.geometric import resize, resize3d
from albucore.utils import get_opencv_max_channels

ArrayFunction = Callable[[], np.ndarray]

QUICK_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (5, 11, 13, 1),
    (5, 11, 13, 3),
    (5, 11, 13, 5),
    (16, 128, 160, 1),
    (16, 128, 160, 5),
)
FULL_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (16, 128, 160, 5),
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


@dataclass(frozen=True, slots=True)
class Candidate:
    """Named full-path resize candidate."""

    name: str
    prepare: Callable[[np.ndarray, tuple[int, int, int]], ArrayFunction | None]


@dataclass(frozen=True, slots=True)
class Row:
    """One candidate timing row."""

    shape: tuple[int, int, int, int]
    target: tuple[int, int, int]
    dtype: str
    candidate: str
    timing: WallTimingMs


def _resize_2d(array: np.ndarray, dsize: tuple[int, int], interpolation: int) -> np.ndarray:
    """Use Albucore's channel-safe 2D router and restore a dropped singleton axis."""
    result = resize(array, dsize, interpolation=interpolation)
    return result[..., np.newaxis] if result.ndim == 2 else result


def _resize_axis_opencv(
    volume: np.ndarray,
    axis: int,
    output_size: int,
    interpolation: int,
) -> np.ndarray:
    """Resize one DHWC spatial axis by packing independent 1D signals as 2D rows."""
    depth, height, width, channels = volume.shape
    source_size = volume.shape[axis]
    if source_size == output_size:
        return volume

    if axis == 0:
        flattened = volume.transpose(1, 2, 0, 3).reshape(height * width, depth, channels)
        resized = _resize_2d(flattened, (output_size, height * width), interpolation)
        return resized.reshape(height, width, output_size, channels).transpose(2, 0, 1, 3)
    if axis == 1:
        flattened = volume.transpose(0, 2, 1, 3).reshape(depth * width, height, channels)
        resized = _resize_2d(flattened, (output_size, depth * width), interpolation)
        return resized.reshape(depth, width, output_size, channels).transpose(0, 2, 1, 3)

    flattened = volume.reshape(depth * height, width, channels)
    resized = _resize_2d(flattened, (output_size, depth * height), interpolation)
    return resized.reshape(depth, height, output_size, channels)


def resize3d_opencv_axis_packing(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Current AlbumentationsX-style three-pass OpenCV implementation."""
    result = volume
    for axis, output_size in enumerate(size):
        result = _resize_axis_opencv(result, axis, output_size, interpolation)
    return result


def _linear_axis_numpy(volume: np.ndarray, axis: int, output_size: int) -> np.ndarray:
    """Reference half-pixel linear interpolation along one axis in float32."""
    input_size = volume.shape[axis]
    if input_size == output_size:
        return volume

    output_coordinates = np.arange(output_size, dtype=np.float32)
    source_coordinates = (output_coordinates + np.float32(0.5)) * (input_size / output_size) - np.float32(0.5)
    source_floor = np.floor(source_coordinates).astype(np.intp)
    left = np.clip(source_floor, 0, input_size - 1)
    right = np.clip(source_floor + 1, 0, input_size - 1)
    weight_shape = [1] * volume.ndim
    weight_shape[axis] = output_size
    right_weight = (source_coordinates - source_floor.astype(np.float32)).reshape(weight_shape)
    left_values = np.take(volume, left, axis=axis)
    right_values = np.take(volume, right, axis=axis)
    return left_values * (np.float32(1.0) - right_weight) + right_values * right_weight


def resize3d_numpy_reference(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Allocation-heavy true trilinear reference with final-only uint8 rounding."""
    result = volume.astype(np.float32, copy=False)
    for axis, output_size in enumerate(size):
        result = _linear_axis_numpy(result, axis, output_size)
    if volume.dtype == np.uint8:
        return np.minimum(result + np.float32(0.5), np.float32(255)).astype(np.uint8)
    return result


def resize3d_opencv_two_stage(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray | None:
    """Joint H/W OpenCV resize followed by packed depth resize when channels fit OpenCV."""
    depth, height, width, channels = volume.shape
    if depth * channels > get_opencv_max_channels():
        return None

    packed = volume.transpose(1, 2, 0, 3).reshape(height, width, depth * channels)
    resized_hw = _resize_2d(packed, (size[2], size[1]), interpolation)
    restored = resized_hw.reshape(size[1], size[2], depth, channels).transpose(2, 0, 1, 3)
    return _resize_axis_opencv(restored, 0, size[0], interpolation)


def resize3d_opencv_per_slice(
    volume: np.ndarray,
    size: tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize H/W per slice before the packed depth pass; avoids depth-times-channel packing."""
    depth, _, _, channels = volume.shape
    resized_slices = np.empty((depth, size[1], size[2], channels), dtype=volume.dtype)
    for index in range(depth):
        resized_slices[index] = _resize_2d(volume[index], (size[2], size[1]), interpolation)
    return _resize_axis_opencv(resized_slices, 0, size[0], interpolation)


def resize3d_torch_round_trip(volume: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    """Complete NumPy DHWC → Torch NCDHW → NumPy DHWC route."""
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).unsqueeze(0)
    with torch.inference_mode():
        if volume.dtype == np.uint8:
            result = torch_f.interpolate(tensor.to(torch.float32), size=size, mode="trilinear", align_corners=False)
            result = torch.minimum(result + 0.5, result.new_tensor(255)).to(torch.uint8)
        else:
            result = torch_f.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    return result.squeeze(0).permute(1, 2, 3, 0).numpy()


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("numpy_three_pass", lambda volume, size: lambda: resize3d_numpy_reference(volume, size)),
    Candidate("opencv_axis_packing", lambda volume, size: lambda: resize3d_opencv_axis_packing(volume, size)),
    Candidate(
        "opencv_two_stage",
        lambda volume, size: (
            None
            if resize3d_opencv_two_stage(volume, size) is None
            else lambda: resize3d_opencv_two_stage(volume, size)  # type: ignore[return-value]
        ),
    ),
    Candidate("opencv_per_slice", lambda volume, size: lambda: resize3d_opencv_per_slice(volume, size)),
    Candidate("torch_round_trip", lambda volume, size: lambda: resize3d_torch_round_trip(volume, size)),
    Candidate("albucore_public", lambda volume, size: lambda: resize3d(volume, size)),
)


def _target_shape(shape: tuple[int, int, int, int], scenario: str) -> tuple[int, int, int]:
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


def _make_volume(rng: np.random.Generator, shape: tuple[int, int, int, int], dtype: np.dtype[Any]) -> np.ndarray:
    if dtype == np.dtype(np.uint8):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _validate_candidates(volume: np.ndarray, size: tuple[int, int, int]) -> None:
    reference = resize3d_numpy_reference(volume, size)
    for candidate in CANDIDATES[1:]:
        fn = candidate.prepare(volume, size)
        if fn is None:
            continue
        result = fn()
        assert result.shape == (*size, volume.shape[-1])  # noqa: S101 - benchmark invariant
        assert result.dtype == volume.dtype  # noqa: S101 - benchmark invariant
        if volume.dtype == np.float32:
            np.testing.assert_allclose(result, reference, rtol=2e-4, atol=3e-5)
        else:
            assert result.min() >= 0  # noqa: S101 - benchmark invariant
            assert result.max() <= 255  # noqa: S101 - benchmark invariant


def _format_rows(rows: list[Row]) -> list[str]:
    lines = ["| Shape | Target | Dtype | Candidate | Median ms | MAD ms |", "|---|---|---|---|---:|---:|"]
    lines.extend(
        "| "
        f"`{'x'.join(map(str, row.shape))}` | `{'x'.join(map(str, row.target))}` | {row.dtype} | "
        f"{row.candidate} | {row.timing.median:.3f} | {row.timing.mad:.3f} |"
        for row in rows
    )
    return lines


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    """Parse one comma-separated DHWC shape for an isolated memory or routing benchmark cell."""
    try:
        shape = tuple(int(axis_size) for axis_size in value.split(","))
    except ValueError as error:
        msg = f"--shape must be four comma-separated positive integers, got {value!r}."
        raise argparse.ArgumentTypeError(msg) from error
    if len(shape) != 4 or any(axis_size <= 0 for axis_size in shape):
        msg = f"--shape must be four comma-separated positive integers, got {value!r}."
        raise argparse.ArgumentTypeError(msg)
    return shape  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument("--quick", action="store_true", help="Use small plus thin-slab shapes.")
    shape_group.add_argument("--full", action="store_true", help="Use the canonical DHWC matrix.")
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Benchmark one explicit DHWC shape; repeat the option for an isolated subset.",
    )
    parser.add_argument("--threads", type=int, default=1, help="Torch and OpenCV CPU thread count.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    rng = np.random.default_rng(137)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    scenarios = ("down", "up", "mixed", "unit")
    rows: list[Row] = []

    for shape in shapes:
        for dtype in (np.dtype(np.uint8), np.dtype(np.float32)):
            volume = _make_volume(rng, shape, dtype)
            for scenario in scenarios:
                size = _target_shape(shape, scenario)
                _validate_candidates(volume, size)
                for candidate in CANDIDATES:
                    fn = candidate.prepare(volume, size)
                    if fn is None:
                        continue
                    timing = bench_wall_ms(fn, repeats=args.repeats, warmup=args.warmup)
                    rows.append(Row(shape, size, dtype.name, candidate.name, timing))
                del fn
            del volume
            gc.collect()

    lines = [
        "# resize3d CPU benchmark",
        "",
        f"Run date: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}. Platform: `{platform.platform()}` "
        f"(`{platform.machine()}`).",
        "",
        f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
        f"OpenCV `{cv2.__version__}`. Torch/OpenCV CPU threads: `{args.threads}`. "
        f"Repeats: `{args.repeats}`; warmup: `{args.warmup}`.",
        "",
        "Each row times the full CPU route, including NumPy/Torch wrapping, axis packing, permutations, dtype "
        "conversion, and output materialization. `opencv_two_stage` is omitted when `D*C` exceeds OpenCV's encoded "
        "channel limit. Float32 candidates are checked against the pure NumPy three-pass half-pixel reference. "
        "Uint8 candidates preserve dtype and range; intermediate rounding differs between OpenCV and final-only "
        "trilinear routes.",
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
