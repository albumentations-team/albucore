# ruff: noqa: INP001
"""Compare complete CPU execution paths for batches sampled by one shared 3D grid."""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import benchmark_threads
import cv2
import numpy as np
import torch

import albucore
from albucore.remap3d import _sampling_grid_to_tensor, remap3d
from albucore.sampling3d import _normalize_border_value, _sample3d_torch_cpu_batch_native

Shape = tuple[int, int, int, int]
Size = tuple[int, int, int]
Volume = np.ndarray | torch.Tensor
Grid = np.ndarray | torch.Tensor
Call = Callable[[], Volume]

QUICK_SHAPES: tuple[Shape, ...] = ((5, 11, 13, 1), (16, 32, 40, 3))
FULL_SHAPES: tuple[Shape, ...] = (
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (16, 128, 160, 5),
    (16, 128, 160, 9),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (48, 240, 320, 3),
)


@dataclass(frozen=True)
class Timing:
    """Wall-clock samples for one candidate."""

    median_ms: float
    mad_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class Result:
    """One workload and its complete candidate timings."""

    shape: Shape
    output_size: Size
    batch_size: int
    dtype: str
    container: str
    layout: str
    grid_container: str
    interpolation: str
    border: str
    candidates: tuple[tuple[str, Timing], ...]


def _parse_shape(value: str) -> Shape:
    axes = tuple(int(axis) for axis in value.split(","))
    if len(axes) != 4 or any(axis <= 0 for axis in axes):
        raise argparse.ArgumentTypeError("A shape must be four positive D,H,W,C integers.")
    return axes[0], axes[1], axes[2], axes[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--quick", action="store_true", help="Use small development shapes and N=1/4.")
    scope.add_argument("--full", action="store_true", help="Use the canonical DHWC shape matrix and N=1/4/16.")
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--batch-sizes", type=int, nargs="+", choices=(1, 4, 16))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _volume(rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if dtype == np.dtype(np.uint8):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _grid(rng: np.random.Generator, size: Size) -> np.ndarray:
    return rng.uniform(-1.15, 1.15, size=(*size, 3)).astype(np.float32)


def _output_size(shape: Shape) -> Size:
    depth, height, width, _ = shape
    return max(1, depth // 2), max(1, height * 3 // 4), max(1, width * 3 // 4)


def _folded(
    volumes: Volume,
    grid: Grid,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> Volume:
    batch_size = volumes.shape[0]
    if isinstance(volumes, np.ndarray):
        _, depth, height, width, channels = volumes.shape
        merged = np.moveaxis(volumes, 0, -2).reshape(depth, height, width, batch_size * channels)
        merged_fill = None if fill is None else np.tile(np.asarray(fill, dtype=np.float32), batch_size)
        result = remap3d(merged, grid, interpolation, border_mode, merged_fill)
        return np.moveaxis(result.reshape(*grid.shape[:3], batch_size, channels), -2, 0)

    _, channels, depth, height, width = volumes.shape
    merged = volumes.reshape(batch_size * channels, depth, height, width)
    merged_fill = None if fill is None else np.tile(np.asarray(fill, dtype=np.float32), batch_size)
    result = remap3d(merged, grid, interpolation, border_mode, merged_fill)
    return result.reshape(batch_size, channels, *grid.shape[:3])


def _single_loop(
    volumes: Volume,
    grid: Grid,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> Volume:
    outputs = [remap3d(volume, grid, interpolation, border_mode, fill) for volume in volumes]
    return np.stack(outputs) if isinstance(volumes, np.ndarray) else torch.stack(outputs)


def _tensor_numpy_bridge(
    volumes: torch.Tensor,
    grid: Grid,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> torch.Tensor:
    numpy_volumes = volumes.permute(0, 2, 3, 4, 1).numpy()
    result = albucore.remap3d(numpy_volumes, grid, interpolation, border_mode, fill)
    if not isinstance(result, np.ndarray):
        raise TypeError("The NumPy batch route must return an ndarray.")
    return torch.from_numpy(result).permute(0, 4, 1, 2, 3)


def _native_batch_sampler(
    volumes: Volume,
    grid: Grid,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> Volume:
    channels = volumes.shape[-1] if isinstance(volumes, np.ndarray) else volumes.shape[1]
    border_values = _normalize_border_value(fill, channels)
    tensor_volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3) if isinstance(volumes, np.ndarray) else volumes
    result = _sample3d_torch_cpu_batch_native(
        tensor_volumes,
        _sampling_grid_to_tensor(grid),
        interpolation,
        border_mode,
        border_values,
    )
    return result.permute(0, 2, 3, 4, 1).numpy() if isinstance(volumes, np.ndarray) else result


def _time_candidates(candidates: dict[str, Call], warmup: int, repeats: int) -> tuple[tuple[str, Timing], ...]:
    samples = {name: [] for name in candidates}
    names = tuple(candidates)
    for iteration in range(warmup + repeats):
        start = iteration % len(names)
        order = names[start:] + names[:start]
        for name in order:
            started = time.perf_counter()
            candidates[name]()
            if iteration >= warmup:
                samples[name].append((time.perf_counter() - started) * 1000.0)
    rows = []
    for name, values in samples.items():
        median = statistics.median(values)
        rows.append((name, Timing(median, statistics.median(abs(value - median) for value in values), tuple(values))))
    return tuple(rows)


def _validate(candidates: dict[str, Call], expected: Volume) -> None:
    for name, call in candidates.items():
        actual = call()
        if isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray):
                raise TypeError(f"{name} returned {type(actual).__name__} for NumPy input.")
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name} differs from the single-volume reference")
        else:
            if not isinstance(actual, torch.Tensor):
                raise TypeError(f"{name} returned {type(actual).__name__} for Tensor input.")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"{name} differs from the reference")


def _measure_case(  # noqa: PLR0913, PLR0917
    rng: np.random.Generator,
    shape: Shape,
    batch_size: int,
    dtype: np.dtype,
    container: str,
    layout: str,
    grid_container: str,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
    warmup: int,
    repeats: int,
) -> Result:
    output_size = _output_size(shape)
    volumes: Volume = _volume(rng, (batch_size, *shape), dtype)
    if container == "tensor":
        volumes = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
        if layout == "contiguous":
            volumes = volumes.contiguous()
    grid: Grid = _grid(rng, output_size)
    if grid_container == "tensor":
        grid = torch.from_numpy(grid)

    def loop() -> Volume:
        return _single_loop(volumes, grid, interpolation, border_mode, fill)

    expected = loop()
    candidates: dict[str, Call] = {
        "single_loop": loop,
        "channel_folded": lambda: _folded(volumes, grid, interpolation, border_mode, fill),
        "native_n_batch_sampler": lambda: _native_batch_sampler(
            volumes,
            grid,
            interpolation,
            border_mode,
            fill,
        ),
        "public_batch_dispatch": lambda: albucore.remap3d(volumes, grid, interpolation, border_mode, fill),
    }
    if container == "tensor":
        candidates["tensor_numpy_bridge"] = lambda: _tensor_numpy_bridge(
            volumes,
            grid,
            interpolation,
            border_mode,
            fill,
        )
    _validate(candidates, expected)
    timings = _time_candidates(candidates, warmup, repeats)
    return Result(
        shape,
        output_size,
        batch_size,
        dtype.name,
        container,
        layout,
        grid_container,
        "nearest" if interpolation == cv2.INTER_NEAREST else "trilinear",
        "replicate" if border_mode == cv2.BORDER_REPLICATE else ("constant-zero" if fill is None else "constant-fill"),
        timings,
    )


def _report(results: list[Result], args: argparse.Namespace, thread_settings: str) -> str:
    lines = [
        "# Batched remap3d CPU benchmark",
        "",
        (
            f"Date: `{dt.datetime.now(tz=dt.timezone.utc).date()}`; host: `{platform.platform()}`; "
            f"machine: `{platform.machine()}`; Python `{platform.python_version()}`."
        ),
        (
            f"Albucore `{albucore.__version__}`; NumPy `{np.__version__}`; Torch `{torch.__version__}`; "
            f"OpenCV `{cv2.__version__}`."
        ),
        (
            f"{thread_settings} Warmup: `{args.warmup}`; repeats: `{args.repeats}`. "
            "Route-specific peak RSS is measured separately by `benchmark_sampling3d_batch_memory.py`."
        ),
        "Timing includes layout conversion, sampling, fill correction, dtype restoration, and returned layout.",
        "",
        (
            "| DHWC | Output DHW | N | Dtype | Container | Layout | Grid | Interpolation | Border | Candidate | "
            "Median ms | MAD ms | Raw ms |"
        ),
        "|---|---|---:|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        shape = "x".join(map(str, result.shape))
        output = "x".join(map(str, result.output_size))
        for name, timing in result.candidates:
            raw = ", ".join(f"{sample:.3f}" for sample in timing.samples_ms)
            lines.append(
                f"| `{shape}` | `{output}` | {result.batch_size} | {result.dtype} | {result.container} | "
                f"{result.layout} | {result.grid_container} | {result.interpolation} | {result.border} | {name} | "
                f"{timing.median_ms:.3f} | {timing.mad_ms:.3f} | `{raw}` |",
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:  # noqa: C901
    args = _parse_args()
    thread_settings = benchmark_threads.configure_libraries(torch, cv2, args.threads)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    batch_sizes = tuple(args.batch_sizes) if args.batch_sizes else ((1, 4, 16) if args.full else (1, 4))
    rng = np.random.default_rng(20260929)
    results: list[Result] = []
    for shape in shapes:
        channels = shape[-1]
        fill = tuple(float(17 + 3 * channel) for channel in range(channels))
        for batch_size in batch_sizes:
            if args.full and shape[0] >= 48 and batch_size == 16:
                continue
            for dtype in (np.dtype(np.uint8), np.dtype(np.float32)):
                for interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR):
                    for border_mode, border_value in (
                        (cv2.BORDER_CONSTANT, None),
                        (cv2.BORDER_CONSTANT, fill),
                        (cv2.BORDER_REPLICATE, None),
                    ):
                        for container in ("numpy", "tensor"):
                            layouts = ("ndhwc",) if container == "numpy" else ("contiguous", "channel_last_strided")
                            for layout in layouts:
                                for grid_container in ("numpy", "tensor"):
                                    results.append(  # noqa: PERF401
                                        _measure_case(
                                            rng,
                                            shape,
                                            batch_size,
                                            dtype,
                                            container,
                                            layout,
                                            grid_container,
                                            interpolation,
                                            border_mode,
                                            border_value,
                                            args.warmup,
                                            args.repeats,
                                        ),
                                    )
    report = _report(results, args, thread_settings)
    if args.output is None:
        print(report)  # noqa: T201 - command-line benchmark output
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}")  # noqa: T201 - command-line benchmark output


if __name__ == "__main__":
    main()
