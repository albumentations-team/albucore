# ruff: noqa: INP001
"""Compare complete CPU execution paths for batched 3D affine resampling."""

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
from albucore.affine3d import (
    _inverse_matrix,
    _normalize_matrix,
    _warp_affine3d_torch_cpu_batch_native,
    warp_affine3d,
)
from albucore.sampling3d import _normalize_border_value

Shape = tuple[int, int, int, int]
Size = tuple[int, int, int]
Volume = np.ndarray | torch.Tensor
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
    batch_size: int
    dtype: str
    container: str
    layout: str
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


def _matrix() -> np.ndarray:
    return np.array(
        ((0.95, 0.1, 0.0, 0.25), (0.0, 1.05, 0.1, -0.25), (0.05, 0.0, 1.0, 0.125)),
        dtype=np.float32,
    )


def _target(shape: Shape) -> Size:
    depth, height, width, _ = shape
    return max(1, depth // 2), max(1, height * 3 // 4), max(1, width * 3 // 4)


def _folded(  # noqa: PLR0913, PLR0917
    volumes: Volume,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> Volume:
    batch_size = volumes.shape[0]
    if isinstance(volumes, np.ndarray):
        _, depth, height, width, channels = volumes.shape
        merged = np.moveaxis(volumes, 0, -2).reshape(depth, height, width, batch_size * channels)
        merged_fill = None if fill is None else np.tile(np.asarray(fill, dtype=np.float32), batch_size)
        result = warp_affine3d(merged, matrix, size, interpolation, border_mode, merged_fill)
        return np.moveaxis(result.reshape(*size, batch_size, channels), -2, 0)

    _, channels, depth, height, width = volumes.shape
    merged = volumes.reshape(batch_size * channels, depth, height, width)
    merged_fill = None if fill is None else np.tile(np.asarray(fill, dtype=np.float32), batch_size)
    result = warp_affine3d(merged, matrix, size, interpolation, border_mode, merged_fill)
    return result.reshape(batch_size, channels, *size)


def _single_loop(  # noqa: PLR0913, PLR0917
    volumes: Volume,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> Volume:
    outputs = [warp_affine3d(volume, matrix, size, interpolation, border_mode, fill) for volume in volumes]
    return np.stack(outputs) if isinstance(volumes, np.ndarray) else torch.stack(outputs)


def _tensor_numpy_bridge(  # noqa: PLR0913, PLR0917
    volumes: torch.Tensor,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> torch.Tensor:
    numpy_volumes = volumes.permute(0, 2, 3, 4, 1).numpy()
    result = albucore.warp_affine3d(numpy_volumes, matrix, size, interpolation, border_mode, fill)
    if not isinstance(result, np.ndarray):
        raise TypeError("The NumPy batch route must return an ndarray.")
    return torch.from_numpy(result).permute(0, 4, 1, 2, 3)


def _native_n_batch_sampler(  # noqa: PLR0913, PLR0917
    volumes: Volume,
    matrix: np.ndarray,
    size: Size,
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
) -> Volume:
    if isinstance(volumes, np.ndarray):
        channels = volumes.shape[-1]
        if not volumes.flags.writeable or any(stride < 0 for stride in volumes.strides):
            volumes = np.array(volumes, copy=True, order="C")
        tensor = torch.from_numpy(volumes).permute(0, 4, 1, 2, 3)
    else:
        channels = volumes.shape[1]
        tensor = volumes
    inverse_matrix = _inverse_matrix(_normalize_matrix(matrix))
    result = _warp_affine3d_torch_cpu_batch_native(
        tensor,
        inverse_matrix,
        size,
        interpolation,
        border_mode,
        _normalize_border_value(fill, channels),
    )
    return np.asarray(result.permute(0, 2, 3, 4, 1).numpy()) if isinstance(volumes, np.ndarray) else result


def _time_candidates(candidates: dict[str, Call], warmup: int, repeats: int) -> tuple[tuple[str, Timing], ...]:
    samples = {name: [] for name in candidates}
    names = tuple(candidates)
    for iteration in range(warmup + repeats):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            started = time.perf_counter()
            candidates[name]()
            if iteration >= warmup:
                samples[name].append((time.perf_counter() - started) * 1000.0)
    return tuple(
        (
            name,
            Timing(
                median_ms := statistics.median(values),
                statistics.median(abs(value - median_ms) for value in values),
                tuple(values),
            ),
        )
        for name, values in samples.items()
    )


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
    interpolation: int,
    border_mode: int,
    fill: tuple[float, ...] | None,
    warmup: int,
    repeats: int,
) -> Result:
    size = _target(shape)
    numpy_volumes = _volume(rng, (batch_size, *shape), dtype)
    volumes: Volume = numpy_volumes
    if container == "tensor":
        volumes = torch.from_numpy(numpy_volumes).permute(0, 4, 1, 2, 3)
        if layout == "contiguous":
            volumes = volumes.contiguous()
    matrix = _matrix()

    def loop() -> Volume:
        return _single_loop(volumes, matrix, size, interpolation, border_mode, fill)

    expected = loop()
    candidates: dict[str, Call] = {
        "single_loop": loop,
        "channel_folded": lambda: _folded(volumes, matrix, size, interpolation, border_mode, fill),
        "native_n_batch_sampler": lambda: _native_n_batch_sampler(
            volumes,
            matrix,
            size,
            interpolation,
            border_mode,
            fill,
        ),
        "public_batch_dispatch": lambda: albucore.warp_affine3d(
            volumes,
            matrix,
            size,
            interpolation,
            border_mode,
            fill,
        ),
    }
    if container == "tensor":
        candidates["tensor_numpy_bridge"] = lambda: _tensor_numpy_bridge(
            volumes,
            matrix,
            size,
            interpolation,
            border_mode,
            fill,
        )
    _validate(candidates, expected)
    timings = _time_candidates(candidates, warmup, repeats)
    return Result(
        shape,
        batch_size,
        dtype.name,
        container,
        layout,
        "nearest" if interpolation == cv2.INTER_NEAREST else "trilinear",
        "replicate" if border_mode == cv2.BORDER_REPLICATE else ("constant-zero" if fill is None else "constant-fill"),
        timings,
    )


def _report(results: list[Result], args: argparse.Namespace, thread_settings: str) -> str:
    lines = [
        "# Batched warp_affine3d CPU benchmark",
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
        (
            "All candidates include grid generation, layout work, sampling, fill correction, dtype restoration, "
            "and returned layout."
        ),
        "",
        "| DHWC | N | Dtype | Container | Layout | Interpolation | Border | Candidate | Median ms | MAD ms | Raw ms |",
        "|---|---:|---|---|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        shape = "x".join(map(str, result.shape))
        for name, timing in result.candidates:
            raw = ", ".join(f"{sample:.3f}" for sample in timing.samples_ms)
            lines.append(
                f"| `{shape}` | {result.batch_size} | {result.dtype} | {result.container} | {result.layout} | "
                f"{result.interpolation} | {result.border} | {name} | {timing.median_ms:.3f} | "
                f"{timing.mad_ms:.3f} | `{raw}` |",
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
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
                                results.append(  # noqa: PERF401
                                    _measure_case(
                                        rng,
                                        shape,
                                        batch_size,
                                        dtype,
                                        container,
                                        layout,
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
