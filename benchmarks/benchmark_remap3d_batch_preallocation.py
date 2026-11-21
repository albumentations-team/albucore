# ruff: noqa: E402, INP001
"""Compare per-volume ``remap3d`` plus stack with direct writes into one batch output."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from functools import partial
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import cv2
import numpy as np
import torch
from benchmark_threads import THREAD_COUNT, configure_libraries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_output import output_path, write_json_atomic

import albucore
from albucore.sampling3d import _normalize_border_value, _sample3d_torch_cpu

if TYPE_CHECKING:
    from collections.abc import Callable

SHAPES = ((4, 3, 12, 48, 64), (8, 3, 16, 64, 80), (8, 3, 32, 128, 160))
MEMORY_SHAPE = (8, 3, 32, 256, 256)


def _out_grid_sample_batch(
    volumes: torch.Tensor,
    grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | None,
) -> torch.Tensor:
    """Write each per-volume grid-sample result directly into its batch output slice."""
    batch_size, channels = volumes.shape[:2]
    output_shape = tuple(grid.shape[:3])
    border_values = _normalize_border_value(border_value, channels)
    output = volumes.new_empty((batch_size, channels, *output_shape))
    interpolation_mode = 0 if interpolation == cv2.INTER_LINEAR else 1
    padding_mode = 0 if border_mode == cv2.BORDER_CONSTANT else 1
    grid_batch = grid.unsqueeze(0)
    fill = torch.from_numpy(border_values).reshape(1, channels, 1, 1, 1)
    use_fill = border_mode == cv2.BORDER_CONSTANT and bool(np.any(border_values))
    sampled_storage = (
        None if volumes.dtype == torch.float32 else torch.empty((1, channels, *output_shape), dtype=torch.float32)
    )

    with torch.no_grad():
        for index in range(batch_size):
            source = volumes[index : index + 1]
            working = source if source.dtype == torch.float32 else source.to(torch.float32)
            if use_fill:
                working = working - fill
            sampled = output[index : index + 1] if sampled_storage is None else sampled_storage
            torch.ops.aten.grid_sampler_3d.out(
                working,
                grid_batch,
                interpolation_mode,
                padding_mode,
                False,
                out=sampled,
            )
            if use_fill:
                sampled.add_(fill)
            if volumes.dtype == torch.float32:
                continue
            if volumes.dtype == torch.uint8:
                sampled.clamp_(0.0, 255.0).add_(0.5)
            output[index : index + 1].copy_(sampled.to(dtype=volumes.dtype))
    return output


def _random_inputs(
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    output_shape: tuple[int, int, int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.uint8:
        volume = torch.randint(0, 256, shape, dtype=dtype, generator=generator)
    else:
        volume = torch.rand(shape, dtype=dtype, generator=generator)
    grid_array = np.random.default_rng(seed + 1).uniform(-1.15, 1.15, (*output_shape, 3)).astype(np.float32)
    return volume, torch.from_numpy(grid_array)


def _stacked_batch_reference(
    volumes: torch.Tensor,
    grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | None,
) -> torch.Tensor:
    border_values = _normalize_border_value(border_value, volumes.shape[1])
    return torch.stack(
        [_sample3d_torch_cpu(volume, grid, interpolation, border_mode, border_values) for volume in volumes],
    )


def _paired_medians(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    times: tuple[list[float], list[float]] = ([], [])
    calls = (baseline, candidate)
    for sample in range(warmup + repeats):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for index in order:
            started = perf_counter_ns()
            calls[index]()
            elapsed_ms = (perf_counter_ns() - started) / 1e6
            if sample >= warmup:
                times[index].append(elapsed_ms)
    return statistics.median(times[0]), statistics.median(times[1])


def _check_case(
    volumes: torch.Tensor,
    grid: torch.Tensor,
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | None,
) -> tuple[dict[str, object], Callable[[], object], Callable[[], object]]:
    original = volumes.clone()
    baseline = _stacked_batch_reference(
        volumes,
        grid,
        interpolation,
        border_mode,
        border_value,
    )
    public = albucore.remap3d(
        volumes,
        grid,
        interpolation=interpolation,
        border_mode=border_mode,
        border_value=border_value,
    )
    candidate = _out_grid_sample_batch(volumes, grid, interpolation, border_mode, border_value)
    torch.testing.assert_close(public, baseline, rtol=0, atol=0)
    torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
    torch.testing.assert_close(volumes, original, rtol=0, atol=0)
    baseline_fn = partial(
        _stacked_batch_reference,
        volumes,
        grid,
        interpolation,
        border_mode,
        border_value,
    )
    candidate_fn = partial(_out_grid_sample_batch, volumes, grid, interpolation, border_mode, border_value)
    row = {
        "shape_ncdhw": list(volumes.shape),
        "output_shape_dhw": list(grid.shape[:3]),
        "dtype": str(volumes.dtype),
        "interpolation": "linear" if interpolation == cv2.INTER_LINEAR else "nearest",
        "border_mode": "replicate" if border_mode == cv2.BORDER_REPLICATE else "constant",
        "border_value": border_value,
    }
    return row, baseline_fn, candidate_fn


def _skip_ineligible_case(
    eligible_only: bool,
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...] | None,
    channels: int,
) -> bool:
    return (
        eligible_only
        and interpolation == cv2.INTER_LINEAR
        and (border_mode != cv2.BORDER_CONSTANT or not np.any(_normalize_border_value(border_value, channels)))
    )


def _read_proc_memory_kib() -> tuple[int, int]:
    status = Path("/proc/self/status").read_text()
    current_kib = peak_kib = 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            current_kib = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            peak_kib = int(line.split()[1])
    return current_kib, peak_kib


def _memory_worker(route: str) -> None:
    volumes, grid = _random_inputs(MEMORY_SHAPE, torch.float32, MEMORY_SHAPE[2:], 9107)
    with Path("/proc/self/clear_refs").open("w") as handle:
        handle.write("5")
    current_before_kib, _ = _read_proc_memory_kib()
    if route == "stacked":
        result = _stacked_batch_reference(volumes, grid, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, None)
    else:
        result = _out_grid_sample_batch(volumes, grid, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, None)
    if result.shape != volumes.shape:
        raise AssertionError(f"{route} returned {tuple(result.shape)}, expected {tuple(volumes.shape)}")
    _, peak_kib = _read_proc_memory_kib()
    print(  # noqa: T201
        json.dumps(
            {
                "route": route,
                "shape_ncdhw": list(volumes.shape),
                "output_elements": int(np.prod(volumes.shape)),
                "baseline_rss_kib": current_before_kib,
                "peak_rss_kib": peak_kib,
                "peak_extra_kib": peak_kib - current_before_kib,
            },
        ),
    )


def _measure_rss(script: Path, repeats: int) -> list[dict[str, object]]:
    summaries = []
    for route in ("stacked", "preallocated"):
        samples = []
        for _ in range(repeats):
            process = subprocess.run(  # noqa: S603 - fixed script path and route choices
                [sys.executable, str(script), "--rss-worker", route],
                check=True,
                capture_output=True,
                text=True,
            )
            samples.append(json.loads(process.stdout.strip().splitlines()[-1]))
        summaries.append(
            {
                "route": route,
                "samples": samples,
                "peak_extra_kib_median": statistics.median(sample["peak_extra_kib"] for sample in samples),
            },
        )
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rss", action="store_true")
    parser.add_argument("--rss-only", action="store_true")
    parser.add_argument("--rss-repeats", type=int, default=3)
    parser.add_argument("--eligible-only", action="store_true")
    parser.add_argument("--rss-worker", choices=("stacked", "preallocated"))
    parser.add_argument("--output-json", type=output_path)
    args = parser.parse_args()
    if args.rss_worker is None and args.output_json is None:
        parser.error("--output-json is required for the benchmark run")
    if args.rss_repeats < 1:
        parser.error("--rss-repeats must be positive")
    if args.rss_only and not args.rss:
        parser.error("--rss-only requires --rss")
    return args


def _rss_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "torch")},
        "threads": args.threads,
        "rss": _measure_rss(Path(__file__).resolve(), args.rss_repeats),
    }


def _timing_report(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    shapes = SHAPES[1:] if args.eligible_only else SHAPES
    dtypes = (torch.float32,) if args.eligible_only else (torch.float32, torch.uint8)
    for shape in shapes:
        output_shape = (shape[2], shape[3] * 3 // 4, shape[4] * 3 // 4)
        for dtype in dtypes:
            volumes, grid = _random_inputs(shape, dtype, output_shape, sum(shape))
            fill_values = (None, 13.0, (13.0, 31.0, 91.0))
            for interpolation in (cv2.INTER_LINEAR, cv2.INTER_NEAREST):
                for border_mode, border_value in (
                    (cv2.BORDER_CONSTANT, fill_values[0]),
                    (cv2.BORDER_CONSTANT, fill_values[1]),
                    (cv2.BORDER_CONSTANT, fill_values[2]),
                    (cv2.BORDER_REPLICATE, None),
                ):
                    if _skip_ineligible_case(
                        args.eligible_only,
                        interpolation,
                        border_mode,
                        border_value,
                        shape[1],
                    ):
                        continue
                    row, baseline_fn, candidate_fn = _check_case(
                        volumes,
                        grid,
                        interpolation,
                        border_mode,
                        border_value,
                    )
                    baseline_ms, candidate_ms = _paired_medians(
                        baseline_fn,
                        candidate_fn,
                        repeats=args.repeats,
                        warmup=args.warmup,
                    )
                    row["baseline_ms"] = baseline_ms
                    row["candidate_ms"] = candidate_ms
                    row["speedup"] = baseline_ms / candidate_ms
                    rows.append(row)

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "torch")},
        "threads": args.threads,
        "benchmark": {"repeats": args.repeats, "warmup": args.warmup, "paired": True},
        "rss": _measure_rss(Path(__file__).resolve(), args.rss_repeats) if args.rss else None,
        "rows": rows,
    }


def main() -> None:
    args = _parse_args()
    configure_libraries(torch, cv2, args.threads)
    if args.rss_worker is not None:
        _memory_worker(args.rss_worker)
        return
    result = _rss_report(args) if args.rss_only else _timing_report(args)
    if args.output_json is None:
        raise AssertionError("benchmark output path was not initialized")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
