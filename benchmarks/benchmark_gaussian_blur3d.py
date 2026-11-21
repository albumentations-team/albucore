# ruff: noqa: INP001, S101
"""Compare full CPU routes for a single true-3D Gaussian blur volume.

Torch is imported before timing. NumPy timings include the complete ``DHWC → CDHW → DHWC``
bridge; Tensor timings receive an existing CPU ``CDHW`` Tensor and exclude its construction.

Run::

    uv run python benchmarks/benchmark_gaussian_blur3d.py --quick --threads 1
    uv run python benchmarks/benchmark_gaussian_blur3d.py --full --threads 1 \
      --output benchmarks/results/gaussian-blur3d-cpu.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import benchmark_threads
import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from shape_grids import parse_dhwc_shape
from timing import WallTimingMs, bench_wall_ms

import albucore
from albucore.filter3d import _restore_uint8, _separable_filter3d_torch_cpu, separable_filter3d
from albucore.utils import get_opencv_max_channels

if TYPE_CHECKING:
    from collections.abc import Callable

Shape = tuple[int, int, int, int]
Kernels = tuple[np.ndarray, np.ndarray, np.ndarray]

QUICK_SHAPES: tuple[Shape, ...] = ((3, 5, 7, 1), (3, 5, 7, 5), (16, 128, 160, 1), (16, 128, 160, 5))
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
SIGMA = (0.75, 1.25, 1.75)
MEMORY_SHAPE: Shape = (16, 128, 160, 3)


@dataclass(frozen=True, slots=True)
class Row:
    """One complete candidate timing result."""

    container: str
    shape: Shape
    dtype: str
    candidate: str
    timing: WallTimingMs


def _paired_wall_ms(
    first: Callable[[], object],
    second: Callable[[], object],
    repeats: int,
    warmup: int,
) -> tuple[WallTimingMs, WallTimingMs]:
    """Alternate full public calls to reduce drift between route timings."""
    samples = ([], [])
    for index in range(warmup):
        order = (first, second) if index % 2 == 0 else (second, first)
        for fn in order:
            fn()

    for index in range(repeats):
        order = (0, 1) if index % 2 == 0 else (1, 0)
        for candidate in order:
            fn = first if candidate == 0 else second
            started = time.perf_counter()
            fn()
            samples[candidate].append((time.perf_counter() - started) * 1000.0)

    timings = []
    for values in samples:
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        timings.append(
            WallTimingMs(
                raw=tuple(values),
                median=median,
                mean=float(array.mean()),
                std=float(array.std(ddof=1)) if repeats > 1 else 0.0,
                mad=float(np.median(np.abs(array - median))),
                n=repeats,
            ),
        )
    return timings[0], timings[1]


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Create the same auto-sized float32 kernel as the public Gaussian router."""
    size = int(sigma * 3.5) * 2 + 1
    coordinates = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    kernel = np.exp(np.float32(-0.5) * (coordinates / np.float32(sigma)) ** 2)
    return cast("np.ndarray", kernel / np.sum(kernel, dtype=np.float32))


def _kernels() -> Kernels:
    """Build fixed D/H/W controls outside the timed call, as AlbumentationsX does after sampling."""
    return _gaussian_kernel(SIGMA[0]), _gaussian_kernel(SIGMA[1]), _gaussian_kernel(SIGMA[2])


def _torch_bridge(volume: np.ndarray, kernels: Kernels) -> np.ndarray:
    """Direct complete NumPy-to-Torch selected route."""
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    result = _separable_filter3d_torch_cpu(tensor, kernels)
    return result.permute(1, 2, 3, 0).numpy()


def _numpy_axis(volume: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Diagnostic NumPy reflect-101 one-axis pass."""
    radius = kernel.size // 2
    padding = [(0, 0)] * volume.ndim
    padding[axis] = radius, radius
    padded = np.pad(volume, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)
    return np.tensordot(windows, kernel, axes=((-1,), (0,))).astype(np.float32)


def _numpy_three_pass(volume: np.ndarray, kernels: Kernels) -> np.ndarray:
    """All-NumPy diagnostic route with one final uint8 restoration."""
    result = volume.astype(np.float32, copy=False)
    for axis, kernel in enumerate(kernels):
        result = _numpy_axis(result, kernel, axis)
    if volume.dtype == np.uint8:
        return np.minimum(result + np.float32(0.5), np.float32(255)).astype(np.uint8)
    return result


def _opencv_hw_torch_depth(volume: np.ndarray, kernels: Kernels) -> np.ndarray:
    """Diagnostic packed-OpenCV H/W plus Torch-depth route."""
    depth, height, width, channels = volume.shape
    if depth * channels > get_opencv_max_channels():
        msg = "Packed D*C channels exceed this OpenCV build's limit."
        raise ValueError(msg)
    working = volume.astype(np.float32, copy=False)
    packed = np.ascontiguousarray(working.transpose(1, 2, 0, 3).reshape(height, width, depth * channels))
    del working
    cv2.sepFilter2D(
        packed,
        ddepth=-1,
        kernelX=kernels[2],
        kernelY=kernels[1],
        dst=packed,
        borderType=cv2.BORDER_REFLECT_101,
    )
    filtered = packed.reshape(height, width, depth, channels).transpose(2, 0, 1, 3)
    tensor = torch.from_numpy(filtered).permute(3, 0, 1, 2)
    identity = np.ones(1, dtype=np.float32)
    result = _separable_filter3d_torch_cpu(tensor, (kernels[0], identity, identity))
    if volume.dtype == np.uint8:
        result = _restore_uint8(result)
    return result.permute(1, 2, 3, 0).numpy()


def _gaussian_blur3d_opencv_numpy(volume: np.ndarray) -> np.ndarray:
    """Time the complete public NumPy candidate, including Gaussian kernel setup."""
    return _opencv_hw_torch_depth(volume, _kernels())


def _gaussian_blur3d_opencv_tensor(volume: torch.Tensor) -> torch.Tensor:
    """Time the complete public Tensor candidate through a channel-last NumPy view."""
    array = volume.permute(1, 2, 3, 0).numpy()
    result = _opencv_hw_torch_depth(array, _kernels())
    return torch.from_numpy(result).permute(3, 0, 1, 2).contiguous()


def _reflect101_indices(size: int, radius: int) -> torch.Tensor:
    """Build the universal reflection map used only by the diagnostic candidate."""
    if size == 1:
        return torch.zeros(size + 2 * radius, dtype=torch.long)
    coordinates = torch.arange(-radius, size + radius, dtype=torch.long)
    period = 2 * size - 2
    folded = torch.remainder(coordinates, period)
    return torch.where(folded < size, folded, period - folded)


def _torch_index_axis(volume: torch.Tensor, kernel: np.ndarray, axis: int) -> torch.Tensor:
    """Always use index padding; it is correct but only selected for unsupported reflect pads."""
    radius = kernel.size // 2
    padded = volume.index_select(axis, _reflect101_indices(volume.shape[axis], radius))
    if axis == 2:
        shape = (1, 1, kernel.size, 1, 1)
    elif axis == 3:
        shape = (1, 1, 1, kernel.size, 1)
    else:
        shape = (1, 1, 1, 1, kernel.size)
    weights = torch.from_numpy(kernel).reshape(shape).expand(volume.shape[1], -1, -1, -1, -1)
    return torch_f.conv3d(padded, weights, groups=volume.shape[1])


def _torch_index_three_pass(volume: torch.Tensor, kernels: Kernels) -> torch.Tensor:
    """Universal-index diagnostic Tensor candidate."""
    result = (volume if volume.dtype == torch.float32 else volume.to(torch.float32)).unsqueeze(0)
    with torch.inference_mode():
        for axis, kernel in zip((2, 3, 4), kernels, strict=True):
            result = _torch_index_axis(result, kernel, axis)
        if volume.dtype == torch.uint8:
            result = _restore_uint8(result)
    return result.squeeze(0)


def _assert_equivalent(result: np.ndarray, expected: np.ndarray) -> None:
    """Reject a candidate whose full result changes the selected route's public contract."""
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    if result.dtype == np.uint8:
        delta = np.abs(result.astype(np.int16) - expected.astype(np.int16))
        assert int(delta.max()) <= 1
    else:
        np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)


def _format_rows(rows: list[Row]) -> list[str]:
    """Render timing results as a compact Markdown table."""
    lines = [
        "| Container | Shape | Dtype | Candidate | Median ms | MAD ms |",
        "|---|---|---|---|---:|---:|",
    ]
    lines.extend(
        "| "
        f"{row.container} | `{'x'.join(map(str, row.shape))}` | {row.dtype} | {row.candidate} | "
        f"{row.timing.median:.3f} | {row.timing.mad:.3f} |"
        for row in rows
    )
    return lines


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
    volume = np.random.default_rng(9107).random(MEMORY_SHAPE, dtype=np.float32)
    with Path("/proc/self/clear_refs").open("w") as handle:
        handle.write("5")
    baseline_kib, _ = _read_proc_memory_kib()
    filter3d_mod = importlib.import_module("albucore.filter3d")
    filter3d_mod._LINUX_X86_64 = route == "opencv"  # noqa: SLF001 - select the measured backend in this worker
    result = albucore.gaussian_blur3d(volume, SIGMA)
    if result.shape != volume.shape or result.dtype != volume.dtype:
        raise AssertionError(f"{route} returned {result.shape}/{result.dtype}, expected {volume.shape}/{volume.dtype}")
    _, peak_kib = _read_proc_memory_kib()
    print(  # noqa: T201
        json.dumps(
            {
                "route": route,
                "shape": list(volume.shape),
                "baseline_rss_kib": baseline_kib,
                "peak_rss_kib": peak_kib,
                "peak_extra_kib": peak_kib - baseline_kib,
            },
        ),
    )


def _measure_rss(repeats: int) -> list[dict[str, object]]:
    summaries = []
    for route in ("torch", "opencv"):
        samples = []
        for _ in range(repeats):
            process = subprocess.run(  # noqa: S603 - fixed script path and route choices
                [sys.executable, str(Path(__file__).resolve()), "--rss-worker", route],
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
    """Read timing controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Use small and thin-slab shapes.")
    group.add_argument("--full", action="store_true", help="Use the full canonical DHWC matrix.")
    parser.add_argument("--shape", action="append", type=parse_dhwc_shape, help="Benchmark an explicit DHWC shape.")
    parser.add_argument("--threads", type=int, default=1, help="CPU thread count; default baseline is 1.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--tensor-contiguous", action="store_true", help="Benchmark contiguous CDHW Tensors.")
    parser.add_argument("--rss", action="store_true", help="Measure fresh-process peak RSS for a float32 C3 volume.")
    parser.add_argument("--rss-repeats", type=int, default=3)
    parser.add_argument("--rss-worker", choices=("torch", "opencv"))
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()
    if args.rss_repeats < 1:
        parser.error("--rss-repeats must be positive")
    return args


def _time_numpy_routes(
    volume: np.ndarray,
    shape: Shape,
    kernels: Kernels,
    repeats: int,
    warmup: int,
) -> list[Row]:
    """Validate and time complete NumPy DHWC candidates."""
    expected = _torch_bridge(volume, kernels)
    candidates: list[tuple[str, Callable[[], np.ndarray]]] = [
        ("torch_reflect_bridge", lambda: _torch_bridge(volume, kernels)),
        ("numpy_three_pass", lambda: _numpy_three_pass(volume, kernels)),
    ]

    def public() -> np.ndarray:
        return separable_filter3d(volume, _kernels())

    def public_opencv() -> np.ndarray:
        return _gaussian_blur3d_opencv_numpy(volume)

    paired_public: tuple[WallTimingMs, WallTimingMs] | None = None
    if volume.shape[0] * volume.shape[-1] <= get_opencv_max_channels():
        candidates.insert(2, ("opencv_hw_torch_depth", lambda: _opencv_hw_torch_depth(volume, kernels)))
        _assert_equivalent(public(), expected)
        _assert_equivalent(public_opencv(), expected)
        paired_public = _paired_wall_ms(public, public_opencv, repeats, warmup)
    else:
        candidates.append(("albucore_public", public))
    rows: list[Row] = []
    for name, candidate in candidates:
        _assert_equivalent(candidate(), expected)
        rows.append(Row("NumPy DHWC", shape, volume.dtype.name, name, bench_wall_ms(candidate, repeats, warmup)))
    if paired_public is not None:
        rows.extend(
            (
                Row("NumPy DHWC", shape, volume.dtype.name, "torch_baseline_paired", paired_public[0]),
                Row("NumPy DHWC", shape, volume.dtype.name, "opencv_hw_torch_depth_public_paired", paired_public[1]),
            ),
        )
    return rows


def _time_tensor_routes(
    tensor: torch.Tensor,
    shape: Shape,
    kernels: Kernels,
    repeats: int,
    warmup: int,
) -> list[Row]:
    """Validate and time complete Tensor CDHW candidates with Torch already imported."""
    expected = _separable_filter3d_torch_cpu(tensor, kernels)
    candidates: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("torch_reflect", lambda: _separable_filter3d_torch_cpu(tensor, kernels)),
        ("torch_index_padding", lambda: _torch_index_three_pass(tensor, kernels)),
    ]

    def public() -> torch.Tensor:
        return separable_filter3d(tensor, _kernels())

    def public_opencv() -> torch.Tensor:
        return _gaussian_blur3d_opencv_tensor(tensor)

    paired_public: tuple[WallTimingMs, WallTimingMs] | None = None
    if tensor.shape[1] * tensor.shape[0] <= get_opencv_max_channels():
        _assert_equivalent(public().permute(1, 2, 3, 0).numpy(), expected.permute(1, 2, 3, 0).numpy())
        _assert_equivalent(public_opencv().permute(1, 2, 3, 0).numpy(), expected.permute(1, 2, 3, 0).numpy())
        paired_public = _paired_wall_ms(public, public_opencv, repeats, warmup)
    else:
        candidates.append(("albucore_public", public))
    container = "Torch CDHW contiguous" if tensor.is_contiguous() else "Torch CDHW channel-last view"
    rows: list[Row] = []
    expected_numpy = expected.permute(1, 2, 3, 0).numpy()
    for name, candidate in candidates:
        _assert_equivalent(candidate().permute(1, 2, 3, 0).numpy(), expected_numpy)
        timing = bench_wall_ms(candidate, repeats, warmup)
        rows.append(Row(container, shape, str(tensor.dtype).removeprefix("torch."), name, timing))
    if paired_public is not None:
        rows.extend(
            (
                Row(
                    container,
                    shape,
                    str(tensor.dtype).removeprefix("torch."),
                    "torch_baseline_paired",
                    paired_public[0],
                ),
                Row(
                    container,
                    shape,
                    str(tensor.dtype).removeprefix("torch."),
                    "opencv_hw_torch_depth_public_paired",
                    paired_public[1],
                ),
            ),
        )
    return rows


def main() -> None:
    """Run the selected shape/dtype matrix."""
    args = _parse_args()
    if args.rss_worker is not None:
        _memory_worker(args.rss_worker)
        return
    thread_settings = benchmark_threads.configure_libraries(torch, cv2, args.threads)
    shapes = tuple(args.shape) if args.shape else (FULL_SHAPES if args.full else QUICK_SHAPES)
    kernels = _kernels()
    rng = np.random.default_rng(20260805)
    rows: list[Row] = []
    for shape in shapes:
        for dtype in (np.dtype(np.uint8), np.dtype(np.float32)):
            volume = (
                rng.integers(0, 256, size=shape, dtype=np.uint8)
                if dtype == np.dtype(np.uint8)
                else rng.random(shape, dtype=np.float32)
            )
            rows.extend(_time_numpy_routes(volume, shape, kernels, args.repeats, args.warmup))
            tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
            if args.tensor_contiguous:
                tensor = tensor.contiguous()
            rows.extend(_time_tensor_routes(tensor, shape, kernels, args.repeats, args.warmup))
            del volume
            gc.collect()

    rss_rows = _measure_rss(args.rss_repeats) if args.rss else None
    report = "\n".join(
        (
            "# GaussianBlur3D CPU benchmark",
            "",
            (
                f"Run date: {dt.datetime.now(tz=dt.UTC).date().isoformat()}. Platform: `{platform.platform()}` "
                f"(`{platform.machine()}`)."
            ),
            "",
            (
                f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
                f"OpenCV `{cv2.__version__}`. {thread_settings} Repeats: `{args.repeats}`; "
                f"warmup: `{args.warmup}`."
            ),
            (
                f"Fresh-process peak RSS extra for float32 `{MEMORY_SHAPE}`: "
                f"Torch `{rss_rows[0]['peak_extra_kib_median']} KiB`, "
                f"OpenCV XY/Torch depth `{rss_rows[1]['peak_extra_kib_median']} KiB`."
            )
            if rss_rows is not None
            else "",
            "",
            (
                "NumPy rows include all wrapper, permutation, conversion, kernel, and output-view costs. Tensor rows "
                "use an already-created CPU CDHW Tensor; import and Tensor creation are excluded. Diagnostic "
                "candidates must match the selected route within float32 `rtol=3e-5`, `atol=3e-5` or uint8 delta "
                "at most one."
            ),
            "",
            *_format_rows(rows),
            "",
        ),
    )
    if args.output is None:
        print(report)  # noqa: T201 - benchmark CLI output
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Wrote {args.output}")  # noqa: T201 - benchmark CLI output


if __name__ == "__main__":
    main()
