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
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from timing import WallTimingMs, bench_wall_ms

import albucore
from albucore.filter3d import _restore_uint8, _separable_filter3d_torch_cpu
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


@dataclass(frozen=True, slots=True)
class Row:
    """One complete candidate timing result."""

    container: str
    shape: Shape
    dtype: str
    candidate: str
    timing: WallTimingMs


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
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


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
    filtered_hw = cv2.sepFilter2D(
        packed,
        ddepth=-1,
        kernelX=kernels[2],
        kernelY=kernels[1],
        borderType=cv2.BORDER_REFLECT_101,
    )
    filtered = filtered_hw.reshape(height, width, depth, channels).transpose(2, 0, 1, 3)
    tensor = torch.from_numpy(filtered).permute(3, 0, 1, 2)
    identity = np.ones(1, dtype=np.float32)
    result = _separable_filter3d_torch_cpu(tensor, (kernels[0], identity, identity))
    if volume.dtype == np.uint8:
        result = _restore_uint8(result)
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


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


def _parse_shape(value: str) -> Shape:
    """Parse one non-batched ``D,H,W,C`` shape."""
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
    """Read timing controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Use small and thin-slab shapes.")
    group.add_argument("--full", action="store_true", help="Use the full canonical DHWC matrix.")
    parser.add_argument("--shape", action="append", type=_parse_shape, help="Benchmark an explicit DHWC shape.")
    parser.add_argument("--threads", type=int, default=1, help="Torch and OpenCV CPU thread count.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions per cell.")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmups per cell.")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


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
        ("albucore_public", lambda: albucore.gaussian_blur3d(volume, SIGMA)),
    ]
    if volume.shape[0] * volume.shape[-1] <= get_opencv_max_channels():
        candidates.insert(2, ("opencv_hw_torch_depth", lambda: _opencv_hw_torch_depth(volume, kernels)))
    rows: list[Row] = []
    for name, candidate in candidates:
        _assert_equivalent(candidate(), expected)
        rows.append(Row("NumPy DHWC", shape, volume.dtype.name, name, bench_wall_ms(candidate, repeats, warmup)))
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
    candidates: tuple[tuple[str, Callable[[], torch.Tensor]], ...] = (
        ("torch_reflect", lambda: _separable_filter3d_torch_cpu(tensor, kernels)),
        ("torch_index_padding", lambda: _torch_index_three_pass(tensor, kernels)),
        ("albucore_public", lambda: albucore.gaussian_blur3d(tensor, SIGMA)),
    )
    rows: list[Row] = []
    expected_numpy = expected.permute(1, 2, 3, 0).numpy()
    for name, candidate in candidates:
        _assert_equivalent(candidate().permute(1, 2, 3, 0).numpy(), expected_numpy)
        timing = bench_wall_ms(candidate, repeats, warmup)
        rows.append(Row("Torch CDHW", shape, str(tensor.dtype).removeprefix("torch."), name, timing))
    return rows


def main() -> None:
    """Run the selected shape/dtype matrix."""
    args = _parse_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
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
            rows.extend(_time_tensor_routes(tensor, shape, kernels, args.repeats, args.warmup))
            del volume
            gc.collect()

    report = "\n".join(
        (
            "# GaussianBlur3D CPU benchmark",
            "",
            f"Run date: {dt.datetime.now(tz=dt.timezone.utc).date().isoformat()}. Platform: `{platform.platform()}` "
            f"(`{platform.machine()}`).",
            "",
            f"Versions: Albucore `{albucore.__version__}`, Torch `{torch.__version__}`, NumPy `{np.__version__}`, "
            f"OpenCV `{cv2.__version__}`. Threads: `{args.threads}`; repeats: `{args.repeats}`; "
            f"warmup: `{args.warmup}`.",
            "",
            "NumPy rows include all wrapper, permutation, conversion, kernel, and output-view costs. Tensor rows use "
            "an already-created CPU CDHW Tensor; import and Tensor creation are excluded. Diagnostic candidates must "
            "match the selected route within float32 `rtol=3e-5`, `atol=3e-5` or uint8 delta at most one.",
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
