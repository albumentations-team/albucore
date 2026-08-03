#!/usr/bin/env python3
"""Measure the cost of Torch's mandatory int64 indices for a byte lookup table.

The experiment isolates shared and per-channel ``uint8[256]`` LUTs. Input
tensors already live on CPU before timing, so NumPy-to-Torch conversion and
compilation time are not charged to the eager kernel. With ``--compile``,
``torch.compile`` is compiled and warmed for five representative static shapes
before timing; it models a long-lived, shape-stable Torch pipeline, not an
Albucore call that compiles on demand.

Run from the repository root::

    uv run python benchmarks/benchmark_torch_uint8_lut.py --threads 1
    uv run python benchmarks/benchmark_torch_uint8_lut.py --threads 12 --volumes --compile

The NumPy and Albucore rows provide byte-indexed CPU reference paths. Torch
interprets uint8 tensors as legacy boolean masks when indexing, so eager lookup
must first construct a signed int32 or int64 index tensor. The reported
temporary sizes are included in both the timing and eager allocation path.
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from shape_grids import ROUTER_HWC_FULL_HW, SCALE_LUT_SHAPES
from timing import WallTimingMs, bench_wall_ms

from albucore import apply_uint8_lut

if TYPE_CHECKING:
    from collections.abc import Callable


# Compiling every rank/channel shape in one process exhausts the compiler cache on
# a development laptop. These cover the small-to-large RGB image, volume, and
# batch-of-volume paths that a fixed-shape training pipeline can precompile.
COMPILED_STATIC_SHAPES = frozenset(
    {
        (240, 320, 3),
        (480, 640, 3),
        (768, 1024, 3),
        (32, 128, 160, 3),
        (2, 32, 128, 160, 3),
    },
)


@dataclass(frozen=True, slots=True)
class Row:
    """One exact LUT timing for an already-resident CPU input."""

    table_kind: str
    shape: tuple[int, ...]
    layout: str
    elements: int
    int32_temp_mib: float
    int64_temp_mib: float
    albucore_ms: float
    numpy_ms: float
    torch_int32_ms: float
    torch_eager_ms: float
    torch_compile_ms: float | None


def _torch_eager_int32(image: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Fastest eager advanced-indexing expression available for byte image values."""
    return table[image.to(torch.int32)]


def _torch_eager(image: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """LongTensor indexing required by ``take`` and ``gather`` APIs."""
    return table[image.to(torch.int64)]


def _torch_eager_per_channel_int32(image: torch.Tensor, table: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """Per-channel lookup with int32 image indices and a small static channel index."""
    return table[channels, image.to(torch.int32)]


def _torch_eager_per_channel(image: torch.Tensor, table: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """Per-channel lookup with int64 indices, required by gather-like APIs."""
    return table[channels, image.to(torch.int64)]


def _numpy_index(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Byte-indexed lower-bound reference with the same table and output dtype."""
    return table[image]


def _numpy_index_per_channel(image: np.ndarray, table: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """Byte-indexed NumPy reference for distinct channel tables."""
    return table[channels, image]


def _layout(shape: tuple[int, ...]) -> str:
    return {3: "HWC", 4: "DHWC", 5: "NDHWC"}[len(shape)]


def _compile_for_shape(
    kernel: Callable[..., torch.Tensor],
    *args: torch.Tensor,
) -> tuple[Callable[[], torch.Tensor], float]:
    """Compile and warm a LUT function; report but do not time the build.

    The reported build may reuse Torch's persistent compiler cache. Use a fresh
    ``TORCHINDUCTOR_CACHE_DIR`` when a cold-start number is required.
    """
    # Dynamo otherwise reaches its process-wide recompile limit after several
    # different image shapes and silently resumes eager execution.
    torch.compiler.reset()
    start = time.perf_counter()
    compiled = torch.compile(kernel, dynamic=False)
    actual = compiled(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    expected = kernel(*args)
    if not torch.equal(actual, expected):
        raise AssertionError("torch.compile changed uint8 LUT values")
    return lambda: compiled(*args), elapsed_ms


def _timing(fn: Callable[[], object], repeats: int, warmup: int) -> WallTimingMs:
    return bench_wall_ms(fn, repeats=repeats, warmup=warmup)


def _row(
    image: np.ndarray,
    table: np.ndarray,
    *,
    repeats: int,
    warmup: int,
    include_compile: bool,
    per_channel: bool,
) -> tuple[Row, float | None]:
    image_t = torch.from_numpy(image)
    table_t = torch.from_numpy(table)
    if per_channel:
        c = image.shape[-1]
        channels_np = np.arange(c, dtype=np.int64).reshape((1,) * (image.ndim - 1) + (c,))
        channels_t = torch.from_numpy(channels_np)
        expected = _numpy_index_per_channel(image, table, channels_np)

        def eager_int32() -> torch.Tensor:
            return _torch_eager_per_channel_int32(image_t, table_t, channels_t)

        def eager() -> torch.Tensor:
            return _torch_eager_per_channel(image_t, table_t, channels_t)

        def numpy() -> np.ndarray:
            return _numpy_index_per_channel(image, table, channels_np)

        albucore_table = table.T[:, None, :]
        compiled_kernel = _torch_eager_per_channel_int32
        compiled_args = (image_t, table_t, channels_t)
        table_kind = "per-channel"
    else:
        expected = _numpy_index(image, table)

        def eager_int32() -> torch.Tensor:
            return _torch_eager_int32(image_t, table_t)

        def eager() -> torch.Tensor:
            return _torch_eager(image_t, table_t)

        def numpy() -> np.ndarray:
            return _numpy_index(image, table)

        albucore_table = table
        compiled_kernel = _torch_eager_int32
        compiled_args = (image_t, table_t)
        table_kind = "shared"

    eager_result = eager().numpy()
    np.testing.assert_array_equal(eager_result, expected)
    np.testing.assert_array_equal(eager_int32().numpy(), expected)
    np.testing.assert_array_equal(apply_uint8_lut(image, albucore_table), expected)

    compiled_fn: Callable[[], torch.Tensor] | None = None
    compile_ms: float | None = None
    if include_compile:
        compiled_fn, compile_ms = _compile_for_shape(compiled_kernel, *compiled_args)

    albucore_t = _timing(lambda: apply_uint8_lut(image, albucore_table), repeats, warmup)
    numpy_t = _timing(numpy, repeats, warmup)
    int32_t = _timing(eager_int32, repeats, warmup)
    eager_t = _timing(eager, repeats, warmup)
    compiled_t = None if compiled_fn is None else _timing(compiled_fn, repeats, warmup)
    elements = image.size
    return (
        Row(
            table_kind=table_kind,
            shape=image.shape,
            layout=_layout(image.shape),
            elements=elements,
            int32_temp_mib=elements * np.dtype(np.int32).itemsize / 2**20,
            int64_temp_mib=elements * np.dtype(np.int64).itemsize / 2**20,
            albucore_ms=albucore_t.median,
            numpy_ms=numpy_t.median,
            torch_int32_ms=int32_t.median,
            torch_eager_ms=eager_t.median,
            torch_compile_ms=None if compiled_t is None else compiled_t.median,
        ),
        compile_ms,
    )


def _format(rows: list[Row]) -> list[str]:
    lines = [
        "| table | layout | shape | elements | int32 / int64 temp | Albucore ms | NumPy byte-LUT ms | Torch eager int32 ms | int32 / NumPy | Torch eager int64 ms | int64 / NumPy | Torch compile ms | compile / NumPy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        compiled_ms = "—" if row.torch_compile_ms is None else f"{row.torch_compile_ms:.4f}"
        compiled_ratio = "—" if row.torch_compile_ms is None else f"{row.torch_compile_ms / row.numpy_ms:.2f}×"
        lines.append(
            f"| {row.table_kind} | {row.layout} | {'×'.join(map(str, row.shape))} | {row.elements:,} | "
            f"{row.int32_temp_mib:.1f} / {row.int64_temp_mib:.1f} MiB | "
            f"{row.albucore_ms:.4f} | {row.numpy_ms:.4f} | {row.torch_int32_ms:.4f} | "
            f"{row.torch_int32_ms / row.numpy_ms:.2f}× | {row.torch_eager_ms:.4f} | "
            f"{row.torch_eager_ms / row.numpy_ms:.2f}× | {compiled_ms} | {compiled_ratio} |",
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use the first two HWC sizes and C=1/3.")
    parser.add_argument("--volumes", action="store_true", help="Include canonical DHWC and NDHWC shapes.")
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also time separately warmed torch.compile on representative static C=3 shapes.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.threads)

    rng = np.random.default_rng(42)
    shared_table = rng.permutation(256).astype(np.uint8)
    hw_sizes = ROUTER_HWC_FULL_HW[:2] if args.quick else ROUTER_HWC_FULL_HW
    channels = (1, 3) if args.quick else (1, 3, 9)
    shapes = [(*hw, channels_count) for hw in hw_sizes for channels_count in channels]
    if args.volumes:
        shapes.extend(shape for shape in SCALE_LUT_SHAPES if len(shape) in (4, 5))

    rows: list[Row] = []
    compile_times: list[float] = []
    for shape in shapes:
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        per_channel_table = np.stack(
            [rng.permutation(256).astype(np.uint8) for _ in range(shape[-1])],
        )
        for table, per_channel in ((shared_table, False), (per_channel_table, True)):
            row, compile_ms = _row(
                image,
                table,
                repeats=args.repeats,
                warmup=args.warmup,
                include_compile=args.compile and shape in COMPILED_STATIC_SHAPES,
                per_channel=per_channel,
            )
            rows.append(row)
            if compile_ms is not None:
                compile_times.append(compile_ms)

    print("# Torch uint8 LUT benchmark")
    print()
    print(
        f"Platform: `{platform.platform()}` (`{platform.machine()}`); Torch `{torch.__version__}`; "
        f"NumPy `{np.__version__}`; OpenCV `{cv2.__version__}`. CPU threads: `{args.threads}`; "
        f"repeats: `{args.repeats}`; warmup: `{args.warmup}`.",
    )
    print()
    print(
        "Inputs are already CPU Tensors during Torch timings. `torch.from_numpy` is deliberately outside the timed "
        "call, while the int32/int64 casts are inside each eager expression. `torch.compile` rows exclude compilation "
        "and only apply to a persistent pipeline that can precompile each static shape.",
    )
    print()
    if compile_times:
        print(
            "Per-shape compile-and-first-call cost in the current compiler cache: "
            f"median `{np.median(compile_times):.0f} ms` (excluded from rows).",
        )
        print()
    print(*_format(rows), sep="\n")


if __name__ == "__main__":
    main()
