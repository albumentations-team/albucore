# ruff: noqa: C901, E402, INP001, PLR0912, S101, T201
"""Paired one-thread padding benchmarks, including complete container bridges.

Run from the checkout with ``python benchmarks/benchmark_pad3d.py --quick``.
Omit ``--quick`` for the canonical and issue #159 volume grids plus stride,
fill, int16 mask, unit-axis, and identity controls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import platform
import statistics
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import TypeAlias

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from albucore import pad3d
from albucore.pad3d import _cast_pad_value, _pad3d_numpy

Shape: TypeAlias = tuple[int, int, int, int]
Padding3D: TypeAlias = tuple[int, int, int, int, int, int]
FillValue: TypeAlias = float | tuple[float, ...]
Volume: TypeAlias = np.ndarray | torch.Tensor
NumpyKernel: TypeAlias = Callable[[np.ndarray, Padding3D, FillValue], np.ndarray]
BenchmarkFunction: TypeAlias = Callable[[], Volume]
Case: TypeAlias = tuple[Shape, type[np.generic], str, str, Padding3D, FillValue]


def _numpy_pad(volume: np.ndarray, padding: Padding3D, value: FillValue) -> np.ndarray:
    if not any(padding):
        return volume
    front, back, top, bottom, left, right = padding
    return np.pad(volume, ((front, back), (top, bottom), (left, right), (0, 0)), constant_values=value)


def _numpy_faces(volume: np.ndarray, padding: Padding3D, value: FillValue) -> np.ndarray:
    if not any(padding):
        return volume
    if isinstance(value, tuple):
        return _numpy_pad(volume, padding, value)
    front, back, top, bottom, left, right = padding
    depth, height, width, channels = volume.shape
    value = _cast_pad_value(value, volume.dtype)
    output = np.empty((depth + front + back, height + top + bottom, width + left + right, channels), volume.dtype)
    output[front : front + depth, top : top + height, left : left + width] = volume
    output[:front] = value
    output[front + depth :] = value
    output[front : front + depth, :top] = value
    output[front : front + depth, top + height :] = value
    output[front : front + depth, top : top + height, :left] = value
    output[front : front + depth, top : top + height, left + width :] = value
    return output


def _numpy_route(volume: Volume, padding: Padding3D, value: FillValue, kernel: NumpyKernel) -> Volume:
    if not any(padding):
        return volume
    if isinstance(volume, np.ndarray):
        return kernel(volume, padding, value)
    result = kernel(volume.permute(1, 2, 3, 0).numpy(), padding, value)
    return torch.from_numpy(result).permute(3, 0, 1, 2)


@torch.no_grad()
def _torch_pad(
    volume: Volume,
    padding: Padding3D,
    value: FillValue,
    *,
    batch: bool = False,
    dhwc: bool = False,
) -> Volume:
    if not any(padding):
        return volume
    if isinstance(value, tuple):
        return _numpy_route(volume, padding, value, _numpy_pad)
    front, back, top, bottom, left, right = padding
    torch_padding = (left, right, top, bottom, front, back)
    is_numpy = isinstance(volume, np.ndarray)
    if is_numpy:
        if not volume.flags.writeable or any(stride < 0 for stride in volume.strides):
            volume = volume.copy()
        tensor = torch.from_numpy(volume)
        torch_padding = (0, 0, *torch_padding)
    else:
        tensor = volume.unsqueeze(0) if batch else volume
        if dhwc:
            tensor = volume.permute(1, 2, 3, 0)
            torch_padding = (0, 0, *torch_padding)
    dtype = (
        volume.dtype
        if is_numpy
        else {torch.uint8: np.uint8, torch.int16: np.int16, torch.float32: np.float32}[volume.dtype]
    )
    fill = _cast_pad_value(value, dtype)
    result = torch_f.pad(tensor, torch_padding, value=fill)
    if batch:
        result = result.squeeze(0)
    if dhwc:
        result = result.permute(3, 0, 1, 2)
    return result.numpy() if is_numpy else result


@torch.no_grad()
def _torch_full(volume: Volume, padding: Padding3D, value: FillValue) -> Volume:
    if not any(padding):
        return volume
    if isinstance(value, tuple):
        return _numpy_route(volume, padding, value, _numpy_pad)
    front, back, top, bottom, left, right = padding
    is_numpy = isinstance(volume, np.ndarray)
    if is_numpy:
        if not volume.flags.writeable or any(stride < 0 for stride in volume.strides):
            volume = volume.copy()
        tensor = torch.from_numpy(volume)
        depth, height, width, channels = volume.shape
        shape = (depth + front + back, height + top + bottom, width + left + right, channels)
        interior = (slice(front, front + depth), slice(top, top + height), slice(left, left + width))
        dtype = volume.dtype
    else:
        tensor = volume
        channels, depth, height, width = volume.shape
        shape = (channels, depth + front + back, height + top + bottom, width + left + right)
        interior = (slice(None), slice(front, front + depth), slice(top, top + height), slice(left, left + width))
        dtype = {torch.uint8: np.uint8, torch.int16: np.int16, torch.float32: np.float32}[volume.dtype]
    fill = _cast_pad_value(value, dtype)
    output = tensor.new_full(shape, fill)
    output[interior] = tensor
    return output.numpy() if is_numpy else output


def _numpy_planar_full(volume: torch.Tensor, padding: Padding3D, value: FillValue) -> torch.Tensor:
    if not any(padding):
        return volume
    if isinstance(value, tuple):
        return _numpy_route(volume, padding, value, _numpy_pad)
    front, back, top, bottom, left, right = padding
    array = volume.numpy()
    channels, depth, height, width = array.shape
    output = np.full(
        (channels, depth + front + back, height + top + bottom, width + left + right),
        _cast_pad_value(value, array.dtype),
        dtype=array.dtype,
    )
    output[:, front : front + depth, top : top + height, left : left + width] = array
    return torch.from_numpy(output)


@torch.no_grad()
def _torch_faces(volume: torch.Tensor, padding: Padding3D, value: FillValue) -> torch.Tensor:
    if not any(padding):
        return volume
    if isinstance(value, tuple):
        return _numpy_route(volume, padding, value, _numpy_pad)
    front, back, top, bottom, left, right = padding
    channels, depth, height, width = volume.shape
    dtype = {torch.uint8: np.uint8, torch.int16: np.int16, torch.float32: np.float32}[volume.dtype]
    fill = _cast_pad_value(value, dtype)
    output = volume.new_empty((channels, depth + front + back, height + top + bottom, width + left + right))
    output[:, front : front + depth, top : top + height, left : left + width] = volume
    output[:, :front] = fill
    output[:, front + depth :] = fill
    output[:, front : front + depth, :top] = fill
    output[:, front : front + depth, top + height :] = fill
    output[:, front : front + depth, top : top + height, :left] = fill
    output[:, front : front + depth, top : top + height, left + width :] = fill
    return output


def _elapsed(function: BenchmarkFunction, iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - start) / 1e9


def _paired(
    functions: dict[str, BenchmarkFunction],
    rounds: int,
    seconds: float,
) -> tuple[dict[str, int], dict[str, list[float]]]:
    for function in functions.values():
        _elapsed(function, 3)
    iterations = {
        name: max(3, math.ceil(seconds / (_elapsed(function, 3) / 3)))
        for name, function in functions.items()
    }
    samples = {name: [] for name in functions}
    names = list(functions)
    gc.disable()
    try:
        for round_index in range(rounds):
            for name in names if round_index % 2 == 0 else names[::-1]:
                samples[name].append(_elapsed(functions[name], iterations[name]) * 1000 / iterations[name])
    finally:
        gc.enable()
    return iterations, samples


def _cases(quick: bool) -> Iterator[Case]:
    shapes = [(16, 24, 32, 1), (64, 96, 128, 3)]
    if not quick:
        shapes = [
            (16, 128, 160, 1),
            (16, 128, 160, 3),
            (32, 128, 160, 1),
            (32, 128, 160, 3),
            (64, 128, 160, 3),
            (96, 128, 160, 1),
            (48, 240, 320, 3),
            (16, 128, 160, 9),
            (64, 96, 128, 9),
            *((*size, channels) for size in ((16, 24, 32), (64, 96, 128), (128, 192, 256)) for channels in (1, 3, 5)),
        ]
    for shape in shapes:
        for dtype in (np.uint8, np.float32):
            for container in ("numpy", "tensor"):
                for layout in ("contiguous", "positive" if container == "numpy" else "channel_last"):
                    yield shape, dtype, container, layout, (4, 4, 8, 8, 12, 12), 0.25 if dtype == np.float32 else 7
    for container in ("numpy", "tensor"):
        for dtype in (np.uint8, np.int16, np.float32):
            for padding, value in (((0, 0, 0, 0, 0, 0), 7), ((1, 7, 13, 3, 2, 22), 7), ((1, 2, 2, 1, 3, 2), (3, 7))):
                yield (16, 24, 32, 5), dtype, container, "contiguous", padding, value
        yield (1, 3, 1, 1), np.float32, container, "contiguous", (1, 0, 0, 1, 2, 0), 0.25
    for layout in ("negative", "readonly"):
        yield (64, 96, 128, 3), np.float32, "numpy", layout, (4, 4, 8, 8, 12, 12), 0.25
    for channels in (1, 3):
        for dtype in (np.uint8, np.float32):
            yield (32, 128, 160, channels), dtype, "numpy", "fortran", (4, 4, 8, 8, 12, 12), 7
    for layout in ("sliced", "planar_sliced", "channel_last"):
        for dtype in (np.uint8, np.int16, np.float32):
            yield (64, 96, 128, 3), dtype, "tensor", layout, (1, 7, 13, 3, 2, 22), 7
    if not quick:
        for depth in (16, 31, 32, 33, 63, 64, 65, 127, 128, 129):
            for channels in (1, 2):
                for dtype in (np.uint8, np.float32):
                    layout = "contiguous" if channels == 1 else "positive"
                    yield (depth, 64, 256, channels), dtype, "numpy", layout, (4, 4, 8, 8, 12, 12), 7


def _make_volume(
    rng: np.random.Generator,
    shape: Shape,
    dtype: type[np.generic],
    container: str,
    layout: str,
) -> Volume:
    array = rng.integers(0, 128, shape, dtype=np.uint8).astype(dtype)
    if layout in ("positive", "sliced"):
        array = np.repeat(array, 2, axis=1)[:, ::2]
    if container == "tensor":
        volume = torch.from_numpy(array).permute(3, 0, 1, 2)
        if layout == "planar_sliced":
            return volume.contiguous().repeat_interleave(2, dim=2)[:, :, ::2]
        return volume.contiguous() if layout == "contiguous" else volume
    if layout == "negative":
        array = array[::-1, :, ::-1]
    elif layout == "readonly":
        array.flags.writeable = False
    elif layout == "fortran":
        array = np.asfortranarray(array)
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--container", choices=("numpy", "tensor"), help="Restrict the input container")
    parser.add_argument("--router-only", action="store_true", help="Compare only public padding and the NumPy baseline")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=0.03, help="Target seconds per candidate per round")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/benchmark_pad3d.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    metadata = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads_reported": cv2.getNumThreads(),
        "opencv_used": False,
        "rounds": args.rounds,
        "target_seconds": args.seconds,
        "quick": args.quick,
        "container": args.container,
        "router_only": args.router_only,
    }
    print(json.dumps(metadata), flush=True)
    rng = np.random.default_rng(137)
    rows = []
    for shape, dtype, container, layout, padding, value in _cases(args.quick):
        if args.container is not None and args.container != container:
            continue
        volume = _make_volume(rng, shape, dtype, container, layout)
        snapshot = volume.copy() if container == "numpy" else volume.clone()
        kernels = {
            "numpy_pad": partial(_numpy_route, kernel=_numpy_pad),
            "numpy_full": partial(_numpy_route, kernel=_pad3d_numpy),
            "numpy_faces": partial(_numpy_route, kernel=_numpy_faces),
            "torch_pad": _torch_pad,
            "public": pad3d,
            "torch_full": _torch_full,
        }
        if container == "tensor":
            kernels["torch_pad_5d"] = partial(_torch_pad, batch=True)
            kernels["torch_pad_dhwc"] = partial(_torch_pad, dhwc=True)
            kernels["numpy_planar_full"] = _numpy_planar_full
            kernels["torch_faces"] = _torch_faces
        if args.router_only:
            kernels = {name: kernels[name] for name in ("numpy_pad", "public")}
        functions = {name: partial(fn, volume, padding, value) for name, fn in kernels.items()}
        expected = functions["numpy_pad"]()
        for fn in functions.values():
            actual = fn()
            assert actual.dtype == expected.dtype
            if container == "numpy":
                np.testing.assert_array_equal(actual, expected)
            else:
                assert torch.equal(actual, expected)
            del actual
        del expected
        iterations, samples = _paired(functions, args.rounds, args.seconds)
        if container == "numpy":
            np.testing.assert_array_equal(volume, snapshot)
        else:
            assert torch.equal(volume, snapshot)
        row = {
            "shape_dhwc": shape,
            "dtype": np.dtype(dtype).name,
            "container": container,
            "layout": layout,
            "strides": volume.strides if container == "numpy" else volume.stride(),
            "padding": padding,
            "value": value,
            "iterations": iterations,
            "samples_ms": samples,
        }
        rows.append(row)
        ratio = statistics.median(samples["numpy_pad"]) / statistics.median(samples["public"])
        print(f"{len(rows)} {container} {shape} {row['dtype']} {layout}: baseline/public {ratio:.2f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
