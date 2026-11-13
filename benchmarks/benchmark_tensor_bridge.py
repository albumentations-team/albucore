"""Pair dense NumPy-to-Tensor bridge calls against the generic NumPy copy."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
from importlib.metadata import version
from itertools import product
from statistics import median
from timeit import Timer

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

import albucore
from albucore.tensor_bridge import TensorLayout, numpy_to_tensor_layout, numpy_to_tensor_storage

_DTYPES = (np.uint8, np.float16, np.float32)
_LAYOUTS: tuple[TensorLayout, ...] = ("CHW", "NCHW", "CDHW", "NCDHW")


def _report(writer, operation, layout, name, source, before, after, minimum_seconds):
    expected = before().numpy()
    candidate = after().numpy()
    if (
        not candidate.flags.c_contiguous
        or candidate.dtype != expected.dtype
        or candidate.shape != expected.shape
        or candidate.tobytes() != expected.tobytes()
    ):
        raise AssertionError((operation, layout, name, source.shape, source.dtype, source.strides))
    timers = (Timer(before), Timer(after))
    count = max(100, min(20_000, round(minimum_seconds / max(timer.timeit(10) / 10 for timer in timers))))
    times = [[], []]
    for repeat in range(5):
        for index in (repeat % 2, 1 - repeat % 2):
            times[index].append(timers[index].timeit(count) * 1e6 / count)
    baseline, result = map(median, times)
    writer.writerow(
        (
            operation,
            layout,
            name,
            source.shape,
            source.strides,
            source.dtype.name,
            baseline,
            result,
            baseline / result,
            count,
        )
    )
    sys.stdout.flush()


def _layout_cases(writer, rng, quick):
    sizes = ((192, 208),) if quick else ((64, 80), (128, 144), (192, 208), (256, 320), (512, 640), (1024, 1024))
    minimum_seconds = 0.012 if quick else 0.05
    for layout, (height, width), channels, dtype in product(_LAYOUTS, sizes, (1, 2, 3, 4, 5), _DTYPES):
        batched = layout.startswith("N")
        prefix = ((4,) if batched else ()) + ((4,) if "D" in layout else ())
        if np.prod((*prefix, height, width, channels)) > 16_777_216:
            continue
        array = rng.integers(0, 256, (*prefix, height, width, channels), dtype=np.uint8).astype(dtype, copy=False)
        axes = (0, array.ndim - 1, *range(1, array.ndim - 1)) if batched else (array.ndim - 1, *range(array.ndim - 1))
        for label, source in (
            ("dense", array),
            ("crop", array[..., 1:-1, 1:-1, :]),
            ("horizontal", array[..., ::-1, :]),
            ("vertical", array[..., ::-1, :, :]),
            ("double", array[..., ::-1, ::-1, :]),
            ("transpose", array.swapaxes(-3, -2)),
            ("rotate", np.rot90(array, axes=(-3, -2))),
            ("stepped", array[..., ::2, ::2, :]),
        ):

            def baseline(source=source, axes=axes):
                return torch.from_numpy(np.ascontiguousarray(source.transpose(axes), dtype=source.dtype))

            def candidate(source=source, layout=layout):
                return numpy_to_tensor_layout(source, layout, contiguous=True)

            _report(writer, "layout", layout, label, source, baseline, candidate, minimum_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run one representative spatial size.")
    parser.add_argument("--storage-only", action="store_true", help="Measure scalar masks only.")
    args = parser.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    packages = {name: version(name) for name in ("numpy", "torch", "numkong", "stringzilla")}
    print(platform.platform(), sys.version.split()[0], albucore.__file__, packages, cv2.__version__, file=sys.stderr)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ("operation", "layout", "view", "shape", "strides", "dtype", "numpy_us", "core_us", "speedup", "iterations")
    )
    rng = np.random.default_rng(137)
    if not args.storage_only:
        _layout_cases(writer, rng, args.quick)
    for dtype in (*_DTYPES, np.int16, np.uint16, np.int32, np.int64):
        array = rng.integers(0, 256, (16, 192, 208), dtype=np.uint8).astype(dtype, copy=False)
        for name, source in (("horizontal", array[..., ::-1]), ("double", array[..., ::-1, ::-1])):
            _report(
                writer,
                "storage",
                "DHW",
                name,
                source,
                lambda source=source: torch.from_numpy(np.ascontiguousarray(source, dtype=source.dtype)),
                lambda source=source: numpy_to_tensor_storage(source, contiguous=True),
                0.012 if args.quick else 0.05,
            )


if __name__ == "__main__":
    main()
