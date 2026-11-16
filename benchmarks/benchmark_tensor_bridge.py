"""Compare dense NumPy-to-Tensor conversions and storage-preserving views.

Report median and median absolute deviation per call in microseconds. Pixel-copy bytes
count result bytes in storage independent of the input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import runpy
import subprocess
import sys
from importlib.metadata import distributions, version
from itertools import product
from pathlib import Path
from statistics import median
from timeit import Timer

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in _THREAD_VARIABLES:
    os.environ[variable] = "1"

import cv2
import numkong as nk
import numpy as np
import stringzilla as sz
import torch

import albucore
from albucore.tensor_bridge import TensorLayout, numpy_to_tensor_layout, numpy_to_tensor_storage

_DTYPES = (np.uint8, np.float16, np.float32)
_LAYOUTS: tuple[TensorLayout, ...] = ("CHW", "NCHW", "CDHW", "NCDHW")


def _cpu_model() -> str:
    if sys.platform == "darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return platform.processor() or platform.machine()


def _report(writer, operation, layout, name, source, before, after, minimum_seconds, *, dense=True):
    expected = before().numpy()
    candidate = after().numpy()
    if (
        (dense and not candidate.flags.c_contiguous)
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
    baseline_mad = median(abs(sample - baseline) for sample in times[0])
    result_mad = median(abs(sample - result) for sample in times[1])
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
            baseline_mad,
            result_mad,
            0 if np.shares_memory(source, expected) else expected.nbytes,
            0 if np.shares_memory(source, candidate) else candidate.nbytes,
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


def _view_cases(writer, rng, quick, baseline):
    sizes = ((128, 160), (768, 1024)) if quick else ((128, 160), (240, 320), (480, 640), (768, 1024))
    minimum_seconds = 0.012 if quick else 0.05
    for (height, width), channels, dtype in product(sizes, (1, 3, 9), _DTYPES):
        array = rng.integers(0, 256, (height, width, channels), dtype=np.uint8).astype(dtype, copy=False)
        owner = torch.from_numpy(array)
        for name, source in (
            ("numpy-owning", array),
            ("numpy-crop", array[1:, 1:]),
            ("tensor-dense", owner.numpy()),
            ("tensor-transpose", owner.numpy().swapaxes(0, 1)),
            ("tensor-crop", owner.numpy()[1:, 1:]),
        ):
            result = numpy_to_tensor_storage(source)
            if name.startswith("tensor-") and result.untyped_storage() != owner.untyped_storage():
                raise AssertionError((name, source.shape, source.dtype, source.strides))
            _report(
                writer,
                "view",
                "HWC",
                name,
                source,
                lambda: baseline(source),
                lambda: numpy_to_tensor_storage(source),
                minimum_seconds,
                dense=False,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run one representative spatial size.")
    parser.add_argument("--storage-only", action="store_true", help="Measure scalar masks only.")
    parser.add_argument("--metadata-json", type=Path, help="Write run metadata to this file instead of stderr.")
    parser.add_argument(
        "--baseline-bridge",
        type=Path,
        help="Compare storage-preserving views with tensor_bridge.py from a baseline checkout.",
    )
    args = parser.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    packages = {name: version(name) for name in ("numpy", "torch", "numkong", "stringzilla")}
    metadata = {
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "albucore_source": albucore.__file__,
        "versions": {**packages, "opencv": cv2.__version__},
        "package_inventory": {dist.metadata["Name"]: dist.version for dist in distributions()},
        "source_paths": {module.__name__: module.__file__ for module in (cv2, np, nk, sz, torch)},
        "numkong_capabilities": nk.get_capabilities(),
        "baseline_bridge": str(args.baseline_bridge) if args.baseline_bridge else None,
        "opencv_threads": cv2.getNumThreads(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ[name] for name in _THREAD_VARIABLES},
    }
    metadata_json = json.dumps(metadata, indent=2)
    if args.metadata_json:
        args.metadata_json.write_text(metadata_json + "\n", encoding="utf-8")
    else:
        print(metadata_json, file=sys.stderr)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "operation",
            "layout",
            "view",
            "shape",
            "strides",
            "dtype",
            "baseline_us",
            "core_us",
            "speedup",
            "iterations",
            "baseline_mad_us",
            "core_mad_us",
            "baseline_pixel_copy_bytes",
            "core_pixel_copy_bytes",
        )
    )
    if args.baseline_bridge:
        baseline = runpy.run_path(str(args.baseline_bridge))["numpy_to_tensor_storage"]
        _view_cases(writer, np.random.default_rng(137), args.quick, baseline)
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
