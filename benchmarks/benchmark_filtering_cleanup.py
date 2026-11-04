"""Paired Core-only convolution and unsharp-mask timings against a saved filtering.py."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import albucore as core
import cv2
import numkong as nk
import numpy as np
import torch
from benchmark_output import output_path, write_json_atomic
from benchmark_router_synthetic import _bench_interleaved_pair
from timing import _check_resume_profile

OPERATIONS = ("convolve", "convolve_channels", "unsharp_mask")


def cases():
    for op in OPERATIONS:
        for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)):
            for channels in (1, 3, 9):
                for dtype in ("uint8", "float16", "float32"):
                    yield op, height, width, channels, dtype, "contiguous", 10
    for channels in (1, 3, 9):
        for dtype in ("uint8", "float16", "float32"):
            for layout in ("row_strided", "column_strided", "fortran"):
                yield "unsharp_mask", 240, 320, channels, dtype, layout, 10
            for threshold in (0, 255):
                yield "unsharp_mask", 240, 320, channels, dtype, "contiguous", threshold


def workload(module, case):
    op, height, width, channels, dtype, layout, threshold = case
    values = np.random.default_rng(137).random((height, width, channels), dtype=np.float32)
    image = (values * 255).astype(np.uint8) if dtype == "uint8" else values.astype(np.float16).astype(dtype)
    if layout == "row_strided":
        image = np.repeat(image, 2, axis=0)[::2]
    elif layout == "column_strided":
        image = np.repeat(image, 2, axis=1)[:, ::2]
    elif layout == "fortran":
        image = np.asfortranarray(image)
    image.flags.writeable = False
    if op == "unsharp_mask":
        return image, partial(module.unsharp_mask, image, 3, 1.0, 0.5, threshold)
    kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]], dtype=np.float32)
    if op == "convolve_channels":
        kernel = np.stack([kernel * (channel + 1) / channels for channel in range(channels)])
    return image, partial(module.convolve, image, kernel)


def repeat(call, count):
    for _ in range(count):
        call()


def current_rss_bytes() -> int:
    """Read the process's current resident memory in bytes."""
    if sys.platform == "linux":
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024


def measure(first, second, samples, target_ns):
    pilot = []
    for call in (first, second):
        start = time.perf_counter_ns()
        repeat(call, 5)
        pilot.append((time.perf_counter_ns() - start) / 5)
    count = max(100 if max(pilot) < 50_000 else 1, min(5000, int(target_ns / max(pilot)) + 1))
    old, new = _bench_interleaved_pair(partial(repeat, first, count), partial(repeat, second, count), samples, 3)
    return {
        "baseline_ns": old.median * 1e6 / count,
        "candidate_ns": new.median * 1e6 / count,
        "ratio": new.median / old.median,
        "calls_per_sample": count,
        "baseline_samples_ns": [value * 1e6 / count for value in old.raw],
        "candidate_samples_ns": [value * 1e6 / count for value in new.raw],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=output_path)
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=OPERATIONS)
    parser.add_argument("--rss-worker", choices=("baseline", "candidate"))
    parser.add_argument("--rss-operation", choices=OPERATIONS, default="unsharp_mask")
    parser.add_argument("--rss-dtype", default="float32")
    args = parser.parse_args()
    if not args.rss_worker and args.output is None:
        parser.error("--output is required unless --rss-worker is set")
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sys.platform == "darwin" and not nk.get_capabilities().get("neon"):
        raise RuntimeError("Native NumKong NEON is required")
    spec = importlib.util.spec_from_file_location("filtering_baseline", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    if args.rss_worker:
        image, call = workload(
            baseline if args.rss_worker == "baseline" else core,
            (args.rss_operation, 768, 1024, 9, args.rss_dtype, "contiguous", 10),
        )
        gc.collect()
        before = current_rss_bytes()
        repeat(call, 10)
        # Includes imports and input preparation; this is not a call-scoped peak.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        print(json.dumps({"before": before, "peak": peak, "steady": current_rss_bytes()}))
        return
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "capabilities": nk.get_capabilities(),
        "baseline_source": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        "candidate_source": hashlib.sha256((Path(core.__file__).parent / "filtering.py").read_bytes()).hexdigest(),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": sorted(
            [{"name": dist.metadata["Name"], "version": dist.version} for dist in importlib.metadata.distributions()],
            key=lambda item: item["name"].lower(),
        ),
        "threads": {
            "opencv": cv2.getNumThreads(),
            "torch": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
    }
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        _check_resume_profile(previous["metadata"], metadata)
        rows = previous["rows"]
    completed = {row["case"] for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases():
        if case[0] not in args.operations:
            continue
        label = "|".join(map(str, case))
        if label in completed:
            continue
        source, first = workload(baseline, case)
        second = partial(getattr(core, first.func.__name__), *first.args, **first.keywords)
        original = source.tobytes()
        expected, actual = first(), second()
        if expected.tobytes() != actual.tobytes() or expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise RuntimeError(f"Bitwise mismatch: {label}")
        if np.shares_memory(actual, source):
            raise RuntimeError(f"Output aliases input: {label}")
        del expected, actual
        row = {"case": label, "bitwise_equal": True, **measure(first, second, 11, 3_000_000)}
        if row["ratio"] > 1.05:
            row["repeat"] = measure(first, second, 41, 10_000_000)
        if source.tobytes() != original:
            raise RuntimeError(f"Input was modified: {label}")
        rows.append(row)
        write_json_atomic(args.output, {"metadata": metadata, "rows": rows})
    allocations = []
    for op in args.operations:
        for dtype in ("uint8", "float16", "float32"):
            peaks = {}
            for variant, module in [("baseline", baseline), ("candidate", core)]:
                image, call = workload(module, (op, 480, 640, 9, dtype, "contiguous", 10))
                gc.collect()
                tracemalloc.start()
                result = call()
                peaks[variant] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                del result
            allocations.append({"op": op, "dtype": dtype, **peaks})
    rss = []
    for op, dtype in [("convolve", "float32"), ("unsharp_mask", "float16"), ("unsharp_mask", "float32")]:
        if op not in args.operations:
            continue
        for variant in ("baseline", "candidate"):
            for sample in range(3):
                command = [
                    sys.executable,
                    __file__,
                    "--baseline-source",
                    str(args.baseline_source),
                    "--rss-worker",
                    variant,
                    "--rss-operation",
                    op,
                    "--rss-dtype",
                    dtype,
                ]
                row = json.loads(subprocess.check_output(command, text=True))
                rss.append({"op": op, "dtype": dtype, "variant": variant, "sample": sample, **row})
    write_json_atomic(args.output, {"metadata": metadata, "rows": rows, "allocations": allocations, "rss": rss})
    flags = [row["case"] for row in rows if row.get("repeat", row)["ratio"] > 1.05]
    print(
        json.dumps(
            {
                "cells": len(rows),
                "flags": flags,
                "median": statistics.median(row.get("repeat", row)["ratio"] for row in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
