"""Paired public uint8 LUT allocation timings and fresh-process memory against a saved lut.py."""

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
import stringzilla as sz
import torch
from benchmark_filtering_cleanup import current_rss_bytes, measure
from benchmark_output import output_path, write_json_atomic
from timing import _check_resume_profile


def cases(quick):
    sizes = ((240, 320),) if quick else ((128, 160), (240, 320), (480, 640), (768, 1024))
    for height, width in sizes:
        for channels in (1, 3, 9):
            for layout in ("contiguous", "column_strided", "negative", "fortran"):
                yield (height, width, channels), layout, "allocate"
    if not quick:
        for channels in (1, 3, 9):
            for shape in ((3, 5, channels), (2, 128, 160, channels), (2, 3, 128, 160, channels)):
                yield shape, "contiguous", "allocate"
    for channels in (1, 3, 9):
        for destination in ("allocate", "contiguous", "strided"):
            yield (240, 320, channels), "readonly", destination


def workload(module, shape, layout, destination):
    source_shape = (*shape[:-2], shape[-2] * 2, shape[-1]) if layout == "column_strided" else shape
    image = np.random.default_rng(137).integers(0, 256, source_shape, dtype=np.uint8)
    if layout == "column_strided":
        image = image[..., ::2, :]
    elif layout == "negative":
        image = image[..., ::-1, :]
    elif layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "readonly":
        image.flags.writeable = False
    table = np.random.default_rng(138).permutation(256).astype(np.uint8)
    out = None
    if destination == "contiguous":
        out = np.empty(shape, dtype=np.uint8)
    elif destination == "strided":
        out = np.empty((*shape[:-2], shape[-2] * 2, shape[-1]), dtype=np.uint8)[..., ::2, :]
    return image, table, out, partial(module.apply_uint8_lut, image, table, out=out)


def memory(module, layout):
    image, _, _, call = workload(module, (2048, 3072, 9), layout, "allocate")
    gc.collect()
    before = current_rss_bytes()
    tracemalloc.start()
    result = call()
    live, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    row = {
        "input_bytes": image.nbytes,
        "result_bytes": result.nbytes,
        "result_writable": bool(result.flags.writeable),
        "traced_live_bytes": live,
        "traced_peak_bytes": peak,
        "rss_before_bytes": before,
        "rss_with_result_bytes": current_rss_bytes(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    del result
    for _ in range(10):
        result = call()
        del result
    gc.collect()
    row["rss_after_release_bytes"] = current_rss_bytes()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=output_path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--memory-worker", choices=("baseline", "candidate"))
    parser.add_argument("--memory-layout", choices=("contiguous", "column_strided"), default="contiguous")
    args = parser.parse_args()
    if not args.memory_worker and args.output is None:
        parser.error("--output is required unless --memory-worker is set")
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sys.platform == "darwin" and not nk.get_capabilities().get("neon"):
        raise RuntimeError("Native NumKong NEON is required")
    spec = importlib.util.spec_from_file_location("lut_baseline", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    if args.memory_worker:
        print(json.dumps(memory(baseline if args.memory_worker == "baseline" else core, args.memory_layout)))
        return
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin"
        else next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "capabilities": nk.get_capabilities(),
        "baseline_source": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        "candidate_source": hashlib.sha256((Path(core.__file__).parent / "lut.py").read_bytes()).hexdigest(),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quick": args.quick,
        "packages": sorted(
            [{"name": dist.metadata["Name"], "version": dist.version} for dist in importlib.metadata.distributions()],
            key=lambda item: item["name"].lower(),
        ),
        "source_paths": {module.__name__: module.__file__ for module in (core, cv2, nk, np, sz, torch)},
        "threads": {
            "environment": {
                name: os.environ[name]
                for name in os.environ
                if name.endswith("NUM_THREADS") or name == "VECLIB_MAXIMUM_THREADS"
            },
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
    for shape, layout, destination in cases(args.quick):
        label = f"{shape}|{layout}|{destination}"
        if label in completed:
            continue
        image, table, out, first = workload(baseline, shape, layout, destination)
        second = partial(core.apply_uint8_lut, image, table, out=out)
        original = image.tobytes()
        expected = table[image]
        for call in (first, second):
            if out is not None:
                np.bitwise_not(expected, out=out)
            actual = call()
            np.testing.assert_array_equal(actual, expected)
        assert actual.flags.writeable
        assert actual.shape == image.shape and actual.dtype == np.uint8
        assert not np.shares_memory(actual, image)
        if out is not None:
            assert actual is out
        del expected, actual
        row = {"case": label, "exact_output": True, **measure(first, second, 31, 3_000_000)}
        if row["ratio"] > 1.05:
            row["repeat"] = measure(first, second, 101, 5_000_000)
        assert image.tobytes() == original
        rows.append(row)
        write_json_atomic(args.output, {"metadata": metadata, "rows": rows})
    allocations = []
    if not args.skip_memory:
        for layout in ("contiguous", "column_strided"):
            for variant in ("baseline", "candidate"):
                for sample in range(3):
                    command = [
                        sys.executable,
                        __file__,
                        "--baseline-source",
                        str(args.baseline_source),
                        "--memory-worker",
                        variant,
                        "--memory-layout",
                        layout,
                    ]
                    row = json.loads(subprocess.check_output(command, text=True))
                    allocations.append({"layout": layout, "variant": variant, "sample": sample, **row})
    write_json_atomic(args.output, {"metadata": metadata, "rows": rows, "memory": allocations})
    ratios = [row.get("repeat", row)["ratio"] for row in rows]
    print(
        json.dumps(
            {
                "cells": len(rows),
                "median_candidate_over_baseline": statistics.median(ratios),
                "flags": [row["case"] for row in rows if row.get("repeat", row)["ratio"] > 1.05],
            }
        )
    )


if __name__ == "__main__":
    main()
