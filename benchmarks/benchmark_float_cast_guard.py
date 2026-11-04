"""Paired guard and public FP16-to-FP32 conversion measurements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import cv2
import numkong as nk
import numpy as np
import torch
from numpy.typing import NDArray

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import albucore
import albucore._float_cast as cast_router
from albucore import convert_dtype
from benchmarks.timing import _check_resume_profile


def _masked_guard(image: NDArray[np.float16]) -> bool:
    return bool(np.max(image.view(np.uint16) & 0x7C00) != 0x7C00)


def _sample(call: Callable[[], object], count: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(count):
        call()
    return (time.perf_counter_ns() - start) / count


def _peak(call: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _memory_worker(layer: str, variant: str) -> dict[str, object]:
    import resource

    candidate = cast_router._numkong_cast_preserves_numpy_errors
    cast_router._numkong_cast_preserves_numpy_errors = _masked_guard if variant == "baseline" else candidate
    image = np.full((2048, 2048, 3), 0.5, dtype=np.float16)
    image[::2] = -0.5
    destination = np.full(image.shape, 0, dtype=np.float32) if layer == "out" else None
    for _ in range(3):
        if layer == "guard":
            cast_router._numkong_cast_preserves_numpy_errors(image)
        else:
            convert_dtype(image, np.float32, out=destination)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak *= 1024
        steady = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    else:
        steady = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()) * 1024
    return {
        "layer": layer,
        "variant": variant,
        "input_bytes": image.nbytes,
        "output_bytes": 0 if destination is None else destination.nbytes,
        "peak_rss_bytes": peak,
        "steady_rss_bytes": steady,
    }


def _raw_unit_scale_pairs(args: argparse.Namespace, metadata: dict[str, object]) -> None:
    from albucore.lut import _cv2_lut_channel_last
    import albucore.convert as conversion

    metadata["conversion_source_sha256"] = hashlib.sha256(Path(conversion.__file__).read_bytes()).hexdigest()
    metadata["baseline"] = "Published 0.3.3 unit-scale float32 LUT; candidate is the complete public converter"
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        _check_resume_profile(previous["metadata"], metadata)
        rows = previous["rows"]
    completed = {row["case"] for row in rows}
    for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)):
        for channels in (1, 3, 9):
            image = np.resize(np.arange(256, dtype=np.uint8), (height, width, channels))
            expected = image.astype(np.float32)
            for layer in ("allocate", "out"):
                if f"{height}x{width}x{channels}/{layer}" in completed:
                    continue
                output = np.empty_like(expected) if layer == "out" else None
                calls = {
                    "baseline": lambda: _cv2_lut_channel_last(
                        image, (np.arange(256, dtype=np.float32) / 1).astype(np.float32), out=output
                    ),
                    "candidate": lambda: convert_dtype(image, np.float32, max_value=1, out=output),
                }
                for call in calls.values():
                    np.testing.assert_array_equal(call(), expected)
                    for _ in range(10):
                        call()
                pilot = max(_sample(call, 10) for call in calls.values())
                count = max(args.minimum_count, min(100_000, math.ceil(25_000_000 / pilot)))
                samples = {name: [] for name in calls}
                for repetition in range(args.samples):
                    for name in ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline"):
                        samples[name].append(_sample(calls[name], count))
                medians = {name: statistics.median(values) for name, values in samples.items()}
                row = {
                    "case": f"{height}x{width}x{channels}/{layer}",
                    "median_ns": medians,
                    "samples_ns": samples,
                    "calls_per_sample": count,
                    "candidate_over_baseline": medians["candidate"] / medians["baseline"],
                    "bitwise_equal": True,
                }
                rows.append(row)
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
                temporary.replace(args.output)
                print(row["case"], round(row["candidate_over_baseline"], 4), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--minimum-count", type=int, default=1)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--raw-unit-scale", action="store_true")
    parser.add_argument("--memory-worker", choices=("guard", "allocate", "out"))
    parser.add_argument("--variant", choices=("baseline", "candidate"), default="candidate")
    args = parser.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not Path(albucore.__file__).is_relative_to(repo_root):
        raise RuntimeError(f"Expected Core source under {repo_root}, got {albucore.__file__}")
    capabilities = nk.get_capabilities()
    if platform.system() == "Darwin" and not capabilities.get("neon"):
        raise RuntimeError("Native ARM measurements require NEON detection")
    cpu = (
        subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Darwin"
        else next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    )
    metadata = {
        "baseline_commit": "3c8769246c0f870145dbb8ba9416025940e92811",
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline": "FP16 exponent-mask guard from published 0.3.3; other public-router code shared",
        "candidate_source_sha256": hashlib.sha256(Path(cast_router.__file__).read_bytes()).hexdigest(),
        "core_source": albucore.__file__,
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "capabilities": capabilities,
        "native_x86_cast_route": cast_router._NUMKONG_X86_CAST,
        "threads": {
            "opencv": cv2.getNumThreads(),
            "torch": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
        },
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "samples": args.samples,
        "minimum_count": args.minimum_count,
    }
    if args.raw_unit_scale:
        _raw_unit_scale_pairs(args, metadata)
        return
    if args.memory_worker:
        result = _memory_worker(args.memory_worker, args.variant)
        args.output.write_text(json.dumps({"metadata": metadata, "memory": result}, indent=2) + "\n")
        return
    selected = set(json.loads(args.case_list.read_text())) if args.case_list else None
    rows: list[dict[str, Any]] = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        _check_resume_profile(previous["metadata"], metadata)
        rows = previous["rows"]
    completed = {row["case"] for row in rows}
    candidate_guard = cast_router._numkong_cast_preserves_numpy_errors
    guards = {"baseline": _masked_guard, "candidate": candidate_guard}
    try:
        for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)):
            for channels in (1, 3, 9):
                for pattern, limits in (("positive", (0.1, 0.9)), ("negative", (-0.9, -0.1)), ("mixed", (-2, 2))):
                    image = (
                        np.linspace(*limits, height * width * channels, dtype=np.float32)
                        .reshape(height, width, channels)
                        .astype(np.float16)
                    )
                    source_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    reference = image.astype(np.float32)
                    for layer in ("guard", "allocate", "out"):
                        case = f"{height}x{width}x{channels}/{pattern}/{layer}"
                        if case in completed or (selected is not None and case not in selected):
                            continue
                        destination = np.empty(image.shape, dtype=np.float32) if layer == "out" else None
                        calls: dict[str, Callable[[], object]] = {}
                        peaks: dict[str, int] = {}
                        for variant, guard in guards.items():
                            cast_router._numkong_cast_preserves_numpy_errors = guard
                            call = (
                                (lambda guard=guard: guard(image))
                                if layer == "guard"
                                else (lambda: convert_dtype(image, np.float32, out=destination))
                            )
                            actual = call()
                            if isinstance(actual, np.ndarray):
                                np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
                            else:
                                assert actual is True
                            for _ in range(10):
                                call()
                            calls[variant] = call
                            peaks[variant] = _peak(call)
                        pilot: list[float] = []
                        for variant in guards:
                            cast_router._numkong_cast_preserves_numpy_errors = guards[variant]
                            pilot.append(_sample(calls[variant], 10))
                        count = max(args.minimum_count, min(100_000, math.ceil(25_000_000 / max(pilot))))
                        samples: dict[str, list[float]] = {variant: [] for variant in guards}
                        for repetition in range(args.samples):
                            for variant in (
                                ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                            ):
                                cast_router._numkong_cast_preserves_numpy_errors = guards[variant]
                                samples[variant].append(_sample(calls[variant], count))
                        medians = {variant: statistics.median(values) for variant, values in samples.items()}
                        assert hashlib.sha256(image.tobytes()).hexdigest() == source_hash
                        row = {
                            "case": case,
                            "shape": image.shape,
                            "dtype": "float16",
                            "calls_per_sample": count,
                            "median_ns": medians,
                            "samples_ns": samples,
                            "candidate_over_baseline": medians["candidate"] / medians["baseline"],
                            "traced_extra_peak_bytes": peaks,
                            "output_buffer_bytes": 0 if destination is None else destination.nbytes,
                            "bitwise_equal": True,
                        }
                        rows.append(row)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        temporary = args.output.with_suffix(".tmp")
                        temporary.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
                        temporary.replace(args.output)
                        print(case, round(row["candidate_over_baseline"], 4), flush=True)
    finally:
        cast_router._numkong_cast_preserves_numpy_errors = candidate_guard


if __name__ == "__main__":
    main()
