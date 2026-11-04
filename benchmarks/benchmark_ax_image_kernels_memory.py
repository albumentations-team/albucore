"""Fresh-process allocation and RSS comparison for the six AX/Core kernels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import resource
import subprocess
import sys
import tracemalloc
from itertools import product
from pathlib import Path


def worker(args: argparse.Namespace) -> dict:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.baseline_ax.resolve())]
    import albumentations as albu
    import cv2
    import numpy as np
    import psutil
    import torch
    from benchmark_ax_image_kernels import OWNER_MODULES, _calls, _facades, _swaps

    import albucore as core

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    baseline = {name: getattr(importlib.import_module(module), name) for name, module in OWNER_MODULES.items()}
    swaps = _swaps(baseline, _facades(core, cv2))
    for module, key, old, new in swaps:
        setattr(module, key, old if args.variant == "baseline" else new)
    name, layer, dtype = args.case.split("|")
    channels = 3 if name == "linear_transformation_rgb" else (9 if layer == "core" else 5)
    values = np.random.default_rng(137).random((1024, 1024, channels), dtype=np.float32)
    image = (values * 255).astype(np.uint8) if dtype == "uint8" else values.astype(np.float16).astype(dtype)
    del values
    image.flags.writeable = False
    calls = _calls(albu, core, np, name, layer, image, baseline)
    call = calls[args.variant]
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    tracemalloc.start()
    for _ in range(10):
        result = call()
        del result
    _, allocation_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    steady = process.memory_info().rss
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    high_water *= 1 if sys.platform == "darwin" else 1024
    result = call()
    if result.shape != image.shape or result.dtype != image.dtype or np.shares_memory(result, image):
        raise RuntimeError("Output metadata or ownership changed")
    digest = hashlib.sha256(result.tobytes()).hexdigest()
    return {
        "case": args.case,
        "variant": args.variant,
        "shape": list(image.shape),
        "output_sha256": digest,
        "output_bytes": result.nbytes,
        "allocation_peak_bytes": allocation_peak,
        "rss_before_bytes": before,
        "rss_steady_bytes": steady,
        "rss_peak_bytes": high_water,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ax", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-results", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(worker(args)))
        return
    from benchmark_ax_image_kernels import OWNER_MODULES

    metadata = {
        "timing_profile": json.loads(args.timing_results.read_text())["metadata"],
        "memory_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "calls_per_worker": 10,
    }
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["metadata"] != metadata:
            raise RuntimeError("Cannot resume a different memory comparison")
        rows = previous["rows"]
    completed_units = {(row["case"], row["repeat"], row["variant"]) for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, layer, dtype in product(OWNER_MODULES, ("core", "compose"), ("uint8", "float16", "float32")):
        case = f"{name}|{layer}|{dtype}"
        for repeat in range(args.repeats):
            for variant in ("baseline", "candidate"):
                if (case, repeat, variant) in completed_units:
                    continue
                completed = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--baseline-ax",
                        str(args.baseline_ax),
                        "--case",
                        case,
                        "--variant",
                        variant,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                rows.append({"repeat": repeat, **json.loads(completed.stdout)})
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
                temporary.replace(args.output)
            paired = [row for row in rows if row["case"] == case and row["repeat"] == repeat]
            if len({row["output_sha256"] for row in paired}) != 1:
                raise RuntimeError(f"Output mismatch: {case}")
        print(case, flush=True)


if __name__ == "__main__":
    main()
