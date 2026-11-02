"""Compare complete XY nearest calls with an unmodified affine3d.py baseline.

Export the baseline with ``git show <revision>:albucore/affine3d.py > /tmp/affine3d_before.py``, then run
``python benchmarks/benchmark_warp_affine3d_xy_nearest.py --baseline /tmp/affine3d_before.py --output result.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import runpy
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import benchmark_threads
import cv2
import numkong as nk
import numpy as np
import torch
from benchmark_warp_affine3d_batch import _time_candidates

import albucore
import albucore.affine3d as affine3d

Volume = np.ndarray | torch.Tensor
Warp = Callable[..., Volume]

SHAPES = (
    (1, 1, 4, 1),
    (2, 9, 11, 3),
    (2, 32, 127, 1),
    (2, 32, 128, 1),
    (2, 32, 129, 1),
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (16, 128, 160, 5),
    (16, 128, 160, 9),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
    (48, 240, 320, 3),
    (32, 288, 288, 1),
)


def _matrix(shape: tuple[int, ...]) -> np.ndarray:
    rotation = cv2.getRotationMatrix2D(((shape[2] - 1) / 2, (shape[1] - 1) / 2), 5, 1)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = rotation[:, :2]
    matrix[:2, 3] = rotation[:, 2]
    return matrix


def _input(source: np.ndarray, layout: str, batched: bool) -> np.ndarray | torch.Tensor:
    if not batched:
        source = source[0]
    if layout == "numpy":
        return source
    tensor = torch.from_numpy(source).movedim(-1, 1 if batched else 0)
    return tensor.contiguous() if layout == "tensor_contiguous" else tensor


def _array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    return value.numpy() if isinstance(value, torch.Tensor) else value


def _measure(
    functions: dict[str, Warp],
    inputs: tuple[Volume, ...],
    matrix: np.ndarray,
    size: tuple[int, ...],
    interpolations: tuple[int, ...],
    border: int,
    fill: float,
    args: argparse.Namespace,
) -> dict[str, dict]:
    def call(function: Warp) -> tuple[Volume, ...]:
        return tuple(
            function(data, matrix, size, mode, border, fill) for data, mode in zip(inputs, interpolations, strict=True)
        )

    before = [_array(data).copy() for data in inputs]
    expected = call(functions["baseline"])
    actual = call(functions["current"])
    for old, new, source in zip(expected, actual, inputs, strict=True):
        assert type(old) is type(new)
        assert new.shape == old.shape and new.dtype == old.dtype
        np.testing.assert_array_equal(_array(new), _array(old))
        assert not np.shares_memory(_array(new), _array(source))
    candidates = {name: lambda fn=fn: call(fn) for name, fn in functions.items()}
    timings = dict(_time_candidates(candidates, args.warmup, args.repeats))
    result = {name: asdict(timing) for name, timing in timings.items()}
    baseline, current = timings["baseline"], timings["current"]
    if (
        current.median_ms > 1.05 * baseline.median_ms
        and current.median_ms - 2 * current.mad_ms > baseline.median_ms + 2 * baseline.mad_ms
    ):
        result["followup"] = {
            "runs": [
                {name: asdict(timing) for name, timing in _time_candidates(candidates, 10, 101)} for _ in range(3)
            ],
        }
    for data, original in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(_array(data), original)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rss", choices=("baseline", "current"), help="Measure the pair in a fresh process.")
    args = parser.parse_args()
    if Path(albucore.__file__).resolve().parent.parent != Path(__file__).resolve().parents[1]:
        raise RuntimeError("Benchmark the checkout with PYTHONPATH set to its repository root.")
    threads = benchmark_threads.configure_libraries(torch, cv2, args.threads)
    capabilities = nk.get_capabilities()
    if platform.machine() == "arm64" and not capabilities["neon"]:
        raise RuntimeError("Native ARM64 acceptance requires NumKong NEON; run outside the sandbox.")
    functions = {"baseline": runpy.run_path(str(args.baseline))["warp_affine3d"], "current": albucore.warp_affine3d}
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": threads,
        "capabilities": capabilities,
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "sources": {module.__name__: module.__file__ for module in (albucore, np, torch, cv2, nk)},
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(Path(affine3d.__file__).read_bytes()).hexdigest(),
    }
    rng = np.random.default_rng(179)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.rss:
        source = rng.random((1, 32, 288, 288, 1), dtype=np.float32)
        mask = (source > 0.2).astype(np.uint8)
        matrix = _matrix(source.shape[1:])
        function = functions[args.rss]
        for _ in range(5):
            volume_result = function(source[0], matrix, (32, 288, 288), cv2.INTER_LINEAR)
            mask_result = function(mask[0], matrix, (32, 288, 288), cv2.INTER_NEAREST)
            volume_result[mask_result == 0] = 0
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        steady_kib = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True))
        args.output.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "peak_rss_mib": peak / (1024**2 if platform.system() == "Darwin" else 1024),
                    "steady_rss_mib": steady_kib / 1024,
                },
                indent=2,
            )
            + "\n"
        )
        return

    rows = []
    shapes = ((16, 128, 160, 1), (16, 128, 160, 3)) if args.quick else SHAPES
    workloads = [(shape, 1, False) for shape in shapes]
    workloads += [
        ((depth, 32, 40, channels), batch, True)
        for depth in (4, 16)
        for channels in (1, 3, 5, 9)
        for batch in (1, 4, 16)
    ]
    for shape, batch, batched in workloads:
        logical = rng.random((batch, *shape), dtype=np.float32).astype(np.float16)
        for layout in ("numpy", "tensor_view", "tensor_contiguous"):
            for dtype in (np.uint8, np.float16, np.float32):
                source = (logical * 3).astype(dtype) if dtype is np.uint8 else logical.astype(dtype)
                data = _input(source, layout, batched)
                for border, fill in ((cv2.BORDER_CONSTANT, 0), (cv2.BORDER_CONSTANT, 0.3), (cv2.BORDER_REPLICATE, 0)):
                    timing = _measure(
                        functions, (data,), _matrix(shape), shape[:3], (cv2.INTER_NEAREST,), border, fill, args
                    )
                    rows.append(
                        {
                            "shape": shape,
                            "batch": batch,
                            "batched": batched,
                            "layout": layout,
                            "dtype": np.dtype(dtype).name,
                            "border": border,
                            "fill": fill,
                            "operation": "nearest",
                            "timings": timing,
                        }
                    )
        args.output.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")

    for shape in ((16, 128, 160, 1), (32, 288, 288, 1), (32, 128, 160, 3)):
        logical = rng.random((1, *shape), dtype=np.float32)
        mask = (logical[..., :1] > 0.2).astype(np.uint8)
        for layout in ("numpy", "tensor_view", "tensor_contiguous"):
            data = (_input(logical, layout, False), _input(mask, layout, False))
            timing = _measure(
                functions,
                data,
                _matrix(shape),
                shape[:3],
                (cv2.INTER_LINEAR, cv2.INTER_NEAREST),
                cv2.BORDER_CONSTANT,
                0,
                args,
            )
            rows.append(
                {
                    "shape": shape,
                    "batch": 1,
                    "batched": False,
                    "layout": layout,
                    "dtype": "float32+uint8",
                    "border": 0,
                    "fill": 0,
                    "operation": "pair",
                    "timings": timing,
                }
            )
    args.output.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    ratios = [row["timings"]["baseline"]["median_ms"] / row["timings"]["current"]["median_ms"] for row in rows]
    print(f"{len(rows)} exact cases; baseline/current median ratios {min(ratios):.3f}–{max(ratios):.3f}; {args.output}")


if __name__ == "__main__":
    main()
