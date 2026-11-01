"""Compare the public distance router with the norm/dot route from before #197.

Write timings to CSV and environment metadata to JSON. Run from the repository root
with Python 3.11; see docs/performance-optimization.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import cv2
import numkong as nk
import numpy as np
import torch

import albucore
from timing import bench_wall_ms


def _source_path(path: str) -> str:
    """Identify the import root without publishing the developer's local paths."""
    source = Path(path).resolve()
    roots = (("<repository>", Path(__file__).resolve().parents[1]), ("<environment>", Path(sys.prefix).resolve()))
    for label, root in roots:
        if source.is_relative_to(root):
            return f"{label}/{source.relative_to(root).as_posix()}"
    return f"<external>/{source.name}"


def norm_dot_baseline(first: np.ndarray | torch.Tensor, second: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    tensor_output = isinstance(first, torch.Tensor)
    if tensor_output:
        first, second = first.numpy(), second.numpy()
    first = first.astype(np.float32, copy=False)
    second = second.astype(np.float32, copy=False)
    if len(first) * len(second) < 1000:
        result = np.asarray(
            nk.cdist(np.ascontiguousarray(first), np.ascontiguousarray(second), metric="sqeuclidean"), dtype=np.float32,
        )
    else:
        first_squared = (first**2).sum(axis=1, keepdims=True)
        second_squared = (second**2).sum(axis=1)[None, :]
        dot_product = albucore.matmul(first, second.T)
        result = first_squared + second_squared - 2 * dot_product
    np.maximum(result, 0.0, out=result)
    return torch.from_numpy(result) if tensor_output else result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True, help="Metadata file; timings use the same stem with .csv")
    parser.add_argument("--repeats", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    capabilities = nk.get_capabilities()
    if sys.platform == "darwin" and platform.machine() == "arm64" and not capabilities["neon"]:
        raise RuntimeError("Native ARM64 timing requires NumKong NEON; run outside the restricted sandbox.")

    rng = np.random.default_rng(197)
    rows = []
    for n, m, dimensions in (
        (10, 10, 2), (31, 31, 2), (32, 32, 2), (25, 40, 2),
        (1, 999, 2), (1, 1000, 2), (1, 1001, 2), (100, 100, 2),
        (1000, 100, 2), (4096, 5, 2), (20480, 16, 2), (307200, 16, 2),
        (100, 100, 1), (100, 100, 3), (100, 100, 5), (100, 100, 8), (100, 100, 9), (100, 100, 64),
    ):
        first_values = rng.random((n, dimensions), dtype=np.float32).astype(np.float16)
        second_values = rng.random((m, dimensions), dtype=np.float32).astype(np.float16)
        for dtype in (np.uint8, np.float16, np.float32):
            first_values_dtype = (first_values * 255).astype(dtype) if dtype == np.uint8 else first_values.astype(dtype)
            second_values_dtype = (second_values * 255).astype(dtype) if dtype == np.uint8 else second_values.astype(dtype)
            for strided in (False, True):
                first = np.empty((n, dimensions * (2 if strided else 1)), dtype=dtype)[:, ::2 if strided else 1]
                second = np.empty((m, dimensions * (2 if strided else 1)), dtype=dtype)[:, ::2 if strided else 1]
                first[:] = first_values_dtype
                second[:] = second_values_dtype
                for tensor_output in (False, True):
                    a = torch.from_numpy(first) if tensor_output else first
                    b = torch.from_numpy(second) if tensor_output else second
                    result = albucore.pairwise_distances_squared(a, b)
                    result_array = result.numpy() if tensor_output else result
                    differences = first[:3].astype(np.float64)[:, None, :] - second[:5].astype(np.float64)[None, :, :]
                    expected = np.sum(differences**2, axis=2).astype(np.float32)
                    np.testing.assert_allclose(result_array[:3, :5], expected, rtol=2e-6, atol=0)
                    del result, result_array
                    baseline = bench_wall_ms(lambda: norm_dot_baseline(a, b), args.repeats, args.warmup)
                    current = bench_wall_ms(lambda: albucore.pairwise_distances_squared(a, b), args.repeats, args.warmup)
                    np.testing.assert_array_equal(first, first_values_dtype)
                    np.testing.assert_array_equal(second, second_values_dtype)
                    row = {
                        "n": n, "m": m, "dimensions": dimensions, "dtype": np.dtype(dtype).name,
                        "container": "tensor" if tensor_output else "numpy", "strided": strided,
                        "baseline_ms": baseline.median, "current_ms": current.median,
                        "baseline_mad_ms": baseline.mad, "current_mad_ms": current.mad,
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    timings_path = args.output_json.with_suffix(".csv")
    with timings_path.open("w", newline="") as timings_file:
        writer = csv.DictWriter(timings_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(), "source": _source_path(albucore.__file__),
        "imports": {module.__name__: _source_path(module.__file__) for module in (albucore, np, nk, cv2, torch)},
        "measured_at": datetime.now(UTC).isoformat(),
        "baseline_commit": "fa126da86ea9b4a660c199549dd791a5f9584811",
        "packages": {item.metadata["Name"]: item.version for item in metadata.distributions()},
        "numkong_neon": capabilities["neon"],
        "opencv_threads": cv2.getNumThreads(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "repeats": args.repeats, "warmup": args.warmup, "timings": timings_path.name,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
