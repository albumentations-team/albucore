"""Measure route-specific peak RSS in fresh CPU processes."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path

import benchmark_threads

OPERATIONS = ("warp_affine3d", "remap3d")
CANDIDATES = ("single_loop", "channel_folded", "native_n_batch_sampler", "public_batch_dispatch", "tensor_numpy_bridge")
CHANNELS = (1, 3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=OPERATIONS)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--channels", type=int, choices=CHANNELS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _peak_rss_mib() -> float:
    peak_rss_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_rss_units / (1024 * 1024) if sys.platform == "darwin" else peak_rss_units / 1024


def _worker(operation: str, candidate: str, channels: int) -> None:
    import cv2
    import numpy as np
    import torch

    import albucore
    from benchmarks import benchmark_remap3d_batch as remap_benchmark
    from benchmarks import benchmark_warp_affine3d_batch as affine_benchmark

    benchmark_threads.configure_libraries(torch, cv2, 1)
    batch_size = 16
    shape = (16, 128, 160, channels)
    raw_volumes = np.random.default_rng(20260929).random((batch_size, *shape), dtype=np.float32)
    volumes = torch.from_numpy(raw_volumes).permute(0, 4, 1, 2, 3).contiguous()
    del raw_volumes
    fill = tuple(float(17 + 3 * channel) for channel in range(channels))

    if operation == "warp_affine3d":
        matrix = affine_benchmark._matrix()
        size = affine_benchmark._target(shape)
        routes = {
            "single_loop": lambda: affine_benchmark._single_loop(
                volumes,
                matrix,
                size,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "channel_folded": lambda: affine_benchmark._folded(
                volumes,
                matrix,
                size,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "native_n_batch_sampler": lambda: affine_benchmark._native_n_batch_sampler(
                volumes,
                matrix,
                size,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "public_batch_dispatch": lambda: albucore.warp_affine3d(
                volumes,
                matrix,
                size,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "tensor_numpy_bridge": lambda: affine_benchmark._tensor_numpy_bridge(
                volumes,
                matrix,
                size,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
        }
    else:
        grid = remap_benchmark._grid(np.random.default_rng(20260930), (8, 96, 120))
        routes = {
            "single_loop": lambda: remap_benchmark._single_loop(
                volumes,
                grid,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "channel_folded": lambda: remap_benchmark._folded(
                volumes,
                grid,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "native_n_batch_sampler": lambda: remap_benchmark._native_batch_sampler(
                volumes,
                grid,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "public_batch_dispatch": lambda: albucore.remap3d(
                volumes,
                grid,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
            "tensor_numpy_bridge": lambda: remap_benchmark._tensor_numpy_bridge(
                volumes,
                grid,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                fill,
            ),
        }

    result = routes[candidate]()
    del result
    print(
        json.dumps(
            {
                "peak_rss_mib": _peak_rss_mib(),
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "torch_version": torch.__version__,
            },
        ),
    )


def _measure(operation: str, candidate: str, channels: int) -> dict[str, str | float]:
    completed = subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--operation",
            operation,
            "--candidate",
            candidate,
            "--channels",
            str(channels),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _report() -> str:
    rows = []
    versions: tuple[str, str, str] | None = None
    for operation in OPERATIONS:
        for channels in CHANNELS:
            for candidate in CANDIDATES:
                measured = _measure(operation, candidate, channels)
                if versions is None:
                    versions = (
                        str(measured["numpy_version"]),
                        str(measured["opencv_version"]),
                        str(measured["torch_version"]),
                    )
                peak = float(measured["peak_rss_mib"])
                rows.append(f"| `{operation}` | {channels} | `{candidate}` | {peak:.1f} |")
    numpy_version, opencv_version, torch_version = versions or ("unknown", "unknown", "unknown")
    return "\n".join(
        (
            "# Batched 3D sampling route memory",
            "",
            f"Host: `{platform.platform()}`; machine: `{platform.machine()}`; Python `{platform.python_version()}`.",
            f"NumPy `{numpy_version}`; OpenCV `{opencv_version}`; PyTorch `{torch_version}`.",
            "",
            "Each row is measured in a fresh process with one CPU thread, using a contiguous CPU Torch float32 batch "
            "with `N=16`, input `DHW=(16,128,160)`, output `DHW=(8,96,120)`, trilinear interpolation, and nonzero "
            "constant fill. The value is the process high-water RSS in MiB, including the interpreter, imported "
            "libraries, input, output, and route temporaries. It is not an allocation delta.",
            "",
            "The parent runs each operation/channel/candidate as a separate child process:",
            "`uv run --locked python benchmarks/benchmark_sampling3d_batch_memory.py`.",
            "",
            "| Operation | C | Candidate | Peak process RSS (MiB) |",
            "|---|---:|---|---:|",
            *rows,
            "",
        ),
    )


def main() -> None:
    args = _parse_args()
    if args.worker:
        if args.operation is None or args.candidate is None or args.channels is None:
            raise ValueError("Worker mode requires operation, candidate, and channels.")
        _worker(args.operation, args.candidate, args.channels)
        return

    report = _report()
    if args.output is None:
        print(report)  # noqa: T201 - command-line benchmark output
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}")  # noqa: T201 - command-line benchmark output


if __name__ == "__main__":
    main()
