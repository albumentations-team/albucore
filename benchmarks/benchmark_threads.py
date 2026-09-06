"""Configure reproducible one-thread CPU benchmark processes."""

from __future__ import annotations

import argparse
import os
from typing import Any

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _thread_count_from_command_line() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", type=int, default=1)
    args, _ = parser.parse_known_args()
    if args.threads < 1:
        msg = "--threads must be positive."
        raise ValueError(msg)
    return args.threads


THREAD_COUNT = _thread_count_from_command_line()

for variable in THREAD_ENVIRONMENT_VARIABLES:
    os.environ[variable] = str(THREAD_COUNT)


def configure_libraries(torch: Any, cv2: Any, threads: int = THREAD_COUNT) -> str:
    """Set and report the numerical-library thread controls."""
    if threads != THREAD_COUNT:
        msg = f"Expected --threads={THREAD_COUNT} from process setup, got {threads}."
        raise ValueError(msg)

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    cv2.setNumThreads(0 if threads == 1 else threads)

    settings = (torch.get_num_threads(), torch.get_num_interop_threads(), cv2.getNumThreads())
    if settings != (threads, threads, threads):
        msg = f"Expected {threads} threads for Torch intra-op/inter-op and OpenCV, got {settings}."
        raise RuntimeError(msg)

    opencv_setting = "internal parallelism disabled" if threads == 1 else f"`{threads}`"
    return (
        f"Threads: Torch intra-op `{threads}`, inter-op `{threads}`, OpenCV {opencv_setting}; "
        f"OpenMP/BLAS environment `{threads}`."
    )
