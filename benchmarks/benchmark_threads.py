"""Configure reproducible one-thread CPU benchmark processes."""

from __future__ import annotations

import os
from typing import Any

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for variable in THREAD_ENVIRONMENT_VARIABLES:
    os.environ[variable] = "1"


def configure_libraries(torch: Any, cv2: Any) -> str:
    """Set and report the numerical-library thread controls."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)

    settings = (torch.get_num_threads(), torch.get_num_interop_threads(), cv2.getNumThreads())
    if settings != (1, 1, 1):
        msg = f"Expected one thread for Torch intra-op/inter-op and OpenCV, got {settings}."
        raise RuntimeError(msg)

    return "Threads: Torch intra-op `1`, inter-op `1`, OpenCV internal parallelism disabled; OpenMP/BLAS environment `1`."
