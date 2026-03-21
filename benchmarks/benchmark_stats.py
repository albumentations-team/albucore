"""Quick timing sweep for albucore.stats vs plain NumPy (axis / dtype)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from albucore.stats import mean_std


def _timeit(fn: Callable[[], Any], *, repeats: int = 7, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def _bench_pair(label: str, arr: np.ndarray) -> None:
    t_albu = _timeit(lambda a=arr: mean_std(a, "global"))
    t_np = _timeit(
        lambda a=arr: (
            float(np.mean(a, dtype=np.float64)),
            float(np.std(a, dtype=np.float64) + 1e-4),
        ),
    )
    ratio = t_np / t_albu if t_albu > 0 else 0.0
    line = (
        f"{label:16}  mean_std(albucore)={t_albu * 1e6:8.1f}µs"
        f"  numpy ref={t_np * 1e6:8.1f}µs  ratio={ratio:.2f}x"
    )
    print(line)  # noqa: T201


def main() -> None:
    rng = np.random.default_rng(0)
    cases: list[tuple[str, np.ndarray]] = [
        ("HWC uint8", rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)),
        ("HWC float32", rng.random((512, 512, 3), dtype=np.float32)),
        ("NHWC uint8", rng.integers(0, 256, size=(8, 256, 256, 3), dtype=np.uint8)),
    ]
    for label, arr in cases:
        _bench_pair(label, arr)


if __name__ == "__main__":
    main()
