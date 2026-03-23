# ruff: noqa: T201, INP001, B023, RUF001
"""Minimal **OpenCV ``cv2.LUT`` vs StringZilla ``translate``** (shared 256-byte **uint8** LUT).

LUT is a fixed-seed **permutation** of ``0..255`` (non-identity). Use this file as a **copy-paste
repro** for upstream issues; it has no ``albucore`` import.

The full shape sweep (HWC / DHWC / NDHWC, per-channel LUTs, SZ full-buffer vs loop) lives in
``benchmark_sz_lut_vs_cv2_lut.py``.

Run from anywhere with ``numpy``, ``opencv-python[-headless]``, ``stringzilla`` installed::

    python benchmarks/benchmark_cv2_lut_vs_sz_lut_minimal.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import stringzilla as sz


def _median_ms(fn: object, repeats: int = 15, warmup: int = 5) -> float:
    f = fn  # type: ignore[assignment]
    for _ in range(warmup):
        f()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _sz_shared_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(img, dtype=np.uint8)
    flat = a.reshape(-1)
    out = flat.copy()
    sz.translate(memoryview(out), memoryview(lut), inplace=True)
    return out.reshape(a.shape)


def main() -> None:
    rng = np.random.default_rng(0)
    # Non-trivial uint8 LUT: permutation of 0..255 (reproducible; not identity ``arange``).
    lut = np.random.default_rng(42).permutation(256).astype(np.uint8)
    shapes: list[tuple[int, ...]] = [
        (128, 128, 1),
        (256, 256, 3),
        (512, 512, 3),
        (640, 640, 3),
        (1024, 1024, 3),
        (64, 128, 128, 3),
        (128, 128, 128, 1),
        (2, 64, 128, 128, 3),
    ]

    print("Median ms (lower better). Shared LUT length 256, uint8 image.\n")
    print(f"OpenCV {cv2.__version__}, numpy {np.__version__}\n")
    print("| shape | pixels | cv2.LUT | SZ translate (1× ravel) | faster |")
    print("|-------|-------:|--------:|--------------------------:|--------|")

    for sh in shapes:
        img = rng.integers(0, 256, size=sh, dtype=np.uint8)
        npx = int(np.prod(sh))

        t_cv = _median_ms(lambda: cv2.LUT(np.ascontiguousarray(img), lut))
        t_sz = _median_ms(lambda: _sz_shared_lut(img, lut))
        faster = "cv2" if t_cv <= t_sz else "StringZilla"
        shape_str = "×".join(str(x) for x in sh)
        print(f"| {shape_str} | {npx} | {t_cv:.4f} | {t_sz:.4f} | {faster} |")


if __name__ == "__main__":
    main()
