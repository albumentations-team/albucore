#!/usr/bin/env python3
# ruff: noqa: T201, EXE001, B023, I001
"""Shared ``(256,)`` uint8 LUT: StringZilla (full-buffer ``translate``) vs ``cv2.LUT`` on **HWC**.

Checks ``opencv_shared_uint8_lut_faster_hwc`` against measured medians. Uses
``bench_wall_ms`` with several RNG seeds; LUT is a fixed-seed **permutation** of ``uint8``
(non-trivial remap). A few **borderline** ``512^2 x C=3/4`` cases may
still prefer the other backend by a few percent - tune ``1_310_000`` / ``409600`` only
after re-measuring on your target hardware.

Run from repo root::

    uv run python benchmarks/benchmark_lut_shared_routing.py
"""

from __future__ import annotations

import argparse
import platform

import cv2
import numpy as np
import stringzilla as sz

from albucore.lut import opencv_shared_uint8_lut_faster_hwc
from timing import bench_wall_ms


def sz_full(img: np.ndarray, lut: np.ndarray) -> None:
    a = np.ascontiguousarray(img, dtype=np.uint8)
    out = a.reshape(-1).copy()
    sz.translate(memoryview(out), memoryview(lut), inplace=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--warmup", type=int, default=7)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()

    lut = np.random.default_rng(42).permutation(256).astype(np.uint8)

    sides = [256, 384, 512, 640, 768, 896, 1024]
    c_max = 12

    print()
    print("### Shared LUT routing calibration (SZ full buffer vs `cv2.LUT`, HWC)")
    print()
    print(
        f"_Median ms +/- sample std, repeats={args.repeats}, warmup={args.warmup}, "
        f"{len(sides)} x {c_max} grid x {args.seeds} seeds; "
        f"{platform.system()} `{platform.machine()}`, OpenCV {cv2.__version__}.",
    )
    print()
    print(
        "| S | C | n_el | heuristic→cv2 | median_sz | median_cv | faster | match |",
    )
    print("|:-:|:-:|:-----:|:-------------:|----------:|----------:|:------:|:-----:|")

    mismatches = 0
    for s in sides:
        for c in range(1, c_max + 1):
            want_cv = opencv_shared_uint8_lut_faster_hwc((s, s, c))
            med_sz: list[float] = []
            med_cv: list[float] = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                img = np.ascontiguousarray(rng.integers(0, 256, size=(s, s, c), dtype=np.uint8))
                ts = bench_wall_ms(lambda: sz_full(img, lut), args.repeats, args.warmup)
                tc = bench_wall_ms(lambda: cv2.LUT(img, lut), args.repeats, args.warmup)
                med_sz.append(ts.median)
                med_cv.append(tc.median)
            msz = float(np.median(med_sz))
            mcv = float(np.median(med_cv))
            faster = "cv2" if mcv < msz else "sz"
            predict = "cv2" if want_cv else "sz"
            match = "ok" if faster == predict else "!!"
            if match == "!!":
                mismatches += 1
            n = s * s * c
            heur = "Y" if want_cv else "N"
            # full grid is long; print rows where heuristic disagrees with timing or C in {1,2,3,4,5,12}
            if match == "!!" or c <= 5 or c == c_max or s in (512, 1024):
                print(
                    f"| {s} | {c} | {n} | {heur} | {msz:.4f} | {mcv:.4f} | {faster} | {match} |",
                )

    print()
    print(f"_Heuristic mismatches (faster backend ≠ prediction): **{mismatches}**._")
    print(
        "\nTuning target: minimize mismatches without over-weighting one seed. "
        "If mismatches grow after an OpenCV/StringZilla upgrade, adjust "
        "`opencv_shared_uint8_lut_faster_hwc` in `albucore/lut.py`.\n",
    )


if __name__ == "__main__":
    main()
