#!/usr/bin/env python3
"""
Grayscale / small-C sanity: where **LUT vs OpenCV** and **NumPy vs cv2.multiply** win.

- **uint8 ``multiply_by_vector``**: production uses **LUT**; OpenCV (float multiply + clip) is correct but slower here.
- **``from_float`` (float32 → uint8, C≤4)**: production uses **NumPy** ``rint(img * max_value)``; ``cv2.multiply`` on
  **(H, W, 1)** float tensors does **not** match NumPy elementwise multiply (OpenCV quirk); squeezed 2D ``cv2`` still
  loses to NumPy on tested sizes.

Run from repo root::

    uv run python benchmarks/benchmark_grayscale_paths.py
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from albucore.arithmetic import multiply_lut, multiply_opencv
from albucore.utils import clip, get_max_value
from timing import bench_wall_ms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    mv = float(get_max_value(np.uint8))

    print("## 1) uint8 per-channel multiply: `multiply_lut` vs `multiply_opencv` + clip\n")
    print("| shape | LUT ms | OpenCV ms | faster |")
    print("|-------|-------:|-----------:|--------|")
    for (h, w), c in [((256, 256), 1), ((512, 512), 1), ((1024, 1024), 1), ((256, 256), 3)]:
        img = rng.integers(1, 256, (h, w, c), dtype=np.uint8)
        vec = rng.random(c, dtype=np.float32) * 0.5 + 0.5

        def run_lut() -> None:
            multiply_lut(np.ascontiguousarray(img.copy()), vec)

        def run_cv() -> None:
            clip(multiply_opencv(np.ascontiguousarray(img.copy()), vec), np.uint8)

        t_l = bench_wall_ms(run_lut, repeats=args.repeats, warmup=args.warmup).median
        t_c = bench_wall_ms(run_cv, repeats=args.repeats, warmup=args.warmup).median
        faster = "LUT" if t_l <= t_c else "OpenCV"
        print(f"| ({h},{w},{c}) | {t_l:.4f} | {t_c:.4f} | {faster} |")

    print("\n## 2) float32 → uint8: NumPy vs `cv2.multiply` (grayscale)\n")
    print("| shape | NumPy `rint(img*255)` ms | cv2 on squeezed (H,W) ms | faster | max |f-np| on product |")
    print("|-------|-------------------------:|-------------------------:|--------|--------------:|")
    for h, w in [(256, 256), (512, 512), (1024, 1024)]:
        img = np.ascontiguousarray(rng.random((h, w, 1), dtype=np.float32))
        g = np.squeeze(img, axis=-1)

        def run_np() -> None:
            clip(np.rint(img * mv), np.uint8)

        def run_cv2() -> None:
            clip(np.rint(cv2.multiply(g, mv)), np.uint8)

        t_n = bench_wall_ms(run_np, repeats=args.repeats, warmup=args.warmup).median
        t_v = bench_wall_ms(run_cv2, repeats=args.repeats, warmup=args.warmup).median
        faster = "NumPy" if t_n <= t_v else "cv2"
        bad = np.max(np.abs(img * mv - cv2.multiply(img, mv)))
        print(f"| ({h},{w},1) | {t_n:.4f} | {t_v:.4f} | {faster} | {bad:.2e} |")

    print(
        "\n**Note:** `cv2.multiply(img_3d, 255)` on (H,W,1) float **differs** from `img * 255` (see max |f-np| column); "
        "do not use raw 3D cv2 multiply for float→uint8 scaling on grayscale.",
    )


if __name__ == "__main__":
    main()
