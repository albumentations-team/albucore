#!/usr/bin/env python3
"""Benchmark multiply/add: albucore vs NumKong ``scale`` (a*x+b) and ``fma``.

Run: ``uv run python benchmarks/benchmark_multiply_add_numkong.py`` (after ``uv sync --extra headless``).

Production paths use OpenCV / NumPy / LUT; ``*_numkong`` helpers use ``blend`` or ``nk.scale`` as in the printed tables.
"""

from __future__ import annotations

import argparse

import numkong as nk
import numpy as np

from albucore.functions import (
    add_array,
    add_array_numkong,
    add_constant,
    add_vector,
    multiply_by_array,
    multiply_by_constant,
    multiply_by_vector,
    multiply_numpy,
)
from albucore.weighted import multiply_by_constant_numkong
from albucore.utils import clip, get_num_channels
from timing import median_ms


def nk_scale_flat(img: np.ndarray, *, alpha: float, beta: float) -> np.ndarray:
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(flat, alpha=alpha, beta=beta)
    raw = np.frombuffer(out, dtype=img.dtype).reshape(img.shape)
    if img.dtype == np.float32:
        return clip(raw, np.float32)
    return clip(raw, np.uint8)


def nk_scale_inplace(img: np.ndarray, *, alpha: float, beta: float) -> np.ndarray:
    """In-place nk.scale: writes result back into img's buffer (requires C-contiguous input)."""
    flat = img.reshape(-1)
    nk.scale(flat, alpha=alpha, beta=beta, out=flat)
    raw = flat.reshape(img.shape)
    if img.dtype == np.float32:
        return clip(raw, np.float32)
    return clip(raw, np.uint8)


def nk_fma_mul_flat(img: np.ndarray, other: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(img.reshape(-1))
    b = np.ascontiguousarray(other.reshape(-1))
    z = np.zeros_like(a)
    out = nk.fma(nk.Tensor(a), nk.Tensor(b), nk.Tensor(z), alpha=1.0, beta=0.0)
    raw = np.frombuffer(out, dtype=np.float32).reshape(img.shape)
    return clip(raw, np.float32)


def nk_channelwise_scale(img: np.ndarray, per_ch: np.ndarray, *, alpha_per_ch: bool) -> np.ndarray:
    """If alpha_per_ch: multiply channel i by per_ch[i]. Else: add per_ch[i] (alpha=1, beta=per_ch[i])."""
    out = np.empty_like(img)
    c = get_num_channels(img)
    for i in range(c):
        flat = np.ascontiguousarray(img[..., i].reshape(-1))
        if alpha_per_ch:
            t = nk.scale(flat, alpha=float(per_ch[i]), beta=0.0)
        else:
            t = nk.scale(flat, alpha=1.0, beta=float(per_ch[i]))
        out[..., i] = np.frombuffer(t, dtype=img.dtype).reshape(img.shape[:-1])
    if img.dtype == np.float32:
        return clip(out, np.float32)
    return clip(out, np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    channels = [1, 3, 9]
    s_mul = 1.127
    s_add = 0.07

    print("# multiply / add: albucore vs NumKong `scale` / `fma` / `blend` helpers")
    print()
    print("Median ms; **prod** = current `@clipped` public API. **NK scale** = `α·x+β` on ravel.")
    print()

    # --- scalar multiply ---
    print("## `multiply_by_constant` — scalar")
    print()
    print("| dtype | H×W | C | prod | NK scale (alloc) | NK scale (inplace) | fastest |")
    print("|-------|-----|---|-----:|-----------------:|-------------------:|---------|")
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                if dtype == np.uint8:
                    img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
                else:
                    img = rng.random((h, w, c), dtype=np.float32)

                t_prod = median_ms(lambda: multiply_by_constant(img, s_mul), args.repeats, args.warmup)
                t_nk = median_ms(lambda: multiply_by_constant_numkong(img, s_mul), args.repeats, args.warmup)
                t_nk_ip = median_ms(lambda: nk_scale_inplace(img, alpha=s_mul, beta=0.0), args.repeats, args.warmup)
                opts = [("prod", t_prod), ("NK_alloc", t_nk), ("NK_inplace", t_nk_ip)]
                best = min(opts, key=lambda x: x[1])[0]
                dname = "uint8" if dtype == np.uint8 else "float32"
                print(f"| {dname} | {h}×{w} | {c} | {t_prod:.4f} | {t_nk:.4f} | {t_nk_ip:.4f} | {best} |")
    print()

    # --- scalar add ---
    print("## `add_constant` — scalar")
    print()
    print("| dtype | H×W | C | prod | NK scale (alloc) | NK scale (inplace) | fastest |")
    print("|-------|-----|---|-----:|-----------------:|-------------------:|---------|")
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                if dtype == np.uint8:
                    img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
                    add_v = 3
                else:
                    img = rng.random((h, w, c), dtype=np.float32)
                    add_v = s_add

                t_prod = median_ms(lambda: add_constant(img, add_v), args.repeats, args.warmup)
                t_sc = median_ms(lambda: nk_scale_flat(img, alpha=1.0, beta=float(add_v)), args.repeats, args.warmup)
                t_sc_ip = median_ms(lambda: nk_scale_inplace(img, alpha=1.0, beta=float(add_v)), args.repeats, args.warmup)
                opts = [("prod", t_prod), ("NK_alloc", t_sc), ("NK_inplace", t_sc_ip)]
                best = min(opts, key=lambda x: x[1])[0]
                dname = "uint8" if dtype == np.uint8 else "float32"
                print(f"| {dname} | {h}×{w} | {c} | {t_prod:.4f} | {t_sc:.4f} | {t_sc_ip:.4f} | {best} |")
    print()

    # --- full-array elementwise multiply ---
    print("## `multiply_by_array` — elementwise `img * value`")
    print()
    print("| dtype | H×W | C | prod (OpenCV) | NK `fma` (a·b+0·z) | NumPy (`multiply_numpy` path) | fastest |")
    print("|-------|-----|---|--------------:|-------------------:|------------------------------:|---------|")
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                if dtype == np.uint8:
                    img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
                    val = rng.random((h, w, c), dtype=np.float32) * 0.01 + 0.99
                else:
                    img = rng.random((h, w, c), dtype=np.float32)
                    val = rng.random((h, w, c), dtype=np.float32) * 0.5 + 0.5

                t_prod = median_ms(lambda: multiply_by_array(img, val), args.repeats, args.warmup)
                if dtype == np.float32:
                    t_fma = median_ms(lambda: nk_fma_mul_flat(img, val), args.repeats, args.warmup)
                else:
                    t_fma = float("nan")
                t_np = median_ms(lambda: multiply_numpy(img, val), args.repeats, args.warmup)
                if dtype == np.float32:
                    opts = [("prod", t_prod), ("NK_fma", t_fma), ("numpy", t_np)]
                else:
                    opts = [("prod", t_prod), ("numpy", t_np)]
                best = min(opts, key=lambda x: x[1])[0]
                dname = "uint8" if dtype == np.uint8 else "float32"
                fma_s = f"{t_fma:.4f}" if dtype == np.float32 else "N/A"
                print(f"| {dname} | {h}×{w} | {c} | {t_prod:.4f} | {fma_s} | {t_np:.4f} | {best} |")
    print()

    # --- full-array elementwise add ---
    print("## `add_array` — elementwise `img + value`")
    print()
    print("| dtype | H×W | C | prod (OpenCV) | NK `add_array_numkong` (`blend`) | fastest |")
    print("|-------|-----|---|--------------:|---------------------------------:|---------|")
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                if dtype == np.uint8:
                    img = rng.integers(0, 200, size=(h, w, c), dtype=np.uint8)
                    val = rng.integers(0, 20, size=(h, w, c), dtype=np.uint8)
                else:
                    img = rng.random((h, w, c), dtype=np.float32)
                    val = rng.random((h, w, c), dtype=np.float32) * 0.1 - 0.05

                t_prod = median_ms(lambda: add_array(img, val), args.repeats, args.warmup)
                t_nk = median_ms(lambda: add_array_numkong(img, val), args.repeats, args.warmup)
                best = "prod" if t_prod <= t_nk else "NK_blend"
                dname = "uint8" if dtype == np.uint8 else "float32"
                print(f"| {dname} | {h}×{w} | {c} | {t_prod:.4f} | {t_nk:.4f} | {best} |")
    print()

    # --- per-channel vector (length C) ---
    print("## `multiply_by_vector` / `add_vector` — per-channel constants (broadcast)")
    print()
    print(
        "| dtype | op | H×W | C | prod | NK channel-wise `scale` | fastest |\n"
        "|-------|----|-----|---|-----:|------------------------:|---------|",
    )
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                if dtype == np.uint8:
                    img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
                    vm = (rng.random(c, dtype=np.float32) * 0.2 + 0.9).astype(np.float32)
                    va = rng.integers(0, 5, size=(c,), dtype=np.int32).astype(np.float32)
                else:
                    img = rng.random((h, w, c), dtype=np.float32)
                    vm = rng.random(c, dtype=np.float32) * 0.5 + 0.5
                    va = rng.random(c, dtype=np.float32) * 0.06 - 0.03

                t_pm = median_ms(
                    lambda: multiply_by_vector(img, vm),
                    args.repeats,
                    args.warmup,
                )
                t_pm_nk = median_ms(
                    lambda: nk_channelwise_scale(img, vm, alpha_per_ch=True),
                    args.repeats,
                    args.warmup,
                )
                w_m = "prod" if t_pm <= t_pm_nk else "NK_loop"
                dname = "uint8" if dtype == np.uint8 else "float32"
                print(f"| {dname} | mul_vec | {h}×{w} | {c} | {t_pm:.4f} | {t_pm_nk:.4f} | {w_m} |")

                t_pa = median_ms(lambda: add_vector(img, va), args.repeats, args.warmup)
                t_pa_nk = median_ms(
                    lambda: nk_channelwise_scale(img, va, alpha_per_ch=False),
                    args.repeats,
                    args.warmup,
                )
                w_a = "prod" if t_pa <= t_pa_nk else "NK_loop"
                print(f"| {dname} | add_vec | {h}×{w} | {c} | {t_pa:.4f} | {t_pa_nk:.4f} | {w_a} |")
    print()

    print("## Readout (this machine)")
    print()
    print(
        "- **Scalar affine** (`multiply_by_constant`, `add_constant`): three paths timed — prod, NK scale (alloc), NK scale (inplace).\n"
        "  `inplace` uses `out=buf` (raw numpy array, not Tensor wrapper) per NumKong #326 fix.\n"
        "- **Elementwise multiply** full array: **NK fma** only timed for **float32** (uint8 prod promotes to f32).\n"
        "- **Elementwise add** full array: **NK** is existing **`add_array_numkong`** (`blend`).\n"
        "- **Per-channel** vector: **NK** is **C separate `scale` calls** — usually loses to one OpenCV/LUT pass.",
    )


if __name__ == "__main__":
    main()
