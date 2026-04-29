#!/usr/bin/env python3
"""
``sz_lut`` (StringZilla ``translate``) vs ``cv2.LUT`` on uint8 data — shared and per-channel LUTs.

Layouts (channel last, Albucore convention):

- ``(H, W, C)``
- ``(D, H, W, C)``
- ``(N, D, H, W, C)``

**Shared LUT** (length 256): one table for every byte.

- **SZ full:** single ``sz.translate`` on ``img.ravel()`` (same semantics as scalar ``apply_lut`` on a contiguous buffer).
- **SZ per-ch:** loop ``sz_lut`` on a **copy** of each channel plane (default in-place translate on that buffer).
- **cv2:** one ``cv2.LUT(img, lut)`` on the full contiguous array (OpenCV accepts these ndims here).

**Per-channel LUT** (different table per channel):

- **SZ per-ch:** loop ``sz_lut`` per channel with ``lut_1d[c]``.
- **cv2 loop:** ``C`` times ``cv2.LUT(img[..., c], lut_1d[c])``.
- **cv2 flat:** ``(256, 1, C)`` one-shot on ``(H, W, C)``; for ``DHWC`` / ``NDHWC``, flatten
  the leading dimensions to a synthetic HWC view first.

Non-uint8 / non-standard cases are not timed; production still routes those through OpenCV.

**LUTs:** fixed-seed **permutations** of ``0..255`` (full ``uint8`` tables, non-identity / non-trivial remap).
Same tables for SZ and OpenCV so timings reflect scatter-gather cost, not “no-op” identity.

Run from repo root::

    uv run python benchmarks/benchmark_sz_lut_vs_cv2_lut.py
"""

from __future__ import annotations

import argparse
import platform

import cv2
import numpy as np
import stringzilla as sz

from albucore.lut import sz_lut
from timing import median_ms

# Reproducible non-trivial uint8 LUTs (not ``arange`` identity).
_LUT_PERM_SEED = 42


def _lut_shared() -> np.ndarray:
    return np.random.default_rng(_LUT_PERM_SEED).permutation(256).astype(np.uint8)


def _lut_per_channel(c: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(256, 1, C)`` for cv2; one independent length-256 permutation per channel for SZ."""
    lut_cv2 = np.zeros((256, 1, c), dtype=np.uint8)
    luts_1d: list[np.ndarray] = []
    for i in range(c):
        col = np.random.default_rng(_LUT_PERM_SEED + 1000 + i).permutation(256).astype(np.uint8)
        lut_cv2[:, 0, i] = col
        luts_1d.append(col.copy())
    return lut_cv2, luts_1d


def sz_lut_full_buffer(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """One StringZilla pass over all bytes (shared LUT only)."""
    a = np.ascontiguousarray(img, dtype=np.uint8)
    flat = a.reshape(-1)
    out = flat.copy()
    sz.translate(memoryview(out), memoryview(lut), inplace=True)
    return out.reshape(a.shape)


def sz_lut_per_channel_loop(img: np.ndarray, lut_shared: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.uint8)
    c = int(img.shape[-1])
    for i in range(c):
        ch = np.ascontiguousarray(img[..., i], dtype=np.uint8).copy()
        out[..., i] = sz_lut(ch, lut_shared)
    return out


def sz_lut_per_channel_distinct(img: np.ndarray, luts_1d: list[np.ndarray]) -> np.ndarray:
    out = np.empty_like(img, dtype=np.uint8)
    c = int(img.shape[-1])
    for i in range(c):
        ch = np.ascontiguousarray(img[..., i], dtype=np.uint8).copy()
        out[..., i] = sz_lut(ch, luts_1d[i])
    return out


def cv2_lut_full(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(np.ascontiguousarray(img, dtype=np.uint8), lut)


def cv2_lut_per_channel_distinct(
    img: np.ndarray,
    lut_256_1_c: np.ndarray,
    luts_1d: list[np.ndarray],
) -> np.ndarray:
    """OpenCV accepts ``(256, 1, C)`` only for **2-D** multi-channel images ``(H, W, C)``.

    For ``(D, H, W, C)`` / ``(N, D, H, W, C)``, ``lut.cpp`` rejects multi-column LUTs; mirror
    production by applying ``cv2.LUT`` per channel on ``img[..., c]`` with a length-256 table.
    """
    a = np.ascontiguousarray(img, dtype=np.uint8)
    if a.ndim == 3 and a.shape[-1] > 1:
        return cv2.LUT(a, lut_256_1_c)
    out = np.empty_like(a)
    c = int(a.shape[-1])
    for i in range(c):
        out[..., i] = cv2.LUT(a[..., i], luts_1d[i])
    return out


def cv2_lut_per_channel_flattened(img: np.ndarray, lut_256_1_c: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(img, dtype=np.uint8)
    c = int(a.shape[-1])
    if a.ndim == 3:
        return cv2.LUT(a, lut_256_1_c)
    flat = a.reshape(-1, a.shape[-2], c)
    return cv2.LUT(flat, lut_256_1_c).reshape(a.shape)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    shapes: list[tuple[int, ...]] = [
        (128, 128, 1),
        (128, 128, 3),
        (256, 256, 1),
        (256, 256, 3),
        (256, 256, 9),
        (512, 512, 1),
        (512, 512, 3),
        (512, 512, 9),
        (1024, 1024, 1),
        (1024, 1024, 3),
        (1024, 1024, 9),
        (96, 96, 9),
        # Volumes (D,H,W,C): deeper stacks + in-plane ≥128 common for 3D segmentation training patches
        # (nnU-Net / similar often use ~128–256 in-plane and tens of slices; exact patch is dataset-specific).
        (32, 128, 128, 1),
        (64, 128, 128, 3),
        (128, 128, 128, 1),
        (48, 256, 256, 3),
        (96, 160, 160, 3),
        (6, 32, 32, 9),
        # Batch of volumes (N,D,H,W,C)
        (2, 32, 128, 128, 3),
        (2, 64, 128, 128, 3),
        (1, 128, 128, 128, 1),
    ]

    lut_s = _lut_shared()

    print()
    print("### Benchmark: `sz.translate` vs `cv2.LUT` (uint8 image, uint8 LUT)")
    print()
    print(
        f"_Median ms, repeats={args.repeats}, warmup={args.warmup}, seed={args.seed}; "
        f"{platform.system()} `{platform.machine()}`, "
        f"OpenCV {cv2.__version__}, numpy {np.__version__}.",
    )
    print()

    print("#### Shared LUT `(256,)` — same table for every byte")
    print()
    print(
        "| layout | shape | pixels | SZ 1× full buffer | SZ loop C× `sz_lut` | cv2 1× `LUT` | fastest (shared) |",
    )
    print("|--------|-------|-------:|--------------------:|--------------------:|-------------:|------------------|")

    for sh in shapes:
        img = rng.integers(0, 256, size=sh, dtype=np.uint8)
        npx = int(np.prod(sh))

        t_sz1 = median_ms(lambda: sz_lut_full_buffer(img, lut_s), args.repeats, args.warmup)
        t_szc = median_ms(lambda: sz_lut_per_channel_loop(img, lut_s), args.repeats, args.warmup)
        t_cv2 = median_ms(lambda: cv2_lut_full(img, lut_s), args.repeats, args.warmup)
        best = min(t_sz1, t_szc, t_cv2)
        if best == t_sz1:
            tag = "SZ full"
        elif best == t_szc:
            tag = "SZ loop"
        else:
            tag = "cv2"
        layout = {3: "HWC", 4: "DHWC", 5: "NDHWC"}.get(len(sh), "?")
        shape_str = "×".join(str(x) for x in sh)
        print(
            f"| {layout} | {shape_str} | {npx} | {t_sz1:.4f} | {t_szc:.4f} | {t_cv2:.4f} | {tag} |",
        )

    print()
    print("#### Per-channel LUT (different `256` table per channel)")
    print()
    print(
        "| layout | shape | pixels | SZ loop C× `sz_lut` | cv2 loop C× `LUT` | cv2 flat 1× `(256,1,C)` | fastest |",
    )
    print("|--------|-------|-------:|--------------------:|------------------:|--------------------------:|---------|")

    for sh in shapes:
        c = sh[-1]
        lut_cv2, luts_1d = _lut_per_channel(c)
        img = rng.integers(0, 256, size=sh, dtype=np.uint8)
        npx = int(np.prod(sh))

        t_szc = median_ms(lambda: sz_lut_per_channel_distinct(img, luts_1d), args.repeats, args.warmup)
        t_cv2 = median_ms(
            lambda: cv2_lut_per_channel_distinct(img, lut_cv2, luts_1d),
            args.repeats,
            args.warmup,
        )
        t_cv2_flat = median_ms(lambda: cv2_lut_per_channel_flattened(img, lut_cv2), args.repeats, args.warmup)
        best = min(t_szc, t_cv2, t_cv2_flat)
        if best == t_szc:
            tag = "SZ loop"
        elif best == t_cv2:
            tag = "cv2 loop"
        else:
            tag = "cv2 flat"
        layout = {3: "HWC", 4: "DHWC", 5: "NDHWC"}.get(len(sh), "?")
        shape_str = "×".join(str(x) for x in sh)
        print(f"| {layout} | {shape_str} | {npx} | {t_szc:.4f} | {t_cv2:.4f} | {t_cv2_flat:.4f} | {tag} |")

    print()
    print(
        "**Notes:**\n"
        "- **SZ full** is only valid when one LUT applies to every byte (scalar `apply_lut` path).\n"
        "- **SZ loop** matches the non-contiguous multi-channel `apply_lut` fallback (`sz_lut` per channel).\n"
        "- **cv2** shared `(256,)`: contiguous `HWC` / `DHWC` / `NDHWC` work here. "
        "Per-channel distinct: direct `(256,1,C)` is **only** valid for `ndim==3`, but contiguous "
        "volumes/batches can be reshaped to HWC and use the same one-shot OpenCV path.\n"
        "- Regenerate on your CPU; routing should follow benchmarks, not assumptions.\n",
    )


if __name__ == "__main__":
    main()
