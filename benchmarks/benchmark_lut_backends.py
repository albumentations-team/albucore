#!/usr/bin/env python3
"""StringZilla ``translate`` vs ``cv2.LUT`` on uint8 data — shared and per-channel LUTs.

Layouts (channel last, Albucore convention):

- ``(H, W, C)``
- ``(D, H, W, C)``

**Shared LUT** (length 256): one table for every byte.

- **SZ full:** one allocating ``sz.translate`` call for the contiguous byte buffer.
- **SZ per-ch:** copy and translate one channel plane at a time into the output.
- **cv2:** one ``cv2.LUT(img, lut)`` on the full contiguous array (OpenCV accepts these ndims here).

**Per-channel LUT** (different table per channel):

- **SZ per-ch:** loop StringZilla translate per channel with ``lut_1d[c]``.
- **cv2 loop:** ``C`` times ``cv2.LUT(img[..., c], lut_1d[c])``.
- **cv2 flat:** ``(256, 1, C)`` one-shot on ``(H, W, C)``; for ``DHWC``, flatten
  the leading dimensions to a synthetic HWC view first.

This comparison covers uint8 images and tables only.

**LUTs:** fixed-seed **permutations** of ``0..255`` (full ``uint8`` tables, non-identity / non-trivial remap).
Same tables for SZ and OpenCV so timings reflect scatter-gather cost, not “no-op” identity.

Run from repo root::

    uv run python benchmarks/benchmark_lut_backends.py
"""

from __future__ import annotations

import argparse
import os
import platform

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import stringzilla as sz
import torch
from shape_grids import SZ_LUT_BENCHMARK_SHAPES
from timing import median_ms

from albucore import apply_uint8_lut
from albucore.lut import _cv2_lut_channel_last

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
    source = np.ascontiguousarray(img)
    raw = sz.translate(memoryview(source), memoryview(lut), inplace=False)
    return np.frombuffer(raw, dtype=np.uint8).reshape(img.shape)


def sz_lut_per_channel_loop(img: np.ndarray, lut_shared: np.ndarray) -> np.ndarray:
    out = np.empty_like(img, dtype=np.uint8)
    c = int(img.shape[-1])
    for i in range(c):
        plane = np.array(img[..., i], copy=True, order="C")
        sz.translate(memoryview(plane), memoryview(lut_shared), inplace=True)
        out[..., i] = plane
    return out


def sz_lut_per_channel_distinct(
    img: np.ndarray,
    luts_1d: list[np.ndarray],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    out = np.empty_like(img, dtype=np.uint8) if out is None else out
    c = int(img.shape[-1])
    for i in range(c):
        plane = np.array(img[..., i], copy=True, order="C")
        sz.translate(memoryview(plane), memoryview(luts_1d[i]), inplace=True)
        out[..., i] = plane
    return out


def cv2_lut_full(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(np.ascontiguousarray(img, dtype=np.uint8), lut)


def cv2_lut_per_channel_distinct(
    img: np.ndarray,
    luts_1d: list[np.ndarray],
) -> np.ndarray:
    """Apply one OpenCV call per channel, preserving the source rank."""
    a = np.ascontiguousarray(img, dtype=np.uint8)
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
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(args.seed)
    shapes = ((128, 160, 1), (128, 160, 3), (4, 128, 160, 3)) if args.quick else SZ_LUT_BENCHMARK_SHAPES

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
        "| layout | shape | pixels | SZ 1× full buffer | SZ loop C× `translate` | cv2 1× `LUT` | fastest (shared) |",
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
        layout = {3: "HWC", 4: "DHWC"}.get(len(sh), "?")
        shape_str = "×".join(str(x) for x in sh)
        print(
            f"| {layout} | {shape_str} | {npx} | {t_sz1:.4f} | {t_szc:.4f} | {t_cv2:.4f} | {tag} |",
        )

    print()
    print("#### Per-channel LUT (different `256` table per channel)")
    print()
    print(
        "| layout | shape | pixels | SZ loop C× `translate` | cv2 loop C× `LUT` | cv2 flat 1× `(256,1,C)` | fastest |",
    )
    print("|--------|-------|-------:|--------------------:|------------------:|--------------------------:|---------|")

    for sh in shapes:
        c = sh[-1]
        lut_cv2, luts_1d = _lut_per_channel(c)
        img = rng.integers(0, 256, size=sh, dtype=np.uint8)
        npx = int(np.prod(sh))

        t_szc = median_ms(lambda: sz_lut_per_channel_distinct(img, luts_1d), args.repeats, args.warmup)
        t_cv2 = median_ms(
            lambda: cv2_lut_per_channel_distinct(img, luts_1d),
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
        layout = {3: "HWC", 4: "DHWC"}.get(len(sh), "?")
        shape_str = "×".join(str(x) for x in sh)
        print(f"| {layout} | {shape_str} | {npx} | {t_szc:.4f} | {t_cv2:.4f} | {t_cv2_flat:.4f} | {tag} |")

    print()
    print("#### Supplied output (shared and per-channel tables)")
    print()
    print("| shape | public shared | shared CV | per-channel SZ | per-channel CV |")
    print("|---|---:|---:|---:|---:|")
    for shape in shapes:
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        output = np.empty_like(image)
        table, channel_tables = _lut_per_channel(shape[-1])
        calls = (
            lambda: apply_uint8_lut(image, lut_s, out=output),
            lambda: _cv2_lut_channel_last(image, lut_s, out=output),
            lambda: sz_lut_per_channel_distinct(image, channel_tables, out=output),
            lambda: _cv2_lut_channel_last(image, table, out=output),
        )
        expected_channels = np.stack([channel_tables[c][image[..., c]] for c in range(shape[-1])], axis=-1)
        for index, call in enumerate(calls):
            assert call() is output
            np.testing.assert_array_equal(output, lut_s[image] if index < 2 else expected_channels)
        times = [median_ms(call, args.repeats, args.warmup) for call in calls]
        print(f"| {'×'.join(map(str, shape))} | " + " | ".join(f"{value:.4f}" for value in times) + " |")

    print()
    print(
        "**Notes:**\n"
        "- **SZ full** is only valid when one LUT applies to every byte (scalar `_apply_lut` path).\n"
        "- **SZ loop** copies and translates one channel plane at a time.\n"
        "- **cv2** shared `(256,)`: contiguous `HWC` / `DHWC` work here. "
        "Per-channel distinct: direct `(256,1,C)` is **only** valid for `ndim==3`, but contiguous "
        "volumes/batches can be reshaped to HWC and use the same one-shot OpenCV path.\n"
        "- Regenerate on your CPU; routing should follow benchmarks, not assumptions.\n",
    )


if __name__ == "__main__":
    main()
