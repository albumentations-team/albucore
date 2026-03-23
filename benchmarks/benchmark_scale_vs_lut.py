#!/usr/bin/env python3
"""``nk.scale`` vs ``sz_lut`` (SZ full-buffer) vs ``cv2.LUT`` on uint8 images.

Both LUT paths implement the same per-byte affine remap (shared table, one lookup per byte).
``nk.scale`` implements the same operation arithmetically (alpha*x + beta, saturated).
This benchmark answers: *for a scalar multiply-by-constant on a uint8 image, which backend
is cheapest?*

Implementations compared
------------------------
- **nk.scale**  — ``nk.scale(nk.Tensor(flat), alpha=alpha, beta=0.0)`` on the ravelled buffer;
  output frombuffer'd back.  Matches production ``multiply_by_constant_numkong``.
- **sz full**   — single ``sz.translate`` over the ravelled buffer with a precomputed LUT.
  Matches production ``sz_lut_full_buffer`` / ``apply_uint8_lut`` (shared-LUT path).
- **cv2.LUT**   — ``cv2.LUT(img, lut)`` on the contiguous HWC/DHWC/NDHWC array.
  OpenCV accepts arbitrary-ndim uint8 arrays for a 1-D shared table.

Shapes (channel-last, Albucore convention)
------------------------------------------
HWC  (H, W, C)         — 2-D spatial images
DHWC (D, H, W, C)      — 3-D volumes (e.g. nnU-Net patches)
NDHWC (N, D, H, W, C)  — batch of volumes

Run::

    uv run python benchmarks/benchmark_scale_vs_lut.py
    uv run python benchmarks/benchmark_scale_vs_lut.py --repeats 41 --warmup 12
"""

from __future__ import annotations

import argparse
import platform

import cv2
import numkong as nk
import numpy as np
import stringzilla as sz

from timing import bench_wall_ms

# Non-trivial multiply factor — not 0, 1, or 255 so every implementation does real work.
_ALPHA = 1.5
_BETA = 0.0

# ── LUT construction ─────────────────────────────────────────────────────────


def _build_lut(alpha: float, beta: float) -> np.ndarray:
    """Precomputed uint8 affine LUT: clip(round(alpha*x + beta), 0, 255)."""
    x = np.arange(256, dtype=np.float32)
    return np.clip(np.round(alpha * x + beta), 0, 255).astype(np.uint8)


_LUT = _build_lut(_ALPHA, _BETA)


# ── Implementations ───────────────────────────────────────────────────────────


def impl_nk_scale(img: np.ndarray) -> np.ndarray:
    """NumKong affine scale on ravelled uint8 buffer."""
    flat = np.ascontiguousarray(img).reshape(-1)
    out = nk.scale(nk.Tensor(flat), alpha=_ALPHA, beta=_BETA)
    return np.frombuffer(out, dtype=np.uint8).reshape(img.shape)


def impl_sz_full(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """StringZilla translate over full contiguous buffer (shared LUT)."""
    a = np.ascontiguousarray(img, dtype=np.uint8).reshape(-1)
    out = a.copy()
    sz.translate(memoryview(out), memoryview(lut), inplace=True)
    return out.reshape(img.shape)


def impl_cv2_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """OpenCV LUT on contiguous array (accepts any ndim for a 1-D table)."""
    return cv2.LUT(np.ascontiguousarray(img, dtype=np.uint8), lut)


# ── Shape grid ───────────────────────────────────────────────────────────────

# Canonical shape grid used throughout albucore benchmarks.
#
# HWC (H, W, C):
#   Small: 128×128 and 256×256 — fast iteration / warm-cache behaviour.
#   Medium: 512×512 — typical augmentation training crop.
#   Large: 1024×1024 — high-res / full-image pass.
#   Channels: 1 (grayscale), 3 (RGB), 9 (hyperspectral / >4-ch OpenCV limit).
#
# DHWC (D, H, W, C):
#   3-D medical patches (nnU-Net style).  In-plane ≥128; depth 16–128.
#   1-ch (CT/MRI single modality) and 3-ch (multi-contrast / RGB video).
#
# NDHWC (N, D, H, W, C):
#   Small batches of volumes as seen in a training DataLoader.
#
_SHAPES: list[tuple[int, ...]] = [
    # ── HWC ──────────────────────────────────────────────────────────────
    (128, 128, 1),
    (128, 128, 3),
    (128, 128, 9),
    (256, 256, 1),
    (256, 256, 3),
    (256, 256, 9),
    (512, 512, 1),
    (512, 512, 3),
    (512, 512, 9),
    (1024, 1024, 1),
    (1024, 1024, 3),
    (1024, 1024, 9),
    # ── DHWC ─────────────────────────────────────────────────────────────
    (16, 128, 128, 1),
    (16, 128, 128, 3),
    (32, 128, 128, 1),
    (32, 128, 128, 3),
    (64, 128, 128, 3),
    (128, 128, 128, 1),
    (48, 256, 256, 3),
    # ── NDHWC ────────────────────────────────────────────────────────────
    (2, 32, 128, 128, 1),
    (2, 32, 128, 128, 3),
    (2, 64, 128, 128, 3),
    (4, 16, 128, 128, 3),
]

_LAYOUT = {3: "HWC", 4: "DHWC", 5: "NDHWC"}


# ── Runner ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print()
    print("### Benchmark: `nk.scale` vs `sz_lut` (full-buffer) vs `cv2.LUT` — uint8 affine")
    print()
    print(
        f"alpha={_ALPHA}, beta={_BETA}  |  "
        f"repeats={args.repeats}, warmup={args.warmup}, seed={args.seed}  |  "
        f"{platform.system()} `{platform.machine()}`  |  "
        f"cv2 {cv2.__version__}  numpy {np.__version__}",
    )
    print()
    print(
        "| layout | shape | bytes | nk.scale ms | sz full ms | cv2.LUT ms | fastest | nk/best | sz/best |",
    )
    print(
        "|--------|-------|------:|------------:|-----------:|-----------:|---------|--------:|--------:|",
    )

    for sh in _SHAPES:
        img = rng.integers(0, 256, size=sh, dtype=np.uint8)
        nbytes = img.nbytes

        t_nk = bench_wall_ms(lambda: impl_nk_scale(img), args.repeats, args.warmup)
        t_sz = bench_wall_ms(lambda: impl_sz_full(img, _LUT), args.repeats, args.warmup)
        t_cv = bench_wall_ms(lambda: impl_cv2_lut(img, _LUT), args.repeats, args.warmup)

        best = min(t_nk.median, t_sz.median, t_cv.median)
        if best == t_nk.median:
            fastest = "nk.scale"
        elif best == t_sz.median:
            fastest = "sz"
        else:
            fastest = "cv2"

        nk_vs = t_nk.median / best
        sz_vs = t_sz.median / best

        layout = _LAYOUT.get(len(sh), "?")
        shape_str = "×".join(str(x) for x in sh)

        print(
            f"| {layout} | {shape_str} | {nbytes:,} "
            f"| {t_nk.median:.4f} ± {t_nk.mad:.4f} "
            f"| {t_sz.median:.4f} ± {t_sz.mad:.4f} "
            f"| {t_cv.median:.4f} ± {t_cv.mad:.4f} "
            f"| **{fastest}** | {nk_vs:.2f}× | {sz_vs:.2f}× |",
        )

    print()
    print("**Notes:**")
    print(
        "- `median ± MAD` ms. MAD = median absolute deviation (robust spread across repeats).\n"
        "- **nk.scale**: `nk.scale(nk.Tensor(flat), alpha, beta=0)` — arithmetic path, "
        "handles saturated uint8 rounding internally.\n"
        "- **sz full**: single `sz.translate` over `img.ravel().copy()` with precomputed "
        "`clip(round(α·x), 0, 255)` LUT — same semantics as `apply_uint8_lut` shared path.\n"
        "- **cv2.LUT**: `cv2.LUT(img, lut)` — OpenCV accepts any-ndim uint8 for a 1-D table.\n"
        "- Routing should follow *this machine's* results; re-run before changing production paths.\n",
    )


if __name__ == "__main__":
    main()
