#!/usr/bin/env python3
"""
NumKong patterns from the public "how to normalize with nk" write-up vs OpenCV/NumPy.

Relates to **``normalize()``** (``albucore.arithmetic``, exported from ``albucore.functions``): fixed
``mean`` / ``denominator``. **``normalize_per_image``** (``albucore.normalize``) is different — it
derives stats from ``img``.

Sections:
  1) ImageNet-style fixed mean/std: per-channel ``nk.scale`` on float32 ravel vs broadcast NumPy / OpenCV.
  2) Global min-max on float32: ``Tensor.minmax`` + ``nk.scale`` vs ``np.min/max`` + affine vs ``cv2.normalize``.
  3) Per-channel mean/std stats: ``Tensor`` (H*W, C) with ``sum``/``norm`` axis=0 vs ``cv2.meanStdDev`` / NumPy.

Run from repo root::

    uv run python benchmarks/benchmark_normalize_numkong_patterns.py
"""

from __future__ import annotations

import argparse
import platform

import cv2
import numkong as nk
import numpy as np

from timing import median_ms

MEAN_01 = np.array([0.485, 0.456, 0.406], dtype=np.float64)
STD_01 = np.array([0.229, 0.224, 0.225], dtype=np.float64)
EPS = 1e-4


def _imagenet_vectors(c: int) -> tuple[np.ndarray, np.ndarray]:
    """Tile ImageNet 3-vector to length ``c`` (synthetic multi-spectral / high-C bench)."""
    k = (c + 2) // 3
    mean = np.tile(MEAN_01, k)[:c]
    std = np.tile(STD_01, k)[:c]
    return mean, std


def imagenet_nk_per_channel_scale(img: np.ndarray) -> np.ndarray:
    """Write-up style: one ``nk.scale`` per channel on contiguous float32 ravel (semantic fix vs uint8 tensor)."""
    h, w, c = img.shape
    img_f = np.ascontiguousarray(img, dtype=np.float32)
    out = np.empty((h, w, c), dtype=np.float32)
    mean_v, std_v = _imagenet_vectors(c)
    std_eff = std_v + EPS
    for i in range(c):
        flat = np.ascontiguousarray(img_f[..., i]).reshape(-1)
        alpha = 1.0 / (float(std_eff[i]) * 255.0)
        beta = -float(mean_v[i]) / float(std_eff[i])
        buf = nk.scale(nk.Tensor(flat), alpha=alpha, beta=beta)
        out[..., i] = np.frombuffer(buf, dtype=np.float32).reshape(h, w)
    return np.clip(out, -20.0, 20.0, out=out)


def imagenet_numpy(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32, copy=False)
    c = img_f.shape[-1]
    mean_v, std_v = _imagenet_vectors(c)
    mean = mean_v.reshape(1, 1, -1).astype(np.float32)
    std = (std_v + EPS).reshape(1, 1, -1).astype(np.float32)
    out = (img_f / 255.0 - mean) / std
    return np.clip(out, -20.0, 20.0, out=out)


def imagenet_opencv(img: np.ndarray) -> np.ndarray:
    """Same math as ``(img/255 - mean) / std`` via subtract/divide (see ``normalize`` OpenCV path)."""
    img_f = np.ascontiguousarray(img, dtype=np.float32)
    c = img_f.shape[-1]
    mean_v, std_v = _imagenet_vectors(c)
    mean_bc = np.reshape(mean_v * 255.0, (1, 1, c)).astype(np.float32)
    std_bc = np.reshape((std_v + EPS) * 255.0, (1, 1, c)).astype(np.float32)
    # OpenCV element-wise ops need spatially broadcast arrays (not just (1,1,C) views).
    mean_255 = np.broadcast_to(mean_bc, img_f.shape).astype(np.float32, copy=False)
    std_255 = np.broadcast_to(std_bc, img_f.shape).astype(np.float32, copy=False)
    if c > 4:
        mean_255 = np.ascontiguousarray(mean_255)
        std_255 = np.ascontiguousarray(std_255)
    normalized = cv2.divide(cv2.subtract(img_f, mean_255, dtype=cv2.CV_32F), std_255, dtype=cv2.CV_32F)
    return np.clip(normalized, -20.0, 20.0, out=normalized)


def minmax_nk_minmax_then_scale(img: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1)
    t = nk.Tensor(flat)
    mn_f, _, mx_f, _ = t.minmax()
    mn, mx = float(mn_f), float(mx_f)
    denom = mx - mn + EPS
    buf = nk.scale(t, alpha=1.0 / denom, beta=-mn / denom)
    out = np.frombuffer(buf, dtype=np.float32).reshape(img.shape)
    return np.clip(out, 0.0, 1.0, out=out)


def minmax_numpy(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    d = mx - mn + EPS
    out = (img - mn) / d
    return np.clip(out, 0.0, 1.0, out=out)


def minmax_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def per_channel_stats_numpy(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ax = tuple(range(img.ndim - 1))
    m = img.mean(axis=ax, dtype=np.float64)
    s = img.std(axis=ax, dtype=np.float64) + EPS
    return np.asarray(m), np.asarray(s)


def per_channel_stats_cv2(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = cv2.meanStdDev(img)
    m = mean[:, 0].astype(np.float64, copy=False)
    st = (std[:, 0] + EPS).astype(np.float64, copy=False)
    return np.asarray(m), np.asarray(st)


def per_channel_stats_nk_writeup(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``sum``/``norm`` on (H*W, C), population std from sums of squares."""
    h, w, c = img.shape
    pix = nk.Tensor(np.ascontiguousarray(img.reshape(-1, c)))
    sums = np.asarray(pix.sum(axis=0), dtype=np.float64)
    norms = np.asarray(pix.norm(axis=0), dtype=np.float64)
    n = float(h * w)
    mean = sums / n
    var = np.maximum(norms * norms / n - mean * mean, 0.0)
    std = np.sqrt(var) + EPS
    return mean, std


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    shapes_hwc: list[tuple[int, int, int]] = [
        (128, 128, 3),
        (256, 256, 3),
        (512, 512, 3),
        (1024, 1024, 3),
        (512, 512, 9),
    ]

    print()
    print("### Benchmark: NumKong normalize patterns vs Albucore-style backends")
    print()
    print(
        f"_Median ms, repeats={args.repeats}, warmup={args.warmup}, seed={args.seed}; "
        f"{platform.system()} `{platform.machine()}`, numkong "
        f"{getattr(__import__('numkong'), '__version__', '?')}, numpy {np.__version__}._",
    )
    print()

    print("#### 1) ImageNet fixed mean/std (uint8 input → float32 output)")
    print()
    print(
        "NumKong: **per-channel** ``nk.scale`` on **float32** ravel (write-up uses uint8 tensor; "
        "that returns uint8 buffers — wrong for training; we time the float32 analogue). "
        "NumPy/OpenCV: **one** vectorized pass with broadcast (same as ``normalize`` math).",
    )
    print()
    print("| H×W×C | NK loop (ms) | NumPy (ms) | OpenCV (ms) | fastest | NK vs best |")
    print("|-------|-------------:|-----------:|------------:|---------|------------:|")

    for h, w, c in shapes_hwc:
        if c not in (3, 9):
            continue
        img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        t_nk = median_ms(lambda: imagenet_nk_per_channel_scale(img), args.repeats, args.warmup)
        t_np = median_ms(lambda: imagenet_numpy(img), args.repeats, args.warmup)
        t_cv = median_ms(lambda: imagenet_opencv(img), args.repeats, args.warmup)
        best = min(t_nk, t_np, t_cv)
        winner = "NK" if best == t_nk else ("NumPy" if best == t_np else "OpenCV")
        ratio_nk = t_nk / max(best, 1e-12)
        print(f"| {h}×{w}×{c} | {t_nk:.4f} | {t_np:.4f} | {t_cv:.4f} | {winner} | {ratio_nk:.2f}× |")

    print()
    print("#### 2) Global min–max normalize (float32 → [0,1])")
    print()
    print("NK: ``minmax`` on flat tensor + ``nk.scale``. NumPy: ``min``/``max`` + affine. OpenCV: ``cv2.normalize``.")
    print()
    print("| H×W×C | NK (ms) | NumPy (ms) | OpenCV (ms) | fastest | NK vs best |")
    print("|-------|--------:|-----------:|------------:|---------|------------:|")

    for h, w, c in shapes_hwc:
        img = rng.random((h, w, c), dtype=np.float32)
        t_nk = median_ms(lambda: minmax_nk_minmax_then_scale(img), args.repeats, args.warmup)
        t_np = median_ms(lambda: minmax_numpy(img), args.repeats, args.warmup)
        t_cv = median_ms(lambda: minmax_cv2(img), args.repeats, args.warmup)
        best = min(t_nk, t_np, t_cv)
        winner = "NK" if best == t_nk else ("NumPy" if best == t_np else "OpenCV")
        ratio_nk = t_nk / max(best, 1e-12)
        print(f"| {h}×{w}×{c} | {t_nk:.4f} | {t_np:.4f} | {t_cv:.4f} | {winner} | {ratio_nk:.2f}× |")

    print()
    print("#### 3) Per-channel mean & std (stats only, float32 HWC)")
    print()
    print(
        "NK write-up: ``Tensor(H*W, C)`` then ``sum(axis=0)`` and ``norm(axis=0)`` → mean/std. "
        "vs ``cv2.meanStdDev`` (C≤4 only in Albucore) vs NumPy ``mean``/``std`` over spatial axes.",
    )
    print()
    print("| H×W×C | NK (ms) | cv2.meanStdDev (ms) | NumPy (ms) | fastest | NK vs best |")
    print("|-------|--------:|--------------------:|-----------:|---------|------------:|")

    for h, w, c in shapes_hwc:
        img = rng.random((h, w, c), dtype=np.float32)
        t_nk = median_ms(lambda: per_channel_stats_nk_writeup(img), args.repeats, args.warmup)
        t_np = median_ms(lambda: per_channel_stats_numpy(img), args.repeats, args.warmup)
        if c <= 4:
            t_cv = median_ms(lambda: per_channel_stats_cv2(img), args.repeats, args.warmup)
            best = min(t_nk, t_np, t_cv)
            winner = "NK" if best == t_nk else ("NumPy" if best == t_np else "cv2")
            ratio_nk = t_nk / max(best, 1e-12)
            print(f"| {h}×{w}×{c} | {t_nk:.4f} | {t_cv:.4f} | {t_np:.4f} | {winner} | {ratio_nk:.2f}× |")
        else:
            best = min(t_nk, t_np)
            winner = "NK" if best == t_nk else "NumPy"
            ratio_nk = t_nk / max(best, 1e-12)
            print(f"| {h}×{w}×{c} | {t_nk:.4f} | — | {t_np:.4f} | {winner} | {ratio_nk:.2f}× |")

    print()
    print("#### Takeaway (regenerate on your machine)")
    print()
    print(
        "- **Fixed constants (same math as ``normalize(img, mean, denominator)`` / ImageNet α,β):** "
        "on **this** run, a **3-channel** loop of ``nk.scale`` on float32 ravels beat **one-shot** NumPy "
        "and OpenCV; with **C=9** (tiled ImageNet vectors) **NumPy** won. Routing **inside** "
        "``normalize`` should stay **benchmark-driven per C**.\n"
        "- **Global min–max:** **OpenCV** ``cv2.normalize`` dominated **NK minmax + nk.scale** and NumPy "
        "here — aligns with keeping **min_max** on the OpenCV path in ``normalize_per_image``.\n"
        "- **Per-channel mean/std stats:** **cv2.meanStdDev** won for **C=3**; for **C=9**, **NumPy** "
        "beat the write-up’s **sum+norm** NK recipe. No change suggested to ``stats._mean_std_per_channel`` "
        "from this bench.\n"
        "- **Write-up caveat:** ``nk.scale`` on **uint8** ravel returns a **uint8** buffer (not a float "
        "feature map); use **float32** ravels (or a documented NK dtype) for training-style normalize.",
    )
    print()


if __name__ == "__main__":
    main()
