#!/usr/bin/env python3
"""
Compare NumKong-backed paths against **other albucore backends** (OpenCV, NumPy, LUT),
not raw NumPy in isolation — same spirit as production routing (pick the fastest known path).

Requires OpenCV (e.g. ``uv sync --extra headless``).

    uv run python benchmarks/benchmark_numkong_vs_albucore_backends.py
"""

from __future__ import annotations

import argparse

import cv2
import numkong as nk
import numpy as np

from albucore.functions import (
    add_weighted_lut,
    add_weighted_numkong,
    add_weighted_numpy,
    add_weighted_opencv,
)
from albucore.stats import mean_std
from timing import median_ms


def per_channel_axes(ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim - 1))


def per_channel_stats_numkong(x: np.ndarray, eps: float) -> None:
    """Population mean/std per channel via one ``moments`` call per channel."""
    for ci in range(x.shape[-1]):
        s_sum, s_sq = nk.moments(x[..., ci])
        n = x[..., ci].size
        m = float(s_sum) / n
        var = max(float(s_sq) / n - m * m, 0.0)
        float(m)
        float(np.sqrt(var)) + eps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    channels = [1, 3, 9]
    w1, w2 = 0.5, 0.5

    print("# Albucore backends: NumKong vs OpenCV / NumPy / LUT")
    print()
    print("Median ms; **fastest alt** = min(OpenCV, NumPy, [LUT for uint8]).")
    print()

    # --- add_weighted uint8 ---
    print("## `add_weighted` — uint8 (weights 0.5 / 0.5)")
    print()
    print("| H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |")
    print("|-----|---|--------|--------:|-------:|------:|----:|------------:|---------------:|")
    for h, w in sizes:
        for c in channels:
            img1 = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            img2 = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)

            t_nk = median_ms(lambda: add_weighted_numkong(img1, w1, img2, w2), args.repeats, args.warmup)
            t_cv = median_ms(lambda: add_weighted_opencv(img1, w1, img2, w2), args.repeats, args.warmup)
            t_np = median_ms(lambda: add_weighted_numpy(img1, w1, img2, w2), args.repeats, args.warmup)
            t_lut = median_ms(lambda: add_weighted_lut(img1, w1, img2, w2), args.repeats, args.warmup)

            alts = {"OpenCV": t_cv, "NumPy": t_np, "LUT": t_lut}
            best_name = min(alts, key=alts.get)
            best_t = alts[best_name]
            if t_nk <= best_t:
                vs = f"NK {best_t / max(t_nk, 1e-12):.2f}× faster than {best_name}"
            else:
                vs = f"{best_name} {t_nk / best_t:.2f}× faster than NK"
            print(
                f"| {h}×{w} | {c} | {h * w * c} | {t_nk:.4f} | {t_cv:.4f} | {t_np:.4f} | {t_lut:.4f} | "
                f"{best_name} ({best_t:.4f}) | {vs} |",
            )
    print()

    # --- add_weighted float32 ---
    print("## `add_weighted` — float32 (weights 0.5 / 0.5; no LUT)")
    print()
    print("| H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |")
    print("|-----|---|--------|--------:|-------:|------:|------------:|---------------:|")
    for h, w in sizes:
        for c in channels:
            img1 = rng.random((h, w, c), dtype=np.float32)
            img2 = rng.random((h, w, c), dtype=np.float32)

            t_nk = median_ms(lambda: add_weighted_numkong(img1, w1, img2, w2), args.repeats, args.warmup)
            t_cv = median_ms(lambda: add_weighted_opencv(img1, w1, img2, w2), args.repeats, args.warmup)
            t_np = median_ms(lambda: add_weighted_numpy(img1, w1, img2, w2), args.repeats, args.warmup)

            alts = {"OpenCV": t_cv, "NumPy": t_np}
            best_name = min(alts, key=alts.get)
            best_t = alts[best_name]
            if t_nk <= best_t:
                vs = f"NK {best_t / max(t_nk, 1e-12):.2f}× faster than {best_name}"
            else:
                vs = f"{best_name} {t_nk / best_t:.2f}× faster than NK"
            print(
                f"| {h}×{w} | {c} | {h * w * c} | {t_nk:.4f} | {t_cv:.4f} | {t_np:.4f} | "
                f"{best_name} ({best_t:.4f}) | {vs} |",
            )
    print()

    # --- add_weighted batch (N,H,W,C) — same APIs, raveled in NumKong path ---
    nb, hb, wb = 4, 256, 256
    print(f"## `add_weighted` — batch / video `(N,H,W,C)`, N={nb}, H×W={hb}×{wb}")
    print()
    print("Same weights **0.5 / 0.5**. Pixels = N×H×W×C.")
    print()
    print("### uint8")
    print()
    print("| N×H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |")
    print("|-------|---|--------|--------:|-------:|------:|----:|------------:|---------------:|")
    for c in channels:
        img1 = rng.integers(0, 256, size=(nb, hb, wb, c), dtype=np.uint8)
        img2 = rng.integers(0, 256, size=(nb, hb, wb, c), dtype=np.uint8)
        npx = nb * hb * wb * c
        t_nk = median_ms(lambda: add_weighted_numkong(img1, w1, img2, w2), args.repeats, args.warmup)
        t_cv = median_ms(lambda: add_weighted_opencv(img1, w1, img2, w2), args.repeats, args.warmup)
        t_np = median_ms(lambda: add_weighted_numpy(img1, w1, img2, w2), args.repeats, args.warmup)
        t_lut = median_ms(lambda: add_weighted_lut(img1, w1, img2, w2), args.repeats, args.warmup)
        alts = {"OpenCV": t_cv, "NumPy": t_np, "LUT": t_lut}
        best_name = min(alts, key=alts.get)
        best_t = alts[best_name]
        if t_nk <= best_t:
            vs = f"NK {best_t / max(t_nk, 1e-12):.2f}× faster than {best_name}"
        else:
            vs = f"{best_name} {t_nk / best_t:.2f}× faster than NK"
        print(
            f"| {nb}×{hb}×{wb} | {c} | {npx} | {t_nk:.4f} | {t_cv:.4f} | {t_np:.4f} | {t_lut:.4f} | "
            f"{best_name} ({best_t:.4f}) | {vs} |",
        )
    print()
    print("### float32 (no LUT)")
    print()
    print("| N×H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |")
    print("|-------|---|--------|--------:|-------:|------:|------------:|---------------:|")
    for c in channels:
        img1 = rng.random((nb, hb, wb, c), dtype=np.float32)
        img2 = rng.random((nb, hb, wb, c), dtype=np.float32)
        npx = nb * hb * wb * c
        t_nk = median_ms(lambda: add_weighted_numkong(img1, w1, img2, w2), args.repeats, args.warmup)
        t_cv = median_ms(lambda: add_weighted_opencv(img1, w1, img2, w2), args.repeats, args.warmup)
        t_np = median_ms(lambda: add_weighted_numpy(img1, w1, img2, w2), args.repeats, args.warmup)
        alts = {"OpenCV": t_cv, "NumPy": t_np}
        best_name = min(alts, key=alts.get)
        best_t = alts[best_name]
        if t_nk <= best_t:
            vs = f"NK {best_t / max(t_nk, 1e-12):.2f}× faster than {best_name}"
        else:
            vs = f"{best_name} {t_nk / best_t:.2f}× faster than NK"
        print(
            f"| {nb}×{hb}×{wb} | {c} | {npx} | {t_nk:.4f} | {t_cv:.4f} | {t_np:.4f} | "
            f"{best_name} ({best_t:.4f}) | {vs} |",
        )
    print()

    eps = 1e-4
    cv_global_note = (
        "OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. "
        "It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and "
        "std-only tables for C=1."
    )

    def fastest_name(*, t_np: float, t_nk: float, t_cv: float | None) -> str:
        opts: list[tuple[str, float]] = [("NumPy", t_np), ("NumKong", t_nk)]
        if t_cv is not None:
            opts.append(("OpenCV", t_cv))
        return min(opts, key=lambda x: x[1])[0]

    # --- global mean only ---
    print("## Global **mean only** (uint8) — NumPy vs NumKong vs OpenCV")
    print()
    print(cv_global_note)
    print()
    print("NumPy: `float(img.mean())`. NumKong: `moments` on contiguous ravel, `mean = s/n`. OpenCV (C=1): `meanStdDev`, read scalar mean.")
    print()
    print("| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
    print("|-----|---|--------|------:|--------:|-------:|--------|")
    for h, w in sizes:
        for c in channels:
            img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            n = h * w * c

            t_np = median_ms(lambda: float(img.mean()), args.repeats, args.warmup)

            def nk_mean_only(a=img) -> None:
                s_sum, _s2 = nk.moments(a)
                float(s_sum) / a.size

            t_nk = median_ms(nk_mean_only, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[0][0, 0]),
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {h}×{w} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
    print()

    print("## Global **std only** (uint8) — NumPy vs NumKong vs OpenCV")
    print()
    print(cv_global_note)
    print()
    print(
        "NumPy: `float(img.std()) + eps`. NumKong: full `moments` → population std + eps. OpenCV (C=1): `meanStdDev`, read scalar std + eps.",
    )
    print()
    print("| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
    print("|-----|---|--------|------:|--------:|-------:|--------|")
    for h, w in sizes:
        for c in channels:
            img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            n = h * w * c

            t_np = median_ms(lambda: float(img.std()) + eps, args.repeats, args.warmup)

            def nk_std_only(a=img) -> None:
                s_sum, s_sq = nk.moments(a)
                m = float(s_sum) / a.size
                v = max(float(s_sq) / a.size - m * m, 0.0)
                float(np.sqrt(v)) + eps

            t_nk = median_ms(nk_std_only, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[1][0, 0]) + eps,
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {h}×{w} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
    print()

    print("## Global **mean only** (float32) — NumPy vs NumKong vs OpenCV")
    print()
    print(cv_global_note)
    print()
    print("NumPy: `float(img.mean())`. NumKong: `moments` on contiguous ravel. OpenCV (C=1): `meanStdDev` on float32 image.")
    print()
    print("| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
    print("|-----|---|--------|------:|--------:|-------:|--------|")
    for h, w in sizes:
        for c in channels:
            img = rng.random((h, w, c), dtype=np.float32)
            n = h * w * c

            t_np = median_ms(lambda: float(img.mean()), args.repeats, args.warmup)

            def nk_mean_only_f(a=img) -> None:
                s_sum, _s2 = nk.moments(a)
                float(s_sum) / a.size

            t_nk = median_ms(nk_mean_only_f, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[0][0, 0]),
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {h}×{w} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
    print()

    print("## Global **std only** (float32) — NumPy vs NumKong vs OpenCV")
    print()
    print(cv_global_note)
    print()
    print("NumPy: `float(img.std()) + eps` (population `ddof=0`). NumKong: `moments` → population std + eps.")
    print()
    print("| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
    print("|-----|---|--------|------:|--------:|-------:|--------|")
    for h, w in sizes:
        for c in channels:
            img = rng.random((h, w, c), dtype=np.float32)
            n = h * w * c

            t_np = median_ms(lambda: float(img.std()) + eps, args.repeats, args.warmup)

            def nk_std_only_f(a=img) -> None:
                s_sum, s_sq = nk.moments(a)
                m = float(s_sum) / a.size
                v = max(float(s_sq) / a.size - m * m, 0.0)
                float(np.sqrt(v)) + eps

            t_nk = median_ms(nk_std_only_f, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[1][0, 0]) + eps,
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {h}×{w} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
    print()

    # --- global stats batch (N,H,W,C) same formulas, full tensor ravel ---
    print(
        f"## Global statistics — batch / video `(N,H,W,C)`, N={nb}, H×W={hb}×{wb} "
        "(same mean/std semantics as image: reduce over **all** pixels)",
    )
    print()
    print(cv_global_note)
    print()

    def row_batch_global_mean_u8() -> None:
        print("### Batch — global **mean only** (uint8)")
        print()
        print("| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
        print("|-------|---|--------|------:|--------:|-------:|--------|")
        for c in channels:
            img = rng.integers(0, 256, size=(nb, hb, wb, c), dtype=np.uint8)
            n = img.size
            t_np = median_ms(lambda: float(img.mean()), args.repeats, args.warmup)

            def nk_m(a=img) -> None:
                s_sum, _s2 = nk.moments(a)
                float(s_sum) / a.size

            t_nk = median_ms(nk_m, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[0][0, 0]),
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {nb}×{hb}×{wb} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
        print()

    def row_batch_global_std_u8() -> None:
        print("### Batch — global **std only** (uint8)")
        print()
        print("| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
        print("|-------|---|--------|------:|--------:|-------:|--------|")
        for c in channels:
            img = rng.integers(0, 256, size=(nb, hb, wb, c), dtype=np.uint8)
            n = img.size
            t_np = median_ms(lambda: float(img.std()) + eps, args.repeats, args.warmup)

            def nk_s(a=img) -> None:
                s_sum, s_sq = nk.moments(a)
                m = float(s_sum) / a.size
                v = max(float(s_sq) / a.size - m * m, 0.0)
                float(np.sqrt(v)) + eps

            t_nk = median_ms(nk_s, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[1][0, 0]) + eps,
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {nb}×{hb}×{wb} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
        print()

    def row_batch_global_mean_f32() -> None:
        print("### Batch — global **mean only** (float32)")
        print()
        print("| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
        print("|-------|---|--------|------:|--------:|-------:|--------|")
        for c in channels:
            img = rng.random((nb, hb, wb, c), dtype=np.float32)
            n = img.size
            t_np = median_ms(lambda: float(img.mean()), args.repeats, args.warmup)

            def nk_m(a=img) -> None:
                s_sum, _s2 = nk.moments(a)
                float(s_sum) / a.size

            t_nk = median_ms(nk_m, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[0][0, 0]),
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {nb}×{hb}×{wb} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
        print()

    def row_batch_global_std_f32() -> None:
        print("### Batch — global **std only** (float32)")
        print()
        print("| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |")
        print("|-------|---|--------|------:|--------:|-------:|--------|")
        for c in channels:
            img = rng.random((nb, hb, wb, c), dtype=np.float32)
            n = img.size
            t_np = median_ms(lambda: float(img.std()) + eps, args.repeats, args.warmup)

            def nk_s(a=img) -> None:
                s_sum, s_sq = nk.moments(a)
                m = float(s_sum) / a.size
                v = max(float(s_sq) / a.size - m * m, 0.0)
                float(np.sqrt(v)) + eps

            t_nk = median_ms(nk_s, args.repeats, args.warmup)
            if c == 1:
                t_cv = median_ms(
                    lambda: float(cv2.meanStdDev(img)[1][0, 0]) + eps,
                    args.repeats,
                    args.warmup,
                )
                cv_s = f"{t_cv:.4f}"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=t_cv)
            else:
                cv_s = "N/A"
                win = fastest_name(t_np=t_np, t_nk=t_nk, t_cv=None)
            print(f"| {nb}×{hb}×{wb} | {c} | {n} | {t_np:.4f} | {t_nk:.4f} | {cv_s} | {win} |")
        print()

    row_batch_global_mean_u8()
    row_batch_global_std_u8()
    row_batch_global_mean_f32()
    row_batch_global_std_f32()

    # --- per-channel: (H,W,C), (N,H,W,C), (N,D,H,W,C) ---
    print("## Per-channel mean + std — `(H,W,C)`, `(N,H,W,C)`, `(N,D,H,W,C)`")
    print()
    print(
        "Reduce over **all axes except channel** (`shape[-1]`). "
        "**NP mean** / **NP std**: separate full reductions over those axes. **NP both**: `mean` then `std` in one timed block. "
        "**albucore**: `mean_std(img, \"per_channel\", eps=…)` (3D: OpenCV + NumPy routing in `stats`; higher rank → NumPy axis-reduce). "
        "**NK**: one NumKong `moments` per channel (no batched per-channel API in this bench).",
    )
    print()

    def row_pc(dtype_name: str, layout: str, img: np.ndarray) -> None:
        axes = per_channel_axes(img.ndim)
        c = img.shape[-1]
        n = img.size
        t_nm = median_ms(lambda: img.mean(axis=axes), args.repeats, args.warmup)
        t_ns = median_ms(lambda: img.std(axis=axes) + eps, args.repeats, args.warmup)

        def np_both_pc() -> None:
            img.mean(axis=axes)
            img.std(axis=axes) + eps

        t_nb = median_ms(np_both_pc, args.repeats, args.warmup)
        t_pr = median_ms(lambda: mean_std(img, "per_channel", eps=eps), args.repeats, args.warmup)
        t_nk = median_ms(lambda: per_channel_stats_numkong(img, eps), args.repeats, args.warmup)
        print(
            f"| {dtype_name} | {layout} | {c} | {n} | {t_nm:.4f} | {t_ns:.4f} | {t_nb:.4f} | {t_pr:.4f} | {t_nk:.4f} |",
        )

    print("| dtype | layout (indices …×C) | C | pixels | NP mean | NP std | NP both | albucore | NK (C×moments) |")
    print("|-------|------------------------|---|--------|--------:|-------:|--------:|---------:|---------------:|")

    for h, w in sizes:
        for c in channels:
            u8 = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
            row_pc("uint8", f"{h}×{w}×{c}", u8)
            f32 = rng.random((h, w, c), dtype=np.float32)
            row_pc("float32", f"{h}×{w}×{c}", f32)

    hb, wb = 256, 256
    for c in channels:
        u8 = rng.integers(0, 256, size=(4, hb, wb, c), dtype=np.uint8)
        row_pc("uint8", f"4×{hb}×{wb}×{c}", u8)
        f32 = rng.random((4, hb, wb, c), dtype=np.float32)
        row_pc("float32", f"4×{hb}×{wb}×{c}", f32)

    h5, w5 = 64, 64
    for c in channels:
        u8 = rng.integers(0, 256, size=(2, 4, h5, w5, c), dtype=np.uint8)
        row_pc("uint8", f"2×4×{h5}×{w5}×{c}", u8)
        f32 = rng.random((2, 4, h5, w5, c), dtype=np.float32)
        row_pc("float32", f"2×4×{h5}×{w5}×{c}", f32)


if __name__ == "__main__":
    main()
