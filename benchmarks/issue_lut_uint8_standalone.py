#!/usr/bin/env python3
# ruff: noqa: T201, EXE001, RUF001, C901, PLR0912, PLR0915
"""Standalone **uint8 → uint8 LUT** benchmark: **OpenCV `cv2.LUT`** vs **StringZilla `translate`**.

Mirrors the routing logic used in Albucore's ``apply_uint8_lut``:

  Shared LUT ``(256,)``
  ├── HWC contiguous, large (hw >= 640² or 512²×C≥6):  cv2.LUT on full array  [production uses cv2]
  ├── HWC contiguous, small / C=1:                      sz.translate on flat ravel  [production uses SZ]
  └── DHWC / NDHWC any size:                            sz.translate on flat ravel  [production ALWAYS SZ]

  Per-channel LUT ``(C,256)``
  ├── HWC contiguous, C>1:   cv2.LUT(img, (256,1,C))  [production uses cv2 one-shot]
  └── DHWC / NDHWC any C:   SZ loop per channel        [production uses SZ loop]

For each path we benchmark both the **current production behaviour** and the
**best alternative** to show the gap.

- **No albucore** — paste this single file anywhere with ``numpy``, ``opencv-python``, ``stringzilla``.
- LUT is a fixed-seed **permutation** of ``0..255`` (non-trivial; not identity ``arange``).
- Buffer reuse: where production allocates, we also time a version with a preallocated ``dst``.

Run::

    python issue_lut_uint8_standalone.py [--repeats 15] [--warmup 5]
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import time

import cv2
import numpy as np
import stringzilla as sz

LUT_PERM_SEED = 42

# ---------------------------------------------------------------------------
# Shape coverage
# ---------------------------------------------------------------------------
# HWC: 2D images / frames — small to large, C=1/3/9
HWC_SHAPES: list[tuple[int, ...]] = [
    (256, 256, 1),
    (256, 256, 3),
    (256, 256, 9),
    (512, 512, 1),
    (512, 512, 3),
    (512, 512, 9),
    (640, 640, 3),
    (640, 640, 9),
    (1024, 1024, 1),
    (1024, 1024, 3),
    (1024, 1024, 9),
]

# DHWC / NDHWC: 3D medical / scientific volumes, video sequences
# D = slices/frames, HW = spatial, C = channels
VOLUME_SHAPES: list[tuple[int, ...]] = [
    # Volumes: medical segmentation patches (typical nnU-Net / monai patch sizes)
    (32, 128, 128, 1),   # grayscale CT patch
    (32, 128, 128, 3),   # multi-modal
    (64, 128, 128, 1),
    (64, 128, 128, 3),
    (64, 128, 128, 9),   # hyperspectral / many modalities
    (128, 128, 128, 1),  # isotropic cube
    (48, 256, 256, 3),
    (96, 160, 160, 3),
    (64, 256, 256, 9),   # large hyperspectral volume
    # Batched volumes: NDHWC (N=batch, D=depth, H, W, C)
    (2, 32, 128, 128, 1),
    (2, 32, 128, 128, 3),
    (2, 64, 128, 128, 3),
    (2, 64, 128, 128, 9),
    # Video-like: many frames, typical robotics / autonomous driving
    (30, 640, 640, 3),   # 30 frames 640×640 RGB
    (8, 1024, 1024, 3),  # 8 frames 1MP RGB
    (16, 256, 256, 9),   # 16 frames, 9-channel lidar/radar fusion
]


# ---------------------------------------------------------------------------
# Routing replication (mirrors albucore/lut.py without importing it)
# ---------------------------------------------------------------------------
def _hwc_cv2_faster_shared(shape: tuple[int, ...]) -> bool:
    """Mirrors ``opencv_shared_uint8_lut_faster_hwc`` in albucore/lut.py."""
    if len(shape) != 3:
        return False
    c = shape[-1]
    if c < 2:
        return False
    n = shape[0] * shape[1] * shape[2]
    hw = n // c
    if hw >= 409600:  # ~640×640
        return True
    return hw >= 262_144 and n >= 1_310_000


# ---------------------------------------------------------------------------
# LUT construction
# ---------------------------------------------------------------------------
def make_shared_lut() -> np.ndarray:
    return np.random.default_rng(LUT_PERM_SEED).permutation(256).astype(np.uint8)


def make_per_channel_luts(c: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(256,1,C)`` for cv2 one-shot; list of ``(256,)`` for SZ loop."""
    lut_cv2 = np.zeros((256, 1, c), dtype=np.uint8)
    luts_1d: list[np.ndarray] = []
    for i in range(c):
        col = np.random.default_rng(LUT_PERM_SEED + 1000 + i).permutation(256).astype(np.uint8)
        lut_cv2[:, 0, i] = col
        luts_1d.append(col)
    return lut_cv2, luts_1d


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def median_ms(fn: object, repeats: int, warmup: int) -> float:
    f = fn  # type: ignore[assignment]
    for _ in range(warmup):
        f()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def layout_name(ndim: int) -> str:
    return {3: "HWC", 4: "DHWC", 5: "NDHWC"}.get(ndim, "?")


# ---------------------------------------------------------------------------
# Section 1: Shared LUT on HWC
# Routes to cv2 or SZ based on size; show *both* for every shape.
# ---------------------------------------------------------------------------
def bench_hwc_shared(img: np.ndarray, lut: np.ndarray, repeats: int, warmup: int) -> dict[str, float]:
    flat = img.reshape(-1)
    n = flat.size
    dst = np.empty_like(img)
    buf = np.empty(n, dtype=np.uint8)

    results: dict[str, float] = {}

    def cv2_new() -> None:
        cv2.LUT(img, lut)

    def cv2_dst() -> None:
        cv2.LUT(img, lut, dst)

    # Production SZ path: sz_lut modifies img in place (or a copy).
    # We benchmark the flat-ravel translate which is what sz_lut does on a contiguous array.
    def sz_reuse() -> None:
        np.copyto(buf, flat)
        sz.translate(memoryview(buf), memoryview(lut), inplace=True)

    results["cv2 new"] = median_ms(cv2_new, repeats, warmup)
    results["cv2→dst"] = median_ms(cv2_dst, repeats, warmup)
    results["SZ ravel+reuse"] = median_ms(sz_reuse, repeats, warmup)
    return results


# ---------------------------------------------------------------------------
# Section 2: Shared LUT on DHWC / NDHWC
# Production ALWAYS uses SZ (flat ravel). cv2 is shown as baseline only.
# ---------------------------------------------------------------------------
def bench_volume_shared(img: np.ndarray, lut: np.ndarray, repeats: int, warmup: int) -> dict[str, float]:
    flat = img.reshape(-1)
    n = flat.size
    dst = np.empty_like(img)
    buf = np.empty(n, dtype=np.uint8)

    results: dict[str, float] = {}

    # cv2 — NOT what production uses for volumes, shown as baseline
    def cv2_new() -> None:
        cv2.LUT(img, lut)

    def cv2_dst() -> None:
        cv2.LUT(img, lut, dst)

    # Production path
    def sz_reuse() -> None:
        np.copyto(buf, flat)
        sz.translate(memoryview(buf), memoryview(lut), inplace=True)

    results["cv2 new (not production)"] = median_ms(cv2_new, repeats, warmup)
    results["cv2→dst (not production)"] = median_ms(cv2_dst, repeats, warmup)
    results["SZ ravel+reuse [production]"] = median_ms(sz_reuse, repeats, warmup)
    return results


# ---------------------------------------------------------------------------
# Section 3: Per-channel LUT on HWC
# Production uses cv2.LUT(img, (256,1,C)) one-shot.
# ---------------------------------------------------------------------------
def bench_hwc_per_channel(
    img: np.ndarray,
    lut_cv2: np.ndarray,
    luts_1d: list[np.ndarray],
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    c = img.shape[-1]
    dst = np.empty_like(img)
    out = np.empty_like(img)
    spatial = img.shape[:2]
    ch = np.empty(spatial, dtype=np.uint8)

    results: dict[str, float] = {}

    if c == 1:
        lut1d = luts_1d[0]

        def cv2_one_shot_new() -> None:
            cv2.LUT(img, lut1d)

        def cv2_one_shot_dst() -> None:
            cv2.LUT(img, lut1d, dst)

        results["cv2 new (C=1)"] = median_ms(cv2_one_shot_new, repeats, warmup)
        results["cv2→dst (C=1)"] = median_ms(cv2_one_shot_dst, repeats, warmup)
    else:
        def cv2_one_shot_new() -> None:
            cv2.LUT(img, lut_cv2)

        def cv2_one_shot_dst() -> None:
            cv2.LUT(img, lut_cv2, dst)

        results["cv2 new [production]"] = median_ms(cv2_one_shot_new, repeats, warmup)
        results["cv2→dst"] = median_ms(cv2_one_shot_dst, repeats, warmup)

    def sz_loop_reuse() -> None:
        for i in range(c):
            np.copyto(ch, img[..., i])
            sz.translate(memoryview(ch), memoryview(luts_1d[i]), inplace=True)
            out[..., i] = ch

    results["SZ loop reuse ch"] = median_ms(sz_loop_reuse, repeats, warmup)
    return results


# ---------------------------------------------------------------------------
# Section 4: Per-channel LUT on DHWC / NDHWC
# Production uses SZ loop per channel. cv2 cannot do (256,1,C) on ndim>3.
# ---------------------------------------------------------------------------
def bench_volume_per_channel(
    img: np.ndarray,
    luts_1d: list[np.ndarray],
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    c = img.shape[-1]
    out = np.empty_like(img)
    spatial = img.shape[:-1]
    ch = np.empty(spatial, dtype=np.uint8)

    results: dict[str, float] = {}

    def sz_loop_reuse() -> None:
        for i in range(c):
            np.copyto(ch, img[..., i])
            sz.translate(memoryview(ch), memoryview(luts_1d[i]), inplace=True)
            out[..., i] = ch

    results["SZ loop reuse ch [production]"] = median_ms(sz_loop_reuse, repeats, warmup)

    # cv2 cannot do per-channel LUT on ndim>3 in one call.
    # For C=1 it can treat the leading dims as a flat 2D array; for C>1 there is no efficient path.
    if c == 1:
        slices = img.size  # all elements, since channel dim is 1
        lut1 = luts_1d[0]
        flat2d = img.reshape(-1, 1)
        out2 = np.empty_like(flat2d)

        def cv2_single_channel_flat() -> None:
            cv2.LUT(flat2d, lut1, out2)

        results["cv2 flat 2D (C=1)"] = median_ms(cv2_single_channel_flat, repeats, warmup)

    return results



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    lut = make_shared_lut()
    r, w = args.repeats, args.warmup

    try:
        sz_ver = importlib.metadata.version("stringzilla")
    except importlib.metadata.PackageNotFoundError:
        sz_ver = "?"
    env = (
        f"_Median ms, repeats={r}, warmup={w}; "
        f"{platform.system()} `{platform.machine()}`, "
        f"OpenCV {cv2.__version__}, numpy {np.__version__}, stringzilla {sz_ver}._"
    )

    # ------------------------------------------------------------------ HWC shared
    print()
    print("## Section 1 — Shared LUT `(256,)` on `HWC`")
    print()
    print("Production routes large multi-channel to **cv2**, small/C=1 to **SZ**.")
    print("Both are shown for every shape; `*` marks the production path.")
    print()
    print(env)
    print("| layout | shape | pixels | cv2 new | cv2→dst | SZ ravel+reuse | prod | best |")
    print("|--------|-------|-------:|--------:|--------:|---------------:|:----:|------|")
    for sh in HWC_SHAPES:
        img = np.ascontiguousarray(rng.integers(0, 256, size=sh, dtype=np.uint8))
        times = bench_hwc_shared(img, lut, r, w)
        uses_cv2 = _hwc_cv2_faster_shared(sh)
        prod = "cv2" if uses_cv2 else "SZ"
        best_key = min(times, key=times.__getitem__)
        best_label = "cv2" if "cv2" in best_key else "SZ"
        shape_str = "×".join(str(x) for x in sh)
        print(
            f"| HWC | {shape_str} | {img.size} | {times['cv2 new']:.4f} | {times['cv2→dst']:.4f} | "
            f"{times['SZ ravel+reuse']:.4f} | {prod} | {best_label} |",
        )

    # ------------------------------------------------------------------ DHWC/NDHWC shared
    print()
    print("## Section 2 — Shared LUT `(256,)` on `DHWC` / `NDHWC`")
    print()
    print("Production **always** uses SZ flat ravel. `cv2` columns are for comparison only.")
    print()
    print(env)
    print("| layout | shape | pixels | cv2 new | cv2→dst | SZ ravel+reuse [prod] | best |")
    print("|--------|-------|-------:|--------:|--------:|----------------------:|------|")
    for sh in VOLUME_SHAPES:
        img = np.ascontiguousarray(rng.integers(0, 256, size=sh, dtype=np.uint8))
        times = bench_volume_shared(img, lut, r, w)
        best_key = min(times, key=times.__getitem__)
        best_label = "cv2" if "cv2" in best_key else "SZ"
        shape_str = "×".join(str(x) for x in sh)
        print(
            f"| {layout_name(img.ndim)} | {shape_str} | {img.size} | "
            f"{times['cv2 new (not production)']:.4f} | {times['cv2→dst (not production)']:.4f} | "
            f"{times['SZ ravel+reuse [production]']:.4f} | {best_label} |",
        )

    # ------------------------------------------------------------------ HWC per-channel
    print()
    print("## Section 3 — Per-channel LUT `(C,256)` on `HWC`")
    print()
    print("Production uses **`cv2.LUT(img, (256,1,C))`** one-shot for C>1 contiguous HWC.")
    print()
    print(env)
    print("| layout | shape | C | pixels | cv2 new [prod C>1] | cv2→dst | SZ loop reuse ch | best |")
    print("|--------|-------|---|-------:|-------------------:|--------:|-----------------:|------|")
    for sh in HWC_SHAPES:
        c = sh[-1]
        lut_cv2, luts_1d = make_per_channel_luts(c)
        img = np.ascontiguousarray(rng.integers(0, 256, size=sh, dtype=np.uint8))
        times = bench_hwc_per_channel(img, lut_cv2, luts_1d, r, w)
        # key names differ for C=1 vs C>1 — normalise
        cv2_new_t = times.get("cv2 new [production]", times.get("cv2 new (C=1)", float("nan")))
        cv2_dst_t = times.get("cv2→dst", times.get("cv2→dst (C=1)", float("nan")))
        sz_t = times.get("SZ loop reuse ch", float("nan"))
        best_t = min(cv2_new_t, cv2_dst_t, sz_t)
        best_label = "SZ" if best_t == sz_t else ("cv2→dst" if best_t == cv2_dst_t else "cv2 new")
        shape_str = "×".join(str(x) for x in sh)
        print(f"| HWC | {shape_str} | {c} | {img.size} | {cv2_new_t:.4f} | {cv2_dst_t:.4f} | {sz_t:.4f} | {best_label} |")

    # ------------------------------------------------------------------ DHWC/NDHWC per-channel
    print()
    print("## Section 4 — Per-channel LUT `(C,256)` on `DHWC` / `NDHWC`")
    print()
    print("Production uses **SZ loop per channel** (cv2 `(256,1,C)` rejected for ndim>3).")
    print()
    print(env)
    print("| layout | shape | C | pixels | SZ loop reuse ch [prod] | cv2 flat (C=1 only) | best |")
    print("|--------|-------|---|-------:|------------------------:|--------------------:|------|")
    for sh in VOLUME_SHAPES:
        c = sh[-1]
        _, luts_1d = make_per_channel_luts(c)
        img = np.ascontiguousarray(rng.integers(0, 256, size=sh, dtype=np.uint8))
        times = bench_volume_per_channel(img, luts_1d, r, w)
        sz_t = times.get("SZ loop reuse ch [production]", float("nan"))
        cv2_t = times.get("cv2 flat 2D (C=1)", float("nan"))
        cv2_str = f"{cv2_t:.4f}" if c == 1 else "n/a (C>1)"
        if c == 1 and cv2_t == cv2_t:
            best_label = "SZ" if sz_t <= cv2_t else "cv2"
        else:
            best_label = "SZ [only option]"
        shape_str = "×".join(str(x) for x in sh)
        print(f"| {layout_name(img.ndim)} | {shape_str} | {c} | {img.size} | {sz_t:.4f} | {cv2_str} | {best_label} |")

    print()
    print("### Legend")
    print()
    print(
        "- `[production]`: path taken by albucore `apply_uint8_lut` for this shape.\n"
        "- `cv2 new`: `cv2.LUT(src, lut)` — allocates output each call.\n"
        "- `cv2→dst`: `cv2.LUT(src, lut, dst)` — reuses a preallocated output buffer.\n"
        "- `SZ ravel+reuse`: `copyto` into a preallocated flat buffer, then `translate(inplace=True)`.\n"
        "- `SZ loop reuse ch`: per-channel loop; one reused `H×W` plane + `translate(inplace=True)`.\n"
        "- `cv2 loop 2D slices`: cv2 per-channel via iterating 2D slices — not standard production.\n",
    )


if __name__ == "__main__":
    main()
