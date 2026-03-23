# GitHub issue — copy everything below the next line into a new issue

---

## Title

**Uint8 LUT: `cv2.LUT` vs StringZilla `translate` — full benchmark (shared + per-channel), allocation modes, volume shapes**

## Problem / motivation

Libraries and apps need **fast `uint8 → uint8` LUT** on **images (`HWC`)** and **volumes / batches (`DHWC`, `NDHWC`)**. OpenCV exposes **`cv2.LUT(src, lut[, dst])`**; StringZilla exposes **`translate`** on a buffer. There is **no single winner**: the gap flips with **shape**, **channel count**, **shared vs per-channel LUT**, and whether the caller **reuses output buffers**.

This issue collects a **self-contained benchmark script** (no external repo), **full tables**, and **action items** for what is slow and what we want improved.

## Action items (what is slow → what we want)

| Area | Observation (see tables below) | Ask |
|------|----------------------------------|-----|
| **Large HWC, high C, shared LUT** | e.g. `1024×1024×9`: StringZilla (full ravel) can be **~2× slower** than **`cv2.LUT(..., dst)`**. | **StringZilla:** improve throughput for **very large** contiguous `uint8` remap on **multi-channel** slabs **or** document when OpenCV is expected to win. |
| **Per-channel LUT on HWC** | **`cv2.LUT` one shot `(256,1,C)`** dominates; **Python loop + C × `translate`** is **orders of magnitude slower** (e.g. **~5.6 ms vs ~0.34 ms** median on `1024×1024×9` in this run). | **StringZilla:** **batched / multi-table** LUT API (one pass, channel axis) **or** documented recipe that matches OpenCV semantics without a Python loop. |
| **Volumes `DHWC` / `NDHWC`, shared LUT** | **`cv2.LUT` on the full array** is often **much slower** than **ravel + one `translate`** on the same data (same permutation LUT). | **OpenCV:** investigate **ndim > 3** / large tensor `LUT` path vs flat buffer remap **or** document recommended layout. |
| **Output allocation** | **`cv2.LUT(src, lut, dst)`** (reuse `dst`) is often **faster** than **`cv2.LUT(src, lut)`** (new array every call). | **Integrators:** reuse **`dst`** in hot paths; **docs** should mention it. |

## Protocol

- **Dtypes:** `uint8` image, `uint8` LUT only.
- **LUT:** fixed-seed **`permutation(256)`** (non-identity, full table).
- **Timers:** median wall ms, warmup + repeats (script defaults: 15 / 5).
- **OpenCV “in-place”:** OpenCV does **not** safely read `src` and write the same buffer for LUT; we benchmark **`cv2.LUT(src, lut, dst)`** with a **reused `dst`** vs **allocating** a new output each call.
- **StringZilla:** **`translate(..., inplace=True)`** on a **copy** of the ravel (`SZ copy+inplace`) vs **`copyto` into a reused flat buffer** then translate (`SZ reuse buf+inplace`).

## Full benchmark script (single file — save as `issue_lut_uint8_standalone.py`)

_Optional `# ruff: noqa` line exists in the albucore repo copy only; omitted here for pasting._

```python
#!/usr/bin/env python3
"""Standalone **uint8 → uint8 LUT** benchmark: **OpenCV `cv2.LUT`** vs **StringZilla `translate`**.

- **No albucore** — paste this single file anywhere with `numpy`, `opencv-python`, `stringzilla`.
- LUT is a fixed-seed **permutation** of `0..255` (non-trivial; not identity).
- Compares **allocation behavior**:
  - **cv2:** `LUT(src, lut)` (new array) vs `LUT(src, lut, dst)` (reuse preallocated `dst`).
  - **SZ:** fresh `ravel().copy()` each call vs **reuse** one flat buffer (`copyto` + `translate` inplace).

Also prints a **per-channel LUT** section for `HWC` only (`cv2` `(256,1,C)` vs SZ loop per channel).

Run: ``python issue_lut_uint8_standalone.py``
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

SHAPES: list[tuple[int, ...]] = [
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
    (32, 128, 128, 1),
    (64, 128, 128, 3),
    (128, 128, 128, 1),
    (48, 256, 256, 3),
    (96, 160, 160, 3),
    (6, 32, 32, 9),
    (2, 32, 128, 128, 3),
    (2, 64, 128, 128, 3),
    (1, 128, 128, 128, 1),
]


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


def shared_lut() -> np.ndarray:
    return np.random.default_rng(LUT_PERM_SEED).permutation(256).astype(np.uint8)


def per_channel_luts(c: int) -> tuple[np.ndarray, list[np.ndarray]]:
    lut_cv2 = np.zeros((256, 1, c), dtype=np.uint8)
    luts_1d: list[np.ndarray] = []
    for i in range(c):
        col = np.random.default_rng(LUT_PERM_SEED + 1000 + i).permutation(256).astype(np.uint8)
        lut_cv2[:, 0, i] = col
        luts_1d.append(col)
    return lut_cv2, luts_1d


def layout_name(ndim: int) -> str:
    return {3: "HWC", 4: "DHWC", 5: "NDHWC"}.get(ndim, "?")


def bench_shared_row(
    img: np.ndarray,
    lut: np.ndarray,
    repeats: int,
    warmup: int,
) -> tuple[float, float, float, float]:
    flat = img.reshape(-1)
    n = flat.size

    def cv2_new() -> None:
        cv2.LUT(img, lut)

    dst = np.empty_like(img, dtype=np.uint8)

    def cv2_dst() -> None:
        cv2.LUT(img, lut, dst)

    def sz_copy_inplace() -> None:
        b = flat.copy()
        sz.translate(memoryview(b), memoryview(lut), inplace=True)

    buf = np.empty(n, dtype=np.uint8)

    def sz_reuse_inplace() -> None:
        np.copyto(buf, flat)
        sz.translate(memoryview(buf), memoryview(lut), inplace=True)

    return (
        median_ms(cv2_new, repeats, warmup),
        median_ms(cv2_dst, repeats, warmup),
        median_ms(sz_copy_inplace, repeats, warmup),
        median_ms(sz_reuse_inplace, repeats, warmup),
    )


def bench_per_channel_row(
    img: np.ndarray,
    lut_cv2: np.ndarray,
    luts_1d: list[np.ndarray],
    repeats: int,
    warmup: int,
) -> tuple[float, float, float, float]:
    c = img.shape[-1]
    dst = np.empty_like(img, dtype=np.uint8)
    spatial = tuple(img.shape[:-1])
    out_sz = np.empty_like(img, dtype=np.uint8)
    ch_reuse = np.empty(spatial, dtype=np.uint8)

    def cv2_new() -> None:
        if c > 1:
            cv2.LUT(img, lut_cv2)
        else:
            cv2.LUT(img, luts_1d[0])

    def cv2_dst() -> None:
        if c > 1:
            cv2.LUT(img, lut_cv2, dst)
        else:
            cv2.LUT(img, luts_1d[0], dst)

    def sz_loop_new_ch_each() -> None:
        """New channel buffer allocation every channel (inside timed loop)."""
        for i in range(c):
            ch = np.ascontiguousarray(img[..., i], dtype=np.uint8).copy()
            sz.translate(memoryview(ch), memoryview(luts_1d[i]), inplace=True)
            out_sz[..., i] = ch

    def sz_loop_reuse_ch() -> None:
        """One reusable channel plane: copyto + translate inplace."""
        for i in range(c):
            np.copyto(ch_reuse, img[..., i])
            sz.translate(memoryview(ch_reuse), memoryview(luts_1d[i]), inplace=True)
            out_sz[..., i] = ch_reuse

    return (
        median_ms(cv2_new, repeats, warmup),
        median_ms(cv2_dst, repeats, warmup),
        median_ms(sz_loop_new_ch_each, repeats, warmup),
        median_ms(sz_loop_reuse_ch, repeats, warmup),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng_img = np.random.default_rng(args.seed)
    lut = shared_lut()

    print()
    print("## Shared `(256,)` uint8 LUT — `cv2.LUT` new vs `dst` vs StringZilla copy vs reuse buffer")
    print()
    try:
        sz_ver = importlib.metadata.version("stringzilla")
    except importlib.metadata.PackageNotFoundError:
        sz_ver = "?"
    print(
        f"_Median ms, repeats={args.repeats}, warmup={args.warmup}, image RNG seed={args.seed}; "
        f"{platform.system()} `{platform.machine()}`, OpenCV {cv2.__version__}, numpy {np.__version__}, "
        f"stringzilla {sz_ver}._",
    )
    print()
    print(
        "| layout | shape | pixels | cv2 new | cv2→dst | SZ copy+inplace | SZ reuse buf+inplace | best |",
    )
    print("|--------|-------|-------:|--------:|--------:|----------------:|-----------------------:|------|")

    for sh in SHAPES:
        img = np.ascontiguousarray(rng_img.integers(0, 256, size=sh, dtype=np.uint8))
        npx = int(img.size)
        t_cn, t_cd, t_sc, t_sr = bench_shared_row(img, lut, args.repeats, args.warmup)
        best_t = min(t_cn, t_cd, t_sc, t_sr)
        if best_t == t_cn:
            tag = "cv2 new"
        elif best_t == t_cd:
            tag = "cv2 dst"
        elif best_t == t_sc:
            tag = "SZ copy"
        else:
            tag = "SZ reuse"
        shape_str = "×".join(str(x) for x in sh)
        print(
            f"| {layout_name(img.ndim)} | {shape_str} | {npx} | {t_cn:.4f} | {t_cd:.4f} | "
            f"{t_sc:.4f} | {t_sr:.4f} | {tag} |",
        )

    print()
    print("## Per-channel LUT — `HWC` only (`cv2` `(256,1,C)` vs SZ loop per channel)")
    print()
    print(
        "| layout | shape | pixels | cv2 new | cv2→dst | SZ loop (new ch buf) | SZ loop (reuse ch buf) | best |",
    )
    print("|--------|-------|-------:|--------:|--------:|-------------------:|-------------------:|------|")

    for sh in SHAPES:
        if len(sh) != 3:
            continue
        c = sh[-1]
        lut_cv2, luts_1d = per_channel_luts(c)
        img = np.ascontiguousarray(rng_img.integers(0, 256, size=sh, dtype=np.uint8))
        npx = int(img.size)
        t_cn, t_cd, t_sa, t_sr = bench_per_channel_row(img, lut_cv2, luts_1d, args.repeats, args.warmup)
        best_t = min(t_cn, t_cd, t_sa, t_sr)
        if best_t == t_cn:
            tag = "cv2 new"
        elif best_t == t_cd:
            tag = "cv2 dst"
        elif best_t == t_sa:
            tag = "SZ new ch"
        else:
            tag = "SZ reuse ch"
        shape_str = "×".join(str(x) for x in sh)
        print(
            f"| HWC | {shape_str} | {npx} | {t_cn:.4f} | {t_cd:.4f} | {t_sa:.4f} | {t_sr:.4f} | {tag} |",
        )

    print()
    print("### Legend")
    print()
    print(
        "- **cv2 new:** `dst = cv2.LUT(src, lut)` — OpenCV allocates output each call.\n"
        "- **cv2→dst:** `cv2.LUT(src, lut, dst)` — same `dst` buffer reused (no output allocation).\n"
        "- **SZ copy+inplace:** `b = ravel().copy()` then `translate(..., inplace=True)` on `b` each call.\n"
        "- **SZ reuse buf+inplace:** one flat buffer; `copyto` from ravel then `translate` inplace.\n"
        "- **Per-channel SZ:** loop channels; **new ch buf** allocates a fresh channel copy each `c`; "
        "**reuse ch buf** uses one `H×W` workspace + `copyto`.\n",
    )


if __name__ == "__main__":
    main()
```

## Full benchmark tables (example run)

_Generated by pasting the script above and running `python issue_lut_uint8_standalone.py --repeats 15 --warmup 5`. **Your numbers will differ** by CPU, OS, OpenCV build, and StringZilla version._

**Reference environment for the tables below:** Darwin `arm64`, OpenCV **4.13.0**, NumPy **2.4.2**, StringZilla **4.6.0**, image RNG seed **0**.

### Shared `(256,)` uint8 LUT

| layout | shape | pixels | cv2 new | cv2→dst | SZ copy+inplace | SZ reuse buf+inplace | best |
|--------|-------|-------:|--------:|--------:|----------------:|-----------------------:|------|
| HWC | 128×128×1 | 16384 | 0.0041 | 0.0040 | 0.0017 | 0.0017 | SZ copy |
| HWC | 128×128×3 | 49152 | 0.0126 | 0.0126 | 0.0041 | 0.0046 | SZ copy |
| HWC | 256×256×1 | 65536 | 0.0167 | 0.0167 | 0.0059 | 0.0059 | SZ copy |
| HWC | 256×256×3 | 196608 | 0.0492 | 0.0492 | 0.0171 | 0.0160 | SZ reuse |
| HWC | 256×256×9 | 589824 | 0.1467 | 0.1446 | 0.0448 | 0.0451 | SZ copy |
| HWC | 512×512×1 | 262144 | 0.0325 | 0.0309 | 0.0202 | 0.0203 | SZ copy |
| HWC | 512×512×3 | 786432 | 0.0690 | 0.0685 | 0.0595 | 0.0593 | SZ reuse |
| HWC | 512×512×9 | 2359296 | 0.2070 | 0.1697 | 0.2321 | 0.1797 | cv2 dst |
| HWC | 1024×1024×1 | 1048576 | 0.0870 | 0.0796 | 0.0792 | 0.0793 | SZ copy |
| HWC | 1024×1024×3 | 3145728 | 0.2234 | 0.1585 | 0.3282 | 0.2392 | cv2 dst |
| HWC | 1024×1024×9 | 9437184 | 0.3278 | 0.3234 | 0.7404 | 0.7543 | cv2 dst |
| HWC | 96×96×9 | 82944 | 0.0190 | 0.0210 | 0.0083 | 0.0077 | SZ reuse |
| DHWC | 32×128×128×1 | 524288 | 0.1306 | 0.1306 | 0.0402 | 0.0427 | SZ copy |
| DHWC | 64×128×128×3 | 3145728 | 0.8562 | 0.7572 | 0.3162 | 0.2567 | SZ reuse |
| DHWC | 128×128×128×1 | 2097152 | 0.5707 | 0.5054 | 0.2076 | 0.1606 | SZ reuse |
| DHWC | 48×256×256×3 | 9437184 | 2.2788 | 2.2814 | 0.7108 | 0.7331 | SZ copy |
| DHWC | 96×160×160×3 | 7372800 | 2.0064 | 1.7805 | 0.8192 | 0.5712 | SZ reuse |
| DHWC | 6×32×32×9 | 55296 | 0.0143 | 0.0144 | 0.0048 | 0.0048 | SZ copy |
| NDHWC | 2×32×128×128×3 | 3145728 | 0.8529 | 0.7613 | 0.3374 | 0.2432 | SZ reuse |
| NDHWC | 2×64×128×128×3 | 6291456 | 1.7394 | 1.5159 | 0.6366 | 0.4914 | SZ reuse |
| NDHWC | 1×128×128×128×1 | 2097152 | 0.5676 | 0.5069 | 0.2231 | 0.1678 | SZ reuse |

### Per-channel LUT — `HWC` only

| layout | shape | pixels | cv2 new | cv2→dst | SZ loop (new ch buf) | SZ loop (reuse ch buf) | best |
|--------|-------|-------:|--------:|--------:|-------------------:|-------------------:|------|
| HWC | 128×128×1 | 16384 | 0.0041 | 0.0040 | 0.0022 | 0.0020 | SZ reuse ch |
| HWC | 128×128×3 | 49152 | 0.0127 | 0.0126 | 0.0318 | 0.0300 | cv2 dst |
| HWC | 256×256×1 | 65536 | 0.0160 | 0.0167 | 0.0075 | 0.0075 | SZ new ch |
| HWC | 256×256×3 | 196608 | 0.0493 | 0.0492 | 0.1180 | 0.1089 | cv2 dst |
| HWC | 256×256×9 | 589824 | 0.1468 | 0.1341 | 0.3358 | 0.3261 | cv2 dst |
| HWC | 512×512×1 | 262144 | 0.0327 | 0.0299 | 0.0235 | 0.0234 | SZ reuse ch |
| HWC | 512×512×3 | 786432 | 0.0692 | 0.0683 | 0.4994 | 0.4362 | cv2 dst |
| HWC | 512×512×9 | 2359296 | 0.2458 | 0.2119 | 1.3475 | 1.3140 | cv2 dst |
| HWC | 1024×1024×1 | 1048576 | 0.0860 | 0.0835 | 0.0905 | 0.0909 | cv2 dst |
| HWC | 1024×1024×3 | 3145728 | 0.2236 | 0.1364 | 1.8094 | 1.7416 | cv2 dst |
| HWC | 1024×1024×9 | 9437184 | 0.3356 | 0.3361 | 5.6373 | 5.1268 | cv2 new |
| HWC | 96×96×9 | 82944 | 0.0212 | 0.0215 | 0.0510 | 0.0484 | cv2 new |

## What to attach when you comment

- CPU model, RAM, OS
- `opencv-python` / `opencv` build info if relevant
- `stringzilla` version
- **Full stdout** of the script on your machine

---

_End of issue body._
