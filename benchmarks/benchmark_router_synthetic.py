#!/usr/bin/env python3
"""Time **public routers** on synthetic `(H,W,C)` / `(N,H,W,C)` arrays (uint8, float32).

Run from repo root with the env that should provide ``albucore`` (editable or pinned wheel)::

    uv run python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/run.json

Compare releases::

    uv run --no-project --with albucore==0.0.40 --with opencv-python-headless \\
      --with simsimd --with stringzilla --with numpy \\
      python benchmarks/benchmark_router_synthetic.py --output-json /tmp/old.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from timing import median_ms


@dataclass
class BenchRow:
    op: str
    layout: str
    shape: tuple[int, ...]
    dtype: str
    ms_median: float | None
    status: str
    detail: str = ""


def _make_img(rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _iter_hwc(quick: bool) -> Iterator[tuple[str, tuple[int, ...], np.dtype]]:
    """3D images only — OpenCV routers expect ``(H,W,C)``."""
    channels = (1, 3) if quick else (1, 3, 9)
    sizes = ((128, 128), (256, 256)) if quick else ((128, 128), (256, 256), (512, 512))
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                yield "HWC", (h, w, c), dtype


def _iter_batch_stats(quick: bool) -> Iterator[tuple[str, tuple[int, ...], np.dtype]]:
    """4D batch — only meaningful for ``stats`` reductions in albucore."""
    channels = (1, 3) if quick else (1, 3, 9)
    sizes = ((256, 256),) if quick else ((128, 128), (256, 256))
    n = 4
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                yield "NHWC", (n, h, w, c), dtype


def _bench(
    alb: Any,
    img: np.ndarray,
    build: Callable[[Any, np.ndarray], Callable[[], object]],
    repeats: int,
    warmup: int,
) -> tuple[float | None, str, str]:
    try:
        fn = build(alb, img)
        ms = median_ms(fn, repeats=repeats, warmup=warmup)
        return ms, "ok", ""
    except Exception as e:  # noqa: BLE001 — benchmark harness
        return None, "error", f"{type(e).__name__}: {e}"


def _registry() -> list[tuple[str, Callable[[Any, np.ndarray], Callable[[], object]]]]:
    """(api_name, build(alb, img) -> thunk)."""

    def add_c(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 3 if img.dtype == np.uint8 else 1.5

        def thunk() -> None:
            alb.add(img, v)

        return thunk

    def mul_c(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 1.05 if img.dtype == np.float32 else 2

        def thunk() -> None:
            alb.multiply(img, v)

        return thunk

    def aw(alb: Any, img: np.ndarray) -> Callable[[], object]:
        img2 = (img // 2) if img.dtype == np.uint8 else (img * 0.5)

        def thunk() -> None:
            alb.add_weighted(img, 0.5, img2.astype(img.dtype, copy=False), 0.5)

        return thunk

    def pw(alb: Any, img: np.ndarray) -> Callable[[], object]:
        exp = np.float32(1.01) if img.dtype == np.float32 else np.array([1.01] * img.shape[-1], dtype=np.float32)

        def thunk() -> None:
            alb.power(img, exp, False)

        return thunk

    def ma(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.multiply_add(img, 1.0, 0.0, False)

        return thunk

    def hf(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.hflip(img)

        return thunk

    def vf(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.vflip(img)

        return thunk

    def npi(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.normalize_per_image(img, "image")

        return thunk

    def norm(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.normalize(img, 0.0, 1.0)

        return thunk

    def tf(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.to_float(img)

        return thunk

    def ff(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.from_float(img, np.uint8)

        return thunk

    def mb(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.median_blur(img, 3)

        return thunk

    def lut(alb: Any, img: np.ndarray) -> Callable[[], object]:
        lut = np.arange(256, dtype=np.uint8)

        def thunk() -> None:
            alb.sz_lut(img, lut, True)

        return thunk

    def cmb(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.copy_make_border(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)

        return thunk

    def rsz(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        nw, nh = max(w // 2, 1), max(h // 2, 1)

        def thunk() -> None:
            alb.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        return thunk

    def waff(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def thunk() -> None:
            alb.warp_affine(img, m, (w, h), flags=cv2.INTER_LINEAR)

        return thunk

    def wper(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        m = np.eye(3, dtype=np.float32)

        def thunk() -> None:
            alb.warp_perspective(img, m, (w, h), flags=cv2.INTER_LINEAR)

        return thunk

    def rmp(alb: Any, img: np.ndarray) -> Callable[[], object]:
        h, w = img.shape[-3], img.shape[-2]
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        def thunk() -> None:
            alb.remap(img, mx, my, interpolation=cv2.INTER_LINEAR)

        return thunk

    def mn(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.mean(img, "global")

        return thunk

    def st(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.std(img, "global")

        return thunk

    def ms(alb: Any, img: np.ndarray) -> Callable[[], object]:
        def thunk() -> None:
            alb.mean_std(img, "global")

        return thunk

    return [
        ("add", add_c),
        ("multiply", mul_c),
        ("add_weighted", aw),
        ("power", pw),
        ("multiply_add", ma),
        ("hflip", hf),
        ("vflip", vf),
        ("normalize_per_image", npi),
        ("normalize", norm),
        ("to_float", tf),
        ("from_float", ff),
        ("median_blur", mb),
        ("sz_lut", lut),
        ("copy_make_border", cmb),
        ("resize", rsz),
        ("warp_affine", waff),
        ("warp_perspective", wper),
        ("remap", rmp),
        ("mean", mn),
        ("std", st),
        ("mean_std", ms),
    ]


def _matmul_bench(alb: Any, repeats: int, warmup: int) -> BenchRow:
    rng = np.random.default_rng(0)
    a = rng.random((128, 64), dtype=np.float32)
    b = rng.random((64, 32), dtype=np.float32)

    def thunk() -> None:
        alb.matmul(a, b)

    try:
        ms = median_ms(thunk, repeats=repeats, warmup=warmup)
        return BenchRow("matmul", "2D", a.shape + b.shape, "float32", ms, "ok")
    except Exception as e:  # noqa: BLE001
        return BenchRow("matmul", "2D", a.shape + b.shape, "float32", None, "error", f"{type(e).__name__}: {e}")


def _pairwise_bench(alb: Any, repeats: int, warmup: int) -> BenchRow:
    rng = np.random.default_rng(1)
    x1 = rng.random((24, 3), dtype=np.float32)
    x2 = rng.random((16, 3), dtype=np.float32)

    def thunk() -> None:
        alb.pairwise_distances_squared(x1, x2)

    try:
        ms = median_ms(thunk, repeats=repeats, warmup=warmup)
        return BenchRow(
            "pairwise_distances_squared",
            "points",
            (x1.shape[0], x2.shape[0], 3),
            "float32",
            ms,
            "ok",
        )
    except Exception as e:  # noqa: BLE001
        return BenchRow(
            "pairwise_distances_squared",
            "points",
            (x1.shape[0], x2.shape[0], 3),
            "float32",
            None,
            "error",
            f"{type(e).__name__}: {e}",
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-json", type=str, default="", help="Write full results JSON")
    p.add_argument("--quick", action="store_true", help="Smaller shape grid")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    import albucore as alb  # noqa: PLC0415 — after argparse for isolated runs

    ver = getattr(alb, "__version__", "unknown")
    try:
        dist_ver = importlib.metadata.version("albucore")
    except importlib.metadata.PackageNotFoundError:
        dist_ver = ver

    rows: list[BenchRow] = []
    rng = np.random.default_rng(42)
    reg = _registry()

    stats_ops = {"mean", "std", "mean_std"}

    for layout, shape, dtype in _iter_hwc(args.quick):
        img = _make_img(rng, shape, dtype)
        dname = "uint8" if dtype == np.uint8 else "float32"
        for op_name, build in reg:
            if op_name == "to_float" and dtype != np.uint8:
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs uint8"))
                continue
            if op_name == "from_float" and dtype != np.float32:
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs float32"))
                continue
            if op_name in ("median_blur", "sz_lut") and dtype != np.uint8:
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs uint8"))
                continue
            if not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            ms, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(BenchRow(op_name, layout, shape, dname, ms, st, det))

    for layout, shape, dtype in _iter_batch_stats(args.quick):
        img = _make_img(rng, shape, dtype)
        dname = "uint8" if dtype == np.uint8 else "float32"
        for op_name, build in reg:
            if op_name not in stats_ops:
                continue
            if not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            ms, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(BenchRow(op_name, layout, shape, dname, ms, st, det))

    rows.append(_matmul_bench(alb, args.repeats, args.warmup))
    rows.append(_pairwise_bench(alb, args.repeats, args.warmup))

    meta = {
        "albucore_version": ver,
        "distribution_version": dist_ver,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "quick": args.quick,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    payload = {"meta": meta, "rows": [asdict(r) for r in rows]}

    txt = json.dumps(payload, indent=2)
    if args.output_json:
        path = args.output_json
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"Wrote {path}", file=sys.stderr)
    else:
        print(txt)


if __name__ == "__main__":
    main()
