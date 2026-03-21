#!/usr/bin/env python3
"""Time routers listed in ``albucore.functions.__all__`` on synthetic arrays.

Layouts:

- **HWC** ``(H,W,C)`` for spatial OpenCV-backed ops (and most arithmetic).
- **NHWC** ``(N,H,W,C)`` for ``mean`` / ``std`` / ``mean_std`` only (stats reductions).

**1024 x 1024** is included in the default (non-``--quick``) HWC grid.

Optional ``--with-geometric`` also times ``copy_make_border``, ``resize``, ``warp_affine``,
``warp_perspective``, ``remap`` (they live in ``albucore.geometric``, not ``functions.__all__``).

Run::

    uv run python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/run.json

Reliability: default ``--repeats`` / ``--warmup`` are tuned for stable medians; JSON includes
``ms_std`` / ``ms_mad`` (spread of timed runs). ``sz_lut`` uses ``inplace=False`` so the shared
bench image is not corrupted across iterations.
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

from timing import WallTimingMs, bench_wall_ms

# When ``albucore.functions`` has no ``__all__`` (e.g. albucore 0.0.40), use this so compare
# runs time the same router names. Keep in sync with ``albucore/functions.py`` ``__all__``.
_FUNCTIONS_PUBLIC_ROUTERS_FALLBACK: tuple[str, ...] = (
    "add",
    "add_array",
    "add_constant",
    "add_vector",
    "add_weighted",
    "float32_io",
    "from_float",
    "hflip",
    "matmul",
    "mean",
    "mean_std",
    "median_blur",
    "multiply",
    "multiply_add",
    "multiply_by_array",
    "multiply_by_constant",
    "multiply_by_vector",
    "normalize",
    "normalize_per_image",
    "pairwise_distances_squared",
    "power",
    "std",
    "sz_lut",
    "to_float",
    "uint8_io",
    "vflip",
)


@dataclass
class BenchRow:
    op: str
    layout: str
    shape: tuple[int, ...]
    dtype: str
    ms_median: float | None
    status: str
    detail: str = ""
    ms_mean: float | None = None
    ms_std: float | None = None
    ms_mad: float | None = None
    timing_n: int | None = None


def _make_img(rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32)


def _iter_hwc(quick: bool) -> Iterator[tuple[str, tuple[int, ...], np.dtype]]:
    """3D images — sizes include 1024 in full mode."""
    channels = (1, 3) if quick else (1, 3, 9)
    if quick:
        sizes = ((128, 128), (256, 256))
    else:
        sizes = ((128, 128), (256, 256), (512, 512), (1024, 1024))
    for h, w in sizes:
        for c in channels:
            for dtype in (np.uint8, np.float32):
                yield "HWC", (h, w, c), dtype


def _iter_batch_stats(quick: bool) -> Iterator[tuple[str, tuple[int, ...], np.dtype]]:
    """4D batch for stats only. (No 1024 batch grid: memory.)"""
    channels = (1, 3) if quick else (1, 3, 9)
    n = 4
    if quick:
        sizes = ((256, 256),)
    else:
        sizes = ((128, 128), (256, 256), (512, 512))
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
) -> tuple[WallTimingMs | None, str, str]:
    try:
        fn = build(alb, img)
        return bench_wall_ms(fn, repeats=repeats, warmup=warmup), "ok", ""
    except Exception as e:  # noqa: BLE001 — benchmark harness
        return None, "error", f"{type(e).__name__}: {e}"


def _bench_row(
    op_name: str,
    layout: str,
    shape: tuple[int, ...],
    dname: str,
    timing: WallTimingMs | None,
    st: str,
    det: str,
) -> BenchRow:
    if timing is not None and st == "ok":
        return BenchRow(
            op_name,
            layout,
            shape,
            dname,
            timing.median,
            st,
            det,
            timing.mean,
            timing.std,
            timing.mad,
            timing.n,
        )
    return BenchRow(op_name, layout, shape, dname, None, st, det)


def _channel_vector(img: np.ndarray) -> np.ndarray:
    c = img.shape[-1]
    if img.dtype == np.uint8:
        return np.full(c, 2, dtype=np.uint8)
    return np.linspace(0.98, 1.02, num=c, dtype=np.float32)


def _registry_geometric() -> list[tuple[str, Callable[[Any, np.ndarray], Callable[[], object]]]]:
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

    return [
        ("copy_make_border", cmb),
        ("resize", rsz),
        ("warp_affine", waff),
        ("warp_perspective", wper),
        ("remap", rmp),
    ]


def _registry_functions() -> list[tuple[str, Callable[[Any, np.ndarray], Callable[[], object]]]]:
    """``albucore.functions.__all__`` routers (except decorators handled separately)."""

    def add_c(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 3 if img.dtype == np.uint8 else 1.5

        def thunk() -> None:
            alb.add(img, v)

        return thunk

    def add_const(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 3 if img.dtype == np.uint8 else 1.5

        def thunk() -> None:
            alb.add_constant(img, v, False)

        return thunk

    def add_vec(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = _channel_vector(img)

        def thunk() -> None:
            alb.add_vector(img, v, False)

        return thunk

    def add_arr(alb: Any, img: np.ndarray) -> Callable[[], object]:
        noise = (np.ones_like(img) * (2 if img.dtype == np.uint8 else 0.01)).astype(img.dtype, copy=False)

        def thunk() -> None:
            alb.add_array(img, noise, False)

        return thunk

    def mul_c(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 1.05 if img.dtype == np.float32 else 2

        def thunk() -> None:
            alb.multiply(img, v)

        return thunk

    def mul_const(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = 1.05 if img.dtype == np.float32 else 2

        def thunk() -> None:
            alb.multiply_by_constant(img, v, False)

        return thunk

    def mul_vec(alb: Any, img: np.ndarray) -> Callable[[], object]:
        v = _channel_vector(img)
        c = int(img.shape[-1])

        def thunk() -> None:
            alb.multiply_by_vector(img, v, c, False)

        return thunk

    def mul_arr(alb: Any, img: np.ndarray) -> Callable[[], object]:
        # ``multiply_opencv`` promotes uint8 image to float32; value must match OpenCV expectations.
        if img.dtype == np.float32:
            factor = np.full(img.shape, np.float32(1.02), dtype=np.float32)
        else:
            factor = np.full(img.shape, np.float32(1.02), dtype=np.float32)

        def thunk() -> None:
            alb.multiply_by_array(img, factor)

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
        lut_u8 = np.arange(256, dtype=np.uint8)

        def thunk() -> None:
            # inplace=True would mutate ``img`` across warmup/repeats and distort timings + semantics
            alb.sz_lut(img, lut_u8, False)

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

    # Order follows functions.__all__ (geometric optional elsewhere)
    return [
        ("add", add_c),
        ("add_array", add_arr),
        ("add_constant", add_const),
        ("add_vector", add_vec),
        ("add_weighted", aw),
        ("multiply", mul_c),
        ("multiply_add", ma),
        ("multiply_by_array", mul_arr),
        ("multiply_by_constant", mul_const),
        ("multiply_by_vector", mul_vec),
        ("normalize", norm),
        ("normalize_per_image", npi),
        ("hflip", hf),
        ("vflip", vf),
        ("median_blur", mb),
        ("matmul", add_c),  # placeholder — replaced by special-case
        ("pairwise_distances_squared", add_c),
        ("power", pw),
        ("mean", mn),
        ("mean_std", ms),
        ("std", st),
        ("sz_lut", lut),
        ("to_float", tf),
        ("from_float", ff),
    ]


def _matmul_bench(alb: Any, repeats: int, warmup: int) -> BenchRow:
    rng = np.random.default_rng(0)
    a = rng.random((128, 64), dtype=np.float32)
    b = rng.random((64, 32), dtype=np.float32)

    def thunk() -> None:
        alb.matmul(a, b)

    try:
        t = bench_wall_ms(thunk, repeats=repeats, warmup=warmup)
        return BenchRow(
            "matmul",
            "2D",
            (128, 64, 64, 32),
            "float32",
            t.median,
            "ok",
            "",
            t.mean,
            t.std,
            t.mad,
            t.n,
        )
    except Exception as e:  # noqa: BLE001
        return BenchRow("matmul", "2D", (128, 64, 64, 32), "float32", None, "error", f"{type(e).__name__}: {e}")


def _pairwise_bench(alb: Any, repeats: int, warmup: int) -> BenchRow:
    rng = np.random.default_rng(1)
    x1 = rng.random((24, 3), dtype=np.float32)
    x2 = rng.random((16, 3), dtype=np.float32)

    def thunk() -> None:
        alb.pairwise_distances_squared(x1, x2)

    try:
        t = bench_wall_ms(thunk, repeats=repeats, warmup=warmup)
        return BenchRow(
            "pairwise_distances_squared",
            "points",
            (x1.shape[0], x2.shape[0], 3),
            "float32",
            t.median,
            "ok",
            "",
            t.mean,
            t.std,
            t.mad,
            t.n,
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
    p.add_argument("--quick", action="store_true", help="Smaller shape/channel grid (no 1024 HWC)")
    p.add_argument(
        "--with-geometric",
        action="store_true",
        help="Also benchmark geometric routers (not in functions.__all__)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=21,
        help="Timed iterations per cell (median + spread in JSON)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed iterations before measurement (JIT, caches, alloc)",
    )
    args = p.parse_args()

    import albucore as alb  # noqa: PLC0415
    import albucore.functions as alb_fn  # noqa: PLC0415

    ver = getattr(alb, "__version__", "unknown")
    try:
        dist_ver = importlib.metadata.version("albucore")
    except importlib.metadata.PackageNotFoundError:
        dist_ver = ver

    rows: list[BenchRow] = []
    rng = np.random.default_rng(42)
    fn_reg = _registry_functions()
    # Drop placeholder entries for matmul / pairwise (special benches)
    fn_reg = [(n, b) for n, b in fn_reg if n not in ("matmul", "pairwise_distances_squared")]
    stats_ops = {"mean", "std", "mean_std"}
    skip_uint8_only = {"median_blur", "sz_lut", "to_float"}
    skip_float_only = {"from_float"}

    declared = getattr(alb_fn, "__all__", None)
    if declared is not None:
        export_names = set(declared)
        functions_all_meta: list[str] = list(declared)
    else:
        export_names = set(_FUNCTIONS_PUBLIC_ROUTERS_FALLBACK)
        functions_all_meta = list(_FUNCTIONS_PUBLIC_ROUTERS_FALLBACK)

    for name in sorted(export_names):
        if name in ("float32_io", "uint8_io"):
            rows.append(
                BenchRow(name, "meta", (-1,), "n/a", None, "skip", "decorator factory, not an image router"),
            )

    geo_reg = _registry_geometric() if args.with_geometric else []

    for layout, shape, dtype in _iter_hwc(args.quick):
        img = _make_img(rng, shape, dtype)
        dname = "uint8" if dtype == np.uint8 else "float32"
        for op_name, build in fn_reg:
            if op_name not in export_names:
                continue
            if op_name in stats_ops:
                continue
            if op_name in skip_uint8_only and dtype != np.uint8:
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs uint8"))
                continue
            if op_name in skip_float_only and dtype != np.float32:
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "needs float32"))
                continue
            if not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            t, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(_bench_row(op_name, layout, shape, dname, t, st, det))

        for op_name, build in geo_reg:
            if op_name in skip_uint8_only and dtype != np.uint8:
                continue
            if op_name in skip_float_only and dtype != np.float32:
                continue
            if not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            t, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(_bench_row(op_name, layout, shape, dname, t, st, det))

    for layout, shape, dtype in _iter_batch_stats(args.quick):
        img = _make_img(rng, shape, dtype)
        dname = "uint8" if dtype == np.uint8 else "float32"
        for op_name, build in fn_reg:
            if op_name not in stats_ops or op_name not in export_names:
                continue
            if not hasattr(alb, op_name):
                rows.append(BenchRow(op_name, layout, shape, dname, None, "skip", "missing API"))
                continue
            t, st, det = _bench(alb, img, build, args.repeats, args.warmup)
            rows.append(_bench_row(op_name, layout, shape, dname, t, st, det))

    if "matmul" in export_names:
        rows.append(_matmul_bench(alb, args.repeats, args.warmup))
    if "pairwise_distances_squared" in export_names:
        rows.append(_pairwise_bench(alb, args.repeats, args.warmup))

    meta = {
        "albucore_version": ver,
        "distribution_version": dist_ver,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "quick": args.quick,
        "with_geometric": args.with_geometric,
        "functions_all": functions_all_meta,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "timing_fields": {
            "ms_median": "median wall time (ms) over timed runs",
            "ms_mean": "mean wall time (ms)",
            "ms_std": "sample std (ms) of timed runs; use as error bar with median",
            "ms_mad": "median absolute deviation from median (robust spread)",
            "timing_n": "number of timed runs (equals repeats)",
        },
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
