# ruff: noqa: INP001, T201
"""Benchmark the NumKong 7.8 buffer-first cast used by ``from_float``.

The complete candidates include float32 scaling, round-to-even, saturation,
output allocation, and the public router. Run from the repository root:

    uv run python benchmarks/benchmark_numkong_astype.py \
      --output /tmp/numkong-astype.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numkong as nk
import numpy as np
from shape_grids import ROUTER_HWC_FULL_HW
from timing import WallTimingMs, bench_wall_ms

from albucore.convert import _from_float_uint8_numkong, from_float_opencv

if TYPE_CHECKING:
    from albucore.utils import ImageFloat32, ImageUInt8

HWC_SHAPES: tuple[tuple[int, int, int], ...] = tuple(
    (height, width, channels) for height, width in ROUTER_HWC_FULL_HW for channels in (1, 3, 5, 9)
)
XHWC_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
)


def _numkong_direct_cast(rounded: ImageFloat32) -> ImageUInt8:
    """Cast a pre-rounded, in-range float32 buffer through NumKong."""
    result = np.empty(rounded.shape, dtype=np.uint8)
    nk.astype(rounded, "uint8", out=result)
    return cast("ImageUInt8", result)


def _numpy_direct_cast(rounded: ImageFloat32) -> ImageUInt8:
    """Reference cast for already rounded values in the uint8 range."""
    return cast("ImageUInt8", rounded.astype(np.uint8))


def _strided(image: ImageFloat32) -> ImageFloat32:
    expanded = np.empty((image.shape[0], image.shape[1] * 2, *image.shape[2:]), dtype=np.float32)
    expanded[:, ::2, ...] = image
    expanded[:, 1::2, ...] = image
    return cast("ImageFloat32", expanded[:, ::2, ...])


def _format(timing: WallTimingMs) -> str:
    return f"{timing.median:.4f} +/- {timing.mad:.4f}"


def _row(
    layout: str,
    image: ImageFloat32,
    repeats: int,
    warmup: int,
) -> str:
    expected = from_float_opencv(image, np.dtype(np.uint8), 255.0)
    candidate = _from_float_uint8_numkong(image, 255.0)
    np.testing.assert_array_equal(candidate, expected, strict=True)

    rounded = np.rint(image * np.float32(255.0))
    rounded = np.clip(rounded, np.float32(0.0), np.float32(255.0))
    np.testing.assert_array_equal(_numkong_direct_cast(rounded), _numpy_direct_cast(rounded), strict=True)

    numpy_full = bench_wall_ms(
        lambda: from_float_opencv(image, np.dtype(np.uint8), 255.0),
        repeats=repeats,
        warmup=warmup,
    )
    numkong_full = bench_wall_ms(lambda: _from_float_uint8_numkong(image, 255.0), repeats=repeats, warmup=warmup)
    numpy_cast = bench_wall_ms(lambda: _numpy_direct_cast(rounded), repeats=repeats, warmup=warmup)
    numkong_cast = bench_wall_ms(lambda: _numkong_direct_cast(rounded), repeats=repeats, warmup=warmup)
    return (
        f"| {layout} | {'x'.join(map(str, image.shape))} | {_format(numpy_full)} | "
        f"{_format(numkong_full)} | {numkong_full.median / numpy_full.median:.3f}x | "
        f"{_format(numpy_cast)} | {_format(numkong_cast)} | {numkong_cast.median / numpy_cast.median:.3f}x |"
    )


def _report(repeats: int, warmup: int) -> str:
    rng = np.random.default_rng(368)
    run_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    platform_name = platform.platform()
    rows = [
        "| layout | shape | NumPy rint+clip ms | NumKong full ms | NK/NumPy | "
        "NumPy cast ms | NumKong cast ms | NK/NumPy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for shape in HWC_SHAPES:
        contiguous = rng.uniform(np.float32(-0.2), np.float32(1.2), size=shape).astype(np.float32)
        rows.append(_row("HWC contiguous", contiguous, repeats, warmup))
        rows.append(_row("HWC strided", _strided(contiguous), repeats, warmup))
    for shape in XHWC_SHAPES:
        contiguous = rng.uniform(np.float32(-0.2), np.float32(1.2), size=shape).astype(np.float32)
        rows.append(_row("XHWC contiguous", contiguous, repeats, warmup))
        rows.append(_row("XHWC strided", _strided(contiguous), repeats, warmup))

    return "\n".join(
        (
            "# NumKong buffer-first cast benchmark",
            "",
            "`nk.astype(..., out=...)` receives an arbitrary-stride float32 source and writes a new C-contiguous "
            "uint8 output. The full candidate uses one float32 working buffer for scale+round, then relies on "
            "NumKong's round-to-even saturating cast; it is exact against Albucore's existing `rint`+clip route.",
            "",
            f"Run date: {run_date}. Platform: `{platform_name}`. Python repeats: {repeats}; warmup: {warmup}.",
            "",
            f"Versions: NumPy `{np.__version__}`, NumKong `{nk.__version__}`.",
            "",
            *rows,
            "",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _report(args.repeats, args.warmup)
    if args.output is None:
        print(report)
        return
    args.output.write_text(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
