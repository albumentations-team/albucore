#!/usr/bin/env python3
"""Compare semantic CPU Torch candidates with Albucore's established CPU backends.

The benchmark deliberately includes ``torch.from_numpy`` and ``Tensor.numpy`` in
every Torch timing. Compatible CPU arrays share storage at both boundaries, so
the comparison measures the path that an Albucore router would execute rather
than a pre-wrapped Tensor kernel. It excludes host-to-device transfers.

Run from the repository root::

    uv run python benchmarks/benchmark_torch_cpu.py --threads 1
    uv run python benchmarks/benchmark_torch_cpu.py --threads 12
    uv run python benchmarks/benchmark_torch_cpu.py --full --output /tmp/torch-cpu.md

The report marks numerically different candidates as ``mismatch``. Those rows
remain useful for investigation, but they are not routing candidates.

By default, the benchmark measures Albucore's configured production router.
Pass ``--disable-public-torch-route`` to measure the prior NumPy, OpenCV,
NumKong, or LUT baseline for a direct comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import platform
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f
from shape_grids import ROUTER_HWC_FULL_HW, ROUTER_HWC_QUICK_HW, SCALE_LUT_SHAPES
from timing import WallTimingMs, bench_wall_ms

import albucore
from albucore import torch_backend

ArrayResult = np.ndarray | float | tuple[np.ndarray | float, np.ndarray | float]
PreparedBenchmark = tuple[Callable[[], ArrayResult], Callable[[], ArrayResult]]


@contextmanager
def _torch_router_disabled() -> Any:
    """Temporarily compare a direct Torch kernel with Albucore's prior CPU backend.

    The benchmark uses Torch to time the candidate, which would otherwise make
    Albucore's runtime router eligible too. This process-wide patch is
    safe because benchmark timing is single-process and sequential.
    """
    original = torch_backend.TORCH_CPU_BACKEND_ENABLED
    torch_backend.TORCH_CPU_BACKEND_ENABLED = False
    try:
        yield
    finally:
        torch_backend.TORCH_CPU_BACKEND_ENABLED = original


@dataclass(frozen=True, slots=True)
class Candidate:
    """One public Albucore call and its complete CPU Torch alternative."""

    name: str
    dtypes: tuple[np.dtype[Any], ...]
    prepare: Callable[[np.ndarray, np.random.Generator], PreparedBenchmark]
    rtol: float = 1e-5
    atol: float = 1e-6


@dataclass(frozen=True, slots=True)
class Row:
    candidate: str
    shape: tuple[int, ...]
    dtype: str
    correctness: str
    albucore_ms: float
    torch_ms: float

    @property
    def winner(self) -> str:
        if self.correctness != "ok":
            return "semantic mismatch"
        return "torch" if self.torch_ms < self.albucore_ms else "albucore"

    @property
    def speedup(self) -> float:
        return self.albucore_ms / self.torch_ms if self.torch_ms else math.inf


def _torch_array(tensor: torch.Tensor) -> np.ndarray:
    """Return a NumPy view of a CPU Torch result, raising if a copy is needed."""
    assert tensor.device.type == "cpu"  # noqa: S101 - benchmark contract
    return tensor.numpy()


def _torch_last_axis_vector(vector: np.ndarray, ndim: int) -> torch.Tensor:
    return torch.from_numpy(vector).reshape((1,) * (ndim - 1) + (-1,))


def _assert_equal(expected: ArrayResult, actual: ArrayResult, rtol: float, atol: float) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)  # noqa: S101 - benchmark contract
        for expected_part, actual_part in zip(expected, actual, strict=True):
            _assert_equal(expected_part, actual_part, rtol, atol)
        return
    if isinstance(expected, (np.ndarray, np.generic)):
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        if expected_array.dtype == np.uint8 or expected_array.dtype == np.uint64:
            np.testing.assert_array_equal(actual_array, expected_array)
        else:
            np.testing.assert_allclose(actual_array, expected_array, rtol=rtol, atol=atol, equal_nan=True)
        return
    assert not isinstance(actual, np.ndarray)  # noqa: S101 - benchmark contract
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def _float_image(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Positive finite inputs exercise the OpenCV-compatible public log route.
    return rng.uniform(np.float32(0.05), np.float32(1.0), size=shape).astype(np.float32)


def _uint8_image(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _prepare_wrap(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return lambda: array, lambda: _torch_array(torch.from_numpy(array))


def _prepare_exp(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return lambda: albucore.exp(array), lambda: _torch_array(torch.exp(torch.from_numpy(array)))


def _prepare_log(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return lambda: albucore.log(array), lambda: _torch_array(torch.log(torch.from_numpy(array)))


def _prepare_sqrt(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return lambda: albucore.sqrt(array), lambda: _torch_array(torch.sqrt(torch.from_numpy(array)))


def _prepare_add(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    value = np.float32(0.125)
    return lambda: albucore.add(array, value), lambda: _torch_array(torch.add(torch.from_numpy(array), value))


def _prepare_multiply(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    value = np.float32(1.125)
    return lambda: albucore.multiply(array, value), lambda: _torch_array(torch.mul(torch.from_numpy(array), value))


def _prepare_power(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    value = np.float32(0.88)
    return lambda: albucore.power(array, value), lambda: _torch_array(torch.pow(torch.from_numpy(array), value))


def _prepare_multiply_add(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    factor, value = np.float32(1.125), np.float32(-0.05)
    return (
        lambda: albucore.multiply_add(array, factor, value),
        lambda: _torch_array(torch.add(torch.mul(torch.from_numpy(array), factor), value)),
    )


def _prepare_add_weighted(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    other = np.ascontiguousarray(array * np.float32(0.7))
    weight1, weight2 = np.float32(0.4), np.float32(0.6)
    return (
        lambda: albucore.add_weighted(array, weight1, other, weight2),
        lambda: _torch_array(
            torch.add(torch.mul(torch.from_numpy(array), weight1), torch.from_numpy(other), alpha=weight2),
        ),
    )


def _prepare_normalize(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    channels = array.shape[-1]
    mean = np.linspace(0.3, 0.5, channels, dtype=np.float32)
    denominator = np.linspace(1.25, 1.75, channels, dtype=np.float32)
    mean_t = _torch_last_axis_vector(mean, array.ndim)
    denominator_t = _torch_last_axis_vector(denominator, array.ndim)
    offset_t = -mean_t * denominator_t
    return (
        lambda: albucore.normalize(array, mean, denominator),
        lambda: _torch_array(torch.addcmul(offset_t, torch.from_numpy(array), denominator_t)),
    )


def _prepare_mean_std_global(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    def torch_candidate() -> tuple[float, float]:
        variance, mean = torch.var_mean(torch.from_numpy(array), correction=0)
        return float(mean), float(torch.sqrt(variance) + 1e-4)

    return lambda: albucore.mean_std(array, "global"), torch_candidate


def _prepare_mean_std_per_channel(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    axes = tuple(range(array.ndim - 1))

    def torch_candidate() -> tuple[np.ndarray, np.ndarray]:
        variance, mean = torch.var_mean(torch.from_numpy(array), dim=axes, correction=0)
        return _torch_array(mean), _torch_array(torch.sqrt(variance) + 1e-4)

    return lambda: albucore.mean_std(array, "per_channel"), torch_candidate


def _prepare_reduce_sum_global_float(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return (
        lambda: albucore.reduce_sum(array, "global"),
        lambda: _torch_array(torch.sum(torch.from_numpy(array), dtype=torch.float64)),
    )


def _prepare_reduce_sum_per_channel_float(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    axes = tuple(range(array.ndim - 1))
    return (
        lambda: albucore.reduce_sum(array, "per_channel"),
        lambda: _torch_array(torch.sum(torch.from_numpy(array), dim=axes, dtype=torch.float64)),
    )


def _uint64_view(array: np.ndarray) -> np.ndarray:
    return array.view(np.uint64)


def _prepare_reduce_sum_global_uint8(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return (
        lambda: albucore.reduce_sum(array, "global"),
        lambda: _uint64_view(_torch_array(torch.sum(torch.from_numpy(array), dtype=torch.int64))),
    )


def _prepare_reduce_sum_per_channel_uint8(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    axes = tuple(range(array.ndim - 1))
    return (
        lambda: albucore.reduce_sum(array, "per_channel"),
        lambda: _uint64_view(_torch_array(torch.sum(torch.from_numpy(array), dim=axes, dtype=torch.int64))),
    )


def _prepare_hflip(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return lambda: albucore.hflip(array), lambda: _torch_array(torch.flip(torch.from_numpy(array), dims=(-2,)))


def _prepare_vflip(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return lambda: albucore.vflip(array), lambda: _torch_array(torch.flip(torch.from_numpy(array), dims=(-3,)))


def _prepare_to_float(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return (
        lambda: albucore.to_float(array),
        lambda: _torch_array(torch.mul(torch.from_numpy(array).to(torch.float32), 1.0 / 255.0)),
    )


def _prepare_from_float(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    return (
        lambda: albucore.from_float(array, np.uint8),
        lambda: _torch_array(torch.round(torch.mul(torch.from_numpy(array), 255.0)).clamp_(0.0, 255.0).to(torch.uint8)),
    )


def _prepare_uint8_lut(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    lut = np.bitwise_xor(np.arange(256, dtype=np.uint8), np.uint8(0xA5))
    lut_t = torch.from_numpy(lut)
    return (
        lambda: albucore.apply_uint8_lut(array, lut),
        lambda: _torch_array(lut_t[torch.from_numpy(array).to(torch.int64)]),
    )


def _prepare_copy_make_border(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    top, bottom, left, right = 3, 5, 7, 11
    return (
        lambda: albucore.copy_make_border(array, top, bottom, left, right, cv2.BORDER_CONSTANT, 0),
        lambda: _torch_array(
            torch_f.pad(
                torch.from_numpy(array).permute(2, 0, 1),
                (left, right, top, bottom),
                mode="constant",
                value=0,
            ).permute(1, 2, 0),
        ),
    )


def _prepare_resize(array: np.ndarray, _: np.random.Generator) -> PreparedBenchmark:
    height, width, _ = array.shape
    output_wh = (max(width // 2, 1), max(height // 2, 1))
    output_hw = (output_wh[1], output_wh[0])
    return (
        lambda: albucore.resize(array, output_wh, interpolation=cv2.INTER_LINEAR),
        lambda: _torch_array(
            torch_f.interpolate(
                torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0),
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0),
        ),
    )


FLOAT32 = (np.dtype(np.float32),)
UINT8 = (np.dtype(np.uint8),)
BOTH = (np.dtype(np.uint8), np.dtype(np.float32))

CANDIDATES: tuple[Candidate, ...] = (
    Candidate("wrap_numpy_tensor_numpy", BOTH, _prepare_wrap, rtol=0, atol=0),
    Candidate("exp", FLOAT32, _prepare_exp),
    Candidate("log", FLOAT32, _prepare_log),
    Candidate("sqrt", FLOAT32, _prepare_sqrt),
    Candidate("add_scalar", FLOAT32, _prepare_add),
    Candidate("multiply_scalar", FLOAT32, _prepare_multiply),
    Candidate("power_scalar", FLOAT32, _prepare_power),
    Candidate("multiply_add_scalar", FLOAT32, _prepare_multiply_add),
    Candidate("add_weighted", FLOAT32, _prepare_add_weighted),
    Candidate("normalize", FLOAT32, _prepare_normalize),
    Candidate("mean_std_global", FLOAT32, _prepare_mean_std_global),
    Candidate("mean_std_per_channel", FLOAT32, _prepare_mean_std_per_channel),
    Candidate("reduce_sum_global_float32", FLOAT32, _prepare_reduce_sum_global_float),
    Candidate("reduce_sum_per_channel_float32", FLOAT32, _prepare_reduce_sum_per_channel_float),
    Candidate("reduce_sum_global_uint8", UINT8, _prepare_reduce_sum_global_uint8, rtol=0, atol=0),
    Candidate("reduce_sum_per_channel_uint8", UINT8, _prepare_reduce_sum_per_channel_uint8, rtol=0, atol=0),
    Candidate("hflip", BOTH, _prepare_hflip, rtol=0, atol=0),
    Candidate("vflip", BOTH, _prepare_vflip, rtol=0, atol=0),
    Candidate("to_float", UINT8, _prepare_to_float),
    Candidate("from_float", FLOAT32, _prepare_from_float, rtol=0, atol=0),
    Candidate("apply_uint8_lut", UINT8, _prepare_uint8_lut, rtol=0, atol=0),
    Candidate("copy_make_border_constant", BOTH, _prepare_copy_make_border, rtol=0, atol=0),
    Candidate("resize_linear", FLOAT32, _prepare_resize),
)


def _make_array(rng: np.random.Generator, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    return _uint8_image(rng, shape) if dtype == np.uint8 else _float_image(rng, shape)


def _correctness(
    baseline: Callable[[], ArrayResult],
    candidate: Callable[[], ArrayResult],
    *,
    rtol: float,
    atol: float,
) -> str:
    try:
        _assert_equal(baseline(), candidate(), rtol, atol)
    except (AssertionError, RuntimeError, TypeError, ValueError) as error:
        return f"mismatch: {type(error).__name__}"
    return "ok"


def _bench_candidate(
    candidate: Candidate,
    array: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    warmup: int,
    *,
    compare_existing_routes: bool,
) -> Row:
    baseline, torch_candidate = candidate.prepare(array, rng)
    if compare_existing_routes:
        with _torch_router_disabled():
            correctness = _correctness(baseline, torch_candidate, rtol=candidate.rtol, atol=candidate.atol)
            albucore_t: WallTimingMs = bench_wall_ms(baseline, repeats=repeats, warmup=warmup)
    else:
        correctness = _correctness(baseline, torch_candidate, rtol=candidate.rtol, atol=candidate.atol)
        albucore_t = bench_wall_ms(baseline, repeats=repeats, warmup=warmup)
    torch_t: WallTimingMs = bench_wall_ms(torch_candidate, repeats=repeats, warmup=warmup)
    return Row(candidate.name, array.shape, array.dtype.name, correctness, albucore_t.median, torch_t.median)


def _matmul_rows(rng: np.random.Generator, repeats: int, warmup: int) -> list[Row]:
    rows: list[Row] = []
    for m, k, n in ((64, 80, 48), (240, 320, 160), (480, 640, 320)):
        a = _float_image(rng, (m, k))
        b = _float_image(rng, (k, n))
        baseline = lambda a=a, b=b: albucore.matmul(a, b)
        torch_candidate = lambda a=a, b=b: _torch_array(torch.matmul(torch.from_numpy(a), torch.from_numpy(b)))
        correctness = _correctness(baseline, torch_candidate, rtol=1e-5, atol=1e-5)
        rows.append(
            Row(
                "matmul",
                (m, k, n),
                np.dtype(np.float32).name,
                correctness,
                bench_wall_ms(baseline, repeats, warmup).median,
                bench_wall_ms(torch_candidate, repeats, warmup).median,
            ),
        )
    return rows


def _cdist_rows(rng: np.random.Generator, repeats: int, warmup: int) -> list[Row]:
    rows: list[Row] = []
    for n1, n2, dim in ((8, 16, 8), (24, 32, 16), (48, 64, 32)):
        a = _float_image(rng, (n1, dim))
        b = _float_image(rng, (n2, dim))
        baseline = lambda a=a, b=b: albucore.pairwise_distances_squared(a, b)
        torch_candidate = lambda a=a, b=b: _torch_array(torch.cdist(torch.from_numpy(a), torch.from_numpy(b)).square())
        correctness = _correctness(baseline, torch_candidate, rtol=1e-4, atol=1e-5)
        rows.append(
            Row(
                "pairwise_distances_squared",
                (n1, n2, dim),
                np.dtype(np.float32).name,
                correctness,
                bench_wall_ms(baseline, repeats, warmup).median,
                bench_wall_ms(torch_candidate, repeats, warmup).median,
            ),
        )
    return rows


def _format_rows(rows: list[Row]) -> list[str]:
    lines = [
        "| candidate | shape | dtype | correctness | albucore ms | Torch ms | winner | speedup |",
        "|---|---:|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.candidate} | {'×'.join(map(str, row.shape))} | {row.dtype} | {row.correctness} | "
            f"{row.albucore_ms:.4f} | {row.torch_ms:.4f} | {row.winner} | {row.speedup:.2f}× |",
        )
    return lines


def _summary(rows: list[Row]) -> list[str]:
    accepted = [row for row in rows if row.correctness == "ok" and row.winner == "torch"]
    mismatches = sorted({row.candidate for row in rows if row.correctness != "ok"})
    lines = ["## Result", ""]
    if accepted:
        by_candidate: dict[str, list[Row]] = {}
        for row in accepted:
            by_candidate.setdefault(row.candidate, []).append(row)
        lines.append("Torch won at least one semantically valid cell:")
        lines.extend(
            f"- `{name}`: {len(items)} cell(s), best {max(item.speedup for item in items):.2f}×."
            for name, items in sorted(by_candidate.items())
        )
    else:
        lines.append("Torch did not win a semantically valid cell in this run.")
    if mismatches:
        lines.extend(["", "Candidates with non-matching public results:"])
        lines.extend(f"- `{name}`" for name in mismatches)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use the first two canonical HWC sizes and C=1/3.")
    parser.add_argument("--full", action="store_true", help="Use every canonical HWC size and C=1/3/9 (default).")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=torch.get_num_threads())
    parser.add_argument(
        "--disable-public-torch-route",
        action="store_true",
        help="Temporarily disable the configured Torch route and measure the prior CPU backend.",
    )
    parser.add_argument(
        "--candidates",
        help="Comma-separated candidate names. The default times every candidate.",
    )
    parser.add_argument(
        "--volumes",
        action="store_true",
        help="Also time generic candidates on the canonical DHWC and NDHWC shape grid.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.quick and args.full:
        parser.error("--quick and --full are mutually exclusive")
    if args.threads < 1:
        parser.error("--threads must be >= 1")

    # Configure before the first parallel kernel. Interop threads can only be changed once per process.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(args.threads)

    rng = np.random.default_rng(0)
    sizes = ROUTER_HWC_QUICK_HW if args.quick else ROUTER_HWC_FULL_HW
    channels = (1, 3) if args.quick else (1, 3, 9)
    requested = None if args.candidates is None else frozenset(args.candidates.split(","))
    names = {candidate.name for candidate in CANDIDATES}
    if requested is not None and not requested <= names | {"matmul", "pairwise_distances_squared"}:
        unknown = ", ".join(sorted(requested - names - {"matmul", "pairwise_distances_squared"}))
        parser.error(f"unknown candidate(s): {unknown}")
    active_candidates = CANDIDATES if requested is None else tuple(c for c in CANDIDATES if c.name in requested)
    rows: list[Row] = []
    for shape_hw in sizes:
        for channels_count in channels:
            shape = (*shape_hw, channels_count)
            for dtype in BOTH:
                array = _make_array(rng, shape, dtype)
                for candidate in active_candidates:
                    if dtype in candidate.dtypes:
                        rows.append(
                            _bench_candidate(
                                candidate,
                                array,
                                rng,
                                args.repeats,
                                args.warmup,
                                compare_existing_routes=args.disable_public_torch_route,
                            ),
                        )
    if args.volumes:
        generic_candidate_names = {
            "wrap_numpy_tensor_numpy",
            "add_scalar",
            "multiply_scalar",
            "power_scalar",
            "multiply_add_scalar",
            "add_weighted",
            "normalize",
            "mean_std_global",
            "mean_std_per_channel",
            "reduce_sum_global_float32",
            "reduce_sum_per_channel_float32",
            "reduce_sum_global_uint8",
            "reduce_sum_per_channel_uint8",
            "to_float",
            "from_float",
            "apply_uint8_lut",
        }
        volume_candidates = tuple(c for c in active_candidates if c.name in generic_candidate_names)
        volume_shapes = tuple(shape for shape in SCALE_LUT_SHAPES if len(shape) in (4, 5))
        for shape in volume_shapes:
            for dtype in BOTH:
                array = _make_array(rng, shape, dtype)
                for candidate in volume_candidates:
                    if dtype in candidate.dtypes:
                        rows.append(
                            _bench_candidate(
                                candidate,
                                array,
                                rng,
                                args.repeats,
                                args.warmup,
                                compare_existing_routes=args.disable_public_torch_route,
                            ),
                        )
    if requested is None or "matmul" in requested:
        rows.extend(_matmul_rows(rng, args.repeats, args.warmup))
    if requested is None or "pairwise_distances_squared" in requested:
        rows.extend(_cdist_rows(rng, args.repeats, args.warmup))

    lines = [
        "# Torch CPU backend benchmark",
        "",
        f"Run date: {dt.date.today().isoformat()}. Platform: `{platform.platform()}` (`{platform.machine()}`).",
        "",
        f"Versions: Torch `{torch.__version__}`, NumPy `{np.__version__}`, OpenCV `{cv2.__version__}`, "
        f"NumKong `{__import__('numkong').__version__}`. Torch/OpenCV CPU threads: `{args.threads}`; "
        f"Torch interop threads: `{torch.get_num_interop_threads()}`. Repeats: `{args.repeats}`; warmup: `{args.warmup}`.",
        "",
        "Each Torch cell includes `torch.from_numpy(array)` and `Tensor.numpy()`. Those operations share CPU storage "
        "for these writable, positive-stride arrays; device transfers and compile time are excluded. Albucore calls "
        "the public router. By default, `albucore ms` measures its configured production route. "
        "`--disable-public-torch-route` instead measures the prior NumPy/NumKong/OpenCV/LUT baseline. "
        "A `mismatch` cannot become a router regardless of its timing.",
        "",
        *_summary(rows),
        "",
        "## Full matrix",
        "",
        *_format_rows(rows),
        "",
    ]
    report = "\n".join(lines)
    if args.output is None:
        print(report)
    else:
        args.output.write_text(report)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
