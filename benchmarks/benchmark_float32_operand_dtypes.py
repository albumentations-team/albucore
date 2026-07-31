#!/usr/bin/env python3
"""Benchmark float32 arithmetic across accepted ndarray operand dtypes.

The legacy candidate intentionally leaves operands unchanged so NumPy creates a
promoted result where required. The current candidate normalizes only operands
that would promote float32 image arithmetic, avoiding copies for uint8, float16,
and float32 operands.

Run from the repository root::

    uv run python benchmarks/benchmark_float32_operand_dtypes.py
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

import albucore.arithmetic as arithmetic

HWC_SHAPES = tuple(
    (height, width, channels)
    for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024))
    for channels in (1, 3, 9)
)
OPERAND_DTYPES = tuple(np.dtype(dtype) for dtype in (np.uint8, np.float16, np.float32, np.float64))


def _legacy_apply_numpy(
    img: np.ndarray,
    value: float | np.ndarray,
    operation: str,
) -> np.ndarray:
    return arithmetic.np_operations[operation](img.astype(np.float32, copy=False), value)


def _legacy_multiply_add_numpy(
    img: np.ndarray,
    factor: float | np.ndarray,
    value: float | np.ndarray,
) -> np.ndarray:
    img_f = img.astype(np.float32, copy=False)

    def _scalar_float(x: float | np.ndarray) -> float | None:
        return None if isinstance(x, np.ndarray) else float(x)

    sf, sv = _scalar_float(factor), _scalar_float(value)
    if sf is not None and sv is not None and sf == 0.0 and sv == 0.0:
        return np.zeros_like(img_f, dtype=np.float32)
    if sf is not None and sv is not None and sf == 1.0 and sv == 0.0:
        return img_f

    n_dim = img_f.ndim
    channels = int(img_f.shape[-1])
    factor_broadcast = arithmetic._broadcast_channel_vector(factor, n_dim, channels)  # noqa: SLF001
    value_broadcast = arithmetic._broadcast_channel_vector(value, n_dim, channels)  # noqa: SLF001
    result = (
        np.zeros_like(img_f)
        if arithmetic._is_all_zero_param(factor_broadcast)  # noqa: SLF001
        else np.multiply(img_f, factor_broadcast)
    )
    return (
        result
        if arithmetic._is_all_zero_param(value_broadcast)  # noqa: SLF001
        else np.add(result, value_broadcast)
    )


def _operand(operation: str, shape: tuple[int, int, int], dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 4, size=shape, dtype=dtype)
    if operation == "add":
        value = rng.uniform(-0.1, 0.1, size=shape)
    elif operation == "multiply":
        value = rng.uniform(0.8, 1.2, size=shape)
    else:
        value = rng.uniform(0.7, 1.3, size=shape)
    return value.astype(dtype, copy=False)


def _apply_thunk(operation: str, img: np.ndarray, value: np.ndarray) -> Callable[[], np.ndarray]:
    public_operation = {
        "add": arithmetic.add,
        "multiply": arithmetic.multiply,
        "power": arithmetic.power,
    }[operation]
    return lambda: public_operation(img, value)


def _multiply_add_thunk(img: np.ndarray, factor: np.ndarray, value: np.ndarray) -> Callable[[], np.ndarray]:
    return lambda: arithmetic.multiply_add(img, factor, value)


def _time_apply(
    operation: str,
    img: np.ndarray,
    value: np.ndarray,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    current_apply_numpy = arithmetic.apply_numpy

    def use_legacy() -> None:
        arithmetic.apply_numpy = _legacy_apply_numpy  # type: ignore[assignment]

    def use_current() -> None:
        arithmetic.apply_numpy = current_apply_numpy

    try:
        return _paired_median_ms(_apply_thunk(operation, img, value), use_legacy, use_current, repeats, warmup)
    finally:
        arithmetic.apply_numpy = current_apply_numpy


def _time_multiply_add(
    img: np.ndarray,
    factor: np.ndarray,
    value: np.ndarray,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    current_multiply_add_numpy = arithmetic.multiply_add_numpy

    def use_legacy() -> None:
        arithmetic.multiply_add_numpy = _legacy_multiply_add_numpy  # type: ignore[assignment]

    def use_current() -> None:
        arithmetic.multiply_add_numpy = current_multiply_add_numpy

    try:
        return _paired_median_ms(_multiply_add_thunk(img, factor, value), use_legacy, use_current, repeats, warmup)
    finally:
        arithmetic.multiply_add_numpy = current_multiply_add_numpy


def _paired_median_ms(
    thunk: Callable[[], np.ndarray],
    use_legacy: Callable[[], None],
    use_current: Callable[[], None],
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        use_legacy()
        thunk()
        use_current()
        thunk()

    legacy_samples: list[float] = []
    current_samples: list[float] = []
    for index in range(repeats):
        candidates = (
            ((use_legacy, legacy_samples), (use_current, current_samples))
            if index % 2 == 0
            else ((use_current, current_samples), (use_legacy, legacy_samples))
        )
        for select, samples in candidates:
            select()
            started = time.perf_counter()
            thunk()
            samples.append((time.perf_counter() - started) * 1_000.0)

    return float(np.median(legacy_samples)), float(np.median(current_samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print("# Float32 arithmetic operand dtype benchmark")
    print()
    print("Median milliseconds. Ratio is current / legacy; values below 1 are faster.")
    print()
    print("| operation | shape | operand dtype | legacy | current | ratio |")
    print("|---|---:|---:|---:|---:|---:|")

    for shape in HWC_SHAPES:
        img = rng.uniform(0.05, 0.95, size=shape).astype(np.float32)
        for operation in ("add", "multiply", "power"):
            for operand_dtype in OPERAND_DTYPES:
                operand = _operand(operation, shape, operand_dtype, rng)
                legacy_ms, current_ms = _time_apply(
                    operation,
                    img,
                    operand,
                    args.repeats,
                    args.warmup,
                )
                print(
                    f"| {operation} | {'×'.join(map(str, shape))} | {operand_dtype.name} | "
                    f"{legacy_ms:.4f} | {current_ms:.4f} | {current_ms / legacy_ms:.3f}× |",
                )

        for operand_dtype in OPERAND_DTYPES:
            factor = _operand("multiply", shape, operand_dtype, rng)
            value = _operand("add", shape, operand_dtype, rng)
            legacy_ms, current_ms = _time_multiply_add(
                img,
                factor,
                value,
                args.repeats,
                args.warmup,
            )
            print(
                f"| multiply_add | {'×'.join(map(str, shape))} | {operand_dtype.name} | "
                f"{legacy_ms:.4f} | {current_ms:.4f} | {current_ms / legacy_ms:.3f}× |",
            )


if __name__ == "__main__":
    main()
