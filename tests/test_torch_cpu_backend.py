# ruff: noqa: S101, SLF001
"""Correctness and dispatch coverage for large CPU Torch backend candidates."""

from __future__ import annotations

import numpy as np
import pytest

from albucore import arithmetic, convert, stats, torch_backend
from albucore.functions import from_float, multiply_add, normalize, reduce_sum


def _enable_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_backend, "TORCH_MIN_ELEMENTS", 1)
    monkeypatch.setattr(torch_backend, "TORCH_CPU_BACKEND_ENABLED", True)


def _strided(array: np.ndarray) -> np.ndarray:
    return array[:, ::2, ...]


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize("max_value", [128.0, 255.0])
def test_from_float_torch_matches_numpy_and_bypasses_opencv(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    max_value: float,
) -> None:
    _enable_torch(monkeypatch)
    source = np.array(
        [[[0.0], [0.5 / 255], [1.5 / 255], [1.0], [1.2], [-0.1]]],
        dtype=np.float32,
    )
    image = {
        "contiguous": source,
        "fortran": np.asfortranarray(source),
        "strided": _strided(np.repeat(source, 2, axis=1)),
    }[layout]
    expected = convert.from_float_numpy(image, np.uint8, max_value)

    def fail_opencv(*args: object, **kwargs: object) -> np.ndarray:
        msg = "Torch-compatible float32 input should not reach the NumPy/OpenCV fallback."
        raise AssertionError(msg)

    monkeypatch.setattr(convert, "from_float_opencv", fail_opencv)
    result = from_float(image, np.uint8, max_value)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
def test_multiply_add_scalar_torch_matches_current_float32_contract(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    _enable_torch(monkeypatch)
    source = np.linspace(-2.0, 3.0, num=60, dtype=np.float32).reshape(3, 20, 1)
    image = {
        "contiguous": source,
        "fortran": np.asfortranarray(source),
        "strided": _strided(np.repeat(source, 2, axis=1)),
    }[layout]
    expected = image * np.float32(1.125) + np.float32(-0.05)

    def fail_numkong(*args: object, **kwargs: object) -> np.ndarray:
        msg = "Torch-compatible scalar float32 input should not reach NumKong."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic, "multiply_add_numkong", fail_numkong)
    result = multiply_add(image, 1.125, -0.05)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    assert result.dtype == np.float32


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
def test_multichannel_normalize_torch_matches_numpy_and_bypasses_fallback(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    _enable_torch(monkeypatch)
    source = np.linspace(-1.0, 2.0, num=180, dtype=np.float32).reshape(3, 20, 3)
    image = {
        "contiguous": source,
        "fortran": np.asfortranarray(source),
        "strided": _strided(np.repeat(source, 2, axis=1)),
    }[layout]
    mean = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    denominator = np.array([1.1, 1.3, 1.7], dtype=np.float32)
    expected = image * denominator - mean * denominator

    def fail_numpy(*args: object, **kwargs: object) -> np.ndarray:
        msg = "Large Torch-compatible multi-channel float32 input should not reach normalize_numpy."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic, "normalize_numpy", fail_numpy)
    result = normalize(image, mean, denominator)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    assert result.dtype == np.float32


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
def test_reduce_sum_torch_is_exact_for_global_and_per_channel(
    monkeypatch: pytest.MonkeyPatch,
    dtype: type[np.generic],
    layout: str,
) -> None:
    _enable_torch(monkeypatch)
    rng = np.random.default_rng(2)
    source = (
        rng.integers(0, 256, size=(2, 6, 10, 3), dtype=np.uint8)
        if dtype is np.uint8
        else rng.random((2, 6, 10, 3), dtype=np.float32)
    )
    image = {
        "contiguous": source,
        "fortran": np.asfortranarray(source),
        "strided": _strided(np.repeat(source, 2, axis=2)),
    }[layout]
    calls = 0
    original = torch_backend.reduce_sum_torch

    def spy(*args: object, **kwargs: object) -> np.generic | np.ndarray | None:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(stats, "reduce_sum_torch", spy)
    accumulator = np.uint64 if dtype is np.uint8 else np.float64
    axes = tuple(range(image.ndim - 1))

    for axis in (None, "global", "per_channel"):
        for keepdims in (False, True):
            expected_axis = None if axis in (None, "global") else axes
            expected = np.sum(image, axis=expected_axis, dtype=accumulator, keepdims=keepdims)
            np.testing.assert_array_equal(reduce_sum(image, axis, keepdims=keepdims), expected)

    # UInt8 keeps its established per-channel NumKong/OpenCV routing; Torch only
    # replaces its full reductions. Float32 also uses Torch for C <= 4 per-channel sums.
    assert calls == (4 if dtype is np.uint8 else 6)


def test_torch_backend_can_be_disabled_for_a_non_torch_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_backend, "TORCH_CPU_BACKEND_ENABLED", False)
    image = np.ones((800, 800, 1), dtype=np.float32)

    assert torch_backend.from_float_uint8_torch(image, 255.0) is None
    assert torch_backend.multiply_add_float32_torch(image, 1.1, 0.2) is None


def test_torch_backend_skips_read_only_and_negative_stride_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_torch(monkeypatch)
    image = np.ones((4, 5, 3), dtype=np.float32)
    read_only = image.copy()
    read_only.flags.writeable = False

    assert not torch_backend._is_torch_compatible_array(read_only)
    assert not torch_backend._is_torch_compatible_array(image[:, ::-1])
