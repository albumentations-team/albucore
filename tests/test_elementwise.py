from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import albucore as ac
from albucore import elementwise

Unary = Callable[..., np.ndarray]


def test_exp_preserves_singleton_channel_shape_and_float32_dtype() -> None:
    array = np.linspace(-2.0, 2.0, 64 * 64, dtype=np.float32).reshape(64, 64, 1)
    original = array.copy()

    result = ac.exp(array)

    assert result.shape == array.shape
    assert result.dtype == np.float32
    assert not np.shares_memory(result, array)
    np.testing.assert_array_equal(array, original)
    np.testing.assert_allclose(result, np.exp(array), rtol=1e-6, atol=0.0)


def test_sqrt_preserves_singleton_channel_shape_and_float32_dtype() -> None:
    array = np.linspace(0.0, 4.0, 35, dtype=np.float32).reshape(5, 7, 1)

    result = ac.sqrt(array)

    assert result.shape == array.shape
    assert result.dtype == np.float32
    assert not np.shares_memory(result, array)
    np.testing.assert_allclose(result, np.sqrt(array), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [(ac.exp, np.exp), (ac.sqrt, np.sqrt)],
)
def test_no_output_does_not_mutate_a_view(operation: Unary, reference: Unary) -> None:
    base = np.linspace(0.1, 4.0, 256 * 512, dtype=np.float32).reshape(256, 512, 1)
    view = base[:, ::2]
    original = base.copy()
    expected = reference(view)

    result = operation(view)

    assert not np.shares_memory(result, view)
    np.testing.assert_array_equal(base, original)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [(ac.exp, np.exp), (ac.sqrt, np.sqrt)],
)
def test_explicit_output_mutates_owned_writable_array(operation: Unary, reference: Unary) -> None:
    array = np.linspace(0.1, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64, 1).copy()
    expected = reference(array.copy())

    result = operation(array, out=array)

    assert result is array
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("operation", [ac.exp, ac.sqrt])
def test_no_output_returns_new_array_for_read_only_input(operation: Unary) -> None:
    array = np.linspace(0.1, 4.0, 105, dtype=np.float32).reshape(5, 7, 3).copy()
    array.flags.writeable = False

    result = operation(array)

    assert not np.shares_memory(result, array)
    assert result.flags.writeable


@pytest.mark.parametrize("operation", [ac.exp, ac.sqrt])
@pytest.mark.parametrize(
    "shape",
    [(5, 7), (5, 7, 1), (5, 7, 3), (5, 7, 5), (2, 5, 7, 1), (3, 5, 7, 5)],
)
def test_elementwise_operations_preserve_all_supported_ranks(operation: Unary, shape: tuple[int, ...]) -> None:
    array = np.linspace(0.1, 4.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)

    result = operation(array)

    assert result.shape == shape
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    ("operation", "reference"),
    [(ac.exp, np.exp), (ac.sqrt, np.sqrt)],
)
@pytest.mark.parametrize("shape", [(4, 32, 40, 3)])
def test_elementwise_operations_match_numpy_on_large_rank4_layouts(
    operation: Unary,
    reference: Unary,
    shape: tuple[int, ...],
) -> None:
    array = np.linspace(0.1, 4.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)

    result = operation(array)

    assert result.shape == shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, reference(array), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("operation", [ac.exp, ac.sqrt])
@pytest.mark.parametrize("shape", [(0,), (0, 7), (0, 5, 7, 3)])
@pytest.mark.parametrize("provided_output", [False, True])
def test_elementwise_operations_support_empty_arrays(
    operation: Unary,
    shape: tuple[int, ...],
    provided_output: bool,
) -> None:
    array = np.empty(shape, dtype=np.float32)

    result = operation(array, out=array if provided_output else None)

    assert result.shape == shape
    assert result.dtype == np.float32
    if provided_output:
        assert result is array
    else:
        assert not np.shares_memory(result, array)


@pytest.mark.parametrize(
    ("operation", "reference", "values"),
    [
        (
            ac.exp,
            np.exp,
            [
                -np.inf,
                -104.0,
                -100.0,
                -1.0,
                -0.0,
                0.0,
                np.nextafter(np.float32(0), np.float32(1)),
                1.0,
                100.0,
                np.inf,
                np.nan,
            ],
        ),
        (
            ac.sqrt,
            np.sqrt,
            [
                -np.inf,
                -1.0,
                -0.0,
                0.0,
                np.nextafter(np.float32(0), np.float32(1)),
                np.finfo(np.float32).tiny,
                1.0,
                np.inf,
                np.nan,
            ],
        ),
    ],
)
@pytest.mark.parametrize("provided_output", [False, True])
def test_elementwise_operations_match_numpy_special_values(
    operation: Unary,
    reference: Unary,
    values: list[float],
    provided_output: bool,
) -> None:
    array = np.resize(np.asarray(values, dtype=np.float32), 8192).copy()

    with np.errstate(all="ignore"):
        expected = reference(array)
        result = operation(array, out=array if provided_output else None)

    if provided_output:
        assert result is array
    else:
        assert not np.shares_memory(result, array)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("layout", ["contiguous", "strided", "out"])
def test_exp_uint8_matches_float32_numpy_for_every_byte(layout: str) -> None:
    values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
    array = np.repeat(values, 3, axis=-1)[:, ::2] if layout == "strided" else values
    with np.errstate(over="ignore"):
        expected = np.exp(array.astype(np.float32))

    if layout == "out":
        out = np.empty(array.shape, dtype=np.float32)
        result = ac.exp(array, out=out)
        assert result is out
    else:
        result = ac.exp(array)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def _uint8_sqrt_reference(array: np.ndarray) -> np.ndarray:
    return np.sqrt(array.astype(np.float32))


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    calls: list[int] = []
    original = getattr(elementwise, name)

    def wrapper(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        if result is not None:
            calls.append(1)
        return result

    monkeypatch.setattr(elementwise, name, wrapper)
    return calls


@pytest.mark.parametrize("shape", [(40, 50, 1), (40, 50, 3), (30, 40, 9), (2, 20, 30, 3)])
@pytest.mark.parametrize("with_out", [False, True])
def test_uint8_sqrt_macos_routes_through_torch_and_matches_numpy(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    with_out: bool,
) -> None:
    monkeypatch.setattr(elementwise, "_MACOS_ARM64", True)
    monkeypatch.setattr(elementwise, "_LINUX_X86_64", False)
    monkeypatch.setattr(elementwise, "_MACOS_UINT8_SQRT_TORCH_MIN_ELEMENTS", 1)
    monkeypatch.setattr(elementwise, "_MACOS_UINT8_SQRT_TORCH_AFTER_LUT_MIN_ELEMENTS", 1)
    calls = _spy(monkeypatch, "sqrt_uint8_torch")
    array = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    out = np.empty(shape, dtype=np.float32) if with_out else None

    result = ac.sqrt(array, out=out)

    assert calls == [1]
    assert result.dtype == np.float32
    assert (result is out) if with_out else not np.shares_memory(result, array)
    # Torch and NumPy float32 sqrt differ by up to 1 ulp on some CPUs (x86), so compare within float32 precision.
    np.testing.assert_allclose(result, _uint8_sqrt_reference(array), rtol=1e-6, atol=0)


def test_uint8_sqrt_macos_torch_route_handles_strided_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elementwise, "_MACOS_ARM64", True)
    monkeypatch.setattr(elementwise, "_LINUX_X86_64", False)
    monkeypatch.setattr(elementwise, "_MACOS_UINT8_SQRT_TORCH_MIN_ELEMENTS", 1)
    calls = _spy(monkeypatch, "sqrt_uint8_torch")
    array = np.random.default_rng(1).integers(0, 256, (40, 60, 3), dtype=np.uint8)[:, ::2]
    out = np.empty((40, 60, 3), dtype=np.float32)[:, ::2]

    result = ac.sqrt(array, out=out)

    assert calls == [1]
    assert result is out
    np.testing.assert_allclose(result, _uint8_sqrt_reference(array), rtol=1e-6, atol=0)


def test_uint8_sqrt_macos_read_only_input_skips_torch_and_matches_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elementwise, "_MACOS_ARM64", True)
    monkeypatch.setattr(elementwise, "_LINUX_X86_64", False)
    monkeypatch.setattr(elementwise, "_MACOS_UINT8_SQRT_TORCH_MIN_ELEMENTS", 1)
    calls = _spy(monkeypatch, "sqrt_uint8_torch")
    array = np.random.default_rng(2).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    array.flags.writeable = False

    result = ac.sqrt(array)

    assert calls == []
    np.testing.assert_array_equal(result, _uint8_sqrt_reference(array))


@pytest.mark.parametrize("shape", [(40, 50, 1), (40, 50, 3), (30, 40, 9), (2, 20, 30, 3), (2, 3, 10, 20, 5)])
@pytest.mark.parametrize("with_out", [False, True])
def test_uint8_sqrt_linux_uses_float_lut_for_every_channel_count_with_and_without_out(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    with_out: bool,
) -> None:
    monkeypatch.setattr(elementwise, "_MACOS_ARM64", False)
    monkeypatch.setattr(elementwise, "_LINUX_X86_64", True)
    monkeypatch.setattr(elementwise, "_LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS", 1)
    calls = _spy(monkeypatch, "_apply_float_lut")
    array = np.random.default_rng(3).integers(0, 256, shape, dtype=np.uint8)
    out = np.empty(shape, dtype=np.float32) if with_out else None

    result = ac.sqrt(array, out=out)

    assert calls == [1]
    assert (result is out) if with_out else not np.shares_memory(result, array)
    # Torch and NumPy float32 sqrt differ by up to 1 ulp on some CPUs (x86), so compare within float32 precision.
    np.testing.assert_allclose(result, _uint8_sqrt_reference(array), rtol=1e-6, atol=0)


def test_uint8_sqrt_linux_below_lut_minimum_and_strided_input_use_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(elementwise, "_MACOS_ARM64", False)
    monkeypatch.setattr(elementwise, "_LINUX_X86_64", True)
    monkeypatch.setattr(elementwise, "_LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS", 1_000_000)
    calls = _spy(monkeypatch, "_apply_float_lut")
    array = np.random.default_rng(4).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    np.testing.assert_array_equal(ac.sqrt(array), _uint8_sqrt_reference(array))

    monkeypatch.setattr(elementwise, "_LINUX_UINT8_SQRT_LUT_MIN_ELEMENTS", 1)
    strided = array[:, ::2]
    np.testing.assert_array_equal(ac.sqrt(strided), _uint8_sqrt_reference(strided))
    assert calls == []
