from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
import torch

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
    np.testing.assert_array_equal(result, _uint8_sqrt_reference(array))


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
    np.testing.assert_array_equal(result, _uint8_sqrt_reference(array))


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
    np.testing.assert_array_equal(result, _uint8_sqrt_reference(array))


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


def _tensor_exp_spy(monkeypatch: pytest.MonkeyPatch, minimums: dict[torch.dtype, int]) -> list[int]:
    monkeypatch.setattr(elementwise, "_TENSOR_EXP_NUMPY_MIN_ELEMENTS", minimums)
    return _spy(monkeypatch, "_exp_tensor_numpy")


def _tensor_exp_input(dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    # Values up to 18 overflow float16 results, which must stay silent like Torch.
    return (torch.rand(shape, generator=generator) * 24 - 6).to(dtype)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(3, 40, 50), (3, 100, 700), (2, 3, 8, 9)])
@pytest.mark.parametrize("destination", ["allocate", "out", "in_place"])
def test_tensor_exp_contiguous_input_uses_numpy_bridge_and_matches_torch(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    destination: str,
) -> None:
    if destination == "in_place" and dtype == torch.uint8:
        pytest.skip("uint8 input cannot be the float32 output")
    calls = _tensor_exp_spy(monkeypatch, {dtype: 1})
    image = _tensor_exp_input(dtype, shape)
    expected = torch.exp(image.to(torch.float32) if dtype == torch.uint8 else image)
    output = {"allocate": None, "out": torch.empty_like(expected), "in_place": image}[destination]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = ac.exp(image, out=output)

    assert calls == [1]
    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    if output is not None:
        assert result is output
    if dtype == torch.float16:
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("case", ["below_minimum", "dtype_absent", "strided_input", "strided_output"])
def test_tensor_exp_keeps_torch_outside_the_numpy_bridge_region(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    minimums = {torch.float32: 10**9 if case == "below_minimum" else 1}
    if case == "dtype_absent":
        minimums = {torch.float16: 1}
    calls = _tensor_exp_spy(monkeypatch, minimums)
    backing = torch.rand(3, 40, 100)
    image = backing[..., ::2] if case == "strided_input" else backing[..., :50].contiguous()
    output = torch.empty(3, 40, 100)[..., ::2] if case == "strided_output" else None

    result = ac.exp(image, out=output)

    assert calls == []
    torch.testing.assert_close(result, torch.exp(image))


@pytest.mark.parametrize(
    ("dtype", "present"),
    [(torch.uint8, True), (torch.float32, True), (torch.float16, False)],
)
def test_tensor_exp_platform_table_matches_measured_regions(dtype: torch.dtype, present: bool) -> None:
    table = elementwise._TENSOR_EXP_NUMPY_MIN_ELEMENTS
    if elementwise._MACOS_ARM64:
        assert dtype in table
    elif elementwise._LINUX_X86_64:
        assert (dtype in table) is present
    else:
        assert table == {}
