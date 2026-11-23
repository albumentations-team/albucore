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
