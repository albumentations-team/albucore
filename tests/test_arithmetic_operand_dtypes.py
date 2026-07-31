from collections.abc import Callable

import numpy as np
import pytest

from albucore import add, multiply, multiply_add, power
from albucore.arithmetic import _prepare_numpy_value, apply_numpy, multiply_add_numpy

PUBLIC_OPERATIONS: dict[str, Callable[[np.ndarray, np.ndarray | float], np.ndarray]] = {
    "add": add,
    "multiply": multiply,
    "power": power,
}


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_prepare_numpy_value_reuses_operands_that_resolve_to_float32(dtype: type[np.generic]) -> None:
    operand = np.ones((5, 7, 3), dtype=dtype)

    assert _prepare_numpy_value(operand) is operand


@pytest.mark.parametrize("operation", ["add", "multiply", "power"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16])
def test_float32_arithmetic_avoids_copy_for_compatible_operand_dtypes(
    operation: str,
    dtype: type[np.generic],
) -> None:
    shape = (5, 7, 3)
    img = np.linspace(0.1, 0.7, np.prod(shape), dtype=np.float32).reshape(shape)
    operand = np.full(shape, 1, dtype=dtype)

    result = apply_numpy(img, operand, operation)  # type: ignore[arg-type]
    expected = getattr(np, operation)(img, operand)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("operation", ["add", "multiply", "power"])
def test_float32_arithmetic_preserves_finite_float64_operands_outside_float32_range(operation: str) -> None:
    img = np.array([0.0, 1e-40, 0.5, 1.0], dtype=np.float32).reshape(2, 2, 1)
    operand = np.full(img.shape, 1e40, dtype=np.float64)

    with np.errstate(over="ignore"):
        narrowed_operand = operand.astype(np.float32)
    assert np.isfinite(operand).all()
    assert np.isinf(narrowed_operand).all()

    result = apply_numpy(img, operand, operation)
    with np.errstate(over="ignore"):
        expected = getattr(np, operation)(img, operand).astype(np.float32)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)

    public_result = PUBLIC_OPERATIONS[operation](img, operand)
    expected_public = np.clip(expected, 0, 1).astype(np.float32)
    assert public_result.dtype == np.float32
    np.testing.assert_array_equal(public_result, expected_public)


@pytest.mark.parametrize("operation", ["add", "multiply", "power"])
@pytest.mark.parametrize("operand", [1e40, np.float64(1e40)], ids=["python_float", "numpy_float64"])
def test_float32_arithmetic_preserves_finite_scalars_outside_float32_range(
    operation: str,
    operand: float | np.float64,
) -> None:
    img = np.array([0.0, 1e-40, 0.5, 1.0], dtype=np.float32).reshape(2, 2, 1)
    operand_float64 = np.float64(operand)

    result = apply_numpy(img, operand, operation)
    with np.errstate(over="ignore"):
        expected = getattr(np, operation)(img, operand_float64).astype(np.float32)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)

    public_result = PUBLIC_OPERATIONS[operation](img, operand)  # type: ignore[arg-type]
    expected_public = np.clip(expected, 0, 1).astype(np.float32)
    assert public_result.dtype == np.float32
    np.testing.assert_array_equal(public_result, expected_public)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16])
def test_float32_multiply_add_avoids_copy_for_compatible_operand_dtypes(dtype: type[np.generic]) -> None:
    shape = (5, 7, 3)
    img = np.linspace(0.1, 0.7, np.prod(shape), dtype=np.float32).reshape(shape)
    factor = np.full(shape, 1, dtype=dtype)
    value = np.full(shape, 0, dtype=dtype)

    result = multiply_add_numpy(img, factor, value)

    assert _prepare_numpy_value(factor) is factor
    assert _prepare_numpy_value(value) is value
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, img)


def test_float32_multiply_add_preserves_finite_float64_operand_outside_float32_range() -> None:
    img = np.array([0.0, 1e-40, 0.5, 1.0], dtype=np.float32).reshape(2, 2, 1)
    factor = np.full(img.shape, 1e40, dtype=np.float64)

    result = multiply_add_numpy(img, factor, 0.0)
    with np.errstate(over="ignore"):
        expected = np.multiply(img, factor).astype(np.float32)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)

    public_result = multiply_add(img, factor, 0.0)
    expected_public = np.clip(expected, 0, 1).astype(np.float32)
    assert public_result.dtype == np.float32
    np.testing.assert_array_equal(public_result, expected_public)


@pytest.mark.parametrize("factor", [1e40, np.float64(1e40)], ids=["python_float", "numpy_float64"])
def test_float32_multiply_add_preserves_finite_scalar_outside_float32_range(
    factor: float | np.float64,
) -> None:
    img = np.array([0.0, 1e-40, 0.5, 1.0], dtype=np.float32).reshape(2, 2, 1)
    factor_float64 = np.float64(factor)

    result = multiply_add_numpy(img, factor, 0.0)  # type: ignore[arg-type]
    with np.errstate(over="ignore"):
        expected = np.multiply(img, factor_float64).astype(np.float32)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)

    public_result = multiply_add(img, factor, 0.0)  # type: ignore[arg-type]
    expected_public = np.clip(expected, 0, 1).astype(np.float32)
    assert public_result.dtype == np.float32
    np.testing.assert_array_equal(public_result, expected_public)


def test_float32_multiply_add_avoids_numkong_for_scalar_outside_float32_range() -> None:
    img = np.zeros((1_000_000, 1, 1), dtype=np.float32)

    result = multiply_add(img, 1e40, 0.0)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, img)


def _float64_operand(kind: str, shape: tuple[int, int, int]) -> np.ndarray | float:
    if kind == "scalar":
        return np.float64(0.9)
    if kind == "vector":
        return np.linspace(0.8, 1.0, shape[-1], dtype=np.float64)
    if kind == "contiguous_array":
        return np.linspace(0.8, 1.0, np.prod(shape), dtype=np.float64).reshape(shape)

    expanded_shape = (shape[0], shape[1] * 2, shape[2])
    operand = np.linspace(0.8, 1.0, np.prod(expanded_shape), dtype=np.float64).reshape(expanded_shape)
    return operand[:, ::2, :]


@pytest.mark.parametrize("operation", ["add", "multiply", "power"])
@pytest.mark.parametrize("operand_kind", ["scalar", "vector", "contiguous_array", "strided_array"])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_float32_arithmetic_normalizes_float64_operands(
    operation: str,
    operand_kind: str,
    channels: int,
) -> None:
    shape = (5, 7, channels)
    img = np.linspace(0.1, 0.7, np.prod(shape), dtype=np.float32).reshape(shape)
    operand = _float64_operand(operand_kind, shape)
    if operand_kind == "strided_array":
        assert isinstance(operand, np.ndarray)
        assert not operand.flags.c_contiguous

    result = apply_numpy(img, operand, operation)  # type: ignore[arg-type]
    operand_float32 = np.asarray(operand, dtype=np.float32)
    expected = getattr(np, operation)(img, operand_float32)

    assert result.dtype == np.float32
    assert result.shape == img.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    public_result = PUBLIC_OPERATIONS[operation](img, operand)
    assert public_result.dtype == np.float32
    assert public_result.shape == img.shape
    np.testing.assert_allclose(public_result, np.clip(expected, 0, 1), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("parameter", ["factor", "value"])
@pytest.mark.parametrize("operand_kind", ["scalar", "vector", "contiguous_array", "strided_array"])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_float32_multiply_add_normalizes_float64_operands(
    parameter: str,
    operand_kind: str,
    channels: int,
) -> None:
    shape = (5, 7, channels)
    img = np.linspace(0.1, 0.7, np.prod(shape), dtype=np.float32).reshape(shape)
    operand = _float64_operand(operand_kind, shape)
    factor = operand if parameter == "factor" else 0.5
    value = operand if parameter == "value" else 0.1

    result = multiply_add_numpy(img, factor, value)  # type: ignore[arg-type]
    expected = np.multiply(img, np.asarray(factor, dtype=np.float32))
    expected = np.add(expected, np.asarray(value, dtype=np.float32))

    assert result.dtype == np.float32
    assert result.shape == img.shape
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    public_result = multiply_add(img, factor, value)  # type: ignore[arg-type]
    assert public_result.dtype == np.float32
    assert public_result.shape == img.shape
    np.testing.assert_allclose(public_result, np.clip(expected, 0, 1), rtol=1e-6, atol=1e-7)
