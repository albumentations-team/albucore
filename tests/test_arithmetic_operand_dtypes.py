from collections.abc import Callable

import numpy as np
import pytest

from albucore import add, multiply, multiply_add, power
from albucore.arithmetic import apply_numpy, multiply_add_numpy

PUBLIC_OPERATIONS: dict[str, Callable[[np.ndarray, np.ndarray | float], np.ndarray]] = {
    "add": add,
    "multiply": multiply,
    "power": power,
}


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
