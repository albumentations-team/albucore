from collections.abc import Callable

import numpy as np
import pytest

from albucore import (
    add,
    add_array,
    add_constant,
    add_vector,
    arithmetic,
    multiply,
    multiply_add,
    multiply_by_array,
    multiply_by_constant,
    multiply_by_vector,
    power,
)
from albucore.utils import clipped


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (lambda img: add(img, 0.5), lambda img: img + 0.5),
        (lambda img: add_constant(img, 0.5), lambda img: img + 0.5),
        (lambda img: add_vector(img, np.array([0.5], dtype=np.float32)), lambda img: img + 0.5),
        (lambda img: add_array(img, np.full_like(img, 0.5)), lambda img: img + 0.5),
        (lambda img: multiply(img, 2.0), lambda img: img * 2.0),
        (lambda img: multiply_by_constant(img, 2.0), lambda img: img * 2.0),
        (lambda img: multiply_by_vector(img, np.array([2.0], dtype=np.float32)), lambda img: img * 2.0),
        (lambda img: multiply_by_array(img, np.full_like(img, 2.0)), lambda img: img * 2.0),
        (lambda img: multiply_add(img, 2.0, -0.5), lambda img: img * 2.0 - 0.5),
        (lambda img: power(img, 3.0), lambda img: img**3.0),
    ],
)
def test_float32_arithmetic_preserves_raw_numeric_range(
    operation: Callable[[np.ndarray], np.ndarray],
    reference: Callable[[np.ndarray], np.ndarray],
) -> None:
    img = np.array([[[-1.0], [2.0]]], dtype=np.float32)

    result = operation(img)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, reference(img))


@pytest.mark.parametrize(
    "operation",
    [
        lambda img: add(img, 0),
        lambda img: add_constant(img, 0),
        lambda img: multiply(img, 1),
        lambda img: multiply_by_constant(img, 1),
        lambda img: multiply_add(img, 1, 0),
        lambda img: power(img, 1),
    ],
)
def test_float32_arithmetic_identity_preserves_values_outside_unit_range(
    operation: Callable[[np.ndarray], np.ndarray],
) -> None:
    img = np.array([[[-2.0], [3.0]]], dtype=np.float32)

    result = operation(img)

    np.testing.assert_array_equal(result, img)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda img: add(img, 10), np.array([[[10], [255]]], dtype=np.uint8)),
        (lambda img: add_constant(img, 10), np.array([[[10], [255]]], dtype=np.uint8)),
        (lambda img: add_vector(img, np.array([10])), np.array([[[10], [255]]], dtype=np.uint8)),
        (lambda img: add_array(img, np.full_like(img, 10)), np.array([[[10], [255]]], dtype=np.uint8)),
        (lambda img: multiply(img, 2), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: multiply_by_constant(img, 2), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: multiply_by_vector(img, np.array([2])), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: multiply_by_array(img, np.full_like(img, 2)), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: multiply_add(img, 2, -10), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: power(img, 2), np.array([[[0], [255]]], dtype=np.uint8)),
    ],
)
def test_uint8_arithmetic_remains_saturating(
    operation: Callable[[np.ndarray], np.ndarray],
    expected: np.ndarray,
) -> None:
    img = np.array([[[0], [250]]], dtype=np.uint8)

    result = operation(img)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.int16, np.float64])
@pytest.mark.parametrize(
    "operation",
    [
        lambda img: add(img, 0),
        lambda img: add_constant(img, 1),
        lambda img: add_vector(img, np.array([1])),
        lambda img: add_array(img, np.ones_like(img)),
        lambda img: multiply(img, 1),
        lambda img: multiply_by_constant(img, 1),
        lambda img: multiply_by_vector(img, np.array([1])),
        lambda img: multiply_by_array(img, np.ones_like(img)),
        lambda img: multiply_add(img, 1, 0),
        lambda img: power(img, 1),
    ],
)
def test_arithmetic_rejects_unsupported_image_dtypes(
    operation: Callable[[np.ndarray], np.ndarray],
    dtype: type[np.generic],
) -> None:
    img = np.ones((2, 3, 1), dtype=dtype)

    with pytest.raises(ValueError, match="Albucore supports only uint8 and float32"):
        operation(img)


def test_clipped_decorator_remains_an_explicit_float32_range_constraint() -> None:
    @clipped
    def copy_image(img: np.ndarray) -> np.ndarray:
        return img.copy()

    img = np.array([[[-1.0], [2.0]]], dtype=np.float32)

    result = copy_image(img)

    np.testing.assert_array_equal(result, np.array([[[0.0], [1.0]]], dtype=np.float32))


@pytest.mark.parametrize("is_contiguous", [True, False])
def test_float32_scalar_multiply_add_routes_to_numkong(
    monkeypatch: pytest.MonkeyPatch,
    is_contiguous: bool,
) -> None:
    storage = np.ones((2, 3, 1), dtype=np.float32)
    img = storage if is_contiguous else storage[:, ::2, :]
    called = False

    def numkong_spy(source: np.ndarray, factor: float, value: float) -> np.ndarray:
        nonlocal called
        called = True
        return source * factor + value

    monkeypatch.setattr(arithmetic, "multiply_add_numkong", numkong_spy)

    result = multiply_add(img, 2.0, -0.5)

    assert called
    np.testing.assert_array_equal(result, img * 2.0 - 0.5)
