# ruff: noqa: S101

from collections.abc import Callable

import numpy as np
import pytest

from albucore import (
    add,
    arithmetic,
    multiply,
    multiply_add,
    power,
    torch_backend,
)
from albucore.utils import clipped


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (lambda img: add(img, 0.5), lambda img: img + 0.5),
        (lambda img: add(img, np.array([0.5], dtype=np.float32)), lambda img: img + 0.5),
        (lambda img: add(img, np.full_like(img, 0.5)), lambda img: img + 0.5),
        (lambda img: multiply(img, 2.0), lambda img: img * 2.0),
        (lambda img: multiply(img, np.array([2.0], dtype=np.float32)), lambda img: img * 2.0),
        (lambda img: multiply(img, np.full_like(img, 2.0)), lambda img: img * 2.0),
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
        lambda img: multiply(img, 1),
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
        (lambda img: add(img, np.array([10])), np.array([[[10], [255]]], dtype=np.uint8)),
        (lambda img: add(img, np.full_like(img, 10)), np.array([[[10], [255]]], dtype=np.uint8)),
        (lambda img: multiply(img, 2), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: multiply(img, np.array([2])), np.array([[[0], [255]]], dtype=np.uint8)),
        (lambda img: multiply(img, np.full_like(img, 2)), np.array([[[0], [255]]], dtype=np.uint8)),
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
    monkeypatch.setattr(torch_backend, "TORCH_CPU_BACKEND_ENABLED", False)
    storage = np.ones((2, 3, 1), dtype=np.float32)
    img = storage if is_contiguous else storage[:, ::2, :]
    called = False
    native_numkong = arithmetic._scale_numkong

    def numkong_spy(source: np.ndarray, factor: float, value: float, *, out=None) -> np.ndarray:
        nonlocal called
        called = True
        return native_numkong(source, factor, value, out=out)

    monkeypatch.setattr(arithmetic, "_scale_numkong", numkong_spy)

    result = multiply_add(img, 2.0, -0.5)

    assert called
    np.testing.assert_array_equal(result, img * 2.0 - 0.5)
