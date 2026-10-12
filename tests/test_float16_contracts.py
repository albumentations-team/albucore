"""Public float16 contracts across common pointwise image operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import torch

import albucore.geometric as geometric_mod
import albucore.stats as stats_mod
import albucore.torch_backend as torch_backend
from albucore import (
    add,
    add_constant,
    add_weighted,
    add_vector,
    arithmetic,
    copy_make_border,
    exp,
    flip_volume,
    gaussian_blur3d,
    log,
    matmul,
    mean,
    mean_std,
    multiply,
    multiply_by_constant,
    multiply_add,
    multiply_by_vector,
    normalize,
    normalize_per_image,
    pad3d,
    power,
    reduce_sum,
    remap,
    remap3d,
    resize,
    resize3d,
    rot90_volume,
    separable_filter3d,
    std,
    sqrt,
    transpose_volume,
    warp_affine,
    warp_affine3d,
    warp_perspective,
)
from albucore.tensor_bridge import tensor_to_numpy_channel_last
from albucore.utils import TensorLayout

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("name", "operation", "reference"),
    [
        ("add_scalar", lambda image, _other: add(image, 0.125), lambda image, _other: image + np.float32(0.125)),
        (
            "add_constant",
            lambda image, _other: add_constant(image, 0.125),
            lambda image, _other: image + np.float32(0.125),
        ),
        (
            "add_vector",
            lambda image, _other: add(image, np.array([0.125, 0.25, 0.375], dtype=np.float32)),
            lambda image, _other: image + np.array([0.125, 0.25, 0.375], dtype=np.float32),
        ),
        (
            "multiply_scalar",
            lambda image, _other: multiply(image, 1.125),
            lambda image, _other: image * np.float32(1.125),
        ),
        (
            "multiply_by_constant",
            lambda image, _other: multiply_by_constant(image, 1.125),
            lambda image, _other: image * np.float32(1.125),
        ),
        (
            "multiply_vector",
            lambda image, _other: multiply(image, np.array([1.125, 0.875, 0.75], dtype=np.float32)),
            lambda image, _other: image * np.array([1.125, 0.875, 0.75], dtype=np.float32),
        ),
        ("multiply_array", multiply, lambda image, other: image * other),
        (
            "multiply_add",
            lambda image, _other: multiply_add(image, 1.125, 0.125),
            lambda image, _other: image * np.float32(1.125) + np.float32(0.125),
        ),
        (
            "add_weighted",
            lambda image, other: add_weighted(image, 0.25, other, 0.75),
            lambda image, other: image * np.float32(0.25) + other * np.float32(0.75),
        ),
        ("power", lambda image, _other: power(image, 1.5), lambda image, _other: np.power(image, np.float32(1.5))),
    ],
    ids=(
        "add-scalar",
        "add-constant",
        "add-vector",
        "multiply-scalar",
        "multiply-by-constant",
        "multiply-vector",
        "multiply-array",
        "multiply-add",
        "add-weighted",
        "power",
    ),
)
def test_float16_pointwise_operations_preserve_dtype_and_match_float32_rounded_once(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    operation: Callable[..., np.ndarray],
    reference: Callable[..., np.ndarray],
) -> None:
    monkeypatch.setattr(arithmetic, "_TORCH_FLOAT16_MIN_ELEMENTS", 1)
    monkeypatch.setattr(arithmetic, "_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS", 1)
    fallbacks = ("add_numpy", "multiply_numpy", "multiply_add_numpy", "add_weighted_numpy", "power_numpy")
    for fallback in fallbacks:
        monkeypatch.setattr(
            arithmetic,
            fallback,
            lambda *_args, _name=fallback, **_kwargs: pytest.fail(f"{name} took the {_name} float32 fallback"),
        )

    source = np.linspace(0.0625, 0.875, num=19 * 23 * 3, dtype=np.float32).reshape(19, 23, 3).astype(np.float16)
    other = np.linspace(0.25, 0.75, num=source.size, dtype=np.float32).reshape(source.shape).astype(np.float16)

    result = operation(source, other)
    expected = reference(source.astype(np.float32), other.astype(np.float32)).astype(np.float16)

    assert result.dtype == np.float16, name
    assert result.shape == source.shape
    observed_ulps = np.abs(result.view(np.uint16).astype(np.int32) - expected.view(np.uint16).astype(np.int32))
    assert observed_ulps.max() <= 1, name


@pytest.mark.parametrize(
    ("operation", "control", "reference"),
    [
        (add_constant, 0.125, np.add),
        (multiply_by_constant, 1.125, np.multiply),
    ],
)
@pytest.mark.parametrize(
    ("layout", "numpy_shape", "tensor_permutation"),
    [
        ("CHW", (19, 23, 3), (2, 0, 1)),
        ("NCHW", (2, 19, 23, 3), (0, 3, 1, 2)),
    ],
)
def test_float16_tensor_scalar_operations_preserve_layout_and_dtype(
    operation: Callable[..., object],
    control: float,
    reference: Callable[..., np.ndarray],
    layout: TensorLayout,
    numpy_shape: tuple[int, ...],
    tensor_permutation: tuple[int, ...],
) -> None:
    source = np.linspace(0.0625, 0.875, num=int(np.prod(numpy_shape)), dtype=np.float32)
    source = source.reshape(numpy_shape).astype(np.float16)
    tensor = torch.from_numpy(source).permute(tensor_permutation)
    expected = reference(source.astype(np.float32), np.float32(control)).astype(np.float16)

    result = operation(tensor, control, layout=layout)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float16
    assert result.shape == tensor.shape
    actual = tensor_to_numpy_channel_last(result, layout)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("operation", "control", "reference"),
    [
        (add_vector, 0.125, np.add),
        (multiply_by_vector, 1.125, np.multiply),
    ],
)
def test_float16_single_channel_vector_controls_use_scalar_torch_parameter(
    operation: Callable[..., np.ndarray],
    control: float,
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    source = np.random.default_rng(610135).random((128, 160, 1), dtype=np.float32).astype(np.float16)
    vector = np.array([control], dtype=np.float32)
    expected = reference(source.astype(np.float32), vector).astype(np.float16)
    original = torch_backend._half_parameter
    parameters: list[torch.Tensor | float | None] = []

    def record_parameter(value: float | np.ndarray, array: np.ndarray) -> torch.Tensor | float | None:
        parameter = original(value, array)
        parameters.append(parameter)
        return parameter

    monkeypatch.setattr(torch_backend, "_half_parameter", record_parameter)
    result = operation(source, vector)

    np.testing.assert_array_equal(result, expected)
    assert len(parameters) == 1
    assert isinstance(parameters[0], float)


@pytest.mark.parametrize(
    ("operation", "control", "alpha", "beta", "reference"),
    [
        (add_vector, 0.0, 1.0, 0.0, np.add),
        (add_vector, -0.0, 1.0, -0.0, np.add),
        (add_vector, 0.5, 1.0, 0.5, np.add),
        (add_vector, -0.5, 1.0, -0.5, np.add),
        (add_vector, 0.875, 1.0, 0.875, np.add),
        (add_vector, -1.25, 1.0, -1.25, np.add),
        (add_vector, 1.0, 1.0, 1.0, np.add),
        (multiply_by_vector, 0.0, 0.0, -0.0, np.multiply),
        (multiply_by_vector, -0.0, -0.0, -0.0, np.multiply),
        (multiply_by_vector, 0.5, 0.5, -0.0, np.multiply),
        (multiply_by_vector, -0.5, -0.5, -0.0, np.multiply),
        (multiply_by_vector, 0.875, 0.875, -0.0, np.multiply),
        (multiply_by_vector, -1.25, -1.25, -0.0, np.multiply),
        (multiply_by_vector, 1.0, 1.0, -0.0, np.multiply),
    ],
)
def test_float16_single_channel_vector_uses_numkong_scale_for_exact_half_controls(
    operation: Callable[..., np.ndarray],
    control: float,
    alpha: float,
    beta: float,
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)
    calls: list[tuple[float, float]] = []
    original = arithmetic.nk.scale

    def record_numkong(img: np.ndarray, *, alpha: float, beta: float) -> np.ndarray:
        calls.append((alpha, beta))
        return original(img, alpha=alpha, beta=beta)

    monkeypatch.setattr(arithmetic.nk, "scale", record_numkong)
    image = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256, 1).view(np.float16)
    vector = np.array([control], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), np.float32(control)).astype(np.float16)

    result = operation(image, vector)

    assert calls == [(alpha, beta)]
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))


@pytest.mark.parametrize(
    ("operation", "shape", "control", "reference"),
    [
        (add_vector, (128, 160, 1), 0.1, np.add),
        (multiply_by_vector, (128, 160, 1), 0.1, np.multiply),
        (add_vector, (127, 160, 1), 0.5, np.add),
        (multiply_by_vector, (127, 160, 1), 0.875, np.multiply),
        (add_vector, (128, 160, 3), 0.5, np.add),
        (multiply_by_vector, (128, 160, 3), 0.875, np.multiply),
    ],
)
def test_float16_single_channel_numkong_vector_route_keeps_measurement_gates(
    operation: Callable[..., np.ndarray],
    shape: tuple[int, ...],
    control: float,
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)

    def fail_numkong(*args: object, **kwargs: object) -> np.ndarray:
        msg = "NumKong scale is limited to the measured contiguous HWC C=1 half-exact region."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic.nk, "scale", fail_numkong)
    image = np.random.default_rng(613).uniform(-0.75, 0.75, size=shape).astype(np.float16)
    vector = np.full((shape[-1],), control, dtype=np.float32)
    expected = reference(image.astype(np.float32), vector).astype(np.float16)

    result = operation(image, vector)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("operation", "opencv_name", "controls", "reference"),
    [
        (add_vector, "add", (0.5, 0.25, -0.5), np.add),
        (multiply_by_vector, "multiply", (0.875, 0.5, -0.75), np.multiply),
    ],
)
def test_float16_hwc_c3_vector_uses_cv16f_for_exact_controls(
    operation: Callable[..., np.ndarray],
    opencv_name: str,
    controls: tuple[float, float, float],
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    calls: list[tuple[tuple[float, ...], object]] = []
    original = getattr(arithmetic.cv2, opencv_name)

    def record_cv16f(img: np.ndarray, scalar: tuple[float, ...], **kwargs: object) -> np.ndarray:
        calls.append((scalar, kwargs.get("dtype")))
        return original(img, scalar, **kwargs)

    monkeypatch.setattr(arithmetic.cv2, opencv_name, record_cv16f)
    bits = np.arange(1 << 16, dtype=np.uint16)
    image = np.resize(bits, (256, 256, 3)).view(np.float16)
    vector = np.array(controls, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), vector).astype(np.float16)

    result = operation(image, vector)

    assert calls == [(controls + (0.0,), cv2.CV_16F)]
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))


@pytest.mark.parametrize(
    ("operation", "opencv_name", "control", "shape", "non_contiguous", "reference"),
    [
        (add_vector, "add", (0.5, 0.25, -0.5), (127, 160, 3), False, np.add),
        (multiply_by_vector, "multiply", (0.875, 0.5, -0.75), (127, 160, 3), False, np.multiply),
        (add_vector, "add", (0.1, 0.25, -0.5), (128, 160, 3), False, np.add),
        (multiply_by_vector, "multiply", (0.875, 0.1, -0.75), (128, 160, 3), False, np.multiply),
        (add_vector, "add", (0.5, 0.25, -0.5), (128, 160, 3), True, np.add),
        (multiply_by_vector, "multiply", (0.875, 0.5, -0.75), (128, 160, 3), True, np.multiply),
    ],
)
def test_float16_hwc_c3_vector_cv16f_route_keeps_measurement_gates(
    operation: Callable[..., np.ndarray],
    opencv_name: str,
    control: tuple[float, float, float],
    shape: tuple[int, int, int],
    non_contiguous: bool,
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cv16f(*args: object, **kwargs: object) -> np.ndarray:
        msg = "CV16F vector routing is limited to the measured contiguous HWC C=3 region."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic.cv2, opencv_name, fail_cv16f)
    image = np.random.default_rng(620).uniform(-0.75, 0.75, size=shape).astype(np.float16)
    if non_contiguous:
        image = np.ascontiguousarray(np.repeat(image, 2, axis=1))[:, ::2, :]
    vector = np.array(control, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), vector).astype(np.float16)

    result = operation(image, vector)

    np.testing.assert_array_equal(result, expected)


def test_float16_hwc_c9_multiply_uses_cv16f_broadcast_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    calls: list[tuple[tuple[int, ...], bool, object]] = []
    original = arithmetic.cv2.multiply

    def record_cv16f(img: np.ndarray, value: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append((value.strides, value.flags.writeable, kwargs.get("dtype")))
        return original(img, value, **kwargs)

    monkeypatch.setattr(arithmetic.cv2, "multiply", record_cv16f)
    bits = np.arange(1 << 16, dtype=np.uint16)
    image = np.resize(bits, (256, 320, 9)).view(np.float16)
    controls = (0.5, -0.25, 0.125, 0.375, -0.5, 0.75, -0.125, 0.875, 0.0)
    vector = np.array(controls, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.multiply(image.astype(np.float32), vector).astype(np.float16)

    result = multiply_by_vector(image, vector)

    assert calls == [((0, 0, 2), False, cv2.CV_16F)]
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))
    np.testing.assert_array_equal(np.signbit(result[expected == 0]), np.signbit(expected[expected == 0]))


def test_float16_hwc_c9_multiply_allows_safe_inexact_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[int, ...], object]] = []
    original = arithmetic.cv2.multiply

    def record_cv16f(img: np.ndarray, value: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append((value.strides, kwargs.get("dtype")))
        return original(img, value, **kwargs)

    monkeypatch.setattr(arithmetic.cv2, "multiply", record_cv16f)
    bits = np.arange(1 << 16, dtype=np.uint16)
    image = np.resize(bits, (256, 320, 9)).view(np.float16)
    vector = np.array([0.1, 0.3, -0.75, -0.3333333, 0.0001, 0.0000611, 0.99999, -0.01, 1.0], np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.multiply(image.astype(np.float32), vector).astype(np.float16)

    result = multiply_by_vector(image, vector)

    assert calls == [((0, 0, 2), cv2.CV_16F)]
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_max_ulp(result[finite], expected[finite], maxulp=1)
    np.testing.assert_array_equal(np.signbit(result[expected == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize(
    ("operation", "numpy_helper", "control", "reference"),
    [
        (add_vector, "add_numpy", np.array([0.5, -0.25, 0.125, 0.375, -0.5, 0.75, -0.125, 0.875, 0.25], np.float32), np.add),
        (
            multiply_by_vector,
            "multiply_numpy",
            np.array([0.875, 1.125, 0.5, -0.75, 1.25, 0.625, 0.25, -1.0, 1.1], np.float32),
            np.multiply,
        ),
    ],
)
def test_float16_hwc_c9_vector_uses_float32_ufunc_loop(
    operation: Callable[..., np.ndarray],
    numpy_helper: str,
    control: np.ndarray,
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    calls: list[tuple[tuple[int, ...], bool]] = []
    original_numpy = getattr(arithmetic, numpy_helper)

    def record_numpy(img: np.ndarray, value: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        calls.append((img.shape, inplace))
        return original_numpy(img, value, inplace=inplace)

    def fail_torch(*args: object, **kwargs: object) -> np.ndarray | None:
        msg = "the measured contiguous HWC C=9 vector route uses the float32 NumPy ufunc loop"
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic, numpy_helper, record_numpy)
    monkeypatch.setattr(arithmetic, "float16_pointwise_torch", fail_torch)
    image = np.random.default_rng(629).uniform(-0.75, 0.75, size=(480, 640, 9)).astype(np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), control).astype(np.float16)

    result = operation(image, control)

    np.testing.assert_array_equal(result, expected)
    assert calls == [((480, 640, 9), False)]


@pytest.mark.parametrize(
    ("operation", "numkong_name", "controls", "reference"),
    [
        (add_vector, "add", (0.5, -0.25, 0.125, 0.375, -0.5, 0.75, -0.125, 0.875, -0.0), np.add),
    ],
)
def test_float16_hwc_c9_vector_uses_numkong_broadcast_controls(
    operation: Callable[..., np.ndarray],
    numkong_name: str,
    controls: tuple[float, ...],
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)
    observed: list[tuple[tuple[int, ...], tuple[int, ...], bool]] = []
    original = getattr(arithmetic.nk, numkong_name)

    def record_numkong(image: np.ndarray, control: np.ndarray) -> object:
        observed.append((control.shape, control.strides, control.flags.writeable))
        return original(image, control)

    monkeypatch.setattr(arithmetic.nk, numkong_name, record_numkong)
    bits = np.arange(1 << 16, dtype=np.uint16)
    image = np.resize(bits, (128, 160, 9)).view(np.float16)
    vector = np.array(controls, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), vector).astype(np.float16)

    result = operation(image, vector)

    assert observed == [((128, 160, 9), (0, 0, 2), False)]
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))


def test_float16_hwc_c9_multiply_zero_control_preserves_negative_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)

    def fail_numkong(*args: object, **kwargs: object) -> np.ndarray:
        msg = "NumKong multiply does not preserve negative zero with a zero broadcast control."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic.nk, "multiply", fail_numkong)
    image = np.zeros((128, 160, 9), dtype=np.float16)
    image[..., -1] = np.float16(-0.0)
    vector = np.array([0.875, 1.125, 0.5, -0.75, 1.25, 0.625, 0.25, -1.0, 0.0], dtype=np.float32)
    expected = np.multiply(image.astype(np.float32), vector).astype(np.float16)

    result = multiply_by_vector(image, vector)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
    assert np.all(np.signbit(result[..., -1]))


def test_float16_hwc_c9_multiply_keeps_numpy_route_with_neonhalf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)
    calls: list[tuple[int, ...]] = []
    original_numpy = arithmetic.multiply_numpy

    def record_numpy(img: np.ndarray, value: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        calls.append(img.shape)
        return original_numpy(img, value, inplace=inplace)

    def fail_torch(*args: object, **kwargs: object) -> np.ndarray | None:
        raise AssertionError("C9 float32 ufunc output path should be selected before Torch half")

    monkeypatch.setattr(arithmetic, "multiply_numpy", record_numpy)
    monkeypatch.setattr(arithmetic, "float16_pointwise_torch", fail_torch)
    monkeypatch.setattr(arithmetic.nk, "multiply", fail_torch)
    image = np.random.default_rng(632).uniform(-0.75, 0.75, size=(480, 640, 9)).astype(np.float16)
    vector = np.array([0.875, 1.125, 0.5, -0.75, 1.25, 0.625, 0.25, -1.0, 1.1], dtype=np.float32)
    expected = np.multiply(image.astype(np.float32), vector).astype(np.float16)

    result = multiply_by_vector(image, vector)

    np.testing.assert_array_equal(result, expected)
    assert calls == [image.shape]


def test_float16_hwc_c9_multiply_falls_back_for_inexact_subnormal_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, ...]] = []
    original_numpy = arithmetic.multiply_numpy

    def fail_cv16f(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("an inexact half-subnormal control must keep the float32 working path")

    def record_numpy(img: np.ndarray, value: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        calls.append(img.shape)
        return original_numpy(img, value, inplace=inplace)

    monkeypatch.setattr(arithmetic.cv2, "multiply", fail_cv16f)
    monkeypatch.setattr(arithmetic, "multiply_numpy", record_numpy)
    image = np.random.default_rng(633).uniform(-0.75, 0.75, size=(128, 160, 9)).astype(np.float16)
    vector = np.array([0.5, -0.25, 0.125, 0.375, -0.5, 0.75, -0.125, 0.875, 0.00006], dtype=np.float32)
    expected = np.multiply(image.astype(np.float32), vector).astype(np.float16)

    result = multiply_by_vector(image, vector)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
    assert calls == [image.shape]


@pytest.mark.parametrize(
    ("operation", "control", "reference"),
    [
        (add_vector, np.array([0.5, -0.25, 0.125, 0.375, -0.5, 0.75, -0.125, 0.875, 0.25], np.float32), np.add),
        (
            multiply_by_vector,
            np.array([0.875, 1.125, 0.5, -0.75, 1.25, 0.625, 0.25, -1.0, 1.5], np.float32),
            np.multiply,
        ),
    ],
)
def test_float16_hwc_c9_vector_keeps_torch_path_below_numpy_threshold(
    operation: Callable[..., np.ndarray],
    control: np.ndarray,
    reference: Callable[..., np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original_torch = arithmetic.float16_pointwise_torch

    def record_torch(img: np.ndarray, operation_name: str, value: float | np.ndarray, *, minimum_elements: int) -> np.ndarray | None:
        calls.append(operation_name)
        return original_torch(img, operation_name, value, minimum_elements=minimum_elements)

    monkeypatch.setattr(arithmetic, "float16_pointwise_torch", record_torch)
    image = np.random.default_rng(630).uniform(-0.75, 0.75, size=(127, 160, 9)).astype(np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), control).astype(np.float16)

    result = operation(image, control)

    np.testing.assert_array_equal(result, expected)
    assert calls == ["add" if operation is add_vector else "multiply"]


def test_float16_single_channel_nonrepresentable_vector_matches_wide_result() -> None:
    source = np.random.default_rng(610140).random((128, 160, 1), dtype=np.float32).astype(np.float16)
    vector = np.array([0.1], dtype=np.float32)

    result = add_vector(source, vector)
    expected = np.add(source.astype(np.float32), vector).astype(np.float16)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [(add, np.add), (multiply, np.multiply)],
)
def test_float16_full_float32_array_operands_match_wide_reference(
    operation: Callable[..., np.ndarray],
    reference: Callable[..., np.ndarray],
) -> None:
    source = np.random.default_rng(20261010).uniform(-0.75, 0.75, size=(240, 320, 3)).astype(np.float16)
    control = np.random.default_rng(20261011).uniform(0.75, 1.25, size=source.shape).astype(np.float32)
    expected = reference(source.astype(np.float32), control).astype(np.float16)

    result = operation(source, control)

    assert result.dtype == np.float16
    np.testing.assert_array_max_ulp(result, expected, maxulp=1)


@pytest.mark.parametrize("normalization", ["image", "image_per_channel", "min_max", "min_max_per_channel"])
def test_float16_per_image_normalization_uses_float32_work_and_restores_half(normalization: str) -> None:
    source = np.linspace(0.05, 0.95, num=29 * 31 * 3, dtype=np.float32).reshape(29, 31, 3).astype(np.float16)
    reference = source.astype(np.float32)

    result = normalize_per_image(source, normalization)
    expected = normalize_per_image(reference, normalization).astype(np.float16)

    assert result.dtype == np.float16
    assert result.shape == source.shape
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("normalization", ["image", "image_per_channel", "min_max", "min_max_per_channel"])
@pytest.mark.parametrize(
    ("layout", "numpy_shape", "tensor_permutation"),
    [
        ("CHW", (7, 9, 3), (2, 0, 1)),
        ("NCHW", (2, 7, 9, 3), (0, 3, 1, 2)),
        ("CDHW", (4, 7, 9, 3), (3, 0, 1, 2)),
        ("NCDHW", (2, 4, 7, 9, 3), (0, 4, 1, 2, 3)),
    ],
)
def test_float16_tensor_per_image_normalization_preserves_layout_and_dtype(
    normalization: str,
    layout: TensorLayout,
    numpy_shape: tuple[int, ...],
    tensor_permutation: tuple[int, ...],
) -> None:
    source = np.linspace(0.05, 0.95, num=int(np.prod(numpy_shape)), dtype=np.float32)
    source = source.reshape(numpy_shape).astype(np.float16)
    tensor = torch.from_numpy(source).permute(tensor_permutation)
    samples = source if layout in ("NCHW", "NCDHW") else source[None]
    expected = np.stack(
        [normalize_per_image(sample.astype(np.float32), normalization).astype(np.float16) for sample in samples],
    )

    result = normalize_per_image(tensor, normalization, layout=layout)

    assert result.dtype == torch.float16
    assert result.shape == tensor.shape
    actual = tensor_to_numpy_channel_last(result, layout)
    if layout in ("CHW", "CDHW"):
        actual = actual[None]
    np.testing.assert_array_equal(actual, expected)


def test_float16_fixed_normalization_preserves_dtype_and_identity_alias() -> None:
    source = np.linspace(-0.5, 1.5, num=19 * 23 * 3, dtype=np.float32).reshape(19, 23, 3).astype(np.float16)

    identity = normalize(source, 0.0, 1.0)
    actual = normalize(source, np.array([0.125, 0.25, 0.375], dtype=np.float32), 2.0)
    expected = ((source.astype(np.float32) - [0.125, 0.25, 0.375]) * 2.0).astype(np.float16)

    assert identity is source
    assert actual.dtype == np.float16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("channels", [3, 9])
def test_float16_fixed_normalization_matches_wide_result_within_one_ulp(channels: int) -> None:
    source = np.linspace(0.0625, 0.9375, num=128 * 160 * channels, dtype=np.float32)
    source = source.reshape(128, 160, channels).astype(np.float16)
    mean = np.resize(np.array([0.375, 0.5, 0.625], dtype=np.float32), channels)
    denominator = np.resize(np.array([1.5, 1.75, 2.0], dtype=np.float32), channels)
    expected = normalize(source.astype(np.float32), mean, denominator).astype(np.float16)

    result = normalize(source, mean, denominator)

    assert result.dtype == np.float16
    np.testing.assert_array_max_ulp(result, expected, maxulp=1)


@pytest.mark.parametrize("shape", [(128, 160, 1), (240, 320, 1)])
def test_float16_single_channel_normalization_matches_wide_result_within_one_ulp(
    shape: tuple[int, ...],
) -> None:
    source = np.linspace(-0.5, 1.5, num=int(np.prod(shape)), dtype=np.float32).reshape(shape).astype(np.float16)
    expected = normalize(source.astype(np.float32), 0.375, 1.5).astype(np.float16)

    result = normalize(source, 0.375, 1.5)

    assert result.dtype == np.float16
    np.testing.assert_array_max_ulp(result, expected, maxulp=1)


@pytest.mark.parametrize(
    ("layout", "numpy_shape", "tensor_permutation"),
    [
        ("CHW", (128, 160, 3), (2, 0, 1)),
        ("NCHW", (2, 7, 11, 3), (0, 3, 1, 2)),
        ("CDHW", (4, 7, 11, 3), (3, 0, 1, 2)),
        ("NCDHW", (2, 4, 7, 11, 3), (0, 4, 1, 2, 3)),
    ],
)
def test_float16_normalize_tensor_preserves_layout_and_matches_wide_result(
    layout: str,
    numpy_shape: tuple[int, ...],
    tensor_permutation: tuple[int, ...],
) -> None:
    source = np.linspace(0.0625, 0.9375, num=int(np.prod(numpy_shape)), dtype=np.float32)
    source = source.reshape(numpy_shape).astype(np.float16)
    tensor = torch.from_numpy(source).permute(tensor_permutation)
    mean = np.array([0.375, 0.5, 0.625], dtype=np.float32)
    denominator = np.array([1.5, 1.75, 2.0], dtype=np.float32)
    expected = normalize(source.astype(np.float32), mean, denominator).astype(np.float16)

    result = normalize(tensor, mean, denominator, layout=layout)

    assert result.dtype == torch.float16
    assert tuple(result.shape) == tuple(tensor.shape)
    actual = tensor_to_numpy_channel_last(result, layout)
    np.testing.assert_array_max_ulp(actual, expected, maxulp=1)


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("CHW", (3, 128, 160)),
        ("CHW", (9, 128, 160)),
        ("CHW", (3, 240, 320)),
        ("NCHW", (4, 3, 128, 160)),
        ("NCHW", (4, 9, 128, 160)),
    ],
)
def test_float16_channel_first_normalize_matches_wide_result(layout: str, shape: tuple[int, ...]) -> None:
    if layout == "CHW":
        source_shape = (shape[1], shape[2], shape[0])
        permutation = (2, 0, 1)
        channels = shape[0]
    else:
        source_shape = (shape[0], shape[2], shape[3], shape[1])
        permutation = (0, 3, 1, 2)
        channels = shape[1]
    source = np.random.default_rng(20261011).random(source_shape, dtype=np.float32).astype(np.float16)
    tensor = torch.from_numpy(source).permute(permutation).contiguous()
    mean = np.linspace(0.375, 0.5, num=channels, dtype=np.float32)
    denominator = np.linspace(1.5, 1.75, num=channels, dtype=np.float32)
    expected = normalize(tensor.to(torch.float32), mean, denominator, layout=layout).to(torch.float16)

    result = normalize(tensor, mean, denominator, layout=layout)

    assert result.dtype == torch.float16
    assert tuple(result.shape) == shape
    actual_array = tensor_to_numpy_channel_last(result, layout)
    expected_array = tensor_to_numpy_channel_last(expected, layout)
    np.testing.assert_array_max_ulp(actual_array, expected_array, maxulp=1)


def test_float16_channel_first_normalize_scalar_controls_match_wide_result() -> None:
    source = np.random.default_rng(20261012).random((128, 160, 3), dtype=np.float32).astype(np.float16)
    tensor = torch.from_numpy(source).permute(2, 0, 1).contiguous()
    expected = normalize(tensor.to(torch.float32), 0.375, 1.5, layout="CHW").to(torch.float16)

    result = normalize(tensor, 0.375, 1.5, layout="CHW")

    np.testing.assert_array_max_ulp(result.numpy(), expected.numpy(), maxulp=1)


def test_float16_channel_first_normalize_preserves_binary16_values() -> None:
    bits = np.arange(65_536, dtype=np.uint16).reshape(256, 256)
    source = np.stack(
        (
            bits.view(np.float16),
            np.roll(bits, 1, axis=0).view(np.float16),
            np.roll(bits, 2, axis=1).view(np.float16),
        ),
        axis=0,
    )[None].copy()
    tensor = torch.from_numpy(source)
    mean = np.array([0.375, 0.5, 0.625], dtype=np.float32)
    denominator = np.array([1.5, 1.75, 2.0], dtype=np.float32)

    with np.errstate(invalid="ignore", over="ignore"):
        expected = normalize(tensor.to(torch.float32), mean, denominator, layout="NCHW").to(torch.float16).numpy()
        actual = normalize(tensor, mean, denominator, layout="NCHW").numpy()

    finite = np.isfinite(expected) & np.isfinite(actual)
    np.testing.assert_array_equal(actual[finite].view(np.uint16), expected[finite].view(np.uint16))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(actual[np.isinf(actual)]), np.signbit(expected[np.isinf(expected)]))


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("CHW", (3, 5, 7)),
        ("NCHW", (2, 3, 5, 7)),
        ("CDHW", (3, 4, 5, 7)),
        ("NCDHW", (2, 3, 4, 5, 7)),
    ],
)
def test_float16_tensor_normalize_identity_returns_original_tensor(layout: str, shape: tuple[int, ...]) -> None:
    image = torch.ones(shape, dtype=torch.float16)

    assert normalize(image, 0.0, 1.0, layout=layout) is image


def test_float16_per_channel_mean_supports_high_channel_count() -> None:
    source = np.linspace(0.0, 1.0, num=31 * 37 * 9, dtype=np.float32).reshape(31, 37, 9).astype(np.float16)

    result = mean(source, "per_channel")

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, source.astype(np.float32).mean(axis=(0, 1), dtype=np.float64))


def test_float16_small_skewed_per_channel_mean_uses_wide_accumulator() -> None:
    source = np.full((5, 7, 1), np.float16(1e-5), dtype=np.float16)
    source[0] = np.float16(1.0)
    source[1, 0, 0] = np.float16(1.0)
    expected = np.mean(source, axis=(0, 1), dtype=np.float64)

    result = mean(source, "per_channel")

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(64, 73, 3), (2, 31, 37, 9)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_global_mean_matches_wide_reference(shape: tuple[int, ...], keepdims: bool) -> None:
    source = np.random.default_rng(97).uniform(-1, 1, size=shape).astype(np.float16)
    expected = np.mean(source, dtype=np.float64, keepdims=keepdims)

    result = mean(source, "global", keepdims=keepdims)

    assert np.asarray(result).dtype == np.float64
    assert np.shape(result) == expected.shape
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    ("first", "expected"),
    [(np.inf, np.inf), (np.nan, np.nan), (np.inf, np.nan)],
)
def test_float16_global_mean_preserves_nonfinite_semantics(first: float, expected: float) -> None:
    source = np.zeros((64, 64, 1), dtype=np.float16)
    source.flat[0] = first
    if first == np.inf and np.isnan(expected):
        source.flat[1] = -np.inf

    result = mean(source, "global")

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("channels", [1, 3, 4, 5, 9])
def test_float16_hwc_statistics_match_wide_reference(channels: int) -> None:
    source = np.random.default_rng(97).uniform(-1, 1, size=(240, 320, channels)).astype(np.float16)
    expected_mean = np.mean(source, axis=(0, 1), dtype=np.float64)
    expected_std = np.std(source, axis=(0, 1), dtype=np.float64) + 1e-4

    actual_mean = mean(source, "per_channel")
    actual_std = std(source, "per_channel")
    actual_mean_std = mean_std(source, "per_channel")

    assert actual_mean.dtype == np.float64
    assert actual_std.dtype == np.float64
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(actual_mean_std[0], expected_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_mean_std[1], expected_std, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("channels", [1, 3, 4, 9])
def test_float16_constant_statistics_preserve_per_channel_values(channels: int) -> None:
    channel_values = np.linspace(0.25, 0.9, num=channels, dtype=np.float32).astype(np.float16)
    source = np.broadcast_to(channel_values, (128, 160, channels)).copy()
    expected_global_mean = np.mean(source, dtype=np.float64)
    expected_mean = np.mean(source, axis=(0, 1), dtype=np.float64)
    expected_std = np.std(source, axis=(0, 1), dtype=np.float64) + 1e-4

    actual_global_mean = mean(source, "global")
    actual_mean = mean(source, "per_channel")
    actual_std = std(source, "per_channel")
    actual_mean_std = mean_std(source, "per_channel")

    np.testing.assert_allclose(actual_global_mean, expected_global_mean, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_std, expected_std, rtol=0.0, atol=5e-9)
    np.testing.assert_array_equal(actual_mean_std[0], expected_mean)
    np.testing.assert_allclose(actual_mean_std[1], expected_std, rtol=0.0, atol=5e-9)


@pytest.mark.parametrize(
    "shape",
    [(128, 160, 3), (64, 73, 4), (4, 40, 43, 3), (4, 31, 37, 9), (2, 3, 4, 5, 7)],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_global_mean_std_matches_wide_reference(shape: tuple[int, ...], keepdims: bool) -> None:
    source = np.random.default_rng(97).uniform(-1, 1, size=shape).astype(np.float16)
    expected_mean = np.mean(source, dtype=np.float64, keepdims=keepdims)
    expected_std = np.std(source, dtype=np.float64, keepdims=keepdims) + 1e-4

    actual_mean, actual_std = mean_std(source, "global", keepdims=keepdims)

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(actual_std, expected_std, rtol=1e-6, atol=2e-6)


def test_float16_global_mean_std_uses_stable_fallback_for_large_nearly_constant_input() -> None:
    source = np.full((128, 160, 3), np.float16(0.5), dtype=np.float16)
    source.flat[0] = np.nextafter(source.flat[0], np.float16(np.inf))
    expected_mean = np.mean(source, dtype=np.float64)
    expected_std = np.std(source, dtype=np.float64) + 1e-4

    actual_mean, actual_std = mean_std(source, "global")

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_std, expected_std, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [np.inf, -np.inf]])
def test_float16_global_mean_std_preserves_nonfinite_semantics(values: list[float]) -> None:
    source = np.zeros((64, 64, 3), dtype=np.float16)
    source.flat[: len(values)] = values

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        expected_mean = np.mean(source, dtype=np.float64)
        expected_std = np.std(source, dtype=np.float64) + 1e-4
        actual_mean, actual_std = mean_std(source, "global")

    np.testing.assert_equal(actual_mean, expected_mean)
    np.testing.assert_equal(actual_std, expected_std)


@pytest.mark.parametrize(
    ("height", "width", "channels"),
    [
        (128, 160, 1),
        (128, 160, 3),
        (128, 160, 4),
        (128, 160, 5),
        (128, 160, 9),
        (240, 320, 1),
        (240, 320, 3),
        (240, 320, 4),
        (240, 320, 9),
    ],
)
def test_float16_per_channel_reduce_sum_preserves_wide_accumulator(
    height: int,
    width: int,
    channels: int,
) -> None:
    source = np.random.default_rng(74).random((height, width, channels), dtype=np.float32).astype(np.float16)
    expected = np.sum(source, axis=(0, 1), dtype=np.float64)

    result = reduce_sum(source, "per_channel")

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_rank4_c3_per_channel_sum_uses_wide_torch_accumulator(
    batch_size: int,
    keepdims: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.random.default_rng(75 + batch_size).random((batch_size, 128, 160, 3), dtype=np.float32).astype(np.float16)
    axes = (0, 1, 2)
    expected = np.sum(source, axis=axes, dtype=np.float64, keepdims=keepdims)
    original = stats_mod.reduce_sum_torch
    minimums: list[int | None] = []

    def record_torch_route(
        array: np.ndarray,
        reduction_axes: tuple[int, ...],
        *,
        keepdims: bool,
        minimum_elements: int | None = None,
    ) -> np.generic | np.ndarray | None:
        minimums.append(minimum_elements)
        return original(array, reduction_axes, keepdims=keepdims, minimum_elements=minimum_elements)

    monkeypatch.setattr(stats_mod, "reduce_sum_torch", record_torch_route)
    result = reduce_sum(source, "per_channel", keepdims=keepdims)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)
    assert minimums == [20_480]


@pytest.mark.parametrize(("height", "width"), [(128, 160), (240, 320)])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_rank4_c1_per_channel_sum_uses_measured_torch_window(
    height: int,
    width: int,
    batch_size: int,
    keepdims: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.random.default_rng(80 + batch_size + height).random(
        (batch_size, height, width, 1),
        dtype=np.float32,
    ).astype(np.float16)
    axes = (0, 1, 2)
    expected = np.sum(source, axis=axes, dtype=np.float64, keepdims=keepdims)
    original = stats_mod.torch.sum
    calls: list[dict[str, object]] = []

    def record_torch_sum(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(stats_mod.torch, "sum", record_torch_sum)
    result = reduce_sum(source, "per_channel", keepdims=keepdims)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)
    assert calls == ([] if keepdims else [{"dim": axes, "dtype": torch.float64}])


def test_float16_rank4_c3_per_channel_sum_preserves_subnormal_residuals() -> None:
    source = np.full((4, 128, 160, 3), np.float16(2**-24), dtype=np.float16)
    source[0, 0, 0, :] = np.float16(65_504)
    source[-1, -1, -1, :] = np.float16(-65_504)
    expected = np.sum(source, axis=(0, 1, 2), dtype=np.float64)

    result = reduce_sum(source, "per_channel")

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(240, 320, 9), (4, 128, 160, 9)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_c9_per_channel_sum_uses_wide_cv_accumulator(
    shape: tuple[int, ...],
    keepdims: bool,
) -> None:
    source = np.random.default_rng(74).random(shape, dtype=np.float32).astype(np.float16)
    axes = tuple(range(source.ndim - 1))
    expected = np.sum(source, axis=axes, dtype=np.float64, keepdims=keepdims)

    result = reduce_sum(source, "per_channel", keepdims=keepdims)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("axis", ["global", "per_channel"])
def test_float16_nearly_constant_mean_std_uses_stable_variance(axis: str) -> None:
    source = np.full((31, 37, 2), np.float16(0.8604), dtype=np.float16)
    expected_mean = np.mean(source, axis=None if axis == "global" else (0, 1), dtype=np.float64)
    expected_std = np.std(source, axis=None if axis == "global" else (0, 1), dtype=np.float64) + 1e-4

    actual_mean, actual_std = mean_std(source, axis)

    np.testing.assert_array_equal(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_std, expected_std, rtol=0.0, atol=1e-8)


def test_float16_matmul_matches_wide_accumulation_and_preserves_dtype() -> None:
    source = np.linspace(-1.0, 1.0, num=128 * 64, dtype=np.float32).reshape(128, 64).astype(np.float16)
    other = np.linspace(0.75, -0.75, num=64 * 32, dtype=np.float32).reshape(64, 32).astype(np.float16)

    result = matmul(source, other)
    expected = (source.astype(np.float32) @ other.astype(np.float32)).astype(np.float16)

    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
def test_float16_tensor_nearest_resize_uses_native_half_kernel(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if layout == "CHW":
        image = torch.from_numpy(np.linspace(-1.0, 1.0, 3 * 128 * 160, dtype=np.float32).reshape(3, 128, 160).astype(np.float16))
        dsize = (80, 64)
    else:
        bits = np.arange(65_536, dtype=np.uint16).reshape(1, 1, 256, 256)
        image = torch.from_numpy(bits.view(np.float16))
        dsize = (173, 119)

    layout_before = image.clone()
    with torch.no_grad():
        expected = resize(image.to(torch.float32), dsize, interpolation=cv2.INTER_NEAREST, layout=layout).to(torch.float16)

    interpolate = geometric_mod.torch_f.interpolate
    input_dtypes: list[torch.dtype] = []

    def record_interpolate(input_tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        input_dtypes.append(input_tensor.dtype)
        assert kwargs.get("mode") == "nearest"
        return interpolate(input_tensor, *args, **kwargs)

    def reject_numpy_bridge(*args: object, **kwargs: object) -> None:
        raise AssertionError("The selected float16 nearest route should not use the NumPy bridge.")

    monkeypatch.setattr(geometric_mod.torch_f, "interpolate", record_interpolate)
    monkeypatch.setattr(geometric_mod, "dispatch_tensor_image", reject_numpy_bridge)
    with torch.no_grad():
        result = resize(image, dsize, interpolation=cv2.INTER_NEAREST, layout=layout)

    assert input_dtypes == [torch.float16]
    assert result.dtype == torch.float16
    assert result.shape == expected.shape
    assert torch.equal(image.view(torch.int16), layout_before.view(torch.int16))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(result.view(torch.int16)[finite], expected.view(torch.int16)[finite], rtol=0, atol=0)
    assert torch.equal(torch.isnan(result), torch.isnan(expected))
    assert torch.equal(torch.isposinf(result), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(result), torch.isneginf(expected))


@pytest.mark.parametrize("axis", ["global", "per_channel"])
def test_float16_sum_uses_wide_results(axis: str) -> None:
    source = np.linspace(0.0, 1.0, num=4 * 240 * 320 * 3, dtype=np.float32).reshape(4, 240, 320, 3).astype(np.float16)
    axes = None if axis == "global" else (0, 1, 2)

    result = reduce_sum(source, axis)
    expected = np.sum(source, axis=axes, dtype=np.float64)

    assert np.asarray(result).dtype == np.float64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(("operation", "reference"), [(exp, np.exp), (log, np.log), (sqrt, np.sqrt)])
def test_float16_transcendentals_preserve_special_values(
    operation: Callable[..., np.ndarray],
    reference: Callable[..., np.ndarray],
) -> None:
    source = np.array([[[-2.0], [-0.0], [0.0], [0.0001], [0.25], [1.0], [8.0]]], dtype=np.float16)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        result = operation(source)
        expected = reference(source.astype(np.float32)).astype(np.float16)

    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize(("operation", "reference"), [(exp, np.exp), (log, np.log), (sqrt, np.sqrt)])
def test_float16_tensor_transcendentals_preserve_dtype_and_match_wide_reference(
    operation: Callable[..., object],
    reference: Callable[..., np.ndarray],
) -> None:
    source = np.linspace(0.001, 8.0, num=2 * 3 * 7 * 9, dtype=np.float32).reshape(2, 3, 7, 9).astype(np.float16)
    tensor = torch.from_numpy(source)
    expected = reference(source.astype(np.float32)).astype(np.float16)

    result = operation(tensor)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float16
    assert result.shape == tensor.shape
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize(("operation", "reference"), [(exp, np.exp), (log, np.log), (sqrt, np.sqrt)])
def test_uint8_tensor_transcendentals_use_raw_values_and_return_float32(
    operation: Callable[..., object],
    reference: Callable[..., np.ndarray],
) -> None:
    tensor = torch.tensor([0, 1, 4, 16, 255], dtype=torch.uint8).reshape(1, 1, 5, 1)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        expected = reference(tensor.numpy().astype(np.float32))
        result = operation(tensor)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_float16_log_bridge_preserves_special_values_for_large_arrays() -> None:
    values = np.array(
        [-2.0, -0.0, 0.0, np.nextafter(np.float16(0), np.float16(1)), 0.25, 1.0, np.inf, np.nan],
        dtype=np.float16,
    )
    source = np.resize(values, (64, 64, 1))

    with np.errstate(invalid="ignore", divide="ignore"):
        result = log(source)
        expected = np.log(source.astype(np.float32)).astype(np.float16)

    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize(("operation", "reference"), [(exp, np.exp), (log, np.log), (sqrt, np.sqrt)])
def test_uint8_transcendentals_apply_to_raw_values_and_return_float32(
    operation: Callable[..., np.ndarray],
    reference: Callable[..., np.ndarray],
) -> None:
    source = np.array([[[0], [1], [4], [16], [255]]], dtype=np.uint8)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        result = operation(source)
        expected = reference(source.astype(np.float32))

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (exp, np.exp),
        (log, np.log),
        (sqrt, np.sqrt),
    ],
)
def test_float16_transcendental_routes_match_one_ulp(
    operation: Callable[..., np.ndarray],
    reference: Callable[..., np.ndarray],
) -> None:
    source = np.linspace(0.125, 1.5, num=128 * 160 * 3, dtype=np.float32).reshape(128, 160, 3).astype(np.float16)

    result = operation(source)
    expected = reference(source.astype(np.float32)).astype(np.float16)
    ulps = np.abs(result.view(np.uint16).astype(np.int32) - expected.view(np.uint16).astype(np.int32))

    assert result.dtype == np.float16
    assert ulps.max() <= 1


@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_float16_volume_operations_preserve_dtype_and_widen_compute(container: str) -> None:
    volume_array = np.linspace(0.05, 0.95, num=4 * 7 * 9 * 3, dtype=np.float32).reshape(4, 7, 9, 3).astype(np.float16)
    volume = volume_array if container == "numpy" else torch.from_numpy(volume_array).permute(3, 0, 1, 2)

    def to_array(result: np.ndarray | torch.Tensor) -> np.ndarray:
        return result if isinstance(result, np.ndarray) else result.permute(1, 2, 3, 0).numpy()

    kernels = tuple(np.array([0.25, 0.5, 0.25], dtype=np.float32) for _ in range(3))
    affine = np.array([[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    depth, height, width, _ = volume_array.shape
    x = (np.arange(width, dtype=np.float32) + 0.5) * (2.0 / width) - 1.0
    y = (np.arange(height, dtype=np.float32) + 0.5) * (2.0 / height) - 1.0
    z = (np.arange(depth, dtype=np.float32) + 0.5) * (2.0 / depth) - 1.0
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    grid = np.stack((xx + 0.05, yy, zz), axis=-1).astype(np.float32)

    results = [
        separable_filter3d(volume, kernels),
        gaussian_blur3d(volume, sigma=(0.0, 0.5, 0.5), kernel_size=(0, 3, 3)),
        pad3d(volume, (1, 0, 1, 0, 1, 0), value=0.25),
        resize3d(volume, (5, 8, 11)),
        warp_affine3d(volume, affine, (depth, height, width)),
        remap3d(volume, grid),
    ]
    expected_dtype = np.float16 if container == "numpy" else torch.float16
    assert all(result.dtype == expected_dtype for result in results)

    for axes in ((2, 3), (0, 2)):
        assert flip_volume(volume, axes[0]).dtype == expected_dtype
        assert transpose_volume(volume, *axes).dtype == expected_dtype
        assert rot90_volume(volume, 1, axes).dtype == expected_dtype

    expected_filter = separable_filter3d(volume_array.astype(np.float32), kernels).astype(np.float16)
    np.testing.assert_array_equal(to_array(results[0]), expected_filter)
    expected_pad = pad3d(volume_array, (1, 0, 1, 0, 1, 0), value=0.25)
    np.testing.assert_array_equal(to_array(results[2]), expected_pad)


@pytest.mark.parametrize("operation", [warp_affine3d, remap3d], ids=("affine", "remap"))
@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_float16_volume_batch_resampling_preserves_container_and_dtype(
    operation: Callable[..., np.ndarray],
    container: str,
) -> None:
    volume_array = (
        np.linspace(0.05, 0.95, num=2 * 4 * 7 * 9 * 3, dtype=np.float32).reshape(2, 4, 7, 9, 3).astype(np.float16)
    )
    volume = volume_array if container == "numpy" else torch.from_numpy(volume_array).permute(0, 4, 1, 2, 3)
    matrix = np.array([[1.0, 0.0, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    if operation is warp_affine3d:
        result = operation(volume, matrix, (4, 7, 9))
        reference = operation(volume_array.astype(np.float32), matrix, (4, 7, 9)).astype(np.float16)
    else:
        width, height, depth = 9, 7, 4
        x = (np.arange(width, dtype=np.float32) + 0.5) * (2.0 / width) - 1.0
        y = (np.arange(height, dtype=np.float32) + 0.5) * (2.0 / height) - 1.0
        z = (np.arange(depth, dtype=np.float32) + 0.5) * (2.0 / depth) - 1.0
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        grid = np.stack((xx + 0.05, yy, zz), axis=-1).astype(np.float32)
        result = operation(volume, grid)
        reference = operation(volume_array.astype(np.float32), grid).astype(np.float16)

    result_array = result if isinstance(result, np.ndarray) else result.permute(0, 2, 3, 4, 1).numpy()
    assert result.dtype == (np.float16 if container == "numpy" else torch.float16)
    np.testing.assert_array_equal(result_array, reference)


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
def test_float16_tensor_2d_geometric_routers_preserve_layout_and_dtype(layout: str) -> None:
    batch_size = 2 if layout == "NCHW" else 1
    images = (
        np.linspace(0.05, 0.95, num=batch_size * 7 * 9 * 3, dtype=np.float32)
        .reshape(batch_size, 7, 9, 3)
        .astype(np.float16)
    )
    tensor = (
        torch.from_numpy(images).permute(0, 3, 1, 2)
        if layout == "NCHW"
        else torch.from_numpy(images[0]).permute(2, 0, 1)
    )
    affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.25]], dtype=np.float32)
    perspective = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]], dtype=np.float32)
    map_x, map_y = np.meshgrid(np.arange(9, dtype=np.float32), np.arange(7, dtype=np.float32))
    cases = [
        (resize(tensor, (11, 8), layout=layout), lambda image: resize(image, (11, 8))),
        (warp_affine(tensor, affine, (11, 8), layout=layout), lambda image: warp_affine(image, affine, (11, 8))),
        (
            warp_perspective(tensor, perspective, (11, 8), layout=layout),
            lambda image: warp_perspective(image, perspective, (11, 8)),
        ),
        (
            copy_make_border(tensor, 1, 2, 3, 0, value=0.25, layout=layout),
            lambda image: copy_make_border(image, 1, 2, 3, 0, value=0.25),
        ),
        (remap(tensor, map_x, map_y, layout=layout), lambda image: remap(image, map_x, map_y)),
    ]
    source_images = images if layout == "NCHW" else images[:1]
    for tensor_result, reference in cases:
        expected = np.stack([reference(image.astype(np.float32)).astype(np.float16) for image in source_images])
        result_array = (
            tensor_result.permute(0, 2, 3, 1).numpy()
            if layout == "NCHW"
            else tensor_result.permute(1, 2, 0).numpy()[None]
        )
        assert tensor_result.dtype == torch.float16
        assert result_array.shape == expected.shape
        np.testing.assert_array_equal(result_array, expected)


@pytest.mark.parametrize(("layout", "channels"), [("CHW", 3), ("CHW", 9), ("NCHW", 3), ("NCHW", 9)])
@pytest.mark.parametrize(
    ("height", "width", "dsize"),
    [(128, 160, (81, 61)), (240, 320, (201, 151)), (480, 640, (401, 301)), (768, 1024, (769, 481))],
)
def test_float16_contiguous_tensor_resize_matches_float32_reference(
    layout: str,
    channels: int,
    height: int,
    width: int,
    dsize: tuple[int, int],
) -> None:
    batch_size = 2 if layout == "NCHW" else 1
    source = np.random.default_rng(height + channels).random((batch_size, height, width, channels), dtype=np.float32).astype(
        np.float16,
    )
    tensor = (
        torch.from_numpy(source).permute(0, 3, 1, 2).contiguous()
        if layout == "NCHW"
        else torch.from_numpy(source[0]).permute(2, 0, 1).contiguous()
    )
    expected = np.stack([resize(image.astype(np.float32), dsize).astype(np.float16) for image in source])

    result = resize(tensor, dsize, layout=layout)

    actual = (
        result.permute(0, 2, 3, 1).numpy()
        if layout == "NCHW"
        else result.permute(1, 2, 0).numpy()[None]
    )
    assert result.dtype == torch.float16
    # OpenCV and Torch bilinear weights differ slightly; bound drift at unit image scale.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=float(np.finfo(np.float16).eps / 2))
