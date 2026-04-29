"""``apply_uint8_lut``: reference equality, volumes, errors, routing, arithmetic parity."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from albucore import apply_uint8_lut, multiply, multiply_add
from albucore.arithmetic import apply_lut, multiply_lut
from albucore.functions import multiply_add_lut
from albucore.lut import opencv_shared_uint8_lut_faster_hwc, sz_lut
from albucore.utils import clip


def _ref_shared(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[img]


def _ref_per_channel(img: np.ndarray, luts: np.ndarray) -> np.ndarray:
    c = img.shape[-1]
    return np.stack([luts[:, 0, i][img[..., i]] for i in range(c)], axis=-1)


# --- HWC (existing coverage expanded) ---


@pytest.mark.parametrize("shape", [(17, 19, 1), (32, 24, 3), (8, 16, 5), (4, 8, 9)])
@pytest.mark.parametrize("seed", [0, 1])
def test_apply_uint8_lut_shared_matches_numpy(shape: tuple[int, ...], seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    out = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, _ref_shared(img, lut))


@pytest.mark.parametrize("shape", [(11, 13, 3), (6, 7, 5), (4, 5, 9)])
@pytest.mark.parametrize("seed", [0, 2])
def test_apply_uint8_lut_per_channel_matches_numpy(shape: tuple[int, ...], seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    c = shape[-1]
    luts = rng.integers(0, 256, size=(256, 1, c), dtype=np.uint8)
    out = apply_uint8_lut(img, luts, inplace=False)
    np.testing.assert_array_equal(out, _ref_per_channel(img, luts))


def test_apply_uint8_lut_inplace_shared() -> None:
    rng = np.random.default_rng(3)
    img = np.ascontiguousarray(rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8))
    lut = np.arange(256, dtype=np.uint8)
    expected = apply_uint8_lut(img, lut, inplace=False)
    img2 = img.copy()
    out = apply_uint8_lut(img2, lut, inplace=True)
    assert out is img2  # noqa: S101
    np.testing.assert_array_equal(out, expected)


# --- DHWC / NDHWC ---


@pytest.mark.parametrize(
    "shape",
    [
        (5, 12, 14, 1),
        (4, 8, 9, 3),
        (3, 6, 7, 5),
    ],
)
@pytest.mark.parametrize("seed", [0, 3])
def test_apply_uint8_lut_shared_volume_matches_numpy(shape: tuple[int, ...], seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    out = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, _ref_shared(img, lut))


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 8, 10, 3),
        (2, 2, 5, 6, 1),
    ],
)
def test_apply_uint8_lut_shared_batch_volume_matches_numpy(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    out = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, _ref_shared(img, lut))


@pytest.mark.parametrize(
    "shape",
    [
        (5, 10, 11, 3),
        (2, 3, 4, 5, 4),
    ],
)
def test_apply_uint8_lut_per_channel_volume_matches_numpy(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    c = shape[-1]
    luts = rng.integers(0, 256, size=(256, 1, c), dtype=np.uint8)
    out = apply_uint8_lut(img, luts, inplace=False)
    np.testing.assert_array_equal(out, _ref_per_channel(img, luts))


def test_apply_uint8_lut_single_channel_table_matches_reference() -> None:
    rng = np.random.default_rng(13)
    img = rng.integers(0, 256, size=(6, 7, 1), dtype=np.uint8)
    luts = rng.integers(0, 256, size=(256, 1, 1), dtype=np.uint8)
    out = apply_uint8_lut(img, luts, inplace=False)
    np.testing.assert_array_equal(out, _ref_per_channel(img, luts))


# --- Non-contiguous / layout ---


def test_apply_uint8_lut_strided_hwc_matches_reference() -> None:
    rng = np.random.default_rng(14)
    base = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    img = base[:, ::2, :]
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    out = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, _ref_shared(img, lut))


def test_apply_uint8_lut_fortran_hwc_matches_reference() -> None:
    rng = np.random.default_rng(15)
    base = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    img = np.asfortranarray(base)
    lut = np.arange(256, dtype=np.uint8)
    out = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, _ref_shared(np.ascontiguousarray(img), lut))


# --- Routing: OpenCV branch still equals NumPy reference ---


@pytest.mark.parametrize(
    "shape",
    [
        (512, 512, 5),
        (640, 640, 2),
        (128, 128, 3),
    ],
)
def test_apply_uint8_lut_shared_large_and_small_hwc_match_numpy(shape: tuple[int, ...]) -> None:
    """OpenCV and StringZilla branches must both agree with ``lut[img]``."""
    rng = np.random.default_rng(16)
    img = np.ascontiguousarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    out = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, _ref_shared(img, lut))


# --- Errors ---


@pytest.mark.parametrize(
    ("img_dtype", "lut_dtype"),
    [
        (np.float32, np.uint8),
        (np.uint8, np.float32),
    ],
)
def test_apply_uint8_lut_type_error(img_dtype: np.dtype, lut_dtype: np.dtype) -> None:
    img = np.zeros((4, 4, 3), dtype=img_dtype)
    lut = np.zeros(256, dtype=lut_dtype)
    with pytest.raises(TypeError, match="apply_uint8_lut expects uint8"):
        apply_uint8_lut(img, lut, inplace=False)


def test_apply_uint8_lut_wrong_1d_length() -> None:
    img = np.zeros((2, 2, 1), dtype=np.uint8)
    lut = np.zeros(255, dtype=np.uint8)
    with pytest.raises(ValueError, match="1D LUT must have length 256"):
        apply_uint8_lut(img, lut, inplace=False)


def test_apply_uint8_lut_rejects_2d_per_channel_luts() -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    lut = np.zeros((3, 256), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"LUT must be \(256,\) or \(256, 1, C\)"):
        apply_uint8_lut(img, lut, inplace=False)


def test_apply_uint8_lut_ndim_too_high_lut() -> None:
    img = np.zeros((2, 2, 1), dtype=np.uint8)
    lut = np.zeros((1, 256, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"LUT must be \(256,\) or \(256, 1, C\)"):
        apply_uint8_lut(img, lut, inplace=False)


# --- Heuristic contract ---


@pytest.mark.parametrize(
    ("shape", "expect"),
    [
        ((256, 256, 1), False),
        ((1024, 1024, 1), False),
        ((512, 512, 2), False),
        ((512, 512, 3), False),
        ((512, 512, 4), False),
        ((512, 512, 5), True),
        ((512, 512, 8), True),
        ((512, 512, 9), True),
        ((640, 640, 2), True),
        ((4, 8, 9, 3), False),
        ((2, 2, 4, 5, 3), False),
    ],
)
def test_opencv_shared_heuristic_examples(shape: tuple[int, ...], expect: bool) -> None:
    assert opencv_shared_uint8_lut_faster_hwc(shape) is expect  # noqa: S101


def test_apply_shared_matches_sz_lut_when_heuristic_false() -> None:
    rng = np.random.default_rng(20)
    img = np.ascontiguousarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    assert not opencv_shared_uint8_lut_faster_hwc(img.shape)  # noqa: S101
    a = apply_uint8_lut(img.copy(), lut, inplace=False)
    b = sz_lut(img.copy(), lut, inplace=False)
    np.testing.assert_array_equal(a, b)


def test_apply_shared_matches_cv2_when_heuristic_true() -> None:
    rng = np.random.default_rng(21)
    shape = (512, 512, 5)
    assert opencv_shared_uint8_lut_faster_hwc(shape)  # noqa: S101
    img = np.ascontiguousarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
    lut = rng.integers(0, 256, size=(256,), dtype=np.uint8)
    a = apply_uint8_lut(img.copy(), lut, inplace=False)
    b = cv2.LUT(img, lut)
    np.testing.assert_array_equal(a, b)


# --- Arithmetic parity (HWC) ---


def test_multiply_lut_parity_with_multiply_small() -> None:
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(12, 14, 3), dtype=np.uint8)
    for m in (1.5, np.array([0.5, 1.0, 2.0], dtype=np.float32)):
        np.testing.assert_array_equal(
            multiply_lut(img, m, inplace=False),
            multiply(img, m, inplace=False),
        )


def test_multiply_add_lut_parity_small() -> None:
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(10, 11, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        multiply_add_lut(img, 1.25, 3, inplace=False),
        multiply_add(img, 1.25, 3, inplace=False),
    )


# --- Arithmetic parity (volumes / batches) ---


def _ref_multiply_uint8(img: np.ndarray, factor: float) -> np.ndarray:
    return clip(np.multiply(img.astype(np.float32, copy=False), float(factor)), np.uint8, inplace=False)


@pytest.mark.parametrize("shape", [(3, 16, 16, 3), (2, 2, 8, 8, 1)])
@pytest.mark.parametrize("factor", [0.5, 2.0, 1.25])
def test_multiply_scalar_volume_matches_float_reference(shape: tuple[int, ...], factor: float) -> None:
    rng = np.random.default_rng(22)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    got = multiply(img, factor, inplace=False)
    np.testing.assert_array_equal(got, _ref_multiply_uint8(img, factor))


@pytest.mark.parametrize("shape", [(2, 4, 5, 3), (2, 2, 3, 4, 3)])
def test_multiply_vector_volume_matches_float_reference(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(23)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    c = shape[-1]
    vec = np.linspace(0.8, 1.2, num=c, dtype=np.float32)
    got = multiply(img, vec, inplace=False)
    np.testing.assert_array_equal(got, _ref_multiply_uint8_vector(img, vec))


def _ref_multiply_uint8_vector(img: np.ndarray, vec: np.ndarray) -> np.ndarray:
    f = vec.reshape((1,) * (img.ndim - 1) + (img.shape[-1],))
    return clip(np.multiply(img.astype(np.float32, copy=False), f), np.uint8, inplace=False)


@pytest.mark.parametrize("shape", [(2, 5, 6, 3), (1, 2, 4, 4, 1)])
def test_multiply_add_scalar_volume_matches_float_reference(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(24)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    got = multiply_add(img, 1.1, 4.0, inplace=False)
    ref = clip(np.multiply(img.astype(np.float32, copy=False), 1.1) + 4.0, np.uint8, inplace=False)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("shape", [(2, 5, 6, 3)])
def test_apply_lut_multiply_volume_matches_apply_uint8_lut(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(25)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    factor = 1.3
    from albucore.arithmetic import create_lut_array

    lut = clip(create_lut_array(np.uint8, factor, "multiply"), np.uint8, inplace=False)
    a = apply_lut(img, factor, "multiply", inplace=False)
    b = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("operation", ["add", "multiply", "power"])
def test_apply_lut_scalar_hwc_matches_apply_uint8_lut(operation: str) -> None:
    rng = np.random.default_rng(26)
    img = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)
    if operation == "add":
        v = 7.0
    elif operation == "multiply":
        v = 1.2
    else:
        v = 1.05
    from albucore.arithmetic import create_lut_array

    lut = clip(create_lut_array(np.uint8, v, operation), np.uint8, inplace=False)
    a = apply_lut(img, v, operation, inplace=False)
    b = apply_uint8_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(a, b)
