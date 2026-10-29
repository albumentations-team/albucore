"""Shared lookup tables match independent OpenCV results."""

from typing import cast

import cv2
import numpy as np
import pytest

import albucore.lut as lut_module
from albucore import apply_uint8_lut


@pytest.mark.parametrize(
    "shape",
    [
        (10, 10, 1),
        (100, 100, 1),
        (224, 224, 1),
        (100, 100, 3),
        (224, 224, 4),
    ],
)
@pytest.mark.parametrize("lut_type", ["identity", "invert", "random", "threshold"])
@pytest.mark.parametrize("alias", [True, False])
def test_shared_lut_matches_cv2_lut(shape: tuple[int, ...], lut_type: str, alias: bool) -> None:
    rng = np.random.default_rng(42)
    img_base = rng.integers(0, 256, size=shape, dtype=np.uint8)

    if lut_type == "identity":
        lut = np.arange(256, dtype=np.uint8)
    elif lut_type == "invert":
        lut = np.arange(255, -1, -1, dtype=np.uint8)
    elif lut_type == "random":
        lut = rng.integers(0, 256, size=256, dtype=np.uint8)
    else:
        lut = np.where(np.arange(256) > 127, 255, 0).astype(np.uint8)

    expected = cv2.LUT(np.ascontiguousarray(img_base), lut).reshape(img_base.shape)

    buf = img_base.copy()
    got = apply_uint8_lut(buf, lut, out=buf if alias else None)

    if alias:
        assert got is buf  # noqa: S101
    np.testing.assert_array_equal(got, expected)


def test_shared_lut_non_contiguous_input_matches_cv2() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    img = np.asfortranarray(base)
    lut = np.roll(np.arange(256, dtype=np.uint8), 17)
    expected = cv2.LUT(np.ascontiguousarray(img), lut).reshape(img.shape)
    out = apply_uint8_lut(img, lut)
    np.testing.assert_array_equal(out, expected)


def test_shared_lut_out_alias_mutates_strided_input() -> None:
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(24, 64, 3), dtype=np.uint8)
    img = base[:, ::2, :]
    lut = np.roll(np.arange(256, dtype=np.uint8), 17)
    expected = cv2.LUT(np.ascontiguousarray(img.copy()), lut).reshape(img.shape)

    out = apply_uint8_lut(img, lut, out=img)

    assert out is img  # noqa: S101
    np.testing.assert_equal(img.flags["C_CONTIGUOUS"], False)
    np.testing.assert_array_equal(img, expected)


def test_apply_uint8_lut_preserves_hw1_shape() -> None:
    """Raw ``cv2.LUT`` returns ``(H, W)`` for ``(H, W, 1)``; wrapper keeps channel axis."""
    img = np.arange(60, dtype=np.uint8).reshape(6, 10, 1)
    lut = np.arange(255, -1, -1, dtype=np.uint8)
    out = apply_uint8_lut(img, lut)
    assert out.shape == (6, 10, 1)  # noqa: S101
    raw = cv2.LUT(img, lut)
    assert raw.shape == (6, 10)  # noqa: S101
    np.testing.assert_array_equal(out[..., 0], raw)


def test_shared_lut_accepts_contiguous_singleton_stride_view() -> None:
    backing = np.arange(99, dtype=np.uint8).reshape(1, 9, 11)
    image = backing.transpose(1, 2, 0)
    original = image.copy()
    lut = 255 - np.arange(256, dtype=np.uint8)
    np.testing.assert_equal(image.flags.c_contiguous, True)

    result = apply_uint8_lut(image, lut)

    np.testing.assert_array_equal(result, lut[original])
    np.testing.assert_array_equal(image, original)


def test_shared_lut_uses_stringzilla_output_for_contiguous_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    original = image.copy()
    lut = np.roll(np.arange(256, dtype=np.uint8), 17)
    expected = lut[image]
    out = np.empty_like(image)
    calls: list[tuple[memoryview, memoryview, dict[str, object]]] = []
    original_translate = lut_module.sz.translate

    def translate(source: memoryview, table: memoryview, **kwargs: object) -> object:
        calls.append((source, table, kwargs))
        return original_translate(source, table, **kwargs)

    monkeypatch.setattr(lut_module.sz, "translate", translate)

    result = apply_uint8_lut(image, lut, out=out)

    np.testing.assert_equal(len(calls), 1)
    source, table, kwargs = calls[0]
    np.testing.assert_equal(type(source), memoryview)
    np.testing.assert_equal(type(table), memoryview)
    np.testing.assert_equal(id(source.obj), id(image))
    np.testing.assert_equal(id(table.obj), id(lut))
    np.testing.assert_equal(set(kwargs), {"out"})
    output_buffer = cast("memoryview", kwargs["out"])
    np.testing.assert_equal(id(output_buffer.obj), id(result))
    np.testing.assert_equal(id(result), id(out))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
