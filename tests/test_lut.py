"""``sz_lut`` matches OpenCV LUT reference (``_cv2_lut_uint8`` applies ``@preserve_channel_dim``)."""

import cv2
import numpy as np
import pytest

from albucore.functions import sz_lut
from albucore.lut import _cv2_lut_uint8


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
@pytest.mark.parametrize("inplace", [True, False])
def test_sz_lut_matches_cv2_lut(shape: tuple[int, ...], lut_type: str, inplace: bool) -> None:
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

    expected = _cv2_lut_uint8(np.ascontiguousarray(img_base), lut)

    buf = img_base.copy()
    got = sz_lut(buf, lut, inplace=inplace)

    if inplace:
        assert got is buf  # noqa: S101
    np.testing.assert_array_equal(got, expected)


def test_sz_lut_non_contiguous_input_matches_cv2() -> None:
    """Fortran-ordered or sliced inputs are made C-contiguous inside ``sz_lut``."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    img = np.asfortranarray(base)
    lut = np.roll(np.arange(256, dtype=np.uint8), 17)
    expected = _cv2_lut_uint8(np.ascontiguousarray(img), lut)
    out = sz_lut(img, lut, inplace=False)
    np.testing.assert_array_equal(out, expected)


def test_cv2_lut_uint8_preserves_hw1_shape() -> None:
    """Raw ``cv2.LUT`` returns ``(H, W)`` for ``(H, W, 1)``; wrapper keeps channel axis."""
    img = np.arange(60, dtype=np.uint8).reshape(6, 10, 1)
    lut = np.arange(255, -1, -1, dtype=np.uint8)
    out = _cv2_lut_uint8(img, lut)
    assert out.shape == (6, 10, 1)  # noqa: S101
    raw = cv2.LUT(img, lut)
    assert raw.shape == (6, 10)  # noqa: S101
    np.testing.assert_array_equal(out[..., 0], raw)
