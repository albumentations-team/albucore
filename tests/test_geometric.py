"""Tests for geometric module: warp_affine, warp_perspective, copy_make_border, remap."""

from __future__ import annotations

import pytest
import cv2
import numpy as np

from albucore.geometric import (
    warp_affine,
    warp_perspective,
    copy_make_border,
    remap,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

CHANNEL_COUNTS = [1, 3, 4, 5, 8, 16]
DTYPES = [np.uint8, np.float32]
RNG = np.random.default_rng(42)


@pytest.fixture
def rng():
    """Reproducible RNG for tests."""
    return np.random.default_rng(42)


def make_image(h: int, w: int, channels: int, dtype: type, rng: np.random.Generator) -> np.ndarray:
    """Create test image with shape (H, W, C)."""
    if dtype == np.uint8:
        img = rng.integers(0, 256, (h, w, channels), dtype=np.uint8)
    else:
        img = rng.random((h, w, channels), dtype=np.float32)
    return np.ascontiguousarray(img)


# -----------------------------------------------------------------------------
# warp_affine
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 4, 8], ids=[f"c{c}" for c in [1, 3, 4, 8]])
def test_warp_affine_shape(channels: int, rng: np.random.Generator) -> None:
    """warp_affine produces correct output shape."""
    img = make_image(16, 16, channels, np.uint8, rng)
    M = np.float32([[1, 0, 2], [0, 1, 2]])
    dsize = (16, 16)
    result = warp_affine(img, M, dsize, border_value=0)
    assert result.shape == (16, 16, channels)


@pytest.mark.parametrize("channels", [1, 3, 4], ids=["1ch", "3ch", "4ch"])
def test_warp_affine_equiv_cv2(channels: int, rng: np.random.Generator) -> None:
    """warp_affine matches cv2.warpAffine when C <= 4 (albucore preserves (H,W,1))."""
    img = make_image(16, 16, channels, np.uint8, rng)
    M = np.float32([[1, 0, 2], [0, 1, 2]])
    dsize = (16, 16)
    expected = cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
    if channels == 1 and expected.ndim == 2:
        expected = np.expand_dims(expected, -1)
    result = warp_affine(img, M, dsize, flags=cv2.INTER_LINEAR, border_value=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "flags",
    [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA],
    ids=["NEAREST", "LINEAR", "AREA"],
)
def test_warp_affine_interpolations(flags: int, rng: np.random.Generator) -> None:
    """warp_affine works with INTER_NEAREST, INTER_LINEAR, INTER_AREA for 8ch."""
    img = make_image(16, 16, 8, np.uint8, rng)
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    result = warp_affine(img, M, (16, 16), flags=flags, border_value=0)
    assert result.shape == (16, 16, 8)


@pytest.mark.parametrize("border_value", [0, 128, (0, 0, 0, 0)], ids=["scalar0", "scalar128", "tuple4"])
def test_warp_affine_border_values(border_value: int | tuple, rng: np.random.Generator) -> None:
    """warp_affine accepts scalar and len<=4 borderValue for 8ch."""
    img = make_image(16, 16, 8, np.uint8, rng)
    M = np.float32([[1, 0, 2], [0, 1, 2]])
    result = warp_affine(img, M, (16, 16), border_value=border_value)
    assert result.shape == (16, 16, 8)


def test_warp_affine_per_channel_border_value(rng: np.random.Generator) -> None:
    """warp_affine handles per-channel borderValue (len=8) for 8ch."""
    img = make_image(16, 16, 8, np.uint8, rng)
    M = np.float32([[1, 0, 2], [0, 1, 2]])
    border_value = tuple(range(8))  # per-channel
    result = warp_affine(img, M, (16, 16), border_value=border_value)
    assert result.shape == (16, 16, 8)


def test_warp_affine_preserve_channel_dim(rng: np.random.Generator) -> None:
    """warp_affine preserves (H, W, 1) for grayscale."""
    img = make_image(16, 16, 1, np.uint8, rng)
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    result = warp_affine(img, M, (16, 16), border_value=0)
    assert result.shape == (16, 16, 1)


# -----------------------------------------------------------------------------
# warp_perspective
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 8], ids=["1ch", "3ch", "8ch"])
def test_warp_perspective_shape(channels: int, rng: np.random.Generator) -> None:
    """warp_perspective produces correct output shape."""
    img = make_image(16, 16, channels, np.uint8, rng)
    M = np.eye(3, dtype=np.float32)
    M[0, 2] = 1
    M[1, 2] = 1
    dsize = (16, 16)
    result = warp_perspective(img, M, dsize, border_value=0)
    assert result.shape == (16, 16, channels)


@pytest.mark.parametrize("channels", [1, 3, 4], ids=["1ch", "3ch", "4ch"])
def test_warp_perspective_equiv_cv2(channels: int, rng: np.random.Generator) -> None:
    """warp_perspective matches cv2.warpPerspective when C <= 4 (albucore preserves (H,W,1))."""
    img = make_image(16, 16, channels, np.uint8, rng)
    M = np.eye(3, dtype=np.float32)
    M[0, 2] = 1
    M[1, 2] = 1
    dsize = (16, 16)
    expected = cv2.warpPerspective(img, M, dsize, borderValue=0)
    if channels == 1 and expected.ndim == 2:
        expected = np.expand_dims(expected, -1)
    result = warp_perspective(img, M, dsize, border_value=0)
    np.testing.assert_array_equal(result, expected)


def test_warp_perspective_per_channel_border_value(rng: np.random.Generator) -> None:
    """warp_perspective handles per-channel borderValue for 8ch."""
    img = make_image(16, 16, 8, np.uint8, rng)
    M = np.eye(3, dtype=np.float32)
    border_value = (0,) * 8
    result = warp_perspective(img, M, (16, 16), border_value=border_value)
    assert result.shape == (16, 16, 8)


# -----------------------------------------------------------------------------
# copy_make_border
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 4, 8], ids=[f"c{c}" for c in [1, 3, 4, 8]])
def test_copy_make_border_shape(channels: int, rng: np.random.Generator) -> None:
    """copy_make_border produces correct padded shape."""
    img = make_image(16, 16, channels, np.uint8, rng)
    result = copy_make_border(img, 2, 2, 2, 2, value=0)
    assert result.shape == (20, 20, channels)


@pytest.mark.parametrize("channels", [1, 3, 4], ids=["1ch", "3ch", "4ch"])
def test_copy_make_border_equiv_cv2(channels: int, rng: np.random.Generator) -> None:
    """copy_make_border matches cv2.copyMakeBorder when C <= 4 (albucore preserves (H,W,1))."""
    img = make_image(16, 16, channels, np.uint8, rng)
    expected = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    if channels == 1 and expected.ndim == 2:
        expected = np.expand_dims(expected, -1)
    result = copy_make_border(img, 2, 2, 2, 2, value=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("value", [0, 128, (0, 0, 0, 0)], ids=["scalar0", "scalar128", "tuple4"])
def test_copy_make_border_values(value: int | tuple, rng: np.random.Generator) -> None:
    """copy_make_border accepts scalar and len<=4 value for 8ch."""
    img = make_image(16, 16, 8, np.uint8, rng)
    result = copy_make_border(img, 2, 2, 2, 2, value=value)
    assert result.shape == (20, 20, 8)


def test_copy_make_border_per_channel_value(rng: np.random.Generator) -> None:
    """copy_make_border handles per-channel value (len=8) for 8ch."""
    img = make_image(16, 16, 8, np.uint8, rng)
    value = tuple(range(8))
    result = copy_make_border(img, 2, 2, 2, 2, value=value)
    assert result.shape == (20, 20, 8)


# -----------------------------------------------------------------------------
# remap
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 4, 8], ids=[f"c{c}" for c in [1, 3, 4, 8]])
def test_remap_shape(channels: int, rng: np.random.Generator) -> None:
    """remap preserves input shape."""
    h, w = 16, 16
    img = make_image(h, w, channels, np.uint8, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    result = remap(img, map_x, map_y)
    assert result.shape == img.shape


@pytest.mark.parametrize("channels", [1, 3, 4], ids=["1ch", "3ch", "4ch"])
def test_remap_equiv_cv2(channels: int, rng: np.random.Generator) -> None:
    """remap matches cv2.remap when C <= 4 (albucore preserves (H,W,1))."""
    h, w = 16, 16
    img = make_image(h, w, channels, np.uint8, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    expected = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    if channels == 1 and expected.ndim == 2:
        expected = np.expand_dims(expected, -1)
    result = remap(img, map_x, map_y)
    np.testing.assert_array_equal(result, expected)


def test_remap_identity_map(rng: np.random.Generator) -> None:
    """remap with identity map returns same as input (within interpolation error)."""
    h, w = 16, 16
    img = make_image(h, w, 3, np.uint8, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    result = remap(img, map_x, map_y)
    np.testing.assert_array_equal(result, img)


def test_remap_preserve_channel_dim(rng: np.random.Generator) -> None:
    """remap preserves (H, W, 1) for grayscale."""
    h, w = 16, 16
    img = make_image(h, w, 1, np.uint8, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    result = remap(img, map_x, map_y)
    assert result.shape == (h, w, 1)


# Interpolations supported by cv2.remap (INTER_LINEAR_EXACT is not)
REMAP_INTERPOLATIONS = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
]
REMAP_BORDER_MODES = [
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT,
    cv2.BORDER_REFLECT_101,
    cv2.BORDER_WRAP,
]


@pytest.mark.parametrize("channels", [6, 8, 16, 32], ids=[f"c{c}" for c in [6, 8, 16, 32]])
@pytest.mark.parametrize("interpolation", REMAP_INTERPOLATIONS, ids=["nearest", "linear", "cubic", "area", "lanczos4"])
@pytest.mark.parametrize("border_mode", REMAP_BORDER_MODES, ids=["constant", "replicate", "reflect", "reflect101", "wrap"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_remap_many_channels_all_modes(
    channels: int,
    interpolation: int,
    border_mode: int,
    dtype: type,
    rng: np.random.Generator,
) -> None:
    """remap works for C > 5 with any interpolation and border mode."""
    h, w = 32, 32
    img = make_image(h, w, channels, dtype, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    result = remap(img, map_x, map_y, interpolation=interpolation, border_mode=border_mode)

    assert result.shape == img.shape
    assert result.dtype == img.dtype

    # For interpolations that cv2.remap supports for >4ch, compare with direct call
    if interpolation in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA):
        expected = cv2.remap(img, map_x, map_y, interpolation, borderMode=border_mode)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("border_value", [0, 128, (0, 0, 0), (128,) * 4], ids=["0", "128", "tuple3", "tuple4"])
def test_remap_border_value_const(border_value: int | tuple, rng: np.random.Generator) -> None:
    """remap with constant border_value (scalar or len<=4) works for 8ch."""
    h, w = 32, 32
    img = make_image(h, w, 8, np.uint8, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    result = remap(
        img, map_x, map_y, border_mode=cv2.BORDER_CONSTANT, border_value=border_value
    )
    assert result.shape == img.shape

    expected = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )
    np.testing.assert_array_equal(result, expected)


def test_remap_border_value_per_channel(rng: np.random.Generator) -> None:
    """remap with per-channel border_value (1,2,3,4,5,6,7,8) works for 8ch via chunking."""
    h, w = 32, 32
    img = make_image(h, w, 8, np.uint8, rng)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    border_value = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

    result = remap(
        img, map_x, map_y, border_mode=cv2.BORDER_CONSTANT, border_value=border_value
    )
    assert result.shape == img.shape

    # Compare with manual chunked reference
    ref_chunks = []
    for i in range(8):
        ch = img[:, :, i : i + 1]
        bv = (border_value[i],) * 4
        out = cv2.remap(ch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
        ref_chunks.append(np.expand_dims(out, -1) if out.ndim == 2 else out)
    expected = np.concatenate(ref_chunks, axis=-1)
    np.testing.assert_array_equal(result, expected)


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_warp_affine_dsize_change(rng: np.random.Generator) -> None:
    """warp_affine with different dsize produces correct shape."""
    img = make_image(32, 32, 3, np.uint8, rng)
    M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
    result = warp_affine(img, M, (16, 16), border_value=0)
    assert result.shape == (16, 16, 3)


@pytest.mark.parametrize("dtype", DTYPES, ids=["uint8", "float32"])
def test_copy_make_border_dtypes(dtype: type, rng: np.random.Generator) -> None:
    """copy_make_border preserves dtype."""
    img = make_image(16, 16, 3, dtype, rng)
    result = copy_make_border(img, 2, 2, 2, 2, value=0)
    assert result.dtype == img.dtype
