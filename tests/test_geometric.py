"""Tests for geometric module: warp_affine, warp_perspective, copy_make_border, remap."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product

import pytest
import cv2
import numpy as np

from albucore.geometric import (
    warp_affine,
    warp_perspective,
    copy_make_border,
    remap,
    resize,
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


def test_warp_affine_accepts_3x3_matrix(rng: np.random.Generator) -> None:
    """warp_affine accepts 3x3 matrix (e.g. from create_affine_transformation_matrix)."""
    img = make_image(16, 16, 3, np.uint8, rng)
    M_2x3 = np.float32([[1, 0, 2], [0, 1, 2]])
    M_3x3 = np.eye(3, dtype=np.float32)
    M_3x3[:2, :] = M_2x3
    result_2x3 = warp_affine(img, M_2x3, (16, 16), border_value=0)
    result_3x3 = warp_affine(img, M_3x3, (16, 16), border_value=0)
    np.testing.assert_array_equal(result_2x3, result_3x3)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_warp_affine_3x3_matrix_dtype(dtype: type, rng: np.random.Generator) -> None:
    """warp_affine with 3x3 matrix works for uint8 and float32."""
    img = make_image(16, 16, 3, dtype, rng)
    M_3x3 = np.eye(3, dtype=np.float32)
    M_3x3[:2, :] = [[1, 0, 2], [0, 1, 2]]
    result = warp_affine(img, M_3x3, (16, 16), border_value=0)
    assert result.shape == (16, 16, 3)
    assert result.dtype == dtype


def test_warp_affine_3x3_matrix_chunked_path(rng: np.random.Generator) -> None:
    """warp_affine with 3x3 matrix works in chunked path (8ch)."""
    img = make_image(16, 16, 8, np.uint8, rng)
    M_3x3 = np.eye(3, dtype=np.float32)
    M_3x3[:2, :] = [[1, 0, 2], [0, 1, 2]]
    result = warp_affine(img, M_3x3, (16, 16), border_value=0)
    assert result.shape == (16, 16, 8)


def test_warp_affine_3x3_matrix_rotation(rng: np.random.Generator) -> None:
    """warp_affine with 3x3 rotation matrix produces same result as 2x3."""
    img = make_image(16, 16, 3, np.uint8, rng)
    angle = np.pi / 4
    c, s = np.cos(angle), np.sin(angle)
    M_2x3 = np.float32([[c, -s, 8], [s, c, 8]])
    M_3x3 = np.eye(3, dtype=np.float32)
    M_3x3[:2, :] = M_2x3
    result_2x3 = warp_affine(img, M_2x3, (16, 16), border_value=0)
    result_3x3 = warp_affine(img, M_3x3, (16, 16), border_value=0)
    np.testing.assert_array_equal(result_2x3, result_3x3)


# From Albumentations geometric tests (fgeometric.warp_affine -> albucore.warp_affine)
_WARP_AFFINE_IMAGE_SHAPES = [(100, 100, 1), (100, 100, 2), (100, 100, 3), (100, 100, 7)]
_WARP_AFFINE_TRANSFORM_PARAMS = [
    (0, (0, 0), 1, (0, 0), (100, 100)),  # No change
    (45, (0, 0), 1, (0, 0), (100, 100)),  # Rotation
    (0, (10, 10), 1, (0, 0), (100, 100)),  # Translation
    (0, (0, 0), 2, (0, 0), (200, 200)),  # Scaling
    (0, (0, 0), 1, (20, 0), (100, 100)),  # Shear in x only
    (0, (0, 0), 1, (0, 20), (100, 100)),  # Shear in y only
    (0, (0, 0), 1, (20, 20), (100, 100)),  # Shear in both x and y
]


@pytest.mark.parametrize(
    "params,image_shape",
    list(product(_WARP_AFFINE_TRANSFORM_PARAMS, _WARP_AFFINE_IMAGE_SHAPES)),
)
def test_warp_affine_transforms(params: tuple, image_shape: tuple[int, ...]) -> None:
    """warp_affine with create_affine_transformation_matrix produces correct output shape."""
    angle, translation, scale, shear, output_shape = params
    image = np.ones(image_shape, dtype=np.uint8) * 255

    translate: Mapping[str, float] = {"x": translation[0], "y": translation[1]}
    shear_dict: Mapping[str, float] = {"x": shear[0], "y": shear[1]}
    scale_dict: Mapping[str, float] = {"x": scale, "y": scale}
    shift = _center(image_shape)

    affine_matrix = _create_affine_matrix(
        translate=translate,
        shear=shear_dict,
        scale=scale_dict,
        rotate=angle,
        shift=shift,
    )

    height_out, width_out = output_shape
    warped = warp_affine(
        image,
        affine_matrix,
        dsize=(width_out, height_out),
        flags=cv2.INTER_LINEAR,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )

    assert warped.shape[:2] == output_shape


@pytest.mark.parametrize("image_shape", _WARP_AFFINE_IMAGE_SHAPES)
@pytest.mark.parametrize(
    "translation,padding_value",
    [
        ((10, 0), 0),
        ((-10, 0), 0),
        ((0, 10), 0),
        ((0, -10), 0),
    ],
)
def test_warp_affine_edge_padding(
    image_shape: tuple[int, ...],
    translation: tuple[int, int],
    padding_value: int,
) -> None:
    """warp_affine applies correct edge padding for translation."""
    image = np.ones(image_shape, dtype=np.uint8) * 255

    translate: Mapping[str, float] = {"x": translation[0], "y": translation[1]}
    shear_dict: Mapping[str, float] = {"x": 0, "y": 0}
    scale_dict: Mapping[str, float] = {"x": 1, "y": 1}
    shift = _center(image_shape)

    affine_matrix = _create_affine_matrix(
        translate=translate,
        shear=shear_dict,
        scale=scale_dict,
        rotate=0,
        shift=shift,
    )

    height, width = image_shape[:2]
    warped = warp_affine(
        image,
        affine_matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        border_value=padding_value,
        border_mode=cv2.BORDER_CONSTANT,
    )

    if translation[0] > 0:
        assert np.all(warped[:, : translation[0]] == padding_value)
    elif translation[0] < 0:
        assert np.all(warped[:, translation[0] :] == padding_value)
    if translation[1] > 0:
        assert np.all(warped[: translation[1], :] == padding_value)
    elif translation[1] < 0:
        assert np.all(warped[translation[1] :, :] == padding_value)

    if translation[0] > 0:
        assert np.all(warped[:, translation[0] :] == 255)
    elif translation[0] < 0:
        assert np.all(warped[:, : translation[0]] == 255)
    if translation[1] > 0:
        assert np.all(warped[translation[1] :, :] == 255)
    elif translation[1] < 0:
        assert np.all(warped[: translation[1], :] == 255)


# Helpers for warp_affine integration tests (match Albumentations create_affine_transformation_matrix)
def _center(image_shape: tuple[int, ...]) -> tuple[float, float]:
    """Center (x, y) for image. Matches albumentations geometric.functional.center."""
    height, width = image_shape[:2]
    return width / 2 - 0.5, height / 2 - 0.5


def _create_affine_matrix(
    translate: Mapping[str, float],
    shear: Mapping[str, float],
    scale: Mapping[str, float],
    rotate: float,
    shift: tuple[float, float],
) -> np.ndarray:
    """3x3 affine matrix. Matches albumentations create_affine_transformation_matrix."""
    rotate_rad = np.deg2rad(rotate % 360)
    shear_x_rad = np.deg2rad(shear["x"])
    shear_y_rad = np.deg2rad(shear["y"])

    m_shift_topleft = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]], dtype=np.float32)
    m_scale = np.array([[scale["x"], 0, 0], [0, scale["y"], 0], [0, 0, 1]], dtype=np.float32)
    m_rotate = np.array(
        [
            [np.cos(rotate_rad), np.sin(rotate_rad), 0],
            [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    m_shear = np.array(
        [[1, np.tan(shear_x_rad), 0], [np.tan(shear_y_rad), 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    m_translate = np.array(
        [[1, 0, translate["x"]], [0, 1, translate["y"]], [0, 0, 1]],
        dtype=np.float32,
    )
    m_shift_center = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=np.float32)

    m = m_shift_center @ m_translate @ m_shear @ m_rotate @ m_scale @ m_shift_topleft
    m[2] = [0, 0, 1]
    return m


def _ensure_hwc(img: np.ndarray) -> np.ndarray:
    """Ensure image has shape (H, W, C) for albucore."""
    if img.ndim == 2:
        return np.expand_dims(img, -1)
    return img


@pytest.mark.parametrize("image_shape", [(100, 100, 3)])  # 1ch has lower IoU after round-trip
@pytest.mark.parametrize("angle", [45, 90, 180])
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("scale", [1, 0.8, 1.2])
def test_warp_affine_inverse_angle_scale(
    image_shape: tuple[int, ...],
    angle: int,
    shape: str,
    scale: float,
) -> None:
    """Forward then inverse affine (rotate+scale) restores image (IoU > 0.97)."""
    image = np.zeros(image_shape, dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)
    else:
        cv2.circle(image, (50, 50), 25, 255, -1)

    center_pt = _center(image_shape)
    forward_matrix = _create_affine_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": scale, "y": scale},
        rotate=angle,
        shift=center_pt,
    )
    inverse_matrix = _create_affine_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": 1 / scale, "y": 1 / scale},
        rotate=-angle,
        shift=center_pt,
    )

    height, width = image_shape[:2]
    img_hwc = _ensure_hwc(image)
    warped = warp_affine(
        img_hwc,
        forward_matrix,
        flags=cv2.INTER_NEAREST,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    restored = warp_affine(
        warped,
        inverse_matrix,
        flags=cv2.INTER_NEAREST,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )

    intersection = np.logical_and(image > 0, restored.squeeze() > 0)
    union = np.logical_or(image > 0, restored.squeeze() > 0)
    iou = np.sum(intersection) / np.sum(union)
    assert iou > 0.97, f"IoU {iou} too low"


@pytest.mark.parametrize(
    "img,expected",
    [
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[4, 8, 12, 16], [3, 7, 11, 15], [2, 6, 10, 14], [1, 5, 9, 13]], dtype=np.uint8),
        ),
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.04, 0.08, 0.12, 0.16],
                    [0.03, 0.07, 0.11, 0.15],
                    [0.02, 0.06, 0.10, 0.14],
                    [0.01, 0.05, 0.09, 0.13],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_warp_affine_rotate_90(img: np.ndarray, expected: np.ndarray) -> None:
    """90 deg rotation with warp_affine matches expected."""
    angle = 90
    center_pt = _center(img.shape[:2])
    transform = _create_affine_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=angle,
        shift=center_pt,
    )
    height, width = img.shape[:2]
    img_hwc = _ensure_hwc(img)
    result = warp_affine(
        img_hwc,
        transform,
        flags=cv2.INTER_LINEAR,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    out = result.squeeze() if result.shape[-1] == 1 else result
    np.testing.assert_array_almost_equal(out, expected, decimal=5)


@pytest.mark.parametrize(
    "img,expected,translate",
    [
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10], [0, 0, 13, 14]], dtype=np.uint8),
            (0, 2),
        ),
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.00, 0.00, 0.01, 0.02],
                    [0.00, 0.00, 0.05, 0.06],
                    [0.00, 0.00, 0.09, 0.10],
                    [0.00, 0.00, 0.13, 0.14],
                ],
                dtype=np.float32,
            ),
            (0, 2),
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8),
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8),
            (2, 0),
        ),
        (
            np.array(
                [
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                    [0.09, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.01, 0.02, 0.03, 0.04],
                    [0.05, 0.06, 0.07, 0.08],
                ],
                dtype=np.float32,
            ),
            (2, 0),
        ),
    ],
)
def test_warp_affine_translate(img: np.ndarray, expected: np.ndarray, translate: tuple[int, int]) -> None:
    """Translation with warp_affine matches expected."""
    transform = _create_affine_matrix(
        translate={"x": translate[1], "y": translate[0]},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=(0, 0),
    )
    height, width = img.shape[:2]
    img_hwc = _ensure_hwc(img)
    result = warp_affine(
        img_hwc,
        transform,
        flags=cv2.INTER_LINEAR,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    out = result.squeeze() if result.shape[-1] == 1 else result
    np.testing.assert_array_almost_equal(out, expected, decimal=5)


@pytest.mark.parametrize("image_shape", [(100, 100, 1), (100, 100, 3)])
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("shear", [0, 20, -20])
def test_warp_affine_inverse_shear(
    image_shape: tuple[int, ...],
    shear: int,
    shape: str,
) -> None:
    """Forward then inverse shear restores image."""
    image = np.zeros(image_shape, dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)
    else:
        cv2.circle(image, (50, 50), 25, 255, -1)

    center_pt = _center(image_shape[:2])
    forward_matrix = _create_affine_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": shear, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center_pt,
    )
    inverse_matrix = _create_affine_matrix(
        translate={"x": 0, "y": 0},
        shear={"x": -shear, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center_pt,
    )

    height, width = image_shape[:2]
    warped = warp_affine(
        image,
        forward_matrix,
        flags=cv2.INTER_NEAREST,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    restored = warp_affine(
        warped,
        inverse_matrix,
        flags=cv2.INTER_NEAREST,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    np.testing.assert_allclose(image, restored, atol=1)
    if shear != 0:
        assert not np.array_equal(image, warped)


@pytest.mark.parametrize("image_shape", [(100, 100, 1), (100, 100, 3)])
@pytest.mark.parametrize("shape", ["circle", "rectangle"])
@pytest.mark.parametrize("translate", [(0, 0), (10, -10), (-10, 10)])
def test_warp_affine_inverse_translate(
    image_shape: tuple[int, ...],
    translate: tuple[int, int],
    shape: str,
) -> None:
    """Forward then inverse translation restores image."""
    image = np.zeros(image_shape, dtype=np.uint8)
    if shape == "rectangle":
        cv2.rectangle(image, (25, 25), (75, 75), 255, -1)
    else:
        cv2.circle(image, (50, 50), 25, 255, -1)

    center_pt = _center(image_shape[:2])
    forward_matrix = _create_affine_matrix(
        translate={"x": translate[0], "y": translate[1]},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center_pt,
    )
    inverse_matrix = _create_affine_matrix(
        translate={"x": -translate[0], "y": -translate[1]},
        shear={"x": 0, "y": 0},
        scale={"x": 1, "y": 1},
        rotate=0,
        shift=center_pt,
    )

    height, width = image_shape[:2]
    warped = warp_affine(
        image,
        forward_matrix,
        flags=cv2.INTER_NEAREST,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    restored = warp_affine(
        warped,
        inverse_matrix,
        flags=cv2.INTER_NEAREST,
        border_value=0,
        dsize=(width, height),
        border_mode=cv2.BORDER_CONSTANT,
    )
    np.testing.assert_allclose(image, restored, atol=1)
    if translate != (0, 0):
        assert not np.array_equal(image, warped)


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


# -----------------------------------------------------------------------------
# resize
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 4, 5, 8, 16], ids=[f"c{c}" for c in [1, 3, 4, 5, 8, 16]])
def test_resize_shape(channels: int, rng: np.random.Generator) -> None:
    """resize produces correct output shape for various channels."""
    img = make_image(16, 16, channels, np.uint8, rng)
    dsize = (32, 24)
    result = resize(img, dsize)
    assert result.shape == (24, 32, channels)


@pytest.mark.parametrize("channels", [1, 3, 4], ids=["1ch", "3ch", "4ch"])
def test_resize_equiv_cv2(channels: int, rng: np.random.Generator) -> None:
    """resize matches cv2.resize exactly when C <= 4 for linear interpolation."""
    img = make_image(16, 16, channels, np.uint8, rng)
    dsize = (32, 24)
    expected = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
    if channels == 1 and expected.ndim == 2:
        expected = np.expand_dims(expected, -1)
    result = resize(img, dsize, interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(result, expected)


def test_resize_preserve_channel_dim(rng: np.random.Generator) -> None:
    """resize preserves (H, W, 1) for grayscale images."""
    img = make_image(16, 16, 1, np.uint8, rng)
    dsize = (20, 20)
    result = resize(img, dsize)
    assert result.shape == (20, 20, 1)


@pytest.mark.parametrize("channels", [5, 8, 16], ids=["5ch", "8ch", "16ch"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_resize_many_channels_interpolations(channels: int, interpolation: int, rng: np.random.Generator) -> None:
    """resize works without crashing for various interpolations and many channels."""
    img = make_image(16, 16, channels, np.uint8, rng)
    dsize = (32, 24)

    # Just checking it runs and produces correct shape and type
    result = resize(img, dsize, interpolation=interpolation)

    assert result.shape == (24, 32, channels)
    assert result.dtype == img.dtype


def test_resize_with_fx_fy(rng: np.random.Generator) -> None:
    """resize works correctly when passing fx and fy instead of dsize."""
    img = make_image(16, 16, 6, np.uint8, rng)
    # Scale by 1.5x
    result = resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    assert result.shape == (24, 24, 6)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_resize_dtypes(dtype: type, rng: np.random.Generator) -> None:
    """resize preserves dtype for large channel dimensions."""
    img = make_image(16, 16, 6, dtype, rng)
    dsize = (32, 32)
    result = resize(img, dsize)
    assert result.dtype == img.dtype
