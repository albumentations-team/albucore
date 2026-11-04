"""OpenCV differential coverage for the six dtype-preserving image kernels."""

import cv2
import numpy as np
import pytest

import albucore as core

KERNEL = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]], dtype=np.float32)
ELEMENT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
MATRIX = np.array([[1.125, 0.125, 0], [0, 0.875, 0.125], [-0.125, 0.25, 1]], dtype=np.float32)
OPERATIONS = {
    "box_blur": lambda image: core.box_blur(image, 3),
    "erode": lambda image: core.erode(image, ELEMENT),
    "dilate": lambda image: core.dilate(image, ELEMENT),
    "convolve": lambda image: core.convolve(image, KERNEL),
    "linear_transformation_rgb": lambda image: core.linear_transformation_rgb(image, MATRIX),
    "unsharp_mask": lambda image: core.unsharp_mask(image, 3, 1.0, 0.5, 10),
}


def _reference(image: np.ndarray, name: str, *, threshold: int = 10) -> np.ndarray:
    original_dtype = image.dtype
    working = image.astype(np.float32, order="C") if image.dtype == np.float16 else image.copy(order="C")
    if name == "box_blur":
        result = cv2.blur(working, (3, 3))
    elif name in {"erode", "dilate"}:
        result = getattr(cv2, name)(working, ELEMENT)
    elif name == "convolve":
        result = cv2.filter2D(working, -1, KERNEL)
    elif name == "linear_transformation_rgb":
        packed = working.reshape(-1, working.shape[-2], working.shape[-1])
        result = cv2.transform(packed, MATRIX)
    else:
        blurred = cv2.GaussianBlur(working, (3, 3), sigmaX=1.0).reshape(working.shape)
        result = cv2.addWeighted(working, 1.5, blurred, -0.5, 0).reshape(working.shape)
        if working.dtype == np.float32:
            result = np.clip(result, 0, 1)
        if threshold > 0:
            difference = cv2.absdiff(working, blurred).reshape(working.shape)
            cutoff = threshold if original_dtype == np.uint8 else threshold / 255.0
            result = np.where(difference > cutoff, result, working)
    result = result.reshape(image.shape)
    if name in {"convolve", "linear_transformation_rgb"}:
        result = np.clip(result, 0, 255 if original_dtype == np.uint8 else 1)
    return result.astype(original_dtype)


def _source(dtype: type[np.generic], channels: int, layout: str) -> np.ndarray:
    values = np.random.default_rng(137).random((13, 17, channels), dtype=np.float32)
    image = (values * 255).astype(dtype) if dtype == np.uint8 else (values * 1.5 - 0.25).astype(dtype)
    if layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "strided":
        image = image[:, ::2]
    elif layout == "negative":
        image = image[:, ::-1]
    elif layout == "readonly":
        image.flags.writeable = False
    return image


@pytest.mark.parametrize(
    ("name", "channels"),
    [
        (name, channels)
        for name in OPERATIONS
        for channels in ((3,) if name == "linear_transformation_rgb" else (1, 3, 9))
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided", "negative", "readonly"])
def test_kernels_match_opencv_without_input_writes(
    name: str, channels: int, dtype: type[np.generic], layout: str
) -> None:
    image = _source(dtype, channels, layout)
    original = image.tobytes()
    expected = _reference(image, name)

    actual = OPERATIONS[name](image)

    unsigned = {np.dtype(np.uint8): np.uint8, np.dtype(np.float16): np.uint16, np.dtype(np.float32): np.uint32}[
        image.dtype
    ]
    np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
    assert actual.dtype == image.dtype
    assert actual.shape == image.shape
    assert actual.flags.writeable
    assert not np.shares_memory(actual, image)
    assert image.tobytes() == original


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "negative", "readonly"])
def test_rgb_batch_matches_packed_opencv_transform(dtype: type[np.generic], layout: str) -> None:
    image = _source(dtype, 3, layout)
    batch = np.stack((image, image[:, ::-1]))
    if layout == "fortran":
        batch = np.asfortranarray(batch)
    elif layout == "negative":
        batch = batch[:, :, ::-1]
    batch.flags.writeable = False
    original = batch.tobytes()

    actual = core.linear_transformation_rgb(batch, MATRIX)

    # One OpenCV call preserves SIMD/tail rounding across packed batch items.
    expected = _reference(batch, "linear_transformation_rgb")
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == batch.dtype
    assert batch.tobytes() == original


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 9])
def test_convolve_uses_independent_channel_kernels(dtype: type[np.generic], channels: int) -> None:
    image = _source(dtype, channels, "readonly")
    kernels = np.stack([KERNEL * (channel + 1) / channels for channel in range(channels)])
    working = image.astype(np.float32) if dtype == np.float16 else image.copy()
    expected = np.stack(
        [cv2.filter2D(working[..., channel], -1, kernel) for channel, kernel in enumerate(kernels)], axis=-1
    )
    expected = np.clip(expected, 0, 255 if dtype == np.uint8 else 1).astype(dtype)

    actual = core.convolve(image, kernels)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == image.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("threshold", [0, 137, 255])
def test_unsharp_threshold_selects_between_clipped_candidate_and_original(
    dtype: type[np.generic], threshold: int
) -> None:
    image = _source(dtype, 9, "readonly")

    actual = core.unsharp_mask(image, 3, 1.0, 0.5, threshold)

    np.testing.assert_array_equal(actual, _reference(image, "unsharp_mask", threshold=threshold))
    assert actual.dtype == image.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
def test_unsharp_positive_threshold_applies_to_every_channel(dtype: type[np.generic], channels: int) -> None:
    image = np.full((9, 11, channels), 100, dtype=np.float32)
    image[4, 5] = 102
    if dtype != np.uint8:
        image /= 255
    image = image.astype(dtype)
    image.flags.writeable = False

    actual = core.unsharp_mask(image, 3, 1.0, 0.5, 10)

    np.testing.assert_array_equal(actual, image)


@pytest.mark.parametrize(
    ("name", "channels"),
    [
        ("convolve", 3),
        ("unsharp_mask", 1),
        ("unsharp_mask", 2),
        ("unsharp_mask", 9),
    ],
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_floating_filters_preserve_nonfinite_and_signed_zero_bits(
    name: str, channels: int, dtype: type[np.generic]
) -> None:
    image = _source(dtype, channels, "contiguous")
    image.flat[:4] = [np.nan, np.inf, -np.inf, -0.0]
    image.flags.writeable = False
    original = image.tobytes()

    with np.errstate(all="ignore"):
        expected = _reference(image, name)
        actual = OPERATIONS[name](image)

    unsigned = np.uint16 if dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
    assert image.tobytes() == original
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("ksize", [1, 3, 15])
def test_convolve_matches_inplace_opencv_on_unaligned_inputs(dtype: type[np.generic], ksize: int) -> None:
    shape = (13, 17, 3)
    storage = bytearray(np.prod(shape) * np.dtype(dtype).itemsize + 1)
    image = np.ndarray(shape, dtype=dtype, buffer=storage, offset=1)
    image[...] = _source(dtype, 3, "contiguous")
    image.flags.writeable = False
    original = image.tobytes()
    working = image.astype(np.float32, order="C") if dtype == np.float16 else image.copy(order="C")
    kernel = np.full((ksize, ksize), 1 / (ksize * ksize), dtype=np.float32)
    cv2.filter2D(working, -1, kernel, dst=working)
    np.clip(working, 0, 255 if dtype == np.uint8 else 1, out=working)
    actual = core.convolve(image, kernel)

    expected = working.astype(dtype)
    unsigned = {np.uint8: np.uint8, np.float16: np.uint16, np.float32: np.uint32}[dtype]
    np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
    assert image.tobytes() == original
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("channels", [2, 9])
@pytest.mark.parametrize("threshold", [1, 10, 137, 255])
def test_unsharp_selection_near_threshold(channels: int, threshold: int) -> None:
    image = np.zeros((5, 7, channels), dtype=np.float32)
    gaussian = cv2.getGaussianKernel(3, 1.0, cv2.CV_32F)
    center = np.float32((threshold / 255) / (1 - gaussian[1, 0] ** 2))
    image[2, 3] = center
    image[2, 3, 0] = np.nextafter(center, np.float32(-np.inf))
    image[2, 3, -1] = np.nextafter(center, np.float32(np.inf))
    image.flags.writeable = False

    actual = core.unsharp_mask(image, 3, 1.0, 0.5, threshold)

    expected = _reference(image, "unsharp_mask", threshold=threshold)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
