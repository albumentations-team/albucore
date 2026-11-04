from collections.abc import Callable

import cv2
import numpy as np
import pytest

from albucore.utils import (
    NPDTYPE_TO_OPENCV_DTYPE,
    clip,
    convert_value,
    get_image_data,
    get_num_channels,
    get_opencv_dtype_from_numpy,
    get_opencv_max_channels,
    is_grayscale_image,
    is_multispectral_image,
    is_rgb_image,
)


@pytest.mark.parametrize(
    "input_img, dtype, expected",
    [
        (
            np.array([[-300, 0], [100, 400]], dtype=np.float32),
            np.uint8,
            np.array([[0, 0], [100, 255]], dtype=np.float32),
        ),
        (
            np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32),
            np.float32,
            np.array([[0, 0], [0.5, 1.0]], dtype=np.float32),
        ),
    ],
)
@pytest.mark.parametrize("provided_output", [False, True])
def test_clip(input_img, dtype, expected, provided_output):
    original = input_img.copy()
    destination = np.empty_like(input_img, dtype=dtype) if provided_output else None
    clipped = clip(input_img, dtype=dtype, out=destination)
    np.testing.assert_array_equal(clipped, expected)
    assert clipped.dtype == dtype
    np.testing.assert_array_equal(input_img, original)
    if destination is not None:
        assert clipped is destination


valid_cv2_types = {
    cv2.CV_8U,
    cv2.CV_16F,
    cv2.CV_16U,
    cv2.CV_32F,
    cv2.CV_64F,
    cv2.CV_32S,
}


@pytest.mark.parametrize("cv_type", NPDTYPE_TO_OPENCV_DTYPE.values())
def test_valid_cv2_types(cv_type):
    assert cv_type in valid_cv2_types, f"{cv_type} is not a valid cv2 type"


def test_cv_dtype_from_np():
    # Image routers support uint8, float16, and float32.
    assert get_opencv_dtype_from_numpy(np.uint8) == cv2.CV_8U
    assert get_opencv_dtype_from_numpy(np.float16) == cv2.CV_16F
    assert get_opencv_dtype_from_numpy(np.float32) == cv2.CV_32F
    assert get_opencv_dtype_from_numpy(np.dtype("uint8")) == cv2.CV_8U
    assert get_opencv_dtype_from_numpy(np.dtype("float16")) == cv2.CV_16F
    assert get_opencv_dtype_from_numpy(np.dtype("float32")) == cv2.CV_32F
    assert get_opencv_dtype_from_numpy(np.zeros((1,), dtype=np.float16)) == cv2.CV_16F


def test_get_opencv_max_channels_matches_matrix_type_encoding():
    max_channels = get_opencv_max_channels()

    assert cv2.CV_MAKETYPE(cv2.CV_8U, max_channels) <= cv2.MAT_TYPE_MASK
    assert cv2.CV_MAKETYPE(cv2.CV_8U, max_channels + 1) > cv2.MAT_TYPE_MASK


@pytest.mark.parametrize(
    "value, num_channels, expected",
    [
        ((1.5), 1, 1.5),
        (np.array([1.5]), 3, 1.5),
        (np.array([1.5, 2.5]), 1, 1.5),
        (np.array([1.5, 2.5, 0.5]), 2, np.array([1.5, 2.5], dtype=np.float32)),
        (3, 2, 3),
        (np.array(1.5), 2, 1.5),
        (np.reciprocal(np.array(0.2, dtype=np.float64)), 2, 5),
    ],
)
def test_convert_value(value, num_channels, expected):
    result = convert_value(value, num_channels)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "shape, expected_channels, description",
    [
        # 3D image with channels (H, W, C)
        ((100, 99, 1), 1, "3D image with 1 channel"),
        ((100, 99, 3), 3, "3D RGB image"),
        ((100, 99, 7), 7, "3D multi-channel image"),
        # 4D batch of images or volume (N/D, H, W, C)
        ((4, 100, 99, 1), 1, "Batch of grayscale images or volume"),
        ((4, 100, 99, 3), 3, "Batch of RGB images or volume"),
        ((10, 224, 224, 3), 3, "Batch of RGB images or volume"),
        # Edge cases
        ((1, 1, 1), 1, "Minimal 3D array"),
    ],
)
def test_get_num_channels(shape, expected_channels, description):
    image = np.zeros(shape)
    assert get_num_channels(image) == expected_channels, f"Failed for {description} with shape {shape}"


@pytest.mark.parametrize(
    "shape, expected_grayscale, description",
    [
        # HWC: shape=(100, 200, C)
        ((100, 200, 3), False, "HWC: RGB image"),
        ((100, 200, 1), True, "HWC: single channel image"),
        ((100, 200, 4), False, "HWC: RGBA image"),
        # XHWC: shape=(X, 100, 200, C) where X is batch or depth
        ((10, 100, 200, 3), False, "XHWC: batch/volume of RGB"),
        ((10, 100, 200, 1), True, "XHWC: batch/volume of single channel"),
    ],
)
def test_is_grayscale_image(shape, expected_grayscale, description):
    image = np.zeros(shape)
    result = is_grayscale_image(image)
    assert result == expected_grayscale, f"Failed for {description} with shape {shape}"


@pytest.mark.parametrize(
    ("shape", "expected_rgb", "expected_multispectral"),
    [
        ((100, 200, 1), False, False),
        ((100, 200, 3), True, False),
        ((100, 200, 4), False, True),
        ((100, 200, 5), False, True),
        ((10, 100, 200, 1), False, False),
        ((10, 100, 200, 3), True, False),
        ((10, 100, 200, 4), False, True),
        ((10, 100, 200, 5), False, True),
    ],
)
def test_image_channel_predicates(
    shape: tuple[int, ...], expected_rgb: bool, expected_multispectral: bool
) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    assert is_rgb_image(image) is expected_rgb
    assert is_multispectral_image(image) is expected_multispectral


@pytest.mark.parametrize(
    "shape",
    [
        # Basic HWC cases
        (100, 200, 1),
        (100, 200, 3),
        # XHWC cases (batch or depth)
        (10, 100, 200, 1),
        (10, 100, 200, 3),
    ],
)
def test_get_num_channels_and_is_grayscale_consistency(shape):
    image = np.zeros(shape)
    num_channels = get_num_channels(image)
    is_grayscale = is_grayscale_image(image)

    assert (num_channels == 1) == is_grayscale, (
        f"Inconsistency for shape {shape}: num_channels={num_channels}, is_grayscale={is_grayscale}"
    )


@pytest.mark.parametrize(
    "dtype, shape",
    [
        (np.uint8, (100, 200, 3)),
        (np.uint16, (150, 250, 1)),
        (np.float32, (224, 224, 3)),
        (np.float64, (64, 128, 4)),
        (np.int32, (32, 32, 1)),
    ],
)
def test_get_image_data_single_image(dtype, shape):
    img = np.zeros(shape, dtype=dtype)
    data = {"image": img}
    result = get_image_data(data)

    assert isinstance(result, dict)
    assert "dtype" in result
    assert "height" in result
    assert "width" in result
    assert result["dtype"] == dtype
    assert result["height"] == shape[0]
    assert result["width"] == shape[1]


@pytest.mark.parametrize(
    "dtype, shape",
    [
        (np.uint8, (5, 100, 200, 3)),
        (np.float32, (10, 256, 256, 1)),
        (np.uint16, (3, 128, 128, 1)),
    ],
)
def test_get_image_data_batch_of_images(dtype, shape):
    imgs = np.zeros(shape, dtype=dtype)
    data = {"images": imgs}
    result = get_image_data(data)

    assert isinstance(result, dict)
    assert result["dtype"] == dtype
    assert result["height"] == shape[1]
    assert result["width"] == shape[2]


@pytest.mark.parametrize(
    "dtype, shape",
    [
        (np.uint16, (10, 100, 200, 1)),
        (np.float64, (5, 256, 256, 3)),
        (np.float32, (20, 64, 64, 1)),
    ],
)
def test_get_image_data_volume(dtype, shape):
    vol = np.zeros(shape, dtype=dtype)
    data = {"volume": vol}
    result = get_image_data(data)

    assert isinstance(result, dict)
    assert result["dtype"] == dtype
    assert result["height"] == shape[1]
    assert result["width"] == shape[2]


@pytest.mark.parametrize(
    "array_specs, expected_dtype, expected_height, expected_width, expected_num_channels, description",
    [
        (
            {
                "image": {"shape": (100, 200, 3), "dtype": np.uint8},
                "images": {"shape": (5, 150, 250, 3), "dtype": np.uint16},
                "volume": {"shape": (10, 120, 220, 3), "dtype": np.float32},
            },
            np.uint8,
            100,
            200,
            3,
            "All keys present - should use 'image'",
        ),
        (
            {
                "images": {"shape": (5, 150, 250, 3), "dtype": np.uint16},
                "volume": {"shape": (10, 120, 220, 3), "dtype": np.float32},
            },
            np.uint16,
            150,
            250,
            3,
            "No 'image' - should use 'images'",
        ),
        (
            {
                "volume": {"shape": (10, 120, 220, 3), "dtype": np.float32},
            },
            np.float32,
            120,
            220,
            3,
            "No 'image' or 'images' - should use 'volume'",
        ),
    ],
)
def test_get_image_data_priority_order(
    array_specs, expected_dtype, expected_height, expected_width, expected_num_channels, description
):
    data = {key: np.zeros(spec["shape"], dtype=spec["dtype"]) for key, spec in array_specs.items()}

    result = get_image_data(data)
    assert result["dtype"] == expected_dtype
    assert result["height"] == expected_height
    assert result["width"] == expected_width
    assert result["num_channels"] == expected_num_channels


@pytest.mark.parametrize(
    "invalid_data, error_message",
    [
        ({}, "No valid image/volume data found"),
        ({"mask": np.zeros((100, 100)), "label": 5}, "No valid image/volume data found"),
        ({"img": np.zeros((100, 100)), "imgs": np.zeros((5, 100, 100))}, "No valid image/volume data found"),
    ],
)
def test_get_image_data_missing_keys(invalid_data, error_message):
    with pytest.raises(ValueError, match=error_message):
        get_image_data(invalid_data)


@pytest.mark.parametrize(
    "additional_keys",
    [
        {"mask": np.zeros((100, 200), dtype=np.float32), "label": 1, "metadata": {"source": "test"}},
        {"segmentation": np.zeros((100, 200)), "class_id": 42},
        {"annotations": [1, 2, 3], "timestamp": 123456},
    ],
)
def test_get_image_data_with_additional_keys(additional_keys):
    dtype = np.uint8
    img = np.zeros((100, 200, 3), dtype=dtype)
    data = {"image": img, **additional_keys}

    result = get_image_data(data)
    assert isinstance(result, dict)
    assert result["dtype"] == dtype
    assert result["height"] == 100
    assert result["width"] == 200


@pytest.mark.parametrize(
    "key, shape, expected_height_idx, expected_width_idx",
    [
        ("image", (100, 200, 3), 0, 1),  # Direct H, W
        ("images", (5, 100, 200, 3), 1, 2),  # Skip batch
        ("volume", (10, 100, 200), 1, 2),  # Skip depth
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64, np.int32])
def test_get_image_data_parametrized(key, shape, expected_height_idx, expected_width_idx, dtype):
    arr = np.zeros(shape, dtype=dtype)
    data = {key: arr}

    result = get_image_data(data)
    assert isinstance(result, dict)
    assert result["dtype"] == dtype
    assert result["height"] == shape[expected_height_idx]
    assert result["width"] == shape[expected_width_idx]


@pytest.mark.parametrize(
    "dtype_str, shape, expected_height, expected_width, expected_num_channels",
    [
        ("uint8", (100, 100, 1), 100, 100, 1),
        ("complex64", (50, 75, 1), 50, 75, 1),
        ("float16", (224, 224, 3), 224, 224, 3),
        ("int64", (32, 64, 1), 32, 64, 1),
    ],
)
def test_get_image_data_preserves_numpy_dtype_object(
    dtype_str, shape, expected_height, expected_width, expected_num_channels
):
    img = np.zeros(shape, dtype=np.dtype(dtype_str))
    data = {"image": img}
    result = get_image_data(data)

    assert isinstance(result["dtype"], np.dtype)
    assert result["dtype"] == np.dtype(dtype_str)
    assert result["height"] == expected_height
    assert result["width"] == expected_width
    assert result["num_channels"] == expected_num_channels


def test_get_image_data_returns_correct_keys():
    img = np.zeros((224, 224, 3), dtype=np.float32)
    data = {"image": img}
    result = get_image_data(data)

    expected_keys = {"dtype", "height", "width", "num_channels"}
    assert set(result.keys()) == expected_keys

    assert isinstance(result["dtype"], np.dtype)
    assert isinstance(result["height"], (int, np.integer))
    assert isinstance(result["width"], (int, np.integer))
    assert isinstance(result["num_channels"], (int, np.integer))


@pytest.mark.parametrize(
    "key, shape, expected_height, expected_width, expected_num_channels, description",
    [
        ("image", (100, 200, 3), 100, 200, 3, "Single image: shape[0] and shape[1] are H, W"),
        ("images", (5, 100, 200, 3), 100, 200, 3, "Batch of images: skip batch dimension to get H, W"),
        ("volume", (10, 100, 200, 1), 100, 200, 1, "Volume: skip depth dimension to get H, W"),
    ],
)
def test_get_image_data_shape_extraction_behavior(
    key, shape, expected_height, expected_width, expected_num_channels, description
):
    arr = np.zeros(shape)
    result = get_image_data({key: arr})
    assert (result["height"], result["width"], result["num_channels"]) == (
        expected_height,
        expected_width,
        expected_num_channels,
    )
