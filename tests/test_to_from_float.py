from typing import Any

import pytest
import numpy as np

from albucore.functions import from_float, from_float_numpy, from_float_opencv, to_float_numpy, to_float_opencv, to_float_lut, to_float, MAX_VALUES_BY_DTYPE
import cv2

CHANNELS = [1, 3, 5]
DTYPES = [np.uint8, np.float32]
BATCHES = [0, 2]


def generate_img(max_value: int, dtype: Any, channels: int, batch: int) -> np.ndarray:
    """Function generates a random image to be used in unit tests.

    The max_value parameter is ignored when `dtype == np.float32`.
    """
    batch_shape = (batch,) if batch > 0 else ()
    channel_shape = (channels,) if channels > 1 else ()
    shape = (*batch_shape, 100, 100, *channel_shape)
    return (
        np.random.rand(*shape).astype(np.float32)
        if dtype == np.float32
        else np.random.randint(0, max_value, shape).astype(dtype)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_functions_match(dtype, channels, batch):
    # Generate random image
    img = generate_img(256, dtype, channels, batch)

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    # Apply each function
    result_numpy = to_float_numpy(img, max_value)
    result_opencv = to_float_opencv(img, max_value)
    result_to_float = to_float(img)

    # Check that results match
    np.testing.assert_allclose(result_numpy, result_opencv, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result_numpy, result_to_float, rtol=1e-5, atol=1e-8)

    # For uint8, also check LUT method
    if dtype == np.uint8:
        result_lut = to_float_lut(img, max_value)
        np.testing.assert_allclose(result_numpy, result_lut, rtol=1e-5, atol=1e-8)



@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_output_type(dtype, channels, batch):
    img = generate_img(256, dtype, channels, batch)

    result = to_float(img)
    assert result.dtype == np.float32

@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_lut_uint8_only(channels, batch):
    img_uint8 = generate_img(256, np.uint8, channels, batch)
    img_uint16 = generate_img(65536, np.uint16, channels, batch)

    # Should work for uint8
    result_uint8 = to_float_lut(img_uint8, MAX_VALUES_BY_DTYPE[np.uint8])
    assert result_uint8.dtype == np.float32

    # Should raise an error for uint16 (unsupported dtype)
    with pytest.raises(ValueError):
        to_float_lut(img_uint16, 65535.0)

@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_shape_preservation(channels, batch):
    img = generate_img(256, np.uint8, channels, batch)

    result = to_float(img)
    assert result.shape == img.shape

def test_to_float_max_value_none():
    img = np.random.randint(0, 256, (100, 100)).astype(np.uint8)
    result = to_float(img)  # max_value not provided
    assert result.dtype == np.float32
    assert np.max(result) <= 1.0

def test_to_float_unsupported_dtype():
    img = np.random.randint(0, 256, (100, 100)).astype(np.int64)
    with pytest.raises(RuntimeError):
        to_float(img)

def test_to_float_custom_max_value():
    img = np.random.randint(0, 256, (100, 100)).astype(np.uint8)
    custom_max = 128.0
    result = to_float(img, max_value=custom_max)
    assert np.max(result) <= 2.0  # Since 255 / 128 = 1.9921875


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_numpy_vs_opencv(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    float_img = generate_img(None, np.float32, channels, batch)

    # Convert using both methods
    numpy_result = from_float_numpy(float_img, dtype, max_value)
    opencv_result = from_float_opencv(float_img, dtype, max_value)
    result = from_float(float_img, dtype, max_value)

    # Check that results are the same
    np.testing.assert_allclose(numpy_result, opencv_result, rtol=1e-5, atol=1)
    np.testing.assert_allclose(numpy_result, result, rtol=1e-5, atol=1)
    np.testing.assert_allclose(opencv_result, result, rtol=1e-5, atol=1)

    assert numpy_result.dtype == dtype
    assert opencv_result.dtype == dtype
    assert result.dtype == dtype

@pytest.mark.parametrize("dtype", [np.uint8])  # float32 roundtrip is lossy (quantizes to 0/1)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_from_float_roundtrip(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random image
    original_img = generate_img(max_value + 1, dtype, channels, batch)

    # Convert to float and back
    float_img = to_float(original_img, max_value)
    roundtrip_img = from_float_opencv(float_img, dtype, max_value)

    # Check that the result is the same as the original
    np.testing.assert_allclose(original_img, roundtrip_img, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_to_float_roundtrip(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    original_float_img = generate_img(None, np.float32, channels, batch)

    # Convert from float and back
    int_img = from_float(original_float_img, dtype, max_value)
    roundtrip_float_img = to_float(int_img, max_value)

    # Check that the result is the same as the original
    np.testing.assert_array_almost_equal(original_float_img, roundtrip_float_img, decimal=2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_opencv_channels(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    float_img = generate_img(None, np.float32, channels, batch)

    # Convert using OpenCV method
    result = from_float_opencv(float_img, dtype, max_value)

    # Check that the result has the correct number of channels
    assert result.shape == float_img.shape

    # Check that the result is correct
    np.testing.assert_allclose(
        result, cv2.multiply(float_img, np.full_like(float_img, max_value)), rtol=1e-5, atol=1
    )


def assert_array_equal_with_copy(original, processed):
    np.testing.assert_array_equal(original, processed)
    assert original is not processed  # Ensure a copy was made

@pytest.mark.parametrize("dtype", DTYPES + [np.float32])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_input_unchanged(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    img = generate_img(max_value + 1, dtype, channels, batch)

    img_copy = img.copy()
    _ = to_float(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_lut_input_unchanged(channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]
    img = generate_img(256, np.uint8, channels, batch)

    img_copy = img.copy()
    _ = to_float_lut(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES + [np.float32])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_numpy_input_unchanged(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    img = generate_img(max_value, dtype, channels, batch)

    img_copy = img.copy()
    _ = to_float_numpy(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES + [np.float32])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_to_float_opencv_input_unchanged(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    img = generate_img(max_value, dtype, channels, batch)

    img_copy = img.copy()
    _ = to_float_opencv(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_input_unchanged(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    img = generate_img(None, np.float32, channels, batch)

    img_copy = img.copy()
    _ = from_float(img, dtype, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_numpy_input_unchanged(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    img = generate_img(None, np.float32, channels, batch)

    img_copy = img.copy()
    _ = from_float_numpy(img, dtype, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES + [np.float32])
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_from_float_opencv_input_unchanged(dtype, channels, batch):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    img = generate_img(None, np.float32, channels, batch)

    img_copy = img.copy()
    _ = from_float_opencv(img, dtype, max_value)
    np.testing.assert_array_equal(img, img_copy)


def test_to_float_returns_same_object_for_float32():
    float32_image = np.random.rand(10, 10, 3).astype(np.float32)
    result = to_float(float32_image)
    assert result is float32_image  # Check if it's the same object
