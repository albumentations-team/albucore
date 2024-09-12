import pytest
import numpy as np

from albucore.functions import from_float, from_float_numpy, from_float_opencv, to_float_numpy, to_float_opencv, to_float_lut, to_float, MAX_VALUES_BY_DTYPE
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS
import cv2

CHANNELS = [1, 3, 5]
DTYPES = [np.uint8, np.uint16]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_functions_match(dtype, channels):
    # Generate random image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.randint(0, 256, shape).astype(dtype)

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
def test_to_float_output_type(dtype, channels):
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.randint(0, 256, shape).astype(dtype)

    result = to_float(img)
    assert result.dtype == np.float32

@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_lut_uint8_only(channels):
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img_uint8 = np.random.randint(0, 256, shape).astype(np.uint8)
    img_uint16 = np.random.randint(0, 65536, shape).astype(np.uint16)

    # Should work for uint8
    result_uint8 = to_float_lut(img_uint8, MAX_VALUES_BY_DTYPE[np.uint8])
    assert result_uint8.dtype == np.float32

    # Should raise an error for uint16
    with pytest.raises(ValueError):
        to_float_lut(img_uint16, MAX_VALUES_BY_DTYPE[np.uint16])

@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_channel_preservation(channels):
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.randint(0, 256, shape).astype(np.uint8)

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
def test_from_float_numpy_vs_opencv(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    float_img = np.random.rand(*shape).astype(np.float32)

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

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_from_float_roundtrip(dtype, channels, max_value):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    if dtype == np.float32:
        original_img = np.random.rand(*shape).astype(dtype)
    else:
        max_val = MAX_VALUES_BY_DTYPE[dtype] if max_value is None else max_value
        original_img = np.random.randint(0, max_val + 1, shape).astype(dtype)

    # Convert to float and back
    float_img = to_float(original_img, max_value)
    roundtrip_img = from_float_opencv(float_img, dtype, max_value)

    # Check that the result is the same as the original
    np.testing.assert_allclose(original_img, roundtrip_img, rtol=1e-5, atol=1)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_to_float_roundtrip(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    original_float_img = np.random.rand(*shape).astype(np.float32)

    # Convert from float and back
    int_img = from_float_opencv(original_float_img, dtype, max_value)
    roundtrip_float_img = to_float(int_img, max_value)

    # Check that the result is the same as the original
    np.testing.assert_allclose(original_float_img, roundtrip_float_img, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_opencv_channels(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    float_img = np.random.rand(*shape).astype(np.float32)

    # Convert using OpenCV method
    result = from_float_opencv(float_img, dtype, max_value)

    # Check that the result has the correct number of channels
    if channels > 1:
        assert result.shape[2] == channels
    else:
        assert len(result.shape) == 2

    # Check that the function uses the correct method based on number of channels
    if channels > MAX_OPENCV_WORKING_CHANNELS:
        # For more than 4 channels, it should use np.full_like
        assert np.array_equal(result, cv2.multiply(float_img, np.full_like(float_img, max_value or MAX_VALUES_BY_DTYPE[dtype])))
    else:
        # For 4 or fewer channels, it should use scalar multiplication
        assert np.array_equal(result, cv2.multiply(float_img, max_value or MAX_VALUES_BY_DTYPE[dtype]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_numpy_vs_opencv(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    float_img = np.random.rand(*shape).astype(np.float32)

    # Convert using both methods
    numpy_result = from_float_numpy(float_img, dtype, max_value)
    opencv_result = from_float_opencv(float_img, dtype, max_value)

    # Check that results are the same
    np.testing.assert_allclose(numpy_result, opencv_result, rtol=1e-5, atol=1)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_from_float_roundtrip(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    if dtype == np.float32:
        original_img = np.random.rand(*shape).astype(dtype)
    else:
        max_val = MAX_VALUES_BY_DTYPE[dtype] if max_value is None else max_value
        original_img = np.random.randint(0, max_val + 1, shape).astype(dtype)

    # Convert to float and back
    float_img = to_float(original_img, max_value)
    roundtrip_img = from_float(float_img, dtype, max_value)

    # Check that the result is the same as the original
    np.testing.assert_array_almost_equal(original_img, roundtrip_img, decimal=4)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_to_float_roundtrip(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    original_float_img = np.random.rand(*shape).astype(np.float32)

    # Convert from float and back
    int_img = from_float(original_float_img, dtype, max_value)
    roundtrip_float_img = to_float(int_img, max_value)

    # Check that the result is the same as the origina
    np.testing.assert_array_almost_equal(original_float_img, roundtrip_float_img, decimal=2)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_opencv_channels(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    # Generate random float image
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    float_img = np.random.rand(*shape).astype(np.float32)

    # Convert using OpenCV method
    result = from_float_opencv(float_img, dtype, max_value)

    assert result.dtype == dtype
    assert result.max() <= max_value

    # Check that the result has the correct number of channels
    if channels > 1:
        assert result.shape[2] == channels
    else:
        assert len(result.shape) == 2

    # Check that the function uses the correct method based on number of channels
    if channels > MAX_OPENCV_WORKING_CHANNELS:
        # For more than 4 channels, it should use np.full_like
        np.testing.assert_allclose(result, cv2.multiply(float_img, np.full_like(float_img, max_value)), rtol=1e-5, atol=1)
    else:
        # For 4 or fewer channels, it should use scalar multiplication
        np.testing.assert_allclose(result, cv2.multiply(float_img, max_value), rtol=1e-5, atol=1)


def assert_array_equal_with_copy(original, processed):
    np.testing.assert_array_equal(original, processed)
    assert original is not processed  # Ensure a copy was made

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_input_unchanged(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    if dtype == np.float32:
        img = np.random.rand(*shape).astype(dtype)
    else:
        max_val = MAX_VALUES_BY_DTYPE[dtype] if max_value is None else max_value
        img = np.random.randint(0, max_val + 1, shape).astype(dtype)

    img_copy = img.copy()
    _ = to_float(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_lut_input_unchanged(channels):
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.randint(0, 256, shape).astype(np.uint8)

    img_copy = img.copy()
    _ = to_float_lut(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_to_float_numpy_input_unchanged(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    if dtype == np.float32:
        img = np.random.rand(*shape).astype(dtype)
    else:
        max_val = MAX_VALUES_BY_DTYPE[dtype] if max_value is None else max_value
        img = np.random.randint(0, max_val + 1, shape).astype(dtype)

    img_copy = img.copy()
    _ = to_float_numpy(img, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_input_unchanged(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.rand(*shape).astype(np.float32)

    img_copy = img.copy()
    _ = from_float(img, dtype, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_numpy_input_unchanged(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.rand(*shape).astype(np.float32)

    img_copy = img.copy()
    _ = from_float_numpy(img, dtype, max_value)
    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
def test_from_float_opencv_input_unchanged(dtype, channels):
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.rand(*shape).astype(np.float32)

    img_copy = img.copy()
    _ = from_float_opencv(img, dtype, max_value)
    np.testing.assert_array_equal(img, img_copy)
