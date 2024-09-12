import pytest
import numpy as np

from albucore.functions import to_float_numpy, to_float_opencv, to_float_lut, to_float, MAX_VALUES_BY_DTYPE


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
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



@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_to_float_output_type(dtype, channels):
    shape = (100, 100, channels) if channels > 1 else (100, 100)
    img = np.random.randint(0, 256, shape).astype(dtype)

    result = to_float(img)
    assert result.dtype == np.float32

@pytest.mark.parametrize("channels", [1, 3, 5])
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

@pytest.mark.parametrize("channels", [1, 3, 5])
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
