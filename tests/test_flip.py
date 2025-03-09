import pytest
import numpy as np
import cv2
from albucore.functions import hflip_numpy, hflip_cv2, vflip, vflip_cv2, vflip_numpy, hflip, _flip_multichannel

@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_hflip_functions(channels, dtype):
    # Create a sample image
    if channels == 1:
        img = np.arange(20).reshape(4, 5).astype(dtype)
    else:
        img = np.arange(20 * channels).reshape(4, 5, channels).astype(dtype)

    # Apply both flip functions
    flipped_numpy = hflip_numpy(img)
    flipped_cv2 = hflip_cv2(img)

    # Test 1: Check if results match
    np.testing.assert_array_equal(flipped_numpy, flipped_cv2)

    # Test 2: Check if the flip is correct

    expected = img[:, ::-1]
    np.testing.assert_array_equal(flipped_numpy, expected)

    # Test 3: Check if results are contiguous
    assert flipped_numpy.flags['C_CONTIGUOUS']
    assert flipped_cv2.flags['C_CONTIGUOUS']

    # Test 4: Check if dtype is preserved
    assert flipped_numpy.dtype == dtype
    assert flipped_cv2.dtype == dtype

@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 1), (10, 10, 3), (10, 10, 5)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_hflip_identity(shape, dtype):
    # Create a random image
    img = np.random.rand(*shape).astype(dtype)

    # Apply flip twice should result in the original image
    flipped_twice_numpy = hflip_numpy(hflip_numpy(img))
    flipped_twice_cv2 = hflip_cv2(hflip_cv2(img))

    np.testing.assert_array_almost_equal(img, flipped_twice_numpy)
    np.testing.assert_array_almost_equal(img, flipped_twice_cv2)


@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_vflip_functions(channels, dtype):
    # Create a sample image
    if channels == 1:
        img = np.arange(20).reshape(4, 5).astype(dtype)
    else:
        img = np.arange(20 * channels).reshape(4, 5, channels).astype(dtype)

    # Apply all flip functions
    flipped_numpy = vflip_numpy(img)
    flipped_cv2 = vflip_cv2(img)
    flipped_main = vflip(img)

    # Test 1: Check if results match
    np.testing.assert_array_equal(flipped_numpy, flipped_cv2)
    np.testing.assert_array_equal(flipped_numpy, flipped_main)

    # Test 2: Check if the flip is correct
    expected = img[::-1, :]
    np.testing.assert_array_equal(flipped_numpy, expected)

    # Test 3: Check if results are contiguous
    assert flipped_numpy.flags['C_CONTIGUOUS']
    assert flipped_cv2.flags['C_CONTIGUOUS']
    assert flipped_main.flags['C_CONTIGUOUS']

    # Test 4: Check if dtype is preserved
    assert flipped_numpy.dtype == dtype
    assert flipped_cv2.dtype == dtype
    assert flipped_main.dtype == dtype

@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 1), (10, 10, 3), (10, 10, 5)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_vflip_identity(shape, dtype):
    # Create a random image
    img = np.random.rand(*shape).astype(dtype)

    # Apply flip twice should result in the original image
    flipped_twice_numpy = vflip_numpy(vflip_numpy(img))
    flipped_twice_cv2 = vflip_cv2(vflip_cv2(img))
    flipped_twice_main = vflip(vflip(img))

    np.testing.assert_array_almost_equal(img, flipped_twice_numpy)
    np.testing.assert_array_almost_equal(img, flipped_twice_cv2)
    np.testing.assert_array_almost_equal(img, flipped_twice_main)

@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_multichannel_hflip(dtype):
    """Test horizontal flip with more than 512 channels."""
    # Create a sample image with 600 channels (exceeds OpenCV's 512 channel limit)
    channels = 600
    height, width = 10, 15
    img = np.arange(height * width * channels).reshape(height, width, channels).astype(dtype)

    # Apply flip functions
    flipped_numpy = hflip_numpy(img)
    flipped_cv2 = hflip_cv2(img)
    flipped_main = hflip(img)

    # Test 1: Check if results match
    np.testing.assert_array_equal(flipped_numpy, flipped_cv2)
    np.testing.assert_array_equal(flipped_numpy, flipped_main)

    # Test 2: Check if the flip is correct
    expected = img[:, ::-1, :]
    np.testing.assert_array_equal(flipped_numpy, expected)

    # Test 3: Check if results are contiguous
    assert flipped_cv2.flags['C_CONTIGUOUS']
    assert flipped_main.flags['C_CONTIGUOUS']

    # Test 4: Check if dtype is preserved
    assert flipped_cv2.dtype == img.dtype
    assert flipped_main.dtype == img.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_multichannel_vflip(dtype):
    """Test vertical flip with more than 512 channels."""
    # Create a sample image with 600 channels (exceeds OpenCV's 512 channel limit)
    channels = 600
    height, width = 10, 15
    img = np.arange(height * width * channels).reshape(height, width, channels).astype(dtype)

    # Apply flip functions
    flipped_numpy = vflip_numpy(img)
    flipped_cv2 = vflip_cv2(img)
    flipped_main = vflip(img)

    # Test 1: Check if results match
    np.testing.assert_array_equal(flipped_numpy, flipped_cv2)
    np.testing.assert_array_equal(flipped_numpy, flipped_main)

    # Test 2: Check if the flip is correct
    expected = img[::-1, :, :]
    np.testing.assert_array_equal(flipped_numpy, expected)

    # Test 3: Check if results are contiguous
    assert flipped_cv2.flags['C_CONTIGUOUS']
    assert flipped_main.flags['C_CONTIGUOUS']

    # Test 4: Check if dtype is preserved
    assert flipped_cv2.dtype == img.dtype
    assert flipped_main.dtype == img.dtype


@pytest.mark.parametrize("channels", [513, 600, 1024])
@pytest.mark.parametrize("flip_code", [0, 1, -1])
def test_flip_multichannel_function(channels, flip_code):
    """Test the _flip_multichannel function directly with different channel counts and flip codes."""
    height, width = 8, 10
    img = np.arange(height * width * channels).reshape(height, width, channels).astype(np.float32)

    # Apply the multichannel flip function
    flipped = _flip_multichannel(img, flip_code)

    # Determine expected result based on flip_code
    if flip_code == 0:  # vertical flip
        expected = img[::-1, :, :]
    elif flip_code == 1:  # horizontal flip
        expected = img[:, ::-1, :]
    else:  # both
        expected = img[::-1, ::-1, :]

    # Check if the result matches the expected output
    np.testing.assert_array_equal(flipped, expected)

    # Check if the result is contiguous
    assert flipped.flags['C_CONTIGUOUS']

    # Check if dtype is preserved
    assert flipped.dtype == img.dtype
