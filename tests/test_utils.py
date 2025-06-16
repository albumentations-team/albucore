import numpy as np
import pytest
import cv2
from albucore.decorators import contiguous
from albucore.functions import float32_io, from_float, to_float, uint8_io
from albucore.utils import NPDTYPE_TO_OPENCV_DTYPE, clip, convert_value, get_opencv_dtype_from_numpy, get_num_channels, is_grayscale_image, get_image_shape


@pytest.mark.parametrize("input_img, dtype, expected", [
    (np.array([[-300, 0], [100, 400]], dtype=np.float32), np.uint8, np.array([[0, 0], [100, 255]], dtype=np.float32)),
    (np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32), np.float32, np.array([[0, 0], [0.5, 1.0]], dtype=np.float32))
])
@pytest.mark.parametrize("inplace", [False, True])
def test_clip(input_img, dtype, expected, inplace):
    clipped = clip(input_img, dtype=dtype, inplace=inplace)
    np.testing.assert_array_equal(clipped, expected)
    assert clipped.dtype == dtype

valid_cv2_types = {
    cv2.CV_8U, cv2.CV_16U, cv2.CV_32F, cv2.CV_64F, cv2.CV_32S
}

@pytest.mark.parametrize("cv_type", NPDTYPE_TO_OPENCV_DTYPE.values())
def test_valid_cv2_types(cv_type):
    assert cv_type in valid_cv2_types, f"{cv_type} is not a valid cv2 type"


def test_cv_dtype_from_np():
    assert get_opencv_dtype_from_numpy(np.uint8) == cv2.CV_8U
    assert get_opencv_dtype_from_numpy(np.uint16) == cv2.CV_16U
    assert get_opencv_dtype_from_numpy(np.float32) == cv2.CV_32F
    assert get_opencv_dtype_from_numpy(np.float64) == cv2.CV_64F
    assert get_opencv_dtype_from_numpy(np.int32) == cv2.CV_32S

    assert get_opencv_dtype_from_numpy(np.dtype("uint8")) == cv2.CV_8U
    assert get_opencv_dtype_from_numpy(np.dtype("uint16")) == cv2.CV_16U
    assert get_opencv_dtype_from_numpy(np.dtype("float32")) == cv2.CV_32F
    assert get_opencv_dtype_from_numpy(np.dtype("float64")) == cv2.CV_64F
    assert get_opencv_dtype_from_numpy(np.dtype("int32")) == cv2.CV_32S


@pytest.mark.parametrize(
    "value, num_channels, expected",
    [
        ((1.5), 1, 1.5),
        (np.array([1.5]), 3, 1.5),
        (np.array([1.5, 2.5]), 1, 1.5),
        (np.array([1.5, 2.5, 0.5]), 2, np.array([1.5, 2.5,], dtype=np.float32)),
        (3, 2, 3),
        (np.array((1.5)), 2, 1.5),
        (np.reciprocal(np.array(0.2, dtype=np.float64)), 2, 5),
    ]
)
def test_convert_value(value, num_channels, expected):
    result = convert_value(value, num_channels)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected

@contiguous
def process_image(img: np.ndarray) -> np.ndarray:
    # For demonstration, let's just return a non-contiguous view
    return img[::-1, ::-1]

@pytest.mark.parametrize(
    "input_array",
    [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),  # C-contiguous array
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[::-1, ::-1],  # Non-contiguous view
        np.arange(100).reshape(10, 10),  # Another C-contiguous array
        np.ones([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)  # 3D array with transpose
    ]
)
def test_contiguous_decorator(input_array):
    # Check if input is made contiguous
    contiguous_input = np.require(input_array, requirements=["C_CONTIGUOUS"])

    assert contiguous_input.flags["C_CONTIGUOUS"], "Input array is not C-contiguous"

    # Process the array using the decorated function
    output_array = process_image(input_array)

    # Check if output is contiguous
    assert output_array.flags["C_CONTIGUOUS"], "Output array is not C-contiguous"

    non_contiguous_array = np.asfortranarray(input_array)

    output_array = process_image(non_contiguous_array)
    assert output_array.flags["C_CONTIGUOUS"], "Output array is not C-contiguous"

    # Check if the content is correct (same as reversing the original array)
    expected_output = input_array[::-1, ::-1]
    np.testing.assert_array_equal(output_array, expected_output), "Output array content is not as expected"


# Sample functions to be wrapped
@float32_io
def dummy_float32_func(img):
    return img * 2

@uint8_io
def dummy_uint8_func(img):
    return np.clip(img + 10, 0, 255).astype(np.uint8)

# Test data
@pytest.fixture(params=[
    np.uint8, np.float32
])
def test_image(request):
    dtype = request.param
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 256, (10, 10, 3), dtype=dtype)
    else:
        return np.random.rand(10, 10, 3).astype(dtype)

# Tests
@pytest.mark.parametrize("wrapper,func, image", [
    (float32_io, dummy_float32_func, np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)),
    (uint8_io, dummy_uint8_func, np.random.rand(10, 10, 3).astype(np.float32))
])
def test_io_wrapper(wrapper, func, image):
    input_dtype = image.dtype
    result = func(image)

    # Check if the output dtype matches the input dtype
    assert result.dtype == input_dtype

    # Check if the function was actually applied
    if wrapper == float32_io:
        expected = from_float(to_float(image) * 2, input_dtype)
    else:  # uint8_io
        expected = to_float(from_float(image, np.uint8) + 10)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("wrapper,func,expected_intermediate_dtype", [
    (float32_io, dummy_float32_func, np.float32),
    (uint8_io, dummy_uint8_func, np.uint8)
])
def test_intermediate_dtype(wrapper, func, expected_intermediate_dtype, test_image):
    original_func = func.__wrapped__  # Access the original function

    def check_dtype(img):
        assert img.dtype == expected_intermediate_dtype
        return original_func(img)

    wrapped_func = wrapper(check_dtype)
    wrapped_func(test_image)  # This will raise an assertion error if the intermediate dtype is incorrect

def test_float32_io_preserves_float32(test_image):
    if test_image.dtype == np.float32:
        result = dummy_float32_func(test_image)
        assert result.dtype == np.float32

def test_uint8_io_preserves_uint8(test_image):
    if test_image.dtype == np.uint8:
        result = dummy_uint8_func(test_image)
        assert result.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_float32_io_does_not_modify_input(dtype):
    @float32_io
    def identity(img):
        return img

    shape = (10, 10, 3)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, np.iinfo(dtype).max, shape).astype(dtype)
    else:
        original = np.random.rand(*shape).astype(dtype)

    original_copy = original.copy()
    _ = identity(original)

    np.testing.assert_array_equal(original, original_copy)

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_uint8_io_does_not_modify_input(dtype):
    @uint8_io
    def identity(img):
        return img

    shape = (10, 10, 3)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, np.iinfo(dtype).max, shape).astype(dtype)
    else:
        original = np.random.rand(*shape).astype(dtype)

    original_copy = original.copy()
    _ = identity(original)

    np.testing.assert_array_equal(original, original_copy)

@pytest.mark.parametrize("wrapper", [float32_io, uint8_io])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_wrapper_preserves_dtype(wrapper, dtype):
    @wrapper
    def identity(img):
        return img

    shape = (10, 10, 3)
    if np.issubdtype(dtype, np.integer):
        original = np.random.randint(0, np.iinfo(dtype).max, shape).astype(dtype)
    else:
        original = np.random.rand(*shape).astype(dtype)

    result = identity(original)

    assert result.dtype == dtype

@pytest.mark.parametrize("wrapper", [float32_io, uint8_io])
def test_wrapper_intermediate_dtype(wrapper):
    intermediate_dtype = np.float32 if wrapper == float32_io else np.uint8

    @wrapper
    def check_dtype(img):
        assert img.dtype == intermediate_dtype
        return img

    original = np.random.rand(10, 10, 3).astype(np.float32)
    _ = check_dtype(original)


@pytest.mark.parametrize("shape, expected_channels, description", [
    # 2D grayscale image (H, W)
    ((100, 99), 1, "2D grayscale image"),
    ((256, 256), 1, "2D grayscale image"),

    # 3D image with channels (H, W, C)
    ((100, 99, 1), 1, "3D image with 1 channel"),
    ((100, 99, 3), 3, "3D RGB image"),
    ((100, 99, 7), 7, "3D multi-channel image"),

    # 3D volume (D, H, W) - WARNING: returns W as channels!
    ((10, 100, 99), 99, "3D volume (D,H,W) - WARNING: returns W as channels"),

    # 4D batch of images (N, H, W, C)
    ((4, 100, 99, 1), 1, "Batch of grayscale images"),
    ((4, 100, 99, 3), 3, "Batch of RGB images"),
    ((10, 224, 224, 3), 3, "Batch of RGB images"),

    # 4D batch of volumes (N, D, H, W) - WARNING: returns W as channels!
    ((2, 10, 100, 99), 99, "Batch of volumes (N,D,H,W) - WARNING: returns W as channels"),

    # 5D batch of volumes with channels (N, D, H, W, C)
    ((2, 10, 100, 99, 1), 1, "Batch of volumes with 1 channel"),
    ((2, 10, 100, 99, 3), 3, "Batch of volumes with 3 channels"),

    # Edge cases
    ((1, 1, 1), 1, "Minimal 3D array"),
    ((1, 1), 1, "Minimal 2D array"),
    ((100,), 1, "1D array"),
])
def test_get_num_channels(shape, expected_channels, description):
    """Test get_num_channels for various array dimensions."""
    image = np.zeros(shape)
    assert get_num_channels(image) == expected_channels, f"Failed for {description} with shape {shape}"


@pytest.mark.parametrize("shape, has_batch_dim, has_depth_dim, expected_channels, description", [
    # HW: shape=(100, 200) → channels=1
    ((100, 200), False, False, 1, "HW: grayscale image"),

    # HWC: shape=(100, 200, 3) → channels=3
    ((100, 200, 3), False, False, 3, "HWC: RGB image"),
    ((100, 200, 1), False, False, 1, "HWC: single channel image"),
    ((100, 200, 4), False, False, 4, "HWC: RGBA image"),

    # NHW: shape=(10, 100, 200) → channels=1 (batch of grayscale)
    ((10, 100, 200), True, False, 1, "NHW: batch of grayscale images"),

    # NHWC: shape=(10, 100, 200, 3) → channels=3 (batch of RGB)
    ((10, 100, 200, 3), True, False, 3, "NHWC: batch of RGB images"),
    ((10, 100, 200, 1), True, False, 1, "NHWC: batch of single channel images"),

    # DHW: shape=(5, 100, 200) → channels=1 (3D volume)
    ((5, 100, 200), False, True, 1, "DHW: 3D volume"),

    # DHWC: shape=(5, 100, 200, 3) → channels=3 (3D volume with RGB slices)
    ((5, 100, 200, 3), False, True, 3, "DHWC: 3D volume with RGB slices"),
    ((5, 100, 200, 1), False, True, 1, "DHWC: 3D volume with single channel"),

    # DNHW: shape=(5, 10, 100, 200) → channels=1 (batch of 3D volumes)
    ((5, 10, 100, 200), True, True, 1, "DNHW: batch of 3D volumes"),

    # DNHWC: shape=(5, 10, 100, 200, 3) → channels=3 (batch of 3D volumes with RGB)
    ((5, 10, 100, 200, 3), True, True, 3, "DNHWC: batch of 3D volumes with RGB"),
    ((5, 10, 100, 200, 1), True, True, 1, "DNHWC: batch of 3D volumes with single channel"),

    # Additional edge cases
    ((32, 32), False, False, 1, "HW: square grayscale"),
    ((224, 224, 3), False, False, 3, "HWC: standard RGB image size"),
    ((1, 512, 512), True, False, 1, "NHW: single image in batch"),
    ((1, 512, 512, 3), True, False, 3, "NHWC: single RGB image in batch"),
])
def test_get_num_channels_with_dimension_flags(shape, has_batch_dim, has_depth_dim, expected_channels, description):
    """Test get_num_channels with batch and depth dimension flags."""
    image = np.zeros(shape)
    result = get_num_channels(image, has_batch_dim=has_batch_dim, has_depth_dim=has_depth_dim)
    assert result == expected_channels, f"Failed for {description} with shape {shape}, has_batch_dim={has_batch_dim}, has_depth_dim={has_depth_dim}"


@pytest.mark.parametrize("shape, has_batch_dim, has_depth_dim, expected_grayscale, description", [
    # HW: shape=(100, 200) → grayscale=True
    ((100, 200), False, False, True, "HW: grayscale image"),

    # HWC: shape=(100, 200, 3) → grayscale=False
    ((100, 200, 3), False, False, False, "HWC: RGB image"),
    ((100, 200, 1), False, False, True, "HWC: single channel image"),
    ((100, 200, 4), False, False, False, "HWC: RGBA image"),

    # NHW: shape=(10, 100, 200) → grayscale=True
    ((10, 100, 200), True, False, True, "NHW: batch of grayscale images"),

    # NHWC: shape=(10, 100, 200, 3) → grayscale=False
    ((10, 100, 200, 3), True, False, False, "NHWC: batch of RGB images"),
    ((10, 100, 200, 1), True, False, True, "NHWC: batch of single channel images"),

    # DHW: shape=(5, 100, 200) → grayscale=True
    ((5, 100, 200), False, True, True, "DHW: 3D volume"),

    # DHWC: shape=(5, 100, 200, 3) → grayscale=False
    ((5, 100, 200, 3), False, True, False, "DHWC: 3D volume with RGB slices"),
    ((5, 100, 200, 1), False, True, True, "DHWC: 3D volume with single channel"),

    # DNHW: shape=(5, 10, 100, 200) → grayscale=True
    ((5, 10, 100, 200), True, True, True, "DNHW: batch of 3D volumes"),

    # DNHWC: shape=(5, 10, 100, 200, 3) → grayscale=False
    ((5, 10, 100, 200, 3), True, True, False, "DNHWC: batch of 3D volumes with RGB"),
    ((5, 10, 100, 200, 1), True, True, True, "DNHWC: batch of 3D volumes with single channel"),
])
def test_is_grayscale_image(shape, has_batch_dim, has_depth_dim, expected_grayscale, description):
    """Test is_grayscale_image with various shape combinations."""
    image = np.zeros(shape)
    result = is_grayscale_image(image, has_batch_dim=has_batch_dim, has_depth_dim=has_depth_dim)
    assert result == expected_grayscale, f"Failed for {description} with shape {shape}, has_batch_dim={has_batch_dim}, has_depth_dim={has_depth_dim}"


@pytest.mark.parametrize("shape, has_batch_dim, has_depth_dim", [
    # Basic 2D cases
    ((100, 200), False, False),
    ((100, 200, 1), False, False),
    ((100, 200, 3), False, False),
    # Batch cases (NHW/NHWC)
    ((10, 100, 200), True, False),
    ((10, 100, 200, 1), True, False),
    ((10, 100, 200, 3), True, False),
    # Depth cases (DHW/DHWC)
    ((5, 100, 200), False, True),
    ((5, 100, 200, 1), False, True),
    ((5, 100, 200, 3), False, True),
    # Batch and depth cases (DNHW/DNHWC)
    ((5, 10, 100, 200), True, True),
    ((5, 10, 100, 200, 1), True, True),
    ((5, 10, 100, 200, 3), True, True),
])
def test_get_num_channels_and_is_grayscale_consistency(shape, has_batch_dim, has_depth_dim):
    """Test that get_num_channels and is_grayscale_image are consistent."""
    image = np.zeros(shape)
    num_channels = get_num_channels(image, has_batch_dim=has_batch_dim, has_depth_dim=has_depth_dim)
    is_grayscale = is_grayscale_image(image, has_batch_dim=has_batch_dim, has_depth_dim=has_depth_dim)

    # is_grayscale should be True if and only if num_channels == 1
    assert (num_channels == 1) == is_grayscale, (
        f"Inconsistency for shape {shape}, has_batch_dim={has_batch_dim}, has_depth_dim={has_depth_dim}: "
        f"num_channels={num_channels}, is_grayscale={is_grayscale}"
    )


@pytest.mark.parametrize("shape, has_batch_dim, has_depth_dim, expected_hwc, description", [
    # HW: shape=(101, 99) → (101, 99, 1)
    ((101, 99), False, False, (101, 99, 1), "HW: grayscale image"),
    ((150, 200), False, False, (150, 200, 1), "HW: another grayscale image"),

    # HWC: shape=(101, 99, 3) → (101, 99, 3)
    ((101, 99, 3), False, False, (101, 99, 3), "HWC: RGB image"),
    ((101, 99, 1), False, False, (101, 99, 1), "HWC: single channel image"),
    ((256, 512, 4), False, False, (256, 512, 4), "HWC: RGBA image"),

    # BHW: shape=(3, 201, 99) → (201, 99, 1)
    ((3, 201, 99), True, False, (201, 99, 1), "BHW: batch of grayscale images"),
    ((10, 150, 200), True, False, (150, 200, 1), "BHW: larger batch of grayscale"),

    # BHWC: shape=(3, 201, 99, 3) → (201, 99, 3)
    ((3, 201, 99, 3), True, False, (201, 99, 3), "BHWC: batch of RGB images"),
    ((1, 512, 512, 3), True, False, (512, 512, 3), "BHWC: single RGB image in batch"),
    ((5, 224, 224, 1), True, False, (224, 224, 1), "BHWC: batch of single channel images"),

    # DHW: shape=(4, 101, 99) → (101, 99, 1)
    ((4, 101, 99), False, True, (101, 99, 1), "DHW: 3D volume"),
    ((10, 256, 256), False, True, (256, 256, 1), "DHW: larger 3D volume"),

    # DHWC: shape=(4, 101, 99, 3) → (101, 99, 3)
    ((4, 101, 99, 3), False, True, (101, 99, 3), "DHWC: 3D volume with RGB slices"),
    ((8, 128, 128, 1), False, True, (128, 128, 1), "DHWC: 3D volume with single channel"),

    # BDHW: shape=(2, 4, 101, 99) → (101, 99, 1)
    ((2, 4, 101, 99), True, True, (101, 99, 1), "BDHW: batch of 3D volumes"),
    ((1, 5, 256, 256), True, True, (256, 256, 1), "BDHW: single 3D volume in batch"),

    # BDHWC: shape=(2, 4, 101, 99, 3) → (101, 99, 3)
    ((2, 4, 101, 99, 3), True, True, (101, 99, 3), "BDHWC: batch of 3D volumes with RGB"),
    ((3, 8, 64, 64, 1), True, True, (64, 64, 1), "BDHWC: batch of 3D volumes with single channel"),
    ((5, 10, 224, 224, 4), True, True, (224, 224, 4), "BDHWC: batch of 3D volumes with RGBA"),

    # Edge cases
    ((1, 1), False, False, (1, 1, 1), "HW: minimal 2D array"),
    ((1, 1, 1), False, False, (1, 1, 1), "HWC: minimal 3D array"),
    ((1, 2, 3), True, False, (2, 3, 1), "BHW: minimal batch"),
    ((1, 1, 2, 3, 4), True, True, (2, 3, 4), "BDHWC: minimal batch of volumes"),
])
def test_get_image_shape(shape, has_batch_dim, has_depth_dim, expected_hwc, description):
    """Test get_image_shape for various array dimensions and configurations."""
    image = np.zeros(shape)
    result = get_image_shape(image, has_batch_dim=has_batch_dim, has_depth_dim=has_depth_dim)
    assert result == expected_hwc, f"Failed for {description} with shape {shape}, has_batch_dim={has_batch_dim}, has_depth_dim={has_depth_dim}"


def test_get_image_shape_errors():
    """Test that get_image_shape raises appropriate errors for invalid inputs."""
    # Not enough dimensions - expecting HW but only have 1D
    with pytest.raises(ValueError, match="doesn't have enough dimensions"):
        get_image_shape(np.zeros(10), has_batch_dim=False, has_depth_dim=False)

    # Not enough dimensions - expecting BHW but only have 2D
    with pytest.raises(ValueError, match="doesn't have enough dimensions"):
        get_image_shape(np.zeros((10, 20)), has_batch_dim=True, has_depth_dim=False)

    # Not enough dimensions - expecting DHW but only have 2D
    with pytest.raises(ValueError, match="doesn't have enough dimensions"):
        get_image_shape(np.zeros((10, 20)), has_batch_dim=False, has_depth_dim=True)

    # Not enough dimensions - expecting BDHW but only have 3D
    with pytest.raises(ValueError, match="doesn't have enough dimensions"):
        get_image_shape(np.zeros((10, 20, 30)), has_batch_dim=True, has_depth_dim=True)


def test_get_image_shape_consistency_with_examples():
    """Test get_image_shape with the exact examples from the user's request."""
    # 1. Single image (101, 99, 3) -> (101, 99, 3)
    img1 = np.zeros((101, 99, 3))
    assert get_image_shape(img1) == (101, 99, 3)

    # 2. Batch of images (3, 201, 99, 3) -> (201, 99, 3)
    img2 = np.zeros((3, 201, 99, 3))
    assert get_image_shape(img2, has_batch_dim=True) == (201, 99, 3)

    # 3. Volume (4, 101, 99, 3) -> (101, 99, 3)
    img3 = np.zeros((4, 101, 99, 3))
    assert get_image_shape(img3, has_depth_dim=True) == (101, 99, 3)

    # 4. Batch of volumes (2, 4, 101, 99, 3) -> (101, 99, 3)
    img4 = np.zeros((2, 4, 101, 99, 3))
    assert get_image_shape(img4, has_batch_dim=True, has_depth_dim=True) == (101, 99, 3)
