import pytest
import numpy as np


from albucore.functions import (
    normalize_per_image_opencv,
    normalize_per_image_numpy,
    normalize_per_image_lut,
    normalize_per_image,
    normalize_per_image_batch,
)


@pytest.mark.parametrize("img, normalization, expected", [
    (np.array([[1, 2], [3, 4]]), "min_max", np.array([[0, 1/3], [2/3, 1]], dtype=np.float32)),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "min_max_per_channel", np.array([[[0, 0], [1/3, 1/3]], [[2/3, 2/3], [1, 1]]], dtype=np.float32)),
    (np.array([[1, 2], [3, 4]]), "image", np.array([[-1.34164079, -0.4472136 ], [ 0.4472136 ,  1.34164079]], dtype=np.float32)),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "image_per_channel", np.array([[[-1.34164079, -1.34164079],
        [-0.4472136 , -0.4472136 ]],
       [[ 0.4472136 ,  0.4472136 ],
        [ 1.34164079,  1.34164079]]], dtype=np.float32))
])
@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_normalize_per_image(img, normalization, expected, dtype):
    img = img.astype(dtype)
    result = normalize_per_image(img, normalization)
    result_np = normalize_per_image_numpy(img, normalization)
    result_cv2 = normalize_per_image_opencv(img, normalization)

    np.testing.assert_array_equal(result_np, expected)
    np.testing.assert_array_equal(result_cv2, expected)
    np.testing.assert_array_equal(result, expected)

    if img.dtype == np.uint8:
        result_lut = normalize_per_image_lut(img, normalization)
        np.testing.assert_array_equal(result_lut, expected)


@pytest.mark.parametrize(
    ["image", "normalization"],
    [
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), "image"],
        [np.random.randint(0, 256, [101, 99, 3], dtype=np.uint8), "image_per_channel"],
        [np.random.randint(0, 256, [101, 99, 1], dtype=np.uint8), "min_max"],
        [np.random.randint(0, 256, [101, 99], dtype=np.uint8), "min_max_per_channel"],
    ],
)
def test_normalize_np_cv_equal(image, normalization):
    np.random.seed(0)
    res1 = normalize_per_image_numpy(image, normalization)
    res1_float = normalize_per_image_numpy(image.astype(np.float32), normalization)

    res2 = normalize_per_image_opencv(image, normalization)
    res2_float = normalize_per_image_opencv(image.astype(np.float32), normalization)

    res3 = normalize_per_image_lut(image, normalization)

    assert np.array_equal(image.shape, res1.shape)
    assert np.array_equal(image.shape, res2.shape)
    assert np.array_equal(image.shape, res3.shape)

    np.testing.assert_allclose(res1, res2, atol=1e-4)
    np.testing.assert_allclose(res1, res1_float, atol=1e-4)
    np.testing.assert_allclose(res2, res2_float, atol=1e-4)

    np.testing.assert_allclose(res1, res3, atol=1e-4)


# Parameterize tests for all combinations
@pytest.mark.parametrize("shape", [
    (100, 100),  # height, width
    (100, 100, 1),  # height, width, 1 channel
    (100, 100, 3),  # height, width, 3 channels
    (100, 100, 7),  # height, width, 7 channels
])
@pytest.mark.parametrize("normalization", [
    "image",
    "image_per_channel",
    "min_max",
    "min_max_per_channel",
])
@pytest.mark.parametrize("dtype", [
    np.uint8,
    np.float32,
])
def test_normalize_per_image(shape, normalization, dtype):
    # Generate a random image of the specified shape and dtype
    if dtype is np.uint8:
        img = np.random.randint(0, 256, size=shape, dtype=dtype)
    else:  # float32
        img = np.random.random(size=shape).astype(dtype) * 255

    # Normalize the image
    normalized_img = normalize_per_image(img, normalization)

    assert normalized_img.dtype == np.float32, "Output dtype should be float32"
    assert normalized_img.shape == img.shape, "Output shape should match input shape"

    # Assert the output shape matches the input shape
    assert normalized_img.shape == img.shape, "Output shape should match input shape"
    assert normalized_img.dtype == np.float32, "Output dtype should be float32"

    # Additional checks based on normalization type
    if normalization in ["min_max", "min_max_per_channel"]:
        # For min-max normalization, values should be in [0, 1]
        assert abs(normalized_img.min()) < 1e-6, "Min value should be >= 0"
        assert abs(normalized_img.max() - 1) < 1e-6, "Max value should be == 1"
    elif normalization in ["image", "image_per_channel"]:
        # For other normalizations, just ensure output dtype is float32
        # and check for expected normalization effects
        assert normalized_img.dtype == np.float32, "Output dtype should be float32"
        if normalization == "image":
            assert np.isclose(normalized_img.mean(), 0, atol=1e-3), "Mean should be close to 0 for 'image' normalization"
            assert np.isclose(normalized_img.std(), 1, atol=1e-3), "STD should be close to 1 for 'image' normalization"
        elif normalization == "image_per_channel":
            # Check channel-wise normalization for multi-channel images
            if len(shape) == 3 and shape[2] > 1:
                for c in range(shape[2]):
                    channel_mean = normalized_img[:, :, c].mean()
                    channel_std = normalized_img[:, :, c].std()
                    assert np.isclose(channel_mean, 0, atol=1e-3), f"Mean for channel {c} should be close to 0"
                    assert np.isclose(channel_std, 1, atol=1e-3), f"STD for channel {c} should be close to 1"



# Check that for constant array min max and min max per channel give 0
@pytest.mark.parametrize("shape", [
    (100, 100),  # height, width
    (100, 100, 1),  # height, width, 1 channel
    (100, 100, 3),  # height, width, 3 channels
    (100, 100, 7),  # height, width, 7 channels
])
@pytest.mark.parametrize("normalization", [
    "min_max",
    "min_max_per_channel",
])
@pytest.mark.parametrize("dtype", [
    np.uint8,
    np.float32,
])
def test_normalize_per_image(shape, normalization, dtype):
    img = np.ones(shape).astype(dtype)

    # Normalize the image
    normalized_img = normalize_per_image(img, normalization)
    np.testing.assert_array_equal(normalized_img, np.zeros_like(normalized_img))


@pytest.mark.parametrize("normalization", ["image", "image_per_channel", "min_max", "min_max_per_channel"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_normalize_per_image_batch(normalization, dtype):
    """Simple test for normalize_per_image_batch with image, volume, batch of images, and batch of volumes."""
    np.random.seed(42)

    shape = (100, 99, 3)
    height, width, num_channels = shape

    base_channel = np.random.rand(height, width)
    if dtype == np.uint8:
        base_channel = (base_channel * 255).astype(dtype)
    else:
        base_channel = base_channel.astype(dtype)

    image = np.stack([base_channel] * num_channels, axis=-1)

    volume = np.stack([image.copy()] * 4, axis=0)  # (4, H, W) or (4, H, W, C)
    images = np.stack([image.copy()] * 3, axis=0)  # (3, H, W) or (3, H, W, C)
    volumes = np.stack([volume.copy()] * 2, axis=0)  # (2, 4, H, W) or (2, 4, H, W, C)

    normalized_image = normalize_per_image_batch(image, normalization, spatial_axes=(0, 1))
    assert normalized_image.shape == image.shape
    assert normalized_image.dtype == np.float32

    normalized_images = normalize_per_image_batch(images, normalization, spatial_axes=(0, 1, 2))
    assert normalized_images.shape == images.shape
    assert normalized_images.dtype == np.float32

    normalized_volume = normalize_per_image_batch(volume, normalization, spatial_axes=(0, 1, 2))
    assert normalized_volume.shape == volume.shape
    assert normalized_volume.dtype == np.float32

    normalized_volumes = normalize_per_image_batch(volumes, normalization, spatial_axes=(0, 1, 2, 3))
    assert normalized_volumes.shape == volumes.shape
    assert normalized_volumes.dtype == np.float32

    np.testing.assert_allclose(normalized_image[0], normalized_image[1], atol=4, rtol=1e-5)
    np.testing.assert_allclose(normalized_images[0], normalized_images[1], atol=4, rtol=1e-5)
    np.testing.assert_allclose(normalized_volume[0], normalized_volume[1], atol=4, rtol=1e-5)
    np.testing.assert_allclose(normalized_volumes[0], normalized_volumes[1], atol=4, rtol=1e-5)
