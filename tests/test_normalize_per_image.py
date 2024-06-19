import pytest
import numpy as np


from albucore.functions import (
    normalize_per_image_opencv,
    normalize_per_image_numpy,
    normalize_per_image_lut,
    normalize_per_image,
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

    assert np.allclose(result_np, expected, atol=1.5e-4), f"Result - expected {(result_np - expected).max()}"
    assert np.allclose(result_cv2, expected, atol=1.5e-4), f"Result - expected {(result_cv2 - expected).max()}"
    assert np.allclose(result, expected, atol=1.5e-4), f"Result - expected {(result - expected).max()}"

    if img.dtype == np.uint8:
        result_lut = normalize_per_image_lut(img, normalization)
        assert np.allclose(result_lut, expected, atol=1.5e-4), f"Result - expected {(result_lut - expected).max()}"


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

    assert np.allclose(res1, res2, atol=1e-4), f"mean: {(res1 - res2).mean()}, max: {(res1 - res2).max()}"
    assert np.allclose(res1, res1_float, atol=1e-4), f"mean: {(res1 - res1_float).mean()}, max: {(res1 - res1_float).max()}"
    assert np.allclose(res2, res2_float, atol=1e-4), f"mean: {(res2 - res2_float).mean()}, max: {(res2 - res2_float).max()}"

    assert np.allclose(res1, res3, atol=1e-4), f"mean: {(res1 - res3).mean()}, max: {(res1 - res3).max()}"


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
    assert np.array_equal(normalized_img, np.zeros_like(normalized_img))
