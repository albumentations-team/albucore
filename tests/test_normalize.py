import numpy as np
import pytest

from albucore import normalize


@pytest.mark.parametrize(
    ("image", "mean", "denominator", "expected"),
    [
        ([1, 2, 3, 4], 2.0, 1.0, [-1, 0, 1, 2]),
        ([0, 1, 2, 3], [2, 3, 4, 5], 1.0, [-2, -2, -2, -2]),
        ([1, 2, 3, 4], 2.0, [1, 0, 0, 1], [-1, 0, 0, 2]),
    ],
)
def test_normalize_float32_values(image, mean, denominator, expected):
    image = np.array(image, dtype=np.float32).reshape(2, 2, 1)
    mean = np.array(mean, dtype=np.float32).reshape(2, 2, 1) if isinstance(mean, list) else mean
    denominator = (
        np.array(denominator, dtype=np.float32).reshape(2, 2, 1) if isinstance(denominator, list) else denominator
    )

    result = normalize(image, mean, denominator)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array(expected, dtype=np.float32).reshape(image.shape))


@pytest.mark.parametrize(
    ("image", "mean", "denominator", "expected"),
    [
        (np.array([1, 2, 3, 4], dtype=np.uint8).reshape(2, 2, 1), 1.0, 2.0, [0, 2, 4, 6]),
        (np.arange(1, 9, dtype=np.uint8).reshape(2, 2, 2), 1.0, 2.0, [0, 2, 4, 6, 8, 10, 12, 14]),
        (np.arange(8, dtype=np.uint8).reshape(2, 2, 2), 1.0, np.array([2.0, 3.0]), [-2, 0, 2, 6, 6, 12, 10, 18]),
    ],
)
def test_normalize_uint8_values(image, mean, denominator, expected):
    result = normalize(image, mean, denominator)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array(expected, dtype=np.float32).reshape(image.shape))


@pytest.mark.parametrize("shape", [(17, 19, 9), (2, 5, 7, 9), (4, 6, 1)])
def test_normalize_uint8_channel_controls(shape):
    image = np.random.default_rng(137).integers(0, 256, size=shape, dtype=np.uint8)
    mean = np.linspace(70.0, 130.0, num=shape[-1], dtype=np.float32)
    denominator = np.linspace(0.01, 0.03, num=shape[-1], dtype=np.float32)

    result = normalize(image, mean, denominator)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, (image.astype(np.float32) - mean) * denominator, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("channels", "mean", "std"),
    [(3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), (3, 0.5, 0.7), (1, 0.5, 0.7), (1, 0.5, 0.5)],
)
def test_normalize_uint8_scaled_controls(channels, mean, std):
    image = np.random.default_rng(137).integers(0, 256, size=(101, 99, channels), dtype=np.uint8)
    mean = np.array(mean, dtype=np.float32)
    denominator = np.reciprocal(np.array(std, dtype=np.float64) * 255)

    result = normalize(image, mean, denominator)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    expected = (image.astype(np.float32) - mean) * denominator.astype(np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("shape", [(99, 101, 3), (99, 101, 1)])
def test_normalize_image_and_batch(dtype, shape):
    image = np.full(shape, 100 if dtype == np.uint8 else 0.4, dtype=dtype)
    batch = np.stack([image] * 3)
    mean = np.array(50, dtype=np.float32)
    denominator = np.array(1 / 3, dtype=np.float32)
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    expected = ((image.astype(np.float32) - mean) * denominator).astype(output_dtype)

    result = normalize(image, mean, denominator)
    batch_result = normalize(batch, mean, denominator)

    assert result.shape == image.shape
    assert batch_result.shape == batch.shape
    assert result.dtype == batch_result.dtype == output_dtype
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    np.testing.assert_allclose(batch_result, np.broadcast_to(expected, batch.shape), rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("shape", [(50, 50, 1), (3, 50, 50, 1), (50, 50, 3)])
def test_normalize_with_channel_arrays(dtype, shape):
    channels = shape[-1]
    image = np.broadcast_to(np.array([100, 150, 200][:channels], dtype=dtype), shape).copy()
    mean = np.array([50, 75, 100][:channels], dtype=np.float32)
    denominator = np.array([2, 3, 4][:channels], dtype=np.float32)
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    expected = ((image.astype(np.float32) - mean) * denominator).astype(output_dtype)

    result = normalize(image, mean, denominator)

    assert result.shape == image.shape
    assert result.dtype == output_dtype
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("shape", [(100, 100, 1), (100, 100, 3)])
def test_normalize_preserves_original_image(dtype, shape):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, size=shape, dtype=dtype) if dtype == np.uint8 else rng.standard_normal(shape).astype(dtype)
    )
    original = image.copy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) if shape[-1] == 3 else 0.5
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) if shape[-1] == 3 else 0.5
    denominator = np.reciprocal(std * (255 if dtype == np.uint8 else 1))

    result = normalize(image, mean, denominator)

    assert result.shape == image.shape
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("width", [8191, 8192, 8193])
@pytest.mark.parametrize("denominator", [5.0, 1000.0, 100000.0])
@pytest.mark.parametrize("vector_controls", [False, True])
def test_normalize_float16_center_is_zero(width, denominator, vector_controls):
    """Scalar controls must not be narrowed to float16 (issue #248)."""
    image = np.full((1, width, 2), 0.7, dtype=np.float16)
    mean = float(image.flat[0])
    if vector_controls:
        mean_control, denominator_control = np.full(2, mean, np.float32), np.full(2, denominator, np.float32)
    else:
        mean_control, denominator_control = mean, denominator
    result = normalize(image, mean_control, denominator_control)
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, 0)


def test_normalize_float16_adjacent_values():
    mean = np.float16(0.7)
    neighbors = np.array(
        [np.nextafter(mean, np.float16(-np.inf)), mean, np.nextafter(mean, np.float16(np.inf))],
        dtype=np.float16,
    )
    image = np.empty((1, 8192, 3), dtype=np.float16)
    image[...] = np.resize(neighbors, 8192)[:, None]
    result = normalize(image, float(mean), 1000.0)
    expected = ((neighbors.astype(np.float64) - float(mean)) * 1000).astype(np.float16)
    np.testing.assert_array_equal(result[0, :3, 0], expected)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 8191, 2), (1, 8192, 2), (2, 16, 16, 5)])
@pytest.mark.parametrize("with_out", [False, True])
def test_normalize_float16_scalar_preserves_infinity(shape, with_out):
    image = np.ones(shape, dtype=np.float16)
    image.flat[0], image.flat[1] = np.inf, -np.inf
    out = np.empty_like(image) if with_out else None
    result = normalize(image, 0.0, 2.0, out=out)
    assert result.flat[0] == np.inf
    assert result.flat[1] == -np.inf
    assert result.flat[2] == 2.0


@pytest.mark.parametrize("width", [8191, 8192])
@pytest.mark.parametrize("denominator", [1e7, 1e8])
def test_normalize_float16_center_is_zero_for_large_denominator(width, denominator):
    image = np.full((1, width, 2), 0.7, dtype=np.float16)
    np.testing.assert_array_equal(normalize(image, float(image.flat[0]), denominator), 0)
