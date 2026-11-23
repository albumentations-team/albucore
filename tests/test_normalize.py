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


def _float32_centered_reference(image: np.ndarray, mean, denominator) -> np.ndarray:
    """Subtract first in float64 from the stored float32 values, then scale and round once."""
    mean64 = np.asarray(mean, dtype=np.float32).astype(np.float64)
    denominator64 = np.asarray(denominator, dtype=np.float32).astype(np.float64)
    return ((image.astype(np.float64) - mean64) * denominator64).astype(np.float32)


@pytest.mark.parametrize("width", [262143, 262144, 262145])
@pytest.mark.parametrize("vector_controls", [False, True])
@pytest.mark.parametrize("output_mode", ["allocate", "output", "alias"])
def test_normalize_float32_center_is_zero(width, vector_controls, output_mode):
    """Issue #247: a pixel equal to the mean must normalize to exact zero on every route."""
    image = np.full((1, width, 2), 0.7, dtype=np.float32)
    mean, denominator = float(image.flat[0]), 1_000_000.0
    if vector_controls:
        mean, denominator = np.full(2, mean, np.float32), np.full(2, denominator, np.float32)
    out = {"allocate": None, "output": np.empty_like(image), "alias": image}[output_mode]
    result = normalize(image, mean, denominator, out=out)
    if out is not None:
        assert result is out
    np.testing.assert_array_equal(result, 0)


@pytest.mark.parametrize("shape", [(1, 174762, 3), (1, 174763, 3), (4, 64, 64, 1), (2, 32, 32, 5), (17, 19, 4)])
def test_normalize_float32_adjacent_values_match_centered_reference(shape):
    mean = np.float32(0.7)
    neighbors = np.array([np.nextafter(mean, np.float32(-1)), mean, np.nextafter(mean, np.float32(2))], np.float32)
    image = np.resize(neighbors, int(np.prod(shape))).reshape(shape)
    result = normalize(image, float(mean), 1_000_000.0)
    np.testing.assert_allclose(result, _float32_centered_reference(image, mean, 1_000_000.0), rtol=4 * np.finfo(np.float32).eps, atol=0)
    np.testing.assert_array_equal(result[image == mean], 0)


@pytest.mark.parametrize("shape", [(99, 101, 3), (512, 512, 3), (8, 32, 32, 3), (64, 64, 1), (64, 64, 4), (64, 64, 9)])
@pytest.mark.parametrize("vector_controls", [False, True])
@pytest.mark.parametrize("output_mode", ["allocate", "output", "strided", "alias"])
def test_normalize_float32_matches_centered_float32(shape, vector_controls, output_mode):
    rng = np.random.default_rng(7)
    image = rng.random(shape, dtype=np.float32)
    channels = shape[-1]
    if vector_controls:
        mean = np.resize(np.array([0.485, 0.456, 0.406], np.float32), channels)
        denominator = np.resize(np.array([1 / 0.229, 1 / 0.224, 1 / 0.225], np.float32), channels)
    else:
        mean, denominator = 0.45, 4.4
    expected = (image - np.asarray(mean, np.float32)) * np.asarray(denominator, np.float32)
    original = image.copy()
    if output_mode == "strided":
        storage = np.full((*shape[:-1], channels * 2), 137, np.float32)
        out = storage[..., ::2]
    else:
        out = {"allocate": None, "output": np.empty_like(image), "alias": image}[output_mode]
    result = normalize(image, mean, denominator, out=out)
    np.testing.assert_array_equal(result, expected)
    if out is not None:
        assert result is out
    if output_mode == "strided":
        np.testing.assert_array_equal(storage[..., 1::2], 137)
    if output_mode != "alias":
        np.testing.assert_array_equal(image, original)
