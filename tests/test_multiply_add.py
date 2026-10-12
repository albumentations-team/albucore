import pytest
import numpy as np
from albucore import arithmetic
from albucore.functions import multiply_add, multiply_add_numpy, multiply_add_opencv, multiply_add_lut
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS, ImageType, clip


@pytest.mark.parametrize(
    "img, value, factor, expected_output",
    [
        # Test case 1: Both factor and value are 1, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            1.0,
            np.array([[2, 3], [4, 5]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            -1.0,
            2.0,
            np.array([[1, 3], [5, 7]], dtype=np.uint8),
        ),
        (np.array([[1, 2], [3, 4]], dtype=np.uint8), 1.0, 0.0, np.ones((2, 2))),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        # Test case 2: Different factor and value, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            2.0,
            0.5,
            np.array([[2, 3], [3, 4]], dtype=np.uint8),
        ),
        # Test case 3: Zero factor and value, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            0.0,
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
        ),
        # Test case 4: Factor 1 and value 0, image of type uint8
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        # Test case 5: Both factor and value are 1, image of type float32
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            1.0,
            np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        ),
        # Test case 6: Different factor and value, image of type float32
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            0.5,
            0.5,
            np.array([[0.55, 0.6], [0.65, 0.7]], dtype=np.float32),
        ),
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            -0.1,
            0.5,
            np.array([[0, 0], [0.05, 0.1]], dtype=np.float32),
        ),
    ],
)
def test_multiply_add_numpy(img, value, factor, expected_output):
    result_numpy = multiply_add_numpy(img, factor, value)
    if img.dtype == np.uint8:
        result_numpy = clip(result_numpy, img.dtype)
    else:
        expected_output = result_numpy
    np.testing.assert_allclose(result_numpy, expected_output, atol=1e-6)

    result_opencv = multiply_add_opencv(img, factor, value)
    if img.dtype == np.uint8:
        result_opencv = clip(result_opencv, img.dtype)
    np.testing.assert_allclose(result_opencv, expected_output, atol=1e-6)

    if img.dtype == np.uint8:
        result_lut = multiply_add_lut(img, factor, value, inplace=False)
        np.testing.assert_allclose(result_lut, expected_output, atol=1e-6)


@pytest.mark.parametrize("img_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize(
    "value, factor",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (-0.2, 0.5),
        (1.0, 1.0),
        (2.0, 0.5),
    ],
)
@pytest.mark.parametrize("is_contiguous", [True, False])
def test_multiply_add(img_dtype, num_channels, value, factor, is_contiguous):
    np.random.seed(0)
    height, width = 9, 11

    if is_contiguous:
        img = (
            np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype)
            if img_dtype == np.uint8
            else np.random.rand(height, width, num_channels).astype(img_dtype)
        )
    else:
        img = (
            np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0)
            if img_dtype == np.uint8
            else np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)
        )

    original_img = img.copy()

    result = multiply_add(img, factor, value)

    assert result.shape == original_img.shape
    assert result.dtype == original_img.dtype

    np.testing.assert_array_equal(img, original_img)

    result_numpy = multiply_add_numpy(img, factor, value)
    if img_dtype == np.uint8:
        result_numpy = clip(result_numpy, img_dtype)
    np.testing.assert_allclose(result, result_numpy, atol=1e-6)

    if img.dtype == np.uint8:
        result_lut = clip(multiply_add_lut(img, factor, value, inplace=False), img.dtype)
        np.testing.assert_array_equal(result, result_lut)

    result_opencv = multiply_add_opencv(img, factor, value)
    if img_dtype == np.uint8:
        result_opencv = clip(result_opencv, img.dtype)
    np.testing.assert_allclose(result, result_opencv, atol=1e-6)


def test_multiply_add_float16_numkong_scale_matches_all_non_nan_binary16_values(monkeypatch) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)
    calls = []
    original_numkong = arithmetic.multiply_add_numkong

    def record_numkong(img: ImageType, factor: float, value: float) -> ImageType:
        calls.append((factor, value))
        return original_numkong(img, factor, value)

    monkeypatch.setattr(arithmetic, "multiply_add_numkong", record_numkong)
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    non_nan_bits = bits[~((exponent == 0x7C00) & (mantissa != 0))]
    image = np.resize(non_nan_bits, (256, 256, 1)).view(np.float16)
    expected = (image.astype(np.float32) * np.float32(0.875) + np.float32(0.125)).astype(np.float16)

    result = multiply_add(image, np.float32(0.875), np.float32(0.125))

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert calls == [(0.875, 0.125)]
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))


@pytest.mark.parametrize(("factor", "bias"), [(0.4, 0.6), (1.127, 0.07)])
def test_multiply_add_float16_numkong_skips_non_half_controls(monkeypatch, factor: float, bias: float) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", True)

    def fail_numkong(*args: object, **kwargs: object) -> np.ndarray:
        msg = "NumKong half scale requires controls exactly representable in float16."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic, "multiply_add_numkong", fail_numkong)
    image = np.random.default_rng(710_011).uniform(-1, 1, (240, 320, 3)).astype(np.float16)
    expected = arithmetic.multiply_add_float16_torch(
        image,
        factor,
        bias,
        minimum_elements=arithmetic._TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS,
    )
    assert expected is not None

    result = multiply_add(image, factor, bias)

    np.testing.assert_array_equal(result, expected)
