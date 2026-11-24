import numpy as np
import pytest
from numpy.typing import NDArray

from albucore import arithmetic, multiply_add


@pytest.mark.parametrize(
    "img, value, factor, expected_output",
    [
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
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            2.0,
            0.5,
            np.array([[2, 3], [3, 4]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            0.0,
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            1.0,
            np.array([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32),
        ),
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
            np.array([[-0.05, 0], [0.05, 0.1]], dtype=np.float32),
        ),
    ],
)
def test_multiply_add_literal_values(img, value, factor, expected_output):
    result = multiply_add(img[..., np.newaxis], factor, value)
    np.testing.assert_allclose(result, expected_output[..., np.newaxis], atol=1e-6)



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

    expected = img.astype(np.float32) * np.float32(factor) + np.float32(value)
    if img_dtype == np.uint8:
        expected = np.clip(expected, 0, 255).astype(np.uint8)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("output_mode", ["allocate", "alias", "output"])
def test_multiply_add_float16_numkong_scale_matches_all_non_nan_binary16_values(monkeypatch, output_mode) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON", True)
    calls = []
    original_numkong = arithmetic._scale_numkong

    def record_numkong(
        img: NDArray[np.float16],
        factor: float,
        value: float,
        *,
        out: NDArray[np.float16] | None = None,
    ) -> NDArray[np.float16]:
        calls.append((factor, value))
        return original_numkong(img, factor, value, out=out)

    monkeypatch.setattr(arithmetic, "_scale_numkong", record_numkong)
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    non_nan_bits = bits[~((exponent == 0x7C00) & (mantissa != 0))]
    image = np.resize(non_nan_bits, (256, 256, 1)).view(np.float16)
    expected = (image.astype(np.float32) * np.float32(0.875) + np.float32(0.125)).astype(np.float16)

    original = image.view(np.uint16).copy()
    destination = image if output_mode == "alias" else np.empty_like(image) if output_mode == "output" else None
    result = multiply_add(image, np.float32(0.875), np.float32(0.125), out=destination)
    if destination is not None:
        assert result is destination
    if output_mode != "alias":
        np.testing.assert_array_equal(image.view(np.uint16), original)

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
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON", True)

    def fail_numkong(*args: object, **kwargs: object) -> np.ndarray:
        msg = "NumKong half scale requires controls exactly representable in float16."
        raise AssertionError(msg)

    monkeypatch.setattr(arithmetic, "_scale_numkong", fail_numkong)
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


@pytest.mark.parametrize("threshold", [1_024, 4_096], ids=["linux", "macos"])
@pytest.mark.parametrize(("factor", "bias"), [(1.1, 0.3), (0.4, -0.6)])
@pytest.mark.parametrize("output", ["allocate", "separate", "strided"])
def test_multiply_add_float16_inexact_scalars_switch_to_torch_at_platform_threshold(
    monkeypatch: pytest.MonkeyPatch,
    threshold: int,
    factor: float,
    bias: float,
    output: str,
) -> None:
    rng = np.random.default_rng(16)
    numpy_calls: list[int] = []
    real_ufunc = arithmetic._float16_numpy_ufunc

    def spy(*args: object, **kwargs: object) -> np.ndarray | None:
        numpy_calls.append(1)
        return real_ufunc(*args, **kwargs)

    for elements, torch_route in ((threshold - 3, False), (threshold, True)):
        image = rng.random((1, elements, 1), dtype=np.float32).astype(np.float16)
        destination = None
        if output == "separate":
            destination = np.empty_like(image)
        elif output == "strided":
            destination = np.empty((1, elements, 2), dtype=np.float16)[..., ::2]
        monkeypatch.setattr(arithmetic, "_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS", 1 << 40)
        numpy_result = multiply_add(image, factor, bias)
        monkeypatch.setattr(arithmetic, "_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS", 1)
        torch_result = multiply_add(image, factor, bias)
        monkeypatch.setattr(arithmetic, "_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS", threshold)
        monkeypatch.setattr(arithmetic, "_float16_numpy_ufunc", spy)
        numpy_calls.clear()

        result = multiply_add(image, factor, bias, out=destination)

        expected = torch_result if torch_route else numpy_result
        np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16), strict=True)
        if destination is not None:
            assert result is destination
        assert bool(numpy_calls) != torch_route
        # For unit-range images the routes differ by at most one half ULP of the larger of product and result.
        product = np.abs(image.astype(np.float32) * np.float32(factor)).astype(np.float16)
        spacing = np.spacing(np.maximum(product, np.abs(numpy_result))).astype(np.float32)
        difference = np.abs(numpy_result.astype(np.float32) - torch_result.astype(np.float32))
        assert np.all(difference <= spacing)
