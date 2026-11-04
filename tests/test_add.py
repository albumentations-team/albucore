import numpy as np
import pytest

from albucore import add, arithmetic, convert_value


@pytest.mark.parametrize(
    "img, value, expected_output",
    [
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            2.0,
            np.array([[[3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14]]], dtype=np.uint8),
        ),
        (
            np.full((2, 2, 3), 254, dtype=np.uint8),
            2,
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ),
        (
            255 * np.ones((2, 2, 3), dtype=np.uint8),
            2 * np.ones((2, 2, 3), dtype=np.uint8),
            255 * np.ones((2, 2, 3), dtype=np.uint8),
        ),
        (
            254 * np.ones((2, 2, 3), dtype=np.uint8),
            np.array([2, 2, 2]),
            255 * np.ones((2, 2, 3), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 3), dtype=np.uint8),
            -1,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 3), dtype=np.uint8),
            np.array([-1, -2, -3]),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 3), dtype=np.uint8),
            -2 * np.ones((2, 2, 3), dtype=np.int8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        ((254 * np.ones((2, 2), dtype=np.uint8))[..., None], 2.0, (255 * np.ones((2, 2), dtype=np.uint8))[..., None]),
        (
            np.full((2, 2), 254, dtype=np.uint8)[..., None],
            (3 * np.ones((2, 2), dtype=np.uint8))[..., None],
            np.full((2, 2), 255, dtype=np.uint8)[..., None],
        ),
        (np.ones((2, 2), dtype=np.uint8)[..., None], -1, np.zeros((2, 2), dtype=np.uint8)[..., None]),
        (
            np.full((2, 2, 5), 254, dtype=np.uint8),
            2.0,
            np.full((2, 2, 5), 255, dtype=np.uint8),
        ),
        (
            np.full((2, 2, 5), 254, dtype=np.uint8),
            (3 * np.ones((2, 2, 5))).astype(np.uint8),
            np.full((2, 2, 5), 255, dtype=np.uint8),
        ),
        (
            np.full((2, 2, 5), 254, dtype=np.uint8),
            np.array([2.0] * 5),
            np.full((2, 2, 5), 255, dtype=np.uint8),
        ),
        (
            (np.ones((2, 2, 5))).astype(np.uint8),
            -1.0,
            np.zeros((2, 2, 5), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 5), dtype=np.uint8),
            np.array([-3.5] * 5),
            np.zeros((2, 2, 5), dtype=np.uint8),
        ),
        (
            np.ones((2, 2, 5), dtype=np.uint8),
            -2 * (np.ones((2, 2, 5), dtype=np.int16)),
            np.zeros((2, 2, 5), dtype=np.uint8),
        ),
        (
            np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8),
            -5.0,
            np.array([[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 115]]], dtype=np.uint8),
        ),
        (
            np.array([[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8),
            -5,
            np.array([[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 115]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[1, 3, 5], [4, 6, 8]], [[7, 9, 11], [10, 12, 14]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([0.5, -1.5, 2.0], dtype=np.float32),
            np.array([[[1, 1, 5], [4, 4, 8]], [[7, 7, 11], [10, 10, 14]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, 0.5, 0.25], [0.5, 1, 1.5]], [[1.5, 2, 0.5], [0.25, 0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 7]], [[8, 10, 9], [10, 11, 13]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[1, -0.5, 0.25], [-0.5, 1, -1.5]], [[-1.5, 2, 0.5], [0.25, -0.75, 1]]], dtype=np.float32),
            np.array([[[2, 2, 3], [4, 6, 5]], [[6, 10, 9], [10, 11, 13]]], dtype=np.uint8),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            0.2,
            np.array([[[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1], [1.2, 1.3, 1.4]]], dtype=np.float32),
        ),
        (
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], [[1.4, 1.6, 1.8], [2.0, 2.2, 2.4]]], dtype=np.float32),
            -0.2,
            np.array([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]], [[1.2, 1.4, 1.6], [1.8, 2.0, 2.2]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, 1.5, 2.0], dtype=np.float32),
            np.array([[[0.6, 1.7, 2.3], [0.9, 2.0, 2.6]], [[1.2, 2.3, 2.9], [1.5, 2.6, 3.2]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]], [[1.4, 1.6, 1.8], [2.0, 2.2, 2.4]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([0.5, -0.2, 0.1], dtype=np.float32),
            np.array([[[0.6, 0.0, 0.4], [0.9, 0.3, 0.7]], [[1.2, 0.6, 1.0], [1.5, 0.9, 1.3]]], dtype=np.float32),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], [[-0.7, 0.8, -0.9], [1.0, -1.1, 1.2]]], dtype=np.float32),
            np.array([[[0.2, 0.0, 0.6], [0.0, 1.0, 0.0]], [[0.0, 1.6, 0.0], [2.0, 0.0, 2.4]]], dtype=np.float32),
        ),
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            60,
            np.array([[[160, 210, 255], [255, 255, 160]], [[110, 135, 185], [235, 255, 255]]], dtype=np.uint8),
        ),
        (
            np.array([[[100, 150, 200], [250, 255, 100]], [[50, 75, 125], [175, 200, 225]]], dtype=np.uint8),
            -60,
            np.array([[[40, 90, 140], [190, 195, 40]], [[0, 15, 65], [115, 140, 165]]], dtype=np.uint8),
        ),
        (
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]], dtype=np.float32),
            1.0,
            np.array([[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], [[1.7, 1.8, 1.9], [2.0, 2.1, 2.2]]], dtype=np.float32),
        ),
    ],
)
def test_add_consistency(img, value, expected_output):
    original = img.copy()
    result = add(img, value)

    np.testing.assert_allclose(result, expected_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(img, original)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


@pytest.mark.parametrize("value", [0, 2, 255])
def test_add_uint8_single_channel_vector_preserves_output_contract(value: int) -> None:
    image = np.array([0, 1, 254, 255], dtype=np.uint8).reshape(1, 4, 1)
    original = image.copy()
    vector = np.array([value], dtype=np.uint8)
    expected = np.minimum(original.astype(np.uint16) + value, 255).astype(np.uint8)

    result = add(image, vector)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)

    destination = original.copy()
    returned = add(destination, vector, out=destination)

    assert returned is destination
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (64, 64, 1),
        (128, 160, 1),
        (128, 160, 3),
        (128, 160, 9),
        (1, 128, 160, 1),
        (2, 128, 80, 1),
        (1, 2, 128, 80, 1),
        (1, 128, 160, 3),
        (2, 64, 80, 3),
        (1, 2, 64, 80, 3),
    ],
)
@pytest.mark.parametrize("output_mode", ["allocate", "separate", "image", "field"])
def test_add_float16_contiguous_field_matches_wide_reference_and_output_aliases(
    shape: tuple[int, ...],
    output_mode: str,
) -> None:
    rng = np.random.default_rng(137)
    image = rng.random(shape, dtype=np.float32).astype(np.float16)
    value = rng.random(shape, dtype=np.float32).astype(np.float16)
    expected = (image.astype(np.float32) + value.astype(np.float32)).astype(np.float16)
    source = image.copy()
    control = value.copy()

    if output_mode == "allocate":
        out = None
    elif output_mode == "separate":
        out = np.empty_like(source)
    elif output_mode == "image":
        out = source
    else:
        out = control

    result = add(source, control, out=out)

    assert result.shape == shape
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, expected)
    if out is not None:
        assert result is out
    if output_mode != "image":
        np.testing.assert_array_equal(source, image)
    if output_mode != "field":
        np.testing.assert_array_equal(control, value)


@pytest.mark.parametrize("shape", [(128, 160, 1), (128, 160, 3)])
@pytest.mark.parametrize("value", [0.125, 1.5, 0.1])
@pytest.mark.parametrize("output_mode", ["allocate", "separate", "image"])
def test_add_float16_scalar_preserves_half_control_rounding_and_output_aliases(
    shape: tuple[int, int, int],
    value: float,
    output_mode: str,
) -> None:
    image = np.random.default_rng(137).uniform(0.1, 0.9, shape).astype(np.float16)
    half_control = np.float32(np.float16(value))
    expected = (image.astype(np.float32) + half_control).astype(np.float16)
    source = image.copy()
    out = None if output_mode == "allocate" else np.empty_like(source) if output_mode == "separate" else source

    result = add(source, value, out=out)

    assert result.shape == shape
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, expected)
    if out is not None:
        assert result is out
    if output_mode != "image":
        np.testing.assert_array_equal(source, image)


@pytest.mark.parametrize("output_mode", ["allocate", "alias", "separate"])
def test_add_float16_numkong_scalar_matches_all_non_nan_binary16_values(
    monkeypatch: pytest.MonkeyPatch,
    output_mode: str,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON", True)
    calls: list[tuple[float, float]] = []
    original_scale = arithmetic._scale_numkong

    def scale_spy(
        image: np.ndarray,
        factor: float,
        value: float,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        calls.append((factor, float(value)))
        return original_scale(image, factor, value, out=out)

    monkeypatch.setattr(arithmetic, "_scale_numkong", scale_spy)
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    non_nan_bits = bits[~((exponent == 0x7C00) & (mantissa != 0))]
    image = np.resize(non_nan_bits, (256, 256, 1)).view(np.float16)
    scalar = 0.875
    with np.errstate(over="ignore"):
        expected = (image.astype(np.float32) + np.float32(scalar)).astype(np.float16)

    original = image.view(np.uint16).copy()
    destination = image if output_mode == "alias" else np.empty_like(image) if output_mode == "separate" else None
    result = add(image, scalar, out=destination)

    if destination is not None:
        assert result is destination
    if output_mode != "alias":
        np.testing.assert_array_equal(image.view(np.uint16), original)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert calls == [(1.0, scalar)]
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))


@pytest.mark.parametrize("output_mode", ["allocate", "separate", "image", "field"])
def test_add_uint8_single_channel_full_field_preserves_output_aliases(output_mode: str) -> None:
    image = np.array([0, 1, 254, 255], dtype=np.uint8).reshape(1, 4, 1)
    value = np.full(image.shape, 2, dtype=np.uint8)
    expected = np.minimum(image.astype(np.uint16) + value.astype(np.uint16), 255).astype(np.uint8)
    source = image.copy()
    control = value.copy()

    if output_mode == "allocate":
        out = None
    elif output_mode == "separate":
        out = np.empty_like(source)
    elif output_mode == "image":
        out = source
    else:
        out = control

    result = add(source, control, out=out)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected)
    if out is not None:
        assert result is out
    if output_mode != "image":
        np.testing.assert_array_equal(source, image)
    if output_mode != "field":
        np.testing.assert_array_equal(control, value)


def test_add_float32_full_field_supports_exact_output_aliases() -> None:
    rng = np.random.default_rng(137)
    image = rng.random((8, 11, 1), dtype=np.float32)
    value = rng.random(image.shape, dtype=np.float32)
    expected = image + value

    image_output = image.copy()
    assert add(image_output, value, out=image_output) is image_output
    np.testing.assert_array_equal(image_output, expected)

    value_output = value.copy()
    assert add(image, value_output, out=value_output) is value_output
    np.testing.assert_array_equal(value_output, expected)


@pytest.mark.parametrize("img_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("num_channels", [1, 3, 5])
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        np.array([1.4]),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1], np.float32),
        np.array([2.0, 1.0, 0.5, 1.5, 1.1, 2.0], np.float32),
    ],
)
@pytest.mark.parametrize("is_contiguous", [True, False])
def test_add(img_dtype, num_channels, value, is_contiguous):
    np.random.seed(0)

    height, width = 9, 11

    if is_contiguous:
        if img_dtype == np.uint8:
            img = np.random.randint(0, 256, size=(height, width, num_channels), dtype=img_dtype)
        else:
            img = np.random.rand(height, width, num_channels).astype(img_dtype)
    elif img_dtype == np.uint8:
        img = np.random.randint(0, 256, size=(num_channels, height, width), dtype=img_dtype).transpose(1, 2, 0)
    else:
        img = np.random.rand(num_channels, height, width).astype(img_dtype).transpose(1, 2, 0)

    original_image = img.copy()

    processed_value = convert_value(value, num_channels)

    result = add(img, value)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    control = np.asarray(processed_value, dtype=np.float32)
    if img_dtype == np.uint8:
        control = np.trunc(control)
    expected = np.add(img.astype(np.float32), control)
    if img_dtype == np.uint8:
        expected = np.clip(expected, 0, 255).astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(img, original_image)
