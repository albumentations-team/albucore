import numpy as np
import pytest

from albucore import normalize_per_image

MODES = ("image", "image_per_channel", "min_max", "min_max_per_channel")


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", [1, 2])
def test_adaptive_small_values(dtype, mode, channels):
    image = np.arange(1, 4 * channels + 1, dtype=dtype).reshape(2, 2, channels)
    wide = image.astype(np.float64)
    axes = (0, 1) if mode.endswith("per_channel") else None
    if mode in ("image", "image_per_channel"):
        expected = (wide - wide.mean(axis=axes)) / (wide.std(axis=axes) + 1e-4)
    else:
        epsilon = 1e-4 if mode == "min_max_per_channel" and (dtype == np.uint8 or channels > 1) else 0
        expected = (wide - wide.min(axis=axes)) / (wide.max(axis=axes) - wide.min(axis=axes) + epsilon)

    result = normalize_per_image(image, mode)

    assert result.shape == image.shape
    assert result.dtype == (np.float16 if dtype == np.float16 else np.float32)
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(result, expected.astype(np.float16), maxulp=1)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["image_per_channel", "min_max_per_channel"])
@pytest.mark.parametrize("shape", [(17, 19, 9), (2, 5, 7, 9)])
def test_adaptive_uint8_many_channels(shape, mode):
    image = np.random.default_rng(137).integers(0, 256, size=shape, dtype=np.uint8)
    wide = image.astype(np.float64)
    axes = tuple(range(image.ndim - 1))
    if mode == "image_per_channel":
        expected = (wide - wide.mean(axis=axes)) / (wide.std(axis=axes) + 1e-4)
    else:
        expected = (wide - wide.min(axis=axes)) / (wide.max(axis=axes) - wide.min(axis=axes) + 1e-4)

    result = normalize_per_image(image, mode)

    assert result.shape == image.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", [(99, 101, 1), (99, 101, 3), (99, 101, 7)])
def test_adaptive_moments_range_and_source_preservation(dtype, mode, shape):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, size=shape, dtype=dtype)
        if dtype == np.uint8
        else (rng.random(shape, dtype=np.float32) * 255).astype(dtype)
    )
    original = image.copy()

    result = normalize_per_image(image, mode)

    assert result.shape == shape
    assert result.dtype == (np.float16 if dtype == np.float16 else np.float32)
    np.testing.assert_array_equal(image, original)
    wide_result = result.astype(np.float64)
    if mode.startswith("min_max"):
        assert abs(wide_result.min()) < 1e-6
        assert abs(wide_result.max() - 1) < 1e-6
    else:
        axes = tuple(range(image.ndim - 1)) if mode.endswith("per_channel") else None
        np.testing.assert_allclose(wide_result.mean(axis=axes), 0, atol=1e-3)
        np.testing.assert_allclose(wide_result.std(axis=axes), 1, atol=1e-3)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", [1, 3, 7])
def test_adaptive_constant_channels(dtype, mode, channels):
    image = np.full((99, 101, channels), 1 if dtype == np.uint8 else 0.5, dtype=dtype)

    result = normalize_per_image(image, mode)

    np.testing.assert_array_equal(result, np.zeros_like(image, dtype=np.float16 if dtype == np.float16 else np.float32))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("mode", MODES)
def test_adaptive_repeated_images_preserve_statistics(dtype, mode):
    rng = np.random.default_rng(137)
    image = (
        rng.integers(0, 256, size=(64, 80, 3), dtype=dtype)
        if dtype == np.uint8
        else rng.standard_normal((64, 80, 3)).astype(dtype)
    )
    batch = np.stack([image] * 4)

    single = normalize_per_image(image, mode)
    result = normalize_per_image(batch, mode)

    assert result.shape == batch.shape
    for item in result:
        if dtype == np.float16:
            np.testing.assert_array_max_ulp(item, single, maxulp=1)
        else:
            tolerance = 1e-3 if mode == "image_per_channel" else 1e-5
            np.testing.assert_allclose(item, single, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("output_mode", ["allocate", "output", "alias"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("normalization", ["image", "image_per_channel", "min_max", "min_max_per_channel"])
@pytest.mark.parametrize("shape", [(11, 13, 1), (11, 13, 3), (11, 13, 9), (2, 5, 11, 13, 3)])
@pytest.mark.parametrize("strided", [False, True])
def test_normalize_non_native_byte_order(output_mode, dtype, normalization, shape, strided):
    native = np.linspace(0.125, 0.875, num=np.prod(shape), dtype=dtype).reshape(shape)
    foreign = native.astype(np.dtype(dtype).newbyteorder("S"))
    if strided:
        native = native[..., ::2, :]
        foreign = foreign[..., ::2, :]
    if output_mode != "alias":
        foreign.setflags(write=False)
    source_bytes = foreign.tobytes()
    expected = normalize_per_image(native, normalization)
    output = foreign if output_mode == "alias" else np.empty_like(foreign) if output_mode == "output" else None

    result = normalize_per_image(foreign, normalization, out=output)

    assert result.shape == foreign.shape
    assert result.dtype == foreign.dtype
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, expected)
    if output is not None:
        assert result is output
    if output_mode != "alias":
        assert foreign.tobytes() == source_bytes
        assert not np.shares_memory(result, foreign)
