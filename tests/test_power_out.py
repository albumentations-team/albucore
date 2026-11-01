import cv2
import numpy as np
import pytest
import torch

from albucore import power


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((7, 11, 3), "CHW", (2, 0, 1)),
        ((2, 7, 11, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 7, 11, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 7, 11, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("output_mode", ["allocate", "separate", "alias"])
@pytest.mark.parametrize("per_channel", [False, True])
def test_power_output_preserves_values_layout_and_source(dtype, shape, layout, permutation, container, output_mode, per_channel):
    values = np.linspace(1, 15, np.prod(shape), dtype=np.float32).reshape(shape)
    array = values.astype(dtype) if dtype == np.uint8 else (values / 16).astype(dtype)
    exponent = np.array([0.5, 1.5, 2.0], dtype=np.float32) if per_channel else 1.5
    expected = np.power(array.astype(np.float32), exponent)
    if dtype == np.uint8:
        expected = np.clip(expected, 0, 255).astype(dtype)
    else:
        expected = expected.astype(dtype)
    image = torch.from_numpy(array.copy()).permute(permutation).contiguous() if container == "tensor" else array.copy()
    original = image.clone() if container == "tensor" else image.copy()
    if output_mode == "allocate":
        destination = None
    elif output_mode == "alias":
        destination = image
    else:
        destination = torch.empty_like(image) if container == "tensor" else np.empty_like(image)

    result = power(image, exponent, out=destination, layout=layout if container == "tensor" else None)

    if destination is not None:
        assert result is destination
    actual = result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result
    assert actual.shape == shape
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-3 if dtype == np.float16 else 1e-5, atol=1e-7)
    if output_mode != "alias":
        np.testing.assert_array_equal(image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("negative_output", [False, True])
def test_power_strided_output_preserves_gaps_and_readonly_input(dtype, negative_output):
    image = np.arange(1, 67, dtype=np.float32).reshape(2, 11, 3)
    image = (image if dtype == np.uint8 else image / 128).astype(dtype)[:, ::-1]
    original = image.copy()
    image.flags.writeable = False
    storage = np.full((2, 22, 3), 137, dtype=dtype)
    output = storage[:, ::2]
    if negative_output:
        output = output[:, ::-1]
    expected = np.power(original.astype(np.float32), np.float32(1.5))
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)

    result = power(image, 1.5, out=output)

    assert result is output
    np.testing.assert_allclose(result, expected, rtol=1e-3 if dtype == np.float16 else 1e-5, atol=1e-7)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(storage[:, 1::2], np.full((2, 11, 3), 137, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("alias", [False, True])
def test_power_floating_field_exponent_with_explicit_output(dtype, alias):
    image = np.linspace(0.0625, 0.875, 19 * 23 * 3, dtype=np.float32).reshape(19, 23, 3).astype(dtype)
    exponents = np.linspace(0.5, 2, image.size, dtype=np.float32).reshape(image.shape)
    original = image.copy()
    expected = np.power(image.astype(np.float32), exponents).astype(dtype)
    output = image if alias else np.empty_like(image)

    result = power(image, exponents, out=output)

    assert result is output
    np.testing.assert_allclose(result, expected, rtol=1e-3 if dtype == np.float16 else 1e-5, atol=1e-7)
    if not alias:
        np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("alias", [False, True])
def test_power_wide_exponents_preserve_the_existing_cast_error_policy(alias):
    image = np.array([[[2.0, 0.5]]], dtype=np.float32)
    exponents = np.array([128.0, 1e300], dtype=np.float64)
    output = image if alias else np.empty_like(image)

    with np.errstate(over="raise", invalid="raise"):
        result = power(image, exponents, out=output)

    assert result is output
    np.testing.assert_array_equal(result, np.array([[[np.inf, 0.0]]], dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((7, 11, 3), "CHW", (2, 0, 1)),
        ((2, 7, 11, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 7, 11, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 7, 11, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
def test_power_tensor_field_uses_the_declared_layout_and_strided_destination(dtype, shape, layout, permutation):
    values = np.linspace(0.0625, 0.875, np.prod(shape), dtype=np.float32).reshape(shape).astype(dtype)
    controls = np.linspace(0.5, 2, values.size, dtype=np.float32).reshape(shape).astype(np.float16).astype(dtype)
    image = torch.from_numpy(values).permute(permutation)
    exponent = torch.from_numpy(controls).permute(permutation).contiguous()
    original = image.clone()
    backing = torch.full((*image.shape[:-1], image.shape[-1] * 2), 137, dtype=image.dtype)
    output = backing[..., ::2]
    expected = np.power(values.astype(np.float32), controls.astype(np.float32)).astype(dtype)

    result = power(image, exponent, layout=layout, out=output)

    assert result is output
    actual = result.permute(tuple(np.argsort(permutation))).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3 if dtype == np.float16 else 1e-5, atol=1e-7)
    np.testing.assert_array_equal(image.numpy(), original.numpy())
    torch.testing.assert_close(backing[..., 1::2], torch.full_like(backing[..., 1::2], 137), rtol=0, atol=0)


@pytest.mark.parametrize("exponent", [0.5, 1.5, 2.0])
@pytest.mark.parametrize("alias", [False, True])
def test_power_scalar_output_matches_float32_reference_for_every_half_encoding(exponent, alias):
    image = np.arange(1 << 16, dtype=np.uint16).view(np.float16).reshape(256, 256, 1)
    original = image.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected = cv2.pow(image.astype(np.float32), exponent).reshape(image.shape).astype(np.float16)
        output = image if alias else np.empty_like(image)
        result = power(image, exponent, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    if not alias:
        np.testing.assert_array_equal(image.view(np.uint16), original.view(np.uint16))


def test_power_contiguous_singleton_output_view_is_used_directly():
    image = np.linspace(0.0625, 0.875, 99, dtype=np.float32).reshape(9, 11, 1)
    backing = np.empty((1, 9, 11), dtype=np.float32)
    output = backing.transpose(1, 2, 0)
    assert output.flags.c_contiguous

    result = power(image, 1.5, out=output)

    assert result is output
    np.testing.assert_allclose(result, np.power(image, np.float32(1.5)), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("output_mode", ["contiguous", "gapped", "negative", "alias"])
def test_power_half_channel_output_matches_wide_reference_within_one_ulp(output_mode):
    image = np.linspace(0.0625, 0.875, 128 * 160 * 3, dtype=np.float32).reshape(128, 160, 3).astype(np.float16)
    original = image.copy()
    exponents = np.array([0.5, 1.5, 2.0], dtype=np.float16)
    expected = np.power(image.astype(np.float32), exponents.astype(np.float32)).astype(np.float16)
    storage = np.full((128, 320, 3), 137, dtype=np.float16)
    if output_mode == "alias":
        output = image
    elif output_mode == "contiguous":
        output = np.empty_like(image)
    else:
        output = storage[:, ::2]
    if output_mode == "negative":
        output = output[:, ::-1]

    result = power(image, exponents, out=output)

    assert result is output
    distances = np.abs(result.view(np.uint16).astype(np.int32) - expected.view(np.uint16).astype(np.int32))
    assert distances.max() <= 1
    if output_mode != "alias":
        np.testing.assert_array_equal(image, original)
    if output_mode in ("gapped", "negative"):
        np.testing.assert_array_equal(storage[:, 1::2], np.full((128, 160, 3), 137, dtype=np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_power_empty_output_returns_supplied_storage(dtype):
    image = np.empty((0, 11, 3), dtype=dtype)
    output = np.empty_like(image)

    result = power(image, 1.5, out=output)

    assert result is output
