"""Destination, source ownership, and precision for unary operations."""

import numpy as np
import pytest
import torch

import albucore as ac

OPERATIONS = [(ac.exp, np.exp), (ac.log, np.log), (ac.sqrt, np.sqrt)]


def assert_unary_values(actual, expected):
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    if expected.dtype == np.float16:
        difference = np.abs(
            actual[finite].view(np.uint16).astype(np.int32) - expected[finite].view(np.uint16).astype(np.int32)
        )
        assert np.all(difference <= 1)
    else:
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-5, atol=1e-7)
    zeros = finite & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("operation, reference", OPERATIONS)
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("shape", [(7, 11, 3), (64, 128, 1)])
@pytest.mark.parametrize("strided_output", [False, True])
def test_unary_supplied_output_preserves_source_and_destination_gaps(
    operation, reference, dtype, container, shape, strided_output
):
    values = [0, 1, 4, 16, 255] if dtype == np.uint8 else [-1, -0.0, 0, 0.125, 0.5, 1, 4]
    array = np.resize(np.asarray(values, dtype=dtype), shape)
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    with np.errstate(all="ignore"):
        expected = reference(array.astype(np.float32)).astype(output_dtype)
    if container == "tensor":
        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        target_shape = (*image.shape[:-1], image.shape[-1] * (2 if strided_output else 1))
        backing = torch.full(target_shape, 17, dtype=torch.float16 if dtype == np.float16 else torch.float32)
        destination = backing[..., ::2] if strided_output else backing
        original = image.clone()
    else:
        image = array
        target_shape = (*shape[:-1], shape[-1] * (2 if strided_output else 1))
        backing = np.full(target_shape, 17, dtype=output_dtype)
        destination = backing[..., ::2] if strided_output else backing
        original = image.copy()

    with np.errstate(all="ignore"):
        result = operation(image, out=destination)

    assert result is destination
    actual = result.permute(1, 2, 0).numpy() if container == "tensor" else result
    assert_unary_values(actual, expected)
    np.testing.assert_array_equal(
        image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original
    )
    if strided_output:
        gaps = backing[..., 1::2].numpy() if container == "tensor" else backing[..., 1::2]
        np.testing.assert_array_equal(gaps, np.full(gaps.shape, 17, dtype=output_dtype))


@pytest.mark.parametrize("operation, reference", OPERATIONS)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_unary_explicit_output_can_mutate_a_view(operation, reference, dtype, container):
    storage = np.linspace(0.125, 4, 64 * 256, dtype=dtype).reshape(64, 256, 1)
    original = storage.copy()
    image = torch.from_numpy(storage)[:, ::2] if container == "tensor" else storage[:, ::2]
    with np.errstate(all="ignore"):
        expected = reference(original[:, ::2].astype(np.float32)).astype(dtype)
        result = operation(image, out=image)

    assert result is image
    assert_unary_values(image.numpy() if container == "tensor" else image, expected)
    np.testing.assert_array_equal(storage[:, 1::2], original[:, 1::2])


@pytest.mark.parametrize("operation, reference", OPERATIONS)
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("alias", [False, True])
def test_unary_explicit_outputs_cover_every_binary16_encoding(operation, reference, container, alias):
    array = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(256, 256, 1)
    with np.errstate(all="ignore"):
        expected = reference(array.astype(np.float32)).astype(np.float16)
    image = torch.from_numpy(array.copy()) if container == "tensor" else array.copy()
    output = image if alias else torch.empty_like(image) if container == "tensor" else np.empty_like(image)

    with np.errstate(all="ignore"):
        result = operation(image, out=output)

    assert result is output
    assert_unary_values(result.numpy() if container == "tensor" else result, expected)


@pytest.mark.parametrize("operation, reference", OPERATIONS)
def test_unary_half_output_accepts_negative_strides(operation, reference):
    image = np.linspace(0.125, 4, 128 * 160, dtype=np.float16).reshape(128, 160, 1)
    backing = np.empty_like(image)
    output = backing[:, ::-1]
    expected = reference(image.astype(np.float32)).astype(np.float16)

    result = operation(image, out=output)

    assert result is output
    assert_unary_values(result, expected)


@pytest.mark.parametrize("operation", [ac.exp, ac.log, ac.sqrt])
def test_unary_tensor_output_can_feed_a_trainable_consumer(operation):
    image = torch.tensor([0.25, 0.5, 1.5], dtype=torch.float32).reshape(1, 3, 1)
    output = torch.empty_like(image)
    result = operation(image, out=output)
    weight = torch.ones_like(result, requires_grad=True)

    (weight * result).sum().backward()

    torch.testing.assert_close(weight.grad, result)


@pytest.mark.parametrize("operation, reference", OPERATIONS)
def test_unary_output_preserves_zero_stride_singleton_axis(operation, reference):
    backing = np.linspace(0.125, 1.5, 64 * 128, dtype=np.float32).reshape(64, 128)
    image = np.lib.stride_tricks.as_strided(backing, shape=(64, 128, 1), strides=(*backing.strides, 0))
    assert image.flags.c_contiguous
    expected = reference(image.copy())

    result = operation(image, out=image)

    assert result is image
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("operation, reference", OPERATIONS)
@pytest.mark.parametrize("shape", [(2, 17, 19, 3), (2, 3, 17, 19, 3)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("reuse_output", [False, True])
def test_unary_batch_and_volume_shapes_with_explicit_output(
    operation, reference, shape, dtype, container, reuse_output
):
    array = np.linspace(0.125, 1.5, int(np.prod(shape)), dtype=dtype).reshape(shape)
    original = array.copy()
    image = torch.from_numpy(array) if container == "tensor" else array
    expected = reference(original.astype(np.float32)).astype(dtype)

    result = operation(image, out=image if reuse_output else None)

    assert result.shape == shape
    assert_unary_values(result.numpy() if container == "tensor" else result, expected)
    if reuse_output:
        assert result is image
    else:
        np.testing.assert_array_equal(array, original)
