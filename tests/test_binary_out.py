import numpy as np
import pytest
import torch

from albucore import add, multiply


@pytest.mark.parametrize(("operation", "reference"), [(add, np.add), (multiply, np.multiply)])
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
@pytest.mark.parametrize("control_kind", ["scalar", "channels", "field"])
def test_binary_output_preserves_values_layout_and_source(
    operation,
    reference,
    dtype,
    shape,
    layout,
    permutation,
    container,
    output_mode,
    control_kind,
):
    values = np.linspace(0, 255, np.prod(shape), dtype=np.float32).reshape(shape)
    array = values.astype(dtype) if dtype == np.uint8 else (values / 128 - 1).astype(dtype)
    if control_kind == "scalar":
        control = 1.5
    elif control_kind == "channels":
        control = np.array([0.5, -1.5, 2.0], dtype=np.float32)
    else:
        control = (
            np.linspace(-1.5, 2, array.size, dtype=np.float32).reshape(shape).astype(np.float16).astype(np.float32)
        )
    operand = np.trunc(control) if dtype == np.uint8 and operation is add else control
    expected = reference(array.astype(np.float32), operand)
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)
    image = torch.from_numpy(array.copy()).permute(permutation).contiguous() if container == "tensor" else array.copy()
    original = image.clone() if container == "tensor" else image.copy()
    if container == "tensor" and control_kind == "field":
        control = torch.from_numpy(control).permute(permutation).contiguous()
    if output_mode == "allocate":
        destination = None
    elif output_mode == "alias":
        destination = image
    else:
        destination = torch.empty_like(image) if container == "tensor" else np.empty_like(image)

    result = operation(image, control, out=destination, layout=layout if container == "tensor" else None)

    if destination is not None:
        assert result is destination
    actual = result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result
    assert actual.shape == shape
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)
    if output_mode != "alias":
        np.testing.assert_array_equal(
            image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original
        )


@pytest.mark.parametrize(("operation", "reference"), [(add, np.add), (multiply, np.multiply)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("negative_output", [False, True])
def test_binary_strided_output_preserves_gaps_and_readonly_input(operation, reference, dtype, negative_output):
    image = np.arange(66, dtype=np.float32).reshape(2, 11, 3)
    image = (image if dtype == np.uint8 else image / 32 - 1).astype(dtype)[:, ::-1]
    original = image.copy()
    image.flags.writeable = False
    storage = np.full((2, 22, 3), 137, dtype=dtype)
    output = storage[:, ::2]
    if negative_output:
        output = output[:, ::-1]
    control = np.array([0.5, -1.5, 2], dtype=np.float32)
    operand = np.trunc(control) if dtype == np.uint8 and operation is add else control
    expected = reference(original.astype(np.float32), operand)
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)

    result = operation(image, control, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(storage[:, 1::2], np.full((2, 11, 3), 137, dtype=dtype))


@pytest.mark.parametrize(("operation", "reference"), [(add, np.add), (multiply, np.multiply)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_binary_full_field_can_be_the_output(operation, reference, dtype, container):
    image = np.arange(66, dtype=dtype).reshape(2, 11, 3)
    field = np.full_like(image, 2)
    expected = reference(image.astype(np.float32), field.astype(np.float32))
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)
    if container == "tensor":
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        field = torch.from_numpy(field).permute(2, 0, 1).contiguous()
    original = image.clone() if container == "tensor" else image.copy()

    result = operation(image, field, out=field, layout="CHW" if container == "tensor" else None)

    assert result is field
    np.testing.assert_array_equal(result.permute(1, 2, 0).numpy() if container == "tensor" else result, expected)
    np.testing.assert_array_equal(
        image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original
    )


@pytest.mark.parametrize(("operation", "reference"), [(add, np.add), (multiply, np.multiply)])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("rank", [3, 4, 5])
def test_binary_half_channel_output_matches_every_half_encoding(operation, reference, channels, alias, rank):
    spatial = {3: (256, 256), 4: (2, 128, 256), 5: (2, 2, 128, 128)}[rank]
    encodings = np.arange(1 << 16, dtype=np.uint16).view(np.float16).reshape(*spatial, 1)
    image = np.broadcast_to(encodings, (*spatial, channels)).copy()
    original = image.view(np.uint16).copy()
    control = np.resize(np.array([0.5, -1.5, 2], dtype=np.float32), channels)
    output = image if alias else np.empty_like(image)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(image.astype(np.float32), control).astype(np.float16)

        result = operation(image, control, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    if not alias:
        np.testing.assert_array_equal(image.view(np.uint16), original)


@pytest.mark.parametrize(("operation", "reference"), [(add, np.add), (multiply, np.multiply)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((128, 160, 3), "CHW", (2, 0, 1)),
        ((2, 128, 160, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 128, 160, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 128, 160, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("control_kind", ["channels", "field"])
def test_binary_tensor_strided_output_preserves_gaps(operation, reference, dtype, shape, layout, permutation, control_kind):
    values = np.linspace(0, 255, np.prod(shape), dtype=np.float32).reshape(shape)
    array = values.astype(dtype) if dtype == np.uint8 else (values / 128 - 1).astype(dtype)
    control = np.array([0.5, -1.5, 2], dtype=np.float32)
    if control_kind == "field":
        control = np.broadcast_to(control, shape).astype(dtype)
    operand = np.trunc(control) if dtype == np.uint8 and operation is add else control
    expected = reference(array.astype(np.float32), operand)
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)
    image = torch.from_numpy(array).permute(permutation)
    original = image.clone()
    if control_kind == "field":
        control = torch.from_numpy(control).permute(permutation).contiguous()
    backing = torch.full((*image.shape[:-1], image.shape[-1] * 2), 137, dtype=image.dtype)
    output = backing[..., ::2]

    result = operation(image, control, out=output, layout=layout)

    assert result is output
    np.testing.assert_array_equal(result.permute(tuple(np.argsort(permutation))).numpy(), expected)
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    torch.testing.assert_close(backing[..., 1::2], torch.full_like(backing[..., 1::2], 137), rtol=0, atol=0)


@pytest.mark.parametrize("alias", [False, True])
def test_add_uint8_full_field_exhausts_saturating_sums(alias):
    image = np.broadcast_to(np.arange(256, dtype=np.uint8)[:, None, None], (256, 256, 1)).copy()
    field = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], image.shape).copy()
    expected = np.minimum(image.astype(np.uint16) + field.astype(np.uint16), 255).astype(np.uint8)
    output = image if alias else np.empty_like(image)

    result = add(image, field, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("operation", [add, multiply])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_binary_empty_output_returns_supplied_storage(operation, dtype):
    image = np.empty((0, 11, 3), dtype=dtype)
    output = np.empty_like(image)

    assert operation(image, 1.5, out=output) is output


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("negative_zero", [False, True])
def test_add_scalar_zero_preserves_input_bits_with_output(dtype, negative_zero):
    image = np.array([[[-0.0, 0.0, np.nan]]], dtype=dtype)
    output = np.empty_like(image)

    result = add(image, -0.0 if negative_zero else 0.0, out=output)

    assert result is output
    np.testing.assert_array_equal(
        result.view(np.uint16 if dtype == np.float16 else np.uint32),
        image.view(np.uint16 if dtype == np.float16 else np.uint32),
    )


@pytest.mark.parametrize("operation", [add, multiply])
@pytest.mark.parametrize("alias", [False, True])
def test_binary_wide_controls_preserve_the_existing_cast_error_policy(operation, alias):
    image = np.array([[[0.0, 0.5]]], dtype=np.float32)
    control = np.array([1e40, 1e40], dtype=np.float64)
    output = image if alias else np.empty_like(image)
    expected = (
        np.array([[[np.inf, np.inf]]], dtype=np.float32)
        if operation is add
        else np.array([[[0.0, np.inf]]], dtype=np.float32)
    )

    with np.errstate(over="raise", invalid="raise"):
        result = operation(image, control, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected)
