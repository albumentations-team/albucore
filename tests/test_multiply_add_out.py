import numpy as np
import pytest
import torch

from albucore import arithmetic, multiply_add


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((7, 11, 3), "CHW", (2, 0, 1)),
        ((2, 7, 11, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 7, 11, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 7, 11, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("control_kind", ["scalar", "channels"])
@pytest.mark.parametrize("output_mode", ["allocate", "alias", "output"])
def test_multiply_add_output_contract(dtype, container, shape, layout, permutation, control_kind, output_mode):
    values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) % 32
    image_array = values.astype(dtype) if dtype == np.uint8 else ((values - 16) / 8).astype(dtype)
    factor = 1.5 if control_kind == "scalar" else np.array([1.5, -0.5, 2.0], dtype=np.float32)
    bias = 0.25 if control_kind == "scalar" else np.array([-0.25, 0.25, 1.0], dtype=np.float32)
    expected = image_array.astype(np.float32) * factor + bias
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)
    image = torch.from_numpy(image_array.copy()).permute(permutation).contiguous() if container == "tensor" else image_array.copy()
    original = image.clone() if container == "tensor" else image.copy()
    destination = image if output_mode == "alias" else (torch.empty_like(image) if container == "tensor" else np.empty_like(image)) if output_mode == "output" else None

    result = multiply_add(image, factor, bias, out=destination, layout=layout if container == "tensor" else None)

    actual = result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result
    assert actual.dtype == dtype and actual.shape == shape
    np.testing.assert_array_equal(actual, expected)
    if destination is not None:
        assert result is destination
    if output_mode != "alias":
        np.testing.assert_array_equal(image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("destination_name", ["image", "factor", "bias"])
@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((7, 11, 3), "CHW", (2, 0, 1)),
        ((2, 7, 11, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 7, 11, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 7, 11, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
def test_multiply_add_full_fields_can_alias_output(dtype, container, destination_name, shape, layout, permutation):
    indices = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    image = ((indices % 32 - 16) / 8).astype(dtype)
    factor = (indices % 3 / 2 + 0.5).astype(dtype)
    bias = (indices % 5 / 4 - 0.5).astype(dtype)
    expected = (image.astype(np.float32) * factor.astype(np.float32) + bias.astype(np.float32)).astype(dtype)
    arrays = {"image": image, "factor": factor, "bias": bias}
    if container == "tensor":
        arrays = {name: torch.from_numpy(value).permute(permutation).contiguous() for name, value in arrays.items()}
    originals = {name: value.clone() if container == "tensor" else value.copy() for name, value in arrays.items()}
    destination = arrays[destination_name]

    result = multiply_add(arrays["image"], arrays["factor"], arrays["bias"], out=destination, layout=layout if container == "tensor" else None)

    assert result is destination
    np.testing.assert_array_equal(result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result, expected)
    for name, value in arrays.items():
        if name != destination_name:
            np.testing.assert_array_equal(value.numpy() if container == "tensor" else value, originals[name].numpy() if container == "tensor" else originals[name])


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("negative_output", [False, True])
def test_multiply_add_readonly_source_and_strided_output(dtype, negative_output):
    image = np.arange(66, dtype=dtype).reshape(2, 11, 3)[:, ::-1]
    original = image.copy()
    image.flags.writeable = False
    storage = np.full((2, 22, 3), 137, dtype=dtype)
    destination = storage[:, ::2]
    if negative_output:
        destination = destination[:, ::-1]
    factor = np.array([1.5, -0.5, 2.0], dtype=np.float32)
    bias = np.array([-0.25, 0.25, 1.0], dtype=np.float32)
    expected = original.astype(np.float32) * factor + bias
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)

    result = multiply_add(image, factor, bias, out=destination)

    assert result is destination
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(storage[:, 1::2], 137)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_multiply_add_identity_preserves_float32_bits(container):
    array = np.array([-0.0, 0.0, -1.5, 2.0], dtype=np.float32).reshape(1, 4, 1)
    image = torch.from_numpy(array.copy()).permute(2, 0, 1) if container == "tensor" else array.copy()
    original = image.numpy().copy() if container == "tensor" else image.copy()

    result = multiply_add(image, 1.0, 0.0, layout="CHW" if container == "tensor" else None)

    assert result is image
    np.testing.assert_array_equal((result.numpy() if container == "tensor" else result).view(np.uint32), original.view(np.uint32))


@pytest.mark.parametrize("shape", [(256, 256, 1), (4, 256, 256, 1)])
@pytest.mark.parametrize("output_mode", ["allocate", "alias", "output"])
def test_multiply_add_half_two_rounding_for_every_encoding(monkeypatch, shape, output_mode):
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    image = np.resize(np.arange(1 << 16, dtype=np.uint16), shape).view(np.float16)
    original = image.view(np.uint16).copy()
    with np.errstate(all="ignore"):
        product = np.multiply(image.astype(np.float32), np.float32(0.875)).astype(np.float16)
        expected = np.add(product.astype(np.float32), np.float32(0.125)).astype(np.float16)
    destination = image if output_mode == "alias" else np.empty_like(image) if output_mode == "output" else None

    result = multiply_add(image, 0.875, 0.125, out=destination)

    if destination is not None:
        assert result is destination
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(result[finite].view(np.uint16), expected[finite].view(np.uint16))
    if output_mode != "alias":
        np.testing.assert_array_equal(image.view(np.uint16), original)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("output_mode", ["allocate", "alias", "output"])
def test_multiply_add_large_two_step_output(container, dtype, output_mode):
    values = ((np.arange(256 * 256 * 9, dtype=np.int32) % 32 - 16) / 8).astype(dtype).reshape(256, 256, 9)
    expected = (values.astype(np.float32) * np.float32(1.5) + np.float32(0.25)).astype(dtype)
    image = torch.from_numpy(values.copy()).permute(2, 0, 1).contiguous() if container == "tensor" else values.copy()
    original = image.clone() if container == "tensor" else image.copy()
    destination = image if output_mode == "alias" else (torch.empty_like(image) if container == "tensor" else np.empty_like(image)) if output_mode == "output" else None

    result = multiply_add(image, 1.5, 0.25, out=destination, layout="CHW" if container == "tensor" else None)

    np.testing.assert_array_equal(result.permute(1, 2, 0).numpy() if container == "tensor" else result, expected)
    if destination is not None:
        assert result is destination
    if output_mode != "alias":
        np.testing.assert_array_equal(image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("shared_control", ["factor", "bias", "both"])
def test_multiply_add_reads_input_control_alias_before_writing(dtype, container, shared_control):
    shape = (256, 256, 9)
    values = ((np.arange(np.prod(shape), dtype=np.int32) % 32 - 16) / 8).astype(dtype).reshape(shape)
    image = torch.from_numpy(values.copy()).permute(2, 0, 1).contiguous() if container == "tensor" else values.copy()
    factor = image if shared_control in ("factor", "both") else 1.5
    bias = image if shared_control in ("bias", "both") else 0.25
    expected = (values.astype(np.float32) * (values.astype(np.float32) if shared_control in ("factor", "both") else 1.5) + (values.astype(np.float32) if shared_control in ("bias", "both") else 0.25)).astype(dtype)

    result = multiply_add(image, factor, bias, out=image, layout="CHW" if container == "tensor" else None)

    assert result is image
    np.testing.assert_array_equal(result.permute(1, 2, 0).numpy() if container == "tensor" else result, expected)


def test_multiply_add_half_alias_preserves_source_for_wide_bias_fallback(monkeypatch):
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    image = np.array([0.5, 1, 1.5], dtype=np.float16).reshape(1, 3, 1)
    bias = np.full(image.shape, 1e40, dtype=np.float64)
    with np.errstate(over="ignore"):
        expected = (image.astype(np.float64) * 1.5 + bias).astype(np.float32).astype(np.float16)
        result = multiply_add(image, 1.5, bias, out=image)
    assert result is image
    np.testing.assert_array_equal(result, expected)


def test_multiply_add_half_alias_keeps_invalid_error_on_wide_bias_fallback(monkeypatch):
    monkeypatch.setattr(arithmetic, "_NUMKONG_NEON_HALF", False)
    image = np.array([0x7C01, 0x3800], dtype=np.uint16).view(np.float16).reshape(1, 2, 1)
    original = image.view(np.uint16).copy()
    with np.errstate(invalid="raise"), pytest.raises(FloatingPointError, match="invalid"):
        multiply_add(image, 0.5, 1e40, out=image)
    np.testing.assert_array_equal(image.view(np.uint16), original)


@pytest.mark.parametrize(
    ("dtype", "pixel", "factor", "bias", "predicate"),
    [
        (np.float32, 1.5, 3e38, -3e38, np.isposinf),
        (np.float16, 0.0, np.inf, 0.125, np.isnan),
    ],
)
def test_multiply_add_negative_output_preserves_selected_torch_math(dtype, pixel, factor, bias, predicate):
    image = np.full((256, 256, 9), pixel, dtype=dtype)
    storage = np.full((256, 512, 9), 137, dtype=dtype)
    destination = storage[:, ::2][:, ::-1]
    original = image.copy()

    with np.errstate(all="raise"):
        result = multiply_add(image, factor, bias, out=destination)

    assert result is destination
    assert predicate(result).all()
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(storage[:, 1::2], 137)


@pytest.mark.parametrize("output_mode", ["allocate", "alias", "output"])
@pytest.mark.parametrize("tensor_controls", [False, True])
@pytest.mark.parametrize(
    ("layout", "shape", "two_step"),
    [
        ("CHW", (3, 128, 160), False),
        ("CHW", (3, 512, 512), True),
        ("CDHW", (3, 4, 128, 160), False),
        ("CDHW", (3, 16, 128, 160), True),
        ("NCHW", (8, 3, 160, 160), False),
        ("NCDHW", (2, 3, 4, 160, 160), False),
    ],
)
def test_multiply_add_tensor_preserves_fused_and_per_image_rounding(layout, shape, two_step, tensor_controls, output_mode):
    image = torch.full(shape, 1.5, dtype=torch.float32)
    factor = torch.tensor(3e38) if tensor_controls else 3e38
    bias = torch.tensor(-3e38) if tensor_controls else -3e38
    destination = None if output_mode == "allocate" else image if output_mode == "alias" else torch.empty_like(image)

    with np.errstate(all="raise"):
        result = multiply_add(image, factor, bias, out=destination, layout=layout)

    torch.testing.assert_close(result, torch.full_like(result, torch.inf if two_step else 1.5e38), rtol=0, atol=0)
    if destination is not None:
        assert result is destination
    if output_mode != "alias":
        torch.testing.assert_close(image, torch.full_like(image, 1.5), rtol=0, atol=0)
