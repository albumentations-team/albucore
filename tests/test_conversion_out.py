"""Conversion values, aliasing, and supplied destination storage."""

import numpy as np
import pytest
import torch

from albucore import from_float, to_float

SHAPES = [(3, 5, 1), (2, 3, 5, 3), (2, 2, 3, 5, 9)]
OUTPUTS = [("numpy", mode) for mode in ("contiguous", "gapped", "fortran", "negative")] + [
    ("tensor", mode) for mode in ("contiguous", "gapped")
]


def destination(shape, dtype, container, mode):
    backing = np.full((*shape[:-1], shape[-1] * 2), 41, dtype=dtype)
    if mode == "fortran":
        output = np.empty(shape, dtype=dtype, order="F")
    elif mode == "contiguous":
        output = np.empty(shape, dtype=dtype)
    else:
        output = backing[..., ::2]
        if mode == "negative":
            output = output[..., ::-1]
    return (torch.from_numpy(output) if container == "tensor" else output), backing


@pytest.mark.parametrize("container,mode", OUTPUTS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_to_float_supplied_output(container, mode, shape, dtype):
    source = np.resize(np.array([0, 1, 127, 254, 255], dtype=dtype), shape)
    if dtype != np.uint8:
        source = source / np.array(128, dtype=dtype)
    original = source.copy()
    image = torch.from_numpy(source) if container == "tensor" else source
    if container == "numpy":
        source.flags.writeable = False
    target_dtype = np.float32 if dtype == np.uint8 else dtype
    output, backing = destination(shape, target_dtype, container, mode)
    expected = source.astype(np.float32) / np.float32(128) if dtype == np.uint8 else source

    result = to_float(image, max_value=128, out=output)

    assert result is output
    actual = result.numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == target_dtype
    np.testing.assert_array_equal(backing[..., 1::2], np.full(shape, 41, dtype=target_dtype))


@pytest.mark.parametrize("container,mode", OUTPUTS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("source_dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target_dtype", [np.uint8, np.float16, np.float32])
def test_from_float_supplied_output(container, mode, shape, source_dtype, target_dtype):
    values = np.array([-np.inf, -2, -0.0, 0.5 / 255, 1.5 / 255, 0.3333, 1, 2, np.inf, np.nan], dtype=source_dtype)
    source = np.resize(values, shape)
    original = source.copy()
    image = torch.from_numpy(source) if container == "tensor" else source
    if container == "numpy":
        source.flags.writeable = False
    output, backing = destination(shape, target_dtype, container, mode)
    with np.errstate(invalid="ignore"):
        expected = (
            np.clip(np.rint(source.astype(np.float32) * np.float32(255)), 0, 255).astype(np.uint8)
            if target_dtype == np.uint8
            else source.astype(target_dtype)
        )

    result = from_float(image, target_dtype, out=output)

    assert result is output
    actual = result.numpy() if container == "tensor" else result
    if container == "tensor" and source_dtype == np.float16 and target_dtype == np.uint8:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)
    else:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == target_dtype
    np.testing.assert_array_equal(backing[..., 1::2], np.full(shape, 41, dtype=target_dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("shape", SHAPES)
def test_float_identity_and_exact_alias(dtype, container, shape):
    source = np.resize(np.array([-2, -0.0, 0.3333, 2, np.inf, np.nan], dtype=dtype), shape)
    image = torch.from_numpy(source) if container == "tensor" else source
    original_bits = source.view(np.uint16 if dtype == np.float16 else np.uint32).copy()

    assert to_float(image) is image
    assert from_float(image, dtype) is image
    assert to_float(image, max_value=255, out=image) is image
    assert from_float(image, dtype, max_value=255, out=image) is image
    np.testing.assert_array_equal(source.view(original_bits.dtype), original_bits)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("scale", [128.0, 255.0, 17.3])
def test_half_quantization_all_encodings_with_output(container, scale):
    source = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(1, 65536, 1)
    image = torch.from_numpy(source) if container == "tensor" else source
    output = (
        torch.empty(source.shape, dtype=torch.uint8)
        if container == "tensor"
        else np.empty(source.shape, dtype=np.uint8)
    )
    with np.errstate(all="ignore"):
        expected = np.clip(np.rint(source.astype(np.float32) * np.float32(scale)), 0, 255).astype(np.uint8)
        result = from_float(image, np.uint8, max_value=scale, out=output)
    actual = result.numpy() if container == "tensor" else result
    assert result is output
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1 if container == "tensor" and scale != 17.3 else 0)
    np.testing.assert_array_equal(source.view(np.uint16).reshape(-1), np.arange(65536, dtype=np.uint16))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("mode", ["contiguous", "gapped", "fortran", "negative"])
def test_numpy_torch_quantization_writes_supplied_storage(monkeypatch, dtype, mode):
    from albucore import convert, torch_backend

    monkeypatch.setattr(torch_backend, "TORCH_CPU_BACKEND_ENABLED", True)
    monkeypatch.setattr(torch_backend, "TORCH_MIN_ELEMENTS", 1)
    monkeypatch.setattr(torch_backend, "TORCH_FLOAT16_TO_UINT8_MIN_ELEMENTS", 1)
    source = np.resize(np.array([-np.inf, -0.0, 0.5 / 255, 1.5 / 255, 1, np.inf, np.nan], dtype=dtype), (3, 5, 3))
    output, backing = destination(source.shape, np.uint8, "numpy", mode)
    with np.errstate(invalid="ignore"):
        expected = np.clip(np.rint(source.astype(np.float32) * np.float32(255)), 0, 255).astype(np.uint8)

    def fail_numkong(*args, **kwargs):
        raise AssertionError("The eligible Torch route must honor the supplied destination")

    monkeypatch.setattr(convert, "_from_float_uint8_numkong", fail_numkong)
    result = from_float(source, np.uint8, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(backing[..., 1::2], np.full(source.shape, 41, dtype=np.uint8))


@pytest.mark.parametrize("dtype,target", [(np.float16, np.float32), (np.float32, np.float16)])
@pytest.mark.parametrize("mode", ["contiguous", "gapped"])
def test_large_tensor_float_cast_supplied_output(dtype, target, mode):
    source = np.resize(np.array([-2, -0.0, 0.3333, 2, np.inf, np.nan], dtype=dtype), (3, 128, 160))
    image = torch.from_numpy(source)
    output, backing = destination(source.shape, target, "tensor", mode)
    expected = source.astype(target)

    result = from_float(image, target, out=output)

    assert result is output
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(backing[..., 1::2], np.full(source.shape, 41, dtype=target))
