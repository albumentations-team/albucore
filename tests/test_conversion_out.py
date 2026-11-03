"""Conversion values, aliasing, and supplied destination storage."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from albucore import convert_dtype

SHAPES = [(3, 5, 1), (2, 3, 5, 3), (2, 2, 3, 5, 9)]
OUTPUTS = [("numpy", mode) for mode in ("contiguous", "gapped", "fortran", "negative")] + [
    ("tensor", mode) for mode in ("contiguous", "gapped")
]


def _record_reused_numkong_working_buffer(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    from albucore import convert

    original = convert._quantize_uint8_numkong
    calls: list[bool] = []

    def record(
        img: NDArray[np.float32],
        max_value: float,
        *,
        out: NDArray[np.uint8] | None = None,
        input_is_working_buffer: bool = False,
    ) -> NDArray[np.uint8]:
        calls.append(input_is_working_buffer)
        return original(img, max_value, out=out, input_is_working_buffer=input_is_working_buffer)

    monkeypatch.setattr(convert, "_quantize_uint8_numkong", record)
    return calls


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
@pytest.mark.parametrize("source_dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("target_dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("max_value", [128.0, 17.3])
def test_convert_dtype_supplied_output(container, mode, shape, source_dtype, target_dtype, max_value):
    values = (
        np.array([0, 1, 127, 254, 255], dtype=np.uint8)
        if source_dtype == np.uint8
        else np.array([-np.inf, -2, -0.0, 0.5 / 255, 1.5 / 255, 0.3333, 1, 2, np.inf, np.nan], dtype=source_dtype)
    )
    source = np.resize(values, shape)
    original = source.copy()
    image = torch.from_numpy(source) if container == "tensor" else source
    if container == "numpy":
        source.flags.writeable = False
    output, backing = destination(shape, target_dtype, container, mode)
    with np.errstate(invalid="ignore"):
        if source_dtype == target_dtype:
            expected = source
        elif source_dtype == np.uint8:
            expected = (source.astype(np.float32) / np.float32(max_value)).astype(target_dtype)
        elif target_dtype == np.uint8:
            expected = np.clip(np.rint(source.astype(np.float32) * np.float32(max_value)), 0, 255).astype(np.uint8)
        else:
            expected = source.astype(target_dtype)

    result = convert_dtype(image, target_dtype, max_value=max_value, out=output)

    assert result is output
    actual = result.numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == target_dtype
    np.testing.assert_array_equal(backing[..., 1::2], np.full(shape, 41, dtype=target_dtype))


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("shape", SHAPES)
def test_same_dtype_identity_and_exact_alias(dtype, container, shape):
    values = [0, 1, 127, 254, 255] if dtype == np.uint8 else [-2, -0.0, 0.3333, 2, np.inf, np.nan]
    source = np.resize(np.array(values, dtype=dtype), shape)
    image = torch.from_numpy(source) if container == "tensor" else source
    original_bits = source.view(np.uint8).copy()

    assert convert_dtype(image, dtype) is image
    assert convert_dtype(image, dtype, max_value=255, out=image) is image
    np.testing.assert_array_equal(source.view(original_bits.dtype), original_bits)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("scale", [128.0, 255.0, 17.3])
@pytest.mark.parametrize("channels", [1, 3, 4, 5])
def test_half_quantization_all_encodings_with_output(monkeypatch, container, scale, channels):
    working_buffer_calls: list[bool] = []
    if container == "numpy":
        from albucore import torch_backend

        monkeypatch.setattr(torch_backend, "TORCH_FLOAT16_TO_UINT8_MIN_ELEMENTS", 1 << 31)
        working_buffer_calls = _record_reused_numkong_working_buffer(monkeypatch)

    values = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(1, 65536, 1)
    source = np.repeat(values, channels, axis=-1)
    original_bits = source.view(np.uint16).copy()
    image = torch.from_numpy(source) if container == "tensor" else source
    output = (
        torch.empty(source.shape, dtype=torch.uint8)
        if container == "tensor"
        else np.empty(source.shape, dtype=np.uint8)
    )
    with np.errstate(all="ignore"):
        expected = np.clip(np.rint(source.astype(np.float32) * np.float32(scale)), 0, 255).astype(np.uint8)
        result = convert_dtype(image, np.uint8, max_value=scale, out=output)
    actual = result.numpy() if container == "tensor" else result
    assert result is output
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1 if container == "tensor" and scale != 17.3 else 0)
    np.testing.assert_array_equal(source.view(np.uint16), original_bits)
    if container == "numpy":
        assert working_buffer_calls == [True]


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
def test_half_quantization_all_encodings_without_output(monkeypatch, channels):
    from albucore import torch_backend

    monkeypatch.setattr(torch_backend, "TORCH_FLOAT16_TO_UINT8_MIN_ELEMENTS", 1 << 31)
    working_buffer_calls = _record_reused_numkong_working_buffer(monkeypatch)

    values = np.arange(65536, dtype=np.uint16).view(np.float16).reshape(256, 256, 1)
    source = np.repeat(values, channels, axis=-1)
    original_bits = source.view(np.uint16).copy()
    with np.errstate(all="ignore"):
        expected = np.clip(np.rint(source.astype(np.float32) * np.float32(255)), 0, 255).astype(np.uint8)

    result = convert_dtype(source, np.uint8)

    assert result.dtype == np.uint8
    assert result.shape == source.shape
    assert not np.shares_memory(result, source)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source.view(np.uint16), original_bits)
    assert working_buffer_calls == [True]


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

    monkeypatch.setattr(convert, "_quantize_uint8_numkong", fail_numkong)
    result = convert_dtype(source, np.uint8, out=output)

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

    result = convert_dtype(image, target, out=output)

    assert result is output
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(backing[..., 1::2], np.full(source.shape, 41, dtype=target))
