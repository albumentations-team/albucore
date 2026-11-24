import numpy as np
import pytest
import torch

from albucore import convert, convert_dtype
from albucore.utils import clip, convert_value, get_max_value

CHANNELS = [1, 3, 5]
DTYPES = [np.uint8, np.float16, np.float32]
BATCHES = [0, 2]


def generate_img(dtype: type[np.uint8 | np.float16 | np.float32], channels: int, batch: int) -> np.ndarray:
    shape = (*((batch,) if batch else ()), 13, 17, channels)
    rng = np.random.default_rng(137)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.random(shape, dtype=np.float32).astype(dtype)


@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("max_value", [None, 128.0, 17.3])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_convert_dtype_reference_and_input_preservation(source_dtype, target_dtype, channels, batch, max_value, container):
    source = generate_img(source_dtype, channels, batch)
    original = source.copy()
    scale = np.float32(255 if max_value is None else max_value)
    if source_dtype == target_dtype:
        expected = source
    elif source_dtype == np.uint8:
        expected = (source.astype(np.float32) / scale).astype(target_dtype)
    elif target_dtype == np.uint8:
        expected = np.clip(np.rint(source.astype(np.float32) * scale), 0, 255).astype(np.uint8)
    else:
        expected = source.astype(target_dtype)
    image = torch.from_numpy(source).movedim(-1, -3) if container == "tensor" else source

    result = convert_dtype(image, target_dtype, max_value=max_value)

    actual = result.movedim(-3, -1).numpy() if container == "tensor" else result
    if container == "tensor" and source_dtype == np.float16 and target_dtype == np.uint8:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)
    else:
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    assert actual.dtype == target_dtype
    assert actual.shape == source.shape
    if source_dtype == target_dtype:
        assert result is image


def test_float16_conversion_helpers_use_float_image_range() -> None:
    image = np.array([[[-1.0], [0.5], [2.0]]], dtype=np.float16)
    result = clip(image.copy(), np.dtype(np.float16))
    assert get_max_value(image.dtype) == 1.0
    assert result.dtype == np.float16
    np.testing.assert_array_equal(result, np.array([[[0.0], [0.5], [1.0]]], dtype=np.float16))
    assert convert_value(np.float16(0.3333), 3) == float(np.float16(0.3333))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("max_value", [255.0, 128.0, 17.3])
@pytest.mark.parametrize("storage", ["contiguous", "strided", "fortran", "negative"])
def test_uint8_full_domain_normalization(dtype, max_value, storage):
    values = np.arange(256, dtype=np.uint8).reshape(8, 32, 1)
    if storage == "strided":
        image = np.repeat(values, 2, axis=1)[:, ::2]
    elif storage == "fortran":
        image = np.asfortranarray(values)
    elif storage == "negative":
        image = values[:, ::-1]
    else:
        image = values
    image.flags.writeable = False
    expected = (image.astype(np.float32) / np.float32(max_value)).astype(dtype)

    result = convert_dtype(image, dtype, max_value=max_value)

    np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize("dtype", [np.float16, np.dtype(np.float16), torch.float16])
@pytest.mark.parametrize("max_value", [255.0, 128.0, 17.3])
def test_tensor_uint8_to_float16_accepts_numpy_and_torch_dtypes(dtype, max_value):
    image = torch.arange(256, dtype=torch.uint8).reshape(1, 8, 32)
    expected = (image.to(torch.float32) / max_value).to(torch.float16)

    result = convert_dtype(image, dtype, max_value=max_value)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("max_value", [255.0, 128.0, 17.3])
@pytest.mark.parametrize("destination", ["allocate", "contiguous", "strided"])
def test_linux_tensor_uint8_half_normalization_preserves_values_and_output(monkeypatch, max_value, destination):
    from albucore import convert

    monkeypatch.setattr(convert, "_LINUX_X86_64", True)
    image = torch.arange(256, dtype=torch.uint8).reshape(1, 8, 32)
    original = image.clone()
    backing = None
    output = None
    if destination == "contiguous":
        output = torch.empty(image.shape, dtype=torch.float16)
    elif destination == "strided":
        backing = torch.full((1, 8, 64), 41, dtype=torch.float16)
        output = backing[..., ::2]
    expected = (image.to(torch.float32) / max_value).to(torch.float16)

    result = convert_dtype(image, torch.float16, max_value=max_value, out=output)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    if backing is not None:
        torch.testing.assert_close(backing[..., 1::2], torch.full(image.shape, 41, dtype=torch.float16), rtol=0, atol=0)
    if output is not None:
        assert result is output


@pytest.mark.parametrize("shape", [(1, 257, 1), (128, 160, 3)])
def test_float16_to_uint8_matches_promoted_reference(shape) -> None:
    image = np.linspace(0, 1, num=np.prod(shape), dtype=np.float32).reshape(shape).astype(np.float16)
    expected = np.clip(np.rint(image.astype(np.float32) * np.float32(255)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(convert_dtype(image, np.dtype(np.uint8)), expected)


def test_large_float16_to_uint8_preserves_float32_rounding_edges() -> None:
    values = np.array(
        [-np.inf, -1, -0.0, 0, 0.5 / 17.3, 1.5 / 17.3, 127.5 / 17.3, 255 / 17.3, 1, np.inf, np.nan],
        dtype=np.float16,
    )
    image = np.resize(values, (128, 160, 3))
    result = convert_dtype(image, np.dtype(np.uint8), max_value=17.3)
    expected = convert_dtype(image.astype(np.float32), np.dtype(np.uint8), max_value=17.3)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize("max_value", [128.0, 255.0])
def test_convert_dtype_uint8_saturates_nonfinite_and_rounding_edges(layout, max_value):
    source = np.array(
        [[[-np.inf], [-0.5 / 255], [0.5 / 255], [1.5 / 255], [1.2], [np.inf], [np.nan]]], dtype=np.float32,
    )
    source = np.broadcast_to(source, (2, 7, 3)).copy()
    image = {
        "contiguous": source,
        "fortran": np.asfortranarray(source),
        "strided": np.repeat(source, 2, axis=1)[:, ::2],
    }[layout]
    with np.errstate(invalid="ignore"):
        expected = np.clip(np.rint(image * np.float32(max_value)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(convert_dtype(image, np.uint8, max_value=max_value), expected)


@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("batch", BATCHES)
def test_uint8_roundtrip(channels, batch):
    image = generate_img(np.uint8, channels, batch)
    np.testing.assert_array_equal(convert_dtype(convert_dtype(image, np.float32), np.uint8), image)


def _uint8_to_float_reference(image: np.ndarray, dtype: type[np.float16 | np.float32], divisor: float) -> np.ndarray:
    return (np.arange(256, dtype=np.float32) / divisor).astype(dtype)[image]


def _forbid_lut(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError

    monkeypatch.setattr(convert, "_cv2_lut_channel_last", fail)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("max_value", [None, 1.0, 100, 255.5, np.float64(0.1)])
@pytest.mark.parametrize("shape", [(40, 50, 1), (40, 50, 3), (30, 40, 9), (2, 20, 30, 3)])
@pytest.mark.parametrize("with_out", [False, True])
def test_macos_uint8_to_float_uses_torch_and_matches_lookup_table(monkeypatch, dtype, max_value, shape, with_out):
    monkeypatch.setattr(convert, "_MACOS_ARM64", True)
    monkeypatch.setattr(convert, "_LINUX_X86_64", False)
    monkeypatch.setattr(convert, "_MACOS_UINT8_TO_FLOAT_TORCH_MIN_ELEMENTS", 1)
    _forbid_lut(monkeypatch)
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    out = np.empty(shape, dtype=dtype) if with_out else None

    result = convert_dtype(image, dtype, max_value=max_value, out=out)

    assert (result is out) if with_out else result.flags.c_contiguous
    expected = _uint8_to_float_reference(image, dtype, 255 if max_value is None else max_value)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("layout", ["read_only", "strided", "strided_out"])
def test_macos_uint8_to_float_falls_back_to_lookup_table_for_unsupported_storage(monkeypatch, layout):
    monkeypatch.setattr(convert, "_MACOS_ARM64", True)
    monkeypatch.setattr(convert, "_LINUX_X86_64", False)
    monkeypatch.setattr(convert, "_MACOS_UINT8_TO_FLOAT_TORCH_MIN_ELEMENTS", 1)
    calls = []
    original = convert._cv2_lut_channel_last
    monkeypatch.setattr(convert, "_cv2_lut_channel_last", lambda *a, **k: calls.append(1) or original(*a, **k))
    image = np.random.default_rng(1).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    out = None
    if layout == "read_only":
        image.flags.writeable = False
    elif layout == "strided":
        image = image[:, ::2]
    else:
        out = np.empty((40, 60, 3), dtype=np.float32)[:, ::2]
        image = image[:, ::2]

    result = convert_dtype(image, np.float32, out=out)

    assert calls
    np.testing.assert_array_equal(result, _uint8_to_float_reference(image, np.float32, 255))


@pytest.mark.parametrize("max_value", [None, 1.0, 100, np.float64(0.1)])
@pytest.mark.parametrize("shape", [(40, 50, 1), (40, 50, 3), (2, 20, 30, 3)])
def test_linux_uint8_to_float32_uses_cast_and_divide_without_lookup_table(monkeypatch, max_value, shape):
    monkeypatch.setattr(convert, "_MACOS_ARM64", False)
    monkeypatch.setattr(convert, "_LINUX_X86_64", True)
    _forbid_lut(monkeypatch)
    image = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)

    result = convert_dtype(image, np.float32, max_value=max_value)

    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, _uint8_to_float_reference(image, np.float32, 255 if max_value is None else max_value))


@pytest.mark.parametrize("case", ["float16", "out", "above_maximum", "strided"])
def test_linux_uint8_to_float_keeps_lookup_table_outside_the_cast_divide_region(monkeypatch, case):
    monkeypatch.setattr(convert, "_MACOS_ARM64", False)
    monkeypatch.setattr(convert, "_LINUX_X86_64", True)
    calls = []
    original = convert._cv2_lut_channel_last
    monkeypatch.setattr(convert, "_cv2_lut_channel_last", lambda *a, **k: calls.append(1) or original(*a, **k))
    image = np.random.default_rng(3).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    dtype = np.float16 if case == "float16" else np.float32
    out = np.empty(image.shape, dtype=dtype) if case == "out" else None
    if case == "above_maximum":
        monkeypatch.setattr(convert, "_LINUX_UINT8_TO_FLOAT32_CAST_DIVIDE_MAX_ELEMENTS", 10)
    if case == "strided":
        image = image[:, ::2]

    result = convert_dtype(image, dtype, out=out)

    assert calls == [1]
    np.testing.assert_array_equal(result, _uint8_to_float_reference(image, dtype, 255))


@pytest.mark.parametrize("max_value", [17.3, 1000, 70000])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
def test_macos_uint8_to_float16_out_divides_with_full_scale_for_any_destination_strides(
    monkeypatch, max_value, layout
):
    monkeypatch.setattr(convert, "_MACOS_ARM64", True)
    monkeypatch.setattr(convert, "_LINUX_X86_64", False)
    monkeypatch.setattr(convert, "_MACOS_UINT8_TO_FLOAT_TORCH_MIN_ELEMENTS", 1)
    image = np.random.default_rng(4).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    if layout == "contiguous":
        out = np.empty(image.shape, dtype=np.float16)
    elif layout == "strided":
        out = np.empty((40, 120, 3), dtype=np.float16)[:, ::2]
    else:
        out = np.empty((60, 40, 3), dtype=np.float16).transpose(1, 0, 2)

    result = convert_dtype(image, np.float16, max_value=max_value, out=out)

    assert result is out
    np.testing.assert_array_equal(result, _uint8_to_float_reference(image, np.float16, max_value))
