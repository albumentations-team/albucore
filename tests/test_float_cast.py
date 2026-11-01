"""Exact storage casts, backend boundaries, and allocation contracts."""

import warnings
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import albucore._float_cast as cast_mod
import albucore.convert as convert_mod
from albucore import from_float


@pytest.fixture
def numkong_calls(monkeypatch):
    calls = []
    original = cast_mod.nk.astype

    def tracked(image, dtype, *, out):
        calls.append((image, out))
        return original(image, dtype, out=out)

    monkeypatch.setattr(cast_mod, "_NUMKONG_X86_CAST", True)
    monkeypatch.setattr(cast_mod.nk, "astype", tracked)
    return calls


@pytest.mark.parametrize("source_dtype,target_dtype", [(np.float16, np.float32), (np.float32, np.float16)])
@pytest.mark.parametrize("layout", ["C", "transpose", "F", "negative", "slice", "readonly"])
def test_float_cast_preserves_bits_ownership_and_source(source_dtype, target_dtype, layout, numkong_calls):
    source = np.linspace(-2, 2, 128 * 160 * 3, dtype=np.float32).reshape(128, 160, 3).astype(source_dtype)
    if layout == "transpose":
        source = source.swapaxes(0, 1)
    elif layout == "F":
        source = np.asfortranarray(source)
    elif layout == "negative":
        source = source[:, ::-1]
    elif layout == "slice":
        source = source[:, ::2]
    elif layout == "readonly":
        source.flags.writeable = False
    original = source.copy()
    expected = source.astype(target_dtype)

    actual = from_float(source, target_dtype, max_value=255)

    unsigned = np.uint16 if target_dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
    np.testing.assert_array_equal(source, original)
    assert actual.shape == source.shape
    assert actual.flags.owndata and actual.flags.writeable
    assert not np.shares_memory(actual, source)
    assert len(numkong_calls) == int(source.flags.c_contiguous)


@pytest.mark.parametrize("size", [16_383, 16_384])
def test_numkong_cast_threshold(size, numkong_calls):
    source = np.ones((1, size, 1), np.float16)

    actual = from_float(source, np.float32)

    np.testing.assert_array_equal(actual, source.astype(np.float32))
    assert len(numkong_calls) == int(size >= 16_384)


@pytest.mark.parametrize("source_dtype", [np.float16, np.float32])
@pytest.mark.parametrize("target_type", [np.float16, np.float32])
@pytest.mark.parametrize("size", [3, 20_000])
@pytest.mark.parametrize("supplied", [False, True])
def test_non_native_float_target_preserves_storage_and_values(
    source_dtype, target_type, size, supplied, numkong_calls
):
    target_dtype = np.dtype(target_type).newbyteorder("S")
    source = np.resize(np.array([-2, -0.0, 0.3333, 1.25], dtype=source_dtype), (1, size, 3))
    source.flags.writeable = False
    original = source.tobytes()
    expected = source.astype(target_dtype)
    backing = np.full((1, size, 6), 41, dtype=target_dtype)
    output = backing[..., ::2] if supplied else None

    result = from_float(source, target_dtype, out=output)

    assert result.dtype == target_dtype
    assert result.tobytes() == expected.tobytes()
    assert source.tobytes() == original
    if supplied:
        assert result is output
        np.testing.assert_array_equal(backing[..., 1::2], 41)
    assert not numkong_calls


def test_numkong_cast_all_finite_binary16_encodings(numkong_calls):
    values = np.arange(65_536, dtype=np.uint16).view(np.float16)
    source = values[np.isfinite(values)].reshape(-1, 1, 1)

    actual = from_float(source, np.float32)

    np.testing.assert_array_equal(actual.view(np.uint32), source.astype(np.float32).view(np.uint32))
    assert len(numkong_calls) == 1


def test_numkong_cast_binary16_rounding_boundaries(numkong_calls):
    encodings = np.arange(0x7BFF, dtype=np.uint16)
    low = encodings.view(np.float16).astype(np.float32)
    high = (encodings + 1).view(np.float16).astype(np.float32)
    midpoint = (low + high) * np.float32(0.5)
    positive = np.concatenate([midpoint, np.nextafter(midpoint, -np.inf), np.nextafter(midpoint, np.inf)])
    source = np.concatenate([positive, -positive]).reshape(-1, 1, 1)

    actual = from_float(source, np.float16)

    np.testing.assert_array_equal(actual.view(np.uint16), source.astype(np.float16).view(np.uint16))
    assert len(numkong_calls) == 1


@pytest.mark.parametrize("source_dtype,target_dtype", [(np.float16, np.float32), (np.float32, np.float16)])
def test_numkong_cast_keeps_numpy_special_values(source_dtype, target_dtype, numkong_calls):
    source = np.resize(np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], source_dtype), (128, 160, 1))

    actual = from_float(source, target_dtype)

    np.testing.assert_array_equal(actual, source.astype(target_dtype))
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(source.astype(target_dtype)))
    assert not numkong_calls


@pytest.mark.parametrize("kind", ["overflow", "underflow", "invalid"])
def test_float_cast_preserves_numpy_raising_error_policy(kind, numkong_calls):
    source = np.ones((128, 160, 1), np.float32)
    source.flat[0] = {"overflow": 65_520, "underflow": 1e-40, "invalid": 1}[kind]
    if kind == "invalid":
        source.view(np.uint32).flat[0] = 0x7F800001

    with np.errstate(all="raise"):
        try:
            expected = source.astype(np.float16)
        except FloatingPointError:
            with pytest.raises(FloatingPointError):
                from_float(source, np.float16)
        else:
            actual = from_float(source, np.float16)
            np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert not numkong_calls


def test_float_cast_preserves_numpy_overflow_warning(numkong_calls):
    source = np.full((128, 160, 1), 65_520, np.float32)

    with np.errstate(over="warn"), pytest.warns(RuntimeWarning, match="overflow"):
        actual = from_float(source, np.float16)

    assert np.isposinf(actual).all()
    assert not numkong_calls


@pytest.mark.parametrize("position", [0, -1])
def test_signaling_nan_warning_matches_numpy(position, numkong_calls):
    source = np.ones((128, 160, 1), np.float32)
    source.view(np.uint32).flat[position] = 0x7F800001
    with warnings.catch_warnings(record=True) as expected_warnings, np.errstate(invalid="warn"):
        warnings.simplefilter("always")
        expected = source.astype(np.float16)

    with warnings.catch_warnings(record=True) as actual_warnings, np.errstate(invalid="warn"):
        warnings.simplefilter("always")
        actual = from_float(source, np.float16)

    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings
    ]
    assert not numkong_calls


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_same_dtype_cast_returns_original_array(dtype, numkong_calls):
    source = np.ones((128, 160, 1), dtype)

    assert from_float(source, dtype) is source
    assert not numkong_calls


@pytest.mark.parametrize("shape", [(3, 128, 160), (4, 3, 128, 160), (1, 16, 128, 160), (2, 3, 16, 128, 160)])
@pytest.mark.parametrize("source_dtype,target_dtype", [(torch.float16, torch.float32), (torch.float32, torch.float16)])
def test_tensor_cast_matches_torch_and_has_independent_storage(shape, source_dtype, target_dtype, monkeypatch):
    monkeypatch.setattr(convert_mod, "_MACOS_ARM64", True)
    numpy_cast = Mock(wraps=convert_mod._cast_float_numpy)
    monkeypatch.setattr(convert_mod, "_cast_float_numpy", numpy_cast)
    source = torch.linspace(-2, 2, int(np.prod(shape)), dtype=source_dtype).reshape(shape)
    original = source.clone()

    actual = from_float(source, target_dtype)

    assert numpy_cast.call_count == 1
    np.testing.assert_array_equal(actual.numpy(), source.to(target_dtype).numpy())
    assert actual.dtype == target_dtype
    assert actual.shape == source.shape
    assert actual.is_contiguous()
    assert actual.data_ptr() != source.data_ptr()
    assert not torch.is_inference(actual)
    actual.zero_()
    np.testing.assert_array_equal(source.numpy(), original.numpy())


@pytest.mark.parametrize("size", [61_439, 61_440])
def test_tensor_numpy_cast_size_boundary(size, monkeypatch):
    monkeypatch.setattr(convert_mod, "_MACOS_ARM64", True)
    numpy_cast = Mock(wraps=convert_mod._cast_float_numpy)
    monkeypatch.setattr(convert_mod, "_cast_float_numpy", numpy_cast)
    source = torch.linspace(-1, 1, size, dtype=torch.float16).reshape(1, 1, size)

    actual = from_float(source, torch.float32)

    assert numpy_cast.call_count == int(size >= 61_440)
    np.testing.assert_array_equal(actual.numpy(), source.to(torch.float32).numpy())


@pytest.mark.skipif(not convert_mod._MACOS_ARM64, reason="NaN payload equality is specific to the macOS ARM route.")
def test_tensor_cast_keeps_torch_error_policy_and_nan_bits(monkeypatch):
    monkeypatch.setattr(convert_mod, "_MACOS_ARM64", True)
    half = np.arange(65_536, dtype=np.uint16).view(np.float16)
    with np.errstate(all="ignore"):
        wide = half.astype(np.float32)
    for array, target, unsigned in [(half, torch.float32, np.uint32), (wide, torch.float16, np.uint16)]:
        source = torch.from_numpy(array)
        expected = source.to(target).numpy().view(unsigned)
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="raise"):
            actual = from_float(source, target)
        assert not caught
        np.testing.assert_array_equal(actual.numpy().view(unsigned), expected)


def test_tensor_cast_keeps_identity_and_strided_torch_route(monkeypatch):
    monkeypatch.setattr(convert_mod, "_MACOS_ARM64", True)
    source = torch.ones((3, 128, 160), dtype=torch.float16).transpose(1, 2)

    def unexpected_numpy_route(*args):
        pytest.fail("The NumPy cast route must not handle identity or strided Tensor calls.")

    monkeypatch.setattr(convert_mod, "_cast_float_numpy", unexpected_numpy_route)
    assert from_float(source, torch.float16) is source
    np.testing.assert_array_equal(from_float(source, torch.float32).numpy(), source.to(torch.float32).numpy())


@pytest.mark.parametrize("source_dtype,target_dtype", [(np.float16, np.float32), (np.float32, np.float16)])
@pytest.mark.parametrize("layout", ["C", "singleton", "negative", "slice"])
def test_float_cast_writes_the_supplied_output(source_dtype, target_dtype, layout, numkong_calls):
    source = np.linspace(-2, 2, 128 * 160, dtype=source_dtype).reshape(128, 160, 1)
    original = source.copy()
    shape = (128, 320, 1) if layout == "slice" else source.shape
    backing = np.full(shape, 17, dtype=target_dtype)
    output = backing[:, ::2] if layout == "slice" else backing[:, ::-1] if layout == "negative" else backing
    if layout == "singleton":
        output = np.lib.stride_tricks.as_strided(backing, shape=source.shape, strides=(*backing.strides[:2], 0))
    expected = source.astype(target_dtype)

    result = cast_mod._cast_float_numpy(source, target_dtype, out=output)

    assert result is output
    unsigned = np.uint16 if target_dtype == np.float16 else np.uint32
    np.testing.assert_array_equal(result.view(unsigned), expected.view(unsigned))
    np.testing.assert_array_equal(source, original)
    assert len(numkong_calls) == int(output.flags.c_contiguous)
    if numkong_calls:
        assert numkong_calls[-1][1] is output
    if layout == "slice":
        np.testing.assert_array_equal(backing[:, 1::2], np.full((128, 160, 1), 17, dtype=target_dtype))


@pytest.mark.parametrize("kind", ["overflow", "underflow", "invalid"])
def test_float_cast_output_preserves_numpy_errors(kind, numkong_calls):
    source = np.ones((128, 160, 1), np.float32)
    source.flat[0] = {"overflow": 65_520, "underflow": 1e-40, "invalid": 1}[kind]
    if kind == "invalid":
        source.view(np.uint32).flat[0] = 0x7F800001
    output = np.empty(source.shape, dtype=np.float16)

    with np.errstate(all="raise"):
        try:
            expected = source.astype(np.float16)
        except FloatingPointError:
            with pytest.raises(FloatingPointError):
                cast_mod._cast_float_numpy(source, np.float16, out=output)
        else:
            actual = cast_mod._cast_float_numpy(source, np.float16, out=output)
            np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert not numkong_calls
