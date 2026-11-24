import numpy as np
import pytest

import albucore.utils as utils
from albucore import clip

_EVERYWHERE = {np.dtype(np.float16): (0, 2**31 - 1), np.dtype(np.float32): (0, 2**31 - 1)}


@pytest.fixture
def torch_route(monkeypatch):
    """Force the torch route and record its calls."""
    calls = []
    original = utils._clip_torch

    def spy(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(utils, "_CLIP_TORCH_WINDOWS", _EVERYWHERE)
    monkeypatch.setattr(utils, "_clip_torch", spy)
    return calls


def _values(dtype, shape):
    special = np.array([-np.inf, -2, -0.0, 0.0, 0.25, 1, 3, np.inf, np.nan], dtype=dtype)
    count = int(np.prod(shape))
    return np.resize(special, count).reshape(shape)


def _assert_same_values(actual, expected):
    # -0.0 == +0.0 and NaN payload bits are not part of the contract; NaN positions and all other values are.
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("kind", ["contiguous", "crop", "transposed", "step", "negative", "batch", "readonly"])
@pytest.mark.parametrize("use_out", [False, True])
def test_torch_route_matches_numpy_clip(torch_route, dtype, kind, use_out):
    base = _values(dtype, (6, 10, 3))
    source = {
        "contiguous": base,
        "crop": base[1:-1, 2:-2],
        "transposed": base.swapaxes(0, 1),
        "step": base[::2, ::2],
        "negative": base[::-1, ::-1],
        "batch": np.stack([base, base[::-1].copy()]),
        "readonly": base.copy(),
    }[kind]
    if kind == "readonly":
        source.flags.writeable = False
    original = source.copy()
    destination = np.full(source.shape, 7, dtype=dtype) if use_out else None

    with np.errstate(invalid="ignore"):
        expected = np.clip(source, 0, 1)
    result = clip(source, out=destination)

    assert len(torch_route) == 1
    _assert_same_values(result, expected)
    if use_out:
        assert result is destination
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_torch_route_writes_only_explicit_output_view(torch_route, dtype):
    source = _values(dtype, (4, 5, 1))
    with np.errstate(invalid="ignore"):
        expected = np.clip(source, 0, 1)
    backing = np.full((4, 5, 2), 137, dtype=dtype)
    destination = backing[..., ::-1][..., :1]  # negative-stride output keeps the NumPy route
    positive = backing[..., :1]

    clip(source, out=destination)
    assert not torch_route
    np.testing.assert_array_equal(backing[..., 1:], expected)
    np.testing.assert_array_equal(backing[..., :1], np.full((4, 5, 1), 137, dtype=dtype))

    result = clip(source, out=positive)
    assert result is positive
    assert len(torch_route) == 1
    np.testing.assert_array_equal(backing[..., :1], expected)
    np.testing.assert_array_equal(backing[..., 1:], expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_torch_route_alias_mutates_input_only_with_out(torch_route, dtype):
    source = _values(dtype, (3, 7, 3))
    original = source.copy()

    allocated = clip(source)
    np.testing.assert_array_equal(source, original)
    assert allocated is not source

    result = clip(source, out=source)
    assert result is source
    np.testing.assert_array_equal(source, allocated)
    assert len(torch_route) == 2


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_torch_route_keeps_infinities_and_nan_positions(torch_route, dtype):
    source = np.array([-np.inf, np.inf, np.nan, -0.0, 0.5, 2], dtype=dtype).reshape(1, 6, 1)

    result = clip(source)

    np.testing.assert_array_equal(np.isnan(result), np.isnan(source))
    np.testing.assert_array_equal(result[~np.isnan(result)], np.array([0, 1, 0, 0.5, 1], dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_torch_route_respects_window(monkeypatch, dtype):
    calls = []
    monkeypatch.setattr(utils, "_CLIP_TORCH_WINDOWS", {np.dtype(dtype): (10, 20)})
    monkeypatch.setattr(utils, "_clip_torch", lambda *args: calls.append(args) or args[0])

    for size in (9, 10, 20, 21):
        clip(np.zeros((1, size, 1), dtype=dtype))

    assert len(calls) == 2


def test_torch_route_skips_other_dtypes_and_conversions(monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "_CLIP_TORCH_WINDOWS", _EVERYWHERE)
    monkeypatch.setattr(utils, "_clip_torch", lambda *args: calls.append(args))

    clip(np.zeros((1, 4, 1), dtype=np.uint8))
    clip(np.zeros((1, 4, 1), dtype=np.float32), np.float16)
    clip(np.zeros((1, 4, 1), dtype=np.float32), out=np.empty((1, 4, 1), dtype=np.float16))

    assert not calls


def _misaligned(dtype, shape=(2, 3, 1)):
    itemsize = np.dtype(dtype).itemsize
    strides = (3 * itemsize + 3, itemsize + 1, itemsize)
    array = np.ndarray(shape, dtype=dtype, buffer=bytearray(strides[0] * shape[0] + itemsize), strides=strides)
    array[...] = np.array([[-1.5, 0.25, 2.0], [0.5, 3.0, -0.0]], dtype=dtype)[..., None]
    return array


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("kind", ["big_endian", "misaligned"])
@pytest.mark.parametrize("use_out", [False, "same_kind"])
def test_torch_route_keeps_numpy_for_storage_torch_cannot_wrap(torch_route, dtype, kind, use_out):
    native = _values(dtype, (2, 3, 1))
    source = native.astype(np.dtype(dtype).newbyteorder(">")) if kind == "big_endian" else _misaligned(dtype)
    expected = np.clip(np.asarray(source, dtype=dtype), 0, 1)
    destination = None
    if use_out:
        destination = (
            np.zeros(source.shape, dtype=np.dtype(dtype).newbyteorder(">")) if kind == "big_endian" else _misaligned(dtype)
        )

    result = clip(source, out=destination)

    assert not torch_route
    np.testing.assert_array_equal(np.asarray(result, dtype=dtype), expected)
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("kind", ["big_endian", "misaligned"])
def test_torch_route_keeps_numpy_for_unwrappable_output_only(torch_route, dtype, kind):
    source = _values(dtype, (2, 3, 1))
    destination = np.zeros(source.shape, dtype=np.dtype(dtype).newbyteorder(">")) if kind == "big_endian" else _misaligned(dtype)

    result = clip(source, out=destination)

    assert not torch_route
    assert result is destination
    np.testing.assert_array_equal(np.asarray(result, dtype=dtype), np.clip(source, 0, 1))
