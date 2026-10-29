import numpy as np
import numkong as nk
import pytest
import torch

from albucore import add_weighted, weighted
from albucore.utils import get_opencv_max_channels


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
@pytest.mark.parametrize(("weight1", "weight2"), [(0.25, 0.75), (1.5, -0.5)])
@pytest.mark.parametrize("output_mode", ["allocate", "first", "second", "output"])
def test_weighted_output_ownership(dtype, container, shape, layout, permutation, weight1, weight2, output_mode):
    indices = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    first = ((indices % 16) * 4).astype(dtype)
    second = (((indices + 5) % 16) * 4).astype(dtype)
    expected = first.astype(np.float32) * weight1 + second.astype(np.float32) * weight2
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)
    if container == "tensor":
        first = torch.from_numpy(first).permute(permutation).contiguous()
        second = torch.from_numpy(second).permute(permutation).contiguous()
    before1 = first.clone() if container == "tensor" else first.copy()
    before2 = second.clone() if container == "tensor" else second.copy()
    destination = {"allocate": None, "first": first, "second": second}.get(output_mode)
    if output_mode == "output":
        destination = torch.empty_like(first) if container == "tensor" else np.empty_like(first)

    result = add_weighted(
        first, weight1, second, weight2, out=destination, layout=layout if container == "tensor" else None
    )

    actual = result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result
    assert actual.dtype == dtype and actual.shape == shape
    np.testing.assert_array_equal(actual, expected)
    if destination is not None:
        assert result is destination
    if output_mode != "first":
        np.testing.assert_array_equal(
            first.numpy() if container == "tensor" else first, before1.numpy() if container == "tensor" else before1
        )
    if output_mode != "second":
        np.testing.assert_array_equal(
            second.numpy() if container == "tensor" else second, before2.numpy() if container == "tensor" else before2
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("output_mode", ["allocate", "alias", "output"])
def test_weighted_same_operand_can_alias_output(dtype, container, output_mode):
    image = np.arange(66, dtype=dtype).reshape(2, 11, 3)
    expected = image.astype(np.float32) * 3
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)
    if container == "tensor":
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    original = image.clone() if container == "tensor" else image.copy()
    destination = image if output_mode == "alias" else None
    if output_mode == "output":
        destination = torch.empty_like(image) if container == "tensor" else np.empty_like(image)

    result = add_weighted(image, 2.0, image, 1.0, out=destination, layout="CHW" if container == "tensor" else None)

    actual = result.permute(1, 2, 0).numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, expected)
    if destination is not None:
        assert result is destination
    if output_mode != "alias":
        np.testing.assert_array_equal(
            image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("has_output", [False, True])
def test_weighted_empty_batch_returns_supplied_storage(dtype, container, has_output):
    first = np.empty((0, 7, 11, 3), dtype=dtype)
    second = np.empty_like(first)
    if container == "tensor":
        first = torch.from_numpy(first).permute(0, 3, 1, 2)
        second = torch.from_numpy(second).permute(0, 3, 1, 2)
    output = (torch.empty_like(first) if container == "tensor" else np.empty_like(first)) if has_output else None

    result = add_weighted(first, 0.25, second, 0.75, out=output, layout="NCHW" if container == "tensor" else None)

    assert result is (output if has_output else first)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("negative_output", [False, True])
def test_weighted_readonly_sources_and_strided_output(dtype, negative_output):
    first = np.arange(66, dtype=dtype).reshape(2, 11, 3)[:, ::-1]
    second = np.full_like(first, 12)
    before1, before2 = first.copy(), second.copy()
    first.flags.writeable = second.flags.writeable = False
    storage = np.full((2, 22, 3), 137, dtype=dtype)
    destination = storage[:, ::2]
    if negative_output:
        destination = destination[:, ::-1]
    expected = first.astype(np.float32) * 2.0 + second.astype(np.float32) * -1.0
    expected = np.clip(expected, 0, 255).astype(dtype) if dtype == np.uint8 else expected.astype(dtype)

    result = add_weighted(first, 2.0, second, -1.0, out=destination)

    assert result is destination
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(first, before1)
    np.testing.assert_array_equal(second, before2)
    np.testing.assert_array_equal(storage[:, 1::2], 137)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("output_mode", ["allocate", "first", "second", "output"])
def test_weighted_zero_coefficient_preserves_nonfinite_arithmetic(dtype, channels, output_mode):
    first = np.repeat(np.array([1, np.inf, -np.inf, np.nan], dtype=dtype).reshape(1, 4, 1), channels, axis=-1)
    second = np.repeat(np.array([np.inf, 1, np.inf, 1], dtype=dtype).reshape(1, 4, 1), channels, axis=-1)
    original_first, original_second = first.copy(), second.copy()
    destination = {"allocate": None, "first": first, "second": second}.get(output_mode)
    if output_mode == "output":
        destination = np.empty_like(first)

    with np.errstate(invalid="ignore"):
        result = add_weighted(first, 1.0, second, 0.0, out=destination)

    if dtype == np.float32 and channels == 1:
        reference_first = original_first.copy()
        reference_second = original_second.copy()
        reference_destination = {
            "first": reference_first,
            "second": reference_second,
            "output": np.empty_like(reference_first),
        }.get(output_mode)
        with np.errstate(invalid="ignore"):
            if reference_destination is None:
                expected = np.asarray(
                    nk.blend(reference_first.reshape(-1), reference_second.reshape(-1), alpha=1.0, beta=0.0),
                    dtype=dtype,
                ).reshape(first.shape)
            else:
                nk.blend(
                    reference_first.reshape(-1),
                    reference_second.reshape(-1),
                    alpha=1.0,
                    beta=0.0,
                    out=reference_destination.reshape(-1),
                )
                expected = reference_destination
    else:
        expected = np.repeat(
            np.array([np.nan, np.inf, np.nan, np.nan], dtype=dtype).reshape(1, 4, 1), channels, axis=-1
        )
    np.testing.assert_array_equal(
        result,
        expected,
    )
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize("output_mode", ["first", "second", "output"])
def test_weighted_half_numkong_output_matches_all_encodings(monkeypatch, output_mode):
    monkeypatch.setattr(weighted, "_NUMKONG_NEON_HALF", True)
    first = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256, 1).view(np.float16)
    second = np.ascontiguousarray(np.roll(first, 137, axis=1))
    before1, before2 = first.copy(), second.copy()
    with np.errstate(invalid="ignore"):
        expected = (first.astype(np.float32) * 0.5 + second.astype(np.float32) * 0.5).astype(np.float16)
    destination = first if output_mode == "first" else second if output_mode == "second" else np.empty_like(first)
    if output_mode != "first":
        first.flags.writeable = False
    if output_mode != "second":
        second.flags.writeable = False

    result = add_weighted(first, 0.5, second, 0.5, out=destination)

    assert result is destination
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_max_ulp(result[finite], expected[finite], maxulp=1)
    if output_mode != "first":
        np.testing.assert_array_equal(first.view(np.uint16), before1.view(np.uint16))
    if output_mode != "second":
        np.testing.assert_array_equal(second.view(np.uint16), before2.view(np.uint16))


@pytest.mark.parametrize("output_mode", ["allocate", "first", "second", "strided"])
def test_weighted_half_wide_working_output_above_opencv_channel_limit(output_mode):
    shape = (2, 3, get_opencv_max_channels() + 1)
    indices = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) % 64
    first = (indices / 64).astype(np.float16)
    second = ((63 - indices) / 64).astype(np.float16)
    before1, before2 = first.copy(), second.copy()
    expected = (first.astype(np.float32) * 0.4 + second.astype(np.float32) * 0.6).astype(np.float16)
    destination = first if output_mode == "first" else second if output_mode == "second" else None
    if output_mode == "strided":
        storage = np.full((shape[0], shape[1] * 2, shape[2]), 137, dtype=np.float16)
        destination = storage[:, ::-2]

    result = add_weighted(first, 0.4, second, 0.6, out=destination)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float16
    if destination is not None:
        assert result is destination
    if output_mode != "first":
        np.testing.assert_array_equal(first, before1)
    if output_mode != "second":
        np.testing.assert_array_equal(second, before2)
    if output_mode == "strided":
        np.testing.assert_array_equal(storage[:, ::2], 137)


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("packed_inputs", [False, True])
@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((7, 11, 3), "CHW", (2, 0, 1)),
        ((2, 7, 11, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 7, 11, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 7, 11, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("force_half_average", [False, True])
def test_weighted_tensor_strided_output_preserves_gaps(
    dtype, packed_inputs, shape, layout, permutation, force_half_average, monkeypatch
):
    if force_half_average:
        monkeypatch.setattr(weighted, "_NUMKONG_NEON_HALF", True)
        monkeypatch.setattr(weighted, "_NUMKONG_HALF_BLEND_MIN_ELEMENTS", 1)
    weight1, weight2 = (0.5, 0.5) if force_half_average else (0.25, 0.75)
    values = (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) % 16) * 4
    first_array = values.astype(dtype)
    second_array = (60 - values).astype(dtype)
    expected = (first_array.astype(np.float32) * weight1 + second_array.astype(np.float32) * weight2).astype(dtype)
    first = torch.from_numpy(first_array).permute(permutation)
    second = torch.from_numpy(second_array).permute(permutation)
    if not packed_inputs:
        first, second = first.contiguous(), second.contiguous()
    before1, before2 = first.clone(), second.clone()
    storage = torch.full((*first.shape[:-1], first.shape[-1] * 2), 137, dtype=first.dtype)
    destination = storage[..., ::2]

    result = add_weighted(first, weight1, second, weight2, out=destination, layout=layout)

    assert result is destination
    np.testing.assert_array_equal(result.permute(tuple(np.argsort(permutation))).numpy(), expected)
    torch.testing.assert_close(first, before1, rtol=0, atol=0)
    torch.testing.assert_close(second, before2, rtol=0, atol=0)
    torch.testing.assert_close(storage[..., 1::2], torch.full_like(storage[..., 1::2], 137), rtol=0, atol=0)
