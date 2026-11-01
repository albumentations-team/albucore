"""Exact lookup values and explicit storage ownership across supported layouts."""

import numpy as np
import pytest
import torch

from albucore import apply_uint8_lut


def _table(channels, per_channel):
    values = np.arange(256, dtype=np.uint8)
    if not per_channel:
        return 255 - values
    return np.stack([np.roll(values, channel + 1) for channel in range(channels)], axis=-1)[:, None, :]


def _reference(image, table):
    if table.ndim == 1:
        return table[image]
    return np.stack([table[:, 0, channel][image[..., channel]] for channel in range(image.shape[-1])], axis=-1)


@pytest.mark.parametrize("shape", [(5, 7, 1), (5, 7, 3), (5, 7, 9), (2, 5, 7, 3), (2, 3, 5, 7, 3)])
@pytest.mark.parametrize("source_layout", ["contiguous", "strided", "fortran"])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("destination", ["allocate", "alias", "contiguous", "strided"])
def test_numpy_lut_output_storage(shape, source_layout, per_channel, destination):
    original = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    source_backing = None
    if source_layout == "strided":
        source_backing = np.full((*shape[:-2], shape[-2] * 2, shape[-1]), 173, dtype=np.uint8)
        image = source_backing[..., ::2, :]
        image[...] = original
    else:
        image = original.copy(order="F" if source_layout == "fortran" else "C")
    table = _table(shape[-1], per_channel)
    original_table = table.copy()
    expected = _reference(original, table)
    backing = None
    if destination == "allocate":
        out = None
        image.flags.writeable = False
    elif destination == "alias":
        out = image
    elif destination == "strided":
        backing = np.full((*shape[:-2], shape[-2] * 2, shape[-1]), 219, dtype=np.uint8)
        out = backing[..., ::-2, :]
    else:
        out = np.empty(shape, dtype=np.uint8)

    result = apply_uint8_lut(image, table, out=out)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(table, original_table)
    assert result.dtype == np.uint8
    if out is not None:
        assert result is out
    if destination != "alias":
        np.testing.assert_array_equal(image, original)
    if backing is not None:
        np.testing.assert_array_equal(backing[..., ::2, :], 219)
    if source_backing is not None:
        np.testing.assert_array_equal(source_backing[..., 1::2, :], 173)


@pytest.mark.parametrize("shape", [(640, 640, 2), (512, 512, 5)])
@pytest.mark.parametrize("alias", [False, True])
def test_large_shared_lut_writes_exact_output(shape, alias):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    expected = 255 - image
    out = image if alias else np.empty_like(image)

    result = apply_uint8_lut(image, _table(shape[-1], False), out=out)

    assert result is out
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("shape", "layout", "permutation"),
    [
        ((16, 16, 1), "CHW", (2, 0, 1)),
        ((16, 16, 3), "CHW", (2, 0, 1)),
        ((16, 16, 9), "CHW", (2, 0, 1)),
        ((2, 16, 16, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 16, 16, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 16, 16, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("destination", ["allocate", "alias", "contiguous", "strided"])
def test_tensor_lut_output_storage(shape, layout, permutation, contiguous, per_channel, destination):
    original = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    image = torch.from_numpy(original.copy()).permute(permutation)
    if contiguous:
        image = image.contiguous()
    before = image.clone()
    table = _table(shape[-1], per_channel)
    original_table = table.copy()
    expected = torch.from_numpy(_reference(original, table)).permute(permutation)
    backing = None
    if destination == "allocate":
        out = None
    elif destination == "alias":
        out = image
    elif destination == "strided":
        backing = torch.full((*image.shape[:-1], image.shape[-1] * 2), 219, dtype=torch.uint8)
        out = backing[..., ::2]
    else:
        out = torch.empty(image.shape, dtype=torch.uint8)

    result = apply_uint8_lut(image, table, out=out, layout=layout if per_channel else None)

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(table, original_table)
    assert result.shape == image.shape
    assert result.dtype == torch.uint8
    if out is not None:
        assert result is out
    if destination != "alias":
        torch.testing.assert_close(image, before, rtol=0, atol=0)
    if backing is not None:
        torch.testing.assert_close(backing[..., 1::2], torch.full_like(out, 219), rtol=0, atol=0)
