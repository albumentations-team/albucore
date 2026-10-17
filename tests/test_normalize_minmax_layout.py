import numpy as np
import pytest
import torch

from albucore import normalize_per_image


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize(
    ("spatial", "layout", "permutation"),
    [
        ((5, 7), "CHW", (2, 0, 1)),
        ((2, 5, 7), "NCHW", (0, 3, 1, 2)),
        ((2, 5, 7), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 5, 7), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
def test_minmax_contiguous_tensor(dtype, channels, spatial, layout, permutation):
    array = (np.arange(np.prod((*spatial, channels))).reshape(*spatial, channels) % 137).astype(dtype)
    if dtype != np.uint8:
        array /= 137
    image = torch.from_numpy(array).permute(permutation).contiguous()
    original = image.clone()
    wide = array.astype(np.float64)
    if layout in ("NCHW", "NCDHW"):
        expected = np.stack([(item - item.min()) / (item.max() - item.min()) for item in wide])
    else:
        expected = (wide - wide.min()) / (wide.max() - wide.min())

    result = normalize_per_image(image, "min_max", layout=layout)

    actual = result.permute(tuple(np.argsort(permutation))).numpy()
    assert result.shape == image.shape
    assert actual.dtype == (np.float16 if dtype == np.float16 else np.float32)
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(actual, expected.astype(np.float16), maxulp=1)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(image.numpy(), original.numpy())


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("readonly", [False, True])
def test_minmax_fortran_numpy(dtype, readonly):
    image = np.asfortranarray(np.arange(105).reshape(5, 7, 3).astype(dtype))
    original = image.copy()
    image.flags.writeable = not readonly
    expected = (original.astype(np.float64) - original.min()) / (original.max() - original.min())

    result = normalize_per_image(image, "min_max")

    assert result.shape == image.shape
    assert result.dtype == (np.float16 if dtype == np.float16 else np.float32)
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(result, expected.astype(np.float16), maxulp=1)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_minmax_single_channel_fortran(dtype):
    image = np.asfortranarray(np.arange(35).reshape(5, 7, 1).astype(dtype))
    expected = image.astype(np.float64) / 34

    result = normalize_per_image(image, "min_max_per_channel")

    assert result.shape == image.shape
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(result, expected.astype(np.float16), maxulp=1)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
