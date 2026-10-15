import numpy as np
import pytest
import torch

from albucore import clipped, preserve_channel_dim
from albucore.functions import add_weighted_opencv


@pytest.mark.parametrize("call_form", ["positional", "callback_keyword", "img_keyword"])
def test_preserve_channel_dim_first_parameter_name(call_form):
    @preserve_channel_dim
    def grayscale(array):
        return array[..., 0]

    image = np.arange(15, dtype=np.float32).reshape(3, 5, 1)
    if call_form == "positional":
        result = grayscale(image)
    elif call_form == "callback_keyword":
        result = grayscale(array=image)
    else:
        result = grayscale(img=image)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, image)


def test_preserve_channel_dim_weighted_keywords():
    image = np.arange(15, dtype=np.float32).reshape(3, 5, 1)
    other = np.full_like(image, 2.0)
    result = add_weighted_opencv(img1=image, weight1=0.25, img2=other, weight2=0.75)
    assert result.shape == image.shape
    np.testing.assert_allclose(result, image * 0.25 + other * 0.75)


def test_preserve_channel_dim_tensor_result():
    @preserve_channel_dim
    def select_width(array):
        return array[..., 0]

    image = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
    result = select_width(image)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, image[..., 0])


def test_preserve_channel_dim_builtin_callback():
    negative = preserve_channel_dim(np.negative)
    image = np.arange(15, dtype=np.float32).reshape(3, 5, 1)
    np.testing.assert_array_equal(negative(img=image), -image)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("readonly", [False, True])
def test_clipped_preserves_borrowed_result(dtype, readonly):
    @clipped
    def select_other(img, other):
        return other

    image = np.full((2, 3, 1), 0.5, dtype=dtype)
    other = np.array([[-0.5, 0.0, 0.25], [0.75, 1.0, 1.5]], dtype=dtype)[..., None].copy()
    original = other.copy()
    other.flags.writeable = not readonly

    result = select_other(image, other)

    expected = np.array([[0.0, 0.0, 0.25], [0.75, 1.0, 1.0]], dtype=dtype)[..., None]
    assert result.dtype == dtype
    assert not np.shares_memory(result, other)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(other, original)
    np.testing.assert_array_equal(image, np.full_like(image, 0.5))
