import numpy as np
import pytest

from albucore.functions import hflip, hflip_cv2, hflip_numpy, vflip, vflip_cv2, vflip_numpy
from albucore.ops_misc import _flip_multichannel


@pytest.mark.parametrize(
    ("main", "numpy_backend", "cv2_backend", "axis"),
    [
        (hflip, hflip_numpy, hflip_cv2, 1),
        (vflip, vflip_numpy, vflip_cv2, 0),
    ],
    ids=["horizontal", "vertical"],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5, 600])
def test_flip_backends_preserve_values_and_public_views(main, numpy_backend, cv2_backend, axis, dtype, channels):
    img = np.arange(4 * 5 * channels, dtype=dtype).reshape(4, 5, channels)

    expected = np.flip(img, axis=axis)
    numpy_result = numpy_backend(img)
    cv2_result = cv2_backend(img)
    main_result = main(img)

    np.testing.assert_array_equal(numpy_result, expected)
    np.testing.assert_array_equal(cv2_result, expected)
    np.testing.assert_array_equal(main_result, expected)
    assert np.shares_memory(numpy_result, img)
    assert np.shares_memory(main_result, img)
    assert not numpy_result.flags["C_CONTIGUOUS"]
    assert not main_result.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("flip", [hflip, vflip], ids=["horizontal", "vertical"])
def test_public_flips_are_involutions(flip, dtype):
    img = np.arange(4 * 5 * 3, dtype=dtype).reshape(4, 5, 3)

    np.testing.assert_array_equal(flip(flip(img)), img)


@pytest.mark.parametrize("channels", [129, 513, 600, 1024])
@pytest.mark.parametrize("flip_code", [0, 1, -1])
def test_flip_multichannel_function(channels, flip_code):
    img = np.arange(8 * 10 * channels, dtype=np.float32).reshape(8, 10, channels)

    flipped = _flip_multichannel(img, flip_code)

    if flip_code == 0:
        expected = img[::-1, :, :]
    elif flip_code == 1:
        expected = img[:, ::-1, :]
    else:  # both
        expected = img[::-1, ::-1, :]

    np.testing.assert_array_equal(flipped, expected)
    assert flipped.dtype == img.dtype
