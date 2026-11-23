import numpy as np
import pytest

from albucore import hflip, vflip


@pytest.mark.parametrize(("flip", "axis"), [(hflip, 1), (vflip, 0)], ids=["horizontal", "vertical"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 5, 600])
def test_flips_return_numpy_views(flip, axis, dtype, channels):
    img = np.arange(4 * 5 * channels, dtype=dtype).reshape(4, 5, channels)

    result = flip(img)

    np.testing.assert_array_equal(result, np.flip(img, axis=axis))
    assert np.shares_memory(result, img)
    assert not result.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("flip", [hflip, vflip], ids=["horizontal", "vertical"])
def test_public_flips_are_involutions(flip, dtype):
    img = np.arange(4 * 5 * 3, dtype=dtype).reshape(4, 5, 3)

    np.testing.assert_array_equal(flip(flip(img)), img)
