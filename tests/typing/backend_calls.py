"""Inferred dtypes when callers explicitly select a NumPy backend."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from albucore.functions import (
    add_numpy,
    add_opencv,
    add_weighted_numpy,
    add_weighted_opencv,
    exp_numpy,
    exp_opencv,
    from_float_numpy,
    from_float_opencv,
    hflip_cv2,
    hflip_numpy,
    log_numpy,
    multiply_add_numpy,
    multiply_numpy,
    multiply_opencv,
    normalize_numpy,
    normalize_opencv,
    normalize_per_image_numpy,
    normalize_per_image_opencv,
    power_numpy,
    power_opencv,
    sqrt_opencv,
    to_float_numpy,
    to_float_opencv,
    vflip_cv2,
    vflip_numpy,
)
from albucore.weighted import add_weighted_numkong, multiply_add_numkong


def check_half(image: NDArray[np.float16], floats: NDArray[np.float32]) -> None:
    assert_type(multiply_numpy(image, 1.5), NDArray[np.float16])
    assert_type(multiply_opencv(image, 1.5), NDArray[np.float16])
    assert_type(add_numpy(image, 0.5), NDArray[np.float16])
    assert_type(add_opencv(image, 0.5), NDArray[np.float16])
    assert_type(normalize_numpy(image, 0.5, 2.0), NDArray[np.float16])
    assert_type(normalize_opencv(image, 0.5, 2.0), NDArray[np.float16])
    assert_type(normalize_per_image_numpy(image, "image"), NDArray[np.float16])
    assert_type(normalize_per_image_opencv(image, "image"), NDArray[np.float16])
    assert_type(power_numpy(image, 2.0), NDArray[np.float16])
    assert_type(power_opencv(image, 2.0), NDArray[np.float16])
    assert_type(add_weighted_numpy(image, 0.5, image, 0.5), NDArray[np.float16])
    assert_type(add_weighted_opencv(image, 0.5, image, 0.5), NDArray[np.float16])
    assert_type(add_weighted_numkong(image, 0.5, image, 0.5), NDArray[np.float16])
    assert_type(multiply_add_numpy(image, 1.5, 0.5), NDArray[np.float16])
    assert_type(multiply_add_numkong(image, 1.5, 0.5), NDArray[np.float16])
    assert_type(to_float_numpy(image), NDArray[np.float16])
    assert_type(to_float_opencv(image), NDArray[np.float16])
    assert_type(from_float_numpy(floats, np.dtype(np.float16)), NDArray[np.float16])
    assert_type(from_float_opencv(floats, np.dtype(np.uint8)), NDArray[np.uint8])
    assert_type(exp_numpy(image), NDArray[np.float16])
    assert_type(exp_opencv(image), NDArray[np.float16])
    assert_type(log_numpy(image), NDArray[np.float16])
    assert_type(sqrt_opencv(image), NDArray[np.float16])
    assert_type(hflip_numpy(image), NDArray[np.float16])
    assert_type(hflip_cv2(image), NDArray[np.float16])
    assert_type(vflip_numpy(image), NDArray[np.float16])
    assert_type(vflip_cv2(image), NDArray[np.float16])


def check_uint8(image: NDArray[np.uint8]) -> None:
    assert_type(multiply_numpy(image, 1.5), NDArray[np.float32])
    assert_type(multiply_opencv(image, 1.5), NDArray[np.float32])
    assert_type(add_numpy(image, 0.5), NDArray[np.float32])
    assert_type(add_opencv(image, 0.5), NDArray[np.uint8] | NDArray[np.float32])
    assert_type(power_opencv(image, 2.0), NDArray[np.uint8] | NDArray[np.float32])
