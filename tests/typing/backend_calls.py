"""Type relationships for internal NumPy backend implementations."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

from albucore.arithmetic import (
    _scale_numkong,
)
from albucore.elementwise import _numpy_unary
from albucore.normalize import _affine_numpy
from albucore.ops_misc import hflip_cv2, hflip_numpy, vflip_cv2, vflip_numpy
from albucore.weighted import _blend_numkong, _blend_numpy, _blend_opencv


def check_half(image: NDArray[np.float16]) -> None:
    assert_type(_affine_numpy(image, 0.5, 2.0), NDArray[np.float32])
    assert_type(_blend_numpy(image, 0.5, image, 0.5), NDArray[np.float32])
    assert_type(_blend_opencv(image, 0.5, image, 0.5), NDArray[np.float16])
    assert_type(_blend_numkong(image, 0.5, image, 0.5), NDArray[np.float16])
    assert_type(_scale_numkong(image, 1.5, 0.5), NDArray[np.float16])
    assert_type(_numpy_unary(image, np.exp), NDArray[np.float16])
    assert_type(_numpy_unary(image, np.log), NDArray[np.float16])
    assert_type(_numpy_unary(image, np.sqrt), NDArray[np.float16])
    assert_type(hflip_numpy(image), NDArray[np.float16])
    assert_type(hflip_cv2(image), NDArray[np.float16])
    assert_type(vflip_numpy(image), NDArray[np.float16])
    assert_type(vflip_cv2(image), NDArray[np.float16])
