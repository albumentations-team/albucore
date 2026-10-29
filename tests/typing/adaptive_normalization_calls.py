"""Adaptive normalization input/result/output dtype relationships."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import normalize_per_image


def check_numpy(byte: NDArray[np.uint8], half: NDArray[np.float16], single: NDArray[np.float32]) -> None:
    assert_type(normalize_per_image(byte, "image"), NDArray[np.float32])
    assert_type(normalize_per_image(half, "min_max", out=half), NDArray[np.float16])
    assert_type(normalize_per_image(single, "image_per_channel", out=single), NDArray[np.float32])
    assert_type(normalize_per_image(byte, "min_max_per_channel", out=single), NDArray[np.float32])
    assert_type(normalize_per_image(img=half, normalization="image"), NDArray[np.float16])


def check_tensor(image: torch.Tensor) -> None:
    assert_type(normalize_per_image(image, "image", layout="CHW"), torch.Tensor)
    assert_type(normalize_per_image(image, "image_per_channel", out=image, layout="NCDHW"), torch.Tensor)


def check_union(image: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor) -> None:
    assert_type(
        normalize_per_image(image, "min_max", layout="CHW"), NDArray[np.float16] | NDArray[np.float32] | torch.Tensor
    )
