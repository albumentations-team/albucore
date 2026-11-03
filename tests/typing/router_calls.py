"""Checked consumer calls for the public NumPy/Tensor type relationships."""

from typing import TypeAlias, assert_type

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from albucore import (
    add,
    add_weighted,
    apply_uint8_lut,
    clip,
    convert_dtype,
    copy_make_border,
    exp,
    hflip,
    log,
    median_blur,
    multiply,
    multiply_add,
    normalize,
    normalize_per_image,
    power,
    remap,
    resize,
    sqrt,
    vflip,
    warp_affine,
    warp_perspective,
)

ConsumerImage: TypeAlias = NDArray[np.uint8] | NDArray[np.float32]


def check_numpy(image: ConsumerImage, half: NDArray[np.float16], floats: NDArray[np.float32]) -> None:
    assert_type(add(image, 0.5), ConsumerImage)
    assert_type(multiply(image, 1.5), ConsumerImage)
    assert_type(multiply_add(image, 1.5, 0.25), ConsumerImage)
    assert_type(power(image, 2.0), ConsumerImage)
    assert_type(add_weighted(image, 0.5, image, 0.5), ConsumerImage)
    assert_type(hflip(image), ConsumerImage)
    assert_type(vflip(image), ConsumerImage)
    assert_type(median_blur(image, 3), ConsumerImage)
    assert_type(multiply(half, 1.5), NDArray[np.float16])
    assert_type(normalize(image, 0.5, 2.0), NDArray[np.float32])
    assert_type(normalize(half, 0.5, 2.0), NDArray[np.float16])
    assert_type(normalize_per_image(image, "image"), NDArray[np.float32])
    assert_type(normalize_per_image(half, "image"), NDArray[np.float16])
    assert_type(exp(image), NDArray[np.float32])
    assert_type(exp(half), NDArray[np.float16])
    assert_type(log(floats), NDArray[np.float32])
    assert_type(sqrt(floats), NDArray[np.float32])


def check_geometry(
    image: ConsumerImage, labels: NDArray[np.int16], matrix: NDArray[np.float32], grid: NDArray[np.float32],
) -> None:
    assert_type(resize(image, (11, 13)), ConsumerImage)
    assert_type(copy_make_border(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT), ConsumerImage)
    assert_type(warp_affine(image, matrix, (11, 13)), ConsumerImage)
    assert_type(warp_perspective(image, matrix, (11, 13)), ConsumerImage)
    assert_type(remap(labels, grid, grid, cv2.INTER_NEAREST), NDArray[np.int16])


def check_conversions(
    image: ConsumerImage,
    half: NDArray[np.float16],
    floats: NDArray[np.float32],
    dtype: np.dtype[np.uint8] | np.dtype[np.float32],
) -> None:
    assert_type(convert_dtype(image, np.float32), NDArray[np.float32])
    assert_type(convert_dtype(half, np.float16), NDArray[np.float16])
    assert_type(clip(image, dtype), ConsumerImage)
    assert_type(clip(floats, np.dtype(np.uint8)), NDArray[np.uint8])
    assert_type(convert_dtype(half, np.dtype(np.uint8)), NDArray[np.uint8])
    assert_type(convert_dtype(floats, np.dtype(np.float16)), NDArray[np.float16])


def check_luts(image: NDArray[np.uint8], lut: NDArray[np.uint8]) -> None:
    assert_type(apply_uint8_lut(image, lut), NDArray[np.uint8])


def check_tensor(image: torch.Tensor, matrix: NDArray[np.float32], lut: NDArray[np.uint8]) -> None:
    assert_type(multiply(image, 1.5, layout="CHW"), torch.Tensor)
    assert_type(normalize(image, 0.5, 2.0, layout="CHW"), torch.Tensor)
    assert_type(normalize_per_image(image, "image", layout="CHW"), torch.Tensor)
    assert_type(convert_dtype(image, np.float32), torch.Tensor)
    assert_type(convert_dtype(image, torch.uint8), torch.Tensor)
    assert_type(hflip(image), torch.Tensor)
    assert_type(median_blur(image, 3, layout="CHW"), torch.Tensor)
    assert_type(resize(image, (11, 13), layout="CHW"), torch.Tensor)
    assert_type(warp_affine(image, matrix, (11, 13), layout="CHW"), torch.Tensor)
    assert_type(apply_uint8_lut(image, lut, layout="CHW"), torch.Tensor)


def check_union_inputs(image: ConsumerImage | torch.Tensor, floats: NDArray[np.float32]) -> None:
    assert_type(multiply(image, 1.5, layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(add(image, 0.5, layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(convert_dtype(image, np.float32), NDArray[np.float32] | torch.Tensor)
    assert_type(normalize(image, 0.5, 2.0, layout="CHW"), NDArray[np.float32] | torch.Tensor)
    assert_type(resize(image, (11, 13), layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(multiply(floats, 1.5), NDArray[np.float32])
    vector = np.ones(3, dtype=np.float32)
    assert_type(multiply(floats, vector), NDArray[np.float32])
    assert_type(multiply(floats, floats), NDArray[np.float32])
    assert_type(add(floats, 0.5), NDArray[np.float32])
    assert_type(add(floats, vector), NDArray[np.float32])
    assert_type(add(floats, floats), NDArray[np.float32])
