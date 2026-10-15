"""Checked consumer calls for the public NumPy/Tensor type relationships."""

from typing import TypeAlias, assert_type

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from albucore import (
    add,
    add_array,
    add_constant,
    add_vector,
    add_weighted,
    apply_uint8_lut,
    clip,
    clipped,
    copy_make_border,
    exp,
    float32_io,
    from_float,
    hflip,
    log,
    median_blur,
    multiply,
    multiply_by_array,
    multiply_by_constant,
    multiply_by_vector,
    multiply_add,
    normalize,
    normalize_per_image,
    power,
    preserve_channel_dim,
    remap,
    resize,
    sqrt,
    sz_lut,
    to_float,
    uint8_io,
    vflip,
    warp_affine,
    warp_perspective,
)

ConsumerImage: TypeAlias = NDArray[np.uint8] | NDArray[np.float32]


@float32_io
@clipped
@preserve_channel_dim
def floating_kernel(img: ConsumerImage, amount: float) -> ConsumerImage:
    result = img.copy()
    np.multiply(result, amount, out=result, casting="unsafe")
    return result


@uint8_io
@preserve_channel_dim
def integer_kernel(img: NDArray[np.uint8], *, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    result = img.copy()
    np.bitwise_and(result, mask, out=result)
    return result


@clipped
def uint8_mask(img: NDArray[np.float32]) -> NDArray[np.uint8]:
    return (img > 0.5).astype(np.uint8)


@clipped
def floating_result(img: NDArray[np.float32]) -> NDArray[np.float32]:
    return img.copy()


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
    image: ConsumerImage, labels: NDArray[np.int16], matrix: NDArray[np.float32], grid: NDArray[np.float32]
) -> None:
    assert_type(resize(image, (11, 13)), ConsumerImage)
    assert_type(copy_make_border(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT), ConsumerImage)
    assert_type(warp_affine(image, matrix, (11, 13)), ConsumerImage)
    assert_type(warp_perspective(image, matrix, (11, 13)), ConsumerImage)
    assert_type(remap(labels, grid, grid, cv2.INTER_NEAREST), NDArray[np.int16])


def check_conversions(
    image: ConsumerImage, half: NDArray[np.float16], floats: NDArray[np.float32], dtype: np.dtype[np.uint8 | np.float32]
) -> None:
    assert_type(to_float(image), NDArray[np.float32])
    assert_type(to_float(half), NDArray[np.float16])
    assert_type(clip(image, dtype), ConsumerImage)
    assert_type(clip(floats, np.dtype(np.uint8)), NDArray[np.uint8])
    assert_type(from_float(half, np.dtype(np.uint8)), NDArray[np.uint8])
    assert_type(from_float(floats, np.dtype(np.float16)), NDArray[np.float16])


def check_luts(image: NDArray[np.uint8], lut: NDArray[np.uint8]) -> None:
    assert_type(apply_uint8_lut(image, lut), NDArray[np.uint8])
    assert_type(sz_lut(image, lut), NDArray[np.uint8])


def check_decorators(image: ConsumerImage, half: NDArray[np.float16], mask: NDArray[np.uint8]) -> None:
    assert_type(uint8_mask(half.astype(np.float32)), NDArray[np.uint8])
    assert_type(floating_result(half.astype(np.float32)), NDArray[np.float32])
    assert_type(floating_kernel(img=image, amount=1.5), ConsumerImage)
    assert_type(floating_kernel(half, amount=1.5), NDArray[np.float16])
    assert_type(integer_kernel(img=image, mask=mask), ConsumerImage)


def check_tensor(image: torch.Tensor, matrix: NDArray[np.float32], lut: NDArray[np.uint8]) -> None:
    assert_type(multiply(image, 1.5, layout="CHW"), torch.Tensor)
    assert_type(normalize(image, 0.5, 2.0, layout="CHW"), torch.Tensor)
    assert_type(normalize_per_image(image, "image", layout="CHW"), torch.Tensor)
    assert_type(to_float(image), torch.Tensor)
    assert_type(from_float(image, torch.uint8), torch.Tensor)
    assert_type(hflip(image), torch.Tensor)
    assert_type(median_blur(image, 3, layout="CHW"), torch.Tensor)
    assert_type(resize(image, (11, 13), layout="CHW"), torch.Tensor)
    assert_type(warp_affine(image, matrix, (11, 13), layout="CHW"), torch.Tensor)
    assert_type(apply_uint8_lut(image, lut, layout="CHW"), torch.Tensor)
    assert_type(floating_kernel.with_layout("CHW")(image, amount=1.5), torch.Tensor)


def check_union_inputs(image: ConsumerImage | torch.Tensor, floats: NDArray[np.float32]) -> None:
    assert_type(floating_kernel.with_layout("CHW")(image, amount=1.5), ConsumerImage | torch.Tensor)
    assert_type(floating_kernel.with_layout("CHW")(img=image, amount=1.5), ConsumerImage | torch.Tensor)
    assert_type(multiply(image, 1.5, layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(add(image, 0.5, layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(add_weighted(image, 0.5, image, 0.5, layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(to_float(image), NDArray[np.float32] | torch.Tensor)
    assert_type(normalize(image, 0.5, 2.0, layout="CHW"), NDArray[np.float32] | torch.Tensor)
    assert_type(resize(image, (11, 13), layout="CHW"), ConsumerImage | torch.Tensor)
    assert_type(multiply_by_constant(floats, 1.5), NDArray[np.float32])
    vector = np.ones(3, dtype=np.float32)
    assert_type(multiply_by_vector(floats, vector), NDArray[np.float32])
    assert_type(multiply_by_array(floats, floats), NDArray[np.float32])
    assert_type(add_constant(floats, 0.5), NDArray[np.float32])
    assert_type(add_vector(floats, vector), NDArray[np.float32])
    assert_type(add_array(floats, floats), NDArray[np.float32])


@clipped
@float32_io
def clipping_outside_float(img: NDArray[np.float32], amount: float) -> NDArray[np.float32]:
    return img * amount


@clipped
@uint8_io
def clipping_outside_integer(img: NDArray[np.uint8], *, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return np.bitwise_and(img, mask)


@uint8_io
@clipped
def clipping_inside_integer(image: NDArray[np.uint8], *, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return np.bitwise_and(image, mask)


@float32_io
def kernel_with_layout_argument(img: NDArray[np.float32], *, layout: float) -> NDArray[np.float32]:
    return img * layout


def check_outer_clipping(
    image: ConsumerImage,
    half: NDArray[np.float16],
    floats: NDArray[np.float32],
    bytes_image: NDArray[np.uint8],
    tensor: torch.Tensor,
) -> None:
    assert_type(clipping_outside_float(image, 1.5), ConsumerImage)
    assert_type(clipping_outside_float(img=image, amount=1.5), ConsumerImage)
    assert_type(clipping_outside_float(half, 1.5), NDArray[np.float16])
    assert_type(clipping_outside_float(floats, 1.5), NDArray[np.float32])
    assert_type(clipping_outside_float(bytes_image, 1.5), NDArray[np.uint8])
    assert_type(clipping_outside_float.with_layout("CHW")(tensor, 1.5), torch.Tensor)
    assert_type(clipping_outside_float.with_layout("CHW")(img=tensor, amount=1.5), torch.Tensor)
    assert_type(clipping_outside_integer(image, mask=bytes_image), ConsumerImage)
    assert_type(clipping_outside_integer(img=half, mask=bytes_image), NDArray[np.float16])
    assert_type(clipping_outside_integer.with_layout("CHW")(tensor, mask=bytes_image), torch.Tensor)
    assert_type(clipping_inside_integer(bytes_image, mask=bytes_image), NDArray[np.uint8])
    assert_type(clipping_inside_integer(half, mask=bytes_image), NDArray[np.float16])
    assert_type(clipping_inside_integer(floats, mask=bytes_image), NDArray[np.float32])
    assert_type(clipping_inside_integer(image, mask=bytes_image), ConsumerImage)
    assert_type(clipping_inside_integer(img=half, mask=bytes_image), NDArray[np.float16])
    assert_type(clipping_inside_integer.with_layout("CHW")(tensor, mask=bytes_image), torch.Tensor)
    assert_type(kernel_with_layout_argument.with_layout("CHW")(tensor, layout=1.5), torch.Tensor)


def check_invalid_decorator_calls(floats: NDArray[np.float32], tensor: torch.Tensor) -> None:
    # warn_unused_ignores makes these checks fail if an invalid call becomes accepted.
    clipping_outside_float(img=floats)  # type: ignore[call-arg]
    clipping_outside_integer(img=floats)  # type: ignore[call-arg]
    clipping_outside_float.with_layout("CHW")(tensor)  # type: ignore[call-arg]
    clipping_inside_integer.with_layout("CHW")(tensor)  # type: ignore[call-arg]
    clipping_outside_float.with_layout(None)  # type: ignore[arg-type]
    clipping_outside_float(img=floats, amount="invalid")  # type: ignore[arg-type]
    clipping_outside_float.with_layout("CHW")(tensor, amount="invalid")  # type: ignore[arg-type]
    clipping_inside_integer.with_layout("CHW")(tensor, mask=1)  # type: ignore[arg-type]
