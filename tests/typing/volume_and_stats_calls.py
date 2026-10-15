"""Inferred output types for valid volume and reduction calls."""

from typing import TypeAlias, assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from albucore import (
    flip_volume,
    gaussian_blur3d,
    matmul,
    mean,
    mean_std,
    pad3d,
    pairwise_distances_squared,
    reduce_sum,
    remap3d,
    resize3d,
    rot90_volume,
    separable_filter3d,
    std,
    transpose_volume,
    warp_affine3d,
)

ConsumerImage: TypeAlias = NDArray[np.uint8] | NDArray[np.float32]
StatsScalar: TypeAlias = float | np.float64


def check_volumes(
    image: ConsumerImage,
    half: NDArray[np.float16],
    labels: NDArray[np.int16],
    matrix: NDArray[np.float32],
    grid: NDArray[np.float32],
) -> None:
    assert_type(flip_volume(image, 0), ConsumerImage)
    assert_type(transpose_volume(image, 0, 1), ConsumerImage)
    assert_type(rot90_volume(image, 1, (0, 1)), ConsumerImage)
    assert_type(resize3d(image, (3, 11, 13)), ConsumerImage)
    assert_type(warp_affine3d(half, matrix, (3, 11, 13)), NDArray[np.float16])
    assert_type(remap3d(half, grid), NDArray[np.float16])
    assert_type(gaussian_blur3d(image, 1.0), ConsumerImage)
    kernel = np.ones(1, dtype=np.float32)
    assert_type(separable_filter3d(half, (kernel, kernel, kernel)), NDArray[np.float16])
    assert_type(pad3d(labels, (1, 1, 1, 1, 1, 1)), NDArray[np.int16])
    assert_type(mean(half), np.float64)
    assert_type(std(half), float)
    assert_type(mean_std(half), tuple[float, float])
    assert_type(matmul(matrix, matrix), NDArray[np.float32])
    assert_type(pairwise_distances_squared(matrix, matrix), NDArray[np.float32])


def check_statistics(image: NDArray[np.float32], bytes_image: NDArray[np.uint8], keepdims: bool) -> None:
    assert_type(mean(image), np.float64)
    assert_type(std(image), float)
    assert_type(mean_std(image), tuple[np.float64, np.float64])
    assert_type(mean(image, "per_channel"), NDArray[np.float64])
    assert_type(std(image, keepdims=True), NDArray[np.float64])
    assert_type(mean_std(image, "per_channel"), tuple[NDArray[np.float64], NDArray[np.float64]])
    assert_type(mean(image, dtype=np.float32), NDArray[np.float32])
    assert_type(std(image, dtype=np.dtype(np.float16)), NDArray[np.float16])
    assert_type(mean(image, keepdims=keepdims), StatsScalar | NDArray[np.float64])
    assert_type(reduce_sum(bytes_image), np.uint64)
    assert_type(reduce_sum(image), np.float64)
    assert_type(reduce_sum(bytes_image, "per_channel"), NDArray[np.uint64])
    assert_type(reduce_sum(image, keepdims=True), NDArray[np.float64])
    assert_type(reduce_sum(image, axis=(0, 1)), np.float64 | NDArray[np.float64])


def check_tensors(
    image: torch.Tensor, matrix: NDArray[np.float32], grid: torch.Tensor, tensor_matrix: torch.Tensor
) -> None:
    assert_type(flip_volume(image, 0), torch.Tensor)
    assert_type(transpose_volume(image, 0, 1), torch.Tensor)
    assert_type(rot90_volume(image, 1, (0, 1)), torch.Tensor)
    assert_type(resize3d(image, (3, 11, 13)), torch.Tensor)
    assert_type(warp_affine3d(image, matrix, (3, 11, 13)), torch.Tensor)
    assert_type(remap3d(image, grid), torch.Tensor)
    assert_type(gaussian_blur3d(image, 1.0), torch.Tensor)
    assert_type(pad3d(image, (1, 1, 1, 1, 1, 1)), torch.Tensor)
    assert_type(mean(image, layout="CDHW"), torch.Tensor)
    tensor_mean, tensor_std = mean_std(image, layout="CDHW")
    assert_type(tensor_mean, torch.Tensor)
    assert_type(tensor_std, torch.Tensor)
    assert_type(std(image, "per_channel", layout="CDHW"), torch.Tensor)
    assert_type(reduce_sum(image, layout="CDHW"), torch.Tensor)
    assert_type(matmul(tensor_matrix, tensor_matrix), torch.Tensor)
    assert_type(pairwise_distances_squared(tensor_matrix, tensor_matrix), torch.Tensor)
