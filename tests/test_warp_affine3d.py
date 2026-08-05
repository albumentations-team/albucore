# ruff: noqa: S101
"""Contract and differential tests for public single-volume ``warp_affine3d``."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from albucore import warp_affine, warp_affine3d


def _volume(dtype: type[np.uint8 | np.float32], channels: int = 5) -> np.ndarray:
    """Build a non-cubic ``DHWC`` volume with values that expose spatial-axis swaps."""
    shape = (3, 4, 5, channels)
    if dtype is np.uint8:
        return np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.float32(10.0)


def _translation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Build one forward ``(x, y, z)`` translation matrix."""
    return np.array(
        (
            (1.0, 0.0, 0.0, x),
            (0.0, 1.0, 0.0, y),
            (0.0, 0.0, 1.0, z),
        ),
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "volume",
    [
        np.zeros((3, 4, 1), dtype=np.uint8),
        torch.zeros((1, 3, 4), dtype=torch.uint8),
    ],
    ids=("numpy_rank_3", "torch_rank_3"),
)
def _numpy_trilinear_constant_reference(
    volume: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int, int],
    fill: np.ndarray,
) -> np.ndarray:
    """Independently sample one small DHWC volume with inverse affine trilinear interpolation."""
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3] = matrix
    inverse = np.linalg.inv(homogeneous)
    result = np.empty((*size, volume.shape[-1]), dtype=np.float32)
    depth, height, width = volume.shape[:3]

    for output_z in range(size[0]):
        for output_y in range(size[1]):
            for output_x in range(size[2]):
                source_x, source_y, source_z, _ = inverse @ (output_x, output_y, output_z, 1.0)
                source = (source_z, source_y, source_x)
                lower = tuple(int(np.floor(coordinate)) for coordinate in source)
                weights = tuple(coordinate - axis_lower for coordinate, axis_lower in zip(source, lower, strict=True))
                value = np.zeros(volume.shape[-1], dtype=np.float32)
                for depth_offset in (0, 1):
                    for height_offset in (0, 1):
                        for width_offset in (0, 1):
                            offsets = depth_offset, height_offset, width_offset
                            weight = np.float32(1.0)
                            for axis, offset in enumerate(offsets):
                                weight *= weights[axis] if offset else 1.0 - weights[axis]
                            indices = tuple(lower[axis] + offset for axis, offset in enumerate(offsets))
                            if all(
                                0 <= index_ < axis_size
                                for index_, axis_size in zip(indices, (depth, height, width), strict=True)
                            ):
                                value += weight * volume[indices]
                            else:
                                value += weight * fill
                result[output_z, output_y, output_x] = value
    return result


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 3, 5, 9], ids=["c1", "c3", "c5", "c9"])
def test_warp_affine3d_identity_preserves_public_container_dtype_and_aliasing(
    dtype: type[np.uint8 | np.float32],
    channels: int,
) -> None:
    """Exact identity is a no-allocation fast path for both documented single-volume layouts."""
    volume = _volume(dtype, channels)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    matrix = np.eye(4, dtype=np.float32)

    assert warp_affine3d(volume, matrix, volume.shape[:3]) is volume
    assert warp_affine3d(tensor, matrix, (tensor.shape[1], tensor.shape[2], tensor.shape[3])) is tensor


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_warp_affine3d_forward_x_translation_uses_constant_per_channel_fill(
    dtype: type[np.uint8 | np.float32],
) -> None:
    """A forward positive x translation pulls source x-1 and fills the first output column."""
    volume = _volume(dtype, channels=5)
    fill = np.array((11.0, 23.0, 37.0, 41.0, 53.0), dtype=np.float32)

    result = warp_affine3d(
        volume,
        _translation(x=1.0),
        volume.shape[:3],
        interpolation=cv2.INTER_NEAREST,
        border_value=fill,
    )
    expected = np.empty_like(volume)
    expected[:, :, 0, :] = fill.astype(dtype)
    expected[:, :, 1:, :] = volume[:, :, :-1, :]

    if dtype is np.uint8:
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)


def test_warp_affine3d_replicate_border_uses_the_nearest_edge_voxel() -> None:
    """Replicate padding does not consume constant fill and keeps the source edge at x=0."""
    volume = _volume(np.uint8, channels=3)

    result = warp_affine3d(
        volume,
        _translation(x=1.0),
        volume.shape[:3],
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REPLICATE,
        border_value=(255.0, 254.0, 253.0),
    )
    expected = np.empty_like(volume)
    expected[:, :, 0, :] = volume[:, :, 0, :]
    expected[:, :, 1:, :] = volume[:, :, :-1, :]

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["depth", "height", "width"])
def test_warp_affine3d_nearest_reflection_matches_numpy_flip(axis: int) -> None:
    """Forward reflections in x/y/z map to the corresponding explicit volume axis."""
    volume = _volume(np.uint8, channels=5)
    matrix = np.eye(4, dtype=np.float32)
    coordinate_axis = 2 - axis
    matrix[coordinate_axis, coordinate_axis] = -1.0
    matrix[coordinate_axis, 3] = volume.shape[axis] - 1

    result = warp_affine3d(volume, matrix, volume.shape[:3], interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result, np.flip(volume, axis=axis))


@pytest.mark.parametrize(
    ("matrix", "size", "axes"),
    [
        (
            np.array(((0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 4.0), (0.0, 0.0, 1.0, 0.0)), dtype=np.float32),
            (3, 5, 4),
            (1, 2),
        ),
        (
            np.array(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 3.0)), dtype=np.float32),
            (4, 3, 5),
            (0, 1),
        ),
        (
            np.array(((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 4.0)), dtype=np.float32),
            (5, 4, 3),
            (0, 2),
        ),
    ],
    ids=["height_width", "depth_height", "depth_width"],
)
def test_warp_affine3d_nearest_90_degree_rotations_match_numpy_rot90(
    matrix: np.ndarray,
    size: tuple[int, int, int],
    axes: tuple[int, int],
) -> None:
    """A 90-degree rotation over every spatial-axis pair uses the documented x/y/z order."""
    volume = _volume(np.uint8, channels=5)

    result = warp_affine3d(volume, matrix, size, interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result, np.rot90(volume, axes=axes))


def test_warp_affine3d_in_plane_d1_rotation_matches_2d_warp_affine() -> None:
    """The D=1 in-plane convention agrees with the existing public 2D geometric router."""
    volume = _volume(np.uint8, channels=3)[:1]
    matrix_3d = np.array(((0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 4.0), (0.0, 0.0, 1.0, 0.0)), dtype=np.float32)
    matrix_2d = matrix_3d[:2, (0, 1, 3)]

    result_3d = warp_affine3d(volume, matrix_3d, (1, 5, 4), interpolation=cv2.INTER_NEAREST)
    result_2d = warp_affine(volume[0], matrix_2d, (4, 5), flags=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result_3d[0], result_2d)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR], ids=["nearest", "linear"])
def test_warp_affine3d_numpy_and_torch_paths_match_after_layout_conversion(
    dtype: type[np.uint8 | np.float32],
    interpolation: int,
) -> None:
    """The NumPy bridge and direct CPU Tensor path have one shared sampling contract."""
    volume = _volume(dtype, channels=5)
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    matrix = np.array(
        (
            (0.9, 0.1, 0.0, 0.4),
            (0.0, 1.1, 0.1, -0.3),
            (0.05, 0.0, 1.0, 0.2),
        ),
        dtype=np.float32,
    )
    size = (2, 5, 4)

    numpy_result = warp_affine3d(volume, matrix, size, interpolation=interpolation, border_value=17.0)
    tensor_result = warp_affine3d(tensor, matrix, size, interpolation=interpolation, border_value=17.0)

    assert numpy_result.shape == (*size, volume.shape[-1])
    assert tensor_result.shape == (volume.shape[-1], *size)
    np.testing.assert_array_equal(numpy_result, tensor_result.permute(1, 2, 3, 0).numpy())


def test_warp_affine3d_linear_matches_an_independent_numpy_trilinear_reference() -> None:
    """Arbitrary trilinear sampling follows the public forward matrix and constant fill semantics."""
    volume = _volume(np.float32, channels=3)
    matrix = np.array(
        (
            (0.9, 0.1, 0.0, 0.4),
            (0.0, 1.1, 0.1, -0.3),
            (0.05, 0.0, 1.0, 0.2),
        ),
        dtype=np.float32,
    )
    size = (2, 5, 4)
    fill = np.array((1.5, -2.0, 7.0), dtype=np.float32)

    result = warp_affine3d(volume, matrix, size, interpolation=cv2.INTER_LINEAR, border_value=fill)
    expected = _numpy_trilinear_constant_reference(volume, matrix, size, fill)

    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_warp_affine3d_linear_constant_fill_has_the_expected_half_voxel_blend(
    dtype: type[np.uint8 | np.float32],
) -> None:
    """Shifted-input fill correction blends a half-voxel boundary without an extra coverage kernel."""
    volume = np.ones((1, 1, 4, 1), dtype=dtype)

    result = warp_affine3d(
        volume,
        _translation(x=0.5),
        volume.shape[:3],
        interpolation=cv2.INTER_LINEAR,
        border_value=3.0,
    )

    expected = np.array((2.0, 1.0, 1.0, 1.0), dtype=dtype).reshape(1, 1, 4, 1)
    np.testing.assert_array_equal(result, expected)


def test_warp_affine3d_nearest_half_voxel_ties_to_even() -> None:
    """Nearest sampling follows the documented CPU Torch tie rule at interior half-voxels."""
    volume = np.arange(4, dtype=np.uint8).reshape(1, 1, 4, 1)

    result = warp_affine3d(
        volume,
        _translation(x=0.5),
        volume.shape[:3],
        interpolation=cv2.INTER_NEAREST,
        border_value=99.0,
    )

    np.testing.assert_array_equal(result.reshape(-1), np.array((0, 0, 2, 2), dtype=np.uint8))


def test_warp_affine3d_accepts_positive_and_negative_stride_numpy_views() -> None:
    """Positive-stride inputs use zero-copy storage and negative strides receive one explicit repair copy."""
    volume = _volume(np.float32, channels=5)
    positive_stride = volume[:, ::2, :, :]
    negative_stride = volume[:, :, ::-1, :]

    positive_result = warp_affine3d(positive_stride, _translation(), (2, 3, 4))
    negative_result = warp_affine3d(negative_stride, _translation(), (2, 3, 4))

    assert positive_result.shape == negative_result.shape == (2, 3, 4, 5)
    assert positive_result.dtype == negative_result.dtype == np.float32


def test_warp_affine3d_accepts_read_only_numpy_storage_with_a_repair_copy() -> None:
    """A read-only NumPy input is copied once before Torch receives mutable shared storage."""
    volume = _volume(np.float32, channels=3)
    volume.setflags(write=False)

    result = warp_affine3d(volume, _translation(x=0.25), (2, 5, 4))

    assert result.shape == (2, 5, 4, 3)
    assert result.dtype == np.float32


def test_warp_affine3d_accepts_noncontiguous_torch_cdhw() -> None:
    """Torch sampling consumes an existing strided CDHW view without a public layout repair."""
    volume = torch.from_numpy(_volume(np.float32, channels=5)).permute(3, 0, 1, 2)[:, :, :, ::2]

    result = warp_affine3d(volume, _translation(), (2, 3, 4))

    assert result.shape == (5, 2, 3, 4)
    assert result.dtype == torch.float32


def test_warp_affine3d_output_can_feed_a_trainable_torch_module() -> None:
    """A non-autograd input still produces a normal Tensor usable by later training layers."""
    volume = torch.from_numpy(_volume(np.float32, channels=1)).permute(3, 0, 1, 2)
    layer = torch.nn.Conv3d(1, 1, kernel_size=1)

    result = warp_affine3d(volume, _translation(x=0.25), (2, 3, 4))
    loss = layer(result.unsqueeze(0)).sum()
    loss.backward()

    assert layer.weight.grad is not None


def test_warp_affine3d_3x4_and_homogeneous_4x4_matrices_are_equivalent() -> None:
    """The two documented affine matrix encodings produce the same output."""
    volume = _volume(np.float32, channels=3)
    matrix_3x4 = _translation(x=0.75, y=-0.25, z=0.5)
    matrix_4x4 = np.eye(4, dtype=np.float32)
    matrix_4x4[:3] = matrix_3x4

    result_3x4 = warp_affine3d(volume, matrix_3x4, (2, 5, 4), border_value=2.0)
    result_4x4 = warp_affine3d(volume, matrix_4x4, (2, 5, 4), border_value=2.0)

    np.testing.assert_array_equal(result_3x4, result_4x4)


def test_warp_affine3d_does_not_mutate_a_real_warp_input() -> None:
    """Affine sampling is allocating because output voxels read source values in arbitrary order."""
    volume = _volume(np.float32, channels=3)
    original = volume.copy()

    result = warp_affine3d(volume, _translation(x=0.25), (2, 5, 4), border_value=7.0)

    np.testing.assert_array_equal(volume, original)
    assert not np.shares_memory(result, volume)
