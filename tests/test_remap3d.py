# ruff: noqa: S101
"""Contract and differential tests for public single-volume ``remap3d``."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from albucore import remap3d


def _volume(dtype: type[np.uint8 | np.float32], channels: int = 5) -> np.ndarray:
    """Build a non-cubic DHWC volume whose values identify all three spatial axes."""
    shape = (2, 3, 4, channels)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if dtype is np.uint8:
        return values.astype(np.uint8)
    return values / np.float32(10.0)


def _normalized_coordinates(indices: np.ndarray, size: int) -> np.ndarray:
    """Convert caller-owned voxel-center indices to align-corners-false normalized coordinates."""
    return (indices.astype(np.float32) * np.float32(2.0) + np.float32(1.0)) / np.float32(size) - np.float32(1.0)


def _identity_grid(size: tuple[int, int, int]) -> np.ndarray:
    """Create a normalized identity pull grid in the public x/y/z coordinate order."""
    depth, height, width = size
    z, y, x = np.meshgrid(
        _normalized_coordinates(np.arange(depth), depth),
        _normalized_coordinates(np.arange(height), height),
        _normalized_coordinates(np.arange(width), width),
        indexing="ij",
    )
    return np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)


def _voxel_grid(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_z: np.ndarray,
    input_size: tuple[int, int, int],
) -> np.ndarray:
    """Build a normalized grid from independent source voxel-coordinate ramps for tests only."""
    depth, height, width = input_size
    return np.stack(
        (
            _normalized_coordinates(source_x, width),
            _normalized_coordinates(source_y, height),
            _normalized_coordinates(source_z, depth),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _nearest_reference_value(
    volume: np.ndarray,
    source: tuple[np.float32, np.float32, np.float32],
    border_mode: int,
    fill: np.ndarray,
) -> np.ndarray:
    """Independently sample one nearest neighbor through either supported border mode."""
    sizes = volume.shape[:3]
    indices = tuple(int(np.rint(coordinate)) for coordinate in source)
    if border_mode == cv2.BORDER_REPLICATE:
        clamped = tuple(np.clip(index, 0, size - 1) for index, size in zip(indices, sizes, strict=True))
        return volume[clamped]
    if all(0 <= index < size for index, size in zip(indices, sizes, strict=True)):
        return volume[indices]
    return fill


def _linear_reference_value(
    volume: np.ndarray,
    source: tuple[np.float32, np.float32, np.float32],
    border_mode: int,
    fill: np.ndarray,
) -> np.ndarray:
    """Independently sample one trilinear value through either supported border mode."""
    sizes = volume.shape[:3]
    lower = tuple(int(np.floor(coordinate)) for coordinate in source)
    weights = tuple(coordinate - index for coordinate, index in zip(source, lower, strict=True))
    value = np.zeros(volume.shape[-1], dtype=np.float32)
    for depth_offset in (0, 1):
        for height_offset in (0, 1):
            for width_offset in (0, 1):
                offsets = (depth_offset, height_offset, width_offset)
                weight = np.float32(1.0)
                for axis, offset in enumerate(offsets):
                    weight *= weights[axis] if offset else np.float32(1.0) - weights[axis]
                indices = tuple(lower[axis] + offset for axis, offset in enumerate(offsets))
                if border_mode == cv2.BORDER_REPLICATE:
                    clamped = tuple(np.clip(index, 0, size - 1) for index, size in zip(indices, sizes, strict=True))
                    value += weight * volume[clamped]
                elif all(0 <= index < size for index, size in zip(indices, sizes, strict=True)):
                    value += weight * volume[indices]
                else:
                    value += weight * fill
    return value


def _scalar_reference(
    volume: np.ndarray,
    sampling_grid: np.ndarray,
    interpolation: int,
    border_mode: int,
    fill: np.ndarray,
) -> np.ndarray:
    """Independently sample one small DHWC volume through the public normalized-grid contract."""
    depth, height, width, channels = volume.shape
    result = np.empty((*sampling_grid.shape[:3], channels), dtype=np.float32)

    for output_z in range(sampling_grid.shape[0]):
        for output_y in range(sampling_grid.shape[1]):
            for output_x in range(sampling_grid.shape[2]):
                grid_x, grid_y, grid_z = sampling_grid[output_z, output_y, output_x]
                source = (
                    np.float32((grid_z + 1.0) * depth / 2.0 - 0.5),
                    np.float32((grid_y + 1.0) * height / 2.0 - 0.5),
                    np.float32((grid_x + 1.0) * width / 2.0 - 0.5),
                )
                sample = _nearest_reference_value if interpolation == cv2.INTER_NEAREST else _linear_reference_value
                result[output_z, output_y, output_x] = sample(volume, source, border_mode, fill)
    return result


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 3, 5, 9], ids=["c1", "c3", "c5", "c9"])
def test_remap3d_identity_grid_samples_once_and_preserves_the_volume_contract(
    dtype: type[np.uint8 | np.float32],
    channels: int,
) -> None:
    """An identity grid returns equal allocated data rather than relying on an O(DHW) identity scan."""
    volume = _volume(dtype, channels)

    result = remap3d(volume, _identity_grid(volume.shape[:3]), interpolation=cv2.INTER_NEAREST)

    assert result is not volume
    assert result.shape == volume.shape
    assert result.dtype == volume.dtype
    np.testing.assert_array_equal(result, volume)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR], ids=["nearest", "linear"])
def test_remap3d_matches_an_independent_scalar_reference_for_normalized_coordinate_ramps(
    interpolation: int,
) -> None:
    """Coordinate ramps cover x/y/z order, a non-cubic output, half voxels, and constant fill."""
    volume = _volume(np.float32, channels=3)
    source_z, source_y, source_x = np.meshgrid(
        np.array((-0.25, 0.5), dtype=np.float32),
        np.array((0.0, 1.5), dtype=np.float32),
        np.array((-0.5, 1.25, 3.2), dtype=np.float32),
        indexing="ij",
    )
    grid = _voxel_grid(source_x, source_y, source_z, volume.shape[:3])
    fill = np.array((1.5, -2.0, 7.0), dtype=np.float32)

    result = remap3d(volume, grid, interpolation=interpolation, border_value=fill)
    expected = _scalar_reference(volume, grid, interpolation, cv2.BORDER_CONSTANT, fill)

    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-5)


def test_remap3d_axis_impulses_confirm_x_y_z_grid_order() -> None:
    """Each grid coordinate selects the matching source spatial axis rather than a permuted axis."""
    volume = np.zeros((3, 4, 5, 1), dtype=np.uint8)
    volume[2, 1, 3, 0] = 255
    source_x = np.full((1, 1, 1), 3, dtype=np.float32)
    source_y = np.full((1, 1, 1), 1, dtype=np.float32)
    source_z = np.full((1, 1, 1), 2, dtype=np.float32)
    grid = _voxel_grid(source_x, source_y, source_z, volume.shape[:3])

    result = remap3d(volume, grid, interpolation=cv2.INTER_NEAREST)

    np.testing.assert_array_equal(result, np.array([[[[255]]]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("border_mode", "border_value", "expected"),
    [
        (cv2.BORDER_CONSTANT, None, np.array((0.0, 0.0, 0.0), dtype=np.float32)),
        (cv2.BORDER_CONSTANT, 17.0, np.array((17.0, 17.0, 17.0), dtype=np.float32)),
        (cv2.BORDER_CONSTANT, (11.0, 23.0, 37.0), np.array((11.0, 23.0, 37.0), dtype=np.float32)),
        (cv2.BORDER_REPLICATE, (11.0, 23.0, 37.0), np.array((0.0, 0.1, 0.2), dtype=np.float32)),
    ],
    ids=["constant_zero", "constant_scalar", "constant_per_channel", "replicate"],
)
def test_remap3d_borders_apply_to_outside_coordinates(
    border_mode: int,
    border_value: float | tuple[float, ...] | None,
    expected: np.ndarray,
) -> None:
    """Outside pulls use constant fill or the nearest source voxel for all channels."""
    volume = _volume(np.float32, channels=3)
    grid = np.array([[[[-1.5, -1.5, -1.5]]]], dtype=np.float32)

    result = remap3d(
        volume,
        grid,
        interpolation=cv2.INTER_NEAREST,
        border_mode=border_mode,
        border_value=border_value,
    )

    np.testing.assert_array_equal(result.reshape(-1), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR], ids=["nearest", "linear"])
def test_remap3d_numpy_and_tensor_grid_storage_match(
    dtype: type[np.uint8 | np.float32],
    interpolation: int,
) -> None:
    """The NumPy route reads equal caller-owned NumPy and CPU Tensor grid storage identically."""
    volume = _volume(dtype, channels=5)
    grid = _identity_grid(volume.shape[:3])
    grid[:, :, :, 0] += np.float32(0.2)

    numpy_grid_result = remap3d(volume, grid, interpolation=interpolation, border_value=17.0)
    tensor_grid_result = remap3d(volume, torch.from_numpy(grid), interpolation=interpolation, border_value=17.0)

    assert tensor_grid_result.shape == volume.shape
    assert tensor_grid_result.dtype == volume.dtype
    np.testing.assert_array_equal(numpy_grid_result, tensor_grid_result)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 3, 5, 9], ids=["c1", "c3", "c5", "c9"])
@pytest.mark.parametrize("grid_container", ["numpy", "tensor"])
@pytest.mark.parametrize("layout", ["contiguous", "channel_last_strided"])
def test_remap3d_tensor_volume_preserves_cdhw_and_matches_numpy_route(
    dtype: type[np.uint8 | np.float32],
    channels: int,
    grid_container: str,
    layout: str,
) -> None:
    """The public Tensor fallback preserves CDHW without mutating either caller-owned input."""
    numpy_volume = _volume(dtype, channels)
    volume = torch.from_numpy(numpy_volume).permute(3, 0, 1, 2)
    if layout == "contiguous":
        volume = volume.contiguous()
    original = volume.clone()
    grid = _identity_grid(numpy_volume.shape[:3])
    original_grid = grid.copy()
    sampling_grid = grid if grid_container == "numpy" else torch.from_numpy(grid.copy())

    result = remap3d(volume, sampling_grid, interpolation=cv2.INTER_LINEAR, border_value=17.0)
    expected = remap3d(numpy_volume, grid, interpolation=cv2.INTER_LINEAR, border_value=17.0)

    assert isinstance(result, torch.Tensor)
    assert result.shape == volume.shape
    assert result.dtype == volume.dtype
    torch.testing.assert_close(result, torch.from_numpy(expected).permute(3, 0, 1, 2), rtol=0, atol=0)
    torch.testing.assert_close(volume, original, rtol=0, atol=0)
    if isinstance(sampling_grid, torch.Tensor):
        torch.testing.assert_close(sampling_grid, torch.from_numpy(original_grid), rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(sampling_grid, original_grid)


def test_remap3d_tensor_result_feeds_a_trainable_module() -> None:
    """The no-grad resampling result remains valid input for a downstream trainable module."""
    volume = torch.from_numpy(_volume(np.float32, channels=3)).permute(3, 0, 1, 2).contiguous()
    result = remap3d(volume, _identity_grid((2, 3, 4)), interpolation=cv2.INTER_LINEAR)
    module = torch.nn.Conv3d(3, 1, kernel_size=1)

    module(result.unsqueeze(0)).sum().backward()

    assert module.weight.grad is not None


def test_remap3d_accepts_positive_negative_and_read_only_numpy_volume_views_without_mutation() -> None:
    """Torch receives one repair copy only where its shared-storage contract requires it."""
    source = _volume(np.float32, channels=3)
    positive = source[:, ::2, :, :]
    negative = source[:, :, ::-1, :]
    readonly = source.copy()
    readonly.setflags(write=False)

    for volume in (positive, negative, readonly):
        original = volume.copy()
        result = remap3d(volume, _identity_grid(volume.shape[:3]), interpolation=cv2.INTER_LINEAR)

        np.testing.assert_allclose(result, volume, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(volume, original)
        assert not np.shares_memory(result, volume)


def test_remap3d_uses_one_grid_sample_per_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-channel fill correction remains fused around one volumetric resampling operation."""
    volume = _volume(np.float32, channels=3)
    calls = 0
    original_grid_sample = torch.nn.functional.grid_sample

    def counted_grid_sample(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_grid_sample(*args, **kwargs)

    monkeypatch.setattr("albucore.sampling3d.torch_f.grid_sample", counted_grid_sample)

    remap3d(volume, _identity_grid(volume.shape[:3]), border_value=(11.0, 23.0, 37.0))

    assert calls == 1


def test_remap3d_uint8_restores_after_one_float32_sampling_pass() -> None:
    """Half-voxel trilinear values receive the documented final saturating round once."""
    volume = np.array([[[[0], [255]]]], dtype=np.uint8)
    grid = _voxel_grid(
        np.array([[[0.5]]], dtype=np.float32),
        np.zeros((1, 1, 1), dtype=np.float32),
        np.zeros((1, 1, 1), dtype=np.float32),
        volume.shape[:3],
    )

    result = remap3d(volume, grid, interpolation=cv2.INTER_LINEAR)

    np.testing.assert_array_equal(result, np.array([[[[128]]]], dtype=np.uint8))
