# ruff: noqa: S101
"""Public single-volume contract tests for ``gaussian_blur3d``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from albucore import gaussian_blur3d, separable_filter3d


def test_gaussian_blur3d_smooths_the_depth_axis_without_touching_channels() -> None:
    """An anisotropic depth-only sigma produces the documented discrete 1D Gaussian."""
    volume = np.zeros((5, 1, 1, 2), dtype=np.float32)
    volume[2, 0, 0, 0] = 1.0

    result = gaussian_blur3d(volume, sigma=(1.0, 0.0, 0.0), kernel_size=(3, 0, 0))

    expected = np.array((0.0, 0.27406862, 0.45186275, 0.27406862, 0.0), dtype=np.float32)
    np.testing.assert_allclose(result[:, 0, 0, 0], expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(result[..., 1], 0.0)


@pytest.mark.parametrize(
    "volume",
    [
        np.zeros((3, 4, 1), dtype=np.uint8),
        torch.zeros((1, 3, 4), dtype=torch.uint8),
    ],
    ids=("numpy_rank_3", "torch_rank_3"),
)
def test_gaussian_blur3d_rejects_invalid_rank(volume: np.ndarray | torch.Tensor) -> None:
    with pytest.raises(ValueError, match="rank-4"):
        gaussian_blur3d(volume, sigma=0.0)


def _numpy_separable_reference(volume: np.ndarray, kernels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Independent reflect-101 reference for tiny float32 volumes."""
    result = volume
    for axis, kernel in enumerate(kernels):
        radius = kernel.size // 2
        if radius == 0:
            continue
        padding = [(0, 0)] * result.ndim
        padding[axis] = radius, radius
        padded = np.pad(result, padding, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)
        result = np.tensordot(windows, kernel, axes=((-1,), (0,))).astype(np.float32)
    return result


def test_separable_filter3d_uses_reflect101_for_unit_and_short_axes() -> None:
    """A small volume exercises the fallback that Torch reflect padding cannot represent directly."""
    volume = np.arange(1 * 2 * 3 * 5, dtype=np.float32).reshape(1, 2, 3, 5)
    kernels = (
        np.array((1.0, 2.0, 1.0), dtype=np.float32) / 4.0,
        np.array((1.0, 3.0, 1.0), dtype=np.float32) / 5.0,
        np.array((1.0, 4.0, 1.0), dtype=np.float32) / 6.0,
    )

    result = separable_filter3d(volume, kernels)

    np.testing.assert_allclose(result, _numpy_separable_reference(volume, kernels), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("channels", [1, 5], ids=["c1", "c5"])
def test_gaussian_blur3d_numpy_and_torch_paths_have_one_contract(
    dtype: type[np.uint8 | np.float32],
    channels: int,
) -> None:
    """Both documented layouts execute the same three-pass kernel without changing the input."""
    rng = np.random.default_rng(20260805)
    shape = (3, 5, 7, channels)
    volume: np.ndarray
    if dtype is np.uint8:
        volume = rng.integers(0, 256, size=shape, dtype=np.uint8)
    else:
        volume = rng.random(shape, dtype=np.float32)
    before = volume.copy()
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)

    numpy_result = gaussian_blur3d(volume, sigma=(0.75, 1.25, 0.0), kernel_size=(5, 0, 0))
    tensor_result = gaussian_blur3d(tensor, sigma=(0.75, 1.25, 0.0), kernel_size=(5, 0, 0))

    assert numpy_result.shape == volume.shape
    assert numpy_result.dtype == volume.dtype
    assert tensor_result.shape == tensor.shape
    assert tensor_result.dtype == tensor.dtype
    np.testing.assert_array_equal(numpy_result, tensor_result.permute(1, 2, 3, 0).numpy())
    np.testing.assert_array_equal(volume, before)


@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_gaussian_blur3d_zero_sigmas_return_the_original_container(container: str) -> None:
    """A no-op Gaussian has no allocation or hidden container conversion."""
    numpy_volume = np.random.default_rng(20260805).random((3, 5, 7, 3), dtype=np.float32)
    volume = numpy_volume if container == "numpy" else torch.from_numpy(numpy_volume).permute(3, 0, 1, 2)

    assert gaussian_blur3d(volume, sigma=(0.0, 0.0, 0.0)) is volume


@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_gaussian_blur3d_float64_falls_back_to_float32(container: str) -> None:
    """An unexpected float64 input follows the documented float32 working/output contract."""
    numpy_volume = np.random.default_rng(20260805).random((3, 5, 7, 3)).astype(np.float64)
    volume = numpy_volume if container == "numpy" else torch.from_numpy(numpy_volume).permute(3, 0, 1, 2)
    expected = gaussian_blur3d(numpy_volume.astype(np.float32), sigma=(0.75, 1.25, 0.0), kernel_size=(5, 0, 0))

    result = gaussian_blur3d(volume, sigma=(0.75, 1.25, 0.0), kernel_size=(5, 0, 0))

    if container == "numpy":
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)
    else:
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        np.testing.assert_array_equal(result.permute(1, 2, 3, 0).numpy(), expected)


@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_separable_filter3d_identity_converts_float64_to_float32(container: str) -> None:
    """The no-op fast path has the same float64 fallback as a non-identity filter."""
    numpy_volume = np.random.default_rng(20260805).random((3, 5, 7, 3)).astype(np.float64)
    volume = numpy_volume if container == "numpy" else torch.from_numpy(numpy_volume).permute(3, 0, 1, 2)
    identity = np.ones(1, dtype=np.float32)

    result = separable_filter3d(volume, (identity, identity, identity))

    if container == "numpy":
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, numpy_volume.astype(np.float32))
    else:
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        np.testing.assert_array_equal(result.permute(1, 2, 3, 0).numpy(), numpy_volume.astype(np.float32))


@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_separable_filter3d_converts_float64_kernels_to_float32(container: str) -> None:
    """Direct custom NumPy kernels cannot mismatch the float32 Torch working volume."""
    numpy_volume = np.random.default_rng(20260805).random((3, 5, 7, 3), dtype=np.float32)
    volume = numpy_volume if container == "numpy" else torch.from_numpy(numpy_volume).permute(3, 0, 1, 2)
    kernel = np.array((1.0, 2.0, 1.0)) / 4.0
    kernels = kernel, kernel, kernel
    expected = separable_filter3d(numpy_volume, tuple(item.astype(np.float32) for item in kernels))

    result = separable_filter3d(volume, kernels)

    if container == "numpy":
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)
    else:
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        np.testing.assert_array_equal(result.permute(1, 2, 3, 0).numpy(), expected)


def test_separable_filter3d_restores_uint8_once_after_all_passes() -> None:
    """The uint8 path clips only the final float32 result and keeps all input channels."""
    volume = np.full((3, 5, 7, 5), 255, dtype=np.uint8)
    kernel = np.array((2.0,), dtype=np.float32)
    kernels = kernel, kernel, kernel

    result = separable_filter3d(volume, kernels)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, 255)


def test_separable_filter3d_does_not_clip_uint8_between_passes() -> None:
    """A value above 255 between passes is restored only after the final pass."""
    volume = np.full((3, 5, 7, 1), 200, dtype=np.uint8)
    kernels = (
        np.array((2.0,), dtype=np.float32),
        np.array((0.5,), dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )

    result = separable_filter3d(volume, kernels)

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, 200)
