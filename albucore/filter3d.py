"""Separable true-3D filtering for one NumPy or CPU Torch volume."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import torch
import torch.nn.functional as torch_f

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["gaussian_blur3d", "separable_filter3d"]


_AXES = (2, 3, 4)


def _three_axis_values(value: float | Sequence[float]) -> tuple[float, float, float]:
    """Normalize caller-prevalidated scalar or explicit D/H/W control data."""
    if np.isscalar(value):
        scalar = float(cast("float", value))
        return scalar, scalar, scalar

    values = tuple(float(item) for item in cast("Sequence[float]", value))
    return values[0], values[1], values[2]


def _three_axis_kernel_sizes(value: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize caller-prevalidated 0-or-odd D/H/W kernel sizes."""
    if isinstance(value, (int, np.integer)):
        return int(value), int(value), int(value)

    values = tuple(value)
    return int(values[0]), int(values[1]), int(values[2])


def _gaussian_kernel_1d(sigma: float, kernel_size: int) -> np.ndarray:
    """Build one float32 Gaussian kernel, using Albumentations' 3.5-sigma automatic radius."""
    if sigma == 0.0:
        return np.ones(1, dtype=np.float32)

    size = int(sigma * 3.5) * 2 + 1 if kernel_size == 0 else kernel_size

    coordinates = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    kernel = np.exp(np.float32(-0.5) * (coordinates / np.float32(sigma)) ** 2)
    return cast("np.ndarray", kernel / np.sum(kernel, dtype=np.float32))


def _is_identity_kernel(kernel: np.ndarray) -> bool:
    """Recognize only the exact one-element identity, which is safe to alias."""
    return kernel.shape == (1,) and bool(kernel[0] == np.float32(1.0))


def _reflect101_indices(size: int, radius: int) -> torch.Tensor:
    """Generate universal OpenCV ``BORDER_REFLECT_101`` indices for one spatial axis."""
    if size == 1:
        return torch.zeros(size + 2 * radius, dtype=torch.long)

    coordinates = torch.arange(-radius, size + radius, dtype=torch.long)
    period = 2 * size - 2
    folded = torch.remainder(coordinates, period)
    return torch.where(folded < size, folded, period - folded)


def _pad_reflect101(volume: torch.Tensor, axis: int, radius: int) -> torch.Tensor:
    """Pad one ``NCDHW`` axis with Torch reflect mode or its singleton/large-radius equivalent."""
    if radius == 0:
        return volume

    if radius >= volume.shape[axis]:
        return volume.index_select(axis, _reflect101_indices(volume.shape[axis], radius))

    if axis == 2:
        padding = (0, 0, 0, 0, radius, radius)
    elif axis == 3:
        padding = (0, 0, radius, radius, 0, 0)
    else:
        padding = (radius, radius, 0, 0, 0, 0)
    return torch_f.pad(volume, padding, mode="reflect")


def _apply_axis_filter(volume: torch.Tensor, kernel: np.ndarray, axis: int) -> torch.Tensor:
    """Apply one same-shape grouped 1D correlation along one NCDHW spatial axis."""
    radius = kernel.size // 2
    padded = _pad_reflect101(volume, axis, radius)
    if axis == 2:
        shape = (1, 1, kernel.size, 1, 1)
    elif axis == 3:
        shape = (1, 1, 1, kernel.size, 1)
    else:
        shape = (1, 1, 1, 1, kernel.size)
    weights = torch.from_numpy(kernel).reshape(shape).expand(volume.shape[1], -1, -1, -1, -1)
    return torch_f.conv3d(padded, weights, groups=volume.shape[1])


def _restore_uint8(result: torch.Tensor) -> torch.Tensor:
    """Round and saturate the one final float32 filtering result."""
    return torch.clamp(result, 0.0, 255.0).add_(0.5).to(torch.uint8)


def _separable_filter3d_torch_cpu(
    volume: torch.Tensor,
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> torch.Tensor:
    """Run the measured three-pass CPU Torch filter for one prevalidated ``CDHW`` volume."""
    working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
    result = working.unsqueeze(0)
    with torch.inference_mode():
        for axis, kernel in zip(_AXES, kernels, strict=True):
            if not _is_identity_kernel(kernel):
                result = _apply_axis_filter(result, kernel, axis)
        if volume.dtype == torch.uint8:
            result = _restore_uint8(result)
    return result.squeeze(0)


def _separable_filter3d_numpy(volume: np.ndarray, kernels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Bridge one ``DHWC`` NumPy volume through the CPU Torch filter without hidden device transfer."""
    if not volume.flags.writeable or any(stride < 0 for stride in volume.strides):
        volume = np.array(volume, copy=True, order="C")
    tensor = torch.from_numpy(volume).permute(3, 0, 1, 2)
    result = _separable_filter3d_torch_cpu(tensor, kernels)
    return np.asarray(result.permute(1, 2, 3, 0).numpy())


def _identity_result(volume: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Keep documented no-op inputs by identity and match the float32 working-type fallback otherwise."""
    if isinstance(volume, np.ndarray):
        if volume.dtype in (np.uint8, np.float32):
            return volume
        return volume.astype(np.float32, copy=False)
    if volume.dtype in (torch.uint8, torch.float32):
        return volume
    return volume.to(torch.float32)


@overload
def separable_filter3d(volume: np.ndarray, kernels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray: ...


@overload
def separable_filter3d(volume: torch.Tensor, kernels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> torch.Tensor: ...


def separable_filter3d(
    volume: np.ndarray | torch.Tensor,
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray | torch.Tensor:
    """Apply separable D/H/W filtering to one NumPy or CPU Torch volume.

    NumPy uses channel-last ``(D, H, W, C)`` layout; Torch uses channel-first ``(C, D, H, W)``.
    The three prevalidated odd-length kernels are applied in ``(depth, height, width)`` order with
    OpenCV-compatible ``BORDER_REFLECT_101`` padding. The router runs in float32 and restores uint8
    once after all three passes. It preserves container, layout, and channels. Supported ``uint8``
    and ``float32`` input preserve their dtype; unexpected ``float64`` input is converted to and
    returned as ``float32``. Three exact one-element identity kernels return a supported input
    ``volume`` itself.

    Callers own rank, layout, CPU-device, strided-layout, and autograd validation. Batches,
    ``volumes``, and ``masks3d`` are intentionally outside this single-volume primitive.
    """
    if all(_is_identity_kernel(kernel) for kernel in kernels):
        return _identity_result(volume)
    if isinstance(volume, np.ndarray):
        return _separable_filter3d_numpy(volume, kernels)
    return _separable_filter3d_torch_cpu(volume, kernels)


@overload
def gaussian_blur3d(
    volume: np.ndarray,
    sigma: float | Sequence[float],
    kernel_size: int | Sequence[int] = 0,
) -> np.ndarray: ...


@overload
def gaussian_blur3d(
    volume: torch.Tensor,
    sigma: float | Sequence[float],
    kernel_size: int | Sequence[int] = 0,
) -> torch.Tensor: ...


def gaussian_blur3d(
    volume: np.ndarray | torch.Tensor,
    sigma: float | Sequence[float],
    kernel_size: int | Sequence[int] = 0,
) -> np.ndarray | torch.Tensor:
    """Blur one volume along D/H/W with scalar or anisotropic Gaussian sigma.

    ``sigma`` and ``kernel_size`` are scalar-isotropic or explicit ``(depth, height, width)``
    controls. A zero sigma skips that axis. A zero kernel-size selects
    ``int(sigma * 3.5) * 2 + 1`` for that axis, matching Albumentations' existing Gaussian
    kernel convention; explicit sizes must be positive and odd. See :func:`separable_filter3d`
    for container layouts, dtype support, borders, and the intentional single-volume scope.
    """
    sigmas = _three_axis_values(sigma)
    kernel_sizes = _three_axis_kernel_sizes(kernel_size)
    kernels = (
        _gaussian_kernel_1d(sigmas[0], kernel_sizes[0]),
        _gaussian_kernel_1d(sigmas[1], kernel_sizes[1]),
        _gaussian_kernel_1d(sigmas[2], kernel_sizes[2]),
    )
    return separable_filter3d(volume, kernels)
