"""Separable true-3D filtering for one NumPy or CPU Torch volume."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as torch_f

from albucore.sampling3d import _restore_uint8
from albucore.tensor_bridge import numpy_to_tensor_layout, tensor_to_numpy_channel_last
from albucore.utils import _LINUX_X86_64, get_opencv_max_channels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from albucore._types import ImageArrayT

__all__ = ["gaussian_blur3d"]


_AXES = (2, 3, 4)
_OPENCV_MAX_CHANNELS = get_opencv_max_channels()
_IDENTITY_KERNEL = np.ones(1, dtype=np.float32)


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
    """Pad one spatial axis of the internal batched Tensor."""
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
    """Apply one same-shape grouped 1D correlation along an internal spatial axis."""
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


def _separable_filter3d_torch_cpu(
    volume: torch.Tensor,
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> torch.Tensor:
    """Run the measured three-pass CPU Torch filter for one prevalidated ``CDHW`` volume."""
    working = volume if volume.dtype == torch.float32 else volume.to(torch.float32)
    result = working.unsqueeze(0)
    with torch.no_grad():
        for axis, kernel in zip(_AXES, kernels, strict=True):
            if not _is_identity_kernel(kernel):
                result = _apply_axis_filter(result, kernel, axis)
        if volume.dtype == torch.uint8:
            result = _restore_uint8(result)
        elif volume.dtype == torch.float16:
            result = result.to(torch.float16)
    return result.squeeze(0)


def _separable_filter3d_numpy(volume: np.ndarray, kernels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Bridge one ``DHWC`` NumPy volume through the CPU Torch filter without hidden device transfer."""
    tensor = numpy_to_tensor_layout(volume, "CDHW")
    result = _separable_filter3d_torch_cpu(tensor, kernels)
    return tensor_to_numpy_channel_last(result, "CDHW")


def _separable_filter3d_numpy_opencv_xy_torch_depth(
    volume: np.ndarray,
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    depth, height, width, channels = volume.shape
    working = volume.astype(np.float32, copy=False)
    packed = np.ascontiguousarray(working.transpose(1, 2, 0, 3).reshape(height, width, depth * channels))
    if np.shares_memory(packed, volume):
        packed = packed.copy()
    del working
    cv2.sepFilter2D(
        packed,
        ddepth=-1,
        kernelX=kernels[2],
        kernelY=kernels[1],
        dst=packed,
        borderType=cv2.BORDER_REFLECT_101,
    )
    filtered = packed.reshape(height, width, depth, channels).transpose(2, 0, 1, 3)
    tensor = torch.from_numpy(filtered).permute(3, 0, 1, 2)
    result = _separable_filter3d_torch_cpu(tensor, (kernels[0], _IDENTITY_KERNEL, _IDENTITY_KERNEL))
    if volume.dtype == np.uint8:
        result = _restore_uint8(result)
    return result.permute(1, 2, 3, 0).numpy()


def _separable_filter3d(
    volume: np.ndarray | torch.Tensor,
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray | torch.Tensor:
    """Apply float32 ``(depth, height, width)`` kernels; three exact identities return ``volume``."""
    if all(_is_identity_kernel(kernel) for kernel in kernels):
        return volume
    if isinstance(volume, np.ndarray):
        return _separable_filter3d_numpy(volume, kernels)
    return _separable_filter3d_torch_cpu(volume, kernels)


@overload
def gaussian_blur3d(
    volume: ImageArrayT,
    sigma: float | Sequence[float],
    kernel_size: int | Sequence[int] = 0,
) -> ImageArrayT: ...


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
    kernel convention; explicit sizes must be positive and odd.

    NumPy uses channel-last ``(D, H, W, C)`` layout; Torch uses channel-first ``(C, D, H, W)``.
    Borders use OpenCV-compatible ``BORDER_REFLECT_101`` padding. The filter runs in float32 and
    restores uint8 once after all three passes. It preserves container, layout, channels, and the
    ``uint8``, ``float16``, or ``float32`` dtype. Zero sigma on every axis returns ``volume`` itself.

    Callers own rank, layout, CPU-device, strided-layout, and autograd validation. Target-level masks
    are intentionally outside this single-volume primitive.
    """
    sigmas = _three_axis_values(sigma)
    kernel_sizes = _three_axis_kernel_sizes(kernel_size)
    kernels = (
        _gaussian_kernel_1d(sigmas[0], kernel_sizes[0]),
        _gaussian_kernel_1d(sigmas[1], kernel_sizes[1]),
        _gaussian_kernel_1d(sigmas[2], kernel_sizes[2]),
    )
    if _LINUX_X86_64 and all(kernel.size > 1 for kernel in kernels):
        if (
            isinstance(volume, np.ndarray)
            and volume.dtype in (np.uint8, np.float32)
            and volume.flags.c_contiguous
            and volume.shape[-1] in (1, 3, 5, 9)
            and volume.shape[0] * volume.shape[-1] <= _OPENCV_MAX_CHANNELS
        ):
            return _separable_filter3d_numpy_opencv_xy_torch_depth(volume, kernels)
        if (
            isinstance(volume, torch.Tensor)
            and volume.dtype in (torch.uint8, torch.float32)
            and volume.shape[0] * volume.shape[1] <= _OPENCV_MAX_CHANNELS
        ):
            channel_last = volume.permute(1, 2, 3, 0)
            contiguous_cdhw = volume.is_contiguous() and volume.shape[0] in (1, 3, 5)
            contiguous_dhwc = (
                channel_last.is_contiguous()
                and volume.shape[0] in (1, 3, 5, 9)
                and (volume.dtype == torch.uint8 or volume.shape[0] in (1, 3, 9))
            )
            if contiguous_cdhw or contiguous_dhwc:
                result = _separable_filter3d_numpy_opencv_xy_torch_depth(channel_last.numpy(), kernels)
                return torch.from_numpy(result).permute(3, 0, 1, 2).contiguous()
    return _separable_filter3d(volume, kernels)
