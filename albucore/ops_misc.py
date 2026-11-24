"""Flips, median blur, matmul, pairwise distances."""

from typing import Any, cast, overload

import cv2
import numkong as nk
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import ArrayT, ImageArrayT, TensorImageLayout
from albucore.convert import convert_dtype
from albucore.tensor_bridge import dispatch_tensor_image
from albucore.utils import (
    _LINUX_X86_64,
    MAX_OPENCV_WORKING_CHANNELS,
    get_num_channels,
)
from albucore.utils import _MACOS_ARM64 as _IS_MACOS_ARM64

_MACOS_ACCELERATE_MATMUL_WARNING_MIN_DIMENSION = 16
_MEDIAN_BLUR_FLOAT16_ORDERED_U16_MIN_ELEMENTS = 76_800
_FLOAT16_INFINITY_BITS = 0x7C00
_LINUX_MEDIAN_BLUR_FLOAT16_TENSOR_PLANES_MAX_ELEMENTS = 8_388_608
_NUMKONG_CDIST_D8_MAX_PAIRS = 2_048


@overload
def hflip(img: ImageArrayT) -> ImageArrayT: ...


@overload
def hflip(img: torch.Tensor) -> torch.Tensor: ...


def hflip(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Flip image horizontally (mirror left-right).

    Routing:
    - All channel counts: NumPy slice view ``img[..., ::-1, :]``.

    Args:
        img: ``(H, W, C)`` image, any dtype.

    Returns:
        Horizontally flipped image, same shape and dtype.
    """
    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float16 and img.is_contiguous():
            return torch.flip(img.view(torch.int16), dims=(-1,)).view(torch.float16)
        return torch.flip(img, dims=(-1,))
    return img[..., ::-1, :]


@overload
def vflip(img: ImageArrayT) -> ImageArrayT: ...


@overload
def vflip(img: torch.Tensor) -> torch.Tensor: ...


def vflip(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Flip image vertically (mirror top-bottom).

    Routing:
    - All channel counts: NumPy slice view ``img[..., ::-1, :, :]``.

    Args:
        img: ``(H, W, C)`` image, any dtype.

    Returns:
        Vertically flipped image, same shape and dtype.
    """
    if isinstance(img, torch.Tensor):
        return torch.flip(img, dims=(-2,))
    return img[..., ::-1, :, :]


def _median_blur_uint8(
    img: NDArray[np.uint8],
    ksize: int,
    dst: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Apply OpenCV directly, or per plane where kernels of 7+ do not support the channel count."""
    if ksize in (3, 5) or get_num_channels(img) <= MAX_OPENCV_WORKING_CHANNELS:
        if img.shape[-1] == 1 and dst is None:
            dst = np.empty_like(img, order="C")
        return cast("NDArray[np.uint8]", cv2.medianBlur(img, ksize, dst=dst))
    result = np.empty_like(img, order="C") if dst is None else dst
    for channel, plane in enumerate(cv2.split(img)):
        result[..., channel] = cv2.medianBlur(plane, ksize)
    return result


def _median_blur_float16_single_channel_5(img: NDArray[np.float16]) -> NDArray[np.float16]:
    """Use ordered CV_16U keys while preserving float32 NaN and signed-zero behavior."""
    bits = img.view(np.uint16)
    # The ordered keys distinguish signed zeros, while the float32 median treats them as equal.
    if np.isnan(img).any() or (np.any(bits == 0) and np.any(bits == np.uint16(0x8000))):
        working = _cast_float_numpy(img, np.float32)
        return _cast_float_numpy(
            cast("NDArray[np.float32]", cv2.medianBlur(working, 5, dst=np.empty_like(working, order="C"))),
            np.float16,
        )

    negative = (bits & np.uint16(0x8000)) != 0
    ordered = np.empty_like(bits)
    np.bitwise_xor(bits, np.uint16(0x8000), out=ordered)
    np.bitwise_not(bits, out=ordered, where=negative)

    filtered = cast(
        "np.ndarray[Any, np.dtype[np.uint16]]",
        cv2.medianBlur(ordered, 5, dst=np.empty_like(ordered, order="C")),
    )
    positive = (filtered & np.uint16(0x8000)) != 0
    result = np.empty_like(filtered)
    np.bitwise_xor(filtered, np.uint16(0x8000), out=result)
    np.bitwise_not(filtered, out=result, where=~positive)
    return cast("NDArray[np.float16]", result.view(np.float16))


def _median_blur_float16_nonnegative(img: NDArray[Any], ksize: int) -> NDArray[np.float16] | None:
    """Filter non-negative finite or infinite float16 data as CV_16U; None selects another route."""
    if img.dtype != np.float16 or ksize not in (3, 5):
        return None
    bits = img.view(np.uint16)
    # Such half bit patterns sort like their values, so OpenCV selects the same elements as the float32 route.
    if bits.max() > _FLOAT16_INFINITY_BITS:
        return None
    destination = np.empty_like(bits) if img.shape[-1] == 1 else None
    return cast("NDArray[np.float16]", cv2.medianBlur(bits, ksize, dst=destination).view(np.float16))


def _use_median_blur_tensor_planes(img: torch.Tensor, ksize: int) -> bool:
    """Select per-plane OpenCV for contiguous Tensors, where it beats the channel-last bridge."""
    if not img.is_contiguous():
        return False
    if img.dtype == torch.uint8:
        return True
    if ksize not in (3, 5) or img.shape[-3] < 3:
        return False
    if _LINUX_X86_64 and img.dtype == torch.float16:
        return img.numel() <= _LINUX_MEDIAN_BLUR_FLOAT16_TENSOR_PLANES_MAX_ELEMENTS
    return True


def _median_blur_tensor_planes(img: torch.Tensor, ksize: int) -> torch.Tensor:
    """Filter each channel plane of a contiguous Tensor without repacking; float16 filters as float32."""
    source = img.to(torch.float32) if img.dtype == torch.float16 else img
    planes = source.numpy().reshape(-1, img.shape[-2], img.shape[-1])
    filtered = np.empty_like(planes)
    for index, plane in enumerate(planes):
        cv2.medianBlur(plane, ksize, dst=filtered[index])
    result = torch.from_numpy(filtered.reshape(img.shape))
    return result.to(torch.float16) if img.dtype == torch.float16 else result


@overload
def median_blur(img: ImageArrayT, ksize: int, *, layout: TensorImageLayout | None = None) -> ImageArrayT: ...


@overload
def median_blur(img: torch.Tensor, ksize: int, *, layout: TensorImageLayout | None = None) -> torch.Tensor: ...


def median_blur(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor,
    ksize: int,
    *,
    layout: TensorImageLayout | None = None,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor:
    """Median blur with dtype- and kernel-aware OpenCV routing.

    uint8 images use OpenCV directly; kernels 7 and larger filter each plane when C > 4, and contiguous uint8 Tensors
    filter their channel planes without repacking. Float32 kernels 3 and 5
    use OpenCV on the original values; larger float32 kernels use a documented float32-to-uint8 fallback because
    OpenCV supports native CV_32F median filtering only for apertures 3 and 5.

    Args:
        img: ``(H, W, C)`` NumPy image or channel-first CPU Tensor with uint8, float16, or float32 dtype.
        ksize: Kernel size (odd: 3, 5, 7, 9, ...).
        layout: Required for Tensor input; selects CHW or NCHW.

    Returns:
        Median-filtered image with the same shape and dtype. It does not share storage with the input.

    The caller validates the image dtype and kernel size before entering this low-level router.
    """
    if isinstance(img, torch.Tensor):
        if _use_median_blur_tensor_planes(img, ksize):
            # Channel-first planes are contiguous, so they need no interleaving copy.
            return _median_blur_tensor_planes(img, ksize)
        return dispatch_tensor_image(img, layout, median_blur, ksize)

    if img.dtype == np.uint8:
        # OpenCV reads row-padded uint8 crops directly and copies other strides itself.
        return _median_blur_uint8(cast("NDArray[np.uint8]", img), ksize)

    if not img.flags["C_CONTIGUOUS"]:
        img = cast("NDArray[np.float16] | NDArray[np.float32]", np.ascontiguousarray(img))

    result: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]
    nonnegative_result = _median_blur_float16_nonnegative(img, ksize)
    if nonnegative_result is not None:
        return nonnegative_result
    if img.dtype == np.float32 and ksize in (3, 5):
        destination = np.empty_like(img, order="C") if img.shape[-1] == 1 else None
        result = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            cv2.medianBlur(img, ksize, dst=destination),
        )
    elif (
        img.dtype == np.float16
        and ksize == 5
        and img.shape[-1] == 1
        and img.size >= _MEDIAN_BLUR_FLOAT16_ORDERED_U16_MIN_ELEMENTS
    ):
        result = _median_blur_float16_single_channel_5(cast("NDArray[np.float16]", img))
    elif img.dtype == np.float16 and ksize in (3, 5):
        working = _cast_float_numpy(img, np.float32)
        destination = np.empty_like(working, order="C") if img.shape[-1] == 1 else None
        result = cast("NDArray[np.float32]", cv2.medianBlur(working, ksize, dst=destination))
        result = _cast_float_numpy(result, np.float16)
    elif img.dtype == np.float16:
        quantized = convert_dtype(_cast_float_numpy(img, np.float32), np.dtype(np.uint8))
        filtered = convert_dtype(_median_blur_uint8(quantized, ksize), np.float32)
        result = _cast_float_numpy(filtered, np.float16)
    else:
        uint8_img = convert_dtype(cast("NDArray[np.float32]", img), np.dtype(np.uint8))
        result = convert_dtype(_median_blur_uint8(uint8_img, ksize), np.float32)
    return result


@overload
def matmul(a: ArrayT, b: np.ndarray) -> ArrayT: ...


@overload
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


def matmul(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Multiply NumPy matrices or CPU Tensors, preserving the input container.

    Tensor inputs must be on CPU with ``requires_grad=False``. Callers validate
    these preconditions before dispatch.

    Args:
        a: First matrix, shape (M, K), dtype float16, float32, float64, or uint8.
        b: Second matrix, shape (K, N), with the same container as ``a``.

    Returns:
        A matrix of shape (M, N). Same-dtype operands preserve dtype; uint8
        multiplication follows NumPy's modulo overflow. Float16 operands compute
        in float32 and round once to float16.

    Notes:
        On macOS arm64, large floating-point matrices use a narrowly scoped
        workaround for spurious divide, overflow, and invalid warnings from
        NumPy's Accelerate kernels. Computed values are returned unchanged.

    Examples:
        >>> matmul(np.ones((2, 3), dtype=np.float32), np.ones((3, 4), dtype=np.float32)).shape
        (2, 4)
    """
    if isinstance(a, torch.Tensor):
        return torch.from_numpy(matmul(a.numpy(), cast("torch.Tensor", b).numpy()))
    b = cast("np.ndarray", b)

    if a.dtype == np.float16:
        result = matmul(_cast_float_numpy(a, np.float32), _cast_float_numpy(b, np.float32))
        return _cast_float_numpy(result, np.float16)

    if a.dtype == np.uint8 and b.dtype == np.uint8:
        # BLAS sums stay exact: float32 holds every partial sum below 2**24, float64 below 2**53.
        working = np.float32 if a.shape[-1] * 255 * 255 < 2**24 else np.float64
        product = a.astype(working) @ b.astype(working)
        return cast("NDArray[np.uint8]", product.astype(np.int64).astype(np.uint8))

    if (
        _IS_MACOS_ARM64
        and a.dtype.kind == "f"
        and b.dtype.kind == "f"
        and max((*a.shape, *b.shape), default=0) >= _MACOS_ACCELERATE_MATMUL_WARNING_MIN_DIMENSION
    ):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return cast("np.ndarray", a @ b)

    return cast("np.ndarray", a @ b)


@overload
def pairwise_distances_squared(points1: np.ndarray, points2: np.ndarray) -> NDArray[np.float32]: ...


@overload
def pairwise_distances_squared(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor: ...


def pairwise_distances_squared(
    points1: np.ndarray | torch.Tensor,
    points2: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Compute squared pairwise Euclidean distances between two point sets.

    Backend selection:
    - NumKong ``cdist`` handles small sets and the measured high-dimensional regions.
    - On macOS ARM64, contiguous float32 eight-dimensional inputs use NumKong through 2,048 output pairs.
    - Other sets: NumPy coordinate differences accumulated into one distance matrix.

    Args:
        points1: First set of points, shape (N, D), dtype float32.
        points2: Second set of points, shape (M, D), dtype float32.

    Returns:
        Float32 matrix of shape (N, M) where element [i, j] is ||points1[i] - points2[j]||².

    Examples:
        >>> import numpy as np
        >>> from albucore import pairwise_distances_squared
        >>> src_points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        >>> dst_points = np.array([[0.1, 0.1], [0.9, 0.1]], dtype=np.float32)
        >>> distances_sq = pairwise_distances_squared(src_points, dst_points)
        >>> distances_sq.shape
        (3, 2)

    Note:
        The caller validates matching coordinate counts. NumPy arithmetic follows
        the caller's floating-point error settings.

        Subtracting coordinates before squaring avoids overflow and cancellation
        from large common offsets. Squared distances exceeding the float32 range
        are returned as infinity. The NumPy route uses one scratch matrix of
        shape (N, M), rather than a temporary of shape (N, M, D).
    """
    tensor_output = isinstance(points1, torch.Tensor)
    points1_array = points1.numpy() if isinstance(points1, torch.Tensor) else points1
    points2_array = points2.numpy() if isinstance(points2, torch.Tensor) else points2

    numkong_d8_eligible = (
        points1_array.dtype == np.float32
        and points2_array.dtype == np.float32
        and points1_array.flags.c_contiguous
        and points2_array.flags.c_contiguous
    )
    points1_array = points1_array.astype(np.float32, copy=False)
    points2_array = points2_array.astype(np.float32, copy=False)

    n1, n2 = points1_array.shape[0], points2_array.shape[0]
    dimensions = points1_array.shape[1]
    if dimensions == 0:
        result = np.zeros((n1, n2), dtype=np.float32)
    elif (
        n1 * n2 < 1000
        or dimensions >= 9
        or (numkong_d8_eligible and dimensions == 8 and n1 * n2 <= _NUMKONG_CDIST_D8_MAX_PAIRS)
    ):
        # nk.cdist requires contiguous inputs on some strided views.
        points1_array = np.ascontiguousarray(points1_array)
        points2_array = np.ascontiguousarray(points2_array)
        result = np.asarray(nk.cdist(points1_array, points2_array, metric="sqeuclidean"), dtype=np.float32)
    else:
        result = np.subtract(points1_array[:, 0, None], points2_array[None, :, 0])
        np.square(result, out=result)
        if dimensions > 1:
            scratch = np.empty_like(result)
            for coordinate in range(1, dimensions):
                np.subtract(points1_array[:, coordinate, None], points2_array[None, :, coordinate], out=scratch)
                np.square(scratch, out=scratch)
                np.add(result, scratch, out=result)
    return torch.from_numpy(result) if tensor_output else result
