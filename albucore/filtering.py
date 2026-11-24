"""Dtype-preserving OpenCV image filters, morphology and RGB transforms.

Callers validate metadata and operation controls. Kernels own float16 promotion,
layout packing and writable workspaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np

from albucore.convert import convert_dtype
from albucore.utils import _LINUX_X86_64, _MACOS_ARM64, clip

_MACOS_FLOAT32_BOX_BLUR_MIN_ELEMENTS = 20_480
_LINUX_FLOAT32_BOX_BLUR_MIN_ELEMENTS = 230_400
_SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS = {
    (3, 7): 921_600,
    (9, 7): 2_764_800,
}
_SEPARABLE_FLOAT32_CONVOLVE_MIN_KSIZE = 9
_KERNEL_SUM_TOLERANCE = 1e-8 + 1e-5
_KERNEL_FACTOR_ATOL = 1e-7
_KERNEL_FACTOR_RTOL = 1e-6
# Bound on the summed reconstruction error relative to sum(|K|); per-coefficient tolerances alone let many
# small errors accumulate into a materially different kernel.
_KERNEL_FACTOR_SUM_ERROR_EPS_MULTIPLE = 4 * float(np.finfo(np.float32).eps)
_MACOS_SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS = 5_120
# (smallest odd kernel size, minimum elements); sizes between measured kernels take the next smaller entry.
_LINUX_SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS = (
    (21, 5_120),
    (15, 12_288),
    (11, 20_480),
    (9, 46_080),
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from albucore._types import ImageArrayT


def _opencv_working_image(
    image: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
    *,
    copy: bool,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]:
    if image.dtype.type is np.float16:
        return convert_dtype(
            image,
            np.float32,
            out=None if image.flags.c_contiguous else np.empty(image.shape, dtype=np.float32),
        )
    return image.copy(order="C") if copy else image


def _identity_morphology_copy(
    image: ImageArrayT,
    kernel: NDArray[Any],
) -> ImageArrayT | None:
    if kernel.size == 1:
        identity = kernel[0, 0] != 0
    else:
        identity = np.count_nonzero(kernel) == 1 and kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] != 0
    return cast("ImageArrayT", image.copy(order="C")) if identity else None


def _separable_float32_min_elements(channels: int, ksize: int) -> int | None:
    """Smallest image size where sepFilter2D beats filter2D for an odd square kernel, per measured platform."""
    if ksize == 7:
        return _SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS.get((channels, ksize))
    if ksize < _SEPARABLE_FLOAT32_CONVOLVE_MIN_KSIZE or ksize % 2 == 0:
        return None
    if _MACOS_ARM64:
        return _MACOS_SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS
    return next(
        (elements for min_ksize, elements in _LINUX_SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS if ksize >= min_ksize),
        None,
    )


def _two_by_two_minors_allow_rank_one(kernel: NDArray[Any]) -> bool:
    """Reject kernels that are clearly not rank one from two 2x2 minors, in about 2 us instead of 25-33 us.

    A rank-one kernel has vanishing minors. A kernel that passes the full factorization differs from one by at
    most the summed-residual bound per coefficient (plus float32 product rounding), so each minor stays within
    ``4 * kmax * e + 4 * e**2`` with ``e`` twice the summed-error bound. The corner minor rejects dense kernels and
    the (1, centre) minor rejects disk-shaped kernels whose corners are zero. NaN never rejects.
    """
    rows, cols = kernel.shape
    error = 2 * _KERNEL_FACTOR_SUM_ERROR_EPS_MULTIPLE
    for row0, row1, col0, col1 in ((0, rows - 1, 0, cols - 1), (1, rows // 2, 1, cols // 2)):
        a, b = kernel.item(row0, col0), kernel.item(row0, col1)
        c, d = kernel.item(row1, col0), kernel.item(row1, col1)
        kmax = max(abs(a), abs(b), abs(c), abs(d))
        if abs(a * d - b * c) > 4 * error * (kmax + error):
            return False
    return True


def _separable_float32_kernel(kernel: NDArray[Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if kernel.dtype != np.float32 or not _two_by_two_minors_allow_rank_one(kernel):
        return None
    # Same tolerances as np.isclose/np.allclose without their per-call overhead.
    total = float(np.sum(kernel, dtype=np.float64))
    if not np.isfinite(kernel).all() or kernel.min() < 0 or abs(total - 1.0) > _KERNEL_SUM_TOLERANCE:
        return None
    row, col = np.unravel_index(kernel.argmax(), kernel.shape)
    pivot = kernel[row, col]
    kernel_x = kernel[row].copy()
    kernel_y = (kernel[:, col] / pivot).astype(np.float32)
    residual = np.abs(kernel_y[:, None] * kernel_x[None, :] - kernel)
    if (
        np.all(residual <= _KERNEL_FACTOR_ATOL + _KERNEL_FACTOR_RTOL * np.abs(kernel))
        and float(residual.sum(dtype=np.float64)) <= _KERNEL_FACTOR_SUM_ERROR_EPS_MULTIPLE * total
    ):
        return kernel_x, kernel_y
    return None


def _convolve_separable_float32(
    img: ImageArrayT,
    kernel: NDArray[Any],
    working: NDArray[Any],
    result: NDArray[Any],
    border_mode: int,
) -> ImageArrayT | None:
    """Filter a finite float32 image with sepFilter2D when its separable kernel is profitable; else None."""
    if not (
        img.dtype.type is np.float32
        and (_LINUX_X86_64 or _MACOS_ARM64)
        and img.flags.c_contiguous
        and kernel.shape[0] == kernel.shape[1]
        and border_mode == cv2.BORDER_DEFAULT
    ):
        return None
    min_elements = _separable_float32_min_elements(img.shape[-1], kernel.shape[0])
    if min_elements is None or img.size < min_elements:
        return None
    factors = _separable_float32_kernel(kernel)
    if factors is None or not cv2.checkRange(img)[0]:
        return None
    cv2.sepFilter2D(working, ddepth=-1, kernelX=factors[0], kernelY=factors[1], dst=result, borderType=border_mode)
    return cast("ImageArrayT", clip(result, out=result))


def box_blur(img: ImageArrayT, ksize: int) -> ImageArrayT:
    """Blur an image with an OpenCV normalized box filter.

    Args:
        img: NumPy image with uint8, float16, or float32 dtype.
        ksize: Positive box kernel size.

    Returns:
        A writable result with the input shape and dtype. Input storage is unchanged.

    Images use HWC with an explicit trailing channel, including C=1.
    """
    if img.dtype == np.uint8 and ksize > 1:
        return cast("ImageArrayT", cv2.blur(img, (ksize, ksize), dst=np.empty_like(img, order="C")))
    if img.dtype == np.float32 and ksize > 1 and img.flags.c_contiguous:
        channels = img.shape[-1]
        macos_route = _MACOS_ARM64 and img.size >= _MACOS_FLOAT32_BOX_BLUR_MIN_ELEMENTS
        linux_route = _LINUX_X86_64 and channels > 1 and img.size >= _LINUX_FLOAT32_BOX_BLUR_MIN_ELEMENTS
        if macos_route or linux_route:
            source = cast("NDArray[np.float32]", img)
            blurred = np.empty(source.shape, dtype=np.float32)
            return cast("ImageArrayT", cv2.blur(source, (ksize, ksize), dst=blurred))
    result = _opencv_working_image(img, copy=True)
    cv2.blur(result, (ksize, ksize), dst=result)
    if img.dtype.type is np.float16:
        return cast("ImageArrayT", convert_dtype(result, cast("np.dtype[np.float16]", img.dtype)))
    return cast("ImageArrayT", result)


def erode(img: ImageArrayT, kernel: NDArray[Any]) -> ImageArrayT:
    """Apply one OpenCV erosion without changing the source.

    Args:
        img: NumPy image with uint8, float16, or float32 dtype.
        kernel: Prevalidated 2D structuring element.

    Returns:
        A writable result with the input shape and dtype. Input storage is unchanged.

    Images use HWC with an explicit trailing channel, including C=1.
    """
    identity_result = _identity_morphology_copy(img, kernel)
    if identity_result is not None:
        return identity_result
    if img.dtype.type is np.float16:
        working = _opencv_working_image(img, copy=False)
        cv2.erode(working, kernel, dst=working, iterations=1)
        return cast("ImageArrayT", convert_dtype(working, cast("np.dtype[np.float16]", img.dtype)))
    if img.shape[-1] == 1:
        destination = np.empty_like(img, order="C")
        return cast("ImageArrayT", cv2.erode(img, kernel, dst=destination, iterations=1))
    return cast("ImageArrayT", cv2.erode(img, kernel, iterations=1))


def dilate(img: ImageArrayT, kernel: NDArray[Any]) -> ImageArrayT:
    """Apply one OpenCV dilation without changing the source.

    Args:
        img: NumPy image with uint8, float16, or float32 dtype.
        kernel: Prevalidated 2D structuring element.

    Returns:
        A writable result with the input shape and dtype. Input storage is unchanged.

    Images use HWC with an explicit trailing channel, including C=1.
    """
    identity_result = _identity_morphology_copy(img, kernel)
    if identity_result is not None:
        return identity_result
    if img.dtype.type is np.float16:
        working = _opencv_working_image(img, copy=False)
        cv2.dilate(working, kernel, dst=working, iterations=1)
        return cast("ImageArrayT", convert_dtype(working, cast("np.dtype[np.float16]", img.dtype)))
    if img.shape[-1] == 1:
        destination = np.empty_like(img, order="C")
        return cast("ImageArrayT", cv2.dilate(img, kernel, dst=destination, iterations=1))
    return cast("ImageArrayT", cv2.dilate(img, kernel, iterations=1))


def convolve(img: ImageArrayT, kernel: NDArray[Any], border_mode: int = cv2.BORDER_DEFAULT) -> ImageArrayT:
    """Apply a shared or per-channel 2D convolution and clip the result.

    Args:
        img: NumPy image with uint8, float16, or float32 dtype.
        kernel: Prevalidated shared 2D or per-channel kernel.
        border_mode: OpenCV border mode.

    Returns:
        A writable result with the input shape and dtype. Input storage is unchanged.

    Images use HWC with an explicit trailing channel, including C=1.
    """
    input_dtype = img.dtype
    is_half = img.dtype.type is np.float16
    reuse_working = is_half or kernel.ndim == 3 or not img.flags.c_contiguous
    working = _opencv_working_image(img, copy=reuse_working)
    result = working if reuse_working else np.empty_like(working)
    if kernel.ndim == 2:
        separable_result = _convolve_separable_float32(img, kernel, working, result, border_mode)
        if separable_result is not None:
            return separable_result
        cv2.filter2D(working, ddepth=-1, kernel=kernel, dst=result, borderType=border_mode)
    else:
        for channel, channel_kernel in enumerate(kernel):
            result[..., channel] = cv2.filter2D(
                working[..., channel],
                ddepth=-1,
                kernel=channel_kernel,
                borderType=border_mode,
            )
    if is_half:
        return cast("ImageArrayT", clip(result, dtype=cast("np.dtype[np.float16]", input_dtype)))
    return cast("ImageArrayT", clip(result, out=result))


def linear_transformation_rgb(img: ImageArrayT, transformation_matrix: NDArray[Any]) -> ImageArrayT:
    """Apply one shared 3x3 RGB matrix to an image or image batch and clip the result.

    Args:
        img: NumPy image with uint8, float16, or float32 dtype.
        transformation_matrix: One shared RGB transformation matrix.

    Returns:
        A writable result with the input shape and dtype. Input storage is unchanged.

    RGB images use HWC or NHWC with one shared matrix.
    """
    input_dtype = img.dtype
    is_half = input_dtype.type is np.float16
    working = _opencv_working_image(img, copy=False)
    if working.ndim == 3:
        result = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            cv2.transform(working, transformation_matrix, dst=working if is_half else None),
        )
    else:
        original_shape = working.shape
        transformed = working.reshape(-1, working.shape[2], working.shape[3])
        transformed = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            cv2.transform(transformed, transformation_matrix, dst=transformed if is_half else None),
        )
        result = cast(
            "NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]",
            transformed.reshape(original_shape),
        )
    if is_half:
        return cast("ImageArrayT", clip(result, dtype=cast("np.dtype[np.float16]", input_dtype)))
    return cast("ImageArrayT", clip(result, out=result))


def unsharp_mask(image: ImageArrayT, ksize: int, sigma: float, alpha: float, threshold: int) -> ImageArrayT:
    """Sharpen an image with Gaussian residuals and a threshold mask.

    Args:
        image: NumPy image with uint8, float16, or float32 dtype.
        ksize: Positive box/Gaussian kernel size.
        sigma: Gaussian sigma.
        alpha: Sharpening strength.
        threshold: Residual threshold in byte intensity units.

    Returns:
        A writable result with the input shape and dtype. Input storage is unchanged.

    Images use HWC with an explicit trailing channel, including C=1.
    """
    input_dtype = image.dtype
    working = _opencv_working_image(image, copy=False)
    ksize_tuple = (ksize, ksize)
    blurred: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] = np.empty_like(working, order="C")
    cv2.GaussianBlur(working, ksize_tuple, sigmaX=sigma, dst=blurred)
    sharpened: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] = np.empty_like(working, order="C")
    cv2.addWeighted(working, 1.0 + alpha, blurred, -alpha, 0, dst=sharpened)
    if working.dtype == np.float32:
        np.clip(sharpened, 0, 1, out=sharpened)
    if threshold > 0:
        cv2.absdiff(working, blurred, dst=blurred)
        threshold_value = (
            np.array(threshold / 255.0, dtype=np.float32)
            if working.dtype == np.float32
            else np.array(min(threshold, 255), dtype=np.uint8)
        )
        mask = cv2.compare(blurred, threshold_value, cv2.CMP_GT)
        np.bitwise_not(mask, out=mask)
        cv2.copyTo(working, mask, sharpened)
    return cast("ImageArrayT", convert_dtype(sharpened, input_dtype) if input_dtype.type is np.float16 else sharpened)
