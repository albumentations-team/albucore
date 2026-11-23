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
_IDENTITY_KERNEL_MIN_ELEMENTS_HALF = 20_480
_IDENTITY_KERNEL_MIN_ELEMENTS_OTHER = 30_720
_SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS = {
    (3, 7): 921_600,
    (3, 15): 230_400,
    (9, 7): 2_764_800,
    (9, 15): 2_764_800,
}

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


def _is_center_only_kernel(kernel: np.ndarray) -> bool:
    center = kernel.shape[0] // 2, kernel.shape[1] // 2
    return bool(kernel[center] != 0) and np.count_nonzero(kernel) == 1


def _identity_morphology_copy(
    image: ImageArrayT,
    kernel: NDArray[Any],
) -> ImageArrayT | None:
    if kernel.size == 1:
        if image.dtype.type is np.float32 or kernel[0, 0] == 0:
            return None
        return cast("ImageArrayT", image.copy(order="C"))
    min_elements = (
        _IDENTITY_KERNEL_MIN_ELEMENTS_HALF if image.dtype.type is np.float16 else _IDENTITY_KERNEL_MIN_ELEMENTS_OTHER
    )
    if image.size < min_elements or not _is_center_only_kernel(kernel):
        return None
    return cast("ImageArrayT", image.copy(order="C"))


def _separable_float32_kernel(kernel: NDArray[Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if kernel.dtype != np.float32 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] not in (7, 15):
        return None
    if not np.isfinite(kernel).all() or np.any(kernel < 0) or not np.isclose(np.sum(kernel, dtype=np.float64), 1.0):
        return None
    row, col = np.unravel_index(np.argmax(np.abs(kernel)), kernel.shape)
    pivot = kernel[row, col]
    kernel_x = kernel[row].copy()
    kernel_y = (kernel[:, col] / pivot).astype(np.float32)
    if np.allclose(kernel_y[:, None] * kernel_x[None, :], kernel, rtol=1e-6, atol=1e-7):
        return kernel_x, kernel_y
    return None


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
        if (
            input_dtype.type is np.float32
            and (_LINUX_X86_64 or _MACOS_ARM64)
            and img.flags.c_contiguous
            and kernel.shape[0] == kernel.shape[1]
            and (img.shape[-1], kernel.shape[0]) in _SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS
            and img.size >= _SEPARABLE_FLOAT32_CONVOLVE_MIN_ELEMENTS[(img.shape[-1], kernel.shape[0])]
            and border_mode == cv2.BORDER_DEFAULT
        ):
            factors = _separable_float32_kernel(kernel)
            if factors is not None and cv2.checkRange(img)[0]:
                cv2.sepFilter2D(
                    working,
                    ddepth=-1,
                    kernelX=factors[0],
                    kernelY=factors[1],
                    dst=result,
                    borderType=border_mode,
                )
                return cast("ImageArrayT", clip(result, out=result))
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
