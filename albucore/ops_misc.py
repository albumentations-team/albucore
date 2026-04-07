"""Flips, median blur, matmul, pairwise distances."""

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import cv2
import numkong as nk
import numpy as np

from albucore.convert import from_float, to_float
from albucore.decorators import contiguous, preserve_channel_dim
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    ImageFloat32,
    ImageType,
    get_num_channels,
    maybe_process_in_chunks,
)


@contiguous
def hflip_numpy(img: ImageType) -> ImageType:
    return img[:, ::-1, ...]


@preserve_channel_dim
def hflip_cv2(img: ImageType) -> ImageType:
    # OpenCV's flip function has a limitation of 512 channels
    if img.ndim > 2 and img.shape[2] > 512:
        return _flip_multichannel(img, flip_code=1)
    return cast("ImageType", cv2.flip(img, 1))


def hflip(img: ImageType) -> ImageType:
    """Flip image horizontally (mirror left-right).

    Routing:
    - All channel counts: ``cv2.flip(img, 1)``.  Images with > 512 channels are split
      into 512-channel chunks and concatenated (OpenCV limit).

    Alternative: ``hflip_numpy`` (NumPy slice ``img[:, ::-1, ...]``; useful when OpenCV is
    unavailable or for non-contiguous arrays).

    Args:
        img: ``(H, W, C)`` image or batch/volume, any dtype.

    Returns:
        Horizontally flipped image, same shape and dtype.
    """
    return hflip_cv2(img)


@preserve_channel_dim
def vflip_cv2(img: ImageType) -> ImageType:
    # OpenCV's flip function has a limitation of 512 channels
    if img.ndim > 2 and img.shape[2] > 512:
        return _flip_multichannel(img, flip_code=0)
    return cast("ImageType", cv2.flip(img, 0))


@contiguous
def vflip_numpy(img: ImageType) -> ImageType:
    return img[::-1, ...]


def vflip(img: ImageType) -> ImageType:
    """Flip image vertically (mirror top-bottom).

    Routing:
    - **C ≤ 4**: ``cv2.flip(img, 0)``.  Images with > 512 channels are chunked.
    - **C > 4**: NumPy slice ``img[::-1, ...]`` (faster than OpenCV on benchmarked shapes).

    Alternative: ``vflip_numpy`` or ``vflip_cv2`` for explicit backend selection.

    Args:
        img: ``(H, W, C)`` image or batch/volume, any dtype.

    Returns:
        Vertically flipped image, same shape and dtype.
    """
    if img.ndim >= 3 and get_num_channels(img) > MAX_OPENCV_WORKING_CHANNELS:
        return vflip_numpy(img)
    return vflip_cv2(img)


def _flip_multichannel(img: ImageType, flip_code: int) -> ImageType:
    """Process images with more than 512 channels by splitting into chunks.

    OpenCV's flip function has a limitation where it can only handle images with up to 512 channels.
    This function works around that limitation by splitting the image into chunks of 512 channels,
    flipping each chunk separately, and then concatenating the results.

    Args:
        img: Input image with many channels
        flip_code: OpenCV flip code (0 for vertical, 1 for horizontal, -1 for both)

    Returns:
        Flipped image with all channels preserved
    """
    # Get image dimensions
    num_channels = img.shape[2]

    # If the image has fewer than 512 channels, use cv2.flip directly
    if num_channels <= 512:
        return cast("ImageType", cv2.flip(img, flip_code))

    chunk_size = 512
    result: np.ndarray | None = None
    offset = 0

    for i in range(0, num_channels, chunk_size):
        end_idx = min(i + chunk_size, num_channels)
        chunk = img[:, :, i:end_idx]
        flipped_chunk = cast("np.ndarray", cv2.flip(chunk, flip_code))

        if flipped_chunk.ndim == 2 and img.ndim == 3:
            flipped_chunk = np.expand_dims(flipped_chunk, axis=2)

        if result is None:
            result = np.empty((*flipped_chunk.shape[:2], num_channels), dtype=img.dtype)
        n = flipped_chunk.shape[-1]
        result[:, :, offset : offset + n] = flipped_chunk
        offset += n

    if result is None:
        msg = "Flip chunking produced no output."
        raise RuntimeError(msg)
    return cast("ImageType", result)


def float32_io(func: Callable[..., ImageType]) -> Callable[..., ImageType]:
    """Decorator: transparently cast input to float32, then cast output back to the original dtype.

    Wraps a function that expects float32 so it accepts any dtype.  If the input is already
    float32, no copy is made.  The output is converted back with ``from_float`` (which rounds
    and clips to the target dtype range).

    Typical use: wrap float-only OpenCV operations (e.g. geometric transforms) so they work
    with uint8 images without manual conversion at every call site.

    Example::

        @float32_io
        def warp(img: np.ndarray, M: np.ndarray) -> np.ndarray:
            return cv2.warpAffine(img, M, ...)

    Args:
        func: Image processing function whose first argument is an ``(H, W, C)`` float32 image.

    Returns:
        Wrapped function that accepts any dtype and returns the original dtype.
    """

    @wraps(func)
    def float32_wrapper(img: ImageType, *args: Any, **kwargs: Any) -> ImageType:
        input_dtype = img.dtype
        if input_dtype != np.float32:
            img = to_float(img)
        result = func(img, *args, **kwargs)

        if input_dtype != np.float32:
            return from_float(cast("ImageFloat32", result), target_dtype=input_dtype)
        return result

    return float32_wrapper


def uint8_io(func: Callable[..., ImageType]) -> Callable[..., ImageType]:
    """Decorator: transparently cast input to uint8, then cast output back to the original dtype.

    Wraps a function that only works with uint8 (e.g. ``cv2.medianBlur``) so it transparently
    handles float32 input.  Float32 is converted to uint8 via ``from_float`` (scales x 255,
    rounds, clips); the uint8 result is then converted back to float32 via ``to_float``.

    Example::

        @uint8_io
        def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
            return cv2.medianBlur(img, ksize)

    Args:
        func: Image processing function whose first argument is a uint8 ``(H, W, C)`` image.

    Returns:
        Wrapped function that accepts uint8 or float32 and returns the original dtype.
    """

    @wraps(func)
    def uint8_wrapper(img: ImageType, *args: Any, **kwargs: Any) -> ImageType:
        input_dtype = img.dtype

        if input_dtype != np.uint8:
            img = from_float(cast("ImageFloat32", img), target_dtype=np.dtype(np.uint8))

        result = func(img, *args, **kwargs)

        return to_float(result) if input_dtype != np.uint8 else result

    return uint8_wrapper


@contiguous
@preserve_channel_dim
@uint8_io
def median_blur(img: ImageType, ksize: int) -> ImageType:
    """Median blur with optimal routing for multi-channel images.

    cv2.medianBlur supports >4 channels only for ksize 3 and 5 (GHA-built OpenCV).
    For ksize 7+, the SIMD path asserts cn <= 4. Uses uint8_io for float32 input.

    Args:
        img: (H, W, C) image, uint8 or float32.
        ksize: Kernel size (odd: 3, 5, 7, 9, ...).

    Returns:
        Median-filtered image, same shape and dtype.
    """
    if ksize % 2 != 1 or ksize < 3:
        raise ValueError(f"ksize must be odd and >= 3, got {ksize}")

    num_channels = get_num_channels(img)

    if ksize in (3, 5):
        return cast("ImageType", cv2.medianBlur(img, ksize))

    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return maybe_process_in_chunks(cv2.medianBlur, ksize)(img)
    return cast("ImageType", cv2.medianBlur(img, ksize))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized matrix multiplication for coordinate transformations.

    Replaces cv2.gemm which has similar performance but doesn't support all dtypes.
    Uses NumPy's @ operator which leverages optimized BLAS libraries:
    - ARM: Apple Accelerate framework
    - x86: MKL or OpenBLAS

    Benchmark results (macOS ARM):
    - Small matrices (2x2, 10x10): Similar to cv2.gemm (~1.0x)
    - Large matrices (1024x1024, 2048x2048): Similar to cv2.gemm (~1.0x)
    - Tall/skinny TPS matrices: Similar to cv2.gemm (0.93-1.02x)
    - uint8: Supported (cv2.gemm doesn't support uint8)

    Args:
        a: First matrix, shape (M, K), dtype float32, float64, or uint8
        b: Second matrix, shape (K, N), dtype float32, float64, or uint8

    Returns:
        Result matrix, shape (M, N). Output dtype follows NumPy's @ promotion rules:
        - float32 @ float32 -> float32
        - float64 @ float64 -> float64
        - uint8 @ uint8 -> int32 (promoted by NumPy)

    Examples:
        >>> import numpy as np
        >>> from albucore import matmul
        >>>
        >>> # ThinPlateSpline pairwise distance computation
        >>> points1 = np.random.randn(10000, 2).astype(np.float32)  # Target points
        >>> points2 = np.random.randn(10, 2).astype(np.float32)     # Control points
        >>> dot_matrix = matmul(points1, points2.T)  # (10000, 10)
        >>>
        >>> # TPS coordinate transformation
        >>> kernel = np.random.randn(10000, 10).astype(np.float32)
        >>> weights = np.random.randn(10, 2).astype(np.float32)
        >>> transformed = matmul(kernel, weights)  # (10000, 2)

    Note:
        This function is a simple wrapper around NumPy's @ operator,
        provided for API consistency and to make it explicit that
        this is the recommended replacement for cv2.gemm in geometric
        transformation contexts.

        Use Cases:
        - ThinPlateSpline geometric transformation (3 uses in AlbumentationsX)
        - Macenko stain normalization for medical imaging (1 use in AlbumentationsX)
    """
    return cast("np.ndarray", a @ b)


def pairwise_distances_squared(
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """Compute squared pairwise Euclidean distances between two point sets.

    Backend selection:
    - Small (``n1 * n2 < 1000``): NumKong ``cdist`` (pure NumPy alone was slower vs 0.0.40 on router).
    - Large: NumPy vectorized. Algorithm: ||a - b||² = ||a||² + ||b||² - 2(a·b)

    Args:
        points1: First set of points, shape (N, D), dtype float32
        points2: Second set of points, shape (M, D), dtype float32

    Returns:
        Matrix of squared distances, shape (N, M), dtype float32
        Element [i, j] contains ||points1[i] - points2[j]||²

    Examples:
        >>> import numpy as np
        >>> from albucore import pairwise_distances_squared
        >>> # Control points for thin plate spline
        >>> src_points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        >>> dst_points = np.array([[0.1, 0.1], [0.9, 0.1]], dtype=np.float32)
        >>> distances_sq = pairwise_distances_squared(src_points, dst_points)
        >>> distances_sq.shape
        (3, 2)

    Note:
        Returns SQUARED distances (not Euclidean distances).
        This is often what's needed (e.g., for RBF kernels in TPS),
        and avoids the expensive sqrt operation.

        For actual Euclidean distances: np.sqrt(result)

        The computation can produce very small negative values (e.g., -1e-6)
        due to floating-point rounding with float32 inputs. The result is
        automatically clamped to enforce non-negativity (distances >= 0).
    """
    # Keep dtype normalization cheap; only force contiguity on the nk.cdist branch below.
    points1 = points1.astype(np.float32, copy=False)
    points2 = points2.astype(np.float32, copy=False)

    n1, n2 = points1.shape[0], points2.shape[0]
    if n1 * n2 < 1000:
        # nk.cdist requires contiguous inputs on some strided views.
        points1 = np.ascontiguousarray(points1)
        points2 = np.ascontiguousarray(points2)
        result = np.asarray(nk.cdist(points1, points2, metric="sqeuclidean"), dtype=np.float32)
        np.maximum(result, 0.0, out=result)
        return result

    # Vectorized: ||a-b||² = ||a||² + ||b||² - 2(a·b)
    p1_squared = (points1**2).sum(axis=1, keepdims=True)  # (N, 1)
    p2_squared = (points2**2).sum(axis=1)[None, :]  # (1, M)
    dot_product = points1 @ points2.T  # (N, M)

    result = p1_squared + p2_squared - 2 * dot_product
    # Clamp to zero to handle numerical errors that can produce small negative values
    np.maximum(result, 0.0, out=result)
    return cast("np.ndarray", result)
