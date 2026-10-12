from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, Literal, ParamSpec, TypeAlias, cast

import cv2
import numpy as np
import torch
from numpy import float16, float32, uint8
from numpy.typing import NDArray

NUM_RGB_CHANNELS = 3
TWO = 2

MAX_OPENCV_WORKING_CHANNELS = 4

NormalizationType = Literal["image", "image_per_channel", "min_max", "min_max_per_channel"]

P = ParamSpec("P")

ImageUInt8: TypeAlias = NDArray[uint8]
ImageFloat16: TypeAlias = NDArray[float16]
ImageFloat32: TypeAlias = NDArray[float32]
ImageType: TypeAlias = ImageUInt8 | ImageFloat16 | ImageFloat32
ImageInput: TypeAlias = ImageType | torch.Tensor
TensorLayout: TypeAlias = Literal["CHW", "NCHW", "CDHW", "NCDHW"]
TensorImageLayout: TypeAlias = Literal["CHW", "NCHW"]

# Supported storage dtypes. Statistics and matrix operations may return wider results.
SupportedDType: TypeAlias = np.dtype[np.uint8] | np.dtype[np.float16] | np.dtype[np.float32]

MAX_VALUES_BY_DTYPE: dict[np.dtype | type, float] = {
    np.dtype("uint8"): 255,
    np.dtype("float16"): 1.0,
    np.dtype("float32"): 1.0,
    np.uint8: 255,
    np.float16: 1.0,
    np.float32: 1.0,
}

NPDTYPE_TO_OPENCV_DTYPE: dict[np.dtype | type, int] = {
    np.uint8: cv2.CV_8U,
    np.float16: cv2.CV_16F,
    np.float32: cv2.CV_32F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("float16"): cv2.CV_16F,
    np.dtype("float32"): cv2.CV_32F,
}

_TENSOR_CHANNEL_AXES: dict[TensorLayout, int] = {"CHW": 0, "NCHW": 1, "CDHW": 0, "NCDHW": 1}
_TORCH_DTYPES: dict[np.dtype | type, torch.dtype] = {
    np.uint8: torch.uint8,
    np.dtype(np.uint8): torch.uint8,
    np.float16: torch.float16,
    np.dtype(np.float16): torch.float16,
    np.float32: torch.float32,
    np.dtype(np.float32): torch.float32,
}
_NUMPY_DTYPES: dict[torch.dtype, type[np.generic]] = {
    torch.uint8: np.uint8,
    torch.float16: np.float16,
    torch.float32: np.float32,
}


def get_opencv_max_channels() -> int:
    """Return the maximum channel count encodable by this OpenCV build.

    OpenCV stores the channel count in the matrix type. The bit layout changed
    between OpenCV 4 and 5, so deriving the limit keeps routing correct for both.
    """
    channel_stride = cv2.CV_MAKETYPE(cv2.CV_8U, 2) - cv2.CV_MAKETYPE(cv2.CV_8U, 1)
    return int((cv2.MAT_TYPE_MASK + 1) // channel_stride)


def maybe_process_in_chunks(
    process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        args: Additional positional arguments.
        kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray, *process_args: P.args, **process_kwargs: P.kwargs) -> np.ndarray:
        # Merge args and kwargs
        all_args = (*args, *process_args)
        all_kwargs: dict[str, Any] = kwargs | process_kwargs

        num_channels = img.shape[-1]
        if num_channels > MAX_OPENCV_WORKING_CHANNELS:
            # dst shape = full image, not chunk shape — strip before passing to cv2
            chunk_kwargs = {k: v for k, v in all_kwargs.items() if k != "dst"}
            out: np.ndarray | None = all_kwargs.get("dst")  # pre-allocated buffer or None
            offset = 0
            for index in range(0, num_channels, 4):
                if num_channels - index == TWO:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, *all_args, **chunk_kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        if out is None:
                            out = np.empty((*chunk.shape[:2], num_channels), dtype=img.dtype)
                        out[:, :, offset : offset + 1] = chunk
                        offset += 1
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, *all_args, **chunk_kwargs)
                    if chunk.ndim == 2:
                        chunk = np.expand_dims(chunk, -1)
                    if out is None:
                        out = np.empty((*chunk.shape[:2], num_channels), dtype=img.dtype)
                    n = chunk.shape[-1]
                    out[:, :, offset : offset + n] = chunk
                    offset += n
            if out is None:
                msg = "Chunked OpenCV processing failed to initialize output buffer."
                raise RuntimeError(msg)
            return out

        return process_fn(img, *all_args, **all_kwargs)

    return __process_fn


def clip(img: ImageInput, dtype: SupportedDType, inplace: bool = False) -> ImageInput:
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if isinstance(img, torch.Tensor):
        torch_dtype = _TORCH_DTYPES[dtype]
        if inplace and img.dtype == torch_dtype:
            return img.clamp_(0, max_value)
        return img.clamp(0, max_value).to(dtype=torch_dtype)
    if inplace and img.dtype == dtype:
        return cast("ImageType", np.clip(img, 0, max_value, out=img).astype(dtype, copy=False))
    return cast("ImageType", np.clip(img, 0, max_value).astype(dtype, copy=False))


def clipped(func: Callable[Concatenate[ImageInput, P], ImageInput]) -> Callable[Concatenate[ImageInput, P], ImageInput]:
    @wraps(func)
    def wrapped_function(img: ImageInput, *args: P.args, **kwargs: P.kwargs) -> ImageInput:
        dtype = img.dtype
        result = func(img, *args, **kwargs)

        if result.dtype in (np.uint8, torch.uint8):
            return result

        if isinstance(img, torch.Tensor):
            numpy_dtype = cast("SupportedDType", np.dtype(_NUMPY_DTYPES[dtype]))
            return clip(cast("torch.Tensor", result), numpy_dtype)

        if result.dtype in (np.float16, np.float32) and result.dtype == dtype and not np.shares_memory(result, img):
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            np.clip(result, 0.0, max_value, out=result)
            return result

        return clip(result, dtype)

    return wrapped_function


def get_num_channels(image: ImageInput, *, layout: TensorLayout | None = None) -> int:
    """Get the number of channels in an image array.

    This function returns the size of the last dimension, which always represents
    the number of channels in our convention.

    Args:
        image: Input image array. Expected shapes:
            - HWC: (height, width, channels) - multi-channel image
            - NHWC: (batch, height, width, channels) - batch of multi-channel images
            - XHWC: (leading, height, width, channels) - image batch or single volume
        layout: Required for Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW channel axis.

    Returns:
        int: Number of channels (the size of the last dimension).

    Examples:
        >>> # Grayscale image with explicit channel
        >>> img = np.zeros((100, 200, 1))
        >>> get_num_channels(img)
        1

        >>> # RGB image
        >>> img = np.zeros((100, 200, 3))
        >>> get_num_channels(img)
        3

        >>> # Batch of grayscale images
        >>> img = np.zeros((10, 100, 200, 1))
        >>> get_num_channels(img)
        1

        >>> # Batch of RGB images
        >>> img = np.zeros((10, 100, 200, 3))
        >>> get_num_channels(img)
        3

    Note:
        NumPy inputs use the last axis. Tensor inputs use the channel axis from ``layout``.
        The caller owns validation of the public layout contract before entering Albucore.
    """
    if isinstance(image, torch.Tensor):
        if layout is None:
            msg = "Tensor image calls require an explicit layout."
            raise TypeError(msg)
        return int(image.shape[_TENSOR_CHANNEL_AXES[layout]])
    return int(image.shape[-1])


def is_grayscale_image(image: ImageInput, *, layout: TensorLayout | None = None) -> bool:
    """Check if an image array represents a grayscale (single-channel) image.

    This function determines whether an image has only one channel by checking if
    shape[-1] == 1.

    Args:
        image: Input image array with explicit channel dimension.
        layout: Required for Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW channel axis.

    Returns:
        bool: True if the image has only 1 channel (shape[-1] == 1), False otherwise.

    Examples:
        >>> # Grayscale image with explicit channel
        >>> img = np.zeros((100, 200, 1))
        >>> is_grayscale_image(img)
        True

        >>> # RGB image
        >>> img = np.zeros((100, 200, 3))
        >>> is_grayscale_image(img)
        False

        >>> # Batch of grayscale images
        >>> img = np.zeros((10, 100, 200, 1))
        >>> is_grayscale_image(img)
        True

        >>> # Batch of RGB images
        >>> img = np.zeros((10, 100, 200, 3))
        >>> is_grayscale_image(img)
        False

    See Also:
        get_num_channels: For getting the exact number of channels.
        is_rgb_image: For checking if an image has exactly 3 channels (RGB).
        is_multispectral_image: For checking if an image has channels other than 1 or 3.
    """
    return get_num_channels(image, layout=layout) == 1


def get_opencv_dtype_from_numpy(value: np.ndarray | int | np.dtype | object) -> int:
    if isinstance(value, np.ndarray):
        value = value.dtype
    if isinstance(value, np.dtype):
        key: np.dtype | type = value
    elif value in (np.uint8, np.float16, np.float32):
        key = cast("type", value)
    else:
        msg = f"Unsupported dtype specifier: {value!r}"
        raise TypeError(msg)
    return int(NPDTYPE_TO_OPENCV_DTYPE[key])


def is_rgb_image(image: ImageInput, *, layout: TensorLayout | None = None) -> bool:
    return get_num_channels(image, layout=layout) == NUM_RGB_CHANNELS


def is_multispectral_image(image: ImageInput, *, layout: TensorLayout | None = None) -> bool:
    return get_num_channels(image, layout=layout) not in {1, 3}


def convert_value(value: object, num_channels: int) -> float | np.ndarray:
    """Convert a value to a float or numpy array based on its shape and number of channels.

    Args:
        value: Input value to convert (numpy array, float, or int)
        num_channels: Number of channels in the target image

    Returns:
        float: If value is a scalar or 1D array that should be converted to scalar
        np.ndarray: If value is a multi-dimensional array or channel vector

    The caller validates value type and channel-vector length before entering the router.
    """
    # Handle scalar types
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, np.floating):
        return float(value)

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            result: float | np.ndarray = float(value.item())
        elif value.ndim > 1:
            result = value
        else:
            num_values = len(value)
            result = float(value[0]) if num_values == 1 or num_channels == 1 else value[:num_channels]
        return result

    return cast("float | np.ndarray", value)


ValueType = np.ndarray | float | int


def get_max_value(dtype: np.dtype | type[np.generic] | torch.dtype) -> float:
    if isinstance(dtype, torch.dtype):
        torch_to_numpy_dtype: dict[torch.dtype, type[np.generic]] = {
            torch.uint8: np.uint8,
            torch.float16: np.float16,
            torch.float32: np.float32,
        }
        dtype = torch_to_numpy_dtype[dtype]
    if dtype not in MAX_VALUES_BY_DTYPE:
        msg = (
            f"Can't infer the maximum value for dtype {dtype}. "
            "You need to specify the maximum value manually by passing the max_value argument."
        )
        raise RuntimeError(msg)
    return MAX_VALUES_BY_DTYPE[dtype]


def get_image_data(data: dict[str, Any]) -> dict[str, np.dtype | torch.dtype | int]:
    """Extract image metadata (dtype, height, width, num_channels) from a dictionary.

    This function checks for image data under specific keys in priority order:
    'image' > 'images' > 'volume'

    The function correctly extracts height and width by accounting for the leading
    dimension based on the key type:
    - 'image': Direct H, W from shape[0], shape[1]
    - 'images': Skip batch dimension (shape[1], shape[2])
    - 'volume': Skip depth dimension (shape[1], shape[2])

    Args:
        data: Dictionary potentially containing image/volume arrays under specific keys.

    Returns:
        dict: Dictionary with 'dtype', 'height', 'width', and 'num_channels' keys.

    Raises:
        ValueError: If no valid image or single-volume data key is found in the dictionary.
    """
    if "image" in data:
        image = data["image"]
        if isinstance(image, torch.Tensor):
            return {
                "dtype": image.dtype,
                "height": image.shape[1],
                "width": image.shape[2],
                "num_channels": get_num_channels(image, layout="CHW"),
            }
        shape = data["image"].shape
        return {
            "dtype": data["image"].dtype,
            "height": shape[0],
            "width": shape[1],
            "num_channels": shape[-1],
        }
    if "images" in data:
        images = data["images"]
        if isinstance(images, torch.Tensor):
            return {
                "dtype": images.dtype,
                "height": images.shape[2],
                "width": images.shape[3],
                "num_channels": get_num_channels(images, layout="NCHW"),
            }
        shape = data["images"].shape
        return {
            "dtype": data["images"].dtype,
            "height": shape[1],
            "width": shape[2],
            "num_channels": shape[-1],
        }
    if "volume" in data:
        volume = data["volume"]
        if isinstance(volume, torch.Tensor):
            return {
                "dtype": volume.dtype,
                "height": volume.shape[2],
                "width": volume.shape[3],
                "num_channels": get_num_channels(volume, layout="CDHW"),
            }
        shape = data["volume"].shape
        return {
            "dtype": data["volume"].dtype,
            "height": shape[1],
            "width": shape[2],
            "num_channels": shape[-1],
        }
    raise ValueError("No valid image/volume data found in data dict")


__all__ = [
    "MAX_OPENCV_WORKING_CHANNELS",
    "MAX_VALUES_BY_DTYPE",
    "NPDTYPE_TO_OPENCV_DTYPE",
    "NUM_RGB_CHANNELS",
    "TWO",
    "ImageFloat16",
    "ImageFloat32",
    "ImageInput",
    "ImageType",
    "ImageUInt8",
    "NormalizationType",
    "P",
    "SupportedDType",
    "TensorImageLayout",
    "TensorLayout",
    "ValueType",
    "clip",
    "clipped",
    "convert_value",
    "get_image_data",
    "get_max_value",
    "get_num_channels",
    "get_opencv_dtype_from_numpy",
    "get_opencv_max_channels",
    "is_grayscale_image",
    "is_multispectral_image",
    "is_rgb_image",
    "maybe_process_in_chunks",
]
