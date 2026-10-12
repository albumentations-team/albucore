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
        all_args = (*args, *process_args)
        all_kwargs: dict[str, Any] = kwargs | process_kwargs

        num_channels = img.shape[-1]
        if num_channels > MAX_OPENCV_WORKING_CHANNELS:
            # dst shape = full image, not chunk shape — strip before passing to cv2
            chunk_kwargs = {k: v for k, v in all_kwargs.items() if k != "dst"}
            out: np.ndarray | None = all_kwargs.get("dst")
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
    """Return the channel count from the input's declared layout.

    NumPy inputs use the last axis (HWC, NHWC, DHWC, or NDHWC). Tensor inputs require
    ``layout=`` to select the channel axis in CHW, NCHW, CDHW, or NCDHW.
    Callers validate rank and layout before entering Albucore.

    Examples:
        >>> get_num_channels(np.zeros((100, 200, 3)))
        3
        >>> get_num_channels(torch.zeros((2, 3, 100, 200)), layout="NCHW")
        3
    """
    if isinstance(image, torch.Tensor):
        if layout is None:
            msg = "Tensor image calls require an explicit layout."
            raise TypeError(msg)
        return int(image.shape[_TENSOR_CHANNEL_AXES[layout]])
    return int(image.shape[-1])


def is_grayscale_image(image: ImageInput, *, layout: TensorLayout | None = None) -> bool:
    """Return whether the declared channel axis has exactly one channel.

    NumPy inputs use the last axis; Tensor inputs require ``layout=``.
    See :func:`get_num_channels` for supported layouts.
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
    if isinstance(value, (float, int, np.floating)):
        return float(value)

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
        dtype = _NUMPY_DTYPES[dtype]
    if dtype not in MAX_VALUES_BY_DTYPE:
        msg = (
            f"Can't infer the maximum value for dtype {dtype}. "
            "You need to specify the maximum value manually by passing the max_value argument."
        )
        raise RuntimeError(msg)
    return MAX_VALUES_BY_DTYPE[dtype]


def get_image_data(data: dict[str, Any]) -> dict[str, np.dtype | torch.dtype | int]:
    """Extract dtype, height, width, and channel count from image data.

    Keys take priority in the order ``image``, ``images``, ``volume``.
    NumPy arrays use HWC for ``image`` and NHWC/DHWC for the other keys;
    Tensor inputs use CHW, NCHW, and CDHW respectively.

    Raises:
        ValueError: If none of the supported keys is present.
    """
    if "image" in data:
        image = data["image"]
        height_axis = 0
        layout: TensorLayout = "CHW"
    elif "images" in data:
        image = data["images"]
        height_axis = 1
        layout = "NCHW"
    elif "volume" in data:
        image = data["volume"]
        height_axis = 1
        layout = "CDHW"
    else:
        raise ValueError("No valid image/volume data found in data dict")

    shape = image.shape
    if isinstance(image, torch.Tensor):
        height_axis += 1
        channel_axis = _TENSOR_CHANNEL_AXES[layout]
    else:
        channel_axis = -1
    return {
        "dtype": image.dtype,
        "height": shape[height_axis],
        "width": shape[height_axis + 1],
        "num_channels": shape[channel_axis],
    }


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
