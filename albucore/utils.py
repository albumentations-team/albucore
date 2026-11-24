import platform
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeGuard, TypeVar, cast, overload

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._types import ImageArrayT, ImageInput, SupportedDType, TensorLayout, ValueType

P = ParamSpec("P")

_FLOAT32_MAX = float(np.finfo(np.float32).max)
_INT32_MAX = 2_147_483_647


def _broadcast_channel_vector(x: ValueType | np.floating[Any], n_dim: int, c: int) -> ValueType | np.floating[Any]:
    if isinstance(x, np.ndarray) and x.shape == (c,):
        return x.reshape((1,) * (n_dim - 1) + (c,))
    return x


def _batch_operand(value: float | np.ndarray, image: np.ndarray, index: int) -> float | np.ndarray:
    if isinstance(value, np.ndarray) and value.ndim == image.ndim and value.shape[0] == image.shape[0]:
        return cast("np.ndarray", value[index])
    return value


NUM_RGB_CHANNELS = 3
MAX_OPENCV_WORKING_CHANNELS = 4

_ImageDTypeT = TypeVar("_ImageDTypeT", bound=np.uint8 | np.float16 | np.float32)


MAX_VALUES_BY_DTYPE: dict[np.dtype | type, float] = {
    np.dtype("uint8"): 255,
    np.dtype("float16"): 1.0,
    np.dtype("float32"): 1.0,
    np.uint8: 255,
    np.float16: 1.0,
    np.float32: 1.0,
}

_MACOS_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"
_LINUX_X86_64 = sys.platform == "linux" and platform.machine() == "x86_64"

# Element-count windows (inclusive) where ``torch.clamp`` beats the NumPy/OpenCV clip route, per platform and dtype.
# Platforms without an entry keep the NumPy/OpenCV route.
_LINUX_X86_64_CLIP_TORCH_WINDOWS: dict[np.dtype, tuple[int, int]] = {
    np.dtype(np.float16): (0, _INT32_MAX),
    # GCP x86-64 public timing: 8,192 elements 1.26x slower, 20,480 tie (1.02), 61,440 0.82x.
    np.dtype(np.float32): (61_440, 131_072),
}
_MACOS_ARM64_CLIP_TORCH_WINDOWS: dict[np.dtype, tuple[int, int]] = {
    np.dtype(np.float16): (0, _INT32_MAX),
    np.dtype(np.float32): (3_072, _INT32_MAX),
}
_CLIP_TORCH_WINDOWS: dict[np.dtype, tuple[int, int]] = (
    _LINUX_X86_64_CLIP_TORCH_WINDOWS if _LINUX_X86_64 else _MACOS_ARM64_CLIP_TORCH_WINDOWS if _MACOS_ARM64 else {}
)

_TENSOR_CHANNEL_AXES: dict[TensorLayout, int] = {"CHW": 0, "NCHW": 1, "CDHW": 0, "NCDHW": 1}
_TORCH_DTYPES: dict[np.dtype | type | torch.dtype, torch.dtype] = {
    np.uint8: torch.uint8,
    np.dtype(np.uint8): torch.uint8,
    np.float16: torch.float16,
    np.dtype(np.float16): torch.float16,
    np.float32: torch.float32,
    np.dtype(np.float32): torch.float32,
    torch.uint8: torch.uint8,
    torch.float16: torch.float16,
    torch.float32: torch.float32,
}
_NUMPY_DTYPES: dict[torch.dtype, type[np.uint8 | np.float16 | np.float32]] = {
    torch.uint8: np.uint8,
    torch.float16: np.float16,
    torch.float32: np.float32,
}


def _is_uint8_image(img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]) -> TypeGuard[NDArray[np.uint8]]:
    return img.dtype == np.uint8


def _is_float32_image(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> TypeGuard[NDArray[np.float32]]:
    return img.dtype == np.float32


def _is_float16_image(
    img: NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32],
) -> TypeGuard[NDArray[np.float16]]:
    return img.dtype == np.float16


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
                if num_channels - index == 2:
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
            return cast("np.ndarray", out)

        return process_fn(img, *all_args, **all_kwargs)

    return __process_fn


def _clip_float16_opencv(
    array: NDArray[np.float16],
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16] | None:
    if (
        not 16_384 <= array.size <= 2_147_483_647
        or not array.flags.c_contiguous
        or (out is not None and not out.flags.c_contiguous)
    ):
        return None
    # Preserve NaN payloads through the NumPy route.
    bits = array.view(np.uint16)
    largest_bits = bits.max()
    signed_bits = bits.view(np.int16)
    if largest_bits > 0x7C00 and (largest_bits > 0xFC00 or signed_bits.max() > 0x7C00):
        return None
    result = np.empty_like(array) if out is None else out
    if largest_bits <= 0x7C00:
        cv2.min(bits.reshape(1, -1), 0x3C00, result.view(np.uint16).reshape(1, -1))
    else:
        destination = result.view(np.int16).reshape(1, -1)
        cv2.max(signed_bits.reshape(1, -1), 0, destination)
        cv2.min(destination, 0x3C00, destination)
    return result


@overload
def clip(img: np.ndarray, dtype: None = None, *, out: ImageArrayT) -> ImageArrayT: ...


@overload
def clip(img: ImageArrayT, dtype: None = None, *, out: None = None) -> ImageArrayT: ...


@overload
def clip(img: NDArray[_ImageDTypeT], dtype: None = None, *, out: None = None) -> NDArray[_ImageDTypeT]: ...


@overload
def clip(
    img: np.ndarray,
    dtype: np.dtype[np.uint8] | type[np.uint8],
    *,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]: ...


@overload
def clip(
    img: np.ndarray,
    dtype: np.dtype[np.float16] | type[np.float16],
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16]: ...


@overload
def clip(
    img: np.ndarray,
    dtype: np.dtype[np.float32] | type[np.float32],
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]: ...


@overload
def clip(
    img: np.ndarray,
    dtype: np.dtype[np.uint8] | np.dtype[np.float32] | type[np.uint8] | type[np.float32],
    *,
    out: None = None,
) -> NDArray[np.uint8] | NDArray[np.float32]: ...


@overload
def clip(
    img: np.ndarray,
    dtype: SupportedDType,
    *,
    out: None = None,
) -> NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]: ...


@overload
def clip(
    img: np.ndarray,
    dtype: np.dtype[_ImageDTypeT] | type[_ImageDTypeT],
    *,
    out: None = None,
) -> NDArray[_ImageDTypeT]: ...


@overload
def clip(
    img: torch.Tensor,
    dtype: SupportedDType | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...


def clip(
    img: np.ndarray | torch.Tensor,
    dtype: np.dtype | type[np.generic] | None = None,
    *,
    out: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    """Saturate to the image dtype range and write only to an explicit output.

    Dtype defaults to the destination dtype, or the input dtype without output.
    The caller validates output shape, dtype, device, writeability, and overlap.
    Without ``out``, the input remains unchanged. ``out=img`` permits mutation
    when the input already has the requested output dtype.

    NumPy float16 and float32 images may be clipped with ``torch.clamp`` where it is faster. That route treats
    ``-0.0`` and ``+0.0`` as the same value and keeps NaN as NaN without preserving payload bits or signalling NaNs.
    Every other value, including infinities, is identical to the NumPy result.
    """
    if isinstance(img, torch.Tensor):
        # Uint8 is the only supported one-byte image dtype.
        if out is img:
            max_value = MAX_VALUES_BY_DTYPE[dtype] if dtype is not None else 255 if img.element_size() == 1 else 1.0
            img.clamp_(0, max_value)
            return img
        tensor_out = None if out is None else cast("torch.Tensor", out)
        if dtype is None:
            target = img if tensor_out is None else tensor_out
            max_value = 255 if target.element_size() == 1 else 1.0
        else:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        if tensor_out is None:
            tensor_result = img.clamp(0, max_value)
            return tensor_result if dtype is None else tensor_result.to(dtype=_TORCH_DTYPES[dtype])
        if tensor_out.dtype == img.dtype:
            torch.clamp(img, 0, max_value, out=tensor_out)
        else:
            tensor_out.copy_(img.clamp(0, max_value))
        return tensor_out
    return _clip_numpy(img, dtype, cast("np.ndarray | None", out))


def _torch_can_wrap(array: np.ndarray) -> bool:
    # ``torch.from_numpy`` rejects non-native byte order and byte strides that are not multiples of the item size.
    return array.dtype.isnative and array.flags.aligned


def _clip_torch(img: np.ndarray, max_value: float, out: np.ndarray | None) -> np.ndarray:
    if not img.flags.writeable or (img.ndim and min(img.strides) < 0):
        # ``torch.from_numpy`` rejects negative strides and warns on read-only storage.
        img = np.array(img, copy=True, order="K")
    source = torch.from_numpy(img)
    if out is None:
        return torch.clamp(source, 0, max_value).numpy()
    torch.clamp(source, 0, max_value, out=torch.from_numpy(out))
    return out


def _clip_numpy(img: np.ndarray, dtype: np.dtype | type[np.generic] | None, out: np.ndarray | None) -> np.ndarray:
    if dtype is None:
        inferred_dtype = img.dtype if out is None else out.dtype
        max_value = MAX_VALUES_BY_DTYPE[inferred_dtype.type]
        dtype = inferred_dtype
    else:
        max_value = MAX_VALUES_BY_DTYPE[dtype]
    if img.dtype == np.uint8 and dtype == np.uint8:
        if out is None:
            return img.copy(order="K")
        if out is not img:
            np.copyto(out, img)
        return out
    window = _CLIP_TORCH_WINDOWS.get(img.dtype)
    if (
        window is not None
        and dtype == img.dtype
        and window[0] <= img.size <= window[1]
        and _torch_can_wrap(img)
        and (
            out is None or (out.dtype == img.dtype and _torch_can_wrap(out) and (not out.ndim or min(out.strides) >= 0))
        )
    ):
        return _clip_torch(img, max_value, out)
    if img.dtype == np.float16 and dtype == np.float16:
        result = _clip_float16_opencv(
            cast("NDArray[np.float16]", img),
            out=cast("NDArray[np.float16] | None", out),
        )
        if result is not None:
            return result
    if out is None:
        clipped_array: np.ndarray = np.clip(img, 0, max_value)
        return clipped_array if clipped_array.dtype == dtype else clipped_array.astype(dtype, copy=False)
    np.clip(img, 0, max_value, out=out, casting="unsafe")
    return out


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


__all__ = [
    "MAX_OPENCV_WORKING_CHANNELS",
    "MAX_VALUES_BY_DTYPE",
    "NUM_RGB_CHANNELS",
    "clip",
    "convert_value",
    "get_max_value",
    "get_num_channels",
    "get_opencv_max_channels",
    "is_grayscale_image",
    "is_multispectral_image",
    "is_rgb_image",
    "maybe_process_in_chunks",
]
