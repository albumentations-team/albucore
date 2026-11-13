"""Storage and layout conversion between NumPy arrays and CPU Tensors."""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING, Any, cast, overload

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from albucore._types import TensorLayout

__all__ = [
    "TensorLayout",
    "numpy_to_tensor_layout",
    "numpy_to_tensor_storage",
    "tensor_to_numpy_channel_last",
    "tensor_to_numpy_storage",
]

_MACOS_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"
_LINUX_X86_64 = sys.platform == "linux" and platform.machine() == "x86_64"
_MIN_SPLIT_PIXELS = 8192
_MIN_ORIENTED_SPLIT_PIXELS = 32768
_MIN_FLIP_PIXELS = 4096
_OPENCV_INT_MAX = 2**31 - 1
_OPENCV_COPY_DTYPES = (np.uint8, np.uint16, np.int16, np.int32, np.float16, np.float32)
_OPENCV_FLIP_DTYPES = (*_OPENCV_COPY_DTYPES, np.int64)

_TENSOR_TO_CHANNEL_LAST: dict[TensorLayout, tuple[int, ...]] = {
    "CHW": (1, 2, 0),
    "NCHW": (0, 2, 3, 1),
    "CDHW": (1, 2, 3, 0),
    "NCDHW": (0, 2, 3, 4, 1),
}
_CHANNEL_LAST_TO_TENSOR: dict[TensorLayout, tuple[int, ...]] = {
    "CHW": (2, 0, 1),
    "NCHW": (0, 3, 1, 2),
    "CDHW": (3, 0, 1, 2),
    "NCDHW": (0, 4, 1, 2, 3),
}


def tensor_to_numpy_channel_last(image: torch.Tensor, layout: TensorLayout | None) -> np.ndarray:
    """Return a channel-last NumPy view for one prevalidated CPU Tensor layout."""
    if layout is None:
        msg = "Tensor image calls require an explicit layout."
        raise TypeError(msg)
    return image.numpy().transpose(_TENSOR_TO_CHANNEL_LAST[layout])


def tensor_to_numpy_storage(tensor: torch.Tensor) -> np.ndarray:
    """Expose a prevalidated CPU Tensor as a NumPy view in the same axis order."""
    return tensor.numpy()


def tensor_operand_to_numpy(value: Any, layout: TensorLayout | None) -> Any:
    """Convert a Tensor scalar, vector, or image operand to a NumPy value or view."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0:
        return value.item()
    if value.ndim >= 3:
        return tensor_to_numpy_channel_last(value, layout)
    return tensor_to_numpy_storage(value)


def numpy_to_tensor_layout(
    array: np.ndarray,
    layout: TensorLayout,
    *,
    contiguous: bool = False,
) -> torch.Tensor:
    """Expose a channel-last NumPy array as a CPU Tensor in the declared layout.

    The caller supplies an array with an explicit channel axis and a rank matching
    CHW, NCHW, CDHW, or NCDHW. The default wraps compatible storage as a view.
    With contiguous=True, the result uses native byte order and dense channel-first
    storage. Dtypes supported by torch.from_numpy are preserved.
    """
    axes = _CHANNEL_LAST_TO_TENSOR[layout]
    channel_first = array.transpose(axes)
    if not contiguous:
        return numpy_to_tensor_storage(channel_first)
    if channel_first.flags.c_contiguous:
        return numpy_to_tensor_storage(channel_first, contiguous=True)
    if 2 <= array.shape[-1] <= 5 and _can_split_channels(array, layout):
        return torch.from_numpy(
            _split_channel_last(array, channel_first.shape, batched=layout in ("NCHW", "NCDHW")),
        )
    return torch.from_numpy(_contiguous_storage(channel_first))


def numpy_to_tensor_storage(array: np.ndarray, *, contiguous: bool = False) -> torch.Tensor:
    """Wrap a NumPy array as a CPU Tensor, preserving its axes and dtype.

    With contiguous=True, materialize C storage when required. Both modes return
    writable native-order storage and preserve views with compatible strides.
    """
    if contiguous:
        return torch.from_numpy(_contiguous_storage(array))
    if not array.flags.writeable or not array.dtype.isnative or (array.ndim and min(array.strides) < 0):
        array = np.array(array, dtype=array.dtype.newbyteorder("="), copy=True, order="K")
    try:
        return torch.from_numpy(array)
    except ValueError:
        if all(stride % array.itemsize == 0 for stride in array.strides):
            raise
        return torch.from_numpy(array.copy(order="K"))


def _opencv_storage(array: np.ndarray) -> np.ndarray:
    # OpenCV's integer kernels copy float16 bit patterns without converting their values.
    return array.view(np.uint16) if array.dtype == np.float16 else array


def _interleaved_rows(array: np.ndarray) -> bool:
    pixel_bytes = array.shape[-1] * array.itemsize
    row_stride, pixel_stride, channel_stride = array.strides[-3:]
    return (
        channel_stride == array.itemsize
        and abs(pixel_stride) == pixel_bytes
        and abs(row_stride) >= array.shape[-2] * pixel_bytes
        and row_stride % array.itemsize == 0
    )


def _split_channel_last(array: np.ndarray, shape: tuple[int, ...], *, batched: bool) -> np.ndarray:
    channels = array.shape[-1]
    output = np.empty(shape, dtype=array.dtype)
    source = _opencv_storage(array)
    planes = _opencv_storage(output)
    mapping = [index for channel in range(channels) for index in (channel, channel)] if channels == 5 else None
    if array.flags.c_contiguous:
        count = array.shape[0] if batched else 1
        source = source.reshape(count, -1, channels)
        planes = planes.reshape(count, channels, -1)
        _write_channel_planes(source, [planes[:, channel] for channel in range(channels)], mapping)
        return output

    reverse_rows = source.strides[-3] < 0
    reverse_cols = source.strides[-2] < 0
    source = source[..., :: -1 if reverse_rows else 1, :: -1 if reverse_cols else 1, :]
    destinations = np.moveaxis(planes, 1 if batched else 0, -3)
    for index in np.ndindex(source.shape[:-3]):
        _write_channel_planes(source[index], list(destinations[index]), mapping)
    if reverse_rows:
        for plane in planes.reshape(-1, *planes.shape[-2:]):
            cv2.flip(plane, -1 if reverse_cols else 0, dst=plane)
    elif reverse_cols:
        rows = planes.reshape(-1, planes.shape[-1])
        cv2.flip(rows, 1, dst=rows)
    return output


def _write_channel_planes(source: np.ndarray, planes: list[np.ndarray], mapping: list[int] | None) -> None:
    if mapping is None:
        cv2.split(source, planes)
    else:
        cv2.mixChannels([source], planes, mapping)


def _can_split_channels(array: np.ndarray, layout: TensorLayout) -> bool:
    channels = array.shape[-1]
    minimum = _MIN_ORIENTED_SPLIT_PIXELS if channels == 5 else _MIN_SPLIT_PIXELS
    if (
        not (_MACOS_ARM64 or _LINUX_X86_64)
        or not minimum * channels <= array.size <= _OPENCV_INT_MAX
        or array.dtype not in _OPENCV_COPY_DTYPES
        or not array.flags.aligned
    ):
        return False
    if array.flags.c_contiguous:
        return True

    area = int(array.shape[-3]) * int(array.shape[-2])
    if area < minimum or not _interleaved_rows(array):
        return False
    return _can_split_strided(array, layout, channels, area)


def _can_split_strided(array: np.ndarray, layout: TensorLayout, channels: int, area: int) -> bool:
    reverse_rows = array.strides[-3] < 0
    reverse_cols = array.strides[-2] < 0
    if _LINUX_X86_64:
        if array.dtype == np.uint8 and channels < 5:
            return not reverse_rows or area >= _MIN_ORIENTED_SPLIT_PIXELS
        return channels < 5 and layout in ("CHW", "NCHW") and not (reverse_rows or reverse_cols)
    if not reverse_rows:
        return True
    if area < _MIN_ORIENTED_SPLIT_PIXELS or (channels == 2 and array.itemsize == 4):
        return False
    return not reverse_cols or (array.itemsize < 4 and (channels < 5 or array.itemsize == 1))


def _contiguous_storage(array: np.ndarray) -> np.ndarray:
    dtype = array.dtype if array.dtype.isnative else array.dtype.newbyteorder("=")
    if array.flags.c_contiguous:
        if array.flags.writeable and dtype == array.dtype and (not array.ndim or min(array.strides) >= 0):
            return array
        return np.array(array, dtype=dtype, order="C", copy=True)

    if (
        array.ndim >= 2
        and array.strides[-1] == -array.itemsize
        and _MIN_FLIP_PIXELS <= array.size <= _OPENCV_INT_MAX
        and array.dtype in _OPENCV_FLIP_DTYPES
        and array.flags.aligned
    ):
        source = _opencv_storage(array[..., ::-1])
        if source.dtype == np.int64:
            # flip also supports 64-bit floats as opaque 8-byte samples.
            source = source.view(np.float64)
        if source.flags.c_contiguous:
            source = source.reshape(-1, array.shape[-1])
            flipped = cast("np.ndarray", cv2.flip(source, 1))
            return flipped.view(array.dtype).reshape(array.shape)
        if (
            source.ndim == 2
            and source.strides[0] >= source.shape[-1] * source.itemsize
            and source.strides[0] % source.itemsize == 0
        ):
            flipped = cast("np.ndarray", cv2.flip(source, 1))
            return flipped.view(array.dtype)
        if (
            source.itemsize < 4
            and source.strides[-2] < 0
            and -source.strides[-2] >= source.shape[-1] * source.itemsize
            and source.strides[-2] % source.itemsize == 0
            and source.shape[-2] * source.shape[-1] >= _MIN_ORIENTED_SPLIT_PIXELS
        ):
            output = np.empty(array.shape, dtype=source.dtype)
            source = source[..., ::-1, :]
            for index in np.ndindex(source.shape[:-2]):
                cv2.flip(source[index], -1, dst=output[index])
            return output.view(array.dtype)
    return np.ascontiguousarray(array, dtype=dtype)


def dispatch_tensor_image(
    image: torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Call an existing channel-last router and restore an image result to the input layout."""
    image_view = tensor_to_numpy_channel_last(image, layout)
    if layout in ("NCHW", "NCDHW"):
        batch_sources = []
        batch_results = []
        for index, image_item in enumerate(image_view):
            batch_sources.append(image_item)
            item_args = tuple(
                value[index]
                if isinstance(value, np.ndarray)
                and value.ndim == image_view.ndim
                and value.shape[0] == image_view.shape[0]
                else value
                for value in args
            )
            item_kwargs = {
                name: value[index]
                if isinstance(value, np.ndarray)
                and value.ndim == image_view.ndim
                and value.shape[0] == image_view.shape[0]
                else value
                for name, value in kwargs.items()
            }
            item_result = numpy_router(image_item, *item_args, **item_kwargs)
            batch_results.append(item_result)
        if len(batch_results) == 0:
            result = numpy_router(image_view, *args, **kwargs)
            if not isinstance(result, np.ndarray):
                msg = "An image Tensor router must return an ndarray."
                raise TypeError(msg)
            return numpy_to_tensor_layout(result, layout)
        if all(
            result is source
            or (
                isinstance(result, np.ndarray)
                and result.shape == source.shape
                and result.dtype == source.dtype
                and result.strides == source.strides
                and result.__array_interface__["data"][0] == source.__array_interface__["data"][0]
            )
            for result, source in zip(batch_results, batch_sources, strict=True)
        ):
            return image
        if not all(isinstance(result, np.ndarray) for result in batch_results):
            msg = "An image Tensor router must return ndarrays."
            raise TypeError(msg)
        result = np.stack(batch_results)
    else:
        result = numpy_router(image_view, *args, **kwargs)
    if result is image_view:
        return image
    if not isinstance(result, np.ndarray):
        msg = "An image Tensor router must return an ndarray."
        raise TypeError(msg)
    return numpy_to_tensor_layout(result, cast("TensorLayout", layout))


def _tensor_reduction_axes_to_numpy(axis: Any, layout: TensorLayout) -> Any:
    if axis is None or isinstance(axis, str):
        return axis
    channel_last_axes = _CHANNEL_LAST_TO_TENSOR[layout]
    rank = len(channel_last_axes)
    tensor_axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(channel_last_axes[int(tensor_axis) % rank] for tensor_axis in tensor_axes)


def _reduction_result_to_tensor(
    result: Any,
    reference: torch.Tensor,
    layout: TensorLayout,
    numpy_axes: Any,
    *,
    keepdims: bool,
) -> Any:
    if isinstance(result, tuple):
        return tuple(
            _reduction_result_to_tensor(item, reference, layout, numpy_axes, keepdims=keepdims) for item in result
        )
    array = np.asarray(result)
    if array.ndim == 0:
        return numpy_to_tensor_storage(array)
    if keepdims or array.ndim == reference.ndim:
        return numpy_to_tensor_layout(array, layout)
    tensor_result = numpy_to_tensor_storage(array)
    if numpy_axes is None or isinstance(numpy_axes, str):
        return tensor_result
    axes = (numpy_axes,) if isinstance(numpy_axes, int) else tuple(numpy_axes)
    reduced_numpy_axes = {int(axis) % reference.ndim for axis in axes}
    tensor_axis_to_numpy = _CHANNEL_LAST_TO_TENSOR[layout]
    numpy_axis_to_tensor = _TENSOR_TO_CHANNEL_LAST[layout]
    reduced_tensor_axes = {numpy_axis_to_tensor[axis] for axis in reduced_numpy_axes}
    remaining_numpy_axes = [axis for axis in range(reference.ndim) if axis not in reduced_numpy_axes]
    desired_numpy_axes = [
        numpy_axis
        for tensor_axis, numpy_axis in enumerate(tensor_axis_to_numpy)
        if tensor_axis not in reduced_tensor_axes
    ]
    permutation = tuple(remaining_numpy_axes.index(axis) for axis in desired_numpy_axes)
    return tensor_result.permute(permutation) if permutation != tuple(range(len(permutation))) else tensor_result


@overload
def dispatch_tensor_reduction(
    image: torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., tuple[object, object]],
    axis: Any,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def dispatch_tensor_reduction(
    image: torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., float | np.generic | np.ndarray | torch.Tensor],
    axis: Any,
    kwargs: dict[str, Any],
) -> torch.Tensor: ...


def dispatch_tensor_reduction(
    image: torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., Any],
    axis: Any,
    kwargs: dict[str, Any],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reduce a channel-last NumPy view and restore scalar/vector/matrix Tensor outputs."""
    if layout is None:
        msg = "Tensor image calls require an explicit layout."
        raise TypeError(msg)
    numpy_axes = _tensor_reduction_axes_to_numpy(axis, layout)
    result = numpy_router(
        tensor_to_numpy_channel_last(image, layout),
        axis=numpy_axes,
        **kwargs,
    )
    return cast(
        "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
        _reduction_result_to_tensor(result, image, layout, numpy_axes, keepdims=kwargs["keepdims"]),
    )
