"""Zero-copy layout views between NumPy image kernels and CPU Tensor callers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from albucore.utils import TensorLayout

__all__ = ["TensorLayout"]

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
    return np.asarray(image.permute(_TENSOR_TO_CHANNEL_LAST[layout]).numpy())


def tensor_operand_to_numpy(value: Any, layout: TensorLayout | None) -> Any:
    """Convert a Tensor scalar, vector, or image operand to a NumPy value or view."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0:
        return value.item()
    if value.ndim >= 3:
        return tensor_to_numpy_channel_last(value, layout)
    return value.numpy()


def numpy_to_tensor_layout(array: np.ndarray, layout: TensorLayout) -> torch.Tensor:
    """Wrap a NumPy result and expose it in the caller's declared channel-first layout."""
    return numpy_to_tensor_storage(array).permute(_CHANNEL_LAST_TO_TENSOR[layout])


def numpy_to_tensor_storage(array: np.ndarray) -> torch.Tensor:
    """Wrap a writable NumPy result without exposing read-only or negative-stride storage to Torch."""
    if not array.flags.writeable or any(stride < 0 for stride in array.strides):
        array = np.array(array, copy=True, order="C")
    return torch.from_numpy(array)


def dispatch_tensor_image(
    image: torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call an existing channel-last router and restore an image result to the input layout."""
    image_view = tensor_to_numpy_channel_last(image, layout)
    router_kwargs = kwargs.copy()
    if kwargs.get("inplace") is True and not image_view.flags.c_contiguous:
        router_kwargs["inplace"] = False
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
                for name, value in router_kwargs.items()
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
        result = numpy_router(image_view, *args, **router_kwargs)
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
        return torch.from_numpy(array.copy() if not array.flags.writeable else array)
    if keepdims or array.ndim == reference.ndim:
        return numpy_to_tensor_layout(array, layout)
    tensor_result = torch.from_numpy(array.copy() if not array.flags.writeable else array)
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


def dispatch_tensor_reduction(
    image: torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., Any],
    axis: Any,
    kwargs: dict[str, Any],
) -> Any:
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
    return _reduction_result_to_tensor(result, image, layout, numpy_axes, keepdims=kwargs["keepdims"])
