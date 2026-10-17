"""Weighted sums with caller-owned output storage."""

from typing import cast, overload

import cv2
import numkong as nk
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import ImageArrayT, ImageType, TensorLayout
from albucore.tensor_bridge import _CHANNEL_LAST_TO_TENSOR, tensor_to_numpy_channel_last
from albucore.utils import get_num_channels, get_opencv_max_channels

_NUMKONG_NEON_HALF = bool(nk.get_capabilities().get("neonhalf", False))
_NUMKONG_HALF_BLEND_MIN_ELEMENTS = 20_480
_MAX_OPENCV_CHANNELS = get_opencv_max_channels()


def _blend_numkong(
    img1: ImageArrayT,
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: ImageArrayT | None = None,
) -> ImageArrayT:
    first = np.ascontiguousarray(img1).reshape(-1)
    second = np.ascontiguousarray(img2).reshape(-1)
    if out is not None and out.flags.c_contiguous:
        nk.blend(first, second, alpha=weight1, beta=weight2, out=out.reshape(-1))
        return out
    blended = nk.blend(first, second, alpha=weight1, beta=weight2)
    if blended is None:
        raise RuntimeError("nk.blend returned None")
    result = np.asarray(blended, dtype=img1.dtype).reshape(img1.shape)
    if out is not None:
        np.copyto(out, result)
        return out
    return cast("ImageArrayT", result)


def _blend_opencv(
    img1: ImageArrayT,
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: ImageArrayT | None = None,
) -> ImageArrayT:
    if out is not None and out.flags.c_contiguous:
        cv2.addWeighted(img1, weight1, img2, weight2, 0, dst=out)
        return out
    result = cv2.addWeighted(img1, weight1, img2, weight2, 0).reshape(img1.shape)
    if out is not None:
        np.copyto(out, result)
        return out
    return cast("ImageArrayT", result)


def _blend_numpy(
    img1: ImageType,
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    if img1.dtype == np.float16:
        result = np.add(
            _cast_float_numpy(img1, np.float32) * weight1,
            _cast_float_numpy(img2, np.float32) * weight2,
            out=out,
        )
    else:
        result = np.add(
            img1.astype(np.float32, copy=False) * weight1,
            img2.astype(np.float32, copy=False) * weight2,
            out=out,
        )
    return cast("NDArray[np.float32]", result)


def _blend_array(
    img1: ImageType,
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: ImageType | None,
) -> ImageType:
    if img1.size == 0:
        return img1 if out is None else out
    if img1.dtype == np.float16:
        if (
            _NUMKONG_NEON_HALF
            and img1.flags.c_contiguous
            and img2.flags.c_contiguous
            and img1.size >= _NUMKONG_HALF_BLEND_MIN_ELEMENTS
            and weight1 == 0.5
            and weight2 == 0.5
        ):
            return _blend_numkong(img1, weight1, img2, weight2, out=out)
        if get_num_channels(img1) <= _MAX_OPENCV_CHANNELS:
            return _blend_opencv(img1, weight1, img2, weight2, out=out)
        return _cast_float_numpy(
            _blend_numpy(img1, weight1, img2, weight2),
            np.float16,
            out=cast("NDArray[np.float16] | None", out),
        )
    if (
        img1.dtype == np.float32
        and get_num_channels(img1) > 1
        and (img1.ndim == 3 or (img1.flags.c_contiguous and img2.flags.c_contiguous))
    ):
        return _blend_opencv(img1, weight1, img2, weight2, out=out)
    return _blend_numkong(img1, weight1, img2, weight2, out=out)


def _blend_batch(
    first: ImageType,
    weight1: float,
    second: ImageType,
    weight2: float,
    *,
    out: ImageType | None,
) -> ImageType:
    result = np.empty_like(first, order="C") if out is None else out
    for index in range(first.shape[0]):
        _blend_array(first[index], weight1, second[index], weight2, out=result[index])
    return result


def _blend_tensor(
    img1: torch.Tensor,
    weight1: float,
    img2: torch.Tensor,
    weight2: float,
    layout: TensorLayout,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if img1.numel() == 0:
        return img1 if out is None else out
    first = tensor_to_numpy_channel_last(img1, layout)
    second = first if img2 is img1 else tensor_to_numpy_channel_last(img2, layout)
    if out is None:
        destination = None
    elif out is img1:
        destination = first
    elif out is img2:
        destination = second
    else:
        destination = tensor_to_numpy_channel_last(out, layout)
    kernel_out = destination
    if destination is not None and not destination.flags.c_contiguous and layout not in ("NCHW", "NCDHW"):
        kernel_out = None
    if layout in ("NCHW", "NCDHW"):
        result = _blend_batch(first, weight1, second, weight2, out=destination)
    else:
        result = _blend_array(first, weight1, second, weight2, out=kernel_out)
    if out is not None and result is destination:
        return out
    tensor = torch.from_numpy(result).permute(_CHANNEL_LAST_TO_TENSOR[layout])
    if out is not None:
        out.copy_(tensor)
        return out
    return tensor


@overload
def add_weighted(
    img1: NDArray[np.uint8],
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: NDArray[np.uint8] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8]: ...


@overload
def add_weighted(
    img1: NDArray[np.float16],
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def add_weighted(
    img1: NDArray[np.float32],
    weight1: float,
    img2: ImageType,
    weight2: float,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def add_weighted(
    img1: torch.Tensor,
    weight1: float,
    img2: torch.Tensor,
    weight2: float,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def add_weighted(
    img1: ImageType | torch.Tensor,
    weight1: float,
    img2: ImageType | torch.Tensor,
    weight2: float,
    *,
    out: ImageType | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Blend matching images as ``img1 * weight1 + img2 * weight2``.

    Both operands have the same shape, dtype and container. NumPy arrays are channel-last;
    Tensor inputs declare CHW, NCHW, CDHW or NCDHW. AX validates this contract before dispatch.
    Floating output retains its raw range; uint8 output saturates using the selected backend's rounding.

    Without ``out``, neither input is changed; views and identity returns are allowed. A supplied
    writable output has matching metadata and is returned by identity. It may alias either complete
    operand; other overlaps are outside the caller-validated contract. Strided outputs can require
    a temporary. Non-finite arithmetic follows the existing backend selected for that input region.
    """
    if isinstance(img1, torch.Tensor):
        second_tensor = cast("torch.Tensor", img2)
        if (
            _NUMKONG_NEON_HALF
            and img1.dtype == torch.float16
            and img1.is_contiguous()
            and second_tensor.is_contiguous()
            and img1.numel() >= _NUMKONG_HALF_BLEND_MIN_ELEMENTS
            and weight1 == 0.5
            and weight2 == 0.5
        ):
            first = img1.numpy()
            second = first if img2 is img1 else second_tensor.numpy()
            if out is None:
                return torch.from_numpy(_blend_numkong(first, weight1, second, weight2))
            tensor_out = cast("torch.Tensor", out)
            if tensor_out.is_contiguous():
                _blend_numkong(first, weight1, second, weight2, out=tensor_out.numpy())
                return tensor_out
            tensor_out.copy_(torch.from_numpy(_blend_numkong(first, weight1, second, weight2)))
            return tensor_out
        return _blend_tensor(
            img1,
            weight1,
            second_tensor,
            weight2,
            cast("TensorLayout", layout),
            out=cast("torch.Tensor | None", out),
        )
    return _blend_array(img1, weight1, cast("ImageType", img2), weight2, out=cast("ImageType | None", out))
