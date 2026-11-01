"""Uint8 shared and per-channel lookup tables with explicit output storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

import cv2
import numpy as np
import stringzilla as sz
import torch

from albucore.tensor_bridge import numpy_to_tensor_layout, numpy_to_tensor_storage, tensor_to_numpy_channel_last
from albucore.utils import MAX_OPENCV_WORKING_CHANNELS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from albucore._types import TensorLayout


__all__ = ["apply_uint8_lut"]


def _cv2_lut_channel_last(img: NDArray[np.uint8], lut: np.ndarray, *, out: np.ndarray | None = None) -> np.ndarray:
    if out is not None and not out.flags.c_contiguous:
        np.copyto(out, _cv2_lut_channel_last(img, lut))
        return out
    num_channels = img.shape[-1]
    source = img if img.ndim == 3 else np.ascontiguousarray(img).reshape(-1, img.shape[-2], num_channels)
    result = np.empty(img.shape, dtype=lut.dtype) if out is None else out
    # Flattening first removes zero strides on singleton axes without copying.
    cv2.LUT(source, lut, dst=result.reshape(-1).reshape(source.shape))
    return result


def _apply_float_lut_per_channel_loop(
    img: NDArray[np.uint8],
    lut: np.ndarray,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    result = np.empty_like(img, dtype=np.float32) if out is None else out
    for i in range(img.shape[-1]):
        result[..., i] = cv2.LUT(img[..., i], lut[:, 0, i])
    return result


def _apply_float_lut(
    img: NDArray[np.uint8],
    lut: np.ndarray,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Apply uint8→float32 LUTs, retaining planar storage where the channel loop wins."""
    lut = lut.astype(np.float32, copy=False)
    if lut.ndim == 1:
        if out is not None and out.flags.c_contiguous:
            cv2.LUT(img, lut, dst=out.reshape(-1).reshape(img.shape))
            return out
        shared_result = cast("NDArray[np.float32]", cv2.LUT(img, lut).reshape(img.shape))
        if out is not None:
            np.copyto(out, shared_result)
            return out
        return shared_result

    num_channels = img.shape[-1]
    if img.flags["C_CONTIGUOUS"] or num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return cast("NDArray[np.float32]", _cv2_lut_channel_last(img, lut, out=out))

    # CHW-derived views avoid interleaving the full input and float32 output.
    return _apply_float_lut_per_channel_loop(img, lut, out=out)


def _strided_uint8_lut_out(img: NDArray[np.uint8], lut: NDArray[np.uint8], out: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if img.flags.c_contiguous:
        np.copyto(out, apply_uint8_lut(img, lut))
    else:
        working = np.ascontiguousarray(img)
        apply_uint8_lut(working, lut, out=working)
        np.copyto(out, working)
    return out


def _translate_contiguous_uint8_lut(img: NDArray[np.uint8], lut: NDArray[np.uint8], out: NDArray[np.uint8]) -> None:
    sz.translate(
        memoryview(cast("Any", img)),
        memoryview(cast("Any", lut)),
        out=memoryview(cast("Any", out)),
    )


def _apply_per_channel_uint8_luts(
    img: NDArray[np.uint8],
    luts: NDArray[np.uint8],
    *,
    out: NDArray[np.uint8] | None,
) -> NDArray[np.uint8]:
    """Apply ``(256, 1, C)`` tables without allocating a second result for contiguous outputs."""
    num_channels = img.shape[-1]
    if num_channels == 1:
        return apply_uint8_lut(img, luts[:, 0, 0], out=out)

    if img.flags["C_CONTIGUOUS"] or num_channels > MAX_OPENCV_WORKING_CHANNELS:
        return cast("NDArray[np.uint8]", _cv2_lut_channel_last(img, luts, out=out))

    result = np.empty_like(img) if out is None else out
    for i in range(num_channels):
        apply_uint8_lut(img[..., i], np.ascontiguousarray(luts[:, 0, i]), out=result[..., i])
    return result


def _tensor_shared_uint8_lut(img: torch.Tensor, lut: NDArray[np.uint8], *, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        result = np.array(img.numpy(), copy=True, order="C")
        apply_uint8_lut(result, lut, out=result)
        return torch.from_numpy(result)
    if out is img:
        view = img.numpy()
        apply_uint8_lut(view, lut, out=view)
        return img
    if out.is_contiguous():
        out.copy_(img)
        view = out.numpy()
        apply_uint8_lut(view, lut, out=view)
    else:
        source = img.numpy()
        working = np.ascontiguousarray(source)
        result = apply_uint8_lut(working, lut, out=working if working is not source else None)
        out.copy_(numpy_to_tensor_storage(result))
    return out


@overload
def apply_uint8_lut(
    img: NDArray[np.uint8],
    lut: NDArray[np.uint8],
    *,
    out: NDArray[np.uint8] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8]: ...


@overload
def apply_uint8_lut(
    img: torch.Tensor,
    lut: NDArray[np.uint8],
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> torch.Tensor: ...


def apply_uint8_lut(
    img: NDArray[np.uint8] | torch.Tensor,
    lut: NDArray[np.uint8],
    *,
    out: NDArray[np.uint8] | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8] | torch.Tensor:
    """Map uint8 values through a shared ``(256,)`` or per-channel ``(256, 1, C)`` table.

    Without ``out``, input storage is never written. A supplied output has the same shape,
    uint8 dtype, container and logical layout as the input and is returned by identity.
    Writable strided outputs and ``out=img`` are supported; other overlapping views and
    overlap with the table are outside the contract. Strides may require a temporary.

    NumPy uses channel-last images, batches and volumes. Per-channel Tensor calls require
    an explicit ``layout``; shared tables operate on bytes independently of layout.
    The caller validates images, outputs, layout, table shape and table contiguity.
    """
    if isinstance(img, torch.Tensor):
        tensor_out = cast("torch.Tensor | None", out)
        if lut.ndim == 1:
            return _tensor_shared_uint8_lut(img, lut, out=tensor_out)
        image_view = tensor_to_numpy_channel_last(img, layout)
        output_view = None
        if tensor_out is not None:
            output_view = image_view if tensor_out is img else tensor_to_numpy_channel_last(tensor_out, layout)
        result = _apply_per_channel_uint8_luts(
            image_view,
            lut,
            out=output_view,
        )
        return numpy_to_tensor_layout(result, cast("TensorLayout", layout)) if tensor_out is None else tensor_out
    if lut.ndim != 1:
        return _apply_per_channel_uint8_luts(img, lut, out=cast("NDArray[np.uint8] | None", out))
    if out is not None:
        numpy_out = cast("NDArray[np.uint8]", out)
        if not numpy_out.flags.c_contiguous:
            return _strided_uint8_lut_out(img, lut, numpy_out)
        if img is not numpy_out and img.flags.c_contiguous:
            _translate_contiguous_uint8_lut(img, lut, numpy_out)
        else:
            if numpy_out is not img:
                np.copyto(numpy_out, img)
            sz.translate(memoryview(cast("Any", numpy_out)), memoryview(cast("Any", lut)), inplace=True)
        return numpy_out
    source = np.ascontiguousarray(img)
    raw = sz.translate(memoryview(cast("Any", source)), memoryview(cast("Any", lut)), inplace=False)
    return np.frombuffer(raw, dtype=np.uint8).reshape(img.shape)
