"""CPU Torch kernels for large NumPy arrays.

Torch is an Albucore dependency. The helpers activate only for large, writable
arrays with compatible strides, so the NumPy wrapper and Torch Tensor can share
CPU storage without an input repair copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeGuard, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from albucore._types import ImageType

TORCH_MIN_ELEMENTS = 512 * 1024
TORCH_FLOAT16_TO_UINT8_MIN_ELEMENTS = 12_288
TORCH_CPU_BACKEND_ENABLED = True


def _is_torch_compatible_array(array: np.ndarray, *, minimum_elements: int | None = None) -> bool:
    """Check whether a NumPy array can use a measured Torch CPU route without an input repair copy."""
    if minimum_elements is None:
        minimum_elements = TORCH_MIN_ELEMENTS
    return (
        TORCH_CPU_BACKEND_ENABLED
        and array.size >= minimum_elements
        and array.flags.writeable
        and all(stride >= 0 for stride in array.strides)
    )


def _is_half_torch_compatible(array: np.ndarray, minimum_elements: int) -> TypeGuard[NDArray[np.float16]]:
    """Return whether a writable, nonnegative-stride half array meets the CPU Torch size threshold."""
    return (
        array.dtype == np.float16
        and array.size >= minimum_elements
        and array.flags.writeable
        and all(stride >= 0 for stride in array.strides)
    )


def _single_channel_half_parameter(value: np.ndarray) -> float | None:
    if value.dtype == np.float16:
        return float(value[0])
    try:
        scalar = float(value[0])
    except (OverflowError, TypeError, ValueError):
        return None
    if not -65_504 <= scalar <= 65_504:
        return None
    half_scalar = np.float16(scalar)
    return float(half_scalar) if float(half_scalar) == scalar else None


def _half_parameter(
    value: float | np.ndarray,
    array: np.ndarray,
) -> torch.Tensor | float | None:
    if not isinstance(value, np.ndarray):
        return float(value)
    if value.dtype != np.float16 and value.ndim == array.ndim:
        return None
    if not value.flags.writeable or any(stride < 0 for stride in value.strides):
        return None
    if value.ndim == 1 and value.size == array.shape[-1] == 1:
        return _single_channel_half_parameter(value)
    if value.dtype != np.float16 and not np.array_equal(value, value.astype(np.float16).astype(value.dtype)):
        return None
    parameter = value.astype(np.float16, copy=False)
    if not parameter.flags.c_contiguous:
        parameter = np.ascontiguousarray(parameter)
    if parameter.ndim == 1 and parameter.size == array.shape[-1]:
        parameter = parameter.reshape((1,) * (array.ndim - 1) + (array.shape[-1],))
    return torch.from_numpy(parameter)


def float16_pointwise_torch(
    array: np.ndarray,
    operation: Literal["add", "multiply", "power"],
    value: float | np.ndarray,
    *,
    minimum_elements: int,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    """Apply a measured eager CPU half operation without widening the image buffer."""
    if not _is_half_torch_compatible(array, minimum_elements):
        return None
    if out is not None and any(stride < 0 for stride in out.strides):
        return None
    parameter = _half_parameter(value, array)
    if parameter is None:
        return None
    image = torch.from_numpy(array)
    destination = None if out is None else torch.from_numpy(out)
    if operation == "add":
        result = torch.add(image, parameter, out=destination)
    elif operation == "multiply":
        result = torch.mul(image, parameter, out=destination)
    else:
        result = torch.pow(image, parameter, out=destination)
    if out is not None:
        return out
    return cast("NDArray[np.float16]", result.numpy())


def _float16_unary_torch(
    array: np.ndarray,
    operation: Literal["exp", "log", "sqrt"],
    *,
    minimum_elements: int,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    """Apply a measured CPU half unary operation without a float32 image buffer."""
    if not _is_half_torch_compatible(array, minimum_elements):
        return None
    if out is not None and any(stride < 0 for stride in out.strides):
        return None
    image = torch.from_numpy(array)
    destination = None if out is None else torch.from_numpy(out)
    if operation == "exp":
        result = torch.exp(image, out=destination)
    elif operation == "log":
        result = torch.log(image, out=destination)
    else:
        result = torch.sqrt(image, out=destination)
    if out is not None:
        return out
    return cast("NDArray[np.float16]", result.numpy())


def multiply_add_float16_torch(
    array: np.ndarray,
    factor: float | np.ndarray,
    bias: float | np.ndarray,
    *,
    minimum_elements: int,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    """Compute the measured two-rounding half route on shared CPU input."""
    if not _is_half_torch_compatible(array, minimum_elements):
        return None
    factor_tensor = _half_parameter(factor, array)
    bias_tensor = _half_parameter(bias, array)
    if factor_tensor is None or bias_tensor is None:
        return None
    image = torch.from_numpy(array)
    product = image * factor_tensor
    if out is not None and all(stride >= 0 for stride in out.strides):
        torch.add(product, bias_tensor, out=torch.from_numpy(out))
        return out
    result_tensor = product + bias_tensor
    del product
    result = result_tensor.numpy()
    if out is not None:
        np.copyto(out, result)
        return out
    return cast("NDArray[np.float16]", result)


def _parameter_tensor(value: float | np.ndarray, ndim: int, channels: int) -> torch.Tensor:
    """Create a float32 scalar or broadcastable channel-last Torch parameter."""
    if not isinstance(value, np.ndarray):
        return torch.tensor(value, dtype=torch.float32)

    value_float32 = value.astype(np.float32, copy=False)
    if not value_float32.flags.writeable:
        value_float32 = value_float32.copy()
    tensor = torch.from_numpy(value_float32)
    return tensor.reshape((1,) * (ndim - 1) + (channels,)) if value_float32.shape == (channels,) else tensor


def from_float_uint8_torch(
    img: NDArray[np.float16] | NDArray[np.float32],
    max_value: float,
    *,
    minimum_elements: int | None = None,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8] | None:
    """Scale, round, and saturate floating input to uint8 with a Torch CPU kernel."""
    if minimum_elements is None and img.dtype == np.float16:
        minimum_elements = TORCH_FLOAT16_TO_UINT8_MIN_ELEMENTS
    if img.dtype not in (np.float16, np.float32) or not _is_torch_compatible_array(
        img,
        minimum_elements=minimum_elements,
    ):
        return None
    tensor = torch.from_numpy(img)
    if img.dtype == np.float16:
        result = tensor.to(torch.float32)
        result.mul_(max_value).round_().clamp_(0.0, 255.0)
    else:
        result = torch.round(tensor * max_value).clamp_(0.0, 255.0)
    if out is not None and all(stride >= 0 for stride in out.strides):
        torch.from_numpy(out).copy_(result)
        return out
    result = result.to(torch.uint8)
    if out is not None:
        np.copyto(out, result.numpy())
        return out
    return cast("NDArray[np.uint8]", result.numpy())


def multiply_add_float32_torch(
    img: NDArray[np.float32],
    factor: float,
    value: float,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32] | None:
    """Compute two-step float32 arithmetic without copying the input buffer."""
    if img.dtype != np.float32 or not _is_torch_compatible_array(img):
        return None
    product = torch.from_numpy(img) * factor
    if out is not None and all(stride >= 0 for stride in out.strides):
        torch.add(product, value, out=torch.from_numpy(out))
        return out
    result_tensor = product + value
    del product
    result = result_tensor.numpy()
    if out is not None:
        np.copyto(out, result)
        return out
    return cast("NDArray[np.float32]", result)


def normalize_float32_torch(
    img: NDArray[np.float32],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32] | None:
    """Compute ``img * denominator - mean * denominator`` for multi-channel float32 input."""
    if img.dtype != np.float32 or img.shape[-1] <= 1 or not _is_torch_compatible_array(img):
        return None
    mean_tensor = _parameter_tensor(mean, img.ndim, img.shape[-1])
    denominator_tensor = _parameter_tensor(denominator, img.ndim, img.shape[-1])
    if out is not None and all(stride >= 0 for stride in out.strides):
        torch.addcmul(
            -mean_tensor * denominator_tensor,
            torch.from_numpy(img),
            denominator_tensor,
            out=torch.from_numpy(out),
        )
        return out
    result = torch.addcmul(-mean_tensor * denominator_tensor, torch.from_numpy(img), denominator_tensor)
    if out is not None:
        np.copyto(out, result.numpy())
        return out
    return cast("NDArray[np.float32]", result.numpy())


def normalize_float16_torch(
    img: NDArray[np.float16],
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
    *,
    minimum_elements: int,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    """Write the rounded fused result into half storage without a Tensor-owned return view."""
    if img.shape[-1] <= 1 or not img.flags.c_contiguous or not _is_half_torch_compatible(img, minimum_elements):
        return None
    mean_tensor = _parameter_tensor(mean, img.ndim, img.shape[-1])
    denominator_tensor = _parameter_tensor(denominator, img.ndim, img.shape[-1])
    image = torch.from_numpy(img)
    result = np.empty_like(img) if out is None else out
    direct = all(stride >= 0 for stride in result.strides)
    destination = torch.from_numpy(result) if direct else torch.empty_like(image)
    torch.addcmul(-mean_tensor * denominator_tensor, image, denominator_tensor, out=destination)
    if not direct:
        np.copyto(result, destination.numpy())
    return result


def reduce_sum_torch(
    arr: ImageType,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    minimum_elements: int | None = None,
) -> np.generic | np.ndarray | None:
    """Match Albucore's uint64/float64 sum accumulator for a preselected CPU Torch path."""
    if not _is_torch_compatible_array(arr, minimum_elements=minimum_elements):
        return None
    if arr.dtype == np.uint8:
        torch_dtype = torch.int64
        output_dtype: type[np.generic] = np.uint64
    elif arr.dtype in (np.float16, np.float32):
        torch_dtype = torch.float64
        output_dtype = np.float64
    else:
        return None
    image = torch.from_numpy(arr)
    if len(axes) == arr.ndim and not keepdims:
        result = torch.sum(image, dtype=torch_dtype).numpy()
    else:
        result = torch.sum(image, dim=axes, dtype=torch_dtype, keepdim=keepdims).numpy()
    if output_dtype is np.uint64:
        result = result.view(np.uint64)
    if len(axes) == arr.ndim and not keepdims:
        return output_dtype(result)
    return np.asarray(result)
