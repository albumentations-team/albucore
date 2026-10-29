"""Pointwise arithmetic: add, multiply, power and multiply-add."""

import math
from typing import Any, Literal, TypeAlias, TypeGuard, cast, overload

import cv2
import numkong as nk
import numpy as np
import torch
from numpy.typing import NDArray

from albucore._float_cast import _cast_float_numpy
from albucore._types import ImageArrayT, ImageType, ValueType
from albucore.lut import apply_uint8_lut as _apply_uint8_lut
from albucore.tensor_bridge import (
    TensorLayout,
    numpy_to_tensor_layout,
    numpy_to_tensor_storage,
    tensor_operand_to_numpy,
    tensor_to_numpy_channel_last,
)
from albucore.torch_backend import (
    float16_pointwise_torch,
    multiply_add_float16_torch,
    multiply_add_float32_torch,
)
from albucore.utils import (
    _FLOAT32_MAX,
    MAX_OPENCV_WORKING_CHANNELS,
    _batch_operand,
    _broadcast_channel_vector,
    clip,
    convert_value,
    get_num_channels,
)
from albucore.weighted import (
    _NUMKONG_NEON_HALF,
    _blend_numkong,
)

_NUMPY_OPERATIONS = {"multiply": np.multiply, "add": np.add, "power": np.power}

_PreparedValue: TypeAlias = float | int | np.floating[Any] | np.ndarray
_FLOAT32_COMPATIBLE_OPERAND_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    np.dtype(dtype) for dtype in (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.float16, np.float32)
)
_TORCH_FLOAT16_MIN_ELEMENTS = 16_384
_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS = 230_400
_NUMKONG_FLOAT16_MIN_ELEMENTS = 20_480
_NUMPY_FLOAT16_C9_VECTOR_MIN_ELEMENTS = 184_320
_CV16F_VECTOR_MIN_ELEMENTS = 61_440
_FLOAT16_MIN_NORMAL = float(np.finfo(np.float16).tiny)


def _scale_numkong(
    img: ImageArrayT,
    factor: float,
    value: float,
    *,
    out: ImageArrayT | None = None,
) -> ImageArrayT:
    """Scalar affine transform via ``nk.scale`` with same-dtype output storage."""
    if out is not None and not out.flags.c_contiguous:
        np.copyto(out, _scale_numkong(img, factor, value))
        return out
    flat = np.ascontiguousarray(img).reshape(-1)
    if out is not None:
        nk.scale(flat, alpha=float(factor), beta=float(value), out=out.reshape(-1))
        return out
    result = nk.scale(flat, alpha=float(factor), beta=float(value))
    if result is None:
        raise RuntimeError("nk.scale returned None")
    return cast("ImageArrayT", np.asarray(result, dtype=img.dtype).reshape(img.shape))


def _is_uint8_image(img: ImageType) -> TypeGuard[NDArray[np.uint8]]:
    return img.dtype == np.uint8


def _is_float32_image(img: ImageType) -> TypeGuard[NDArray[np.float32]]:
    return img.dtype == np.float32


def _is_float16_image(img: ImageType) -> TypeGuard[NDArray[np.float16]]:
    return img.dtype == np.float16


def _half_exact_scalar(value: object) -> float | None:
    if not isinstance(value, (float, int)):
        return None
    try:
        scalar = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(scalar) or abs(scalar) > 65_504:
        return None
    return scalar if float(np.float16(scalar)) == scalar else None


def _opencv_float16_three_channel_vector(
    img: NDArray[np.float16],
    value: object,
    operation: Literal["add", "multiply"],
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    if (
        not img.flags.c_contiguous
        or not img.flags.writeable
        or img.ndim < 3
        or img.shape[-1] != 3
        or img.size < _CV16F_VECTOR_MIN_ELEMENTS
        or not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.size != 3
    ):
        return None
    control0 = _half_exact_scalar(float(value[0]))
    control1 = _half_exact_scalar(float(value[1]))
    control2 = _half_exact_scalar(float(value[2]))
    if control0 is None or control1 is None or control2 is None:
        return None
    scalar = (control0, control1, control2, 0.0)
    # OpenCV stubs: https://github.com/opencv/opencv/issues/30296 (review 2026-11-15).
    if out is not None and not out.flags.c_contiguous:
        return None
    source = img.reshape(-1, img.shape[-2], 3)
    destination = None if out is None else out.reshape(source.shape)
    if operation == "add":
        result = cv2.add(source, cast("Any", scalar), dtype=cv2.CV_16F, dst=destination)
    else:
        result = cv2.multiply(source, cast("Any", scalar), dtype=cv2.CV_16F, dst=destination)
    return cast("NDArray[np.float16]", result.reshape(img.shape)) if out is None else out


def _numpy_float16_c9_vector_eligible(img: NDArray[np.float16], value: object) -> bool:
    return (
        img.flags.c_contiguous
        and img.flags.writeable
        and img.ndim == 3
        and img.shape[-1] == 9
        and img.size >= _NUMPY_FLOAT16_C9_VECTOR_MIN_ELEMENTS
        and isinstance(value, np.ndarray)
        and value.dtype == np.float32
        and value.flags.c_contiguous
        and value.ndim == 1
        and value.size == 9
    )


def _opencv_float16_c9_multiply_vector(
    img: NDArray[np.float16],
    value: object,
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    if not isinstance(value, np.ndarray) or not _numpy_float16_c9_vector_eligible(img, value):
        return None
    if not np.all(np.isfinite(value)):
        return None
    with np.errstate(over="ignore"):
        controls = value.astype(np.float16)
    if not np.array_equal(value, controls.astype(np.float32)):
        magnitudes = np.abs(value)
        if np.any(magnitudes > 1.0) or np.any((magnitudes != 0) & (magnitudes < _FLOAT16_MIN_NORMAL)):
            return None
    control_view = np.broadcast_to(controls.reshape(1, 1, 9), img.shape)
    if out is not None and not out.flags.c_contiguous:
        return None
    result = cv2.multiply(img, control_view, dtype=cv2.CV_16F, dst=out)
    return cast("NDArray[np.float16]", result) if out is None else out


def _numkong_float16_c9_add_channels(
    img: NDArray[np.float16],
    value: object,
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    if (
        not _NUMKONG_NEON_HALF
        or not img.flags.c_contiguous
        or not img.flags.writeable
        or img.ndim != 3
        or img.shape[-1] != 9
        or img.size < _NUMPY_FLOAT16_C9_VECTOR_MIN_ELEMENTS
        or not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or not value.flags.c_contiguous
        or value.ndim != 1
        or value.size != 9
        or not np.all(np.isfinite(value))
    ):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        controls = value.astype(np.float16)
    if not np.array_equal(value, controls.astype(np.float32)):
        return None
    control_image = np.broadcast_to(controls.reshape(1, 1, 9), img.shape)
    if out is not None:
        if not out.flags.c_contiguous:
            return None
        nk.add(img, control_image, out=out)
        return out
    result = nk.add(img, control_image)
    if result is None:
        raise RuntimeError("nk.add returned None")
    return cast("NDArray[np.float16]", np.asarray(result).reshape(img.shape))


def _numkong_float16_single_channel_scale(
    img: NDArray[np.float16],
    value: object,
    operation: Literal["add", "multiply"],
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    if (
        not _NUMKONG_NEON_HALF
        or not img.flags.c_contiguous
        or not img.flags.writeable
        or img.ndim != 3
        or img.shape[-1] != 1
        or img.size < _NUMKONG_FLOAT16_MIN_ELEMENTS
        or not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.size != 1
    ):
        return None
    scalar = _half_exact_scalar(float(value[0]))
    if scalar is None:
        return None
    if out is not None:
        if not out.flags.c_contiguous:
            return None
        nk.scale(
            img.reshape(-1),
            alpha=1.0 if operation == "add" else scalar,
            beta=scalar if operation == "add" else -0.0,
            out=out.reshape(-1),
        )
        return out
    result = (
        nk.scale(img.reshape(-1), alpha=1.0, beta=scalar)
        if operation == "add"
        else nk.scale(
            img.reshape(-1),
            alpha=scalar,
            beta=-0.0,
        )
    )
    if result is None:
        raise RuntimeError("nk.scale returned None")
    return cast("NDArray[np.float16]", np.asarray(result).reshape(img.shape))


def _float16_numpy_ufunc(
    img: NDArray[np.float16],
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    *,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16] | None:
    if isinstance(value, np.ndarray) and value.dtype not in _FLOAT32_COMPATIBLE_OPERAND_DTYPES:
        return None
    if not isinstance(value, np.ndarray) and abs(float(value)) > _FLOAT32_MAX:
        return None
    output = np.empty_like(img) if out is None else out
    with np.errstate(over="ignore", invalid="ignore"):
        _NUMPY_OPERATIONS[operation](img, value, dtype=np.float32, out=output, casting="unsafe")
    return output


def _apply_lut(
    img: NDArray[np.uint8],
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    *,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    domain: NDArray[np.float32] = np.arange(256, dtype=np.float32)
    operand = np.asarray(np.trunc(value) if operation == "add" else value, dtype=np.float32)
    if operand.ndim:
        domain = domain.reshape(256, 1, 1)
        operand = operand.reshape(1, 1, -1)
    table = clip(_NUMPY_OPERATIONS[operation](domain, operand), np.uint8)
    return _apply_uint8_lut(img, table, out=out)


def _prepare_value_opencv(
    img: ImageType,
    value: np.ndarray | float,
    operation: Literal["add", "multiply"],
) -> np.ndarray | float | int:
    return (
        _prepare_scalar_value(img, value, operation)
        if isinstance(value, (int, float))
        else _prepare_array_value(img, value, operation)
    )


def _prepare_scalar_value(
    img: ImageType,
    value: float,
    operation: Literal["add", "multiply"],
) -> np.ndarray | float | int:
    if operation == "add" and img.dtype == np.uint8:
        value = int(value)
    num_channels = get_num_channels(img)
    if num_channels > MAX_OPENCV_WORKING_CHANNELS:
        if operation == "add":
            # Cast to float32 if value is negative to handle potential underflow issues
            cast_type = np.float32 if value < 0 else img.dtype
            return np.full(img.shape, value, dtype=cast_type)
        if operation == "multiply":
            return np.full(img.shape, value, dtype=np.float32)
    return value


def _prepare_array_value(
    img: ImageType,
    value: np.ndarray,
    operation: Literal["add", "multiply"],
) -> np.ndarray:
    if value.dtype == np.float64:
        value = value.astype(np.float32, copy=False)
    if value.ndim == 1:
        value = value.reshape(1, 1, -1)
    value = np.broadcast_to(value, img.shape)
    if operation == "add" and img.dtype == np.uint8:
        if np.all(value >= 0):
            return cast("np.ndarray", clip(cast("ImageType", value), np.dtype(np.uint8)))
        return cast("np.ndarray", np.trunc(value).astype(np.float32, copy=False))
    return value


def _prepare_numpy_value(value: ValueType | np.floating[Any]) -> _PreparedValue:
    if isinstance(value, np.ndarray):
        if value.dtype in _FLOAT32_COMPATIBLE_OPERAND_DTYPES or np.result_type(np.float32, value.dtype) == np.float32:
            return value
        try:
            with np.errstate(over="raise"):
                return value.astype(np.float32, copy=False)
        except FloatingPointError:
            # Preserve finite values outside the float32 range until after the
            # arithmetic. Narrowing them first would turn them into infinities
            # and make otherwise defined operations such as 0 * 1e40 return NaN.
            return value
    if isinstance(value, np.floating):
        try:
            with np.errstate(over="raise"):
                return np.float32(value)
        except FloatingPointError:
            pass
    elif isinstance(value, float) and math.isfinite(value) and not -_FLOAT32_MAX <= value <= _FLOAT32_MAX:
        value = np.float64(value)
    return value


def _apply_numpy(
    img: ImageType,
    value: ValueType | np.floating[Any],
    operation: Literal["add", "multiply", "power"],
    *,
    out: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    prepared: _PreparedValue = (
        np.asarray(value, dtype=np.int16)
        if operation == "add" and img.dtype == np.uint8
        else _prepare_numpy_value(value)
    )
    ufunc = _NUMPY_OPERATIONS[operation]
    if out is not None and np.result_type(np.float32, prepared) == np.dtype(np.float32):
        ufunc(_cast_float_numpy(img, np.float32), prepared, out=out)
        return out
    result = ufunc(_cast_float_numpy(img, np.float32), prepared)
    if out is not None:
        with np.errstate(over="ignore"):
            np.copyto(out, result)
        return out
    if result.dtype == np.float32:
        return cast("NDArray[np.float32]", result)
    with np.errstate(over="ignore"):
        return cast("NDArray[np.float32]", result.astype(np.float32, copy=False))


def _numpy_pointwise(
    img: ImageType,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    *,
    out: ImageType | None = None,
) -> ImageType:
    if _is_float16_image(img):
        destination = cast("NDArray[np.float16] | None", out)
        result = _float16_numpy_ufunc(img, value, operation, out=destination)
        if result is not None:
            return result
        return _cast_float_numpy(_apply_numpy(img, value, operation), np.float16, out=destination)
    return _apply_numpy(img, value, operation, out=cast("NDArray[np.float32] | None", out))


def _multiply_uint8_opencv(
    img: NDArray[np.uint8],
    value: np.ndarray | float,
    *,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    prepared = _prepare_value_opencv(img, value, "multiply")
    operand: Any = prepared.astype(np.float32, copy=False) if isinstance(prepared, np.ndarray) else prepared
    result = cast("NDArray[np.float32]", cv2.multiply(img.astype(np.float32), operand).reshape(img.shape))
    return clip(result, np.uint8, out=out)


def _multiply_scalar(img: ImageType, value: float, *, out: ImageType | None = None) -> ImageType:
    if _is_uint8_image(img):
        return _apply_lut(img, value, "multiply", out=cast("NDArray[np.uint8] | None", out))
    if _is_float16_image(img):
        result = float16_pointwise_torch(
            img,
            "multiply",
            value,
            minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
            out=cast("NDArray[np.float16] | None", out),
        )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "multiply", out=out)


def _multiply_channels(img: ImageType, value: np.ndarray, *, out: ImageType | None = None) -> ImageType:
    if _is_uint8_image(img):
        return _apply_lut(img, value, "multiply", out=cast("NDArray[np.uint8] | None", out))
    if _is_float16_image(img):
        destination = cast("NDArray[np.float16] | None", out)
        result = _opencv_float16_three_channel_vector(img, value, "multiply", out=destination)
        if result is None:
            result = _opencv_float16_c9_multiply_vector(img, value, out=destination)
        if result is None and _numpy_float16_c9_vector_eligible(img, value):
            return _numpy_pointwise(img, value, "multiply", out=out)
        if result is None:
            result = _numkong_float16_single_channel_scale(img, value, "multiply", out=destination)
        if result is None:
            result = float16_pointwise_torch(
                img,
                "multiply",
                value,
                minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
                out=destination,
            )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "multiply", out=out)


def _multiply_field(img: ImageType, value: np.ndarray, *, out: ImageType | None = None) -> ImageType:
    if _is_uint8_image(img):
        return _multiply_uint8_opencv(img, value, out=cast("NDArray[np.uint8] | None", out))
    if _is_float16_image(img) and not (value.dtype == np.float32 and value.ndim == img.ndim):
        result = float16_pointwise_torch(
            img,
            "multiply",
            value,
            minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
            out=cast("NDArray[np.float16] | None", out),
        )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "multiply", out=out)


def _prepare_pointwise_operand(img: ImageType, value: ValueType) -> float | np.ndarray:
    channels = get_num_channels(img)
    if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == channels:
        return value
    return convert_value(value, channels)


@overload
def multiply(
    img: NDArray[np.uint8],
    value: ValueType,
    *,
    out: NDArray[np.uint8] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8]: ...


@overload
def multiply(
    img: NDArray[np.float16],
    value: ValueType,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def multiply(
    img: NDArray[np.float32],
    value: ValueType,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def multiply(
    img: torch.Tensor,
    value: ValueType | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def multiply(
    img: ImageType | torch.Tensor,
    value: ValueType | torch.Tensor,
    *,
    out: ImageType | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Multiply by a scalar, channel vector or broadcastable field.

    Without ``out``, input storage is unchanged; views and identity are allowed.
    A supplied writable output has the input shape, dtype, container and layout and is returned
    by identity. Exact image/field aliases are supported; other overlaps, including broadcast
    controls sharing output storage, are outside the caller-validated contract.
    NumPy uses channel-last images, batches and volumes. Tensor calls require an explicit layout.
    Uint8 saturates to [0, 255]; floating results keep their raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        tensor_out = cast("torch.Tensor | None", out)
        if (
            isinstance(value, torch.Tensor)
            and value.ndim == img.ndim
            and value.dtype == img.dtype
            and img.dtype in (torch.float16, torch.float32)
        ):
            return torch.mul(img, value) if tensor_out is None else torch.mul(img, value, out=tensor_out)
        return _tensor_pointwise(img, value, "multiply", layout, out=tensor_out)
    destination = cast("ImageType | None", out)
    value = _prepare_pointwise_operand(img, cast("ValueType", value))
    if img.size == 0:
        return img if destination is None else destination
    if isinstance(value, (float, int)):
        return _multiply_scalar(img, value, out=destination)
    return (
        _multiply_channels(img, value, out=destination)
        if value.ndim == 1
        else _multiply_field(img, value, out=destination)
    )


def _add_uint8_opencv(
    img: NDArray[np.uint8],
    value: float | np.ndarray,
    *,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    prepared = _prepare_value_opencv(img, value, "add")
    if (isinstance(prepared, (float, int)) and prepared < 0) or (
        isinstance(prepared, np.ndarray) and prepared.dtype != np.uint8
    ):
        operand: Any = prepared.astype(np.float32, copy=False) if isinstance(prepared, np.ndarray) else prepared
        float_result = cast("NDArray[np.float32]", cv2.add(img.astype(np.float32), operand).reshape(img.shape))
        return clip(float_result, np.uint8, out=out)
    if out is not None and not out.flags.c_contiguous:
        np.copyto(out, _add_uint8_opencv(img, value))
        return out
    destination = None if out is None else out.reshape(-1).reshape(img.shape)
    value_cv2: Any = prepared
    result = cv2.add(img, value_cv2, dst=destination).reshape(img.shape)
    return cast("NDArray[np.uint8]", result) if out is None else out


def _add_scalar(img: ImageType, value: float, *, out: ImageType | None = None) -> ImageType:
    if _is_uint8_image(img):
        return _add_uint8_opencv(img, value, out=cast("NDArray[np.uint8] | None", out))
    if _is_float16_image(img):
        result = float16_pointwise_torch(
            img,
            "add",
            value,
            minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
            out=cast("NDArray[np.float16] | None", out),
        )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "add", out=out)


def _add_channels(img: ImageType, value: np.ndarray, *, out: ImageType | None = None) -> ImageType:
    if _is_uint8_image(img):
        return _apply_lut(img, value, "add", out=cast("NDArray[np.uint8] | None", out))
    if _is_float16_image(img):
        destination = cast("NDArray[np.float16] | None", out)
        result = _opencv_float16_three_channel_vector(img, value, "add", out=destination)
        if result is None:
            result = _numkong_float16_c9_add_channels(img, value, out=destination)
        if result is None and _numpy_float16_c9_vector_eligible(img, value):
            return _numpy_pointwise(img, value, "add", out=out)
        if result is None:
            result = _numkong_float16_single_channel_scale(img, value, "add", out=destination)
        if result is None:
            result = float16_pointwise_torch(
                img,
                "add",
                value,
                minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
                out=destination,
            )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "add", out=out)


def _add_field(img: ImageType, value: np.ndarray, *, out: ImageType | None = None) -> ImageType:
    if _is_uint8_image(img):
        destination = cast("NDArray[np.uint8] | None", out)
        if value.shape == img.shape and value.dtype == img.dtype:
            return _blend_numkong(img, 1.0, value, 1.0, out=destination)
        return _add_uint8_opencv(img, value, out=destination)
    if _is_float16_image(img) and not (value.dtype == np.float32 and value.ndim == img.ndim):
        result = float16_pointwise_torch(
            img,
            "add",
            value,
            minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
            out=cast("NDArray[np.float16] | None", out),
        )
        if result is not None:
            return result
    return _numpy_pointwise(img, value, "add", out=out)


@overload
def add(
    img: NDArray[np.uint8],
    value: ValueType,
    *,
    out: NDArray[np.uint8] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8]: ...


@overload
def add(
    img: NDArray[np.float16],
    value: ValueType,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def add(
    img: NDArray[np.float32],
    value: ValueType,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def add(
    img: torch.Tensor,
    value: ValueType | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def add(
    img: ImageType | torch.Tensor,
    value: ValueType | torch.Tensor,
    *,
    out: ImageType | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Add a scalar, channel vector or broadcastable field.

    Without ``out``, input storage is unchanged; views and identity are allowed.
    A supplied writable output has the input shape, dtype, container and layout and is returned
    by identity. Exact image/field aliases are supported; other overlaps, including broadcast
    controls sharing output storage, are outside the caller-validated contract.
    NumPy uses channel-last images, batches and volumes. Tensor calls require an explicit layout.
    Uint8 saturates to [0, 255]; floating results keep their raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        tensor_out = cast("torch.Tensor | None", out)
        if isinstance(value, torch.Tensor) and value.ndim == img.ndim and img.dtype in (torch.float16, torch.float32):
            result = torch.add(img, value) if tensor_out is None else torch.add(img, value, out=tensor_out)
            return result if result.dtype == img.dtype else result.to(img.dtype)
        return _tensor_pointwise(img, value, "add", layout, out=tensor_out)
    destination = cast("ImageType | None", out)
    value = _prepare_pointwise_operand(img, cast("ValueType", value))
    if img.size == 0:
        return img if destination is None else destination
    if isinstance(value, (float, int)):
        if value == 0:
            if destination is not None:
                np.copyto(destination, img)
            return img if destination is None else destination
        if img.dtype == np.uint8:
            value = int(value)
        return _add_scalar(img, value, out=destination)
    return _add_channels(img, value, out=destination) if value.ndim == 1 else _add_field(img, value, out=destination)


def _power_opencv(
    img: NDArray[np.float16] | NDArray[np.float32],
    exponent: float,
    *,
    out: NDArray[np.float16] | NDArray[np.float32] | None,
) -> NDArray[np.float16] | NDArray[np.float32]:
    if _is_float16_image(img):
        result = cast("NDArray[np.float32]", cv2.pow(_cast_float_numpy(img, np.float32), exponent).reshape(img.shape))
        return _cast_float_numpy(result, np.float16, out=cast("NDArray[np.float16] | None", out))
    if out is None:
        return cast("NDArray[np.float32]", cv2.pow(img, exponent).reshape(img.shape))
    if out.flags.c_contiguous and out is not img:
        source = img.reshape(-1) if img.flags.c_contiguous else img
        cv2.pow(source, exponent, dst=out.reshape(-1).reshape(source.shape))
    else:
        # Allocating and copying beats OpenCV's exact-alias power kernel on the measured ARM64 route.
        np.copyto(out, cv2.pow(img, exponent).reshape(img.shape))
    return out


@overload
def power(
    img: NDArray[np.uint8],
    exponent: ValueType,
    *,
    out: NDArray[np.uint8] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8]: ...


@overload
def power(
    img: NDArray[np.float16],
    exponent: ValueType,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def power(
    img: NDArray[np.float32],
    exponent: ValueType,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def power(
    img: torch.Tensor,
    exponent: ValueType | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def power(
    img: ImageType | torch.Tensor,
    exponent: ValueType | torch.Tensor,
    *,
    out: ImageType | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Raise pixels to a power, preserving shape, dtype and container.

    Uint8 scalar/channel exponents use clipped float32 lookup tables and truncate to uint8.
    Floating scalar exponents use OpenCV; half results compute in float32 and round once.
    Floating channel/field exponents use the measured NumPy or eager CPU half Torch route.

    Args:
        img: Channel-last NumPy image, batch or volume, or a CPU Tensor in the declared layout.
        exponent: Scalar or channel vector; floating inputs also accept a broadcastable field.
        out: Optional writable destination with the input shape, dtype, container and layout.
            Exact input aliasing is supported; other overlaps are outside the caller-validated contract.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        The supplied destination by identity. Without ``out``, input storage is unchanged;
        empty inputs may reuse their storage.
        Uint8 saturates to [0, 255]; floating results preserve their raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        parameter = tensor_operand_to_numpy(exponent, layout)
        image_view = tensor_to_numpy_channel_last(img, layout)
        tensor_out = cast("torch.Tensor | None", out)
        output_view = None
        if tensor_out is not None:
            output_view = image_view if tensor_out is img else tensor_to_numpy_channel_last(tensor_out, layout)
        result = power(image_view, parameter, out=output_view)
        return numpy_to_tensor_layout(result, cast("TensorLayout", layout)) if tensor_out is None else tensor_out

    num_channels = get_num_channels(img)
    if img.size == 0:
        return img if out is None else out
    exponent = convert_value(exponent, num_channels)
    if _is_uint8_image(img):
        return _apply_lut(img, exponent, "power", out=cast("NDArray[np.uint8] | None", out))

    output = cast("NDArray[np.float16] | NDArray[np.float32] | None", out)
    floating_image = cast("NDArray[np.float16] | NDArray[np.float32]", img)
    if isinstance(exponent, (float, int)):
        return _power_opencv(floating_image, float(exponent), out=output)

    if _is_float16_image(img):
        torch_result = float16_pointwise_torch(
            img,
            "power",
            exponent,
            minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
            out=cast("NDArray[np.float16] | None", output),
        )
        if torch_result is not None:
            return torch_result
    return _numpy_pointwise(floating_image, exponent, "power", out=output)


def _torch_half_control(
    value: ValueType | torch.Tensor,
    image: torch.Tensor,
    layout: TensorLayout,
) -> torch.Tensor | float | None:
    operand = tensor_operand_to_numpy(value, layout)
    if isinstance(operand, np.ndarray):
        if operand.ndim == 0:
            operand = operand.item()
        elif operand.ndim == 1 and operand.size == get_num_channels(image, layout=layout):
            with np.errstate(over="ignore", invalid="ignore"):
                half = operand.astype(np.float16)
            if not np.array_equal(operand.astype(np.float64), half.astype(np.float64)):
                return None
            channel_axis = 0 if layout in ("CHW", "CDHW") else 1
            broadcast_shape = [1] * image.ndim
            broadcast_shape[channel_axis] = operand.size
            return torch.from_numpy(np.array(half, copy=True)).reshape(broadcast_shape)
        else:
            return None

    try:
        scalar = float(operand)
    except (TypeError, ValueError):
        return None
    return scalar


def _torch_half_pointwise_tensor(
    image: torch.Tensor,
    value: ValueType | torch.Tensor,
    layout: TensorLayout | None,
    operation: Literal["add", "multiply"],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if (
        layout is None
        or image.dtype != torch.float16
        or not image.is_contiguous()
        or image.numel() < _TORCH_FLOAT16_MIN_ELEMENTS
    ):
        return None
    control = _torch_half_control(value, image, layout)
    if control is None:
        return None
    return torch.add(image, control, out=out) if operation == "add" else torch.mul(image, control, out=out)


def _tensor_pointwise(
    image: torch.Tensor,
    value: ValueType | torch.Tensor,
    operation: Literal["add", "multiply"],
    layout: TensorLayout | None,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    operand = tensor_operand_to_numpy(value, layout)
    if operation == "add" and np.asarray(operand).ndim == 0 and float(operand) == 0.0:
        if out is not None:
            out.copy_(image)
            return out
        return image
    half_result = _torch_half_pointwise_tensor(image, operand, layout, operation, out=out)
    if half_result is not None:
        return half_result
    image_view = tensor_to_numpy_channel_last(image, layout)
    destination = None if out is None else image_view if out is image else tensor_to_numpy_channel_last(out, layout)
    numpy_operation = add if operation == "add" else multiply
    result_array = numpy_operation(image_view, operand, out=destination)
    if out is not None:
        return out
    return image if result_array is image_view else numpy_to_tensor_layout(result_array, cast("TensorLayout", layout))


def _is_all_zero_param(x: _PreparedValue) -> bool:
    if isinstance(x, np.ndarray):
        return not np.any(x)
    return float(x) == 0.0


def _use_numkong_scalar_multiply_add(img: ImageType, factor: ValueType, value: ValueType) -> bool:
    return (
        _is_float32_image(img)
        and isinstance(factor, (float, int))
        and isinstance(value, (float, int))
        and -_FLOAT32_MAX <= factor <= _FLOAT32_MAX
        and -_FLOAT32_MAX <= value <= _FLOAT32_MAX
    )


def _multiply_add_numpy(
    img: ImageType,
    factor: ValueType,
    value: ValueType,
) -> NDArray[np.float32]:
    img_f = _cast_float_numpy(img, np.float32)
    factor_prepared = _prepare_numpy_value(factor)
    value_prepared = _prepare_numpy_value(value)

    if not isinstance(factor_prepared, np.ndarray) and not isinstance(value_prepared, np.ndarray):
        if factor_prepared == 0.0 and value_prepared == 0.0:
            return np.zeros_like(img_f, dtype=np.float32)
        if factor_prepared == 1.0 and value_prepared == 0.0:
            return img_f

    n_dim = img_f.ndim
    c = int(img_f.shape[-1])
    f_b = _broadcast_channel_vector(factor_prepared, n_dim, c)
    v_b = _broadcast_channel_vector(value_prepared, n_dim, c)

    multiply_result = np.zeros_like(img_f) if _is_all_zero_param(f_b) else np.multiply(img_f, cast("Any", f_b))
    multiply_result = multiply_result if _is_all_zero_param(v_b) else np.add(multiply_result, cast("Any", v_b))
    if multiply_result.dtype == np.float32:
        return cast("NDArray[np.float32]", multiply_result)
    with np.errstate(over="ignore"):
        return cast("NDArray[np.float32]", multiply_result.astype(np.float32, copy=False))


def _multiply_add_lut(
    img: NDArray[np.uint8],
    factor: float | np.ndarray,
    value: float | np.ndarray,
    *,
    out: NDArray[np.uint8] | None,
) -> NDArray[np.uint8]:
    domain = np.arange(256, dtype=np.float32)
    if isinstance(factor, (float, int)) and isinstance(value, (float, int)):
        table = clip(domain * factor + value, np.uint8)
    else:
        if isinstance(factor, np.ndarray) and factor.shape != ():
            factor = factor.reshape(1, 1, -1)
        if isinstance(value, np.ndarray) and value.shape != ():
            value = value.reshape(1, 1, -1)
        table = clip(np.asarray(domain.reshape(-1, 1, 1) * factor + value, dtype=np.float32), np.uint8)
    return _apply_uint8_lut(img, table, out=out)


def _multiply_add_half(
    img: NDArray[np.float16],
    factor: float | np.ndarray,
    value: float | np.ndarray,
    *,
    out: NDArray[np.float16] | None,
) -> NDArray[np.float16]:
    numkong_factor = _half_exact_scalar(factor)
    numkong_value = _half_exact_scalar(value)
    if (
        _NUMKONG_NEON_HALF
        and img.flags.c_contiguous
        and img.size >= _NUMKONG_FLOAT16_MIN_ELEMENTS
        and numkong_factor is not None
        and numkong_value is not None
    ):
        return _scale_numkong(img, numkong_factor, numkong_value, out=out)
    result = multiply_add_float16_torch(
        img,
        factor,
        value,
        minimum_elements=_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS,
        out=out,
    )
    if result is not None:
        return result
    target = (
        out if out is not None and not isinstance(value, np.ndarray) and not abs(float(value)) > _FLOAT32_MAX else None
    )
    product = _float16_numpy_ufunc(img, factor, "multiply", out=target)
    if product is not None:
        result = _float16_numpy_ufunc(product, value, "add", out=product if out is None else out)
        if result is not None:
            return result
    del product
    return _cast_float_numpy(_multiply_add_numpy(img, factor, value), np.float16, out=out)


def _multiply_add_tensor_batch(
    img: torch.Tensor,
    factor: float | np.ndarray,
    value: float | np.ndarray,
    layout: TensorLayout,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    image = tensor_to_numpy_channel_last(img, layout)
    destination = None if out is None else image if out is img else tensor_to_numpy_channel_last(out, layout)
    results = []
    unchanged = True
    for index, item in enumerate(image):
        result = multiply_add(
            item,
            _batch_operand(factor, image, index),
            _batch_operand(value, image, index),
            out=None if destination is None else destination[index],
        )
        unchanged = unchanged and result is item
        results.append(result)
    if out is not None:
        return out
    return img if unchanged else numpy_to_tensor_layout(np.stack(results), layout)


def _multiply_add_scalar_tensor(
    img: torch.Tensor,
    factor: float,
    value: float,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Avoid channel packing for axis-independent scalars; keep the selected NumPy kernel."""
    image = img.numpy().reshape(1, -1, 1)
    destination = None if out is None else image if out is img else out.numpy().reshape(image.shape)
    result = multiply_add(image, factor, value, out=destination)
    if out is not None:
        return out
    return img if result is image else numpy_to_tensor_storage(result).reshape(img.shape)


def _multiply_add_tensor(
    img: torch.Tensor,
    factor: ValueType | torch.Tensor,
    value: ValueType | torch.Tensor,
    layout: TensorLayout,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if img.dtype == torch.float16:
        half_factor = _torch_half_control(factor, img, layout)
        half_value = _torch_half_control(value, img, layout)
        if half_factor is not None and half_value is not None:
            product = torch.mul(img, half_factor)
            return torch.add(product, half_value) if out is None else torch.add(product, half_value, out=out)
    factor_value = tensor_operand_to_numpy(factor, layout)
    bias_value = tensor_operand_to_numpy(value, layout)
    if layout in ("NCHW", "NCDHW") and img.shape[0]:
        return _multiply_add_tensor_batch(img, factor_value, bias_value, layout, out=out)
    if (
        img.dtype == torch.float32
        and isinstance(factor_value, (float, int))
        and isinstance(bias_value, (float, int))
        and img.is_contiguous()
        and (out is None or out.is_contiguous())
    ):
        return _multiply_add_scalar_tensor(img, factor_value, bias_value, out=out)
    image_view = tensor_to_numpy_channel_last(img, layout)
    destination = None if out is None else image_view if out is img else tensor_to_numpy_channel_last(out, layout)
    result = multiply_add(image_view, factor_value, bias_value, out=destination)
    if out is not None:
        return out
    return img if result is image_view else numpy_to_tensor_layout(result, layout)


@overload
def multiply_add(
    img: NDArray[np.uint8],
    factor: ValueType,
    value: ValueType,
    *,
    out: NDArray[np.uint8] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.uint8]: ...


@overload
def multiply_add(
    img: NDArray[np.float16],
    factor: ValueType,
    value: ValueType,
    *,
    out: NDArray[np.float16] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float16]: ...


@overload
def multiply_add(
    img: NDArray[np.float32],
    factor: ValueType,
    value: ValueType,
    *,
    out: NDArray[np.float32] | None = None,
    layout: TensorLayout | None = None,
) -> NDArray[np.float32]: ...


@overload
def multiply_add(
    img: torch.Tensor,
    factor: ValueType | torch.Tensor,
    value: ValueType | torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    layout: TensorLayout,
) -> torch.Tensor: ...


def multiply_add(
    img: ImageType | torch.Tensor,
    factor: ValueType | torch.Tensor,
    value: ValueType | torch.Tensor,
    *,
    out: ImageType | torch.Tensor | None = None,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Compute ``img * factor + value`` with the existing fused or two-rounding route.

    Uint8 scalar/channel controls saturate and truncate one float32 table.
    Floating images also accept broadcastable fields and preserve their raw range.
    Half Torch/NumPy kernels round the product before addition; the selected
    NumKong scalar kernel retains its fused rounding.

    Without ``out``, inputs stay unchanged and identity results are allowed.
    A supplied output matches the image dtype, shape, container and layout and is
    returned by identity. Writable strides and exact image/full-field aliases are
    supported; other overlapping views and broadcast controls sharing output are
    outside the caller-validated contract. Some routes keep an intermediate buffer.
    NumPy is channel-last; Tensor calls require an explicit layout.
    """
    if isinstance(img, torch.Tensor):
        return _multiply_add_tensor(
            img,
            factor,
            value,
            cast("TensorLayout", layout),
            out=cast("torch.Tensor | None", out),
        )
    destination = cast("ImageType | None", out)
    if img.size == 0:
        return img if destination is None else destination
    channels = get_num_channels(img)
    factor = convert_value(factor, channels)
    value = convert_value(value, channels)
    if _is_uint8_image(img):
        return _multiply_add_lut(img, factor, value, out=cast("NDArray[np.uint8] | None", destination))
    if _is_float16_image(img):
        return _multiply_add_half(img, factor, value, out=cast("NDArray[np.float16] | None", destination))
    image = cast("NDArray[np.float32]", img)
    if isinstance(factor, (float, int)) and isinstance(value, (float, int)) and factor == 1 and value == 0:
        return (
            image
            if destination is None or destination is image
            else _cast_float_numpy(image, np.float32, out=cast("NDArray[np.float32]", destination))
        )
    if _use_numkong_scalar_multiply_add(image, factor, value):
        result = multiply_add_float32_torch(
            image,
            float(factor),
            float(value),
            out=cast("NDArray[np.float32] | None", destination),
        )
        if result is None:
            result = _scale_numkong(
                image,
                float(factor),
                float(value),
                out=cast("NDArray[np.float32] | None", destination),
            )
    else:
        result = _multiply_add_numpy(image, factor, value)
    if destination is not None and result is not destination:
        np.copyto(destination, result)
    return result if destination is None else destination
