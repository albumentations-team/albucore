"""Arithmetic: add, multiply, power, value-normalize, weighted sum, multiply-add."""

import math
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypeGuard, cast

import cv2
import numkong as nk
import numpy as np
import torch

from albucore._float_cast import _cast_float_numpy
from albucore.decorators import preserve_channel_dim
from albucore.lut import _apply_float_lut
from albucore.lut import apply_uint8_lut as _apply_uint8_lut
from albucore.tensor_bridge import (
    TensorLayout,
    dispatch_tensor_image,
    numpy_to_tensor_layout,
    tensor_operand_to_numpy,
    tensor_to_numpy_channel_last,
)
from albucore.torch_backend import (
    float16_pointwise_torch,
    multiply_add_float16_torch,
    multiply_add_float32_torch,
    normalize_float16_torch,
    normalize_float32_torch,
)
from albucore.utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    ImageFloat16,
    ImageFloat32,
    ImageType,
    ImageUInt8,
    SupportedDType,
    ValueType,
    clip,
    convert_value,
    get_num_channels,
    get_opencv_max_channels,
)
from albucore.weighted import (
    _NUMKONG_NEON_HALF,
    add_array_numkong,
    add_weighted_numkong,
    multiply_add_numkong,
)

np_operations = {"multiply": np.multiply, "add": np.add, "power": np.power}

cv2_operations = {"multiply": cv2.multiply, "add": cv2.add, "power": cv2.pow}

_PreparedValue: TypeAlias = float | int | np.floating[Any] | np.ndarray
_FLOAT32_COMPATIBLE_OPERAND_DTYPES: frozenset[np.dtype[Any]] = frozenset(
    np.dtype(dtype) for dtype in (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.float16, np.float32)
)
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_TORCH_FLOAT16_MIN_ELEMENTS = 16_384
_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS = 230_400
_TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS = 16_384
_NUMKONG_FLOAT16_MIN_ELEMENTS = 20_480
_NUMPY_FLOAT16_C9_VECTOR_MIN_ELEMENTS = 184_320
_CV16F_VECTOR_MIN_ELEMENTS = 61_440
_FLOAT16_MIN_NORMAL = float(np.finfo(np.float16).tiny)
_CV16F_ADD_WEIGHTED_MAX_CHANNELS = get_opencv_max_channels()


def _is_uint8_image(img: ImageType) -> TypeGuard[ImageUInt8]:
    return img.dtype == np.uint8


def _is_float32_image(img: ImageType) -> TypeGuard[ImageFloat32]:
    return img.dtype == np.float32


def _is_float16_image(img: ImageType) -> TypeGuard[ImageFloat16]:
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
    img: ImageFloat16,
    value: object,
    operation: Literal["add", "multiply"],
) -> ImageFloat16 | None:
    if (
        not img.flags.c_contiguous
        or not img.flags.writeable
        or img.ndim != 3
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
    if operation == "add":
        result = cv2.add(img, scalar, dtype=cv2.CV_16F)
    else:
        result = cv2.multiply(img, scalar, dtype=cv2.CV_16F)
    return cast("ImageFloat16", result)


def _numpy_float16_c9_vector_eligible(img: ImageFloat16, value: object, inplace: bool) -> bool:
    return (
        not inplace
        and img.flags.c_contiguous
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
    img: ImageFloat16,
    value: object,
    inplace: bool,
) -> ImageFloat16 | None:
    if not isinstance(value, np.ndarray) or not _numpy_float16_c9_vector_eligible(img, value, inplace):
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
    result = cv2.multiply(img, control_view, dtype=cv2.CV_16F)
    return cast("ImageFloat16", result)


def _numkong_float16_c9_add_vector(img: ImageFloat16, value: object, inplace: bool) -> ImageFloat16 | None:
    if (
        not _NUMKONG_NEON_HALF
        or inplace
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
    result = nk.add(img, control_image)
    if result is None:
        raise RuntimeError("nk.add returned None")
    return cast("ImageFloat16", np.asarray(result).reshape(img.shape))


def _numkong_float16_single_channel_scale(
    img: ImageFloat16,
    value: object,
    operation: Literal["add", "multiply"],
) -> ImageFloat16 | None:
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
    return cast("ImageFloat16", np.asarray(result).reshape(img.shape))


def _restore_float_dtype(image: ImageType, result: ImageFloat32) -> ImageFloat16 | ImageFloat32:
    return _cast_float_numpy(result, np.float16) if _is_float16_image(image) else result


def create_lut_array(
    dtype: SupportedDType,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
) -> np.ndarray:
    op = np_operations.get(operation)
    if op is None:
        raise ValueError(f"Unsupported operation: {operation}")

    if dtype == np.uint8 and operation == "add":
        value = np.trunc(value)

    lut = np.arange(int(MAX_VALUES_BY_DTYPE[dtype]) + 1, dtype=np.float32)
    value_arr = np.asarray(value, dtype=np.float32)
    if value_arr.ndim == 0:
        return op(lut, value_arr)
    return op(lut.reshape(-1, 1, 1), value_arr.reshape(1, 1, -1))


def _float16_numpy_ufunc(
    img: ImageFloat16,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    *,
    inplace: bool = False,
) -> ImageFloat16 | None:
    """Run a float32 ufunc loop directly into a half output buffer for supported operands."""
    if isinstance(value, np.ndarray) and value.dtype not in _FLOAT32_COMPATIBLE_OPERAND_DTYPES:
        return None
    if not isinstance(value, np.ndarray) and abs(float(value)) > _FLOAT32_MAX:
        return None

    can_mutate = inplace and img.flags.owndata and img.flags.writeable
    output = img if can_mutate else np.empty_like(img)
    with np.errstate(over="ignore", invalid="ignore"):
        np_operations[operation](img, value, dtype=np.float32, out=output, casting="unsafe")
    return output


def apply_lut(
    img: ImageUInt8,
    value: float | np.ndarray,
    operation: Literal["add", "multiply", "power"],
    inplace: bool,
) -> ImageUInt8:
    """Apply lookup table operation. Only works with uint8 images.

    ``inplace=True`` avoids an extra buffer when the whole image uses one LUT (scalar path) or when
    callers write channel-wise into a preallocated ``result`` (see per-channel loop).
    """
    dtype = img.dtype

    if isinstance(value, (int, float)):
        lut = clip(create_lut_array(dtype, value, operation), dtype, inplace=True)
        return _apply_uint8_lut(img, lut, inplace=inplace)

    luts = clip(create_lut_array(dtype, value, operation), dtype, inplace=True)
    return _apply_uint8_lut(img, luts, inplace=inplace)


def prepare_value_opencv(
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
            return cast("np.ndarray", clip(cast("ImageType", value), np.dtype(np.uint8), inplace=False))
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


def apply_numpy(
    img: ImageType,
    value: ValueType | np.floating[Any],
    operation: Literal["add", "multiply", "power"],
) -> ImageFloat32:
    if operation == "add" and img.dtype == np.uint8:
        value_prepared: _PreparedValue = np.asarray(value, dtype=np.int16)
    else:
        value_prepared = _prepare_numpy_value(value)

    result = np_operations[operation](_cast_float_numpy(img, np.float32), value_prepared)
    if result.dtype == np.float32:
        return cast("ImageFloat32", result)
    with np.errstate(over="ignore"):
        return cast("ImageFloat32", result.astype(np.float32, copy=False))


def multiply_lut(img: ImageUInt8, value: np.ndarray | float, inplace: bool = False) -> ImageUInt8:
    return apply_lut(img, value, "multiply", inplace)


@preserve_channel_dim
def multiply_opencv(img: ImageType, value: np.ndarray | float) -> ImageFloat32:
    value = prepare_value_opencv(img, value, "multiply")
    value_cv2: Any = value
    if _is_uint8_image(img):
        if isinstance(value, np.ndarray):
            value_cv2 = value.astype(np.float32, copy=False)
        return cast("ImageFloat32", cv2.multiply(img.astype(np.float32, copy=False), value_cv2))
    return cast("ImageFloat32", cv2.multiply(img, value_cv2))


def multiply_numpy(
    img: ImageType,
    value: float | np.ndarray,
    *,
    inplace: bool = False,
) -> ImageFloat16 | ImageFloat32:
    if _is_float16_image(img):
        float16_result = _float16_numpy_ufunc(img, value, "multiply", inplace=inplace)
        if float16_result is not None:
            return float16_result
    result = apply_numpy(img, value, "multiply")
    return _cast_float_numpy(result, np.float16) if _is_float16_image(img) else result


def multiply_by_constant(
    img: ImageType | torch.Tensor,
    value: float | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    if isinstance(img, torch.Tensor):
        return _dispatch_tensor_multiply(img, value, layout, multiply_by_constant, inplace)
    if _is_uint8_image(img):
        return multiply_lut(img, value, inplace)
    if _is_float16_image(img):
        torch_result = float16_pointwise_torch(img, "multiply", value, minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS)
        if torch_result is not None:
            return cast("ImageType", torch_result)
        return cast("ImageFloat16", multiply_numpy(img, value, inplace=inplace))
    # float32: match 0.0.40 (`multiply_numpy`); OpenCV / NumKong scalar paths regressed on router grid
    return multiply_numpy(img, value)


def multiply_by_vector(
    img: ImageType | torch.Tensor,
    value: np.ndarray | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    if isinstance(img, torch.Tensor):
        return _dispatch_tensor_multiply(img, value, layout, multiply_by_vector, inplace)
    if _is_uint8_image(img):
        # LUT beats OpenCV float-multiply + clip for per-channel uint8 (see benchmarks/benchmark_grayscale_paths.py).
        return multiply_lut(img, value, inplace)
    if _is_float16_image(img):
        half_result = _opencv_float16_three_channel_vector(img, value, "multiply")
        if half_result is None:
            half_result = _opencv_float16_c9_multiply_vector(img, value, inplace)
        if half_result is None and _numpy_float16_c9_vector_eligible(img, value, inplace):
            half_result = cast("ImageFloat16", multiply_numpy(img, value))
        if half_result is None:
            half_result = _numkong_float16_single_channel_scale(img, value, "multiply")
        if half_result is None:
            half_result = float16_pointwise_torch(img, "multiply", value, minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS)
        if half_result is None:
            half_result = cast("ImageFloat16", multiply_numpy(img, value, inplace=inplace))
        return half_result
    return multiply_numpy(img, value)


def multiply_by_array(
    img: ImageType | torch.Tensor,
    value: np.ndarray | torch.Tensor,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Multiply by a full-shape operand, preserving the image dtype.

    Equal-dtype float16/float32 Tensor operands use direct Torch elementwise multiplication.
    """
    if isinstance(img, torch.Tensor):
        if (
            layout is not None
            and isinstance(value, torch.Tensor)
            and value.dtype == img.dtype
            and img.dtype in (torch.float16, torch.float32)
        ):
            return torch.mul(img, value)
        operand = tensor_operand_to_numpy(value, layout)
        return dispatch_tensor_image(img, layout, multiply_by_array, operand)
    if _is_float32_image(img):
        return multiply_numpy(img, value)
    if _is_float16_image(img):
        if isinstance(value, np.ndarray) and value.dtype == np.float32 and value.ndim == img.ndim:
            half_result: np.ndarray | None = cast("np.ndarray", multiply_numpy(img, value))
        else:
            half_result = float16_pointwise_torch(img, "multiply", value, minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS)
            if half_result is None:
                half_result = cast("np.ndarray", multiply_numpy(img, value))
        return cast("ImageType", half_result)
    return clip(multiply_opencv(img, value), img.dtype)


def multiply(
    img: ImageType | torch.Tensor,
    value: ValueType | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Multiply image pixels by a scalar, per-channel vector, or full array.

    Routes to the fastest backend per dtype and value shape:

    - **uint8 scalar / vector**: LUT (precomputed 256-entry table, clipped to [0, 255]).
    - **uint8 array** (same HxWxC shape): OpenCV ``multiply`` then clip.
    - **float16 scalar and half-compatible operands**: eager CPU Torch from 16,384 elements; other operands use the
      NumPy float32 ufunc loop and write directly to a half output.
    - **float32 scalar / vector / array**: raw NumPy broadcast multiply.

    Alternative: ``multiply_by_constant``, ``multiply_by_vector``, ``multiply_by_array``
    for explicit routing without dtype dispatch.

    Args:
        img: ``(H, W, C)`` image, uint8, float16, or float32.
        value: Scalar, length-``C`` 1-D array, or full ``img``-shaped array.
        inplace: Mutate ``img`` in-place when the LUT/NumPy path allows it.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Pixel-wise product with the same shape and dtype as ``img``. Uint8 results saturate to
        [0, 255]; floating results preserve the raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        return _dispatch_tensor_multiply(img, value, layout, multiply, inplace)

    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        return multiply_by_constant(img, value, inplace)

    if isinstance(value, np.ndarray) and value.ndim == 1:
        return multiply_by_vector(img, value, inplace)

    return multiply_by_array(img, value)


@preserve_channel_dim
def add_opencv(img: ImageType, value: np.ndarray | float, inplace: bool = False) -> ImageType:
    value = prepare_value_opencv(img, value, "add")

    # Convert to float32 if:
    # 1. uint8 image with negative scalar value
    # 2. uint8 image with non-uint8 array value
    needs_float = img.dtype == np.uint8 and (
        (isinstance(value, (int, float)) and value < 0) or (isinstance(value, np.ndarray) and value.dtype != np.uint8)
    )

    if needs_float:
        value_cv2_float: Any = value if isinstance(value, (int, float)) else value.astype(np.float32, copy=False)
        return cast(
            "ImageType",
            cv2.add(
                img.astype(np.float32, copy=False),
                value_cv2_float,
            ),
        )

    dst = img if inplace else None
    value_cv2: Any = value
    return cast("ImageType", cv2.add(img, value_cv2, dst=dst))


def add_numpy(
    img: ImageType,
    value: float | np.ndarray,
    *,
    inplace: bool = False,
) -> ImageFloat16 | ImageFloat32:
    if _is_float16_image(img):
        float16_result = _float16_numpy_ufunc(img, value, "add", inplace=inplace)
        if float16_result is not None:
            return float16_result
    result = apply_numpy(img, value, "add")
    return _cast_float_numpy(result, np.float16) if _is_float16_image(img) else result


def add_lut(img: ImageUInt8, value: np.ndarray | float, inplace: bool = False) -> ImageUInt8:
    return apply_lut(img, value, "add", inplace)


def add_constant(
    img: ImageType | torch.Tensor,
    value: float | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    if isinstance(img, torch.Tensor):
        return _dispatch_tensor_add(img, value, layout, add_constant, inplace)
    if _is_float32_image(img):
        return add_numpy(img, value)
    if _is_float16_image(img):
        torch_result = float16_pointwise_torch(img, "add", value, minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS)
        if torch_result is not None:
            return cast("ImageType", torch_result)
        return cast("ImageFloat16", add_numpy(img, value, inplace=inplace))
    # uint8 (all C): OpenCV + ``prepare_value_opencv`` broadcast. NumKong ``add_constant_numkong``
    # regressed vs 0.0.40 on large (H,W,C>4) router cells; helper remains in ``weighted`` for benches.
    result = add_opencv(img, value, inplace)
    return result if result.dtype == np.uint8 else clip(result, img.dtype)


def add_vector(
    img: ImageType | torch.Tensor,
    value: np.ndarray | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    if isinstance(img, torch.Tensor):
        return _dispatch_tensor_add(img, value, layout, add_vector, inplace)
    if _is_uint8_image(img):
        return add_lut(img, value, inplace)
    if _is_float16_image(img):
        half_result = _opencv_float16_three_channel_vector(img, value, "add")
        if half_result is None:
            half_result = _numkong_float16_c9_add_vector(img, value, inplace)
        if half_result is None and _numpy_float16_c9_vector_eligible(img, value, inplace):
            half_result = cast("ImageFloat16", add_numpy(img, value))
        if half_result is None:
            half_result = _numkong_float16_single_channel_scale(img, value, "add")
        if half_result is None:
            half_result = float16_pointwise_torch(img, "add", value, minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS)
        if half_result is None:
            half_result = cast("ImageFloat16", add_numpy(img, value, inplace=inplace))
        return half_result
    return add_numpy(img, value)


def add_array(
    img: ImageType | torch.Tensor,
    value: np.ndarray | torch.Tensor,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Elementwise ``img + value`` for a full array ``value``.

    NumPy float32 returns the raw NumPy sum. Floating Tensor-Tensor inputs use native Torch addition and preserve the
    image dtype. Uint8 uses saturating arithmetic: same-shape and same-dtype operands route to NumKong ``blend``;
    other operand layouts route to OpenCV.
    There is no ``inplace`` flag: an in-place OpenCV path was slower than the NumKong out-of-place
    path for same-shape uint8 in benchmarks; use ``numpy.copyto`` yourself if you must reuse a buffer.
    """
    if isinstance(img, torch.Tensor):
        if layout is not None and isinstance(value, torch.Tensor) and img.dtype in (torch.float16, torch.float32):
            tensor_result = torch.add(img, value)
            return tensor_result if tensor_result.dtype == img.dtype else tensor_result.to(img.dtype)
        operand = tensor_operand_to_numpy(value, layout)
        return dispatch_tensor_image(img, layout, add_array, operand)

    if _is_float32_image(img):
        # Benchmarks: NumPy broadcast add beats NumKong and OpenCV scalar/tensor prep on float32.
        return add_numpy(img, value)
    if _is_float16_image(img):
        half_result: np.ndarray | None
        if isinstance(value, np.ndarray) and value.dtype == np.float32 and value.ndim == img.ndim:
            half_result = cast("np.ndarray", add_numpy(img, value))
        else:
            half_result = float16_pointwise_torch(img, "add", value, minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS)
        if half_result is None:
            half_result = cast("np.ndarray", add_numpy(img, value))
        return cast("ImageType", half_result)
    if value.shape == img.shape and value.dtype == img.dtype:
        return add_array_numkong(img, value)
    result = add_opencv(img, value)
    return result if result.dtype == np.uint8 else clip(result, img.dtype)


def add(
    img: ImageType | torch.Tensor,
    value: ValueType | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Add a scalar, per-channel vector, or full array to image pixels.

    Routes to the fastest backend per dtype and value shape:

    - **uint8 scalar**: OpenCV ``add`` (saturate arithmetic, clipped to [0, 255]).
      Zero-value short-circuits with a no-op.
    - **uint8 vector**: LUT (one 256-entry table per channel).
    - **uint8 array**: NumKong ``blend`` when shapes match; otherwise OpenCV.
    - **float16**: a benchmark-gated eager CPU Torch path for large arrays; remaining sizes use
      float32 NumPy arithmetic and one final cast.
    - **float32 scalar / vector / array**: raw NumPy broadcast add.

    Alternative: ``add_constant``, ``add_vector``, ``add_array`` for explicit routing.

    Args:
        img: ``(H, W, C)`` image, uint8, float16, or float32.
        value: Scalar, length-``C`` 1-D array, or full ``img``-shaped array.
        inplace: Mutate ``img`` in-place when the backend supports it (scalar/vector paths).
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Pixel-wise sum with the same shape and dtype as ``img``. Uint8 results saturate to
        [0, 255]; floating results preserve the raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        numpy_value = tensor_operand_to_numpy(value, layout)
        if not inplace and np.asarray(numpy_value).ndim == 0 and float(numpy_value) == 0.0:
            return img
        return _dispatch_tensor_add(img, numpy_value, layout, add, inplace)

    num_channels = get_num_channels(img)
    value = convert_value(value, num_channels)

    if isinstance(value, (float, int)):
        if value == 0:
            return img

        if img.dtype == np.uint8:
            value = int(value)

        return add_constant(img, value, inplace)

    return add_vector(img, value, inplace) if value.ndim == 1 else add_array(img, value)


def normalize_numpy(
    img: ImageType,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
) -> ImageFloat16 | ImageFloat32:
    mean_f = mean.astype(np.float32, copy=False) if isinstance(mean, np.ndarray) else np.float32(mean)
    denom_f = (
        denominator.astype(np.float32, copy=False) if isinstance(denominator, np.ndarray) else np.float32(denominator)
    )
    # Reuse the float32 product buffer for subtraction instead of allocating a second image-sized result.
    if isinstance(mean_f, np.ndarray) or isinstance(denom_f, np.ndarray):
        n_dim = img.ndim
        c = int(img.shape[-1])

        def _broadcast_last(x: np.ndarray | np.float32) -> np.ndarray | np.float32:
            if isinstance(x, np.ndarray) and x.shape == (c,):
                return x.reshape((1,) * (n_dim - 1) + (c,))
            return x

        mean_b = _broadcast_last(mean_f)
        denom_b = _broadcast_last(denom_f)
        offset = mean_b * denom_b
        result = cast("ImageFloat32", np.multiply(img, denom_b, dtype=np.float32))
        np.subtract(result, offset, out=result)
        return _restore_float_dtype(img, result)

    result = cast("ImageFloat32", np.multiply(img, denom_f, dtype=np.float32))
    np.subtract(result, mean_f * denom_f, out=result)
    return _restore_float_dtype(img, result)


def _normalize_float16_single_channel_opencv(
    img: ImageFloat16,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
) -> ImageFloat16 | None:
    if isinstance(mean, np.ndarray):
        if mean.size != 1:
            return None
        mean_value = float(mean.item())
    else:
        mean_value = float(mean)
    if isinstance(denominator, np.ndarray):
        if denominator.size != 1:
            return None
        denominator_value = float(denominator.item())
    else:
        denominator_value = float(denominator)
    if (
        not math.isfinite(mean_value)
        or abs(mean_value) > _FLOAT32_MAX
        or not math.isfinite(denominator_value)
        or abs(denominator_value) > _FLOAT32_MAX
    ):
        return None
    bias = -mean_value * denominator_value
    if abs(bias) > _FLOAT32_MAX:
        return None
    result = cv2.addWeighted(img, denominator_value, img, 0.0, bias)
    if result.shape != img.shape:
        result = result.reshape(img.shape)
    return cast("ImageFloat16", result)


def _normalize_float16_channel_values(value: float | np.ndarray, channels: int) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != channels:
            return None
        values = value.astype(np.float32, copy=False)
    else:
        values = np.full(channels, np.float32(value), dtype=np.float32)
    return values if np.isfinite(values).all() else None


def _normalize_float16_tensor_controls(
    mean: object,
    denominator: object,
    channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mean_values = _normalize_float16_channel_values(convert_value(mean, channels), channels)
    denominator_values = _normalize_float16_channel_values(convert_value(denominator, channels), channels)
    if mean_values is None or denominator_values is None:
        return None
    biases = -mean_values * denominator_values
    if not np.isfinite(biases).all():
        return None
    return mean_values, denominator_values, biases


def _normalize_float16_tensor_opencv(
    img: torch.Tensor,
    mean: object,
    denominator: object,
    layout: TensorLayout | None,
) -> torch.Tensor | None:
    """Normalize measured channel-first half Tensors through one CV_16F call per plane."""
    if img.dtype != torch.float16 or layout not in ("CHW", "NCHW") or not img.is_contiguous():
        return None
    if img.numel() < _TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS:
        return None

    channel_axis = 0 if layout == "CHW" else 1
    channels = img.shape[channel_axis]
    if channels not in (3, 9):
        return None
    controls = _normalize_float16_tensor_controls(mean, denominator, channels)
    if controls is None:
        return None
    mean_values, denominator_values, biases = controls
    if _normalize_is_identity(mean_values, denominator_values):
        return img

    source = img.numpy()
    result = torch.empty_like(img)
    _normalize_float16_opencv_planes(source, result.numpy(), denominator_values, biases, layout)
    return result


def _normalize_float16_opencv_planes(
    source: np.ndarray,
    destination: np.ndarray,
    denominator: np.ndarray,
    biases: np.ndarray,
    layout: Literal["CHW", "NCHW"],
) -> None:
    zero_plane = np.zeros(source.shape[-2:], dtype=np.float16)
    source_batches = source[None, ...] if layout == "CHW" else source
    destination_batches = destination[None, ...] if layout == "CHW" else destination
    for batch in range(source_batches.shape[0]):
        for channel in range(source_batches.shape[1]):
            cv2.addWeighted(
                source_batches[batch, channel],
                float(denominator[channel]),
                zero_plane,
                0.0,
                float(biases[channel]),
                dst=destination_batches[batch, channel],
            )


@preserve_channel_dim
def normalize_opencv(
    img: ImageType,
    mean: float | np.ndarray,
    denominator: float | np.ndarray,
) -> ImageFloat16 | ImageFloat32:
    img_f = _cast_float_numpy(img, np.float32)
    if isinstance(mean, (int, float)):
        mean_img = np.full_like(img_f, mean)
    else:
        mean_img = np.broadcast_to(mean.astype(np.float32, copy=False), img_f.shape)
    if isinstance(denominator, (int, float)):
        denom_img = np.full_like(img_f, denominator)
    else:
        denom_img = np.broadcast_to(denominator.astype(np.float32, copy=False), img_f.shape)
    result = cast("ImageFloat32", cv2.subtract(img_f, mean_img))
    result = cast("ImageFloat32", cv2.multiply(result, denom_img, dtype=cv2.CV_32F))
    return _restore_float_dtype(img, result)


@preserve_channel_dim
def normalize_lut(img: ImageUInt8, mean: float | np.ndarray, denominator: float | np.ndarray) -> ImageFloat32:
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    lut_size = int(max_value) + 1

    x = np.arange(lut_size, dtype=np.float32)
    if isinstance(denominator, (float, int)) and isinstance(mean, (float, int)):
        return _apply_float_lut(img, (x - mean) * denominator)

    x_lut = x.reshape(-1, 1, 1)
    mean_lut = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    denominator_lut = np.asarray(denominator, dtype=np.float32).reshape(1, 1, -1)
    luts = (x_lut - mean_lut) * denominator_lut
    return _apply_float_lut(img, luts)


def _normalize_is_identity(mean: float | np.ndarray, denominator: float | np.ndarray, *, eps: float = 1e-5) -> bool:
    if isinstance(mean, np.ndarray):
        if not bool(np.all(np.abs(mean.astype(np.float64, copy=False)) <= eps)):
            return False
    elif abs(float(mean)) > eps:
        return False
    if isinstance(denominator, np.ndarray):
        d = denominator.astype(np.float64, copy=False)
        if not bool(np.all(np.abs(d - 1.0) <= eps)):
            return False
    elif abs(float(denominator) - 1.0) > eps:
        return False
    return True


def normalize(
    img: ImageType | torch.Tensor,
    mean: ValueType | torch.Tensor,
    denominator: ValueType | torch.Tensor,
    *,
    layout: TensorLayout | None = None,
) -> ImageFloat16 | ImageFloat32 | torch.Tensor:
    """Affine normalize with caller-supplied constants: ``(img - mean) * denominator``.

    Float16 inputs return float16. Uint8 and float32 inputs return float32.

    This is **not** ``normalize_per_image``, which computes stats from the image itself.
    Here, ``mean`` and ``denominator`` are fixed caller values (e.g. ImageNet statistics).

    Typical use — ImageNet normalization of a uint8 image:
    ``mean = [123.675, 116.28, 103.53]`` (mean_01 * 255),
    ``denominator = [1/58.395, 1/57.12, 1/57.375]`` (1 / (std_01 * 255)).

    Routing:
    - **uint8**: LUT (one 256-entry float32 table per channel, applied via ``cv2.LUT``).
      Very fast — 256 floats cover all possible pixel values.
    - **float16 NumPy, multiple channels**: a fused eager CPU Torch route writes directly to the half output when
      the input is compatible; other half inputs use one float32 working buffer.
    - **contiguous CHW/NCHW float16 Tensor, C=3/9, scalar/channel controls, and 16,384+ elements**: OpenCV
      ``CV_16F addWeighted`` processes contiguous channel planes directly. Other Tensor half inputs use the
      zero-copy channel-last float32 route.
    - **float32, identity** (mean ≈ 0 and denominator ≈ 1): cast-only, no arithmetic.
    - **large multi-channel float32 input**: a zero-copy Torch CPU kernel for the fused affine expression.
    - **float32**: NumPy fused ``img * denominator - mean * denominator``.

    Alternative: ``normalize_per_image`` for image-adaptive mean/std normalization.

    Args:
        img: ``(H, W, C)`` image (or batch/volume), uint8, float16, or float32.
        mean: Scalar or length-``C`` array of values to subtract before scaling.
        denominator: Scalar or length-``C`` array — reciprocal of std (multiply, not divide).
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Normalized image with the same shape. Float16 input returns float16; uint8 and float32 input return float32.
    """
    if isinstance(img, torch.Tensor):
        mean_value = tensor_operand_to_numpy(mean, layout)
        denominator_value = tensor_operand_to_numpy(denominator, layout)
        if img.dtype == torch.float16:
            tensor_result = _normalize_float16_tensor_opencv(img, mean_value, denominator_value, layout)
            if tensor_result is not None:
                return tensor_result
            image_view = tensor_to_numpy_channel_last(img, layout)
            result = normalize(image_view, mean_value, denominator_value)
            tensor_result = (
                img if result is image_view else numpy_to_tensor_layout(result, cast("TensorLayout", layout))
            )
        else:
            tensor_result = dispatch_tensor_image(img, layout, normalize, mean_value, denominator_value)
        return tensor_result

    num_channels = get_num_channels(img)
    denominator = convert_value(denominator, num_channels)
    mean = convert_value(mean, num_channels)

    if _is_uint8_image(img):
        return normalize_lut(img, mean, denominator)

    if _normalize_is_identity(mean, denominator):
        return cast("ImageFloat16 | ImageFloat32", img.astype(img.dtype, copy=False))

    if _is_float16_image(img):
        half_result = normalize_float16_torch(
            img,
            mean,
            denominator,
            minimum_elements=_TORCH_FLOAT16_NORMALIZE_MIN_ELEMENTS,
        )
        if half_result is None and img.ndim == 3 and num_channels == 1 and img.flags.c_contiguous:
            half_result = _normalize_float16_single_channel_opencv(img, mean, denominator)
        return half_result if half_result is not None else normalize_numpy(img, mean, denominator)

    float32_result = normalize_float32_torch(cast("ImageFloat32", img), mean, denominator)
    return float32_result if float32_result is not None else normalize_numpy(img, mean, denominator)


def power_numpy(img: ImageType, exponent: float | np.ndarray) -> ImageFloat16 | ImageFloat32:
    if _is_float16_image(img):
        float16_result = _float16_numpy_ufunc(img, exponent, "power")
        if float16_result is not None:
            return float16_result
    result = apply_numpy(img, exponent, "power")
    return _cast_float_numpy(result, np.float16) if _is_float16_image(img) else result


@preserve_channel_dim
def power_opencv(img: ImageType, value: float) -> ImageFloat16 | ImageFloat32:
    if _is_float16_image(img):
        result = cv2.pow(_cast_float_numpy(img, np.float32), value)
        return _cast_float_numpy(result, np.float16)
    if _is_float32_image(img):
        # For float32 images, cv2.pow works directly
        return cast("ImageFloat32", cv2.pow(img, value))
    if int(value) == value:
        # For uint8 images, cv2.pow works directly if value is actual integer, even if it's type is float
        return cast("ImageFloat32", cv2.pow(img, value))
    # For uint8 images, convert to float32 before applying a fractional power.
    return cast("ImageFloat32", cv2.pow(img.astype(np.float32, copy=False), value))


def power_lut(img: ImageUInt8, exponent: float | np.ndarray, inplace: bool = False) -> ImageUInt8:
    return apply_lut(img, exponent, "power", inplace)


def power(
    img: ImageType | torch.Tensor,
    exponent: ValueType | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Raise image pixels to a power (gamma correction / contrast adjustment).

    Routes to the fastest backend per dtype and exponent shape:

    - **uint8 scalar / vector**: LUT (one 256-entry table per channel or shared).
    - **float16 scalar and half-compatible array exponents**: eager CPU Torch from 16,384 elements. Scalar exponents
      use the float32 OpenCV path because OpenCV's direct half ``pow`` is incorrect.
    - **float32 scalar**: ``cv2.pow`` (operates directly on float32).
    - **float32 vector / array**: NumPy ``np.power`` broadcast.

    Args:
        img: ``(H, W, C)`` image, uint8, float16, or float32.
        exponent: Scalar, length-``C`` 1-D array, or full ``img``-shaped array.
        inplace: Mutate ``img`` buffer when the LUT path allows it.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Pixel-wise power with the same shape and dtype as ``img``. Uint8 results saturate to
        [0, 255]; floating results preserve the raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        parameter = tensor_operand_to_numpy(exponent, layout)
        return dispatch_tensor_image(img, layout, power, parameter, inplace=inplace)

    num_channels = get_num_channels(img)
    exponent = convert_value(exponent, num_channels)
    if _is_uint8_image(img):
        return power_lut(img, exponent, inplace)

    if _is_float16_image(img):
        if isinstance(exponent, (float, int)):
            result = power_opencv(img, float(exponent))
        else:
            torch_result = float16_pointwise_torch(
                img,
                "power",
                exponent,
                minimum_elements=_TORCH_FLOAT16_MIN_ELEMENTS,
            )
            result = (
                cast("ImageType", torch_result)
                if torch_result is not None
                else cast("ImageFloat16", power_numpy(img, exponent))
            )
        return result

    if isinstance(exponent, (float, int)):
        return power_opencv(img, exponent)

    return power_numpy(img, exponent)


def add_weighted_numpy(
    img1: ImageType,
    weight1: float,
    img2: ImageType,
    weight2: float,
) -> ImageFloat16 | ImageFloat32:
    if _is_float16_image(img1):
        result = _cast_float_numpy(img1, np.float32) * weight1 + _cast_float_numpy(img2, np.float32) * weight2
        return _cast_float_numpy(result, np.float16)
    return img1.astype(np.float32, copy=False) * weight1 + img2.astype(np.float32, copy=False) * weight2


@preserve_channel_dim
def add_weighted_opencv(img1: ImageType, weight1: float, img2: ImageType, weight2: float) -> ImageType:
    return cast("ImageType", cv2.addWeighted(img1, weight1, img2, weight2, 0))


@preserve_channel_dim
def add_weighted_lut(
    img1: ImageUInt8,
    weight1: float,
    img2: ImageUInt8,
    weight2: float,
    inplace: bool = False,
) -> ImageType:
    """Add weighted using LUT. Only works with uint8 images."""
    dtype = img1.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if weight1 == 1 and weight2 == 0:
        return cast("ImageType", img1)

    if weight1 == 0 and weight2 == 1:
        return cast("ImageType", img2)

    if weight1 == 0 and weight2 == 0:
        return cast("ImageType", np.zeros_like(img1))

    if weight1 == 1 and weight2 == 1:
        return add_array(img1, img2)

    base = np.arange(max_value + 1, dtype=np.float32)
    result1 = cast("ImageType", cv2.LUT(img1, base * weight1))
    result2 = cast("ImageType", cv2.LUT(img2, base * weight2))

    return add_opencv(result1, result2, inplace)


def add_weighted(
    img1: ImageType | torch.Tensor,
    weight1: float,
    img2: ImageType | torch.Tensor,
    weight2: float,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Blend two images: ``img1 * weight1 + img2 * weight2``.

    Both images must have identical shapes and dtypes.

    Routing:
    - **float16 NumPy/Tensor**: NumKong NEON-half ``blend`` for contiguous inputs with weights ``0.5, 0.5``; NumPy
      inputs otherwise use direct OpenCV ``CV_16F`` through the backend channel limit, then a float32 working route.
      Tensor inputs keep their measured channel-last bridge when the NumKong condition is not met.
    - **float32 with more than one channel**: ``cv2.addWeighted`` for HWC or two C-contiguous inputs;
      NumKong SIMD ``blend`` for strided batch/volume inputs.
    - **single-channel float32 and uint8**: NumKong SIMD ``blend``.

    The split is calibrated by ``benchmarks/benchmark_add_weighted.py`` across the issue #130 and canonical grids.

    Alternative low-level paths: ``add_weighted_numpy``, ``add_weighted_opencv``, ``add_weighted_lut``.

    Args:
        img1: First image, shape ``(H, W, C)`` (or batch/volume), uint8, float16, or float32.
        img2: Second image, must match ``img1`` shape exactly.
        weight1: Scalar weight for ``img1``.
        weight2: Scalar weight for ``img2``.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        Blended image with the same shape and dtype as the inputs. Floating results are not clipped to the image
        range; uint8 results retain the backend's saturating arithmetic semantics.

    The caller validates matching shapes and dtypes before entering this low-level router.
    """
    if isinstance(img1, torch.Tensor):
        if not isinstance(img2, torch.Tensor):
            msg = "Both image operands must use the same container."
            raise TypeError(msg)
        if (
            _NUMKONG_NEON_HALF
            and layout is not None
            and img1.dtype == torch.float16
            and img1.is_contiguous()
            and img2.is_contiguous()
            and img1.numel() >= _NUMKONG_FLOAT16_MIN_ELEMENTS
            and weight1 == 0.5
            and weight2 == 0.5
        ):
            # NumKong's blend is elementwise, so flattening the contiguous Tensor preserves its layout.
            result = add_weighted_numkong(img1.numpy(), weight1, img2.numpy(), weight2)
            return torch.from_numpy(result)
        image2 = tensor_operand_to_numpy(img2, layout)
        return dispatch_tensor_image(img1, layout, add_weighted, weight1, image2, weight2)

    if _is_float16_image(img1):
        if (
            _NUMKONG_NEON_HALF
            and img1.flags.c_contiguous
            and img2.flags.c_contiguous
            and img1.size >= _NUMKONG_FLOAT16_MIN_ELEMENTS
            and weight1 == 0.5
            and weight2 == 0.5
        ):
            # This half-representable weight pair is the measured NumKong blend region.
            result = add_weighted_numkong(img1, weight1, img2, weight2)
        elif get_num_channels(img1) <= _CV16F_ADD_WEIGHTED_MAX_CHANNELS:
            result = add_weighted_opencv(img1, weight1, img2, weight2)
        else:
            result = add_weighted_numpy(img1, weight1, img2, weight2)
        return result

    if (
        _is_float32_image(img1)
        and get_num_channels(img1) > 1
        and (img1.ndim == 3 or (img1.flags["C_CONTIGUOUS"] and img2.flags["C_CONTIGUOUS"]))
    ):
        return add_weighted_opencv(img1, weight1, img2, weight2)

    return add_weighted_numkong(img1, weight1, img2, weight2)


def _broadcast_channel_vector(x: _PreparedValue, n_dim: int, c: int) -> _PreparedValue:
    if isinstance(x, np.ndarray) and x.shape == (c,):
        return x.reshape((1,) * (n_dim - 1) + (c,))
    return x


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
    return torch.add(image, control) if operation == "add" else torch.mul(image, control)


def _dispatch_tensor_add(
    image: torch.Tensor,
    value: ValueType | torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., ImageType | torch.Tensor],
    inplace: bool,
) -> ImageType | torch.Tensor:
    numpy_value = tensor_operand_to_numpy(value, layout)
    if not inplace:
        result = _torch_half_pointwise_tensor(image, numpy_value, layout, "add")
        if result is not None:
            return result
    return dispatch_tensor_image(image, layout, numpy_router, numpy_value, inplace=inplace)


def _dispatch_tensor_multiply(
    image: torch.Tensor,
    value: ValueType | torch.Tensor,
    layout: TensorLayout | None,
    numpy_router: Callable[..., ImageType | torch.Tensor],
    inplace: bool,
) -> ImageType | torch.Tensor:
    numpy_value = tensor_operand_to_numpy(value, layout)
    if not inplace:
        result = _torch_half_pointwise_tensor(image, numpy_value, layout, "multiply")
        if result is not None:
            return result
    return dispatch_tensor_image(image, layout, numpy_router, numpy_value, inplace=inplace)


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


def multiply_add_numpy(img: ImageType, factor: ValueType, value: ValueType) -> ImageFloat16 | ImageFloat32:
    if _is_float16_image(img):
        float16_product = _float16_numpy_ufunc(img, factor, "multiply")
        if float16_product is not None:
            float16_result = _float16_numpy_ufunc(float16_product, value, "add", inplace=True)
            if float16_result is not None:
                return float16_result

    img_f = _cast_float_numpy(img, np.float32)
    factor_prepared = _prepare_numpy_value(factor)
    value_prepared = _prepare_numpy_value(value)

    def _scalar_float(x: _PreparedValue) -> float | None:
        if isinstance(x, np.ndarray):
            return None
        return float(x)

    sf, sv = _scalar_float(factor_prepared), _scalar_float(value_prepared)
    if sf is not None and sv is not None and sf == 0.0 and sv == 0.0:
        return np.zeros_like(img_f, dtype=np.float32)

    if sf is not None and sv is not None and sf == 1.0 and sv == 0.0:
        return img_f

    n_dim = img_f.ndim
    c = int(img_f.shape[-1])
    f_b = _broadcast_channel_vector(factor_prepared, n_dim, c)
    v_b = _broadcast_channel_vector(value_prepared, n_dim, c)

    multiply_result = np.zeros_like(img_f) if _is_all_zero_param(f_b) else np.multiply(img_f, cast("Any", f_b))
    multiply_result = multiply_result if _is_all_zero_param(v_b) else np.add(multiply_result, cast("Any", v_b))
    if multiply_result.dtype == np.float32:
        return cast("ImageFloat32", multiply_result)
    with np.errstate(over="ignore"):
        return cast("ImageFloat32", multiply_result.astype(np.float32, copy=False))


@preserve_channel_dim
def multiply_add_opencv(img: ImageType, factor: ValueType, value: ValueType) -> ImageFloat32:
    if isinstance(value, (int, float)) and value == 0 and isinstance(factor, (int, float)) and factor == 0:
        return np.zeros_like(img, dtype=np.float32)

    result = _cast_float_numpy(img, np.float32)
    num_channels = result.shape[-1]
    if factor != 0:
        if isinstance(factor, (int, float)) and num_channels <= MAX_OPENCV_WORKING_CHANNELS:
            result = cast("ImageFloat32", cv2.multiply(result, cast("Any", factor), dtype=cv2.CV_32F))
        else:
            factor_img = (
                np.full_like(result, factor)
                if isinstance(factor, (int, float))
                else np.broadcast_to(
                    factor.astype(np.float32, copy=False),
                    result.shape,
                )
            )
            result = cast("ImageFloat32", cv2.multiply(result, factor_img, dtype=cv2.CV_32F))
    else:
        result = np.zeros_like(result)
    if value != 0:
        if isinstance(value, (int, float)) and num_channels <= MAX_OPENCV_WORKING_CHANNELS:
            result = cast("ImageFloat32", cv2.add(result, cast("Any", value), dtype=cv2.CV_32F))
        else:
            value_img = (
                np.full_like(result, value)
                if isinstance(value, (int, float))
                else np.broadcast_to(
                    value.astype(np.float32, copy=False),
                    result.shape,
                )
            )
            result = cast("ImageFloat32", cv2.add(result, value_img, dtype=cv2.CV_32F))
    return result


def multiply_add_lut(img: ImageUInt8, factor: ValueType, value: ValueType, inplace: bool) -> ImageUInt8:
    """Apply multiply-add operation using LUT. Only works with uint8 images."""
    dtype = img.dtype
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    lut_size = int(max_value) + 1

    domain = np.arange(lut_size, dtype=np.float32)

    if isinstance(factor, (float, int)) and isinstance(value, (float, int)):
        lut = clip(domain * factor + value, dtype, inplace=False)
        return _apply_uint8_lut(img, lut, inplace=inplace)

    if isinstance(factor, np.ndarray) and factor.shape != ():
        factor = factor.reshape(1, 1, -1)

    if isinstance(value, np.ndarray) and value.shape != ():
        value = value.reshape(1, 1, -1)

    lut_values = np.asarray(domain.reshape(-1, 1, 1) * factor + value, dtype=np.float32)
    luts = clip(lut_values, dtype, inplace=False)
    return _apply_uint8_lut(img, luts, inplace=inplace)


def multiply_add(
    img: ImageType | torch.Tensor,
    factor: ValueType | torch.Tensor,
    value: ValueType | torch.Tensor,
    inplace: bool = False,
    *,
    layout: TensorLayout | None = None,
) -> ImageType | torch.Tensor:
    """Fused multiply-add: ``img * factor + value``.

    Equivalent to ``add(multiply(img, factor), value)`` but avoids intermediate allocations
    for the uint8 LUT path.

    Routing:
    - **uint8**: LUT (``img * factor + value`` computed once for all 256 values, then applied
      via ``apply_uint8_lut``). Scalar and per-channel vectors both supported.
    - **NumPy float16**: NumKong NEON-half ``scale`` for contiguous arrays of 20,480+ elements when both scalar
      controls are exactly representable in float16; remaining controls and layouts use float32 arithmetic.
    - **float16**: a benchmark-gated eager CPU Torch path for large arrays; remaining sizes use float32 arithmetic
      and one final cast.
    - **float16 Tensor with half-representable scalar/channel controls**: direct Torch multiply and add,
      preserving the existing two-rounding result.
    - **large float32 scalar factor and value**: Torch CPU arithmetic through a zero-copy NumPy wrapper.
    - **other float32 scalar factor and value**: NumKong ``scale``.
    - **float32 vector / array operands**: NumPy broadcast.

    Args:
        img: ``(H, W, C)`` image, uint8, float16, or float32.
        factor: Scalar or length-``C`` array to multiply by.
        value: Scalar or length-``C`` array to add.
        inplace: Reuse ``img`` buffer when the LUT path is used and ``factor``/``value`` are scalar.
        layout: Required for channel-first Tensor inputs; selects CHW, NCHW, CDHW, or NCDHW.

    Returns:
        ``img * factor + value`` with the same shape and dtype as ``img``. Uint8 results saturate
        to [0, 255]; float32 results preserve the raw numeric range.
    """
    if isinstance(img, torch.Tensor):
        if layout is not None and img.dtype == torch.float16 and not inplace:
            half_factor = _torch_half_control(factor, img, layout)
            half_value = _torch_half_control(value, img, layout)
            if half_factor is not None and half_value is not None:
                return torch.add(torch.mul(img, half_factor), half_value)
        factor_value = tensor_operand_to_numpy(factor, layout)
        bias_value = tensor_operand_to_numpy(value, layout)
        return dispatch_tensor_image(img, layout, multiply_add, factor_value, bias_value, inplace=inplace)

    num_channels = get_num_channels(img)
    factor = convert_value(factor, num_channels)
    value = convert_value(value, num_channels)

    if _is_uint8_image(img):
        return multiply_add_lut(img, factor, value, inplace)

    if _is_float16_image(img):
        numkong_factor = _half_exact_scalar(factor)
        numkong_value = _half_exact_scalar(value)
        if (
            _NUMKONG_NEON_HALF
            and img.flags.c_contiguous
            and img.size >= _NUMKONG_FLOAT16_MIN_ELEMENTS
            and numkong_factor is not None
            and numkong_value is not None
        ):
            torch_result: ImageType | None = multiply_add_numkong(img, numkong_factor, numkong_value)
        else:
            torch_result = multiply_add_float16_torch(
                img,
                factor,
                value,
                minimum_elements=_TORCH_FLOAT16_COMPOUND_MIN_ELEMENTS,
            )
        if torch_result is None:
            float16_result = multiply_add_numpy(img, factor, value)
            torch_result = _cast_float_numpy(float16_result, np.float16)
        return torch_result

    float32_img = cast("ImageFloat32", img)
    if isinstance(factor, (float, int)) and isinstance(value, (float, int)) and factor == 1 and value == 0:
        return float32_img
    if _use_numkong_scalar_multiply_add(float32_img, factor, value):
        torch_result = multiply_add_float32_torch(float32_img, float(factor), float(value))
        result = (
            multiply_add_numkong(float32_img, float(factor), float(value)) if torch_result is None else torch_result
        )
    else:
        result = cast("ImageFloat32", multiply_add_numpy(float32_img, factor, value))
    return result
