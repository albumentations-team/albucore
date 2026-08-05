from collections.abc import Callable
from functools import wraps
from typing import Any, Concatenate, Literal

import numpy as np

from albucore.utils import P

Array = np.ndarray[Any, Any]


def contiguous(
    func: Callable[Concatenate[Array, P], Array],
) -> Callable[Concatenate[Array, P], Array]:
    """Ensure that input img is contiguous and the output array is also contiguous.

    Note: This decorator enforces C-contiguous memory layout. Fortran-contiguous
    arrays will be converted to C-contiguous, which involves copying the data.
    This may impact performance for large arrays but is required for compatibility
    with certain operations (e.g., stringzilla).
    """

    @wraps(func)
    def wrapped_function(img: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        # Ensure the input array is contiguous only if needed
        if not img.flags["C_CONTIGUOUS"]:
            img = np.require(img, requirements=["C_CONTIGUOUS"])
        # Call the original function with the contiguous input
        result = func(img, *args, **kwargs)
        # Ensure the output array is contiguous only if needed
        if not result.flags["C_CONTIGUOUS"]:
            return np.require(result, requirements=["C_CONTIGUOUS"])
        return result

    return wrapped_function


def preserve_channel_dim(
    func: Callable[Concatenate[Array, P], Array],
) -> Callable[Concatenate[Array, P], Array]:
    """Preserve single channel dimension when OpenCV drops it."""

    @wraps(func)
    def wrapped_function(img: Array, *args: P.args, **kwargs: P.kwargs) -> Array:
        shape = img.shape
        result = func(img, *args, **kwargs)
        # If input had 3 dims with last dim = 1, and OpenCV dropped it to 2 dims
        if len(shape) == 3 and shape[-1] == 1 and result.ndim == 2:
            return np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


BatchTransformType = Literal["spatial", "channel", "full"]

ShapeType = Literal[
    "HWC",  # (H,W,C)
    "XHWC",  # (X,H,W,C) where X is either batch N or depth D
]


def get_shape_type(shape: tuple[int, ...]) -> ShapeType:
    """Determine the shape type based on number of dimensions."""
    ndim = len(shape)

    if ndim == 3:
        return "HWC"
    if ndim == 4:
        return "XHWC"  # Could be NHWC or DHWC, but they're treated the same
    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def reshape_for_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose appropriate reshape function based on data dimensions."""
    shape_type = get_shape_type(data.shape)
    reshape_func = CHANNEL_RESHAPE_FUNCS[shape_type]
    return reshape_func(data)


def restore_from_channel(
    data: np.ndarray,
    original_shape: tuple[int, ...],
) -> np.ndarray:
    """Choose appropriate restore function based on data dimensions."""
    shape_type = get_shape_type(original_shape)
    restore_func = CHANNEL_RESTORE_FUNCS[shape_type]
    return restore_func(data, original_shape)


def reshape_for_spatial(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose appropriate reshape function based on data dimensions."""
    shape_type = get_shape_type(data.shape)

    reshape_func = SPATIAL_RESHAPE_FUNCS[shape_type]
    return reshape_func(data)


def restore_from_spatial(
    data: np.ndarray,
    original_shape: tuple[int, ...],
) -> np.ndarray:
    """Choose appropriate restore function based on data dimensions."""
    shape_type = get_shape_type(original_shape)

    restore_func = SPATIAL_RESTORE_FUNCS[shape_type]
    return restore_func(data, original_shape)


def batch_transform(
    transform_type: BatchTransformType,
) -> Callable[[Callable[..., Array]], Callable[..., Array]]:
    """Decorator to handle batch transformations."""

    def decorator(func: Callable[..., Array]) -> Callable[..., Array]:
        @wraps(func)
        def wrapper(self: Any, data: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
            if not data.flags["C_CONTIGUOUS"]:
                data = np.require(data, requirements=["C_CONTIGUOUS"])

            if transform_type == "full":
                return func(self, data, *args, **params)

            # Define the function mappings with proper types
            reshape_funcs: dict[str, Callable[..., tuple[np.ndarray, tuple[int, ...]]]] = {
                "spatial": reshape_for_spatial,
                "channel": reshape_for_channel,
            }

            restore_funcs: dict[str, Callable[..., np.ndarray]] = {
                "spatial": restore_from_spatial,
                "channel": restore_from_channel,
            }

            reshape_func = reshape_funcs[transform_type]
            restore_func = restore_funcs[transform_type]

            reshaped, original_shape = reshape_func(data)
            transformed = func(self, reshaped, *args, **params)
            return restore_func(transformed, original_shape)

        return wrapper

    return decorator


def reshape_hwc(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Identity reshape for (H,W,C) - no transformation needed."""
    return data, data.shape


def reshape_xhwc(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (X,H,W,C) where X is batch N or depth D for spatial transforms."""
    # (X,H,W,C) => (H,W,X*C)
    _, height, width, _ = data.shape
    reshaped = np.moveaxis(data, 0, -2)  # (H,W,X,C)
    final = reshaped.reshape(height, width, -1)  # (H,W,X*C)
    return final, data.shape


def restore_hwc(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Identity restore for (H,W,C) - no transformation needed."""
    return data


def restore_xhwc(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,X*C) back to (X,H,W,C)."""
    height, width = data.shape[:2]

    # (H',W',X*C) => (X,H',W',C)
    x_dim, _, _, channels = original_shape
    reshaped = data.reshape(height, width, x_dim, channels)  # (H',W',X,C)
    return np.moveaxis(reshaped, -2, 0)  # (X,H',W',C)


def reshape_hwc_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Identity reshape for (H,W,C) channel transforms - already in correct format."""
    return data, data.shape


def reshape_xhwc_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (X,H,W,C) for channel transforms."""
    # (X,H,W,C) => (X*H,W,C)
    x_dim, height, width, channels = data.shape
    reshaped = data.reshape(x_dim * height, width, channels)
    return reshaped, data.shape


def restore_hwc_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Identity restore for (H,W,C) channel transforms."""
    return data


def restore_xhwc_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (X*H,W',C) back to (X,H,W',C)."""
    # (X*H,W',C) => (X,H,W',C)
    x_dim, height, _, channels = original_shape
    return data.reshape(x_dim, height, data.shape[1], channels)


# Dictionary mapping shape types to spatial reshape functions
SPATIAL_RESHAPE_FUNCS = {
    "HWC": reshape_hwc,
    "XHWC": reshape_xhwc,
}

# Dictionary mapping shape types to spatial restore functions
SPATIAL_RESTORE_FUNCS = {
    "HWC": restore_hwc,
    "XHWC": restore_xhwc,
}

# Dictionary mapping shape types to channel reshape functions
CHANNEL_RESHAPE_FUNCS = {
    "HWC": reshape_hwc_channel,
    "XHWC": reshape_xhwc_channel,
}

# Dictionary mapping shape types to channel restore functions
CHANNEL_RESTORE_FUNCS = {
    "HWC": restore_hwc_channel,
    "XHWC": restore_xhwc_channel,
}

__all__ = [
    "CHANNEL_RESHAPE_FUNCS",
    "CHANNEL_RESTORE_FUNCS",
    "SPATIAL_RESHAPE_FUNCS",
    "SPATIAL_RESTORE_FUNCS",
    "BatchTransformType",
    "ShapeType",
    "batch_transform",
    "contiguous",
    "get_shape_type",
    "preserve_channel_dim",
    "reshape_for_channel",
    "reshape_for_spatial",
    "reshape_hwc",
    "reshape_hwc_channel",
    "reshape_xhwc",
    "reshape_xhwc_channel",
    "restore_from_channel",
    "restore_from_spatial",
    "restore_hwc",
    "restore_hwc_channel",
    "restore_xhwc",
    "restore_xhwc_channel",
]
