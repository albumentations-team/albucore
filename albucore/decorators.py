import sys
from functools import wraps
from typing import Any, Callable, Literal, TypeVar, cast

import numpy as np

from albucore.utils import MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS, P

if sys.version_info >= (3, 10):
    from typing import Concatenate
else:
    from typing_extensions import Concatenate

F = TypeVar("F", bound=Callable[..., Any])


def contiguous(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Ensure that input img is contiguous and the output array is also contiguous."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        # Ensure the input array is contiguous
        img = np.require(img, requirements=["C_CONTIGUOUS"])
        # Call the original function with the contiguous input
        result = func(img, *args, **kwargs)
        # Ensure the output array is contiguous
        return np.require(result, requirements=["C_CONTIGUOUS"])

    return wrapped_function


def preserve_channel_dim(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Preserve dummy channel dim."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS and shape[-1] == 1 and result.ndim == MONO_CHANNEL_DIMENSIONS:
            return np.expand_dims(result, axis=-1)

        if len(shape) == MONO_CHANNEL_DIMENSIONS and result.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            return result[:, :, 0]
        return result

    return wrapped_function


BatchTransformType = Literal["spatial", "channel", "full"]

ShapeType = Literal[
    "DHW",  # (D,H,W)
    "DHWC",  # (D,H,W,C)
    "NHW",  # (N,H,W)
    "NHWC",  # (N,H,W,C)
    "NDHW",  # (N,D,H,W)
    "NDHWC",  # (N,D,H,W,C)
]


def get_shape_type(
    shape: tuple[int, ...],
    has_batch_dim: bool,
    has_depth_dim: bool,
) -> ShapeType:
    """Determine the shape type based on dimensions and flags."""
    ndim = len(shape)

    if has_batch_dim and has_depth_dim:
        return "NDHWC" if ndim == 5 else "NDHW"
    if has_batch_dim:
        return "NHWC" if ndim == 4 else "NHW"
    if has_depth_dim:
        return "DHWC" if ndim == 4 else "DHW"
    raise ValueError("Either batch or depth dimension must be True")


def reshape_3d(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (D,H,W) or (D,H,W,C) for spatial transforms."""
    if data.ndim == 3:  # (D,H,W) => (H,W,D)
        _, height, width = data.shape
        reshaped = np.require(np.moveaxis(data, 0, -1), requirements=["C_CONTIGUOUS"])  # (H,W,D)
        return reshaped, data.shape
    if data.ndim == 4:  # (D,H,W,C) => (H,W,D*C)
        _, height, width, channels = data.shape
        reshaped = np.moveaxis(data, 0, -2)  # (H,W,D,C)
        final = np.require(reshaped.reshape(height, width, -1), requirements=["C_CONTIGUOUS"])  # (H,W,D*C)
        return final, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (H,W,N*D*C)
        _, _, height, width, channels = data.shape
        flat = data.reshape(-1, height, width, channels)  # (N*D,H,W,C)
        reshaped = np.moveaxis(flat, 0, -2)  # (H,W,N*D,C)
        final = np.require(reshaped.reshape(height, width, -1), requirements=["C_CONTIGUOUS"])  # (H,W,N*D*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,H,W) or (N,H,W,C) for spatial transforms."""
    if data.ndim == 3:  # (N,H,W) => (H,W,N)
        _, height, width = data.shape
        reshaped = np.require(np.moveaxis(data, 0, -1), requirements=["C_CONTIGUOUS"])  # (H,W,N)
        return reshaped, data.shape
    if data.ndim == 4:  # (N,H,W,C) => (H,W,N*C)
        _, height, width, channels = data.shape
        reshaped = np.moveaxis(data, 0, -2)  # (H,W,N,C)
        final = np.require(reshaped.reshape(height, width, -1), requirements=["C_CONTIGUOUS"])  # (H,W,N*C)
        return final, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (H,W,N*D*C)
        _, _, height, width, channels = data.shape
        flat = data.reshape(-1, height, width, channels)  # (N*D,H,W,C)
        reshaped = np.moveaxis(flat, 0, -2)  # (H,W,N*D,C)
        final = np.require(reshaped.reshape(height, width, -1), requirements=["C_CONTIGUOUS"])  # (H,W,N*D*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_3d(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,D,H,W) or (N,D,H,W,C) for spatial transforms."""
    if data.ndim == 4:  # (N,D,H,W) => (H,W,N*D)
        _, _, height, width = data.shape
        flat = data.reshape(-1, height, width)  # (N*D,H,W)
        reshaped = np.require(np.moveaxis(flat, 0, -1), requirements=["C_CONTIGUOUS"])  # (H,W,N*D)
        return reshaped, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (H,W,N*D*C)
        _, _, height, width, channels = data.shape
        flat = data.reshape(-1, height, width, channels)  # (N*D,H,W,C)
        reshaped = np.require(np.moveaxis(flat, 0, -2), requirements=["C_CONTIGUOUS"])  # (H,W,N*D,C)
        final = np.require(reshaped.reshape(height, width, -1), requirements=["C_CONTIGUOUS"])  # (H,W,N*D*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_3d_keep_depth(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,D,H,W) or (N,D,H,W,C) preserving depth dimension."""
    if data.ndim == 4:  # (N,D,H,W) => (D,H,W,N)
        _, depth, height, width = data.shape
        reshaped = np.moveaxis(data, 0, -1)  # (D,H,W,N)
        return np.require(reshaped, requirements=["C_CONTIGUOUS"]), data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (D,H,W,N*C)
        _, depth, height, width, _ = data.shape
        reshaped = np.moveaxis(data, 0, -2)  # (D,H,W,N,C)
        final = np.require(reshaped.reshape(depth, height, width, -1), requirements=["C_CONTIGUOUS"])  # (D,H,W,N*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_3d_keep_depth(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (D,H,W) or (D,H,W,C) preserving depth dimension."""
    if data.ndim in {3, 4}:  # (D,H,W) or (D,H,W,C)
        return data, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def restore_3d(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,D) or (H,W,D*C) back to (D,H,W) or (D,H,W,C)."""
    height, width = data.shape[:2]

    if len(original_shape) == 3:  # (H',W',D) => (D,H',W')
        depth = original_shape[0]
        # Reshape using the transformed H',W' dimensions
        reshaped = data.reshape(height, width, depth)
        return np.ascontiguousarray(np.moveaxis(reshaped, -1, 0))
    # (H',W',D*C) => (D,H',W',C)
    depth, _, _, channels = original_shape
    # Use transformed H',W' dimensions instead of original ones
    reshaped = data.reshape(height, width, depth, channels)  # (H',W',D,C)
    return np.ascontiguousarray(np.moveaxis(reshaped, -2, 0))  # (D,H',W',C)


def restore_batch(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,N) or (H,W,N*C) back to (N,H,W) or (N,H,W,C)."""
    height, width = data.shape[:2]

    if len(original_shape) == 3:  # (H',W',N) => (N,H',W')
        num_images = original_shape[0]
        reshaped = data.reshape(height, width, num_images)
        return np.require(np.moveaxis(reshaped, -1, 0), requirements=["C_CONTIGUOUS"])
    # (H',W',N*C) => (N,H',W',C)
    num_images, _, _, channels = original_shape
    reshaped = data.reshape(height, width, num_images, channels)  # (H',W',N,C)
    return np.require(np.moveaxis(reshaped, -2, 0), requirements=["C_CONTIGUOUS"])  # (N,H',W',C)


def restore_batch_3d(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,N*D) or (H,W,N*D*C) back to (N,D,H,W) or (N,D,H,W,C)."""
    height, width = data.shape[:2]

    if len(original_shape) == 4:  # (H',W',N*D) => (N,D,H',W')
        num_images, depth = original_shape[:2]
        reshaped = np.moveaxis(data, -1, 0)  # (N*D,H',W')
        return np.require(reshaped.reshape(num_images, depth, height, width), requirements=["C_CONTIGUOUS"])
    # (H',W',N*D*C) => (N,D,H',W',C)
    num_images, depth, _, _, channels = original_shape
    reshaped = data.reshape(height, width, -1, channels)  # (H',W',N*D,C)
    moved = np.moveaxis(reshaped, -2, 0)  # (N*D,H',W',C)
    return np.require(moved.reshape(num_images, depth, height, width, channels), requirements=["C_CONTIGUOUS"])


def restore_batch_3d_keep_depth(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore data that kept its depth dimension.

    For data that was reshaped with keep_depth_dim=True:
    - (D',H',W',N) => (N,D',H',W')
    - (D',H',W',N*C) => (N,D',H',W',C)

    Note: D',H',W' can be different from original D,H,W after transforms like RandomCrop3D
    """
    # Use transformed D,H,W dimensions
    new_depth, new_height, new_width = data.shape[:3]

    if len(original_shape) == 4:  # (D',H',W',N) => (N,D',H',W')
        return np.moveaxis(data, -1, 0)
    # (D',H',W',N*C) => (N,D',H',W',C)
    num_images = original_shape[0]
    channels = original_shape[-1]
    # Use new_depth, new_height, new_width instead of original dimensions
    reshaped = data.reshape(new_depth, new_height, new_width, num_images, channels)
    return np.moveaxis(reshaped, -2, 0)


def restore_3d_keep_depth(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore data that kept its depth dimension.

    For data that was reshaped with keep_depth_dim=True:
    - (D,H,W) => (D,H,W)
    - (D,H,W,C) => (D,H,W,C)
    """
    # No reshape needed since we kept the original shape
    return data


def reshape_3d_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (D,H,W) or (D,H,W,C) for channel transforms."""
    if data.ndim == 3:  # (D,H,W) => (D*H,W)
        depth, height, width = data.shape
        reshaped = np.require(data.reshape(depth * height, width), requirements=["C_CONTIGUOUS"])
        return reshaped, data.shape
    if data.ndim == 4:  # (D,H,W,C) => (D*H,W,C)
        depth, height, width, channels = data.shape
        reshaped = np.require(data.reshape(depth * height, width, channels), requirements=["C_CONTIGUOUS"])
        return reshaped, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,H,W) or (N,H,W,C) for channel transforms."""
    if data.ndim == 3:  # (N,H,W) => (N*H,W)
        num_images, height, width = data.shape
        reshaped = np.require(data.reshape(num_images * height, width), requirements=["C_CONTIGUOUS"])
        return reshaped, data.shape
    if data.ndim == 4:  # (N,H,W,C) => (N*H,W,C)
        num_images, height, width, channels = data.shape
        reshaped = np.require(data.reshape(num_images * height, width, channels), requirements=["C_CONTIGUOUS"])
        return reshaped, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_3d_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,D,H,W) or (N,D,H,W,C) for channel transforms."""
    if data.ndim == 4:  # (N,D,H,W) => (N*D*H,W)
        _, _, _, width = data.shape
        # Flatten N,D,H together, keep W separate
        reshaped = np.require(data.reshape(-1, width), requirements=["C_CONTIGUOUS"])
        return reshaped, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (N*D*H,W,C)
        _, _, _, width, channels = data.shape
        # Flatten N,D,H together, keep W and C separate
        reshaped = np.require(data.reshape(-1, width, channels), requirements=["C_CONTIGUOUS"])
        return reshaped, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def restore_3d_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (D*H,W) or (D*H,W,C) back to (D,H,W) or (D,H,W,C).

    Args:
        data: Array of shape (D*H,W') or (D*H,W',C)
        original_shape: Original shape (D,H,W) or (D,H,W,C)
    """
    if len(original_shape) == 3:  # (D*H,W') => (D,H,W')
        depth, height, _ = original_shape
        return data.reshape(depth, height, data.shape[1])
    # (D*H,W',C) => (D,H,W',C)
    depth, height, _, channels = original_shape
    return data.reshape(depth, height, data.shape[1], channels)


def restore_batch_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (N*H,W) or (N*H,W,C) back to (N,H,W) or (N,H,W,C).

    Args:
        data: Array of shape (N*H,W') or (N*H,W',C)
        original_shape: Original shape (N,H,W) or (N,H,W,C)
    """
    if len(original_shape) == 3:  # (N*H,W') => (N,H,W')
        num_images, height, _ = original_shape
        return data.reshape(num_images, height, data.shape[1])
    # (N*H,W',C) => (N,H,W',C)
    num_images, height, _, channels = original_shape
    return data.reshape(num_images, height, data.shape[1], channels)


def restore_batch_3d_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (N*D*H,W') or (N*D*H,W',C) back to (N,D,H,W',C).

    Note: W' can be different from original W after transforms.
    """
    if len(original_shape) == 4:  # (N*D*H,W') => (N,D,H,W')
        num_images, depth, height, _ = original_shape  # Don't use original width
        new_width = data.shape[1]  # Use transformed width
        return data.reshape(num_images, depth, height, new_width)

    # (N*D*H,W',C) => (N,D,H,W',C)
    num_images, depth, height, _, channels = original_shape  # Don't use original width
    new_width = data.shape[1]  # Use transformed width
    return data.reshape(num_images, depth, height, new_width, channels)


# Dictionary mapping shape types to channel reshape functions
CHANNEL_RESHAPE_FUNCS = {
    "DHW": reshape_3d_channel,
    "DHWC": reshape_3d_channel,
    "NHW": reshape_batch_channel,
    "NHWC": reshape_batch_channel,
    "NDHW": reshape_batch_3d_channel,
    "NDHWC": reshape_batch_3d_channel,
}

# Dictionary mapping shape types to channel restore functions
CHANNEL_RESTORE_FUNCS = {
    "DHW": restore_3d_channel,
    "DHWC": restore_3d_channel,
    "NHW": restore_batch_channel,
    "NHWC": restore_batch_channel,
    "NDHW": restore_batch_3d_channel,
    "NDHWC": restore_batch_3d_channel,
}


def reshape_for_channel(
    data: np.ndarray,
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose appropriate reshape function based on data dimensions."""
    if data.size == 0:
        raise ValueError("Empty arrays are not supported")

    shape_type = get_shape_type(data.shape, has_batch_dim, has_depth_dim)
    reshape_func = CHANNEL_RESHAPE_FUNCS[shape_type]
    return reshape_func(data)


def restore_from_channel(
    data: np.ndarray,
    original_shape: tuple[int, ...],
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> np.ndarray:
    """Choose appropriate restore function based on data dimensions."""
    shape_type = get_shape_type(original_shape, has_batch_dim, has_depth_dim)
    restore_func = CHANNEL_RESTORE_FUNCS[shape_type]
    result = restore_func(data, original_shape)
    return np.require(result, requirements=["C_CONTIGUOUS"])


# Dictionary mapping shape types to reshape functions
SPATIAL_RESHAPE_FUNCS = {
    "DHW": reshape_3d,
    "DHWC": reshape_3d,
    "NHW": reshape_batch,
    "NHWC": reshape_batch,
    "NDHW": reshape_batch_3d,
    "NDHWC": reshape_batch_3d,
}

# Dictionary mapping shape types to restore functions
SPATIAL_RESTORE_FUNCS = {
    "DHW": restore_3d,
    "DHWC": restore_3d,
    "NHW": restore_batch,
    "NHWC": restore_batch,
    "NDHW": restore_batch_3d,
    "NDHWC": restore_batch_3d,
}


def reshape_for_spatial(
    data: np.ndarray,
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose appropriate reshape function based on data dimensions."""
    if data.size == 0:
        raise ValueError("Empty arrays are not supported")

    shape_type = get_shape_type(data.shape, has_batch_dim, has_depth_dim)

    if keep_depth_dim:
        if shape_type in {"NDHW", "NDHWC"}:
            return reshape_batch_3d_keep_depth(data)
        if shape_type in {"DHW", "DHWC"}:
            return reshape_3d_keep_depth(data)

    reshape_func = SPATIAL_RESHAPE_FUNCS[shape_type]
    return reshape_func(data)


def restore_from_spatial(
    data: np.ndarray,
    original_shape: tuple[int, ...],
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> np.ndarray:
    """Choose appropriate restore function based on data dimensions."""
    shape_type = get_shape_type(original_shape, has_batch_dim, has_depth_dim)

    if keep_depth_dim:
        if shape_type in {"NDHW", "NDHWC"}:
            result = restore_batch_3d_keep_depth(data, original_shape)
        elif shape_type in {"DHW", "DHWC"}:
            result = restore_3d_keep_depth(data, original_shape)
    else:
        restore_func = SPATIAL_RESTORE_FUNCS[shape_type]
        result = restore_func(data, original_shape)

    return np.require(result, requirements=["C_CONTIGUOUS"])


def batch_transform(
    transform_type: BatchTransformType,
    has_batch_dim: bool = True,
    has_depth_dim: bool = False,
    keep_depth_dim: bool = False,
) -> Callable[[F], F]:
    """Decorator to handle batch transformations."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, data: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
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

            reshaped, original_shape = reshape_func(
                data,
                has_batch_dim,
                has_depth_dim,
                keep_depth_dim,
            )
            transformed = func(self, reshaped, *args, **params)
            return restore_func(
                transformed,
                original_shape,
                has_batch_dim,
                has_depth_dim,
                keep_depth_dim,
            )

        return cast(F, wrapper)

    return decorator
