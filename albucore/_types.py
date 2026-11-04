"""Supported array storage types and explicit CPU Tensor layouts."""

from typing import Literal, TypeAlias, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

NormalizationType = Literal["image", "image_per_channel", "min_max", "min_max_per_channel"]

ImageInput: TypeAlias = NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32] | torch.Tensor
ImageInputT = TypeVar("ImageInputT", bound=ImageInput)
ImageArrayT = TypeVar("ImageArrayT", bound=NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32])
ArrayT = TypeVar("ArrayT", bound=np.ndarray)
TensorLayout: TypeAlias = Literal["CHW", "NCHW", "CDHW", "NCDHW"]
TensorImageLayout: TypeAlias = Literal["CHW", "NCHW"]
SupportedDType: TypeAlias = (
    np.dtype[np.uint8]
    | np.dtype[np.float16]
    | np.dtype[np.float32]
    | type[np.uint8]
    | type[np.float16]
    | type[np.float32]
)
ValueType = np.ndarray | float | int


__all__ = [
    "ImageInput",
    "ImageInputT",
    "NormalizationType",
    "SupportedDType",
    "TensorImageLayout",
    "TensorLayout",
    "ValueType",
]
