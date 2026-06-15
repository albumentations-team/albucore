"""Shared constants for verification tests and router contracts."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

from benchmarks.shape_grids import (
    NON_SQUARE_IMAGE_HW,
    NON_SQUARE_RESIZE_TARGET_WH,
    PROPERTY_CHANNELS,
    PROPERTY_NON_SQUARE_HW,
)

DTypeName: TypeAlias = Literal["uint8", "float32", "float64", "other numeric fallback"]
LayoutName: TypeAlias = Literal["HWC", "XHWC", "NDHWC", "2D", "points"]
ChannelSpec: TypeAlias = Literal["1", "3", "4", ">4"]
ValueKind: TypeAlias = Literal[
    "alpha",
    "beta",
    "bias",
    "border_mode",
    "border_sizes",
    "border_value",
    "denominator",
    "dsize",
    "eps",
    "explicit_axis",
    "factor",
    "gamma",
    "global_axis",
    "image_array",
    "interpolation",
    "kernel_size",
    "map_x",
    "map_y",
    "matrix",
    "matrix_a",
    "matrix_b",
    "max_value",
    "mean",
    "normalization_mode",
    "per_channel_axis",
    "per_channel_exponent",
    "per_channel_lut",
    "per_channel_vector",
    "points_a",
    "points_b",
    "scalar",
    "scalar_exponent",
    "second_image",
    "shared_lut",
    "target_dtype",
]

DTYPE_UINT8: Final[DTypeName] = "uint8"
DTYPE_FLOAT32: Final[DTypeName] = "float32"
DTYPE_FLOAT64: Final[DTypeName] = "float64"
DTYPE_OTHER_NUMERIC_FALLBACK: Final[DTypeName] = "other numeric fallback"

LAYOUT_HWC: Final[LayoutName] = "HWC"
LAYOUT_XHWC: Final[LayoutName] = "XHWC"
LAYOUT_NDHWC: Final[LayoutName] = "NDHWC"
LAYOUT_2D: Final[LayoutName] = "2D"
LAYOUT_POINTS: Final[LayoutName] = "points"

CHANNEL_1: Final[ChannelSpec] = "1"
CHANNEL_3: Final[ChannelSpec] = "3"
CHANNEL_4: Final[ChannelSpec] = "4"
CHANNEL_GT4: Final[ChannelSpec] = ">4"

VALUE_ALPHA: Final[ValueKind] = "alpha"
VALUE_BETA: Final[ValueKind] = "beta"
VALUE_BIAS: Final[ValueKind] = "bias"
VALUE_BORDER_MODE: Final[ValueKind] = "border_mode"
VALUE_BORDER_SIZES: Final[ValueKind] = "border_sizes"
VALUE_BORDER_VALUE: Final[ValueKind] = "border_value"
VALUE_DENOMINATOR: Final[ValueKind] = "denominator"
VALUE_DSIZE: Final[ValueKind] = "dsize"
VALUE_EPS: Final[ValueKind] = "eps"
VALUE_EXPLICIT_AXIS: Final[ValueKind] = "explicit_axis"
VALUE_FACTOR: Final[ValueKind] = "factor"
VALUE_GAMMA: Final[ValueKind] = "gamma"
VALUE_GLOBAL_AXIS: Final[ValueKind] = "global_axis"
VALUE_IMAGE_ARRAY: Final[ValueKind] = "image_array"
VALUE_INTERPOLATION: Final[ValueKind] = "interpolation"
VALUE_KERNEL_SIZE: Final[ValueKind] = "kernel_size"
VALUE_MAP_X: Final[ValueKind] = "map_x"
VALUE_MAP_Y: Final[ValueKind] = "map_y"
VALUE_MATRIX: Final[ValueKind] = "matrix"
VALUE_MATRIX_A: Final[ValueKind] = "matrix_a"
VALUE_MATRIX_B: Final[ValueKind] = "matrix_b"
VALUE_MAX_VALUE: Final[ValueKind] = "max_value"
VALUE_MEAN: Final[ValueKind] = "mean"
VALUE_NORMALIZATION_MODE: Final[ValueKind] = "normalization_mode"
VALUE_PER_CHANNEL_AXIS: Final[ValueKind] = "per_channel_axis"
VALUE_PER_CHANNEL_EXPONENT: Final[ValueKind] = "per_channel_exponent"
VALUE_PER_CHANNEL_LUT: Final[ValueKind] = "per_channel_lut"
VALUE_PER_CHANNEL_VECTOR: Final[ValueKind] = "per_channel_vector"
VALUE_POINTS_A: Final[ValueKind] = "points_a"
VALUE_POINTS_B: Final[ValueKind] = "points_b"
VALUE_SCALAR: Final[ValueKind] = "scalar"
VALUE_SCALAR_EXPONENT: Final[ValueKind] = "scalar_exponent"
VALUE_SECOND_IMAGE: Final[ValueKind] = "second_image"
VALUE_SHARED_LUT: Final[ValueKind] = "shared_lut"
VALUE_TARGET_DTYPE: Final[ValueKind] = "target_dtype"

ALL_DTYPE_NAMES: Final[tuple[DTypeName, ...]] = (
    DTYPE_UINT8,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_OTHER_NUMERIC_FALLBACK,
)
ALL_LAYOUT_NAMES: Final[tuple[LayoutName, ...]] = (
    LAYOUT_HWC,
    LAYOUT_XHWC,
    LAYOUT_NDHWC,
    LAYOUT_2D,
    LAYOUT_POINTS,
)
ALL_CHANNEL_SPECS: Final[tuple[ChannelSpec, ...]] = (CHANNEL_1, CHANNEL_3, CHANNEL_4, CHANNEL_GT4)
ALL_VALUE_KINDS: Final[tuple[ValueKind, ...]] = (
    VALUE_ALPHA,
    VALUE_BETA,
    VALUE_BIAS,
    VALUE_BORDER_MODE,
    VALUE_BORDER_SIZES,
    VALUE_BORDER_VALUE,
    VALUE_DENOMINATOR,
    VALUE_DSIZE,
    VALUE_EPS,
    VALUE_EXPLICIT_AXIS,
    VALUE_FACTOR,
    VALUE_GAMMA,
    VALUE_GLOBAL_AXIS,
    VALUE_IMAGE_ARRAY,
    VALUE_INTERPOLATION,
    VALUE_KERNEL_SIZE,
    VALUE_MAP_X,
    VALUE_MAP_Y,
    VALUE_MATRIX,
    VALUE_MATRIX_A,
    VALUE_MATRIX_B,
    VALUE_MAX_VALUE,
    VALUE_MEAN,
    VALUE_NORMALIZATION_MODE,
    VALUE_PER_CHANNEL_AXIS,
    VALUE_PER_CHANNEL_EXPONENT,
    VALUE_PER_CHANNEL_LUT,
    VALUE_PER_CHANNEL_VECTOR,
    VALUE_POINTS_A,
    VALUE_POINTS_B,
    VALUE_SCALAR,
    VALUE_SCALAR_EXPONENT,
    VALUE_SECOND_IMAGE,
    VALUE_SHARED_LUT,
    VALUE_TARGET_DTYPE,
)

HWC_XHWC_NDHWC: Final[tuple[LayoutName, ...]] = (LAYOUT_HWC, LAYOUT_XHWC, LAYOUT_NDHWC)
IMAGE_DTYPES: Final[tuple[DTypeName, ...]] = (DTYPE_UINT8, DTYPE_FLOAT32)
IMAGE_CHANNELS: Final[tuple[ChannelSpec, ...]] = (CHANNEL_1, CHANNEL_3, CHANNEL_4, CHANNEL_GT4)
SCALAR_VECTOR_ARRAY: Final[tuple[ValueKind, ...]] = (
    VALUE_SCALAR,
    VALUE_PER_CHANNEL_VECTOR,
    VALUE_IMAGE_ARRAY,
)

NON_SQUARE_HW: Final[tuple[tuple[int, int], ...]] = PROPERTY_NON_SQUARE_HW
RESIZE_TARGET_WH: Final[tuple[int, int]] = NON_SQUARE_RESIZE_TARGET_WH
