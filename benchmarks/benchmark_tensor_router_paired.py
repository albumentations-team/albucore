#!/usr/bin/env python3
"""Compare float16 and float32 public routers with channel-first CPU Tensor inputs.

Tensor construction and control-Tensor preparation happen before timed samples.
The call timing includes Albucore's Tensor-to-NumPy bridge, routing, output
wrapping, and any operation-specific work. Pass ``--with-geometric`` to include
2D/3D resampling and volume-batch routers.

Run from the repository root::

    uv run python benchmarks/benchmark_tensor_router_paired.py --threads 1 --output-json _internal/tensor.json
    uv run python benchmarks/benchmark_tensor_router_paired.py --threads 1 --with-geometric --output-json _internal/tensor-full.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Set thread controls before loading numerical libraries.
from benchmark_threads import THREAD_COUNT, configure_libraries  # noqa: I001

import cv2
import numkong as nk
import numpy as np
import torch
from shape_grids import ROUTER_BATCH_STATS_FULL_HW, ROUTER_BATCH_STATS_QUICK_HW

import benchmark_router_synthetic as router_bench

_TENSOR_PERMUTATIONS: dict[str, tuple[int, ...]] = {
    "CHW": (2, 0, 1),
    "NCHW": (0, 3, 1, 2),
    "CDHW": (3, 0, 1, 2),
    "NCDHW": (0, 4, 1, 2, 3),
}
_VOLUME_OPS = {
    "gaussian_blur3d",
    "flip_volume",
    "pad3d",
    "resize3d",
    "rot90_volume",
    "separable_filter3d",
    "warp_affine3d",
    "remap3d",
    "transpose_volume",
}
_BATCH_VOLUME_OPS = {"warp_affine3d_batch", "remap3d_batch"}
_QUICK_VOLUME_SHAPES = ((5, 11, 13, 1), (5, 11, 13, 3), (16, 128, 160, 1), (16, 128, 160, 3))
_FULL_VOLUME_SHAPES = (
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (16, 128, 160, 9),
    (32, 128, 160, 3),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
    (48, 240, 320, 3),
)
_FULL_IMAGE_OPERANDS: dict[str, tuple[int, ...]] = {
    "add_array": (0,),
    "multiply_by_array": (0,),
    "add_weighted": (1,),
}
_UINT8_LUT = np.random.default_rng(7).permutation(256).astype(np.uint8)


def _tensor_layout(operation: str, array_layout: str, array: np.ndarray) -> str:
    if array_layout == "NHWC":
        return "NCHW"
    if array_layout == "NDHWC":
        return "NCDHW"
    if array_layout == "DHWC":
        return "CDHW"
    if operation in _BATCH_VOLUME_OPS and array.ndim == 5:
        return "NCDHW"
    if operation in _VOLUME_OPS and array.ndim == 4:
        return "CDHW"
    return "CHW"


def _channel_first_tensor(array: np.ndarray, layout: str) -> torch.Tensor:
    if not array.flags.writeable or any(stride < 0 for stride in array.strides):
        array = np.array(array, copy=True, order="C")
    return torch.from_numpy(array).permute(_TENSOR_PERMUTATIONS[layout]).contiguous()


class TensorRouterAdapter:
    """Adapt the shared NumPy benchmark builders to existing Tensor router calls."""

    def __init__(self, alb: Any, operation: str, array_layout: str, image: np.ndarray) -> None:
        self._alb = alb
        self._operation = operation
        self._array_layout = array_layout
        self._tensors: dict[int, tuple[torch.Tensor, str]] = {}
        self._base_shape = image.shape
        self._base_dtype = image.dtype
        self._base_tensor = self._tensor_for(image)
        self._float16_operands: dict[int, np.ndarray | None] = {}

    def _tensor_for(self, array: np.ndarray) -> tuple[torch.Tensor, str]:
        key = id(array)
        if key not in self._tensors:
            layout = _tensor_layout(self._operation, self._array_layout, array)
            self._tensors[key] = (_channel_first_tensor(array, layout), layout)
        return self._tensors[key]

    def _exact_float16_operand(self, operand: np.ndarray) -> np.ndarray:
        key = id(operand)
        if key not in self._float16_operands:
            half = operand.astype(np.float16)
            self._float16_operands[key] = half if np.array_equal(half.astype(np.float32), operand) else None
        half_operand = self._float16_operands[key]
        return operand if half_operand is None else half_operand

    def __getattr__(self, name: str) -> Any:
        function = getattr(self._alb, name)
        try:
            accepts_layout = "layout" in inspect.signature(function).parameters
        except (TypeError, ValueError):
            accepts_layout = False

        def call(image: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            if (
                self._operation == "sz_lut"
                and image.shape == self._base_shape
                and image.dtype == self._base_dtype
            ):
                tensor, layout = self._base_tensor
            else:
                tensor, layout = self._tensor_for(image)

            arguments = list(args)
            if self._operation == "apply_uint8_lut" and arguments:
                arguments[0] = _UINT8_LUT
            for index in _FULL_IMAGE_OPERANDS.get(self._operation, ()):
                if index < len(arguments):
                    operand = arguments[index]
                    if isinstance(operand, np.ndarray) and operand.shape == image.shape:
                        if self._operation == "multiply_by_array" and self._base_dtype == np.float16:
                            operand = self._exact_float16_operand(operand)
                        arguments[index] = self._tensor_for(operand)[0]
            if accepts_layout:
                kwargs["layout"] = layout
            return function(tensor, *arguments, **kwargs)

        return call


def _tensor_builder(
    alb: Any,
    operation: str,
    array_layout: str,
    builder: Any,
    image: np.ndarray,
) -> Any:
    if operation == "sz_lut":
        tensor = _channel_first_tensor(image, _tensor_layout(operation, array_layout, image))
        thunk = lambda: alb.sz_lut(tensor.clone(), _UINT8_LUT, inplace=True)
        thunk()
        return thunk
    adapter = TensorRouterAdapter(alb, operation, array_layout, image)
    thunk = builder(adapter, image)
    thunk()
    return thunk


def _tensor_builder_factory(operation: str, array_layout: str, builder: Any) -> Any:
    def build(alb: Any, image: np.ndarray) -> Any:
        return _tensor_builder(alb, operation, array_layout, builder, image)

    return build


def _tensor_row_metadata(operation: str, array_layout: str, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    if operation in _BATCH_VOLUME_OPS:
        channels = shape[-1]
        height, width = min(32, shape[-3]), min(40, shape[-2])
        return "NCDHW", (4, channels, 5, height, width)
    if operation in _VOLUME_OPS:
        depth, height, width, channels = shape[-4:]
        return "CDHW", (channels, depth, height, width)
    if array_layout == "NHWC":
        batch, height, width, channels = shape
        return "NCHW", (batch, channels, height, width)
    if array_layout == "NDHWC":
        batch, depth, height, width, channels = shape
        return "NCDHW", (batch, channels, depth, height, width)
    if array_layout == "DHWC":
        depth, height, width, channels = shape
        return "CDHW", (channels, depth, height, width)
    height, width, channels = shape
    return "CHW", (channels, height, width)


def _record_single_image(
    rows: list[router_bench.BenchRow],
    alb: Any,
    op_name: str,
    array_layout: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    image: np.ndarray,
    builder: Any,
    repeats: int,
    warmup: int,
) -> None:
    tensor_layout, tensor_shape = _tensor_row_metadata(op_name, array_layout, shape)
    timing, status, detail = router_bench._bench(
        alb,
        image,
        _tensor_builder_factory(op_name, array_layout, builder),
        repeats,
        warmup,
    )
    rows.append(
        router_bench._bench_row(op_name, tensor_layout, tensor_shape, np.dtype(dtype).name, timing, status, detail),
    )


def _record_image_pair(
    rows: list[router_bench.BenchRow],
    alb: Any,
    op_name: str,
    array_layout: str,
    shape: tuple[int, ...],
    half_image: np.ndarray,
    float32_image: np.ndarray,
    builder: Any,
    repeats: int,
    warmup: int,
) -> None:
    tensor_layout, tensor_shape = _tensor_row_metadata(op_name, array_layout, shape)
    rows.extend(
        router_bench._bench_pair_rows(
            alb,
            op_name,
            tensor_layout,
            tensor_shape,
            _tensor_builder_factory(op_name, array_layout, builder),
            half_image,
            float32_image,
            repeats,
            warmup,
        ),
    )


def _matrix_rows(alb: Any, rng: np.random.Generator, repeats: int, warmup: int) -> list[router_bench.BenchRow]:
    rows: list[router_bench.BenchRow] = []
    half_a = rng.random((128, 64), dtype=np.float32).astype(np.float16)
    half_b = rng.random((64, 32), dtype=np.float32).astype(np.float16)
    float32_a, float32_b = half_a.astype(np.float32), half_b.astype(np.float32)
    tensors = (
        ("float16", torch.from_numpy(half_a), torch.from_numpy(half_b)),
        ("float32", torch.from_numpy(float32_a), torch.from_numpy(float32_b)),
    )
    half_a_t, half_b_t = tensors[0][1:]
    float_a_t, float_b_t = tensors[1][1:]
    timing_half, timing_float32 = router_bench._bench_interleaved_pair(
        lambda: alb.matmul(half_a_t, half_b_t),
        lambda: alb.matmul(float_a_t, float_b_t),
        repeats,
        warmup,
    )
    shape = (128, 64, 64, 32)
    rows.append(router_bench._bench_row("matmul", "matrix", shape, "float16", timing_half, "ok", ""))
    rows.append(router_bench._bench_row("matmul", "matrix", shape, "float32", timing_float32, "ok", ""))

    first_u8 = rng.integers(0, 256, size=(128, 64), dtype=np.uint8)
    second_u8 = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    u8_a, u8_b = torch.from_numpy(first_u8), torch.from_numpy(second_u8)
    timing = router_bench.bench_wall_ms(lambda: alb.matmul(u8_a, u8_b), repeats, warmup)
    rows.append(router_bench._bench_row("matmul", "matrix", shape, "uint8", timing, "ok", ""))
    return rows


def _pairwise_rows(alb: Any, rng: np.random.Generator, repeats: int, warmup: int) -> list[router_bench.BenchRow]:
    rows: list[router_bench.BenchRow] = []
    first_half = rng.random((24, 3), dtype=np.float32).astype(np.float16)
    second_half = rng.random((16, 3), dtype=np.float32).astype(np.float16)
    first_float32, second_float32 = first_half.astype(np.float32), second_half.astype(np.float32)
    first_half_t, second_half_t = torch.from_numpy(first_half), torch.from_numpy(second_half)
    first_float32_t, second_float32_t = torch.from_numpy(first_float32), torch.from_numpy(second_float32)
    timing_half, timing_float32 = router_bench._bench_interleaved_pair(
        lambda: alb.pairwise_distances_squared(first_half_t, second_half_t),
        lambda: alb.pairwise_distances_squared(first_float32_t, second_float32_t),
        repeats,
        warmup,
    )
    shape = (24, 16, 3)
    rows.append(
        router_bench._bench_row("pairwise_distances_squared", "points", shape, "float16", timing_half, "ok", ""),
    )
    rows.append(
        router_bench._bench_row("pairwise_distances_squared", "points", shape, "float32", timing_float32, "ok", ""),
    )

    first_u8 = rng.integers(0, 256, size=(24, 3), dtype=np.uint8)
    second_u8 = rng.integers(0, 256, size=(16, 3), dtype=np.uint8)
    first_u8_t, second_u8_t = torch.from_numpy(first_u8), torch.from_numpy(second_u8)
    timing = router_bench.bench_wall_ms(
        lambda: alb.pairwise_distances_squared(first_u8_t, second_u8_t),
        repeats,
        warmup,
    )
    rows.append(
        router_bench._bench_row("pairwise_distances_squared", "points", shape, "uint8", timing, "ok", ""),
    )
    return rows


def _registry_tensor_volume_ops() -> list[tuple[str, Any]]:
    def gaussian_blur(alb: Any, volume: np.ndarray) -> Any:
        return lambda: alb.gaussian_blur3d(volume, sigma=(0.75, 1.25, 1.75))

    def pad(alb: Any, volume: np.ndarray) -> Any:
        return lambda: alb.pad3d(volume, (1, 2, 2, 3, 3, 4), value=7)

    def resize(alb: Any, volume: np.ndarray) -> Any:
        depth, height, width, _ = volume.shape
        target = max(depth // 2, 1), max(height // 2, 1), max(width // 2, 1)
        return lambda: alb.resize3d(volume, target, interpolation=cv2.INTER_LINEAR)

    def separable_filter(alb: Any, volume: np.ndarray) -> Any:
        kernels = tuple(np.array((1.0, 2.0, 1.0), dtype=np.float32) / 4.0 for _ in range(3))
        return lambda: alb.separable_filter3d(volume, kernels)

    def warp_affine(alb: Any, volume: np.ndarray) -> Any:
        depth, height, width, _ = volume.shape
        matrix = np.array(
            ((0.95, 0.1, 0.0, 0.25), (0.0, 1.05, 0.1, -0.25), (0.05, 0.0, 1.0, 0.125)),
            dtype=np.float32,
        )
        target = max(depth // 2, 1), max(height // 2, 1), max(width // 2, 1)
        return lambda: alb.warp_affine3d(volume, matrix, target, interpolation=cv2.INTER_LINEAR)

    def remap(alb: Any, volume: np.ndarray) -> Any:
        depth, height, width, _ = volume.shape
        z, y, x = np.meshgrid(
            (np.arange(depth, dtype=np.float32) * 2 + 1) / depth - 1,
            (np.arange(height, dtype=np.float32) * 2 + 1) / height - 1,
            (np.arange(width, dtype=np.float32) * 2 + 1) / width - 1,
            indexing="ij",
        )
        grid = np.stack((x, y, z), axis=-1)
        return lambda: alb.remap3d(volume, grid, interpolation=cv2.INTER_LINEAR)

    def flip(alb: Any, volume: np.ndarray) -> Any:
        return lambda: alb.flip_volume(volume, (1, 3))

    def rotate(alb: Any, volume: np.ndarray) -> Any:
        return lambda: alb.rot90_volume(volume, 1, (2, 3))

    def transpose(alb: Any, volume: np.ndarray) -> Any:
        return lambda: alb.transpose_volume(volume, 2, 3)

    return [
        ("gaussian_blur3d", gaussian_blur),
        ("pad3d", pad),
        ("resize3d", resize),
        ("separable_filter3d", separable_filter),
        ("warp_affine3d", warp_affine),
        ("remap3d", remap),
        ("flip_volume", flip),
        ("rot90_volume", rotate),
        ("transpose_volume", transpose),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use the first two canonical HWC sizes and C=1/3.")
    parser.add_argument("--with-geometric", action="store_true", help="Also benchmark 2D/3D geometric routers.")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--volume-repeats", type=int, default=7)
    parser.add_argument("--volume-warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--skip-ops", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if min(args.repeats, args.volume_repeats, args.threads) < 1 or min(args.warmup, args.volume_warmup) < 0:
        parser.error("repeat/thread counts must be >= 1 and warmup counts must be >= 0")

    thread_settings = configure_libraries(torch, cv2, args.threads)
    skip_ops = router_bench._parse_skip_ops(args.skip_ops)
    import albucore as alb  # noqa: PLC0415
    import albucore.functions as alb_fn  # noqa: PLC0415
    import albucore.geometric as alb_geo  # noqa: PLC0415

    declared = getattr(alb_fn, "__all__", None)
    export_names = set(declared if declared is not None else router_bench._FUNCTIONS_PUBLIC_ROUTERS_FALLBACK)
    functions_all = list(declared if declared is not None else router_bench._FUNCTIONS_PUBLIC_ROUTERS_FALLBACK)
    fn_reg = [(name, builder) for name, builder in router_bench._registry_functions() if name not in ("matmul", "pairwise_distances_squared")]
    stats_ops = {
        "mean",
        "std",
        "mean_std",
        "reduce_sum",
        "mean_per_channel",
        "std_per_channel",
        "mean_std_per_channel",
        "reduce_sum_per_channel",
    }
    skip_uint8_only = {"apply_uint8_lut", "sz_lut"}
    skip_float_only = {"from_float"}
    geometric = (
        [(name, builder) for name, builder in router_bench._registry_geometric() if name not in _VOLUME_OPS]
        if args.with_geometric
        else []
    )
    batch_geometric = router_bench._registry_volume_batches() if args.with_geometric else []
    volume_geometric = _registry_tensor_volume_ops() if args.with_geometric else []
    rows: list[router_bench.BenchRow] = []
    rng = np.random.default_rng(42)
    paired_half_images: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}

    for array_layout, shape, dtype in router_bench._iter_hwc(args.quick):
        key = (array_layout, shape)
        if dtype == np.float32 and key in paired_half_images:
            half_image = paired_half_images[key]
            image = half_image.astype(np.float32)
        else:
            image = router_bench._make_img(rng, shape, dtype)
        if dtype == np.float16:
            paired_half_images[key] = image
            continue
        if dtype == np.float32:
            half_image = paired_half_images.get(key)
        else:
            half_image = None

        for op_name, builder in fn_reg:
            if (op_name not in export_names and op_name not in stats_ops) or op_name in skip_ops:
                continue
            if op_name in skip_uint8_only and dtype != np.uint8:
                continue
            if op_name in skip_float_only and dtype not in (np.float16, np.float32):
                continue
            if op_name not in stats_ops and not hasattr(alb, op_name):
                continue
            if half_image is not None:
                _record_image_pair(rows, alb, op_name, array_layout, shape, half_image, image, builder, args.repeats, args.warmup)
            else:
                _record_single_image(rows, alb, op_name, array_layout, shape, dtype, image, builder, args.repeats, args.warmup)

        for op_name, builder in geometric:
            if op_name in skip_ops or (op_name in skip_uint8_only and dtype != np.uint8):
                continue
            if op_name in skip_float_only and dtype not in (np.float16, np.float32):
                continue
            api_name = router_bench._GEOMETRIC_API_ALIASES.get(op_name, op_name.removesuffix("_batch"))
            if not hasattr(alb, api_name):
                continue
            if half_image is not None:
                _record_image_pair(rows, alb, op_name, array_layout, shape, half_image, image, builder, args.repeats, args.warmup)
            else:
                _record_single_image(rows, alb, op_name, array_layout, shape, dtype, image, builder, args.repeats, args.warmup)

    if args.with_geometric:
        volume_shapes = _QUICK_VOLUME_SHAPES if args.quick else _FULL_VOLUME_SHAPES
        half_volumes: dict[tuple[int, ...], np.ndarray] = {}
        for shape in volume_shapes:
            for dtype in (np.uint8, np.float16, np.float32):
                if dtype == np.float16:
                    half_volumes[shape] = router_bench._make_img(rng, shape, dtype)
                    continue
                if dtype == np.float32 and shape in half_volumes:
                    half_volume = half_volumes[shape]
                    volume = half_volume.astype(np.float32)
                else:
                    volume = router_bench._make_img(rng, shape, dtype)
                half_volume = half_volumes.get(shape) if dtype == np.float32 else None
                for op_name, builder in volume_geometric:
                    if op_name in skip_ops:
                        continue
                    if half_volume is not None:
                        _record_image_pair(
                            rows,
                            alb,
                            op_name,
                            "DHWC",
                            shape,
                            half_volume,
                            volume,
                            builder,
                            args.volume_repeats,
                            args.volume_warmup,
                        )
                    else:
                        _record_single_image(
                            rows,
                            alb,
                            op_name,
                            "DHWC",
                            shape,
                            dtype,
                            volume,
                            builder,
                            args.volume_repeats,
                            args.volume_warmup,
                        )

        for channels in (1, 3, 9):
            shape = (32, 40, channels)
            half_image = router_bench._make_img(rng, shape, np.float16)
            float32_image = half_image.astype(np.float32)
            uint8_image = router_bench._make_img(rng, shape, np.uint8)
            for op_name, builder in batch_geometric:
                if op_name in skip_ops:
                    continue
                _record_image_pair(
                    rows,
                    alb,
                    op_name,
                    "HWC",
                    shape,
                    half_image,
                    float32_image,
                    builder,
                    args.volume_repeats,
                    args.volume_warmup,
                )
                _record_single_image(
                    rows,
                    alb,
                    op_name,
                    "HWC",
                    shape,
                    np.dtype(np.uint8),
                    uint8_image,
                    builder,
                    args.volume_repeats,
                    args.volume_warmup,
                )

    paired_half_batches: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    for array_layout, shape, dtype in router_bench._iter_batch_stats(args.quick):
        key = (array_layout, shape)
        if dtype == np.float32 and key in paired_half_batches:
            half_image = paired_half_batches[key]
            image = half_image.astype(np.float32)
        else:
            image = router_bench._make_img(rng, shape, dtype)
        if dtype == np.float16:
            paired_half_batches[key] = image
            continue
        half_image = paired_half_batches.get(key) if dtype == np.float32 else None

        for op_name, builder in fn_reg:
            if op_name in ("matmul", "pairwise_distances_squared") or op_name in skip_ops:
                continue
            if op_name not in export_names and not op_name.endswith("_per_channel"):
                continue
            if op_name in skip_uint8_only and dtype != np.uint8:
                continue
            if op_name in skip_float_only and dtype not in (np.float16, np.float32):
                continue
            if half_image is not None:
                _record_image_pair(rows, alb, op_name, array_layout, shape, half_image, image, builder, args.repeats, args.warmup)
            else:
                _record_single_image(rows, alb, op_name, array_layout, shape, dtype, image, builder, args.repeats, args.warmup)

        for op_name, builder in geometric:
            if op_name in skip_ops or (op_name in skip_uint8_only and dtype != np.uint8):
                continue
            api_name = router_bench._GEOMETRIC_API_ALIASES.get(op_name, op_name.removesuffix("_batch"))
            if not hasattr(alb, api_name):
                continue
            if half_image is not None:
                _record_image_pair(rows, alb, op_name, array_layout, shape, half_image, image, builder, args.repeats, args.warmup)
            else:
                _record_single_image(rows, alb, op_name, array_layout, shape, dtype, image, builder, args.repeats, args.warmup)

    if "matmul" in export_names and "matmul" not in skip_ops:
        rows.extend(_matrix_rows(alb, rng, args.repeats, args.warmup))
    if "pairwise_distances_squared" in export_names and "pairwise_distances_squared" not in skip_ops:
        rows.extend(_pairwise_rows(alb, rng, args.repeats, args.warmup))

    root = Path(alb.__file__).resolve().parent.parent
    try:
        distribution_version = importlib.metadata.version("albucore")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = getattr(alb, "__version__", "unknown")
    payload = {
        "meta": {
            "albucore_version": getattr(alb, "__version__", "unknown"),
            "distribution_version": distribution_version,
            "git_head_short": router_bench._git_head_short(root),
            "container": "torch_cpu",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "supported_dtypes": ["uint8", "float16", "float32"],
            "tensor_layouts": ["CHW", "NCHW", "CDHW", "NCDHW"],
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "thread_settings": thread_settings,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numkong": importlib.metadata.version("numkong"),
            "numkong_capabilities": nk.get_capabilities(),
            "stringzilla": importlib.metadata.version("stringzilla"),
            "quick": args.quick,
            "with_geometric": args.with_geometric,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "volume_repeats": args.volume_repeats,
            "volume_warmup": args.volume_warmup,
            "float16_control_preparation": "exact-half full-array controls are cast once before timing",
            "functions_all": functions_all,
            "geometric_all": list(alb_geo.__all__),
            "batch_shapes": [
                (4, height, width, channels)
                for height, width in (
                    ROUTER_BATCH_STATS_QUICK_HW if args.quick else ROUTER_BATCH_STATS_FULL_HW
                )
                for channels in ((1, 3) if args.quick else (1, 3, 9))
            ],
            "volume_shapes": _QUICK_VOLUME_SHAPES if args.quick else _FULL_VOLUME_SHAPES,
        },
        "rows": [asdict(row) for row in rows],
    }
    result = json.dumps(payload, indent=2)
    if args.output_json:
        path = Path(args.output_json)
        path.write_text(result)
        print(f"Wrote {path}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
