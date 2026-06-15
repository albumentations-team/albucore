"""Shared non-square shape grids for benchmark scripts."""

from __future__ import annotations

PROPERTY_NON_SQUARE_HW: tuple[tuple[int, int], ...] = ((5, 7), (8, 11), (13, 17))
NON_SQUARE_IMAGE_HW: tuple[int, int] = (17, 23)
NON_SQUARE_RESIZE_TARGET_WH: tuple[int, int] = (31, 19)
PROPERTY_CHANNELS: tuple[int, ...] = (1, 3, 9)

SQUARE_BASE_HW: tuple[tuple[int, int], ...] = ((128, 128), (256, 256), (512, 512), (1024, 1024))
SMALL_SQUARE_HW: tuple[tuple[int, int], ...] = SQUARE_BASE_HW[:2]
MEDIUM_SQUARE_HW: tuple[tuple[int, int], ...] = SQUARE_BASE_HW[1:3]
LARGE_SQUARE_HW: tuple[tuple[int, int], ...] = SQUARE_BASE_HW[1:]

ROUTER_HWC_QUICK_HW: tuple[tuple[int, int], ...] = ((128, 160), (240, 320))
ROUTER_HWC_FULL_HW: tuple[tuple[int, int], ...] = (*ROUTER_HWC_QUICK_HW, (480, 640), (768, 1024))

ROUTER_BATCH_STATS_QUICK_HW: tuple[tuple[int, int], ...] = ((240, 320),)
ROUTER_BATCH_STATS_FULL_HW: tuple[tuple[int, int], ...] = ((128, 160), (240, 320), (480, 640))

TARGETED_HWC_HW: tuple[tuple[int, int], ...] = ((240, 320), (480, 640), (768, 1024))
TARGETED_BATCH_NHWC: tuple[int, int, int] = (4, 240, 320)
TARGETED_VOLUME_NDHWC_PREFIX_AND_HW: tuple[int, int, int, int] = (2, 4, 64, 80)

STATS_HWC_SHAPES: tuple[tuple[int, int, int], ...] = (
    (240, 320, 3),
    (480, 640, 3),
    (480, 640, 9),
    (768, 1024, 3),
)
STATS_NHWC_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (4, 240, 320, 3),
    (4, 240, 320, 9),
)

LUT_SHARED_ROUTING_HW: tuple[tuple[int, int], ...] = (
    (240, 320),
    (320, 480),
    (384, 512),
    (480, 640),
    (640, 800),
    (768, 1024),
    (896, 1152),
)
LUT_SHARED_ROUTING_REPORT_HW: tuple[tuple[int, int], ...] = ((480, 640), (768, 1024))

MEMORY_SMOKE_RGB_SHAPE: tuple[int, int, int] = (240, 320, 3)
MEMORY_SMOKE_C9_SHAPE: tuple[int, int, int] = (128, 160, 9)

SCALE_LUT_SHAPES: tuple[tuple[int, ...], ...] = (
    (128, 160, 1),
    (128, 160, 3),
    (128, 160, 9),
    (240, 320, 1),
    (240, 320, 3),
    (240, 320, 9),
    (480, 640, 1),
    (480, 640, 3),
    (480, 640, 9),
    (768, 1024, 1),
    (768, 1024, 3),
    (768, 1024, 9),
    (16, 128, 160, 1),
    (16, 128, 160, 3),
    (32, 128, 160, 1),
    (32, 128, 160, 3),
    (64, 128, 160, 3),
    (96, 128, 160, 1),
    (48, 240, 320, 3),
    (2, 32, 128, 160, 1),
    (2, 32, 128, 160, 3),
    (2, 64, 128, 160, 3),
    (4, 16, 128, 160, 3),
)

MINIMAL_SHARED_LUT_SHAPES: tuple[tuple[int, ...], ...] = (
    (128, 128, 1),
    (256, 256, 3),
    (512, 512, 3),
    (640, 640, 3),
    (1024, 1024, 3),
    (64, 128, 128, 3),
    (128, 128, 128, 1),
    (2, 64, 128, 128, 3),
)

MINMAX_GLOBAL_HWC_SHAPES: tuple[tuple[int, int, int], ...] = (
    (224, 224, 3),
    (512, 512, 3),
    (1024, 1024, 1),
)
MINMAX_PER_CHANNEL_SHAPES: tuple[tuple[int, ...], ...] = (
    (128, 128, 1),
    (128, 128, 3),
    (128, 128, 9),
    (512, 512, 3),
    (512, 512, 9),
    (1024, 1024, 3),
    (1024, 1024, 9),
    (4, 256, 256, 3),
    (4, 256, 256, 9),
)

REDUCE_SUM_RESEARCH_SHAPES: tuple[tuple[int, ...], ...] = (
    (256, 256, 1),
    (256, 256, 3),
    (256, 256, 9),
    (512, 512, 1),
    (512, 512, 3),
    (512, 512, 9),
    (1024, 1024, 3),
    (16, 128, 128, 3),
    (32, 128, 128, 3),
    (4, 256, 256, 3),
)

GRAYSCALE_MULTIPLY_SHAPES: tuple[tuple[tuple[int, int], int], ...] = (
    ((256, 256), 1),
    ((512, 512), 1),
    ((1024, 1024), 1),
    ((256, 256), 3),
)

NORMALIZE_NUMKONG_HWC_SHAPES: tuple[tuple[int, int, int], ...] = (
    (128, 128, 3),
    (256, 256, 3),
    (512, 512, 3),
    (1024, 1024, 3),
    (512, 512, 9),
)

NUMKONG_BLEND_UINT8_HWC_SHAPES: tuple[tuple[int, int, int], ...] = (
    (128, 128, 1),
    (128, 128, 3),
    (256, 256, 3),
    (512, 512, 3),
)
NUMKONG_BLEND_FLOAT32_HWC_SHAPES: tuple[tuple[int, int, int], ...] = (
    (256, 256, 3),
    (512, 512, 3),
)
NUMKONG_REDUCTION_HWC_SHAPES: tuple[tuple[int, int, int], ...] = NUMKONG_BLEND_FLOAT32_HWC_SHAPES
NUMKONG_FLOAT32_REDUCTION_SHAPE: tuple[int, int, int] = (512, 512, 3)

SZ_LUT_BENCHMARK_SHAPES: tuple[tuple[int, ...], ...] = (
    (128, 128, 1),
    (128, 128, 3),
    (256, 256, 1),
    (256, 256, 3),
    (256, 256, 9),
    (512, 512, 1),
    (512, 512, 3),
    (512, 512, 9),
    (1024, 1024, 1),
    (1024, 1024, 3),
    (1024, 1024, 9),
    (96, 96, 9),
    (32, 128, 128, 1),
    (64, 128, 128, 3),
    (128, 128, 128, 1),
    (48, 256, 256, 3),
    (96, 160, 160, 3),
    (6, 32, 32, 9),
    (2, 32, 128, 128, 3),
    (2, 64, 128, 128, 3),
    (1, 128, 128, 128, 1),
)
