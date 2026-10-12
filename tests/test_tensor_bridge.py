"""Contract tests for channel-last NumPy views on channel-first CPU Tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from albucore import (
    add,
    add_array,
    add_vector,
    apply_uint8_lut,
    clip,
    clipped,
    exp,
    float32_io,
    from_float,
    get_image_data,
    get_num_channels,
    hflip,
    is_grayscale_image,
    is_rgb_image,
    matmul,
    mean,
    mean_std,
    multiply,
    multiply_by_vector,
    multiply_add,
    multiply_by_array,
    pairwise_distances_squared,
    reduce_sum,
    std,
    sz_lut,
    to_float,
    uint8_io,
    vflip,
)
from albucore.tensor_bridge import dispatch_tensor_image, numpy_to_tensor_layout, tensor_to_numpy_channel_last
from albucore.utils import TensorLayout


@pytest.mark.parametrize(
    ("layout", "shape", "channel_axis"),
    [
        ("CHW", (3, 7, 11), 0),
        ("NCHW", (2, 3, 7, 11), 1),
        ("CDHW", (3, 5, 7, 11), 0),
        ("NCDHW", (2, 3, 5, 7, 11), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_tensor_bridge_uses_views_and_restores_declared_layout(
    layout: TensorLayout,
    shape: tuple[int, ...],
    channel_axis: int,
    dtype: torch.dtype,
) -> None:
    tensor = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape).to(dtype)

    array = tensor_to_numpy_channel_last(tensor, layout)
    restored = numpy_to_tensor_layout(array, layout)

    assert array.shape[-1] == shape[channel_axis]
    assert array.dtype == tensor.numpy().dtype
    assert np.shares_memory(array, tensor.numpy())
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert restored.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(restored, tensor)


@pytest.mark.parametrize("layout", ["CHW", "NCHW", "CDHW", "NCDHW"])
def test_tensor_router_returns_the_original_tensor_for_identity(layout: Any) -> None:
    shape = {"CHW": (3, 7, 11), "NCHW": (2, 3, 7, 11), "CDHW": (3, 5, 7, 11), "NCDHW": (2, 3, 5, 7, 11)}[layout]
    tensor = torch.rand(shape, dtype=torch.float16)

    result = dispatch_tensor_image(tensor, layout, lambda image: image)

    assert result is tensor


def test_tensor_router_requires_layout_when_numpy_kernel_needs_channels() -> None:
    tensor = torch.rand((3, 7, 11), dtype=torch.float16)

    with pytest.raises(TypeError, match="explicit layout"):
        tensor_to_numpy_channel_last(tensor, None)


def test_layout_aware_tensor_shape_helpers_and_clip() -> None:
    tensor = torch.linspace(-1, 2, steps=3 * 5 * 7, dtype=torch.float32).reshape(3, 5, 7)

    assert get_num_channels(tensor, layout="CHW") == 3
    assert is_rgb_image(tensor, layout="CHW")
    assert not is_grayscale_image(tensor, layout="CHW")
    assert clip(tensor, np.dtype(np.float16)).dtype == torch.float16
    metadata = get_image_data({"image": tensor})
    assert metadata == {"dtype": torch.float32, "height": 5, "width": 7, "num_channels": 3}


def test_clipped_decorator_supports_tensor_results() -> None:
    @clipped
    def double(image: torch.Tensor) -> torch.Tensor:
        return image * 2

    tensor = torch.full((1, 3, 4), 0.75, dtype=torch.float16)

    result = double(tensor)

    assert result.dtype == torch.float16
    torch.testing.assert_close(result, torch.ones_like(tensor))


def test_tensor_inplace_unary_mutates_leaf_view_and_its_alias() -> None:
    base = torch.linspace(0.1, 0.9, steps=12, dtype=torch.float16)
    view = base.view(3, 4)
    original = base.clone()

    result = exp(view, inplace=True)

    assert result is view
    torch.testing.assert_close(base, torch.exp(original))


def test_tensor_dtype_conversion_preserves_container_and_float16_policy() -> None:
    half = torch.tensor([0.0, 0.3333, 1.25], dtype=torch.float16)
    uint8 = torch.tensor([0, 127, 255], dtype=torch.uint8)

    assert to_float(half) is half
    torch.testing.assert_close(to_float(uint8), uint8.to(torch.float32) / 255.0)
    torch.testing.assert_close(from_float(half, np.dtype(np.float32)), half.to(torch.float32))
    torch.testing.assert_close(from_float(half, np.dtype(np.float16)), half)
    torch.testing.assert_close(
        from_float(torch.tensor([0.5 / 255, 1.0], dtype=torch.float16), np.dtype(np.uint8)),
        torch.tensor([0, 255], dtype=torch.uint8),
    )


def test_contiguous_float16_tensor_to_uint8_stays_within_one_over_full_half_domain() -> None:
    half_bits = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256, 1)
    half_values = torch.from_numpy(half_bits.view(np.float16).copy())

    actual = from_float(half_values, np.dtype(np.uint8))
    reference = from_float(half_values.to(torch.float32), np.dtype(np.uint8))

    difference = (actual.to(torch.int16) - reference.to(torch.int16)).abs()
    assert int(difference.max()) <= 1


def test_float16_tensor_to_uint8_uses_wide_rounding_for_unrepresentable_scale() -> None:
    half_bits = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256, 1)
    half_values = torch.from_numpy(half_bits.view(np.float16).copy())

    actual = from_float(half_values, np.dtype(np.uint8), max_value=17.3)
    reference = from_float(half_values.to(torch.float32), np.dtype(np.uint8), max_value=17.3)

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_dtype_io_decorators_restore_half_and_tensor_layout() -> None:
    @float32_io
    def twice(image: np.ndarray) -> np.ndarray:
        return image * np.float32(2.0)

    @uint8_io
    def invert(image: np.ndarray) -> np.ndarray:
        return np.subtract(np.uint8(255), image, dtype=np.uint8)

    image = torch.linspace(0.0, 1.0, steps=3 * 5 * 7, dtype=torch.float16).reshape(3, 5, 7)
    result_float = twice(image, layout="CHW")
    result_uint8 = invert(image, layout="CHW")
    quantized = np.rint(image.permute(1, 2, 0).numpy().astype(np.float32) * 255.0).astype(np.uint8)
    expected_uint8 = ((255 - quantized).astype(np.float32) / 255.0).astype(np.float16)

    assert result_float.dtype == torch.float16
    assert result_uint8.dtype == torch.float16
    np.testing.assert_allclose(
        result_float.permute(1, 2, 0).numpy(),
        (image.permute(1, 2, 0).numpy() * 2).astype(np.float16),
    )
    np.testing.assert_array_equal(result_uint8.permute(1, 2, 0).numpy(), expected_uint8)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_matrix_routers_accept_tensor_inputs_and_preserve_output_policy(dtype: torch.dtype) -> None:
    first = torch.arange(15, dtype=torch.float32).reshape(5, 3).to(dtype)
    second = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype)

    product = matmul(first, second)
    distances = pairwise_distances_squared(first, first)

    assert isinstance(product, torch.Tensor)
    assert product.dtype == dtype
    assert product.shape == (5, 4)
    assert distances.dtype == torch.float32
    torch.testing.assert_close(product, torch.from_numpy(first.numpy() @ second.numpy()))
    torch.testing.assert_close(distances, torch.from_numpy(pairwise_distances_squared(first.numpy(), first.numpy())))


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("CHW", (3, 5, 7)),
        ("NCHW", (2, 3, 5, 7)),
        ("CDHW", (3, 4, 5, 7)),
        ("NCDHW", (2, 3, 4, 5, 7)),
    ],
)
@pytest.mark.parametrize("image_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("operand_dtype", [torch.float16, torch.float32])
def test_tensor_add_array_preserves_input_dtype_and_layout(
    layout: str,
    shape: tuple[int, ...],
    image_dtype: torch.dtype,
    operand_dtype: torch.dtype,
) -> None:
    image = torch.linspace(-0.75, 0.875, steps=int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(image_dtype)
    operand = torch.full(shape, 0.015625, dtype=operand_dtype)
    expected = (image.to(torch.float32) + operand.to(torch.float32)).to(image_dtype)

    result = add_array(image, operand, layout=layout)

    assert result.dtype == image_dtype
    assert result.shape == image.shape
    assert result.data_ptr() != image.data_ptr()
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("CHW", (3, 5, 7)),
        ("NCHW", (2, 3, 5, 7)),
        ("CDHW", (3, 4, 5, 7)),
        ("NCDHW", (2, 3, 4, 5, 7)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tensor_multiply_by_array_preserves_input_dtype_and_layout(
    layout: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    image = torch.linspace(0.125, 0.875, steps=int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(dtype)
    factor = torch.full(shape, 1.015625, dtype=dtype)
    expected = (image.to(torch.float32) * factor.to(torch.float32)).to(dtype)

    result = multiply_by_array(image, factor, layout=layout)

    assert result.dtype == dtype
    assert result.shape == image.shape
    assert result.data_ptr() != image.data_ptr()
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("layout", "numpy_shape", "tensor_permutation"),
    [
        ("CHW", (5, 7, 3), (2, 0, 1)),
        ("NCHW", (2, 5, 7, 3), (0, 3, 1, 2)),
        ("CDHW", (4, 5, 7, 3), (3, 0, 1, 2)),
        ("NCDHW", (2, 4, 5, 7, 3), (0, 4, 1, 2, 3)),
    ],
)
def test_tensor_multiply_add_preserves_two_step_half_rounding(
    layout: TensorLayout,
    numpy_shape: tuple[int, ...],
    tensor_permutation: tuple[int, ...],
) -> None:
    source = np.linspace(0.125, 0.875, num=int(np.prod(numpy_shape)), dtype=np.float32)
    source = source.reshape(numpy_shape).astype(np.float16)
    image = torch.from_numpy(source).permute(tensor_permutation).contiguous()
    channels = numpy_shape[-1]
    factor = np.array([1.125, 1.0, 0.875] * ((channels + 2) // 3), dtype=np.float32)[:channels]
    bias = np.array([0.25, -0.125, 0.125] * ((channels + 2) // 3), dtype=np.float32)[:channels]
    expected = numpy_to_tensor_layout(multiply_add(source, factor, bias), layout)

    result = multiply_add(image, factor, bias, layout=layout)

    assert result.dtype == torch.float16
    assert result.shape == image.shape
    assert result.data_ptr() != image.data_ptr()
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


def test_uint8_lut_routers_accept_tensors_and_preserve_inplace_contract() -> None:
    image = torch.arange(3 * 7 * 9, dtype=torch.uint8).reshape(3, 7, 9)
    shared_lut = (255 - np.arange(256, dtype=np.uint16)).astype(np.uint8)
    expected = torch.from_numpy(shared_lut[image.numpy()])

    result = sz_lut(image, shared_lut, inplace=False)
    assert result.dtype == torch.uint8
    torch.testing.assert_close(result, expected)
    inplace = image.clone()
    returned = sz_lut(inplace, shared_lut, inplace=True)
    assert returned is inplace
    torch.testing.assert_close(inplace, expected)
    routed = apply_uint8_lut(image, shared_lut)
    torch.testing.assert_close(routed, expected)

    channel_luts = np.stack([(np.arange(256, dtype=np.uint16) + index * 13) % 256 for index in range(3)], axis=-1)
    channel_luts = channel_luts.astype(np.uint8)[:, None, :]
    channel_last = image.permute(1, 2, 0).numpy()
    expected_channelwise = torch.from_numpy(apply_uint8_lut(channel_last, channel_luts)).permute(2, 0, 1)
    channelwise = apply_uint8_lut(image, channel_luts, layout="CHW")
    torch.testing.assert_close(channelwise, expected_channelwise)


@pytest.mark.parametrize(
    ("numpy_shape", "torch_layout", "torch_shape", "tensor_permutation"),
    [
        ((5, 7, 3), "CHW", (3, 5, 7), (2, 0, 1)),
        ((2, 5, 7, 3), "NCHW", (2, 3, 5, 7), (0, 3, 1, 2)),
        ((4, 5, 7, 3), "CDHW", (3, 4, 5, 7), (3, 0, 1, 2)),
        ((2, 4, 5, 7, 3), "NCDHW", (2, 3, 4, 5, 7), (0, 4, 1, 2, 3)),
    ],
)
def test_horizontal_and_vertical_flips_use_spatial_axes(
    numpy_shape: tuple[int, ...],
    torch_layout: str,
    torch_shape: tuple[int, ...],
    tensor_permutation: tuple[int, ...],
) -> None:
    image = np.arange(np.prod(numpy_shape), dtype=np.float16).reshape(numpy_shape)
    tensor = torch.from_numpy(image).permute(tensor_permutation)
    expected_horizontal = image[..., ::-1, :]
    expected_vertical = image[..., ::-1, :, :]

    np.testing.assert_array_equal(hflip(image), expected_horizontal)
    np.testing.assert_array_equal(vflip(image), expected_vertical)
    channel_last_permutation = tuple(np.argsort(tensor_permutation))
    torch.testing.assert_close(
        hflip(tensor).permute(channel_last_permutation),
        torch.from_numpy(expected_horizontal.copy()),
    )
    torch.testing.assert_close(
        vflip(tensor).permute(channel_last_permutation),
        torch.from_numpy(expected_vertical.copy()),
    )
    assert tensor.shape == torch_shape


def test_contiguous_float16_tensor_hflip_preserves_special_value_bits() -> None:
    half_bits = np.resize(
        np.array([0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x0001, 0x3C00], dtype=np.uint16),
        3 * 5 * 7,
    ).reshape(3, 5, 7)
    image = torch.from_numpy(half_bits.view(np.float16))

    result = hflip(image)
    expected_bits = torch.flip(image.view(torch.int16), dims=(-1,))

    assert torch.equal(result.view(torch.int16), expected_bits)


@pytest.mark.parametrize(
    ("layout", "shape"),
    [("CHW", (3, 5, 7)), ("NCHW", (2, 3, 5, 7)), ("CDHW", (3, 4, 5, 7)), ("NCDHW", (2, 3, 4, 5, 7))],
)
def test_tensor_statistics_restore_scalar_channel_and_explicit_axis_outputs(
    layout: str,
    shape: tuple[int, ...],
) -> None:
    tensor = torch.arange(np.prod(shape), dtype=torch.float16).reshape(shape)
    view = tensor_to_numpy_channel_last(tensor, layout)  # type: ignore[arg-type]

    for axis in (None, "per_channel", (0, 2)):
        actual_mean = mean(tensor, axis, layout=layout)
        actual_std = std(tensor, axis, layout=layout)
        if axis == "per_channel":
            expected_mean = torch.from_numpy(mean(view, axis))
            expected_std = torch.from_numpy(std(view, axis))
        else:
            tensor_to_numpy_axes = {
                "CHW": (2, 0, 1),
                "NCHW": (0, 3, 1, 2),
                "CDHW": (3, 0, 1, 2),
                "NCDHW": (0, 4, 1, 2, 3),
            }[layout]
            numpy_axes = None if axis is None else tuple(tensor_to_numpy_axes[index] for index in axis)
            numpy_values = view
            mean_values = np.mean(numpy_values, axis=numpy_axes, dtype=np.float64)
            std_values = np.std(numpy_values, axis=numpy_axes, dtype=np.float64) + 1e-4
            if numpy_axes is not None:
                reduced_tensor_axes = set(axis)
                remaining_numpy_axes = [axis_index for axis_index in range(tensor.ndim) if axis_index not in numpy_axes]
                desired_numpy_axes = [
                    numpy_axis
                    for tensor_axis, numpy_axis in enumerate(tensor_to_numpy_axes)
                    if tensor_axis not in reduced_tensor_axes
                ]
                permutation = tuple(remaining_numpy_axes.index(axis_index) for axis_index in desired_numpy_axes)
                mean_values = np.transpose(mean_values, permutation)
                std_values = np.transpose(std_values, permutation)
            expected_mean = torch.from_numpy(np.asarray(mean_values, dtype=np.float64))
            expected_std = torch.from_numpy(np.asarray(std_values, dtype=np.float64))
        torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual_std, expected_std, rtol=1e-12, atol=1e-12)

    actual_pair = mean_std(tensor, "per_channel", layout=layout)
    expected_pair = mean_std(view, "per_channel")
    torch.testing.assert_close(actual_pair[0], torch.as_tensor(expected_pair[0], dtype=torch.float64))
    torch.testing.assert_close(actual_pair[1], torch.as_tensor(expected_pair[1], dtype=torch.float64))
    assert reduce_sum(tensor, layout=layout).dtype == torch.float64


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("channels", [2, 3, 9])
@pytest.mark.parametrize("keepdims", [False, True])
def test_large_float16_tensor_global_mean_matches_wide_reference(
    layout: str,
    channels: int,
    keepdims: bool,
) -> None:
    batch = 4 if layout == "NCHW" else 1
    source = np.random.default_rng(401).uniform(-1, 1, size=(batch, 128, 160, channels)).astype(np.float16)
    tensor = (
        torch.from_numpy(source).permute(0, 3, 1, 2).contiguous()
        if layout == "NCHW"
        else torch.from_numpy(source[0]).permute(2, 0, 1).contiguous()
    )
    reference = source if layout == "NCHW" else source[0]
    expected = np.mean(reference, dtype=np.float64, keepdims=keepdims)

    result = mean(tensor, "global", keepdims=keepdims, layout=layout)

    assert result.dtype == torch.float64
    assert tuple(result.shape) == expected.shape
    torch.testing.assert_close(result, torch.as_tensor(expected, dtype=torch.float64), rtol=0, atol=1e-7)


@pytest.mark.parametrize("layout", ["CHW", "NCHW", "CDHW", "NCDHW"])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("keepdims", [False, True])
def test_large_float16_tensor_per_channel_mean_matches_wide_reference(
    layout: str,
    channels: int,
    keepdims: bool,
) -> None:
    shapes = {
        "CHW": (channels, 240, 320),
        "NCHW": (4, channels, 128, 160),
        "CDHW": (channels, 16, 32, 80),
        "NCDHW": (4, channels, 5, 32, 64),
    }
    image = torch.from_numpy(
        np.random.default_rng(607).uniform(-1, 1, size=shapes[layout]).astype(np.float16),
    )
    channel_axis = 0 if layout in ("CHW", "CDHW") else 1
    axes = tuple(axis for axis in range(image.ndim) if axis != channel_axis)
    expected = image.to(torch.float64).mean(dim=axes, keepdim=keepdims)

    actual = mean(image, "per_channel", keepdims=keepdims, layout=layout)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "layout",
    ["CHW", "NCHW", "CDHW", "NCDHW"],
)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("axis", [None, "per_channel", (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_large_float16_tensor_reduce_sum_matches_wide_reference(
    layout: str,
    channels: int,
    axis: Any,
    keepdims: bool,
) -> None:
    shape = {
        "CHW": (channels, 128, 160),
        "NCHW": (4, channels, 64, 64),
        "CDHW": (channels, 16, 32, 32),
        "NCDHW": (4, channels, 4, 32, 32),
    }[layout]
    image = torch.from_numpy(
        np.random.default_rng(727).uniform(-1, 1, size=shape).astype(np.float16),
    )
    channel_axis = 0 if layout in ("CHW", "CDHW") else 1
    if axis is None:
        axes = tuple(range(image.ndim))
    elif axis == "per_channel":
        axes = tuple(index for index in range(image.ndim) if index != channel_axis)
    else:
        axes = axis
    expected = np.sum(image.numpy(), axis=axes, dtype=np.float64, keepdims=keepdims)

    actual = reduce_sum(image, axis, keepdims=keepdims, layout=layout)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=torch.float64), rtol=0, atol=1e-10)


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("keepdims", [False, True])
def test_large_float16_tensor_global_std_and_mean_std_match_wide_reference(
    layout: str,
    channels: int,
    keepdims: bool,
) -> None:
    batch = 4 if layout == "NCHW" else 1
    source = np.random.default_rng(503).uniform(-1, 1, size=(batch, 128, 160, channels)).astype(np.float16)
    tensor = (
        torch.from_numpy(source).permute(0, 3, 1, 2).contiguous()
        if layout == "NCHW"
        else torch.from_numpy(source[0]).permute(2, 0, 1).contiguous()
    )
    reference = source if layout == "NCHW" else source[0]
    expected_mean = np.mean(reference, dtype=np.float64, keepdims=keepdims)
    expected_std = np.std(reference, dtype=np.float64, keepdims=keepdims) + 1e-4

    actual_std = std(tensor, "global", keepdims=keepdims, layout=layout)
    actual_mean_std = mean_std(tensor, "global", keepdims=keepdims, layout=layout)

    assert actual_std.dtype == torch.float64
    assert tuple(actual_std.shape) == expected_std.shape
    torch.testing.assert_close(actual_std, torch.as_tensor(expected_std), rtol=0, atol=1e-7)
    torch.testing.assert_close(actual_mean_std[0], torch.as_tensor(expected_mean), rtol=0, atol=1e-7)
    torch.testing.assert_close(actual_mean_std[1], torch.as_tensor(expected_std), rtol=0, atol=1e-7)


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("CHW", (1, 480, 640)),
        ("CHW", (1, 768, 1024)),
        ("NCHW", (4, 1, 240, 320)),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_large_single_channel_float16_tensor_std_matches_wide_reference(
    layout: str,
    shape: tuple[int, ...],
    keepdims: bool,
) -> None:
    source = np.random.default_rng(809).uniform(-1, 1, size=shape).astype(np.float16)
    image = torch.from_numpy(source)
    expected = np.std(source, dtype=np.float64, keepdims=keepdims) + 1e-4

    actual = std(image, "global", keepdims=keepdims, layout=layout)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=1e-7)


def test_large_single_channel_float16_tensor_std_handles_near_constant_input() -> None:
    source = np.full((1, 480, 640), np.float16(0.5), dtype=np.float16)
    source.flat[0] = np.nextafter(source.flat[0], np.float16(np.inf))
    expected = np.std(source, dtype=np.float64) + 1e-4

    actual = std(torch.from_numpy(source), "global", layout="CHW")

    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=1e-12)


def test_large_float16_tensor_std_f32_accumulator_stays_within_tolerance_on_half_domain() -> None:
    bits = np.arange(1 << 16, dtype=np.uint16)
    values = bits.view(np.float16)
    values = values[np.isfinite(values) & (values >= -1.0) & (values <= 1.0)]
    source = np.resize(values, (1, 256, 256)).astype(np.float16)
    expected = np.std(source, dtype=np.float64) + 1e-4

    actual = std(torch.from_numpy(source), "global", layout="CHW")

    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=1e-7)


def test_large_float16_tensor_std_widens_out_of_range_values() -> None:
    source = np.full((1, 480, 640), np.float16(60_000), dtype=np.float16)
    source.flat[::997] = np.nextafter(np.float16(60_000), np.float16(0))
    expected = np.std(source, dtype=np.float64) + 1e-4

    actual = std(torch.from_numpy(source), "global", layout="CHW")

    torch.testing.assert_close(actual, torch.as_tensor(expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_large_float16_tensor_std_preserves_nonfinite_values(value: float) -> None:
    source = np.zeros((1, 480, 640), dtype=np.float16)
    source.flat[0] = value
    with np.errstate(invalid="ignore"):
        expected = np.std(source, dtype=np.float64) + 1e-4
        actual = std(torch.from_numpy(source), "global", layout="CHW")

    np.testing.assert_equal(float(actual), expected)


@pytest.mark.parametrize("first", [np.inf, np.nan])
def test_large_float16_tensor_global_moments_widen_nonfinite_values(first: float) -> None:
    source = np.zeros((128, 160, 3), dtype=np.float16)
    source.flat[0] = first
    tensor = torch.from_numpy(source).permute(2, 0, 1).contiguous()

    with np.errstate(invalid="ignore"):
        actual_mean, actual_std = mean_std(tensor, "global", layout="CHW")
        expected_mean = np.mean(source, dtype=np.float64)
        expected_std = np.std(source, dtype=np.float64) + 1e-4

    np.testing.assert_equal(float(actual_mean), expected_mean)
    np.testing.assert_equal(float(actual_std), expected_std)


@pytest.mark.parametrize(
    ("layout", "shape"),
    [
        ("CHW", (3, 128, 160)),
        ("NCHW", (4, 3, 128, 160)),
        ("CDHW", (3, 16, 32, 32)),
        ("NCDHW", (4, 3, 16, 32, 32)),
    ],
)
@pytest.mark.parametrize(
    ("operation", "values"),
    [("add", [0.125, 0.25, 0.375]), ("multiply", [0.5, 1.0, 1.5])],
)
def test_float16_tensor_vector_arithmetic_uses_declared_channel_axis(
    layout: str,
    shape: tuple[int, ...],
    operation: str,
    values: list[float],
) -> None:
    image = torch.linspace(-1.0, 1.0, steps=int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.float16)
    channel_axis = 0 if layout in ("CHW", "CDHW") else 1
    parameter_shape = [1] * image.ndim
    parameter_shape[channel_axis] = 3
    parameter = torch.tensor(values, dtype=torch.float16).reshape(parameter_shape)
    control = np.asarray(values, dtype=np.float32)

    if operation == "add":
        result = add_vector(image, control, layout=layout)
        expected = image + parameter
    else:
        result = multiply_by_vector(image, control, layout=layout)
        expected = image * parameter

    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["add", "multiply"])
@pytest.mark.parametrize(("contiguous", "value"), [(True, 0.1), (False, 0.125)])
def test_float16_tensor_pointwise_scalar_matches_half_backend(
    operation: str,
    contiguous: bool,
    value: float,
) -> None:
    image = torch.linspace(-1.0, 1.0, steps=3 * 128 * 320, dtype=torch.float32).reshape(3, 128, 320).to(torch.float16)
    if not contiguous:
        image = image[:, :, ::2]

    actual = add(image, value, layout="CHW") if operation == "add" else multiply(image, value, layout="CHW")
    expected = image + value if operation == "add" else image * value

    assert actual.dtype == torch.float16
    assert actual.shape == image.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["add", "multiply"])
def test_float16_tensor_pointwise_unrepresentable_vector_uses_wide_path(operation: str) -> None:
    image = torch.linspace(-1.0, 1.0, steps=3 * 128 * 320, dtype=torch.float32).reshape(3, 128, 320).to(torch.float16)
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    control = torch.from_numpy(values).reshape(3, 1, 1)

    actual = add_vector(image, values, layout="CHW") if operation == "add" else multiply_by_vector(image, values, layout="CHW")
    expected = (
        image.to(torch.float32) + control
        if operation == "add"
        else image.to(torch.float32) * control
    ).to(torch.float16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("layout", "shape", "numpy_permutation"),
    [
        ("CHW", (3, 5, 7), (1, 2, 0)),
        ("NCHW", (2, 3, 5, 7), (0, 2, 3, 1)),
        ("CDHW", (3, 5, 7, 9), (1, 2, 3, 0)),
        ("NCDHW", (2, 3, 5, 7, 9), (0, 2, 3, 4, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.float32])
def test_common_pointwise_tensor_routers_preserve_layout_dtype_and_values(
    layout: str,
    shape: tuple[int, ...],
    numpy_permutation: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    tensor = torch.linspace(0.1, 0.9, steps=int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(dtype)
    channel_last = tensor.permute(numpy_permutation).numpy()
    channel_values = np.linspace(0.125, 0.375, num=channel_last.shape[-1], dtype=np.float32)
    expected = (channel_last.astype(np.float32) + channel_values) * np.float32(1.125) + np.float32(0.125)
    if dtype == torch.uint8:
        image_u8 = torch.arange(int(np.prod(shape)), dtype=torch.uint8).reshape(shape)
        channel_last = image_u8.permute(numpy_permutation).numpy()
        added = np.clip(channel_last.astype(np.int16) + np.trunc(channel_values).astype(np.int16), 0, 255).astype(
            np.uint8,
        )
        lut = np.clip(np.arange(256, dtype=np.float32) * 1.125 + 0.125, 0, 255).astype(np.uint8)
        expected = lut[added]
        result = multiply_add(add(image_u8, channel_values, layout=layout), 1.125, 0.125, layout=layout)
    else:
        result = multiply_add(add(tensor, channel_values, layout=layout), 1.125, 0.125, layout=layout)
        expected = expected.astype(np.float16 if dtype == torch.float16 else np.float32)

    expected_shape = tensor.shape if dtype != torch.uint8 else image_u8.shape
    assert result.shape == expected_shape
    assert result.dtype == dtype
    result_last = result.permute(numpy_permutation).numpy()
    if dtype == torch.float16:
        np.testing.assert_allclose(result_last, expected, rtol=1e-3, atol=1e-3)
    else:
        np.testing.assert_allclose(result_last, expected, rtol=1e-6, atol=1e-6)
