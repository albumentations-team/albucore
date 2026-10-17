import numpy as np
import pytest
import torch

from albucore import normalize


@pytest.mark.parametrize(
    ("dtype", "output_mode"),
    [(dtype, mode) for dtype in (np.uint8, np.float16, np.float32) for mode in ("allocate", "output")]
    + [(np.float16, "alias"), (np.float32, "alias")],
)
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize(
    "specification",
    [
        ((7, 11, 3), "CHW", (2, 0, 1)),
        ((2, 7, 11, 3), "NCHW", (0, 3, 1, 2)),
        ((2, 7, 11, 3), "CDHW", (3, 0, 1, 2)),
        ((2, 3, 7, 11, 3), "NCDHW", (0, 4, 1, 2, 3)),
    ],
)
@pytest.mark.parametrize("per_channel", [False, True])
def test_normalize_output_contract(dtype, output_mode, container, specification, per_channel):
    shape, layout, permutation = specification
    image_array = (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) % 32).astype(dtype)
    mean = np.array([1, 2, 3], dtype=np.float32) if per_channel else 2.0
    denominator = np.array([0.5, 2, 4], dtype=np.float32) if per_channel else 0.5
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    expected = ((image_array.astype(np.float32) - mean) * denominator).astype(output_dtype)
    image = torch.from_numpy(image_array).permute(permutation).contiguous() if container == "tensor" else image_array
    original = image.clone() if container == "tensor" else image.copy()
    output = image if output_mode == "alias" else None
    if output_mode == "output":
        output = (
            torch.empty_like(image, dtype=torch.float16 if dtype == np.float16 else torch.float32)
            if container == "tensor"
            else np.empty_like(image, dtype=output_dtype)
        )

    result = normalize(image, mean, denominator, out=output, layout=layout if container == "tensor" else None)

    actual = result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result
    assert actual.dtype == output_dtype
    assert actual.shape == shape
    np.testing.assert_array_equal(actual, expected)
    if output is not None:
        assert result is output
    if output_mode != "alias":
        np.testing.assert_array_equal(
            image.numpy() if container == "tensor" else image,
            original.numpy() if container == "tensor" else original,
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("negative", [False, True])
def test_normalize_readonly_input_and_strided_output(dtype, negative):
    image = np.arange(66, dtype=dtype).reshape(2, 11, 3)[:, ::-1]
    original = image.copy()
    image.flags.writeable = False
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    storage = np.full((2, 22, 3), 137, dtype=output_dtype)
    output = storage[:, ::2]
    if negative:
        output = output[:, ::-1]
    expected = (image.astype(np.float32) - 2) * 0.5

    result = normalize(image, 2.0, 0.5, out=output)

    assert result is output
    np.testing.assert_array_equal(result, expected.astype(output_dtype))
    np.testing.assert_array_equal(storage[:, 1::2], 137)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("channel_controls", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_normalize_near_identity_retains_bits_and_copies_to_output(dtype, container, channel_controls, alias):
    array = np.array([-0.0, 0.0, 1.0, np.nan], dtype=dtype).reshape(1, 4, 1)
    image = torch.from_numpy(array).permute(2, 0, 1) if container == "tensor" else array
    destination = image if alias else torch.empty_like(image) if container == "tensor" else np.empty_like(image)
    kwargs = {"layout": "CHW"} if container == "tensor" else {}

    mean = np.array([9e-6], dtype=np.float64) if channel_controls else 9e-6
    denominator = np.array([1.0], dtype=np.float64) if channel_controls else 1.0
    result = normalize(image, mean, denominator, **kwargs)
    copied = normalize(image, mean, denominator, out=destination, **kwargs)

    assert result is image
    assert copied is destination
    actual = copied.permute(1, 2, 0).numpy() if container == "tensor" else copied
    np.testing.assert_array_equal(
        actual.view(np.uint16 if dtype == np.float16 else np.uint32),
        array.view(np.uint16 if dtype == np.float16 else np.uint32),
    )


@pytest.mark.parametrize("mode", ["allocate", "alias", "negative"])
def test_normalize_strided_output_keeps_selected_fused_float32_kernel(mode):
    image = np.full((480, 640, 3), 3e38, dtype=np.float32)
    output = image if mode == "alias" else np.empty_like(image)[:, ::-1] if mode == "negative" else None

    result = normalize(image, 1e38, 1.5, out=output)

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, np.full_like(image, 3e38), rtol=1e-6)
    if output is not None:
        assert result is output


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 9])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("output_mode", ["allocate", "output", "strided"])
def test_normalize_large_output_routes(dtype, channels, container, output_mode):
    rng = np.random.default_rng(137)
    array = (rng.random((128, 160, channels), dtype=np.float32) * 32).astype(dtype)
    mean = np.resize(np.array([0.125, 0.25, 0.5], dtype=np.float32), channels)
    denominator = np.resize(np.array([0.5, 1.5, 2], dtype=np.float32), channels)
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    expected = ((array.astype(np.float32) - mean) * denominator).astype(output_dtype)
    image = torch.from_numpy(array).permute(2, 0, 1).contiguous() if container == "tensor" else array
    original = image.clone() if container == "tensor" else image.copy()
    output = None
    storage = None
    if output_mode != "allocate":
        if container == "tensor":
            shape = (*image.shape[:-1], image.shape[-1] * 2) if output_mode == "strided" else image.shape
            storage = torch.full(shape, 137, dtype=torch.float16 if dtype == np.float16 else torch.float32)
            output = storage[..., ::2] if output_mode == "strided" else storage
        else:
            shape = (128, 320, channels) if output_mode == "strided" else image.shape
            storage = np.full(shape, 137, dtype=output_dtype)
            output = storage[:, ::2] if output_mode == "strided" else storage

    result = normalize(image, mean, denominator, out=output, layout="CHW" if container == "tensor" else None)

    actual = result.permute(1, 2, 0).numpy() if container == "tensor" else result
    assert actual.dtype == output_dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    if output is not None:
        assert result is output
    if output_mode == "strided":
        untouched = storage[..., 1::2].numpy() if container == "tensor" else storage[:, 1::2]
        np.testing.assert_array_equal(untouched, 137)
    np.testing.assert_array_equal(
        image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("with_output", [False, True])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_normalize_empty_batch(dtype, with_output, container):
    array = np.empty((0, 7, 11, 3), dtype=dtype)
    image = torch.from_numpy(array).permute(0, 3, 1, 2) if container == "tensor" else array
    output_dtype = np.float16 if dtype == np.float16 else np.float32
    output_array = np.empty_like(array, dtype=output_dtype)
    output = torch.from_numpy(output_array).permute(0, 3, 1, 2) if container == "tensor" else output_array

    result = normalize(
        image, 2.0, 0.5, out=output if with_output else None, layout="NCHW" if container == "tensor" else None
    )

    actual = result.numpy() if container == "tensor" else result
    assert actual.size == 0
    assert actual.shape == tuple(image.shape)
    assert actual.dtype == output_dtype
    if with_output:
        assert result is output


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize(
    ("mean", "denominator", "expected_channels"),
    [([0, 0, 0.25], [1, 1, 1], [0.5, 0.5, 0.25]), ([0, 0, 0], [1, 1, 2], [0.5, 0.5, 1.0])],
)
def test_normalize_identity_checks_all_channels(dtype, container, mean, denominator, expected_channels):
    array = np.full((7, 11, 3), 0.5, dtype=dtype)
    image = torch.from_numpy(array).permute(2, 0, 1) if container == "tensor" else array

    result = normalize(
        image,
        np.array(mean, dtype=np.float64),
        np.array(denominator, dtype=np.float64),
        layout="CHW" if container == "tensor" else None,
    )

    actual = result.permute(1, 2, 0).numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, np.broadcast_to(np.array(expected_channels, dtype=dtype), array.shape))
    np.testing.assert_array_equal(array, 0.5)
