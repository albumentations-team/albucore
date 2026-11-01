import numpy as np
import pytest
import torch

from albucore import normalize_per_image

MODES = ("image", "image_per_channel", "min_max", "min_max_per_channel")
SPECIFICATIONS = [
    ((4, 6, 1), "CHW", (2, 0, 1)),
    ((4, 6, 3), "CHW", (2, 0, 1)),
    ((4, 6, 9), "CHW", (2, 0, 1)),
    ((2, 4, 6, 3), "NCHW", (0, 3, 1, 2)),
    ((2, 4, 6, 3), "CDHW", (3, 0, 1, 2)),
    ((2, 3, 4, 6, 3), "NCDHW", (0, 4, 1, 2, 3)),
]


def reference(image, mode, dtype):
    wide = image.astype(np.float64)
    axes = tuple(range(wide.ndim - 1)) if mode.endswith("per_channel") else None
    if mode in ("image", "image_per_channel"):
        result = (wide - wide.mean(axis=axes)) / (wide.std(axis=axes) + 1e-4)
        return np.clip(result, -20, 20)
    lower, upper = wide.min(axis=axes), wide.max(axis=axes)
    epsilon = 1e-4 if mode == "min_max_per_channel" and (dtype == np.uint8 or image.shape[-1] > 1) else 0
    divisor = upper - lower + epsilon
    return np.divide(wide - lower, divisor, out=np.zeros_like(wide), where=divisor != 0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("specification", SPECIFICATIONS)
@pytest.mark.parametrize("container", ("numpy", "tensor"))
@pytest.mark.parametrize(
    ("dtype", "output_mode"),
    [(dtype, output) for dtype in (np.uint8, np.float16, np.float32) for output in ("allocate", "output")]
    + [(np.float16, "alias"), (np.float32, "alias")],
)
def test_adaptive_output_contract(mode, specification, container, dtype, output_mode):
    shape, layout, permutation = specification
    base = np.arange(np.prod(shape[:-1])).reshape((*shape[:-1], 1)) % 16
    values = base * np.resize([1, 2, 4, 8], shape[-1]) + np.arange(shape[-1]) * 7
    if layout in ("NCHW", "NCDHW"):
        values = values + np.arange(shape[0]).reshape((shape[0], *(1 for _ in range(len(shape) - 1)))) * 17
    array = values.astype(dtype)
    expected = (
        np.stack([reference(item, mode, dtype) for item in array])
        if container == "tensor" and layout in ("NCHW", "NCDHW")
        else reference(array, mode, dtype)
    )
    image = torch.from_numpy(array).permute(permutation).contiguous() if container == "tensor" else array
    original = image.clone() if container == "tensor" else image.copy()
    result_dtype = np.float16 if dtype == np.float16 else np.float32
    output = image if output_mode == "alias" else None
    if output_mode == "output":
        output = (
            torch.empty_like(image, dtype=torch.float16 if dtype == np.float16 else torch.float32)
            if container == "tensor"
            else np.empty_like(image, dtype=result_dtype)
        )

    result = normalize_per_image(image, mode, out=output, layout=layout if container == "tensor" else None)

    actual = result.permute(tuple(np.argsort(permutation))).numpy() if container == "tensor" else result
    assert actual.dtype == result_dtype
    assert actual.shape == array.shape
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(actual, expected.astype(np.float16), maxulp=1)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    if output is not None:
        assert result is output
    if output_mode != "alias":
        np.testing.assert_array_equal(
            image.numpy() if container == "tensor" else image, original.numpy() if container == "tensor" else original
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", (np.uint8, np.float16, np.float32))
@pytest.mark.parametrize("channels", (1, 3, 9))
def test_adaptive_readonly_input_and_gapped_output(mode, dtype, channels):
    image = (np.arange(4 * 6 * channels).reshape(4, 6, channels) % 32).astype(dtype)[:, ::-1]
    original = image.copy()
    image.flags.writeable = False
    storage = np.full((4, 12, channels), 137, dtype=np.float16 if dtype == np.float16 else np.float32)
    output = storage[:, ::2][:, ::-1]

    result = normalize_per_image(image, mode, out=output)

    assert result is output
    expected = reference(image, mode, dtype)
    if dtype == np.float16:
        np.testing.assert_array_max_ulp(result, expected.astype(np.float16), maxulp=1)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(storage[:, 1::2], 137)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", (np.uint8, np.float16, np.float32))
def test_adaptive_constant_input(mode, dtype):
    image = np.full((5, 7, 3), 2 if dtype == np.uint8 else 0.5, dtype=dtype)
    result = normalize_per_image(image, mode)
    np.testing.assert_array_equal(result, np.zeros_like(image, dtype=np.float16 if dtype == np.float16 else np.float32))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", (np.uint8, np.float16, np.float32))
def test_adaptive_empty_batch(mode, dtype):
    image = np.empty((0, 5, 7, 3), dtype=dtype)
    output = np.empty_like(image, dtype=np.float16 if dtype == np.float16 else np.float32)
    assert normalize_per_image(image, mode, out=output) is output
    result = normalize_per_image(image, mode)
    assert result.shape == image.shape
    assert result.dtype == output.dtype


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", (np.float16, np.float32))
@pytest.mark.parametrize("supplied", (False, True))
def test_adaptive_non_native_empty_batch(mode, dtype, supplied):
    image = np.empty((0, 5, 7, 3), dtype=np.dtype(dtype).newbyteorder("S"))
    output = np.empty_like(image) if supplied else None

    result = normalize_per_image(image, mode, out=output)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert result is (output if supplied else image)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
@pytest.mark.parametrize("layout,shape", [("NCHW", (2, 3, 5, 7)), ("NCDHW", (2, 3, 4, 5, 7))])
def test_adaptive_tensor_batch_gapped_output(mode, dtype, layout, shape):
    image = (torch.arange(np.prod(shape)).reshape(shape) % 23).to(dtype) / 32
    original = image.clone()
    storage = torch.full((*shape[:-1], shape[-1] * 2), 137, dtype=dtype)
    output = storage[..., ::2]
    expected = normalize_per_image(image, mode, layout=layout)

    result = normalize_per_image(image, mode, layout=layout, out=output)

    assert result is output
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)
    torch.testing.assert_close(storage[..., 1::2], torch.full_like(output, 137), rtol=0, atol=0)
