import numpy as np
import pytest
import torch

from albucore import clip


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_clip_defaults_to_input_dtype(container, dtype):
    values = [0, 1, 127, 255] if dtype == np.uint8 else [-0.25, 0.125, 0.875, 1.25]
    expected = values if dtype == np.uint8 else [0, 0.125, 0.875, 1]
    array = np.array(values, dtype=dtype).reshape(2, 2, 1)
    image = torch.from_numpy(array.copy()).permute(2, 0, 1) if container == "tensor" else array.copy()

    result = clip(image)

    actual = result.permute(1, 2, 0).numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, np.array(expected, dtype=dtype).reshape(array.shape))
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(image.permute(1, 2, 0).numpy() if container == "tensor" else image, array)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_clip_defaults_to_destination_dtype(container, dtype):
    array = np.array([-2.5, 0.5, 128.5, 300], dtype=np.float32).reshape(2, 2, 1)
    output = np.empty(array.shape, dtype=dtype)
    image = torch.from_numpy(array).permute(2, 0, 1) if container == "tensor" else array
    destination = torch.from_numpy(output).permute(2, 0, 1) if container == "tensor" else output

    result = clip(image, out=destination)

    expected = [0, 0, 128, 255] if dtype == np.uint8 else [0, 0.5, 1, 1]
    assert result is destination
    np.testing.assert_array_equal(output, np.array(expected, dtype=dtype).reshape(array.shape))
    np.testing.assert_array_equal(array, np.array([-2.5, 0.5, 128.5, 300], dtype=np.float32).reshape(array.shape))


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("storage", ["allocate", "output", "alias"])
def test_clip_inferred_dtype_preserves_numpy_byte_order(dtype, storage):
    dtype = np.dtype(dtype).newbyteorder("S")
    image = np.array([-0.25, -0.0, 0.5, 1.25, np.nan, np.inf], dtype=dtype).reshape(1, 6, 1)
    original = image.copy()
    expected = np.array([0, -0.0, 0.5, 1, np.nan, 1], dtype=dtype).reshape(image.shape)
    destination = image if storage == "alias" else np.empty_like(image) if storage == "output" else None

    result = clip(image, out=destination)

    assert result.dtype == dtype
    np.testing.assert_array_equal(result.view(np.uint8), expected.view(np.uint8))
    if destination is not None:
        assert result is destination
    if storage != "alias":
        np.testing.assert_array_equal(image.view(np.uint8), original.view(np.uint8))


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("source_dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("target_dtype", [np.uint8, np.float16, np.float32])
@pytest.mark.parametrize("provided_output", [False, True])
def test_clip_output_preserves_conversion_and_input(container, source_dtype, target_dtype, provided_output):
    values = [0, 1, 15, 128, 254, 255] if source_dtype == np.uint8 else [-0.5, -0.0, 0.5, 1.0, 1.5, 255.5]
    array = np.array(values, dtype=source_dtype).reshape(2, 3, 1)
    maximum = 255 if target_dtype == np.uint8 else 1
    expected = np.clip(array, 0, maximum).astype(target_dtype)
    image = torch.from_numpy(array).permute(2, 0, 1) if container == "tensor" else array
    original = image.clone() if container == "tensor" else image.copy()
    if container == "tensor":
        output_dtype = torch.from_numpy(np.empty((), dtype=target_dtype)).dtype
        destination = torch.empty_like(image, dtype=output_dtype) if provided_output else None
    else:
        destination = np.empty_like(image, dtype=target_dtype) if provided_output else None

    result = clip(image, target_dtype, out=destination)

    if container == "tensor":
        torch.testing.assert_close(result, torch.from_numpy(expected).permute(2, 0, 1), rtol=0, atol=0)
        torch.testing.assert_close(image, original, rtol=0, atol=0)
    else:
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(image, original)
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32])
def test_clip_mutates_only_the_explicit_input_output(container, dtype):
    values = [0, 1, 255] if dtype == np.uint8 else [-0.5, 0.5, 1.5]
    array = np.array(values, dtype=dtype).reshape(1, 3, 1)
    expected = np.clip(array, 0, 255 if dtype == np.uint8 else 1)
    image = torch.from_numpy(array.copy()) if container == "tensor" else array.copy()

    result = clip(image, dtype, out=image)

    assert result is image
    actual = result.numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_clip_writes_strided_output_without_touching_gaps(container):
    array = np.linspace(-1, 2, 15, dtype=np.float32).reshape(1, 3, 5)
    storage = np.full((1, 3, 10), 137, dtype=np.float32)
    image = torch.from_numpy(array) if container == "tensor" else array
    backing = torch.from_numpy(storage) if container == "tensor" else storage
    destination = backing[..., ::2]

    result = clip(image, np.float32, out=destination)

    assert result is destination
    actual = result.numpy() if container == "tensor" else result
    np.testing.assert_array_equal(actual, np.clip(array, 0, 1))
    np.testing.assert_array_equal(storage[..., 1::2], np.full((1, 3, 5), 137, dtype=np.float32))
    np.testing.assert_array_equal(array, np.linspace(-1, 2, 15, dtype=np.float32).reshape(1, 3, 5))


def test_clip_accepts_readonly_numpy_source_with_separate_output():
    image = np.array([-0.5, 0.5, 1.5], dtype=np.float32).reshape(1, 3, 1)
    original = image.copy()
    image.flags.writeable = False
    destination = np.empty_like(image)

    result = clip(image, np.float32, out=destination)

    assert result is destination
    np.testing.assert_array_equal(result, np.array([0, 0.5, 1], dtype=np.float32).reshape(1, 3, 1))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("container", ["numpy", "tensor"])
@pytest.mark.parametrize("target_dtype", [np.uint8, np.float16, np.float32])
def test_clip_output_matches_allocating_semantics_for_all_half_encodings(container, target_dtype):
    array = np.arange(1 << 16, dtype=np.uint16).view(np.float16).reshape(256, 256, 1)
    maximum = 255 if target_dtype == np.uint8 else 1
    if container == "tensor":
        image = torch.from_numpy(array.copy())
        output_dtype = torch.from_numpy(np.empty((), dtype=target_dtype)).dtype
        expected = image.clamp(0, maximum).to(dtype=output_dtype).numpy()
        destination = torch.empty_like(image, dtype=output_dtype)
        actual = clip(image, target_dtype, out=destination).numpy()
    else:
        with np.errstate(invalid="ignore"):
            expected = np.clip(array, 0, maximum).astype(target_dtype)
            destination = np.empty(array.shape, dtype=target_dtype)
            actual = clip(array, target_dtype, out=destination)

    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("storage", ["allocate", "alias", "output", "strided_output", "readonly", "strided_input"])
def test_clip_preserves_every_finite_half_encoding(signed, storage):
    encodings = np.arange(1 << 16, dtype=np.uint16)
    selected = ((encodings & 0x7FFF) <= 0x7C00) & (encodings != 0x8000) if signed else encodings <= 0x7C00
    image = encodings[selected].view(np.float16).reshape(1, -1, 1)
    if storage == "strided_input":
        image = np.repeat(image, 2, axis=1)[:, ::2]
    elif storage == "readonly":
        image.flags.writeable = False
    original = image.view(np.uint16).copy()
    expected = np.clip(image, 0, 1)
    backing = None
    if storage == "strided_output":
        backing = np.full((1, image.shape[1], 2), 137, dtype=np.float16)
        destination = backing[..., :1]
    elif storage == "alias":
        destination = image
    elif storage in ("output", "readonly", "strided_input"):
        destination = np.empty_like(image)
    else:
        destination = None

    with np.errstate(all="raise"):
        result = clip(image, np.float16, out=destination)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16))
    if destination is not None:
        assert result is destination
    if storage != "alias":
        np.testing.assert_array_equal(image.view(np.uint16), original)
    if backing is not None:
        np.testing.assert_array_equal(backing[..., 1], 137)
