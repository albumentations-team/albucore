import importlib

import numpy as np
import pytest
import torch

import albucore.stats as stats_mod
from albucore import normalize_per_image

normalize_mod = importlib.import_module("albucore.normalize")

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


@pytest.mark.parametrize("pattern", ["random", "constant", "near_constant"])
@pytest.mark.parametrize("with_output", (False, True))
def test_float32_image_normalization_uses_stable_global_stats(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    with_output: bool,
) -> None:
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", True)
    monkeypatch.setattr(stats_mod, "_MACOS_ARM64", False)
    rng = np.random.default_rng(9917)
    if pattern == "random":
        image = rng.random((128, 160, 3), dtype=np.float32)
        expect_cv2 = True
    elif pattern == "constant":
        image = np.full((128, 160, 3), np.float32(0.5), dtype=np.float32)
        expect_cv2 = True
    else:
        image = np.full((128, 160, 3), np.float32(1e6), dtype=np.float32)
        image[::3] = np.nextafter(np.float32(1e6), np.float32(np.inf))
        expect_cv2 = False
    source = image.copy()
    image.flags.writeable = False
    stats = stats_mod._mean_std_global_float32_opencv(image, 1e-4)  # noqa: SLF001
    expected = np.empty_like(image)
    np.subtract(image, image.mean(), out=expected)
    np.divide(expected, image.std() + 1e-4, out=expected)
    np.clip(expected, -20, 20, out=expected)
    output = np.empty_like(image) if with_output else None

    result = normalize_per_image(image, "image", out=output)

    if (stats is not None) != expect_cv2:
        pytest.fail("float32 image normalization chose the wrong statistics route")
    if with_output:
        if result is not output:
            pytest.fail("float32 image normalization did not return the supplied output")
    np.testing.assert_allclose(result, expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_array_equal(image, source)


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


@pytest.mark.parametrize(
    "layout,shape,item_layout",
    [("NCHW", (4, 3, 7, 9), "CHW"), ("NCDHW", (4, 3, 5, 7, 9), "CDHW")],
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("output_mode", ("allocate", "output", "alias"))
def test_adaptive_tensor_batch_float32_vectorizes_global_statistics(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    shape: tuple[int, ...],
    item_layout: str,
    mode: str,
    output_mode: str,
) -> None:
    monkeypatch.setattr(normalize_mod, "_TENSOR_BATCH_FLOAT32_MIN_ELEMENTS", 0)
    image = torch.from_numpy(np.random.default_rng(251121).random(shape, dtype=np.float32))
    original = image.clone()
    expected = torch.stack([normalize_per_image(item, mode, layout=item_layout) for item in image])
    output = image if output_mode == "alias" else torch.empty_like(image) if output_mode == "output" else None
    route_calls = 0
    original_route = normalize_mod._adaptive_tensor_batch_float32  # noqa: SLF001
    should_route = normalize_mod._use_tensor_batch_float32_route(image, mode, layout, output)  # noqa: SLF001

    def count_route(
        img: torch.Tensor,
        normalization: str,
        out: torch.Tensor | None,
    ) -> torch.Tensor | None:
        nonlocal route_calls
        route_calls += 1
        return original_route(img, normalization, out)

    monkeypatch.setattr(normalize_mod, "_adaptive_tensor_batch_float32", count_route)

    result = normalize_per_image(image, mode, layout=layout, out=output)

    if route_calls != int(should_route):
        pytest.fail("float32 Tensor batch selected the wrong statistics route")
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    if output is not None and result is not output:
        pytest.fail("normalization did not return the supplied output")
    if output_mode != "alias":
        torch.testing.assert_close(image, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode,pattern",
    [("image", "nan"), ("image_per_channel", "near_constant"), ("min_max", "nan"), ("min_max_per_channel", "inf")],
)
def test_adaptive_tensor_batch_float32_statistics_fall_back_for_unstable_data(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    pattern: str,
) -> None:
    monkeypatch.setattr(normalize_mod, "_TENSOR_BATCH_FLOAT32_MIN_ELEMENTS", 0)
    image = np.full((4, 3, 7, 9), np.float32(0.5), dtype=np.float32)
    if pattern == "near_constant":
        image.fill(np.float32(1e6))
        image[:, :, ::3, :] = np.nextafter(np.float32(1e6), np.float32(np.inf))
    elif pattern == "nan":
        image[0, 0, 1, 2] = np.nan
    else:
        image[0, 0, 1, 2] = np.inf
    tensor = torch.from_numpy(image)
    expected = torch.stack([normalize_per_image(item, mode, layout="CHW") for item in tensor])
    route_results: list[torch.Tensor | None] = []
    original_route = normalize_mod._adaptive_tensor_batch_float32  # noqa: SLF001

    def record_route(
        batch: torch.Tensor,
        normalization: str,
        out: torch.Tensor | None,
    ) -> torch.Tensor | None:
        result = original_route(batch, normalization, out)
        route_results.append(result)
        return result

    monkeypatch.setattr(normalize_mod, "_adaptive_tensor_batch_float32", record_route)

    result = normalize_per_image(tensor, mode, layout="NCHW")

    if len(route_results) != 1 or route_results[0] is not None:
        pytest.fail("unstable batch statistics did not fall back to the per-item route")
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=0, atol=0, equal_nan=True)


def test_adaptive_tensor_batch_float32_small_input_keeps_the_per_item_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = torch.from_numpy(np.random.default_rng(251121).random((4, 3, 7, 9), dtype=np.float32))
    expected = torch.stack([normalize_per_image(item, "image", layout="CHW") for item in image])
    route_calls = 0
    original_route = normalize_mod._adaptive_tensor_batch_float32  # noqa: SLF001

    def count_route(
        batch: torch.Tensor,
        normalization: str,
        out: torch.Tensor | None,
    ) -> torch.Tensor | None:
        nonlocal route_calls
        route_calls += 1
        return original_route(batch, normalization, out)

    monkeypatch.setattr(normalize_mod, "_adaptive_tensor_batch_float32", count_route)

    result = normalize_per_image(image, "image", layout="NCHW")

    if route_calls != 0:
        pytest.fail("small Tensor batches should keep the established per-item route")
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "layout,shape,mode,expected",
    [
        ("NCHW", (4, 3, 480, 640), "image", False),
        ("NCHW", (4, 3, 480, 640), "image_per_channel", True),
        ("NCHW", (4, 3, 480, 640), "min_max", True),
        ("NCHW", (4, 3, 480, 640), "min_max_per_channel", True),
        ("NCDHW", (4, 3, 32, 128, 160), "image", False),
        ("NCDHW", (4, 3, 32, 128, 160), "image_per_channel", False),
        ("NCDHW", (4, 3, 32, 128, 160), "min_max", True),
        ("NCDHW", (4, 3, 32, 128, 160), "min_max_per_channel", True),
        ("NCHW", (4, 1, 128, 160), "image", False),
        ("NCHW", (4, 1, 128, 160), "image_per_channel", True),
        ("NCHW", (4, 9, 128, 160), "image", True),
        ("NCHW", (4, 9, 128, 160), "image_per_channel", False),
    ],
)
def test_adaptive_tensor_batch_float32_route_selects_measured_regions(
    layout: str,
    shape: tuple[int, ...],
    mode: str,
    expected: bool,
) -> None:
    image = torch.empty(shape, dtype=torch.float32)
    if layout == "NCDHW" and mode == "image_per_channel":
        expected = normalize_mod._MACOS_ARM64  # noqa: SLF001

    selected = normalize_mod._use_tensor_batch_float32_route(image, mode, layout, None)  # noqa: SLF001

    if selected is not expected:
        pytest.fail(f"unexpected {layout} {mode} route decision for shape {shape}")
