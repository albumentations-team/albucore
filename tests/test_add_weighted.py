import numpy as np
import pytest
import torch

from albucore import add, add_weighted, weighted
from albucore._types import ImageType


def test_add_weighted_does_not_clip_float32_result_to_unit_range():
    img1 = np.full((1, 1, 1), 128, dtype=np.float32)
    img2 = np.full((1, 1, 1), 64, dtype=np.float32)

    result = add_weighted(img1, 0.5, img2, 0.5)

    np.testing.assert_array_equal(result, np.full_like(img1, 96))


@pytest.mark.parametrize(("weight1", "weight2"), [(2.0, 3.0), (-2.0, 3.0)])
def test_add_weighted_preserves_float32_values_outside_image_range(weight1: float, weight2: float):
    img1 = np.full((1, 1, 1), 128, dtype=np.float32)
    img2 = np.full((1, 1, 1), 64, dtype=np.float32)
    expected = img1 * weight1 + img2 * weight2

    result = add_weighted(img1, weight1, img2, weight2)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(9, 11, 1), (9, 11, 3), (9, 11, 9), (2, 4, 9, 11, 3)])
@pytest.mark.parametrize(("weight1", "weight2"), [(0.4, 0.6), (1.25, -0.25)])
def test_add_weighted_float16_matches_wide_result_within_one_ulp(
    shape: tuple[int, ...],
    weight1: float,
    weight2: float,
) -> None:
    rng = np.random.default_rng(15)
    img1 = rng.uniform(-1, 1, size=shape).astype(np.float16)
    img2 = rng.uniform(-1, 1, size=shape).astype(np.float16)
    original_img1 = img1.copy()
    original_img2 = img2.copy()
    expected = (img1.astype(np.float32) * weight1 + img2.astype(np.float32) * weight2).astype(np.float16)

    result = add_weighted(img1, weight1, img2, weight2)

    assert result.dtype == np.float16
    assert result.shape == shape
    np.testing.assert_array_max_ulp(result, expected, maxulp=1)
    np.testing.assert_array_equal(img1, original_img1)
    np.testing.assert_array_equal(img2, original_img2)


def test_add_weighted_float16_preserves_wide_accumulation_on_large_cancellation() -> None:
    large_positive = np.array([0x7BF1], dtype=np.uint16).view(np.float16)[0]
    cancelling_negative = np.array([0xF94B], dtype=np.uint16).view(np.float16)[0]
    img1 = np.full((9, 11, 3), large_positive, dtype=np.float16)
    img2 = np.full_like(img1, cancelling_negative)
    expected = (img1.astype(np.float32) * np.float32(0.4) + img2.astype(np.float32) * np.float32(0.6)).astype(
        np.float16,
    )

    result = add_weighted(img1, 0.4, img2, 0.6)

    np.testing.assert_array_max_ulp(result, expected, maxulp=1)


def test_add_weighted_float16_numkong_half_route_matches_binary16_domain(monkeypatch) -> None:
    monkeypatch.setattr(weighted, "_NUMKONG_NEON_HALF", True)
    calls: list[tuple[float, float]] = []
    original_numkong = weighted._blend_numkong

    def record_numkong(
        img1: ImageType, weight1: float, img2: ImageType, weight2: float, *, out: ImageType | None = None
    ) -> ImageType:
        calls.append((weight1, weight2))
        return original_numkong(img1, weight1, img2, weight2, out=out)

    monkeypatch.setattr(weighted, "_blend_numkong", record_numkong)
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    non_nan_bits = bits[~((exponent == 0x7C00) & (mantissa != 0))]
    img1 = np.resize(non_nan_bits, (256, 256, 1)).view(np.float16)
    img2 = np.ascontiguousarray(np.roll(img1, 12345, axis=1))
    expected = (img1.astype(np.float32) * np.float32(0.5) + img2.astype(np.float32) * np.float32(0.5)).astype(
        np.float16
    )

    result = add_weighted(img1, 0.5, img2, 0.5)

    assert calls == [(0.5, 0.5)]
    assert result.shape == img1.shape
    assert result.dtype == np.float16
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_array_equal(np.signbit(result[np.isinf(result)]), np.signbit(expected[np.isinf(expected)]))
    finite = np.isfinite(result) & np.isfinite(expected)
    np.testing.assert_array_max_ulp(result[finite], expected[finite], maxulp=1)
    assert not np.shares_memory(result, img1)
    assert not np.shares_memory(result, img2)


def test_add_weighted_float16_numkong_route_skips_other_weights(monkeypatch) -> None:
    monkeypatch.setattr(weighted, "_NUMKONG_NEON_HALF", True)

    def fail_numkong(*args: object, **kwargs: object) -> np.ndarray:
        msg = "NumKong float16 blend is enabled only for the measured half-exact 0.5/0.5 weights."
        raise AssertionError(msg)

    monkeypatch.setattr(weighted, "_blend_numkong", fail_numkong)
    rng = np.random.default_rng(910_014)
    img1 = rng.uniform(-1, 1, (240, 320, 3)).astype(np.float16)
    img2 = rng.uniform(-1, 1, (240, 320, 3)).astype(np.float16)
    expected = (img1.astype(np.float32) * np.float32(0.4) + img2.astype(np.float32) * np.float32(0.6)).astype(
        np.float16,
    )

    result = add_weighted(img1, 0.4, img2, 0.6)

    np.testing.assert_array_max_ulp(result, expected, maxulp=1)


@pytest.mark.parametrize("layout", ["CHW", "NCHW"])
@pytest.mark.parametrize("channels", [3, 9])
@pytest.mark.parametrize("output_mode", ["allocate", "first", "second", "output"])
def test_add_weighted_float16_numkong_tensor_route_preserves_layout(
    monkeypatch,
    layout: str,
    channels: int,
    output_mode: str,
) -> None:
    monkeypatch.setattr(weighted, "_NUMKONG_NEON_HALF", True)
    calls: list[tuple[float, float]] = []
    original_numkong = weighted._blend_numkong

    def record_numkong(
        img1: ImageType, weight1: float, img2: ImageType, weight2: float, *, out: ImageType | None = None
    ) -> ImageType:
        calls.append((weight1, weight2))
        return original_numkong(img1, weight1, img2, weight2, out=out)

    monkeypatch.setattr(weighted, "_blend_numkong", record_numkong)
    batch = 2 if layout == "NCHW" else 1
    rng = np.random.default_rng(910_011 + channels)
    img1 = rng.uniform(-1, 1, (batch, 240, 320, channels)).astype(np.float32).astype(np.float16)
    img2 = (img1.astype(np.float32) * np.float32(0.7)).astype(np.float16)
    if layout == "CHW":
        tensor1 = torch.from_numpy(img1[0]).permute(2, 0, 1).contiguous()
        tensor2 = torch.from_numpy(img2[0]).permute(2, 0, 1).contiguous()
    else:
        tensor1 = torch.from_numpy(img1).permute(0, 3, 1, 2).contiguous()
        tensor2 = torch.from_numpy(img2).permute(0, 3, 1, 2).contiguous()
    expected = add_weighted(tensor1.to(torch.float32), 0.5, tensor2.to(torch.float32), 0.5, layout=layout).to(
        torch.float16,
    )
    before1, before2 = tensor1.clone(), tensor2.clone()

    output = {"allocate": None, "first": tensor1, "second": tensor2}.get(output_mode)
    if output_mode == "output":
        output = torch.empty_like(tensor1)
    result = add_weighted(tensor1, 0.5, tensor2, 0.5, layout=layout, out=output)

    assert calls == [(0.5, 0.5)]
    assert result.shape == tensor1.shape
    assert result.dtype == torch.float16
    if output is not None:
        assert result is output
    if output_mode != "first":
        assert result.data_ptr() != tensor1.data_ptr()
        np.testing.assert_array_equal(tensor1.numpy(), before1.numpy())
    if output_mode != "second":
        assert result.data_ptr() != tensor2.data_ptr()
        np.testing.assert_array_equal(tensor2.numpy(), before2.numpy())
    np.testing.assert_array_max_ulp(result.numpy(), expected.numpy(), maxulp=1)


def test_add_weighted_float16_numkong_half_route_matches_all_non_nan_patterns(monkeypatch) -> None:
    monkeypatch.setattr(weighted, "_NUMKONG_NEON_HALF", True)
    bits = np.arange(1 << 16, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    non_nan_bits = bits[~((exponent == 0x7C00) & (mantissa != 0))]
    img1 = np.resize(non_nan_bits, (256, 256, 1)).view(np.float16)
    img2 = np.ascontiguousarray(np.roll(img1, 12345, axis=1))
    expected = (img1.astype(np.float32) * np.float32(0.5) + img2.astype(np.float32) * np.float32(0.5)).astype(
        np.float16
    )
    before1, before2 = img1.copy(), img2.copy()

    result = add_weighted(img1, 0.5, img2, 0.5)

    assert result.shape == img1.shape
    assert result.dtype == np.float16
    assert not np.shares_memory(result, img1)
    assert not np.shares_memory(result, img2)
    np.testing.assert_array_equal(img1, before1)
    np.testing.assert_array_equal(img2, before2)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_max_ulp(result[np.isfinite(expected)], expected[np.isfinite(expected)], maxulp=1)


@pytest.mark.parametrize(
    "shape",
    [
        (9, 11, 1),
        (9, 11, 3),
        (9, 11, 5),
        (2, 9, 11, 1),
        (2, 9, 11, 3),
        (2, 9, 11, 5),
    ],
)
@pytest.mark.parametrize("is_contiguous", [True, False])
def test_add_weighted_preserves_raw_float32_range(shape: tuple[int, ...], is_contiguous: bool):
    rng = np.random.default_rng(42)
    storage_width = shape[-2] if is_contiguous else shape[-2] * 2
    storage_shape = (*shape[:-2], storage_width, shape[-1])
    img1 = rng.random(storage_shape, dtype=np.float32) * np.float32(255)
    img2 = rng.random(storage_shape, dtype=np.float32) * np.float32(255)
    if not is_contiguous:
        img1 = img1[..., ::2, :]
        img2 = img2[..., ::2, :]

    original_img1 = img1.copy()
    original_img2 = img2.copy()
    expected = img1 * 0.5 + img2 * 0.5

    result = add_weighted(img1, 0.5, img2, 0.5)

    assert result.shape == img1.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(img1, original_img1)
    np.testing.assert_array_equal(img2, original_img2)


@pytest.mark.parametrize(
    ("shape", "contiguous1", "contiguous2", "expected_backend"),
    [
        ((9, 11, 3), False, False, "opencv"),
        ((2, 9, 11, 3), True, True, "opencv"),
        ((2, 9, 11, 3), True, False, "numkong"),
        ((2, 9, 11, 3), False, True, "numkong"),
        ((9, 11, 1), True, True, "numkong"),
    ],
)
def test_add_weighted_float32_backend_routing(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    contiguous1: bool,
    contiguous2: bool,
    expected_backend: str,
) -> None:
    def make_input(contiguous: bool) -> np.ndarray:
        if contiguous:
            return np.ones(shape, dtype=np.float32)
        storage = np.ones((*shape[:-2], shape[-2] * 2, shape[-1]), dtype=np.float32)
        return storage[..., ::2, :]

    called: list[str] = []

    native_opencv, native_numkong = weighted._blend_opencv, weighted._blend_numkong

    def opencv_spy(img1, weight1, img2, weight2, *, out=None):
        called.append("opencv")
        return native_opencv(img1, weight1, img2, weight2, out=out)

    def numkong_spy(img1, weight1, img2, weight2, *, out=None):
        called.append("numkong")
        return native_numkong(img1, weight1, img2, weight2, out=out)

    monkeypatch.setattr(weighted, "_blend_opencv", opencv_spy)
    monkeypatch.setattr(weighted, "_blend_numkong", numkong_spy)

    result = add_weighted(make_input(contiguous1), 0.5, make_input(contiguous2), 0.5)
    np.testing.assert_array_equal(result, np.ones(shape, dtype=np.float32))

    assert called == [expected_backend]


@pytest.mark.parametrize(
    "img1, weight1, img2, weight2, expected_output",
    [
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            1.0,
            np.array([[6, 8], [10, 12]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            -1.0,
            np.zeros((2, 2)),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.5,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            0.5,
            np.array([[3, 4], [5, 6]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.5,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            -0.5,
            np.zeros((2, 2)),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            0.0,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            0.0,
            np.array([[0, 0], [0, 0]], dtype=np.uint8),
        ),
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            1.0,
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
            0.0,
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
        ),
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            1.0,
            np.array([[0.6, 0.8], [1.0, 1.2]], dtype=np.float32),
        ),
        (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            0.5,
            np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            0.5,
            np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        ),
        (
            np.array([[1, 0.2], [0.3, 0.4]], dtype=np.float32),
            1.0,
            np.array([[0.5, 0.6], [0.7, 0.2]], dtype=np.float32),
            -0.5,
            np.array([[0.75, -0.10000001], [-0.049999982, 0.3]], dtype=np.float32),
        ),
    ],
)
def test_weighted_reference_values(img1, weight1, img2, weight2, expected_output):
    img1, img2, expected_output = (array[..., None] for array in (img1, img2, expected_output))
    before1, before2 = img1.copy(), img2.copy()

    result = add_weighted(img1, weight1, img2, weight2)

    assert result.dtype == img1.dtype
    assert result.shape == img1.shape
    np.testing.assert_array_equal(result, expected_output)
    np.testing.assert_array_equal(img1, before1)
    np.testing.assert_array_equal(img2, before2)


@pytest.mark.parametrize(
    "img_dtype",
    [np.uint8, np.float16, np.float32],
)
@pytest.mark.parametrize(
    "num_channels",
    [1, 3, 5],
)
@pytest.mark.parametrize(
    "weight1, weight2",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 0.5),
        (2, -0.5),
    ],
)
@pytest.mark.parametrize(
    "is_contiguous",
    [True, False],
)
def test_add_weighted(img_dtype, num_channels, weight1, weight2, is_contiguous):
    rng = np.random.default_rng(137)
    shape = (9, 11 if is_contiguous else 22, num_channels)
    img1 = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if img_dtype == np.uint8
        else rng.random(shape).astype(img_dtype)
    )
    img2 = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if img_dtype == np.uint8
        else rng.random(shape).astype(img_dtype)
    )
    if not is_contiguous:
        img1, img2 = img1[:, ::2], img2[:, ::2]
        assert not img1.flags.c_contiguous
        assert not img2.flags.c_contiguous
    before1, before2 = img1.copy(), img2.copy()
    expected = img1.astype(np.float32) * weight1 + img2.astype(np.float32) * weight2
    expected = (
        np.clip(np.rint(expected), 0, 255).astype(img_dtype) if img_dtype == np.uint8 else expected.astype(img_dtype)
    )

    result = add_weighted(img1, weight1, img2, weight2)

    assert result.dtype == img1.dtype
    assert result.shape == img1.shape
    if img_dtype == np.uint8:
        np.testing.assert_allclose(result, expected, atol=1, rtol=0)
    elif img_dtype == np.float16:
        np.testing.assert_array_max_ulp(result, expected, maxulp=1)
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    if weight1 == weight2 == 1:
        np.testing.assert_array_equal(result, add(img1, img2))
    np.testing.assert_array_equal(img1, before1)
    np.testing.assert_array_equal(img2, before2)
