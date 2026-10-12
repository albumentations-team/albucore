# ruff: noqa: S101
from collections.abc import Callable

import cv2
import numpy as np
import pytest

import albucore.stats as stats_mod
from albucore.stats import DEFAULT_EPS, mean, mean_std, reduce_sum, std


def _rng(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    keepdims: bool,
    dtype: type,
    base: int,
) -> np.random.Generator:
    ax_part = axis if isinstance(axis, int) else sum(v * 31**i for i, v in enumerate(axis))
    seed = (base + sum(shape) * 17 + ax_part * 13 + int(keepdims) + (0 if dtype == np.uint8 else 1)) % 2**32
    return np.random.default_rng(seed)


@pytest.mark.parametrize("shape", [(8, 9, 1), (4, 8, 9, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mean_std_global_matches_numpy_float64_reference(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)

    m, s = mean_std(arr, "global")
    ref_m = np.mean(arr, dtype=np.float64)
    ref_s = np.std(arr, dtype=np.float64) + DEFAULT_EPS
    assert np.isclose(float(m), float(ref_m), rtol=1e-5, atol=1e-5)
    assert np.isclose(float(s), float(ref_s), rtol=1e-4, atol=1e-4)


# Covers low/high channel HWC, NHWC, float32, and uint8 per-channel routing.
@pytest.mark.parametrize(
    "shape",
    [
        (16, 17, 1),  # C=1, 3D
        (16, 17, 3),  # C=3, 3D
        (16, 17, 4),  # C=4, 3D
        (16, 17, 5),  # C=5, 3D, above OpenCV meanStdDev limit
        (16, 17, 9),  # C=9, 3D
        (2, 16, 17, 3),  # NHWC, ndim=4
        (2, 16, 17, 9),  # NHWC C=9, ndim=4
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mean_std_per_channel_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    axes = tuple(range(arr.ndim - 1))
    m, s = mean_std(arr, "per_channel")
    assert m.shape == (shape[-1],), f"mean shape mismatch: {m.shape}"
    assert s.shape == (shape[-1],), f"std shape mismatch: {s.shape}"
    np.testing.assert_allclose(m, arr.mean(axis=axes, dtype=np.float64), rtol=1e-5, atol=1e-5, equal_nan=False)
    np.testing.assert_allclose(
        s,
        arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS,
        rtol=1e-4,
        atol=1e-4,
        equal_nan=False,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (16, 17, 1),
        (16, 17, 4),  # OpenCV boundary
        (16, 17, 5),  # above OpenCV limit — was buggy
        (16, 17, 9),
        (2, 16, 17, 3),
        (2, 16, 17, 9),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mean_per_channel_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    axes = tuple(range(arr.ndim - 1))
    m = mean(arr, "per_channel")
    np.testing.assert_allclose(m, arr.mean(axis=axes, dtype=np.float64), rtol=1e-5, atol=1e-5, equal_nan=False)


def test_small_skewed_float16_mean_per_channel_matches_float64() -> None:
    image = np.full((13, 17, 3), np.float16(1e-7), dtype=np.float16)
    image.reshape(-1, 3)[:2, 0] = 1.0

    result = mean(image, "per_channel")
    expected = image.mean(axis=(0, 1), dtype=np.float64)

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


def test_large_float16_mean_per_channel_matches_float64() -> None:
    image = np.random.default_rng(2).random((128, 160, 3), dtype=np.float32).astype(np.float16)

    result = mean(image, "per_channel")
    expected = image.mean(axis=(0, 1), dtype=np.float64)

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize(
    ("shape", "writeable", "torch_method"),
    [
        ((1, 16, 32, 3), True, None),
        ((1, 48, 64, 3), True, None),
        ((1, 80, 86, 3), True, "mean"),
        ((4, 128, 160, 3), True, "mean"),
        ((4, 240, 320, 3), True, "mean"),
        ((4, 128, 160, 3), False, None),
        ((1, 16, 32, 1), True, None),
        ((1, 128, 160, 1), True, "sum"),
        ((4, 768, 1024, 1), True, "sum"),
        ((10, 768, 1024, 1), True, "sum"),
        ((12, 768, 1024, 1), True, None),
        ((4, 128, 160, 1), False, None),
    ],
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_rank4_mean_uses_wide_torch_accumulator(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    writeable: bool,
    torch_method: str | None,
    keepdims: bool,
) -> None:
    source = np.random.default_rng(621135).random(shape, dtype=np.float32).astype(np.float16)
    if torch_method is not None:
        source[0, 0, 0, 0] = np.nextafter(np.float16(0), np.float16(1))
    if not writeable:
        source.setflags(write=False)
    before = source.copy()
    expected = np.mean(source, axis=(0, 1, 2), dtype=np.float64, keepdims=keepdims)
    original_mean = stats_mod.torch.mean
    original_sum = stats_mod.torch.sum
    mean_calls: list[dict[str, object]] = []
    sum_calls: list[dict[str, object]] = []

    def record_mean(image: object, *args: object, **kwargs: object) -> object:
        mean_calls.append(kwargs)
        return original_mean(image, *args, **kwargs)

    def record_sum(image: object, *args: object, **kwargs: object) -> object:
        sum_calls.append(kwargs)
        return original_sum(image, *args, **kwargs)

    monkeypatch.setattr(stats_mod.torch, "mean", record_mean)
    monkeypatch.setattr(stats_mod.torch, "sum", record_sum)

    result = mean(source, "per_channel", keepdims=keepdims)

    assert np.asarray(result).dtype == np.float64
    if torch_method == "mean":
        assert len(mean_calls) == 1
        assert mean_calls[0]["dim"] == (0, 1, 2)
        assert mean_calls[0]["dtype"] == stats_mod.torch.float64
        assert mean_calls[0]["keepdim"] is keepdims
        assert not sum_calls
        np.testing.assert_array_equal(result, expected)
    elif torch_method == "sum":
        assert not mean_calls
        assert len(sum_calls) == 1
        assert sum_calls[0]["dtype"] == stats_mod.torch.float64
        np.testing.assert_array_equal(result, expected)
    else:
        assert not mean_calls
        assert not sum_calls
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)
    np.testing.assert_array_equal(source, before)


def test_float16_c1_mean_per_channel_preserves_near_constant_accuracy() -> None:
    image = np.clip(np.random.default_rng(71).normal(0.5, 0.002, (128, 160, 1)), 0, 1).astype(np.float16)
    assert np.min(image) > np.finfo(np.float16).tiny

    result = mean(image, "per_channel")
    expected = image.mean(axis=(0, 1), dtype=np.float64)

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


def test_global_float16_mean_preserves_subnormal_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    subnormal = np.nextafter(np.float16(0), np.float16(1))
    image = np.full((128, 128, 1), subnormal, dtype=np.float16)
    image[0, 0, 0] = np.float16(1.0)
    expected = np.mean(image, dtype=np.float64)
    original_mean = stats_mod.cv2.mean
    input_dtypes: list[np.dtype] = []

    def record_mean(source: np.ndarray) -> tuple[float, ...]:
        input_dtypes.append(source.dtype)
        return original_mean(source)

    monkeypatch.setattr(stats_mod.cv2, "mean", record_mean)

    result = mean(image, "global")

    assert input_dtypes == [np.dtype(np.float32)]
    assert np.asarray(result).dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [np.inf, -np.inf]])
def test_global_float16_mean_preserves_nonfinite_semantics(values: list[float]) -> None:
    image = np.zeros((64, 64, 1), dtype=np.float16)
    image.flat[: len(values)] = values
    with np.errstate(invalid="ignore"):
        expected = np.mean(image, dtype=np.float64)
        result = mean(image, "global")

    assert np.asarray(result).dtype == np.float64
    np.testing.assert_equal(result, expected)


def test_small_float16_mean_std_preserves_subnormals() -> None:
    smallest_subnormal = np.nextafter(np.float16(0), np.float16(1))
    image = np.full((5, 7, 1), smallest_subnormal, dtype=np.float16)
    image[0, 0, 0] = 1.0

    result_mean, result_std = mean_std(image, "per_channel")
    expected_mean = image.mean(axis=(0, 1), dtype=np.float64)
    expected_std = image.std(axis=(0, 1), dtype=np.float64) + DEFAULT_EPS

    np.testing.assert_allclose(result_mean, expected_mean, rtol=1e-7, atol=0)
    np.testing.assert_allclose(result_std, expected_std, rtol=1e-7, atol=1e-7)


def test_large_float16_mean_std_preserves_subnormal_accumulation() -> None:
    subnormal = np.nextafter(np.float16(np.finfo(np.float16).tiny), np.float16(0))
    image = np.full((128, 128, 3), subnormal, dtype=np.float16)
    image.reshape(-1, 3)[0, 0] = 1.0

    result_mean, result_std = mean_std(image, "per_channel")
    expected_mean = image.mean(axis=(0, 1), dtype=np.float64)
    expected_std = image.std(axis=(0, 1), dtype=np.float64) + DEFAULT_EPS

    np.testing.assert_allclose(result_mean, expected_mean, rtol=1e-7, atol=0)
    np.testing.assert_allclose(result_std, expected_std, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "shape",
    [
        (16, 17, 1),
        (16, 17, 4),  # OpenCV boundary
        (16, 17, 5),  # above OpenCV limit — was buggy
        (16, 17, 9),
        (2, 16, 17, 3),
        (2, 16, 17, 9),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_std_per_channel_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    axes = tuple(range(arr.ndim - 1))
    s = std(arr, "per_channel")
    np.testing.assert_allclose(
        s,
        arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS,
        rtol=1e-4,
        atol=1e-4,
        equal_nan=False,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (11, 13, 1),
        (5, 6, 4),
        (5, 6, 5),  # C>4: NumKong per-channel chain (3D)
        (5, 6, 9),  # C=9, 3D
        (2, 8, 9, 3),  # NHWC: must fall back to NumPy (ndim>3)
        (2, 8, 9, 9),  # NHWC C=9
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_reduce_sum_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    acc = np.uint64 if dtype == np.uint8 else np.float64
    axes = tuple(range(arr.ndim - 1))

    g = reduce_sum(arr, "global")
    expected_g = np.sum(arr, dtype=acc)
    assert np.shape(g) == np.shape(expected_g), f"global shape {shape} {dtype}"
    np.testing.assert_array_equal(g, expected_g, err_msg=f"global {shape} {dtype}")

    pc = reduce_sum(arr, "per_channel")
    np.testing.assert_array_equal(
        pc,
        np.sum(arr, axis=axes, dtype=acc),
        err_msg=f"per_channel {shape} {dtype}",
    )

    gk = reduce_sum(arr, "global", keepdims=True)
    np.testing.assert_array_equal(
        gk,
        np.sum(arr, dtype=acc, keepdims=True),
        err_msg=f"global keepdims {shape} {dtype}",
    )

    pck = reduce_sum(arr, "per_channel", keepdims=True)
    np.testing.assert_array_equal(
        pck,
        np.sum(arr, axis=axes, dtype=acc, keepdims=True),
        err_msg=f"per_channel keepdims {shape} {dtype}",
    )


@pytest.mark.parametrize("keepdims", [False, True])
def test_float16_global_reduce_sum_uses_measured_torch_window(
    keepdims: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.random.default_rng(73).uniform(-1, 1, (128, 160, 1)).astype(np.float16)
    expected = np.sum(source, dtype=np.float64, keepdims=keepdims)
    original = stats_mod.torch.sum
    calls: list[dict[str, object]] = []

    def record_torch_sum(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(stats_mod.torch, "sum", record_torch_sum)
    result = reduce_sum(source, "global", keepdims=keepdims)

    assert np.asarray(result).dtype == np.float64
    np.testing.assert_array_equal(result, expected)
    assert calls == ([] if keepdims else [{"dtype": stats_mod.torch.float64}])


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [np.inf, -np.inf]])
def test_float16_global_reduce_sum_preserves_nonfinite_values(values: list[float]) -> None:
    source = np.zeros((128, 160, 1), dtype=np.float16)
    source.reshape(-1)[: len(values)] = values
    with np.errstate(invalid="ignore"):
        expected = np.sum(source, dtype=np.float64)
        result = reduce_sum(source, "global")

    assert np.asarray(result).dtype == np.float64
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_mean_and_std_delegation() -> None:
    arr = np.full((3, 4, 2), 10, dtype=np.float32)
    m1, _ = mean_std(arr, "global")
    m2 = mean(arr, "global")
    assert float(m1) == float(m2)
    _, s1 = mean_std(arr, "global")
    s2 = std(arr, "global")
    assert float(s1) == float(s2)


# Why the cv2/axes bug slipped through: older tests only used axis="per_channel", which
# resolves to tuple(range(ndim - 1)) — never explicit axis=-1 on HWC (channel reduction).
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    ("shape", "axis", "keepdims"),
    [
        # Channel reduction (must match np.mean(..., axis=-1) → 2D for HWC)
        ((14, 15, 3), -1, False),
        ((14, 15, 3), -1, True),
        ((14, 15, 3), 2, False),
        ((14, 15, 1), -1, False),
        ((14, 15, 4), -1, False),
        ((14, 15, 5), -1, False),
        # Spatial / mixed explicit axes
        ((14, 15, 3), 0, False),
        ((14, 15, 3), (0, 1), False),
        ((14, 15, 3), (0, 1), True),
        ((2, 8, 9, 3), -1, False),
        ((2, 8, 9, 3), (1, 2), False),
        ((2, 8, 9, 3), (1, 2), True),
    ],
)
def test_mean_std_explicit_axis_matches_numpy(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    keepdims: bool,
    dtype: type,
) -> None:
    rng = _rng(shape, axis, keepdims, dtype, base=20250323)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    ref_m = np.mean(arr, axis=axis, dtype=np.float64, keepdims=keepdims)
    ref_s = np.std(arr, axis=axis, dtype=np.float64, keepdims=keepdims) + DEFAULT_EPS
    m, s = mean_std(arr, axis, keepdims=keepdims)
    assert m.shape == ref_m.shape
    assert s.shape == ref_s.shape
    np.testing.assert_allclose(m, ref_m, rtol=1e-5, atol=1e-5, equal_nan=False)
    np.testing.assert_allclose(s, ref_s, rtol=1e-4, atol=1e-4, equal_nan=False)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    ("shape", "axis", "keepdims"),
    [
        ((14, 15, 3), -1, False),
        ((14, 15, 3), -1, True),
        ((14, 15, 4), -1, False),
        ((14, 15, 3), (0, 1), False),
        ((2, 8, 9, 3), (1, 2), True),
    ],
)
def test_mean_explicit_axis_matches_numpy(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    keepdims: bool,
    dtype: type,
) -> None:
    rng = _rng(shape, axis, keepdims, dtype, base=20250324)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    ref = np.mean(arr, axis=axis, dtype=np.float64, keepdims=keepdims)
    out = mean(arr, axis, keepdims=keepdims)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5, equal_nan=False)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    ("shape", "axis", "keepdims"),
    [
        ((14, 15, 3), -1, False),
        ((14, 15, 3), -1, True),
        ((14, 15, 4), -1, False),
        ((14, 15, 3), (0, 1), False),
        ((2, 8, 9, 3), (1, 2), True),
    ],
)
def test_std_explicit_axis_matches_numpy(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    keepdims: bool,
    dtype: type,
) -> None:
    rng = _rng(shape, axis, keepdims, dtype, base=20250325)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    ref = np.std(arr, axis=axis, dtype=np.float64, keepdims=keepdims) + DEFAULT_EPS
    out = std(arr, axis, keepdims=keepdims)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, equal_nan=False)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    ("shape", "axis", "keepdims"),
    [
        ((14, 15, 3), -1, False),
        ((14, 15, 3), (0, 1), True),
        ((2, 8, 9, 3), (1, 2), False),
    ],
)
def test_reduce_sum_explicit_axis_matches_numpy(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    keepdims: bool,
    dtype: type,
) -> None:
    rng = _rng(shape, axis, keepdims, dtype, base=20250326)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    acc = np.uint64 if dtype == np.uint8 else np.float64
    ref = np.sum(arr, axis=axis, dtype=acc, keepdims=keepdims)
    out = reduce_sum(arr, axis, keepdims=keepdims)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("shape", [(5, 6, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_axis_none_equals_string_global(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(404)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    assert float(mean(arr)) == float(mean(arr, "global"))
    assert float(std(arr)) == float(std(arr, "global"))
    m0, s0 = mean_std(arr)
    m1, s1 = mean_std(arr, "global")
    assert float(m0) == float(m1)
    assert float(s0) == float(s1)
    acc = np.uint64 if dtype == np.uint8 else np.float64
    g = reduce_sum(arr)
    named_g = reduce_sum(arr, "global")
    expected_g = np.sum(arr, dtype=acc)
    assert np.shape(g) == np.shape(named_g)
    np.testing.assert_array_equal(g, named_g)
    assert np.shape(g) == np.shape(expected_g)
    np.testing.assert_array_equal(g, expected_g)


@pytest.mark.parametrize("shape", [(7, 8, 3), (2, 5, 6, 2)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_global_keepdims_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(405)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    ref_m = np.mean(arr, dtype=np.float64, keepdims=True)
    ref_s = np.std(arr, dtype=np.float64, keepdims=True) + DEFAULT_EPS
    m = mean(arr, keepdims=True)
    s = std(arr, keepdims=True)
    ms_m, ms_s = mean_std(arr, keepdims=True)
    assert m.shape == ref_m.shape
    np.testing.assert_allclose(m, ref_m, rtol=1e-5, atol=1e-5, equal_nan=False)
    assert s.shape == ref_s.shape
    np.testing.assert_allclose(s, ref_s, rtol=1e-4, atol=1e-4, equal_nan=False)
    assert ms_m.shape == ref_m.shape
    np.testing.assert_allclose(ms_m, ref_m, rtol=1e-5, atol=1e-5, equal_nan=False)
    assert ms_s.shape == ref_s.shape
    np.testing.assert_allclose(ms_s, ref_s, rtol=1e-4, atol=1e-4, equal_nan=False)

    acc = np.uint64 if dtype == np.uint8 else np.float64
    gk = reduce_sum(arr, "global", keepdims=True)
    np.testing.assert_array_equal(gk, np.sum(arr, dtype=acc, keepdims=True))


@pytest.mark.parametrize("shape", [(9, 10, 2), (4, 5, 6, 1)])
@pytest.mark.parametrize("axis", [None, "global", "per_channel"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mean_dtype_kwarg_matches_cast_numpy(
    shape: tuple[int, ...],
    axis: None | str,
    dtype: type,
) -> None:
    rng = np.random.default_rng(406)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    ref64 = np.mean(
        arr,
        axis=None if axis in (None, "global") else tuple(range(arr.ndim - 1)),
        dtype=np.float64,
        keepdims=False,
    )
    out = mean(arr, axis, dtype=np.float32)
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        out,
        np.asarray(ref64, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
        equal_nan=False,
    )


@pytest.mark.parametrize("shape", [(6, 7, 2)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_std_custom_eps_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(407)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    eps = 1e-3
    ref = np.std(arr, dtype=np.float64) + eps
    assert np.isclose(float(std(arr, eps=eps)), float(ref), rtol=1e-4, atol=1e-4)
    _, s = mean_std(arr, eps=eps)
    assert np.isclose(float(s), float(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("dtype", "expected_acc_dtype"),
    [
        (np.int32, np.int64),
        (np.int64, np.int64),
        (np.uint32, np.uint64),
        (np.uint64, np.uint64),
        (np.float64, np.float64),
        (np.bool_, np.int64),
    ],
)
def test_reduce_sum_accumulator_dtype(dtype: type, expected_acc_dtype: type) -> None:
    arr = np.ones((4, 4, 3), dtype=dtype)
    result = reduce_sum(arr)
    assert result.dtype == expected_acc_dtype
    per_ch = reduce_sum(arr, "per_channel")
    assert per_ch.dtype == expected_acc_dtype


@pytest.mark.parametrize(
    ("dtype", "fill", "expected_acc_dtype"),
    [
        # near int32 max — would overflow into int32 but must not with int64 accumulator
        (np.int32, np.iinfo(np.int32).max, np.int64),
        # unsigned — must not sign-extend into int64
        (np.uint32, np.iinfo(np.uint32).max, np.uint64),
        # float64 precision — all-same values, check no precision loss in sum
        (np.float64, 1.0 / 3.0, np.float64),
    ],
)
def test_reduce_sum_overflow_and_precision(dtype: type, fill: float, expected_acc_dtype: type) -> None:
    arr = np.full((8, 8, 1), fill, dtype=dtype)
    expected = np.sum(arr, dtype=expected_acc_dtype)
    result = reduce_sum(arr)
    assert result.dtype == expected_acc_dtype
    assert result == expected


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_per_channel_mean_std_uint8_matches_numpy_and_opencv_reference(c: int) -> None:
    """NumKong route should still agree with NumPy and OpenCV's per-channel values."""
    rng = np.random.default_rng(408 + c)
    arr = rng.integers(0, 256, size=(19, 21, c), dtype=np.uint8)
    axes = (0, 1)
    m_al, s_al = mean_std(arr, "per_channel")
    m_np = arr.mean(axis=axes, dtype=np.float64)
    s_np = arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS
    mean_cv, std_cv = cv2.meanStdDev(arr)
    m_cv = mean_cv[:, 0].astype(np.float64, copy=False)
    s_cv = (std_cv[:, 0] + DEFAULT_EPS).astype(np.float64, copy=False)
    np.testing.assert_allclose(m_al, m_np, rtol=1e-5, atol=1e-5, equal_nan=False)
    np.testing.assert_allclose(s_al, s_np, rtol=1e-4, atol=1e-4, equal_nan=False)
    np.testing.assert_allclose(m_cv, m_np, rtol=1e-5, atol=1e-5, equal_nan=False)
    np.testing.assert_allclose(s_cv, s_np, rtol=1e-4, atol=1e-4, equal_nan=False)
    mu_only = np.asarray(cv2.mean(arr)[:c], dtype=np.float64)
    np.testing.assert_allclose(mean(arr, "per_channel"), m_np, rtol=1e-5, atol=1e-5, equal_nan=False)
    np.testing.assert_allclose(mu_only, m_np, rtol=1e-5, atol=1e-5, equal_nan=False)


def test_std_per_channel_uint8_hwc_uses_numkong_route(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(412)
    arr = rng.integers(0, 256, size=(19, 21, 3), dtype=np.uint8)
    called = False

    def fake_numkong(arr_arg: np.ndarray, eps_arg: float) -> tuple[np.ndarray, np.ndarray]:
        nonlocal called
        called = True
        axes = tuple(range(arr_arg.ndim - 1))
        mean_value = np.asarray(arr_arg.mean(axis=axes, dtype=np.float64), dtype=np.float64)
        std_value = np.asarray(arr_arg.std(axis=axes, dtype=np.float64) + eps_arg, dtype=np.float64)
        return mean_value, std_value

    def fail_mean_std_dev(arr_arg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("uint8 per-channel std should not use cv2.meanStdDev")

    monkeypatch.setattr(stats_mod, "_mean_std_per_channel_numkong", fake_numkong)
    monkeypatch.setattr(stats_mod.cv2, "meanStdDev", fail_mean_std_dev)

    out = std(arr, "per_channel")
    ref = arr.std(axis=(0, 1), dtype=np.float64) + DEFAULT_EPS

    assert called
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, equal_nan=False)


def test_std_per_channel_float32_rgb_keeps_opencv_route(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(413)
    arr = rng.random((19, 21, 3), dtype=np.float32)
    called = False

    def fake_mean_std_dev(arr_arg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal called
        called = True
        axes = (0, 1)
        mean_value = np.asarray(arr_arg.mean(axis=axes, dtype=np.float64), dtype=np.float64).reshape(-1, 1)
        std_value = np.asarray(arr_arg.std(axis=axes, dtype=np.float64), dtype=np.float64).reshape(-1, 1)
        return mean_value, std_value

    def fail_numkong(arr_arg: np.ndarray, eps_arg: float) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("float32 RGB per-channel std should use cv2.meanStdDev")

    monkeypatch.setattr(stats_mod.cv2, "meanStdDev", fake_mean_std_dev)
    monkeypatch.setattr(stats_mod, "_mean_std_per_channel_numkong", fail_numkong)

    out = std(arr, "per_channel")
    ref = arr.std(axis=(0, 1), dtype=np.float64) + DEFAULT_EPS

    assert called
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, equal_nan=False)


@pytest.mark.parametrize("statistic", ["std", "mean_std"])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("pattern", ["random", "near_constant", "subnormal_residual"])
def test_float16_hwc_numkong_moments_route_matches_wide_reference(
    monkeypatch: pytest.MonkeyPatch,
    statistic: str,
    channels: int,
    pattern: str,
) -> None:
    monkeypatch.setattr(stats_mod, "_NUMKONG_NEON_HALF", True)
    original_numkong = stats_mod._mean_std_per_channel_numkong
    called = False

    def record_numkong(arr: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
        nonlocal called
        called = True
        return original_numkong(arr, eps)

    monkeypatch.setattr(stats_mod, "_mean_std_per_channel_numkong", record_numkong)
    shape = (240, 320, channels)
    if pattern == "random":
        source = np.random.default_rng(920_011 + channels).uniform(-1, 1, shape).astype(np.float16)
    elif pattern == "near_constant":
        values = np.linspace(0.25, 0.9, num=channels, dtype=np.float32).astype(np.float16)
        source = np.broadcast_to(values, shape).copy()
    else:
        smallest = np.nextafter(np.float16(0), np.float16(1))
        source = np.full(shape, smallest, dtype=np.float16)
        source[0, 0, :] = np.float16(1.0)
    expected_mean = np.mean(source, axis=(0, 1), dtype=np.float64)
    expected_std = np.std(source, axis=(0, 1), dtype=np.float64) + DEFAULT_EPS

    if statistic == "std":
        result = std(source, "per_channel")
        np.testing.assert_allclose(result, expected_std, rtol=1e-4, atol=1e-8)
    else:
        result_mean, result_std = mean_std(source, "per_channel")
        np.testing.assert_allclose(result_mean, expected_mean, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(result_std, expected_std, rtol=1e-4, atol=1e-8)

    assert called
