# ruff: noqa: S101
import cv2
import numpy as np
import pytest

from albucore.stats import DEFAULT_EPS, mean, mean_std, reduce_sum, std


def _rng(shape: tuple[int, ...], axis: int | tuple[int, ...], keepdims: bool, dtype: type, base: int) -> np.random.Generator:
    ax_part = axis if isinstance(axis, int) else sum(v * 31**i for i, v in enumerate(axis))
    seed = (base + sum(shape) * 17 + ax_part * 13 + int(keepdims) + (0 if dtype == np.uint8 else 1)) % 2**32
    return np.random.default_rng(seed)


@pytest.mark.parametrize("shape", [(8, 9, 1), (4, 8, 9, 3), (2, 4, 8, 9, 1)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mean_std_global_matches_numpy_float64_reference(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)

    m, s = mean_std(arr, "global")
    ref_m = np.mean(arr, dtype=np.float64)
    ref_s = np.std(arr, dtype=np.float64) + DEFAULT_EPS
    assert np.isclose(float(m), float(ref_m), rtol=1e-5, atol=1e-5)
    assert np.isclose(float(s), float(ref_s), rtol=1e-4, atol=1e-4)


# Covers: C<=4 (OpenCV path), C>4 (NumPy fallback), NHWC (ndim>3), float32 and uint8.
@pytest.mark.parametrize(
    "shape",
    [
        (16, 17, 1),   # C=1, 3D, OpenCV path
        (16, 17, 3),   # C=3, 3D, OpenCV path
        (16, 17, 4),   # C=4, 3D, OpenCV boundary
        (16, 17, 5),   # C=5, 3D, must use NumPy (previously buggy for cv2.meanStdDev)
        (16, 17, 9),   # C=9, 3D, well above OpenCV limit
        (2, 16, 17, 3),  # NHWC, ndim=4, NumPy path
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
    assert np.allclose(m, arr.mean(axis=axes, dtype=np.float64), rtol=1e-5, atol=1e-5)
    assert np.allclose(s, arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (16, 17, 1),
        (16, 17, 4),   # OpenCV boundary
        (16, 17, 5),   # above OpenCV limit — was buggy
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
    assert np.allclose(m, arr.mean(axis=axes, dtype=np.float64), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "shape",
    [
        (16, 17, 1),
        (16, 17, 4),   # OpenCV boundary
        (16, 17, 5),   # above OpenCV limit — was buggy
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
    assert np.allclose(s, arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [
        (11, 13, 1),
        (5, 6, 4),
        (5, 6, 5),   # C>4: NumKong per-channel chain (3D)
        (5, 6, 9),   # C=9, 3D
        (2, 8, 9, 3),   # NHWC: must fall back to NumPy (ndim>3)
        (2, 8, 9, 9),   # NHWC C=9
        (2, 3, 4, 5, 3),  # NDHWC: ndim=5
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_reduce_sum_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(7)
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    acc = np.uint64 if dtype == np.uint8 else np.float64
    axes = tuple(range(arr.ndim - 1))

    g = reduce_sum(arr, "global")
    assert np.array_equal(g, np.sum(arr, dtype=acc)), f"global {shape} {dtype}"

    pc = reduce_sum(arr, "per_channel")
    assert np.array_equal(pc, np.sum(arr, axis=axes, dtype=acc)), f"per_channel {shape} {dtype}"

    gk = reduce_sum(arr, "global", keepdims=True)
    assert np.array_equal(gk, np.sum(arr, dtype=acc, keepdims=True)), f"global keepdims {shape} {dtype}"

    pck = reduce_sum(arr, "per_channel", keepdims=True)
    assert np.array_equal(pck, np.sum(arr, axis=axes, dtype=acc, keepdims=True)), f"per_channel keepdims {shape} {dtype}"


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
    "shape,axis,keepdims",
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
        ((2, 3, 4, 5, 3), -2, False),
    ],
)
def test_mean_std_explicit_axis_matches_numpy(
    shape: tuple[int, ...],
    axis: int | tuple[int, ...],
    keepdims: bool,
    dtype: type,
) -> None:
    rng = _rng(shape, axis, keepdims, dtype, base=20250323)
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    ref_m = np.mean(arr, axis=axis, dtype=np.float64, keepdims=keepdims)
    ref_s = np.std(arr, axis=axis, dtype=np.float64, keepdims=keepdims) + DEFAULT_EPS
    m, s = mean_std(arr, axis, keepdims=keepdims)
    assert m.shape == ref_m.shape
    assert s.shape == ref_s.shape
    assert np.allclose(m, ref_m, rtol=1e-5, atol=1e-5)
    assert np.allclose(s, ref_s, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape,axis,keepdims",
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
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    ref = np.mean(arr, axis=axis, dtype=np.float64, keepdims=keepdims)
    out = mean(arr, axis, keepdims=keepdims)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape,axis,keepdims",
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
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    ref = np.std(arr, axis=axis, dtype=np.float64, keepdims=keepdims) + DEFAULT_EPS
    out = std(arr, axis, keepdims=keepdims)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape,axis,keepdims",
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
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    acc = np.uint64 if dtype == np.uint8 else np.float64
    ref = np.sum(arr, axis=axis, dtype=acc, keepdims=keepdims)
    out = reduce_sum(arr, axis, keepdims=keepdims)
    assert out.shape == ref.shape
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("shape", [(5, 6, 3), (2, 4, 8, 9, 1)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_axis_none_equals_string_global(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(404)
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    assert float(mean(arr)) == float(mean(arr, "global"))
    assert float(std(arr)) == float(std(arr, "global"))
    m0, s0 = mean_std(arr)
    m1, s1 = mean_std(arr, "global")
    assert float(m0) == float(m1) and float(s0) == float(s1)
    acc = np.uint64 if dtype == np.uint8 else np.float64
    assert np.array_equal(reduce_sum(arr), reduce_sum(arr, "global"))
    assert np.array_equal(reduce_sum(arr), np.sum(arr, dtype=acc))


@pytest.mark.parametrize("shape", [(7, 8, 3), (2, 5, 6, 2)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_global_keepdims_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(405)
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    ref_m = np.mean(arr, dtype=np.float64, keepdims=True)
    ref_s = np.std(arr, dtype=np.float64, keepdims=True) + DEFAULT_EPS
    m = mean(arr, keepdims=True)
    s = std(arr, keepdims=True)
    ms_m, ms_s = mean_std(arr, keepdims=True)
    assert m.shape == ref_m.shape and np.allclose(m, ref_m, rtol=1e-5, atol=1e-5)
    assert s.shape == ref_s.shape and np.allclose(s, ref_s, rtol=1e-4, atol=1e-4)
    assert ms_m.shape == ref_m.shape and np.allclose(ms_m, ref_m, rtol=1e-5, atol=1e-5)
    assert ms_s.shape == ref_s.shape and np.allclose(ms_s, ref_s, rtol=1e-4, atol=1e-4)

    acc = np.uint64 if dtype == np.uint8 else np.float64
    gk = reduce_sum(arr, "global", keepdims=True)
    assert np.array_equal(gk, np.sum(arr, dtype=acc, keepdims=True))


@pytest.mark.parametrize("shape", [(9, 10, 2), (4, 5, 6, 1)])
@pytest.mark.parametrize("axis", [None, "global", "per_channel"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_mean_dtype_kwarg_matches_cast_numpy(
    shape: tuple[int, ...],
    axis: None | str,
    dtype: type,
) -> None:
    rng = np.random.default_rng(406)
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    ref64 = np.mean(
        arr,
        axis=None if axis in (None, "global") else tuple(range(arr.ndim - 1)),
        dtype=np.float64,
        keepdims=False,
    )
    out = mean(arr, axis, dtype=np.float32)
    assert out.dtype == np.float32
    assert np.allclose(out, np.asarray(ref64, dtype=np.float32), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("shape", [(6, 7, 2)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_std_custom_eps_matches_numpy(shape: tuple[int, ...], dtype: type) -> None:
    rng = np.random.default_rng(407)
    arr = (
        rng.integers(0, 256, size=shape, dtype=np.uint8)
        if dtype == np.uint8
        else rng.random(shape, dtype=np.float32)
    )
    eps = 1e-3
    ref = np.std(arr, dtype=np.float64) + eps
    assert np.isclose(float(std(arr, eps=eps)), float(ref), rtol=1e-4, atol=1e-4)
    _, s = mean_std(arr, eps=eps)
    assert np.isclose(float(s), float(ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [np.int32, np.float64, np.bool_, np.complex64])
def test_unsupported_dtype_raises_mean_std(dtype: type) -> None:
    arr = np.ones((2, 2, 1), dtype=dtype)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mean(arr)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        std(arr)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mean_std(arr)


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
def test_opencv_per_channel_mean_std_matches_numpy_uint8(c: int) -> None:
    """Fast path uses cv2; reference is NumPy — catches OpenCV / ordering drift."""
    rng = np.random.default_rng(408 + c)
    arr = rng.integers(0, 256, size=(19, 21, c), dtype=np.uint8)
    axes = (0, 1)
    m_al, s_al = mean_std(arr, "per_channel")
    m_np = arr.mean(axis=axes, dtype=np.float64)
    s_np = arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS
    mean_cv, std_cv = cv2.meanStdDev(arr)
    m_cv = mean_cv[:, 0].astype(np.float64, copy=False)
    s_cv = (std_cv[:, 0] + DEFAULT_EPS).astype(np.float64, copy=False)
    assert np.allclose(m_al, m_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(s_al, s_np, rtol=1e-4, atol=1e-4)
    assert np.allclose(m_cv, m_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(s_cv, s_np, rtol=1e-4, atol=1e-4)
    mu_only = np.asarray(cv2.mean(arr)[:c], dtype=np.float64)
    assert np.allclose(mean(arr, "per_channel"), m_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(mu_only, m_np, rtol=1e-5, atol=1e-5)
