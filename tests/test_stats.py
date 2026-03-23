# ruff: noqa: S101
import numpy as np
import pytest

from albucore.stats import DEFAULT_EPS, mean, mean_std, reduce_sum, std


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
