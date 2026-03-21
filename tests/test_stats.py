# ruff: noqa: S101
import numpy as np
import pytest

from albucore.stats import DEFAULT_EPS, mean, mean_std, std


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


@pytest.mark.parametrize("shape", [(16, 17, 3), (2, 16, 17, 1)])
def test_mean_std_per_channel_matches_numpy(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    arr = rng.random(shape, dtype=np.float32)
    axes = tuple(range(arr.ndim - 1))
    m, s = mean_std(arr, "per_channel")
    assert np.allclose(m, arr.mean(axis=axes), rtol=1e-5)
    assert np.allclose(s, arr.std(axis=axes) + DEFAULT_EPS, rtol=1e-4)


def test_mean_std_uint8_per_channel_3d_opencv_path() -> None:
    arr = np.random.default_rng(1).integers(0, 256, size=(5, 6, 4), dtype=np.uint8)
    axes = (0, 1)
    m, s = mean_std(arr, "per_channel")
    assert np.allclose(m, arr.mean(axis=axes, dtype=np.float64), rtol=1e-5)
    assert np.allclose(s, arr.std(axis=axes, dtype=np.float64) + DEFAULT_EPS, rtol=1e-4)


def test_mean_and_std_delegation() -> None:
    arr = np.full((3, 4, 2), 10, dtype=np.float32)
    m1, _ = mean_std(arr, "global")
    m2 = mean(arr, "global")
    assert float(m1) == float(m2)
    _, s1 = mean_std(arr, "global")
    s2 = std(arr, "global")
    assert float(s1) == float(s2)
