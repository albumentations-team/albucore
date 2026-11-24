# ruff: noqa: S101, SLF001
"""Float32 global reduce_sum and mean use NumKong on Linux x86-64 and keep Torch/NumPy elsewhere."""

import numpy as np
import pytest

import albucore.stats as stats_mod
from albucore import mean, reduce_sum


def _forbid_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        msg = "Torch reduction must not be reached."
        raise AssertionError(msg)

    monkeypatch.setattr(stats_mod, "reduce_sum_torch", fail)


@pytest.mark.parametrize("shape", [(11, 13, 1), (64, 64, 3), (2, 40, 50, 9), (1024, 520, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_linux_float32_global_sum_and_mean_use_numkong(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], keepdims: bool
) -> None:
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", True)
    _forbid_torch(monkeypatch)
    image = np.random.default_rng(1).random(shape, dtype=np.float32)

    total = reduce_sum(image, keepdims=keepdims)
    average = mean(image, keepdims=keepdims)

    expected = np.sum(image, dtype=np.float64, keepdims=keepdims)
    assert total.dtype == np.float64
    np.testing.assert_array_equal(total, expected)
    assert average.dtype == np.float64
    assert np.shape(average) == np.shape(expected)
    np.testing.assert_allclose(average, np.mean(image, dtype=np.float64, keepdims=keepdims), rtol=1e-15)


def test_linux_float32_global_sum_keeps_wide_accumulation_and_special_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", True)
    rng = np.random.default_rng(2)
    hard = (rng.standard_normal((300, 400, 3)) * 1e6).astype(np.float32)
    hard[::7] *= np.float32(1e-6)
    reference = np.sum(hard, dtype=np.float64)
    assert abs(float(reduce_sum(hard)) - reference) <= 1e-15 * float(np.sum(np.abs(hard), dtype=np.float64))

    huge = np.array([3e38, 3e38, -3e38, 1.0], dtype=np.float32).reshape(2, 2, 1)
    assert reduce_sum(huge) == np.sum(huge, dtype=np.float64)
    for special in (np.inf, -np.inf, np.nan):
        image = np.ones((8, 8, 1), dtype=np.float32)
        image[3, 3, 0] = special
        np.testing.assert_array_equal(reduce_sum(image), np.sum(image, dtype=np.float64))
    zeros = np.full((8, 8, 1), -0.0, dtype=np.float32)
    np.testing.assert_array_equal(reduce_sum(zeros), np.sum(zeros, dtype=np.float64))


def test_linux_float32_global_sum_handles_strided_and_read_only_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", True)
    image = np.random.default_rng(3).random((40, 60, 3), dtype=np.float32)
    strided = image[:, ::2]
    fortran = np.asfortranarray(image)
    read_only = image.copy()
    read_only.flags.writeable = False
    for array in (strided, fortran, read_only):
        np.testing.assert_array_equal(reduce_sum(array), np.sum(array, dtype=np.float64))
        np.testing.assert_allclose(mean(array), np.mean(array, dtype=np.float64), rtol=1e-15)


def test_non_linux_float32_global_sum_keeps_torch_and_numpy_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod, "_LINUX_X86_64", False)
    monkeypatch.setattr("albucore.torch_backend.TORCH_MIN_ELEMENTS", 1)
    calls = 0
    original = stats_mod.reduce_sum_torch

    def spy(
        arr: np.ndarray, axes: tuple[int, ...], *, keepdims: bool, minimum_elements: int | None = None
    ) -> np.generic | np.ndarray | None:
        nonlocal calls
        calls += 1
        return original(arr, axes, keepdims=keepdims, minimum_elements=minimum_elements)

    monkeypatch.setattr(stats_mod, "reduce_sum_torch", spy)
    image = np.random.default_rng(4).random((16, 16, 3), dtype=np.float32)
    np.testing.assert_array_equal(reduce_sum(image), np.sum(image, dtype=np.float64))
    assert calls == 1
