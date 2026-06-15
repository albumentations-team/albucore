from __future__ import annotations

import numpy as np
from hypothesis import given

import albucore as ac
from albucore.stats import DEFAULT_EPS
from tests.property.strategies import hwc_images


@given(hwc_images())
def test_reduce_sum_global_matches_numpy(img: np.ndarray) -> None:
    expected_dtype = np.uint64 if img.dtype == np.uint8 else np.float64
    expected = np.sum(img, dtype=expected_dtype)

    np.testing.assert_allclose(ac.reduce_sum(img, "global"), expected)


@given(hwc_images())
def test_reduce_sum_per_channel_matches_numpy(img: np.ndarray) -> None:
    axes = tuple(range(img.ndim - 1))
    expected_dtype = np.uint64 if img.dtype == np.uint8 else np.float64
    expected = np.sum(img, axis=axes, dtype=expected_dtype)

    np.testing.assert_allclose(ac.reduce_sum(img, "per_channel"), expected)


@given(hwc_images())
def test_reduce_sum_explicit_axis_matches_numpy(img: np.ndarray) -> None:
    expected_dtype = np.uint64 if img.dtype == np.uint8 else np.float64
    expected = np.sum(img, axis=0, dtype=expected_dtype)

    np.testing.assert_allclose(ac.reduce_sum(img, 0), expected)


@given(hwc_images())
def test_mean_global_matches_numpy(img: np.ndarray) -> None:
    expected = np.mean(img, dtype=np.float64)

    np.testing.assert_allclose(ac.mean(img, "global"), expected)


@given(hwc_images())
def test_mean_per_channel_matches_numpy(img: np.ndarray) -> None:
    axes = tuple(range(img.ndim - 1))
    expected = np.mean(img, axis=axes, dtype=np.float64)

    np.testing.assert_allclose(ac.mean(img, "per_channel"), expected)


@given(hwc_images())
def test_mean_explicit_axis_matches_numpy(img: np.ndarray) -> None:
    expected = np.mean(img, axis=0, dtype=np.float64)

    np.testing.assert_allclose(ac.mean(img, 0), expected)


@given(hwc_images())
def test_mean_std_global_matches_numpy(img: np.ndarray) -> None:
    expected_mean = np.mean(img, dtype=np.float64)
    expected_std = np.std(img, dtype=np.float64) + DEFAULT_EPS

    actual_mean, actual_std = ac.mean_std(img, "global")

    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_std, expected_std, atol=1e-5)


@given(hwc_images())
def test_mean_std_per_channel_matches_numpy(img: np.ndarray) -> None:
    axes = tuple(range(img.ndim - 1))
    expected_mean = np.mean(img, axis=axes, dtype=np.float64)
    expected_std = np.std(img, axis=axes, dtype=np.float64) + DEFAULT_EPS

    actual_mean, actual_std = ac.mean_std(img, "per_channel")

    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_std, expected_std, atol=1e-5)
