from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import albucore


@pytest.fixture
def conversion_benchmarks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return import_module("benchmark_router_synthetic"), import_module("benchmark_tensor_router_paired")


@pytest.mark.parametrize("op_name", ["convert_dtype_float16", "convert_dtype_float32"])
@pytest.mark.parametrize("container", ["numpy", "tensor"])
def test_fixed_float_target_pairs_skip_identity_vs_cast(conversion_benchmarks, op_name, container):
    router, tensor_router = conversion_benchmarks
    shape = (3, 5, 1)
    half = np.linspace(0, 1, np.prod(shape), dtype=np.float16).reshape(shape)
    wide = half.astype(np.float32)
    builder = dict(router._registry_functions())[op_name]
    if container == "tensor":
        rows = []
        tensor_router._record_image_pair(rows, albucore, op_name, "HWC", shape, half, wide, builder, 1, 0)
    else:
        rows = router._bench_pair_rows(albucore, op_name, "HWC", shape, builder, half, wide, 1, 0)

    assert [row.status for row in rows] == ["skip", "skip"]
    assert [row.dtype for row in rows] == ["float16", "float32"]
    assert all(row.ms_median is None and "identity" in row.detail for row in rows)


@pytest.mark.parametrize("op_name", ["convert_dtype_float16", "convert_dtype_float32"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_fixed_float_targets_keep_standalone_measurements(conversion_benchmarks, op_name, dtype):
    router, _ = conversion_benchmarks
    image = np.ones((3, 5, 1), dtype=dtype)
    builder = dict(router._registry_functions())[op_name]

    timing, status, detail = router._bench(albucore, image, builder, 1, 0)

    assert status == "ok" and timing is not None and detail == ""


def test_uint8_target_keeps_equivalent_float_pair(conversion_benchmarks):
    router, _ = conversion_benchmarks
    shape = (3, 5, 1)
    half = np.linspace(0, 1, np.prod(shape), dtype=np.float16).reshape(shape)
    builder = dict(router._registry_functions())["convert_dtype_uint8"]

    rows = router._bench_pair_rows(
        albucore, "convert_dtype_uint8", "HWC", shape, builder, half, half.astype(np.float32), 1, 0
    )

    assert all(row.status == "ok" and row.ms_median is not None for row in rows)
