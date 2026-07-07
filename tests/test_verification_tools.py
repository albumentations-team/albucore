from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import (
    check_benchmark_regressions,
    check_router_contracts,
    ci_matrix,
    resolve_previous_pypi_release,
    summarize_benchmarks,
    verify_golden_vectors,
)
from tools.golden_vectors import array_metadata
from tests.router_contracts import ROUTER_CONTRACTS


def _router_json(ms: float) -> dict[str, object]:
    return {
        "meta": {
            "albucore_version": "test",
            "distribution_version": "test",
            "python": "3.12",
            "platform": "test",
            "quick": True,
            "with_geometric": False,
        },
        "rows": [
            {
                "op": "normalize",
                "layout": "HWC",
                "shape": [128, 160, 3],
                "dtype": "float32",
                "ms_median": ms,
                "status": "ok",
                "detail": "",
                "ms_mean": ms,
                "ms_std": 0.0,
                "ms_mad": 0.0,
                "timing_n": 3,
            },
        ],
    }


def test_router_contract_check_passes() -> None:
    assert check_router_contracts.main() == 0


def test_ci_matrix_check_passes() -> None:
    assert ci_matrix.check() == []


def test_benchmark_regression_check_blocks_release(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_router_json(1.0)))
    current.write_text(json.dumps(_router_json(1.2)))

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_benchmark_regressions.py",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--mode",
            "release",
        ],
    )

    assert check_benchmark_regressions.main() == 1


def test_benchmark_blocking_ops_come_from_router_contracts() -> None:
    expected = {
        benchmark_name
        for contract in ROUTER_CONTRACTS.values()
        if contract.release_blocking_performance
        for benchmark_name in contract.benchmark_names
    }

    assert check_benchmark_regressions.RELEASE_BLOCKING_OPS == expected


def test_benchmark_summary_reads_router_json(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "current.json"
    path.write_text(json.dumps(_router_json(1.0)))

    monkeypatch.setattr("sys.argv", ["summarize_benchmarks.py", str(path)])

    assert summarize_benchmarks.main() == 0
    captured = capsys.readouterr()
    assert "Benchmark Summary" in captured.out
    assert "`normalize`" in captured.out


def test_previous_pypi_release_skips_unpublished_release() -> None:
    releases = {
        "0.1.5": [{"filename": "albucore-0.1.5.tar.gz", "yanked": False}],
        "0.1.6": [{"filename": "albucore-0.1.6.tar.gz", "yanked": False}],
        "0.2.0": [],
        "0.2.1": [{"filename": "albucore-0.2.1.tar.gz", "yanked": False}],
    }

    previous = resolve_previous_pypi_release.previous_published_version("0.2.1", releases)

    assert previous == "0.1.6"


def test_previous_pypi_release_cli_writes_github_env(tmp_path, monkeypatch, capsys) -> None:
    env_path = tmp_path / "github-env"
    releases = {
        "0.1.6": [{"filename": "albucore-0.1.6.tar.gz", "yanked": False}],
        "0.2.0": [],
    }

    monkeypatch.setattr(resolve_previous_pypi_release, "fetch_pypi_releases", lambda project: releases)
    monkeypatch.setattr(
        "sys.argv",
        [
            "resolve_previous_pypi_release.py",
            "--current-version",
            "0.2.1",
            "--github-env",
            str(env_path),
        ],
    )

    assert resolve_previous_pypi_release.main() == 0
    assert env_path.read_text() == "PREVIOUS_RELEASE_VERSION=0.1.6\n"
    assert "PREVIOUS_RELEASE_VERSION=0.1.6" in capsys.readouterr().out


def test_pypi_release_payload_must_be_json_object() -> None:
    with pytest.raises(TypeError, match="not a JSON object"):
        resolve_previous_pypi_release.pypi_releases_from_payload("albucore", [])


def test_golden_vector_verify_checks_computed_dtype(tmp_path, monkeypatch) -> None:
    expected = np.array([1.0, 2.0], dtype=np.float32)
    manifest_path = tmp_path / "manifest.json"
    outputs_path = tmp_path / "outputs.npz"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "dtype_case",
                        "outputs": {"result": array_metadata(expected)},
                    },
                ],
            },
        ),
    )
    np.savez(outputs_path, dtype_case__result=expected)
    case = SimpleNamespace(
        name="dtype_case",
        compute=lambda: {"result": expected.astype(np.float64)},
        rtol=0.0,
        atol=0.0,
    )

    monkeypatch.setattr(verify_golden_vectors, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(verify_golden_vectors, "OUTPUTS_PATH", outputs_path)
    monkeypatch.setattr(verify_golden_vectors, "GOLDEN_CASES", (case,))

    errors = verify_golden_vectors.verify()

    assert any("dtype_case.result.computed: metadata dtype mismatch" in error for error in errors)
