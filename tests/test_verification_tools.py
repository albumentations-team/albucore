from __future__ import annotations

import hashlib
import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from tests.router_contracts import ROUTER_CONTRACTS
from tools import (
    check_benchmark_regressions,
    check_router_contracts,
    ci_matrix,
    generate_release_summary,
    resolve_previous_pypi_release,
    summarize_benchmarks,
    validate_release_candidate,
    verify_golden_vectors,
    verify_publish_artifacts,
)
from tools.golden_vectors import array_metadata


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


def test_ci_matrix_missing_workflow_error_is_not_duplicated(monkeypatch) -> None:
    missing_workflow = ci_matrix.REPO_ROOT / ".github" / "workflows" / "__missing__.yml"

    monkeypatch.setattr(ci_matrix, "RELEASE_CANDIDATE_WORKFLOW", missing_workflow)

    errors = ci_matrix.check()

    assert errors.count(f"Required workflow {missing_workflow.relative_to(ci_matrix.REPO_ROOT)} is missing") == 1


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


def test_benchmark_regression_check_prints_summary_and_writes_report(tmp_path, monkeypatch, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    report = tmp_path / "report.md"
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
            "--output-md",
            str(report),
        ],
    )

    assert check_benchmark_regressions.main() == 1
    assert "Benchmark Regression Check" in report.read_text()
    captured = capsys.readouterr()
    assert "Benchmark regression summary" in captured.out
    assert "normalize HWC 128x160x3 float32" in captured.out


def test_benchmark_regression_check_accepts_explicit_release_regression(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    accepted = tmp_path / "accepted.json"
    report = tmp_path / "report.md"
    baseline.write_text(json.dumps(_router_json(1.0)))
    current.write_text(json.dumps(_router_json(1.2)))
    accepted.write_text(
        json.dumps(
            {
                "regressions": [
                    {
                        "op": "normalize",
                        "layout": "HWC",
                        "shape": [128, 160, 3],
                        "dtype": "float32",
                        "reason": "intentional | correctness fix",
                        "approved_by": "maintainer | release owner",
                    },
                ],
            },
        ),
    )

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
            "--accepted-regressions",
            str(accepted),
            "--output-md",
            str(report),
        ],
    )

    assert check_benchmark_regressions.main() == 0
    report_text = report.read_text()
    assert "`accepted`" in report_text
    assert r"`intentional \| correctness fix`" in report_text
    assert r"`maintainer \| release owner`" in report_text


def test_benchmark_regression_check_rejects_undocumented_accepted_regression(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    accepted = tmp_path / "accepted.json"
    baseline.write_text(json.dumps(_router_json(1.0)))
    current.write_text(json.dumps(_router_json(1.2)))
    accepted.write_text(
        json.dumps(
            {
                "regressions": [
                    {
                        "op": "normalize",
                        "layout": "HWC",
                        "shape": [128, 160, 3],
                        "dtype": "float32",
                    },
                ],
            },
        ),
    )

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
            "--accepted-regressions",
            str(accepted),
        ],
    )

    with pytest.raises(TypeError, match="reason"):
        check_benchmark_regressions.main()


def test_benchmark_regression_check_rejects_duplicate_accepted_regression_key(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    accepted = tmp_path / "accepted.json"
    baseline.write_text(json.dumps(_router_json(1.0)))
    current.write_text(json.dumps(_router_json(1.2)))
    accepted.write_text(
        json.dumps(
            {
                "regressions": [
                    {
                        "op": "normalize",
                        "layout": "HWC",
                        "shape": [128, 160, 3],
                        "dtype": "float32",
                        "reason": "first approval",
                        "approved_by": "maintainer-a",
                    },
                    {
                        "op": "normalize",
                        "layout": "HWC",
                        "shape": [128, 160, 3],
                        "dtype": "float32",
                        "reason": "second approval",
                        "approved_by": "maintainer-b",
                    },
                ],
            },
        ),
    )

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
            "--accepted-regressions",
            str(accepted),
        ],
    )

    with pytest.raises(ValueError, match="duplicate accepted regression keys"):
        check_benchmark_regressions.main()


def test_benchmark_blocking_ops_come_from_router_contracts() -> None:
    expected = {
        benchmark_name
        for contract in ROUTER_CONTRACTS.values()
        if contract.release_blocking_performance
        for benchmark_name in contract.benchmark_names
    }

    assert check_benchmark_regressions.RELEASE_BLOCKING_OPS == expected


def test_release_candidate_ci_run_validator_requires_success() -> None:
    runs = [
        {
            "status": "completed",
            "conclusion": "failure",
            "url": "https://example.invalid/failure",
        },
        {
            "status": "completed",
            "conclusion": "success",
            "url": "https://example.invalid/success",
        },
    ]

    assert validate_release_candidate.validate_ci_runs(runs) == "https://example.invalid/success"

    with pytest.raises(ValueError, match="No successful CI workflow run"):
        validate_release_candidate.validate_ci_runs(runs[:1])


def test_publish_artifact_verifier_checks_provenance_files_and_checksums(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    version = "0.2.2"
    commit_sha = "abc123"
    candidate_run = dist / "candidate-run.json"
    candidate_run.write_text(
        json.dumps(
            {
                "status": "completed",
                "conclusion": "success",
                "workflowName": "Release Candidate",
                "headSha": commit_sha,
            },
        ),
    )
    (dist / "release-candidate-metadata.json").write_text(
        json.dumps(
            {
                "version": version,
                "commit_sha": commit_sha,
            },
        ),
    )
    files = [
        dist / f"albucore-{version}.tar.gz",
        dist / f"albucore-{version}-py3-none-any.whl",
        dist / "release-candidate-metadata.json",
    ]
    for path in files:
        if not path.exists():
            path.write_text(path.name)
    (dist / "SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.name}\n" for path in files),
    )

    verify_publish_artifacts.verify_prepublish_artifacts(dist, candidate_run, version, commit_sha)

    pypi_dist = tmp_path / "pypi-dist"
    verify_publish_artifacts.copy_distribution_files(dist, pypi_dist, version)
    assert sorted(path.name for path in pypi_dist.iterdir()) == [
        f"albucore-{version}-py3-none-any.whl",
        f"albucore-{version}.tar.gz",
    ]

    (dist / f"albucore-{version}.tar.gz").write_text("changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_publish_artifacts.verify_checksums(dist)


def test_publish_artifact_verifier_rejects_candidate_run_mismatches() -> None:
    valid = {
        "status": "completed",
        "conclusion": "success",
        "workflowName": "Release Candidate",
        "headSha": "abc123",
    }

    with pytest.raises(ValueError, match="did not succeed"):
        verify_publish_artifacts.verify_candidate_run({**valid, "conclusion": "failure"}, "abc123")
    with pytest.raises(ValueError, match="expected 'Release Candidate'"):
        verify_publish_artifacts.verify_candidate_run({**valid, "workflowName": "Other"}, "abc123")
    with pytest.raises(ValueError, match="does not match abc123"):
        verify_publish_artifacts.verify_candidate_run({**valid, "headSha": "deadbeef"}, "abc123")


def test_publish_artifact_verifier_rejects_release_metadata_mismatches() -> None:
    metadata = {"version": "0.2.2", "commit_sha": "abc123"}

    with pytest.raises(ValueError, match="version"):
        verify_publish_artifacts.verify_release_metadata({**metadata, "version": "0.2.3"}, "0.2.2", "abc123")
    with pytest.raises(ValueError, match="SHA"):
        verify_publish_artifacts.verify_release_metadata({**metadata, "commit_sha": "deadbeef"}, "0.2.2", "abc123")


def test_publish_artifact_verifier_rejects_missing_and_malformed_checksums(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()

    with pytest.raises(FileNotFoundError, match="SHA256SUMS"):
        verify_publish_artifacts.verify_checksums(dist)

    checksums = dist / "SHA256SUMS.txt"
    checksums.write_text("0" * 64 + "missing-separator")
    with pytest.raises(ValueError, match="Malformed checksum line"):
        verify_publish_artifacts.verify_checksums(dist)

    checksums.write_text("0" * 10 + "  ./artifact.whl\n")
    with pytest.raises(ValueError, match="Malformed sha256 digest"):
        verify_publish_artifacts.verify_checksums(dist)


class _FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, status: int = 200) -> None:
        self.payload = payload or {}
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_publish_artifact_verifier_rejects_existing_pypi_version() -> None:
    def existing_version(url: str, *, timeout: float) -> _FakeResponse:
        return _FakeResponse(status=200)

    with pytest.raises(ValueError, match="already has"):
        verify_publish_artifacts.verify_pypi_absent("albucore", "0.2.2", urlopen=existing_version)


def test_publish_artifact_verifier_accepts_missing_pypi_version() -> None:
    def missing_version(url: str, *, timeout: float) -> _FakeResponse:
        raise urllib.error.HTTPError(url, 404, "missing", hdrs=None, fp=None)

    verify_publish_artifacts.verify_pypi_absent("albucore", "0.2.2", urlopen=missing_version)


def test_publish_artifact_verifier_waits_for_pypi_files() -> None:
    def published_version(url: str, *, timeout: float) -> _FakeResponse:
        return _FakeResponse(payload={"urls": [{"filename": "albucore-0.2.2.tar.gz"}]})

    file_count = verify_publish_artifacts.verify_pypi_publication(
        "albucore",
        "0.2.2",
        verify_publish_artifacts.PyPIWaitConfig(
            attempts=1,
            sleep_seconds=0.0,
            urlopen=published_version,
        ),
    )

    assert file_count == 1


def test_publish_artifact_verifier_requires_pypi_urls_list() -> None:
    with pytest.raises(TypeError, match="urls"):
        verify_publish_artifacts.pypi_file_count({"urls": {"filename": "albucore-0.2.2.tar.gz"}})


def test_publish_artifact_verifier_rejects_pypi_publication_without_files() -> None:
    def empty_version(url: str, *, timeout: float) -> _FakeResponse:
        return _FakeResponse(payload={"urls": []})

    with pytest.raises(ValueError, match="metadata but no files"):
        verify_publish_artifacts.verify_pypi_publication(
            "albucore",
            "0.2.2",
            verify_publish_artifacts.PyPIWaitConfig(
                attempts=1,
                sleep_seconds=0.0,
                urlopen=empty_version,
            ),
        )


def test_benchmark_summary_reads_router_json(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "current.json"
    path.write_text(json.dumps(_router_json(1.0)))

    monkeypatch.setattr("sys.argv", ["summarize_benchmarks.py", str(path)])

    assert summarize_benchmarks.main() == 0
    captured = capsys.readouterr()
    assert "Benchmark Summary" in captured.out
    assert "`normalize`" in captured.out


def test_release_summary_omits_release_benchmark_artifacts(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()

    summary = generate_release_summary.generate_summary("0.2.2", dist)

    assert "Router benchmark" not in summary
    assert "memory-smoke" not in summary


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
