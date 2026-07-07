"""Validate release-candidate metadata and benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PROJECT_NAME: Final = "albucore"
BENCHMARK_HASH_FILES: Final = (
    "benchmarks/benchmark_router_synthetic.py",
    "tools/check_benchmark_regressions.py",
    "uv.lock",
)
BENCHMARK_EVIDENCE_DESTINATIONS: Final = {
    "router-current.json": "router-release.json",
    "router-baseline.json": "router-baseline.json",
    "REPORT_router_regressions.md": "router-release-regressions.md",
    "REPORT_router_current_summary.md": "router-release-summary.md",
    "memory-smoke.json": "memory-smoke.json",
    "release-benchmark-metadata.json": "release-benchmark-metadata.json",
}


@dataclass(frozen=True)
class GitHubRunContext:
    """GitHub Actions run metadata captured in release evidence."""

    workflow: str
    run_id: str
    run_attempt: str
    repository: str


@dataclass(frozen=True)
class ReleaseEnvironment:
    """Release metadata exported to later workflow steps."""

    version: str
    commit_sha: str
    sbom_filename: str
    headless_runtime_dependencies: tuple[str, ...]


def _load_toml(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text())
    if not isinstance(payload, dict):
        msg = f"{path} did not parse as a TOML table."
        raise TypeError(msg)
    return payload


def _project_table(pyproject_path: Path) -> dict[str, Any]:
    project = _load_toml(pyproject_path).get("project")
    if not isinstance(project, dict):
        msg = f"{pyproject_path} is missing [project]."
        raise TypeError(msg)
    return project


def load_pyproject_version(pyproject_path: Path) -> str:
    """Return the package version declared in pyproject.toml."""
    version = _project_table(pyproject_path).get("version")
    if not isinstance(version, str):
        msg = f"{pyproject_path} project.version must be a string."
        raise TypeError(msg)
    return version


def load_lock_version(lock_path: Path, package_name: str = PROJECT_NAME) -> str:
    """Return the package version recorded in uv.lock."""
    packages = _load_toml(lock_path).get("package")
    if not isinstance(packages, list):
        msg = f"{lock_path} is missing a package list."
        raise TypeError(msg)

    for package in packages:
        if isinstance(package, dict) and package.get("name") == package_name:
            version = package.get("version")
            if isinstance(version, str):
                return version
            msg = f"{lock_path} package {package_name!r} version must be a string."
            raise TypeError(msg)

    msg = f"{lock_path} does not contain package {package_name!r}."
    raise ValueError(msg)


def load_headless_runtime_dependencies(pyproject_path: Path) -> tuple[str, ...]:
    """Return project.optional-dependencies.headless from pyproject.toml."""
    optional_dependencies = _project_table(pyproject_path).get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        msg = f"{pyproject_path} project.optional-dependencies must be a table."
        raise TypeError(msg)

    headless_dependencies = optional_dependencies.get("headless", [])
    if not isinstance(headless_dependencies, list) or not all(
        isinstance(dependency, str) for dependency in headless_dependencies
    ):
        msg = f"{pyproject_path} project.optional-dependencies.headless must be an array of strings."
        raise TypeError(msg)
    return tuple(headless_dependencies)


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()  # noqa: S603,S607


def fetch_origin_main(repo_root: Path) -> None:
    """Fetch origin/main so ancestry can be checked in GitHub Actions."""
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_root), "fetch", "origin", "main:refs/remotes/origin/main", "--depth=1"],  # noqa: S607
        check=True,
    )


def require_origin_main_ancestor(repo_root: Path, commit_sha: str) -> None:
    """Fail unless commit_sha is reachable from origin/main."""
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit_sha, "origin/main"],  # noqa: S607
        check=True,
    )


def validate_release_environment(
    repo_root: Path,
    requested_version: str,
    requested_commit_sha: str,
) -> ReleaseEnvironment:
    """Validate local package metadata for a release candidate."""
    actual_sha = _git_output(repo_root, "rev-parse", "HEAD")
    if actual_sha != requested_commit_sha:
        msg = f"Checked out {actual_sha}, expected {requested_commit_sha}."
        raise ValueError(msg)

    pyproject_path = repo_root / "pyproject.toml"
    package_version = load_pyproject_version(pyproject_path)
    if package_version != requested_version:
        msg = f"Requested version {requested_version!r} does not match pyproject version {package_version!r}."
        raise ValueError(msg)

    lock_version = load_lock_version(repo_root / "uv.lock")
    if lock_version != requested_version:
        msg = f"Requested version {requested_version!r} does not match uv.lock {lock_version!r}."
        raise ValueError(msg)

    return ReleaseEnvironment(
        version=package_version,
        commit_sha=actual_sha,
        sbom_filename=f"{PROJECT_NAME}-{package_version}-sbom.cdx.json",
        headless_runtime_dependencies=load_headless_runtime_dependencies(pyproject_path),
    )


def release_environment_lines(environment: ReleaseEnvironment) -> list[str]:
    """Format release metadata for a GitHub Actions environment file."""
    return [
        f"PACKAGE_VERSION={environment.version}",
        f"CANDIDATE_COMMIT_SHA={environment.commit_sha}",
        f"SBOM_FILENAME={environment.sbom_filename}",
        "HEADLESS_RUNTIME_DEPENDENCIES<<EOF",
        *environment.headless_runtime_dependencies,
        "EOF",
    ]


def write_release_environment(path: Path, environment: ReleaseEnvironment) -> None:
    """Append release metadata to a GitHub Actions environment file."""
    with path.open("a") as file:
        file.write("\n".join(release_environment_lines(environment)) + "\n")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object."
        raise TypeError(msg)
    return payload


def validate_ci_runs(runs: object) -> str:
    """Return the first successful CI run URL, or fail if none exists."""
    if not isinstance(runs, list):
        msg = "CI run payload must be a JSON list."
        raise TypeError(msg)

    for run in runs:
        if isinstance(run, dict) and run.get("status") == "completed" and run.get("conclusion") == "success":
            url = run.get("url")
            return str(url) if url else "unknown URL"

    msg = (
        "No successful CI workflow run found for the release candidate commit. "
        "Run CI on main before validating a release candidate."
    )
    raise ValueError(msg)


def validate_benchmark_run(run: dict[str, Any], expected_sha: str) -> None:
    """Validate that a reusable benchmark run is a successful Performance run."""
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        msg = f"Benchmark run did not succeed: {run}."
        raise ValueError(msg)
    if run.get("workflowName") != "Performance":
        msg = f"Benchmark run used workflow {run.get('workflowName')!r}, expected 'Performance'."
        raise ValueError(msg)
    if run.get("headSha") != expected_sha:
        msg = f"Benchmark run SHA {run.get('headSha')} does not match candidate SHA {expected_sha}."
        raise ValueError(msg)


def validate_benchmark_metadata(
    metadata: dict[str, Any],
    expected_sha: str,
    expected_version: str,
    expected_baseline: str,
) -> None:
    """Validate reusable benchmark metadata against the release candidate."""
    if metadata.get("commit_sha") != expected_sha:
        msg = "Benchmark metadata commit does not match candidate SHA."
        raise ValueError(msg)
    if metadata.get("version") != expected_version:
        msg = "Benchmark metadata version does not match package version."
        raise ValueError(msg)
    if metadata.get("baseline_version") != expected_baseline:
        msg = "Benchmark metadata baseline does not match previous PyPI version."
        raise ValueError(msg)


def _evidence_files(evidence_dir: Path) -> dict[str, Path]:
    evidence_files: dict[str, Path] = {}
    for path in sorted((candidate for candidate in evidence_dir.rglob("*") if candidate.is_file()), key=str):
        existing = evidence_files.get(path.name)
        if existing is not None:
            msg = (
                f"Reusable benchmark evidence contains duplicate basename {path.name!r}: "
                f"{existing.relative_to(evidence_dir)} and {path.relative_to(evidence_dir)}."
            )
            raise ValueError(msg)
        evidence_files[path.name] = path
    return evidence_files


def validate_reusable_benchmark_evidence(
    evidence_dir: Path,
    dist_dir: Path,
    expected_sha: str,
    expected_version: str,
    expected_baseline: str,
) -> None:
    """Validate and copy reusable Performance workflow benchmark evidence."""
    evidence_files = _evidence_files(evidence_dir)
    missing = sorted(filename for filename in BENCHMARK_EVIDENCE_DESTINATIONS if filename not in evidence_files)
    if missing:
        msg = f"Reusable benchmark evidence is missing files: {missing}."
        raise FileNotFoundError(msg)

    validate_benchmark_run(_load_json_object(evidence_dir / "benchmark-run.json"), expected_sha)
    validate_benchmark_metadata(
        _load_json_object(evidence_files["release-benchmark-metadata.json"]),
        expected_sha,
        expected_version,
        expected_baseline,
    )

    dist_dir.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in BENCHMARK_EVIDENCE_DESTINATIONS.items():
        shutil.copy2(evidence_files[source_name], dist_dir / destination_name)


def github_context_from_env() -> GitHubRunContext:
    """Build release evidence context from GitHub Actions environment variables."""
    return GitHubRunContext(
        workflow=required_env("GITHUB_WORKFLOW"),
        run_id=required_env("GITHUB_RUN_ID"),
        run_attempt=required_env("GITHUB_RUN_ATTEMPT"),
        repository=required_env("GITHUB_REPOSITORY"),
    )


def required_env(name: str) -> str:
    """Return a required environment variable."""
    value = os.environ.get(name)
    if not value:
        msg = f"Required environment variable {name} is not set."
        raise ValueError(msg)
    return value


def candidate_metadata(
    version: str,
    commit_sha: str,
    previous_pypi_version: str,
    context: GitHubRunContext,
) -> dict[str, str]:
    """Build release-candidate metadata."""
    return {
        "version": version,
        "commit_sha": commit_sha,
        "previous_pypi_version": previous_pypi_version,
        "workflow": context.workflow,
        "run_id": context.run_id,
        "run_attempt": context.run_attempt,
        "repository": context.repository,
    }


def write_candidate_metadata(
    dist_dir: Path,
    version: str,
    commit_sha: str,
    previous_pypi_version: str,
    context: GitHubRunContext,
) -> None:
    """Write release-candidate metadata to dist."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    payload = candidate_metadata(version, commit_sha, previous_pypi_version, context)
    (dist_dir / "release-candidate-metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def _file_hash(repo_root: Path, relative_path: str) -> str:
    return hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest()


def benchmark_metadata(
    repo_root: Path,
    version: str,
    commit_sha: str,
    baseline_version: str,
    context: GitHubRunContext,
) -> dict[str, object]:
    """Build reusable release benchmark metadata."""
    return {
        "version": version,
        "commit_sha": commit_sha,
        "baseline_version": baseline_version,
        "workflow": context.workflow,
        "run_id": context.run_id,
        "run_attempt": context.run_attempt,
        "repository": context.repository,
        "hashes": {filename: _file_hash(repo_root, filename) for filename in BENCHMARK_HASH_FILES},
    }


def write_benchmark_metadata(
    output_json: Path,
    repo_root: Path,
    commit_sha: str,
    baseline_version: str,
    context: GitHubRunContext,
) -> None:
    """Write reusable release benchmark metadata."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    version = load_pyproject_version(repo_root / "pyproject.toml")
    payload = benchmark_metadata(repo_root, version, commit_sha, baseline_version, context)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")


def _metadata_command(args: argparse.Namespace) -> None:
    environment = validate_release_environment(args.repo_root, args.version, args.commit_sha)
    if args.fetch_origin_main:
        fetch_origin_main(args.repo_root)
    if args.require_origin_main:
        require_origin_main_ancestor(args.repo_root, environment.commit_sha)
    if args.github_env is not None:
        write_release_environment(args.github_env, environment)
    else:
        sys.stdout.write("\n".join(release_environment_lines(environment)) + "\n")


def _ci_runs_command(args: argparse.Namespace) -> None:
    url = validate_ci_runs(_load_json(args.runs_json))
    sys.stdout.write(f"Found successful CI run: {url}\n")


def _benchmark_evidence_command(args: argparse.Namespace) -> None:
    validate_reusable_benchmark_evidence(
        args.evidence_dir,
        args.dist_dir,
        args.commit_sha,
        args.version,
        args.baseline_version,
    )


def _candidate_metadata_command(args: argparse.Namespace) -> None:
    write_candidate_metadata(
        args.dist_dir,
        args.version,
        args.commit_sha,
        args.previous_pypi_version,
        github_context_from_env(),
    )


def _benchmark_metadata_command(args: argparse.Namespace) -> None:
    write_benchmark_metadata(
        args.output_json,
        args.repo_root,
        args.commit_sha,
        args.baseline_version,
        github_context_from_env(),
    )


def _package_version_command(args: argparse.Namespace) -> None:
    sys.stdout.write(load_pyproject_version(args.pyproject) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--repo-root", type=Path, default=Path())
    metadata.add_argument("--version", required=True)
    metadata.add_argument("--commit-sha", required=True)
    metadata.add_argument("--github-env", type=Path)
    metadata.add_argument("--fetch-origin-main", action="store_true")
    metadata.add_argument("--require-origin-main", action="store_true")
    metadata.set_defaults(func=_metadata_command)

    ci_runs = subparsers.add_parser("ci-runs")
    ci_runs.add_argument("--runs-json", type=Path, required=True)
    ci_runs.set_defaults(func=_ci_runs_command)

    benchmark_evidence = subparsers.add_parser("benchmark-evidence")
    benchmark_evidence.add_argument("--evidence-dir", type=Path, default=Path("benchmark-evidence"))
    benchmark_evidence.add_argument("--dist-dir", type=Path, default=Path("dist"))
    benchmark_evidence.add_argument("--commit-sha", required=True)
    benchmark_evidence.add_argument("--version", required=True)
    benchmark_evidence.add_argument("--baseline-version", required=True)
    benchmark_evidence.set_defaults(func=_benchmark_evidence_command)

    candidate = subparsers.add_parser("candidate-metadata")
    candidate.add_argument("--dist-dir", type=Path, default=Path("dist"))
    candidate.add_argument("--version", required=True)
    candidate.add_argument("--commit-sha", required=True)
    candidate.add_argument("--previous-pypi-version", required=True)
    candidate.set_defaults(func=_candidate_metadata_command)

    benchmark = subparsers.add_parser("benchmark-metadata")
    benchmark.add_argument("--repo-root", type=Path, default=Path())
    benchmark.add_argument("--output-json", type=Path, required=True)
    benchmark.add_argument("--commit-sha", required=True)
    benchmark.add_argument("--baseline-version", required=True)
    benchmark.set_defaults(func=_benchmark_metadata_command)

    package_version = subparsers.add_parser("package-version")
    package_version.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    package_version.set_defaults(func=_package_version_command)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Release candidate validation failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
