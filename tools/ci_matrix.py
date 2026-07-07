"""Validate support policy, package metadata, and CI matrix consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_PR_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "benchmark-pr.yml"
PERFORMANCE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "performance.yml"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"
RELEASE_CANDIDATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-candidate.yml"
SECURITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security.yml"
SUPPORT_POLICY = REPO_ROOT / "docs" / "maintaining" / "support-policy.md"
VALIDATE_RELEASE_CANDIDATE_TOOL = REPO_ROOT / "tools" / "validate_release_candidate.py"
VERIFY_PUBLISH_ARTIFACTS_TOOL = REPO_ROOT / "tools" / "verify_publish_artifacts.py"


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text())


def _classifier_python_versions(classifiers: list[str]) -> set[str]:
    pattern = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")
    versions: set[str] = set()
    for classifier in classifiers:
        match = pattern.match(classifier)
        if match is not None:
            versions.add(match.group(1))
    return versions


def _check_pyproject(errors: list[str]) -> set[str]:
    pyproject = _load_pyproject()
    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        errors.append("pyproject.toml is missing [project]")
        return set()

    requires_python = project.get("requires-python")
    if requires_python != ">=3.10":
        errors.append(f"Expected requires-python >=3.10, found {requires_python!r}")

    classifiers = project.get("classifiers", [])
    if not isinstance(classifiers, list) or not all(isinstance(item, str) for item in classifiers):
        errors.append("project.classifiers must be a list of strings")
        return set()

    versions = _classifier_python_versions(classifiers)
    expected = {"3.10", "3.11", "3.12", "3.13", "3.14"}
    if versions != expected:
        errors.append(f"Expected Python classifiers {sorted(expected)}, found {sorted(versions)}")

    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict) or "headless" not in optional_dependencies:
        errors.append("pyproject.toml must define project.optional-dependencies.headless")

    return versions


def _check_ci(errors: list[str], versions: set[str]) -> None:
    text = CI_WORKFLOW.read_text()
    errors.extend(
        f"CI matrix does not mention Python {version}"
        for version in sorted(versions)
        if f'"{version}"' not in text and f"'{version}'" not in text
    )
    if "declared-dependency-ranges" not in text:
        errors.append("CI workflow is missing declared-dependency-ranges job")
    if "tools/check_router_contracts.py" not in text:
        errors.append("CI workflow does not run router contract check")
    if "permissions:" not in text or "contents: read" not in text:
        errors.append("CI workflow must declare minimal GITHUB_TOKEN permissions")


def _check_support_policy(errors: list[str], versions: set[str]) -> None:
    text = SUPPORT_POLICY.read_text()
    errors.extend(
        f"Support policy does not mention Python {version}" for version in sorted(versions) if version not in text
    )
    if "opencv-python-headless" not in text:
        errors.append("Support policy does not name opencv-python-headless")
    if "(H, W, 1)" not in text:
        errors.append("Support policy does not state explicit grayscale channel convention")


def _check_file_fragments(errors: list[str], path: Path, required_fragments: dict[str, str]) -> None:
    if not path.exists():
        errors.append(f"Required workflow {path.relative_to(REPO_ROOT)} is missing")
        return
    text = path.read_text()
    errors.extend(
        f"{path.relative_to(REPO_ROOT)} is missing {label}"
        for label, fragment in required_fragments.items()
        if fragment not in text
    )


def _check_release_workflows(errors: list[str]) -> None:
    _check_file_fragments(
        errors,
        BENCHMARK_PR_WORKFLOW,
        {
            "PR trigger": "pull_request:",
            "PR router benchmark": "benchmarks/benchmark_router_synthetic.py",
            "PR advisory regression check": "--mode advisory",
            "PR benchmark artifacts": "pr-router-benchmark-results",
        },
    )
    _check_file_fragments(
        errors,
        PERFORMANCE_WORKFLOW,
        {
            "previous PyPI release resolver": "tools/resolve_previous_pypi_release.py",
            "package version resolver": "tools/validate_release_candidate.py package-version",
            "release benchmark metadata writer": "tools/validate_release_candidate.py benchmark-metadata",
            "benchmark artifact upload on failure": "if: always()",
            "baseline benchmark artifact": "benchmarks/results/router-baseline.json",
            "release regression mode": "--mode release",
            "reusable benchmark evidence": "release-benchmark-evidence",
        },
    )
    _check_file_fragments(
        errors,
        RELEASE_CANDIDATE_WORKFLOW,
        {
            "manual release candidate trigger": "workflow_dispatch:",
            "exact commit input": "commit_sha:",
            "release metadata validator": "tools/validate_release_candidate.py metadata",
            "candidate CI success check": "Verify CI workflow succeeded for candidate",
            "candidate CI validator": "tools/validate_release_candidate.py ci-runs",
            "previous PyPI release resolver": "tools/resolve_previous_pypi_release.py",
            "baseline benchmark artifact": "dist/router-baseline.json",
            "release regression checker": "tools/check_benchmark_regressions.py",
            "release regression mode": "--mode release",
            "accepted regression support": "--accepted-regressions",
            "reusable benchmark run input": "benchmark_run_id:",
            "reusable benchmark evidence validator": "tools/validate_release_candidate.py benchmark-evidence",
            "regression report artifact": "dist/router-release-regressions.md",
            "release validation headless extra": "uv sync --frozen --extra headless --group dev",
            "project-free runtime dependency export": "uv export --frozen --no-dev --no-emit-project",
            "candidate metadata writer": "tools/validate_release_candidate.py candidate-metadata",
            "release candidate artifact upload": "release-candidate-artifacts",
        },
    )
    _check_file_fragments(
        errors,
        PUBLISH_WORKFLOW,
        {
            "manual publish trigger": "workflow_dispatch:",
            "candidate run input": "candidate_run_id:",
            "candidate artifact download": "release-candidate-artifacts",
            "prepublish verifier": "tools/verify_publish_artifacts.py prepublish",
            "PyPI distribution staging": "tools/verify_publish_artifacts.py prepare-pypi-dist",
            "PyPI publication verifier": "tools/verify_publish_artifacts.py publication",
            "trusted publishing": "pypa/gh-action-pypi-publish",
            "GitHub Release only after PyPI": "Create or update GitHub Release",
        },
    )
    _check_file_fragments(
        errors,
        VALIDATE_RELEASE_CANDIDATE_TOOL,
        {
            "release metadata validator": "validate_release_environment",
            "candidate CI run validator": "validate_ci_runs",
            "reusable benchmark evidence validator": "validate_reusable_benchmark_evidence",
            "release benchmark metadata writer": "write_benchmark_metadata",
        },
    )
    _check_file_fragments(
        errors,
        VERIFY_PUBLISH_ARTIFACTS_TOOL,
        {
            "candidate run validator": "verify_candidate_run",
            "checksum verifier": "verify_checksums",
            "distribution staging": "copy_distribution_files",
            "PyPI existing-version guard": "verify_pypi_absent",
            "PyPI publication verifier": "verify_pypi_publication",
        },
    )

    github_release_published_workflows = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / ".github" / "workflows").glob("*.yml")
        if "release:" in path.read_text(errors="ignore") and "published" in path.read_text(errors="ignore")
    ]
    if github_release_published_workflows:
        errors.append(
            f"GitHub Release published-trigger workflows must not exist: {github_release_published_workflows}",
        )


def _check_release_docs(errors: list[str]) -> None:
    process = (REPO_ROOT / "docs" / "maintaining" / "release-process.md").read_text()
    required_fragments = {
        "release candidate workflow": "release-candidate.yml",
        "publish workflow": "publish.yml",
        "PyPI-before-GitHub-Release ordering": "Publish to PyPI before creating or publishing the GitHub Release",
    }
    errors.extend(
        f"release-process.md is missing {label}"
        for label, fragment in required_fragments.items()
        if fragment not in process
    )


def _check_release_workflows_and_docs(errors: list[str]) -> None:
    _check_release_workflows(errors)
    _check_release_docs(errors)


def _check_release_workflow(errors: list[str]) -> None:
    _check_release_workflows_and_docs(errors)


def _check_security_workflow(errors: list[str]) -> None:
    text = SECURITY_WORKFLOW.read_text()
    if "uv export --frozen --no-dev --no-emit-project" not in text:
        errors.append("Security workflow runtime audit must omit the editable project from exported requirements")
    if "uv export --frozen --no-emit-project" not in text:
        errors.append("Security workflow dev audit must omit the editable project from exported requirements")


def check() -> list[str]:
    """Return support-matrix consistency errors."""
    errors: list[str] = []
    versions = _check_pyproject(errors)
    if versions:
        _check_ci(errors, versions)
        _check_support_policy(errors, versions)
        _check_release_workflow(errors)
        _check_security_workflow(errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",), nargs="?", default="check")
    parser.parse_args()

    errors = check()
    if errors:
        sys.stderr.write("CI matrix check failed:\n")
        for error in errors:
            sys.stderr.write(f"- {error}\n")
        return 1

    sys.stdout.write("CI matrix check passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
