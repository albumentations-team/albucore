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
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "upload_to_pypi.yml"
SECURITY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "security.yml"
SUPPORT_POLICY = REPO_ROOT / "docs" / "maintaining" / "support-policy.md"


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


def _check_release_workflow(errors: list[str]) -> None:
    text = RELEASE_WORKFLOW.read_text()
    required_fragments = {
        "previous release resolver": "PREVIOUS_RELEASE_VERSION",
        "baseline benchmark artifact": "dist/router-baseline.json",
        "release regression checker": "tools/check_benchmark_regressions.py",
        "release regression mode": "--mode release",
        "regression report artifact": "dist/router-release-regressions.md",
        "project-free runtime dependency export": "uv export --frozen --no-dev --no-emit-project",
    }
    errors.extend(
        f"Release workflow is missing {label}" for label, fragment in required_fragments.items() if fragment not in text
    )


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
