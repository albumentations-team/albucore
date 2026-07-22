"""Classify whether a commit range only changes Albucore package versions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

VERSION_FILES: Final = frozenset({"pyproject.toml", "uv.lock"})
PROJECT_NAME: Final = "albucore"


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        text=True,
    ).strip()


def _toml_at(repo: Path, revision: str, path: str) -> dict[str, Any]:
    payload = tomllib.loads(_git_output(repo, "show", f"{revision}:{path}"))
    if not isinstance(payload, dict):
        msg = f"{path} at {revision} did not parse as a TOML table."
        raise TypeError(msg)
    return payload


def _without_project_version(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    project = payload.get("project")
    if not isinstance(project, dict):
        msg = "pyproject.toml is missing [project]."
        raise TypeError(msg)

    version = project.pop("version", None)
    if not isinstance(version, str):
        msg = "pyproject.toml project.version must be a string."
        raise TypeError(msg)
    return version, payload


def _without_lock_version(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    packages = payload.get("package")
    if not isinstance(packages, list):
        msg = "uv.lock is missing a package list."
        raise TypeError(msg)

    matching_packages = [
        package for package in packages if isinstance(package, dict) and package.get("name") == PROJECT_NAME
    ]
    if len(matching_packages) != 1:
        msg = f"uv.lock must contain exactly one {PROJECT_NAME!r} package."
        raise ValueError(msg)

    version = matching_packages[0].pop("version", None)
    if not isinstance(version, str):
        msg = f"uv.lock package {PROJECT_NAME!r} version must be a string."
        raise TypeError(msg)
    return version, payload


def is_version_only_change(repo: Path, base_revision: str, head_revision: str) -> bool:
    """Return true only when both package metadata files change version and nothing else."""
    try:
        changed_files = set(_git_output(repo, "diff", "--name-only", base_revision, head_revision, "--").splitlines())
        if changed_files != VERSION_FILES:
            return False

        base_project_version, base_project = _without_project_version(
            _toml_at(repo, base_revision, "pyproject.toml"),
        )
        head_project_version, head_project = _without_project_version(
            _toml_at(repo, head_revision, "pyproject.toml"),
        )
        base_lock_version, base_lock = _without_lock_version(_toml_at(repo, base_revision, "uv.lock"))
        head_lock_version, head_lock = _without_lock_version(_toml_at(repo, head_revision, "uv.lock"))
    except (subprocess.CalledProcessError, tomllib.TOMLDecodeError, TypeError, ValueError):
        return False

    return (
        base_project_version == base_lock_version
        and head_project_version == head_lock_version
        and base_project_version != head_project_version
        and base_project == head_project
        and base_lock == head_lock
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_revision")
    parser.add_argument("head_revision")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    version_only = is_version_only_change(args.repo, args.base_revision, args.head_revision)
    sys.stdout.write(f"run_tests={str(not version_only).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
