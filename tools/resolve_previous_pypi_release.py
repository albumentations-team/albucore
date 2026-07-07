"""Resolve the previous published PyPI release for release benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_NAME = "albucore"
PYPI_JSON_URL = "https://pypi.org/pypi/{project}/json"
_FINAL_RELEASE_RE = re.compile(r"^\d+(?:\.\d+)*$")


def final_release_key(version: str) -> tuple[int, ...] | None:
    """Return a sortable key for final numeric releases, ignoring pre/dev/local versions."""
    normalized = version.removeprefix("v")
    if _FINAL_RELEASE_RE.fullmatch(normalized) is None:
        return None
    return tuple(int(part) for part in normalized.split("."))


def has_installable_files(files: object) -> bool:
    """Return True when a PyPI release has at least one non-yanked file."""
    if not isinstance(files, list):
        return False
    return any(isinstance(file, dict) and not file.get("yanked", False) for file in files)


def previous_published_version(current_version: str, releases: dict[str, Any]) -> str:
    """Return the latest published final release lower than current_version."""
    current_key = final_release_key(current_version)
    if current_key is None:
        msg = f"Current version must be a final numeric release, got {current_version!r}."
        raise ValueError(msg)

    candidates = [
        (key, version)
        for version, files in releases.items()
        if (key := final_release_key(version)) is not None and key < current_key and has_installable_files(files)
    ]
    if not candidates:
        msg = f"No published PyPI release found before {current_version}."
        raise ValueError(msg)

    return max(candidates, key=lambda item: item[0])[1]


def fetch_pypi_releases(project: str, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch project releases from PyPI JSON metadata."""
    url = PYPI_JSON_URL.format(project=project)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)

    return pypi_releases_from_payload(project, payload)


def pypi_releases_from_payload(project: str, payload: object) -> dict[str, Any]:
    """Extract the releases table from a PyPI JSON payload."""
    if not isinstance(payload, dict):
        msg = f"PyPI response for {project!r} is not a JSON object."
        raise TypeError(msg)

    releases = payload.get("releases")
    if not isinstance(releases, dict):
        msg = f"PyPI response for {project!r} does not contain a releases table."
        raise TypeError(msg)
    return releases


def write_github_env(path: Path, previous_version: str) -> None:
    """Append the resolved version to a GitHub Actions environment file."""
    with path.open("a") as file:
        file.write(f"PREVIOUS_RELEASE_VERSION={previous_version}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT_NAME)
    parser.add_argument("--current-version", required=True)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()

    try:
        previous_version = previous_published_version(
            args.current_version,
            fetch_pypi_releases(args.project),
        )
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Failed to resolve previous PyPI release: {exc}\n")
        return 1

    if args.github_env is not None:
        write_github_env(args.github_env, previous_version)

    sys.stdout.write(f"PREVIOUS_RELEASE_VERSION={previous_version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
